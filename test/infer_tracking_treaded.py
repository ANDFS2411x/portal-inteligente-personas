
# 3 hilos: captura + detección + preview suave (persistente, sin colas)
# No guarda nada. Ventana siempre abierta; si no hay detección, muestra un placeholder.

import os
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import threading
import queue
import signal
import sys
import collections
import warnings
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ========= Config =========
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.30
IOU_THRESHOLD = 0.65
ENGINE_PATH = "../best4_100epocas.engine"

CAP_QUEUE_MAX = 5          # cola de captura -> detección
VIZ_MAX_FPS = 10           # límite de FPS para mostrar (preview)
VIZ_TTL = 0.6              # si no llega frame anotado en este tiempo, mostrar placeholder
SHOW_FPS_ON_FRAME = True

# ========= Buffers & control =========
frame_queue = queue.Queue(maxsize=CAP_QUEUE_MAX)
stop_event = threading.Event()

# Buffer atómico para preview (último frame anotado)
viz_lock = threading.Lock()
viz_frame = None           # np.ndarray BGR anotado
viz_last_ts = 0.0          # timestamp de última actualización
viz_event = threading.Event()  # señal: “hay frame nuevo”

# ========= Utilidades =========
def letterbox(im, new_shape=(640, 640), color=(128, 128, 128)):
    h, w = im.shape[:2]
    r = min(new_shape[1] / h, new_shape[0] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[0] - new_unpad[0]) // 2
    dh = (new_shape[1] - new_unpad[1]) // 2
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(
        im_resized, dh, new_shape[1] - new_unpad[1] - dh,
        dw, new_shape[0] - new_unpad[0] - dw,
        cv2.BORDER_CONSTANT, value=color,
    )
    return im_padded, r, dw, dh

def plot_one_box(x, img, color=(0, 255, 0), label=None, line_thickness=None):
    tl = line_thickness or max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 4), c2, color, -1, cv2.LINE_AA)  # filled background
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    (0, 0, 0), thickness=tf, lineType=cv2.LINE_AA)

def nms_numpy(boxes, scores, iou_thres=IOU_THRESHOLD):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep

# ========= SORT Tracking Functions =========
def iou(bb_test, bb_gt):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
             (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    if union <= 0:
        return 0.0
    return wh / union

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# ========= Wrapper TensorRT =========
class YoLov5TRT:
    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"No se encontró el engine en: {engine_path}")
        assert cuda.Context.get_current() is not None, \
            "No hay contexto CUDA activo. Crea make_context() en este hilo antes de cargar el engine."

        self.logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_inputs, self.cuda_inputs = [], []
        self.host_outputs, self.cuda_outputs = [], []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(device_mem)

        try:
            if not self.engine.has_implicit_batch_dimension:
                input_binding = next(b for b in self.engine if self.engine.binding_is_input(b))
                self.context.set_binding_shape(
                    self.engine.get_binding_index(input_binding),
                    (1, 3, INPUT_H, INPUT_W),
                )
        except Exception:
            pass

    def preprocess(self, frame_bgr):
        lb_img, r, dw, dh = letterbox(frame_bgr, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return np.ascontiguousarray(img), (r, dw, dh), frame_bgr.shape[:2]

    def infer(self, frame_bgr):
        inp, meta, orig_hw = self.preprocess(frame_bgr)
        r, dw, dh = meta
        H, W = orig_hw
        np.copyto(self.host_inputs[0], inp.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        out = self.host_outputs[0]
        assert out.size % 6 == 0, f"Salida inesperada del engine (size={out.size})."
        preds = out.reshape(-1, 6)

        confs = preds[:, 4]
        keep_mask = confs >= CONF_THRESH
        preds = preds[keep_mask]
        if preds.size == 0:
            return [], []

        cxcywh = preds[:, :4].astype(np.float32)
        x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2
        y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2
        x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2
        y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2

        x1 = (x1 - dw) / r;  y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r;  y2 = (y2 - dh) / r

        x1 = np.clip(x1, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        x2 = np.clip(x2, 0, W - 1)
        y2 = np.clip(y2, 0, H - 1)

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = preds[:, 4].astype(np.float32)

        keep_idx = nms_numpy(boxes, scores, IOU_THRESHOLD)
        return boxes[keep_idx].tolist(), scores[keep_idx].tolist()

# ========= Threads =========
def capture_thread(cap):
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        # evitar lag: si lleno, descarta el más viejo
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(frame)
    stop_event.set()

def detection_thread():
    # Contexto CUDA SOLO en este hilo
    ctx = cuda.Device(0).make_context()
    frame_id = 0
    last_sent_to_viz = 0.0  # para limitar VIZ_MAX_FPS

    trackers = []
    max_age = 4
    idstp = collections.defaultdict(list)
    idcnt = []
    incnt, outcnt = 0, 0

    try:
        model = YoLov5TRT(ENGINE_PATH)
        print(f"🚀 Detección iniciada | CONF={CONF_THRESH} | IOU={IOU_THRESHOLD}")
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            t0 = time.time()
            boxes, scores = model.infer(frame)
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)

            boxes_np = np.array(boxes) if boxes else np.empty((0, 4))
            H, W = frame.shape[:2]

            # Predict trackers
            trks = np.zeros((len(trackers), 5))
            to_del = []
            for t, trk in enumerate(trks):
                pos = trackers[t].predict()[0]
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                trackers.pop(t)

            # Associate
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(boxes_np, trks)

            # Update matched trackers
            for t, trk in enumerate(trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0][0]
                    trk.update(boxes_np[d, :])
                    xmin, ymin, xmax, ymax = map(int, boxes_np[d, :])
                    cy = int((ymin + ymax) / 2)

                    # Counting logic (using initial position)
                    initial_y = idstp[trk.id][0][1] if idstp[trk.id] else 0
                    if initial_y < H // 2 and cy > H // 2 and trk.id not in idcnt:
                        incnt += 1
                        print("id: " + str(trk.id) + " - IN ")
                        idcnt.append(trk.id)
                    elif initial_y > H // 2 and cy < H // 2 and trk.id not in idcnt:
                        outcnt += 1
                        print("id: " + str(trk.id) + " - OUT ")
                        idcnt.append(trk.id)

            # Create new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(boxes_np[i, :])
                trackers.append(trk)
                u = trk.kf.x[0][0]
                v = trk.kf.x[1][0]
                idstp[trk.id].append([u, v])

            # Remove old trackers
            i = len(trackers) - 1
            while i >= 0:
                if trackers[i].time_since_update > max_age:
                    trackers.pop(i)
                i -= 1

            if boxes:
                confs_txt = ", ".join(f"{s:.2f}" for s in scores)
                print(f"[DETECCIÓN] frame {frame_id} | fps: {fps:.1f} | "
                      f"{len(scores)} persona(s) | confs: {confs_txt}")
                sys.stdout.flush()

            # Throttle de preview
            now = time.time()
            if (now - last_sent_to_viz) >= (1.0 / VIZ_MAX_FPS):
                frame_vis = frame.copy()

                # Draw all trackers (using current state, predicted or updated)
                for trk in trackers:
                    pos = trk.get_state()[0]
                    plot_one_box(pos, frame_vis, color=(0, 255, 0), label=f"person:{trk.id}")

                # Draw text (Total, IN, OUT, FPS) at the bottom with green background
                text = f"Total: {incnt + outcnt}  IN: {incnt}  OUT: {outcnt}  FPS: {fps:.1f}"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                thickness = 1
                t_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                bg_height = t_size[1] + 10
                bg_top = H - bg_height
                # Draw semi-transparent green background
                overlay = frame_vis.copy()
                cv2.rectangle(overlay, (0, bg_top), (W, H), (0, 255, 0), -1)
                alpha = 0.5  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame_vis, 1 - alpha, 0, frame_vis)
                # Draw black text
                cv2.putText(frame_vis, text, (10, H - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                # Publicar como “último frame” (last-writer-wins)
                with viz_lock:
                    global viz_frame, viz_last_ts
                    viz_frame = frame_vis
                    viz_last_ts = now
                viz_event.set()  # avisa al hilo de preview
                last_sent_to_viz = now

            frame_id += 1
    finally:
        try:
            ctx.pop()
        except Exception:
            pass

def preview_thread():
    """Ventana persistente. Dibuja el último frame anotado si es reciente; sino, un placeholder."""
    win = "Detección (preview)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 450)

    target_dt = 1.0 / max(1.0, VIZ_MAX_FPS)  # igual usamos esto para no saturar UI

    while not stop_event.is_set():
        # Espera corta por nuevos frames, pero no se bloquea de por vida
        viz_event.wait(timeout=target_dt)
        viz_event.clear()

        # Obtiene el último frame si existe
        with viz_lock:
            frame = None if viz_frame is None else viz_frame.copy()
            last_ts = viz_last_ts

        now = time.time()
        fresh = frame is not None and (now - last_ts) <= VIZ_TTL

        if fresh:
            to_show = frame
        else:
            # placeholder negro del tamaño más común
            to_show = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(to_show, "Esperando deteccion...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        try:
            cv2.imshow(win, to_show)
            # Mantener UI responsiva y permitir cerrar con ESC/Q
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                stop_event.set()
                break
        except Exception:
            # En caso de error de backend de ventana, evita crash
            time.sleep(target_dt)

        # no más de VIZ_MAX_FPS
        time.sleep(max(0.0, target_dt))

    try:
        cv2.destroyWindow(win)
    except Exception:
        pass

# ========= Main =========
def main():
    # Ctrl+C -> cierre limpio
    def handle_sigint(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # Suele ayudar a que OpenCV no compita por CPU en Jetson
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Inicializa CUDA
    cuda.init()

    # Cámara (Jetson/GStreamer)
    gst_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg,width=640,height=360,framerate=30/1 ! "
        "jpegdec ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    # Lanzar hilos
    t_cap = threading.Thread(target=capture_thread, args=(cap,), daemon=True)
    t_det = threading.Thread(target=detection_thread, daemon=True)
    t_viz = threading.Thread(target=preview_thread, daemon=True)
    t_cap.start(); t_det.start(); t_viz.start()

    t_cap.join(); t_det.join(); t_viz.join()

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("✅ Finalizado.")

if __name__ == "__main__":
    main()
