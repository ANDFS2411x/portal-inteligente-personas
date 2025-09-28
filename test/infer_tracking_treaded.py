# detect_preview_smooth_tracking.py
# 3 hilos: captura + detección (con tracking) + preview persistente
# Ligero para Jetson: tracker IoU/SORT-lite sin dependencias externas.

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

# ========= Config =========
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.30
IOU_THRESHOLD = 0.65                 # NMS
ENGINE_PATH = "../best4_100epocas.engine"

CAP_QUEUE_MAX = 5                     # cola captura -> detección
VIZ_MAX_FPS = 12.0                    # límite FPS de preview
VIZ_TTL = 0.6                         # si no hay frame anotado reciente, placeholder
SHOW_FPS_ON_FRAME = True

# Tracker (SORT-lite) hiperparámetros
TRACK_IOU_MATCH = 0.35                # IoU mínimo para asociar track<->detección
TRACK_MAX_AGE = 20                    # frames sin ver antes de eliminar track
TRACK_MIN_HITS = 3                    # hits para “confirmar” un track

# ========= Buffers & control =========
frame_queue = queue.Queue(maxsize=CAP_QUEUE_MAX)
stop_event  = threading.Event()

viz_lock = threading.Lock()
viz_frame = None
viz_last_ts = 0.0
viz_event = threading.Event()

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

def plot_one_box(x, img, color=(0, 0, 255), label=None, line_thickness=None):
    tl = line_thickness or max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

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

# ========= Tracker (SORT-lite sin Kalman) =========
def iou_matrix(tracks, dets):
    """tracks: (N,4) dets: (M,4) -> IoU NxM"""
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)), dtype=np.float32)
    t = np.array(tracks, dtype=np.float32)
    d = np.array(dets, dtype=np.float32)

    # t: (N,4), d: (M,4)
    t_x1, t_y1, t_x2, t_y2 = t[:,0:1], t[:,1:2], t[:,2:3], t[:,3:4]
    d_x1, d_y1, d_x2, d_y2 = d[:,0], d[:,1], d[:,2], d[:,3]

    xx1 = np.maximum(t_x1, d_x1)
    yy1 = np.maximum(t_y1, d_y1)
    xx2 = np.minimum(t_x2, d_x2)
    yy2 = np.minimum(t_y2, d_y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    area_t = (t_x2 - t_x1 + 1) * (t_y2 - t_y1 + 1)
    area_d = (d_x2 - d_x1 + 1) * (d_y2 - d_y1 + 1)
    union = area_t + area_d - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)

class Track:
    __slots__ = ("tid", "bbox", "score", "age", "hits", "time_since_update")
    def __init__(self, tid, bbox, score):
        self.tid = tid
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = float(score)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

class IoUTracker:
    def __init__(self, iou_match=0.35, max_age=20, min_hits=3):
        self.iou_match = iou_match
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    def _greedy_assign(self, iou_mat, thr):
        if iou_mat.size == 0:
            return [], set(range(iou_mat.shape[0])), set(range(iou_mat.shape[1]))
        matches = []
        t_used, d_used = set(), set()
        m = iou_mat.copy()
        while True:
            i, j = np.unravel_index(np.argmax(m), m.shape)
            if m[i, j] < thr:
                break
            matches.append((i, j))
            t_used.add(i); d_used.add(j)
            m[i, :] = -1.0
            m[:, j] = -1.0
        t_all = set(range(iou_mat.shape[0]))
        d_all = set(range(iou_mat.shape[1]))
        return matches, (t_all - t_used), (d_all - d_used)

    def update(self, det_boxes, det_scores):
        # 1) Actualiza envejecimiento
        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

        # 2) Asociar detecciones a tracks por IoU (greedy)
        T = len(self.tracks)
        D = len(det_boxes)
        if T > 0 and D > 0:
            iou_mat = iou_matrix([tr.bbox for tr in self.tracks], det_boxes)
            matches, un_t, un_d = self._greedy_assign(iou_mat, self.iou_match)
        else:
            matches = []
            un_t = set(range(T))
            un_d = set(range(D))

        # 3) Actualizar tracks emparejados
        for ti, dj in matches:
            tr = self.tracks[ti]
            tr.bbox = np.array(det_boxes[dj], dtype=np.float32)
            tr.score = float(det_scores[dj])
            tr.hits += 1
            tr.time_since_update = 0

        # 4) Crear tracks nuevos por detecciones no emparejadas
        for dj in un_d:
            self.tracks.append(Track(self.next_id, det_boxes[dj], det_scores[dj]))
            self.next_id += 1

        # 5) Eliminar tracks demasiado viejos
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]

        # 6) Devuelve los tracks “visibles” (confirmados o recién actualizados)
        visible = []
        for tr in self.tracks:
            confirmed = tr.hits >= self.min_hits or tr.time_since_update == 0
            if confirmed:
                visible.append(tr)
        return visible

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
    global viz_frame, viz_last_ts
    ctx = cuda.Device(0).make_context()
    frame_id = 0
    last_sent_to_viz = 0.0
    tracker = IoUTracker(iou_match=TRACK_IOU_MATCH, max_age=TRACK_MAX_AGE, min_hits=TRACK_MIN_HITS)

    try:
        model = YoLov5TRT(ENGINE_PATH)
        print(f"🚀 Detección+Tracking | CONF={CONF_THRESH} | IOU_NMS={IOU_THRESHOLD} | IOU_MATCH={TRACK_IOU_MATCH}")
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            t0 = time.time()
            det_boxes, det_scores = model.infer(frame)
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)

            if det_scores:
                tracks = tracker.update(det_boxes, det_scores)
                ids = [tr.tid for tr in tracks]
                confs = [tr.score for tr in tracks]
                print(f"[TRACK] frame {frame_id} | fps:{fps:.1f} | tracks:{len(tracks)} | ids:{ids} | confs:{[round(c,2) for c in confs]}")
                sys.stdout.flush()

                now = time.time()
                if (now - last_sent_to_viz) >= (1.0 / VIZ_MAX_FPS):
                    frame_vis = frame.copy()
                    for tr in tracks:
                        x1, y1, x2, y2 = map(int, tr.bbox)
                        label = f"ID:{tr.tid} {tr.score:.2f}"
                        plot_one_box([x1, y1, x2, y2], frame_vis, label=label)
                    if SHOW_FPS_ON_FRAME:
                        cv2.putText(frame_vis, f"FPS:{fps:.1f}", (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    with viz_lock:
                        viz_frame = frame_vis
                        viz_last_ts = now
                    viz_event.set()
                    last_sent_to_viz = now
            else:
                # también hay que avanzar el envejecimiento del tracker
                tracker.update([], [])

            frame_id += 1
    finally:
        try:
            ctx.pop()
        except Exception:
            pass

def preview_thread():
    win = "Detección (preview)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 450)

    target_dt = 1.0 / max(1.0, VIZ_MAX_FPS)

    while not stop_event.is_set():
        viz_event.wait(timeout=target_dt)
        viz_event.clear()

        with viz_lock:
            frame = None if viz_frame is None else viz_frame.copy()
            last_ts = viz_last_ts

        now = time.time()
        fresh = frame is not None and (now - last_ts) <= VIZ_TTL

        if fresh:
            to_show = frame
        else:
            to_show = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(to_show, "Esperando deteccion...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        try:
            cv2.imshow(win, to_show)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                stop_event.set()
                break
        except Exception:
            time.sleep(target_dt)

        time.sleep(max(0.0, target_dt))

# ========= Main =========
def main():
    def handle_sigint(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    cuda.init()

    gst_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg,width=640,height=360,framerate=30/1 ! "
        "jpegdec ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

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
