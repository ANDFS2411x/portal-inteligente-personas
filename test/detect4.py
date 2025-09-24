import os
import time
import csv
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import threading
import queue

# ========= Config =========
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.2           # más bajo para detectar más
IOU_THRESHOLD = 0.65
ENGINE_PATH = "../best4_100epocas.engine"
HEADLESS = False
WRITE_VIDEO = True          # ✅ puedes desactivar para más FPS
WRITE_CSV = True            # ✅ puedes desactivar para más FPS
OUTPUT_VIDEO = "deteccion_output.avi"
OUTPUT_FPS = 10.0
METRICS_CSV = "metrics_log.csv"

frame_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

def letterbox(im, new_shape=(640, 640), color=(128, 128, 128)):
    h, w = im.shape[:2]
    r = min(new_shape[1] / h, new_shape[0] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[0] - new_unpad[0]) // 2
    dh = (new_shape[1] - new_unpad[1]) // 2
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(
        im_resized, dh, new_shape[1]-new_unpad[1]-dh,
        dw, new_shape[0]-new_unpad[0]-dw,
        cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, dw, dh

def plot_one_box(x, img, color=(0, 0, 255), label=None, line_thickness=None):
    tl = line_thickness or max(1, round(0.002 * (img.shape[0]+img.shape[1])/2)+1)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

def nms_fast(boxes, scores, iou_thres=0.5):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(-scores)
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs[1:]] - inter)
        idxs = idxs[np.where(iou <= iou_thres)[0] + 1]
    return keep

class YoLov5TRT:
    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine no encontrado en {engine_path}")
        self.logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host_inputs, self.cuda_inputs = [], [], []
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

    def preprocess(self, frame):
        lb_img, r, dw, dh = letterbox(frame, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(img), (r, dw, dh), frame.shape[:2]

    def infer(self, frame):
        inp, meta, orig_shape = self.preprocess(frame)
        r, dw, dh = meta
        H, W = orig_shape
        np.copyto(self.host_inputs[0], inp.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        out = self.host_outputs[0].reshape(-1, 6)
        confs = out[:, 4]
        keep = confs >= CONF_THRESH
        preds = out[keep]
        if preds.shape[0] == 0:
            return [], []
        cxcywh = preds[:, :4]
        x1 = (cxcywh[:, 0] - cxcywh[:, 2] / 2 - dw) / r
        y1 = (cxcywh[:, 1] - cxcywh[:, 3] / 2 - dh) / r
        x2 = (cxcywh[:, 0] + cxcywh[:, 2] / 2 - dw) / r
        y2 = (cxcywh[:, 1] + cxcywh[:, 3] / 2 - dh) / r
        x1 = np.clip(x1, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        x2 = np.clip(x2, 0, W - 1)
        y2 = np.clip(y2, 0, H - 1)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = preds[:, 4]
        keep_idx = nms_fast(boxes, scores, IOU_THRESHOLD)
        return boxes[keep_idx].tolist(), scores[keep_idx].tolist()

def capture_thread(cap):
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(frame)
    stop_event.set()

def detection_thread(video_writer, csv_writer):
    ctx = cuda.Device(0).make_context()
    frame_id = 0
    try:
        model = YoLov5TRT(ENGINE_PATH)
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.time()
            boxes, scores = model.infer(frame)
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            for b, s in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, b)
                plot_one_box([x1, y1, x2, y2], frame, label=f"person:{s:.2f}")
            if not HEADLESS:
                cv2.putText(frame, f"{fps:.1f} FPS", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("TensorRT Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
            if WRITE_VIDEO and video_writer:
                video_writer.write(frame)
            if WRITE_CSV and csv_writer:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes] if boxes else []
                confs = np.array(scores)
                csv_writer.writerow([
                    frame_id, time.time(), round(fps, 2), len(scores),
                    round(confs.mean(), 3) if confs.size else 0,
                    round(confs.min(), 3) if confs.size else 0,
                    round(confs.max(), 3) if confs.size else 0,
                    round(np.mean(areas), 1) if areas else 0])
            frame_id += 1
    finally:
        ctx.pop()

def main():
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if WRITE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, OUTPUT_FPS, (width, height))
    csv_writer = None
    if WRITE_CSV:
        csv_file = open(METRICS_CSV, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_id", "timestamp", "fps", "n_dets",
                             "mean_conf", "min_conf", "max_conf", "mean_area"])
    t_cap = threading.Thread(target=capture_thread, args=(cap,), daemon=True)
    t_det = threading.Thread(target=detection_thread, args=(out, csv_writer), daemon=True)
    t_cap.start()
    t_det.start()
    t_cap.join()
    t_det.join()
    cap.release()
    if out: out.release()
    if WRITE_CSV: csv_file.close()
    if not HEADLESS: cv2.destroyAllWindows()
    print("✅ Finalizado. Video y CSV guardados.")

if __name__ == "__main__":
    main()
