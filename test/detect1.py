#detect1.py
import os
import time
import cv2
import numpy as np
import tensorrt as trt
import csv

# CUDA / PyCUDA
import pycuda.autoinit
import pycuda.driver as cuda

# ========= Config =========
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.20
IOU_THRESHOLD = 0.45
MIN_BOX_PIXELS = 30000
ENGINE_PATH = "../best4_100epocas.engine"
HEADLESS = False
OUTPUT_VIDEO = "deteccion_output.avi"
OUTPUT_CSV = "detecciones_log.csv"

REF_AREA = 77000
AREA_TOL = 40000

# ========= Utilidades =========
def letterbox(im, new_shape=(640, 640), color=(128, 128, 128)):
    h, w = im.shape[:2]
    r = min(new_shape[1] / h, new_shape[0] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[0] - new_unpad[0]) // 2
    dh = (new_shape[1] - new_unpad[1]) // 2
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(im_resized, dh, new_shape[1]-new_unpad[1]-dh,
                                   dw, new_shape[0]-new_unpad[0]-dw,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, dw, dh

def plot_one_box(x, img, color=(0, 0, 255), label=None, line_thickness=None):
    tl = line_thickness or max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2f = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2f, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

def nms_numpy(boxes, scores, iou_thres=0.45):
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
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        idx = np.where(iou <= iou_thres)[0]
        order = order[idx + 1]
    return keep

# ========= Wrapper TensorRT =========
class YoLov5TRT:
    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"No se encontró el engine en: {engine_path}")

        self.logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs, self.cuda_inputs = [], []
        self.host_outputs, self.cuda_outputs = [], []

        for i, binding in enumerate(self.engine):
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape) * (self.engine.max_batch_size if self.engine.has_implicit_batch_dimension else 1)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(device_mem)

    def preprocess(self, frame_bgr):
        lb_img, r, dw, dh = letterbox(frame_bgr, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
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
        preds = out.reshape(-1, 6)

        confs = preds[:, 4]
        keep = confs >= CONF_THRESH
        preds = preds[keep]
        if preds.size == 0:
            return [], []

        cxcywh = preds[:, :4].astype(np.float32)
        x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2
        y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2
        x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2
        y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2

        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r

        x1 = np.clip(x1, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        x2 = np.clip(x2, 0, W - 1)
        y2 = np.clip(y2, 0, H - 1)

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = preds[:, 4].astype(np.float32)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        size_mask = areas >= MIN_BOX_PIXELS
        boxes = boxes[size_mask]
        scores = scores[size_mask]

        if boxes.shape[0] == 0:
            return [], []

        keep_idx = nms_numpy(boxes, scores, IOU_THRESHOLD)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        return boxes.tolist(), scores.tolist()

# ========= Main =========
def main():
    if not os.path.exists(ENGINE_PATH):
        print(f"❌ No se encontró el engine en: {ENGINE_PATH}")
        return

    model = YoLov5TRT(ENGINE_PATH)

    gst_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg,width=640,height=360,framerate=30/1 ! "
        "jpegdec ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara con GStreamer")
        return
    print("✅ Cámara abierta con GStreamer (MJPG 640x360@30fps)")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (width, height))

    prev_box = None
    alpha = 0.7

    # === CSV Writer ===
    csv_file = open(OUTPUT_CSV, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "conf", "area"])

    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ No se pudo leer frame")
                break

            t1 = time.time()
            boxes, scores = model.infer(frame)
            t2 = time.time()
            fps = 1.0 / max(t2 - t1, 1e-6)

            if len(boxes) == 0:
                print(f"⏳ Sin detecciones | FPS: {fps:.1f}")
            else:
                print(f"📸 Detecciones: {len(boxes)} | FPS: {fps:.1f}")
                for b, s in zip(boxes, scores):
                    x1, y1, x2, y2 = map(int, b)
                    area = (x2 - x1) * (y2 - y1)

                    if abs(area - REF_AREA) > AREA_TOL:
                        continue

                    if prev_box is not None:
                        x1 = int(alpha * prev_box[0] + (1 - alpha) * x1)
                        y1 = int(alpha * prev_box[1] + (1 - alpha) * y1)
                        x2 = int(alpha * prev_box[2] + (1 - alpha) * x2)
                        y2 = int(alpha * prev_box[3] + (1 - alpha) * y2)

                    prev_box = (x1, y1, x2, y2)

                    print(f" - person ({s:.2f}) box=({x1},{y1},{x2},{y2}) area={area}")

                    # === Guardar en CSV ===
                    csv_writer.writerow([frame_id, x1, y1, x2, y2, round(float(s), 3), area])

                    if not HEADLESS:
                        plot_one_box([x1, y1, x2, y2], frame, label=f"person:{s:.2f}")

            out.write(frame)
            frame_id += 1

            if not HEADLESS:
                cv2.imshow("Detección Personas (TensorRT + suavizado)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        out.release()
        csv_file.close()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print(f"💾 Video guardado en {OUTPUT_VIDEO}")
        print(f"📑 CSV guardado en {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
