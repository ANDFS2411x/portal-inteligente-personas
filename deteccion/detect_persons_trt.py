import cv2
import numpy as np
import threading
import queue
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time

# — Parámetros de detección —
CONF_THRESH   = 0.5             # umbral mínimo de confianza
NMS_THRESH    = 0.4             # umbral de NMS
ENGINE_PATH = "../best3.engine"
INPUT_WIDTH   = 640
INPUT_HEIGHT  = 640
CLASS_PERSON  = 0               # solo mostrar detecciones de clase 0

# — Clase para YOLOv5 TensorRT —
class YoLov5TRT:
    def __init__(self, engine_path):
        self.logger  = trt.Logger(trt.Logger.WARNING)
        runtime     = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine  = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self.inputs, self.outputs, self.bindings = [], [], []
        for b in self.engine:
            shape = self.engine.get_binding_shape(b)
            size  = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(b))
            host  = cuda.pagelocked_empty(size, dtype)
            dev   = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            (self.inputs if self.engine.binding_is_input(b) else self.outputs).append({
                "host": host, "device": dev
            })

    def infer(self, frame):
        img = self.preprocess(frame)
        # copiar a GPU
        np.copyto(self.inputs[0]["host"], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # inferencia
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        # traer resultados
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        return self.postprocess(self.outputs[0]["host"], frame.shape)

    @staticmethod
    def preprocess(image):
        r = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float16) / 255.0
        r = np.transpose(r, (2,0,1))
        return np.expand_dims(r, 0)

    @staticmethod
    def postprocess(output, orig_shape):
        dets = output.reshape(-1,6)
        H, W = orig_shape[:2]
        boxes, scores = [], []
        for cx,cy,w,h,conf,cls in dets:
            if conf < CONF_THRESH or int(cls) != CLASS_PERSON:
                continue
            x1 = int((cx - w/2)*W/INPUT_WIDTH)
            y1 = int((cy - h/2)*H/INPUT_HEIGHT)
            boxes.append([x1,y1,int(w*W/INPUT_WIDTH), int(h*H/INPUT_HEIGHT)])
            scores.append(float(conf))

        # NMS seguro
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
        results = []
        if len(idxs) > 0:
            # idxs podría ser lista de listas o tupla
            for idx in idxs:
                i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                x,y,w,h = boxes[i]
                results.append((x, y, x+w, y+h, scores[i]))
        return results

# — Reader en hilo para mejor FPS —
def camera_reader(q):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return
    while True:
        ret, frame = cap.read()
        if not ret: break
        if q.full(): q.get()   # descartar si no se procesó
        q.put(frame)
    cap.release()

# — Main —
if __name__ == "__main__":
    q = queue.Queue(maxsize=1)
    threading.Thread(target=camera_reader, args=(q,), daemon=True).start()
    model = YoLov5TRT(ENGINE_PATH)

    while True:
        if q.empty(): continue
        frame = q.get()
        t0 = time.time()
        dets = model.infer(frame)
        fps = 1/(time.time()-t0)

        # dibujar cajas y conteo
        for x1,y1,x2,y2,conf in dets:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, f"Persons: {len(dets)}  FPS: {fps:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Person Detector TRT", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
