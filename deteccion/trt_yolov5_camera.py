import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time

CONF_THRESH = 0.4
ENGINE_PATH = "../best3.engine"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 📦 Clase para manejar el engine de TensorRT
class YoLov5TRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, image_raw):
        image = preprocess(image_raw)
        np.copyto(self.inputs[0]["host"], image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        output = self.outputs[0]["host"]
        return postprocess(output, image_raw)

# 🔁 Preprocesamiento como YOLOv5
def preprocess(image):
    resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# 🧠 Postprocesamiento para salida [1, 25200, 6]
def postprocess(output, original_img):
    output = output.reshape((25200, 6))
    H, W = original_img.shape[:2]
    results = []
    for det in output:
        conf = det[4]
        if conf < CONF_THRESH:
            continue
        cx, cy, w, h = det[0:4]
        x1 = int((cx - w / 2) * W / INPUT_WIDTH)
        y1 = int((cy - h / 2) * H / INPUT_HEIGHT)
        x2 = int((cx + w / 2) * W / INPUT_WIDTH)
        y2 = int((cy + h / 2) * H / INPUT_HEIGHT)
        cls = int(det[5])
        results.append((x1, y1, x2, y2, conf, cls))
    return results

# 🎥 Main loop
if __name__ == "__main__":
    model = YoLov5TRT(ENGINE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = model.infer(frame)
        end = time.time()
        fps = 1 / (end - start)

        for x1, y1, x2, y2, conf, cls in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("YOLOv5 TRT", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
