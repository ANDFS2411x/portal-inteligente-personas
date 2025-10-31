import os, cv2, numpy as np, tensorrt as trt, pycuda.driver as cuda
from configuracion import INPUT_W, INPUT_H, CONF_THRESH, IOU_THRESHOLD, ENGINE_PATH
from utils_visuales import letterbox, nms_numpy

class YoLov5TRT:
    """Wrapper para ejecutar YOLOv5 optimizado en TensorRT."""
    def __init__(self, engine_path=ENGINE_PATH):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"❌ No se encontró el modelo: {engine_path}")
        assert cuda.Context.get_current() is not None, \
            "⚠️ No hay contexto CUDA activo."

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
        inp, meta, orig_hw = self.preprocess(frame)
        r, dw, dh = meta
        H, W = orig_hw
        np.copyto(self.host_inputs[0], inp.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        preds = self.host_outputs[0].reshape(-1, 6)
        preds = preds[preds[:, 4] >= CONF_THRESH]
        if preds.size == 0:
            return [], []

        cxcywh = preds[:, :4].astype(np.float32)
        x1 = (cxcywh[:, 0] - cxcywh[:, 2] / 2 - dw) / r
        y1 = (cxcywh[:, 1] - cxcywh[:, 3] / 2 - dh) / r
        x2 = (cxcywh[:, 0] + cxcywh[:, 2] / 2 - dw) / r
        y2 = (cxcywh[:, 1] + cxcywh[:, 3] / 2 - dh) / r

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = preds[:, 4]
        keep = nms_numpy(boxes, scores, IOU_THRESHOLD)
        return boxes[keep].tolist(), scores[keep].tolist()
