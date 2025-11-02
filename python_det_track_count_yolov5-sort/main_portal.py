import cv2
import signal
import threading
import pycuda.driver as cuda
from hilos_sistema import capture_thread, detection_thread, preview_thread, stop_event
from configuracion import CAP_QUEUE_MAX

def main():
    # ====== Señales de salida limpia ======
    def handle_sigint(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # ====== Inicialización de CUDA ======
    cuda.init()

    # ====== Fuente de cámara ======
    gst_pipeline = (
"v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=360,framerate=30/1 ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    # ====== Lanzar hilos ======
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

    print("✅ Finalizado correctamente.")

if __name__ == "__main__":
    main()
