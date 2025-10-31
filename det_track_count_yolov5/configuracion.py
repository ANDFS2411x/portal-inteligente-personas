# ================= CONFIGURACIÓN GLOBAL =================

INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.30
IOU_THRESHOLD = 0.65
ENGINE_PATH = "/home/andfs/portal-inteligente-personas/modelos/YoloV5s/100_Epocas/best4_100epocas.engine"
CAP_QUEUE_MAX = 5          # Cola captura → detección
VIZ_MAX_FPS = 10           # FPS máximo para preview
SHOW_FPS_ON_FRAME = True   # Mostrar FPS en el frame

MAX_TRACK_AGE = 6          # Máx. frames sin actualizar tracker
