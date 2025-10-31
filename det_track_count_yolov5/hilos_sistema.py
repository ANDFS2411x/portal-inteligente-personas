import time
import cv2
import sys
import numpy as np
import threading
import queue
import collections
import pycuda.driver as cuda
from configuracion import (
    CAP_QUEUE_MAX, VIZ_MAX_FPS, ENGINE_PATH, CONF_THRESH, IOU_THRESHOLD, MAX_TRACK_AGE
)
from detector_trt import YoLov5TRT
from tracker_kalman import KalmanBoxTracker, associate_detections_to_trackers
from utils_visuales import plot_box

# ==================== VARIABLES GLOBALES ====================
frame_queue = queue.Queue(maxsize=CAP_QUEUE_MAX)
stop_event = threading.Event()

viz_lock = threading.Lock()
viz_frame = None
viz_last_ts = 0.0
viz_event = threading.Event()

# ==================== HILO DE CAPTURA ====================
def capture_thread(cap):
    """Lee frames de la cámara y los pasa a la cola compartida."""
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

# ==================== HILO DE DETECCIÓN ====================
def detection_thread():
    """Corre la inferencia YOLOv5-TRT + tracking SORT + conteo IN/OUT + rastro visual."""
    ctx = cuda.Device(0).make_context()
    frame_id = 0
    last_sent_to_viz = 0.0

    trackers = []
    idstp = collections.defaultdict(list)  # trayectoria (rastro)
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

            # === PREDICCIÓN DE KALMAN ===
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

            # === ASOCIACIÓN (HÚNGARO) ===
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(boxes_np, trks)

            # === ACTUALIZAR TRACKERS EXISTENTES ===
            for t, trk in enumerate(trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0][0]
                    trk.update(boxes_np[d, :])
                    xmin, ymin, xmax, ymax = map(int, boxes_np[d, :])
                    cy = int((ymin + ymax) / 2)
                    cx = int((xmin + xmax) / 2)
                    idstp[trk.id].append([cx, cy])  # guardar trayectoria

                    # === LÓGICA DE CONTEO (línea media) ===
                    initial_y = idstp[trk.id][0][1] if idstp[trk.id] else 0
                    if initial_y < H // 2 and cy > H // 2 and trk.id not in idcnt:
                        incnt += 1
                        print(f"id: {trk.id} - OUT")
                        idcnt.append(trk.id)
                    elif initial_y > H // 2 and cy < H // 2 and trk.id not in idcnt:
                        outcnt += 1
                        print(f"id: {trk.id} - IN")
                        idcnt.append(trk.id)

            # === CREAR NUEVOS TRACKERS ===
            for i in unmatched_dets:
                trk = KalmanBoxTracker(boxes_np[i, :])
                trackers.append(trk)
                u = trk.kf.x[0][0]
                v = trk.kf.x[1][0]
                idstp[trk.id].append([u, v])

            # === ELIMINAR TRACKERS VIEJOS ===
            i = len(trackers) - 1
            while i >= 0:
                if trackers[i].time_since_update > MAX_TRACK_AGE:
                    trackers.pop(i)
                i -= 1

            # === MOSTRAR RESULTADOS ===
            if boxes:
                confs_txt = ", ".join(f"{s:.2f}" for s in scores)
                print(f"[DETECCIÓN] frame {frame_id} | fps: {fps:.1f} | "
                      f"{len(scores)} persona(s) | confs: {confs_txt}")
                sys.stdout.flush()

            # === VISUALIZACIÓN ===
            now = time.time()
            if (now - last_sent_to_viz) >= (1.0 / VIZ_MAX_FPS):
                frame_vis = frame.copy()
                for trk in trackers:
                    pos = trk.get_state()[0]
                    color = (trk.id * 37 % 255, trk.id * 17 % 255, trk.id * 91 % 255)
                    plot_box(pos, frame_vis, color=color, label=f"ID:{trk.id}")

                    # === DIBUJAR TRAZADO DEL TRACK ===
                    if trk.id in idstp and len(idstp[trk.id]) > 1:
                        pts = np.array(idstp[trk.id], np.int32)
                        cv2.polylines(frame_vis, [pts], False, color, 2)

                # === TEXTO DE INFORMACIÓN ===
                text = f"Total: {incnt + outcnt}  OUT: {incnt}  IN: {outcnt}  FPS: {fps:.1f}"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                thickness = 1
                t_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                bg_height = t_size[1] + 10
                bg_top = H - bg_height
                overlay = frame_vis.copy()
                cv2.rectangle(overlay, (0, bg_top), (W, H), (0, 255, 0), -1)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame_vis, 1 - alpha, 0, frame_vis)
                cv2.putText(frame_vis, text, (10, H - 5), font, font_scale,
                            (0, 0, 0), thickness, cv2.LINE_AA)

                with viz_lock:
                    global viz_frame, viz_last_ts
                    viz_frame = frame_vis
                    viz_last_ts = now
                viz_event.set()
                last_sent_to_viz = now

            frame_id += 1

    finally:
        try:
            ctx.pop()
        except Exception:
            pass

# ==================== HILO DE VISTA PREVIA ====================
def preview_thread():
    """Muestra siempre el último frame procesado en una ventana persistente."""
    win = "Detección (preview)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 450)

    target_dt = 1.0 / max(1.0, VIZ_MAX_FPS)
    while not stop_event.is_set():
        viz_event.wait(timeout=target_dt)
        viz_event.clear()

        with viz_lock:
            frame = None if viz_frame is None else viz_frame.copy()

        if frame is not None:
            try:
                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')):
                    stop_event.set()
                    break
            except Exception:
                pass
        time.sleep(max(0.0, target_dt))

    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
