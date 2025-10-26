import cv2
import numpy as np
from configuracion import IOU_THRESHOLD

def letterbox(im, new_shape=(640, 640), color=(128, 128, 128)):
    """Redimensiona manteniendo proporciones (relleno gris)."""
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

def plot_box(x, img, color=(0, 255, 0), label=None, line_thickness=None):
    """Dibuja un bounding box con etiqueta."""
    tl = line_thickness or max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 4), c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    (0, 0, 0), thickness=tf, lineType=cv2.LINE_AA)

def nms_numpy(boxes, scores, iou_thres=IOU_THRESHOLD):
    """Non-Maximum Suppression (NMS) en NumPy."""
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
