import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
             (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return wh / union if union > 0 else 0.

def convert_bbox_to_z(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = bbox[0] + w/2., bbox[1] + h/2.
    s, r = w*h, w/float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w, h = np.sqrt(x[2]*x[3]), x[2]/np.sqrt(x[2]*x[3])
    bbox = np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.])
    if score is not None:
        bbox = np.append(bbox, score)
    return bbox.reshape(1, -1)

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0,4] = self.kf.F[1,5] = self.kf.F[2,6] = 1
        self.kf.H = np.eye(4,7)
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0: self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.2):
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind)))
    unmatched_dets = [d for d in range(len(detections)) if d not in row_ind]
    unmatched_trks = [t for t in range(len(trackers)) if t not in col_ind]
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    matches = np.concatenate(matches, axis=0) if len(matches) else np.empty((0,2),dtype=int)
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
