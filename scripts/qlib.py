"""Shared helpers for direction C (FP32 / fake-quant / TensorRT evaluation on TT100K45).
Protocol (preregistration_C.md §4): square letterbox (no rect), conf 0.001, NMS IoU 0.7,
multi_label=True, max_det 300, pycocotools AP; extra bins by sqrt(GT area) in original px.
"""
import json, os, random, time
import cv2, numpy as np, torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = "./work/edgets"
DATA = os.environ.get("QDATA", f"{ROOT}/data/TT100K45")
GT_VAL = os.environ.get("QGT_VAL", f"{ROOT}/eval/gt_val.json")
GT_TEST = os.environ.get("QGT_TEST", f"{ROOT}/eval/gt_test.json")
FP32 = {  # (imgsz, seed) -> best.pt  (read-only references)
    **{(640, s): f"{ROOT}/runs/{d}_yolo11n_s{s}/weights/best.pt" for s, d in
       [(0, "20261008"), (1, "20261008"), (2, "20261008"), (3, "20261011"), (4, "20261011")]},
    **{(896, s): f"{ROOT}/runs/20261014_yolo11n-896_s{s}/weights/best.pt" for s in range(5)},
    **{(1024, s): f"{ROOT}/runs/{'20261015' if s == 0 else '20261017'}_yolo11n-1024_s{s}/weights/best.pt" for s in range(5)},
    **{(1280, s): f"{ROOT}/runs/20261015_yolo11n-1280_s{s}/weights/best.pt" for s in range(5)},
}
BINS = [("lt32", 0, 32), ("32_64", 32, 64), ("ge64", 64, 1e5)]  # sqrt(area), original pixels


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, left, top


def to_tensor(batch_np, device):
    x = np.ascontiguousarray(np.stack(batch_np)[..., ::-1].transpose(0, 3, 1, 2))  # BGR->RGB, NCHW
    return torch.from_numpy(x).to(device).float() / 255.0


def split_images(split):
    d = f"{DATA}/images/{split}"
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))


def calib_images(n=512, seed=0, mode="random"):
    """Calibration images from TRAIN only. mode: random | smallrich (images with most GT side<32px)."""
    ims = split_images("train")
    rng = random.Random(seed)
    if mode == "random":
        return rng.sample(ims, n)
    if mode == "smallrich":
        scores = []
        for p in ims:
            lab = p.replace("/images/", "/labels/").replace(".jpg", ".txt")
            k = 0
            if os.path.exists(lab):
                for line in open(lab):
                    v = line.split()
                    if len(v) >= 5 and max(float(v[3]), float(v[4])) * 2048 < 32:
                        k += 1
            scores.append((k, rng.random(), p))
        scores.sort(reverse=True)
        return [p for _, _, p in scores[:n]]
    raise ValueError(mode)


class CalibLoader:
    def __init__(self, paths, imgsz, bs=16, device="cuda"):
        self.paths, self.imgsz, self.bs, self.device = paths, imgsz, bs, device

    def __iter__(self):
        for i in range(0, len(self.paths), self.bs):
            ims = [letterbox(cv2.imread(p), self.imgsz)[0] for p in self.paths[i:i + self.bs]]
            yield to_tensor(ims, self.device)

    def __len__(self):
        return (len(self.paths) + self.bs - 1) // self.bs


def load_fp32(pt, device="cuda"):
    from ultralytics import YOLO
    m = YOLO(pt).model.float().to(device).eval()
    m = m.fuse()
    if getattr(m.model[-1], "end2end", False):  # YOLO26: evaluate the one2many branch with the same NMS protocol
        m.model[-1].end2end = False
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def build_cache(imgsz, split="val"):
    ims = split_images(split)
    return ims, [letterbox(cv2.imread(p), imgsz) for p in ims]


def predict_coco(forward, imgsz, split="val", bs=16, device="cuda", conf=0.001, iou=0.7, max_det=300, cache=None):
    """forward(x)->raw head tensor [B,4+nc,N] (xywh input px, sigmoid scores). Returns COCO det list + stats."""
    from ultralytics.utils.nms import non_max_suppression
    ims, allb = cache if cache is not None else (split_images(split), None)
    dets, maxscores = [], []
    for i in range(0, len(ims), bs):
        chunk = ims[i:i + bs]
        lbs = allb[i:i + bs] if allb is not None else [letterbox(cv2.imread(p), imgsz) for p in chunk]
        x = to_tensor([l[0] for l in lbs], device)
        y = forward(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        y = y.float()
        maxscores.extend(y[:, 4:].amax(dim=(1, 2)).tolist())
        out = non_max_suppression(y, conf, iou, multi_label=True, max_det=max_det, max_time_img=1e4)  # no NMS time limit: the default silently truncates detections when the GPU is shared
        for p, (_, r, left, top), o in zip(chunk, lbs, out):
            iid = int(os.path.basename(p)[:-4])
            o = o.cpu().numpy()
            for x1, y1, x2, y2, s, c in o[:, :6]:
                x1, x2 = (x1 - left) / r, (x2 - left) / r
                y1, y2 = (y1 - top) / r, (y2 - top) / r
                dets.append({"image_id": iid, "category_id": int(c) + 1,
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return dets, {"max_score_mean": float(np.mean(maxscores)), "max_score_max": float(np.max(maxscores))}


_GT = {}


def coco_eval(dets, split="val"):
    import contextlib, io
    gtp = GT_VAL if split == "val" else GT_TEST
    if gtp not in _GT:
        with contextlib.redirect_stdout(io.StringIO()):
            _GT[gtp] = COCO(gtp)
    gt = _GT[gtp]
    res = {}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets) if dets else None
        E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
        for k, v in zip(["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"], E.stats[:6]):
            res[k] = round(float(v) * 100, 2)
        res["AR100"] = round(float(E.stats[8]) * 100, 2)
        E2 = COCOeval(gt, dt, "bbox")
        E2.params.areaRng = [[0, 1e10]] + [[lo * lo, hi * hi] for _, lo, hi in BINS]
        E2.params.areaRngLbl = ["all"] + [b[0] for b in BINS]
        E2.evaluate(); E2.accumulate()
        P = E2.eval["precision"]  # T,R,K,A,M
        Rc = E2.eval["recall"]    # T,K,A,M
        for a, (lbl, _, _) in enumerate(BINS, start=1):
            p = P[:, :, :, a, -1]; p = p[p > -1]
            r = Rc[:, :, a, -1]; r = r[r > -1]
            res[f"AP_{lbl}"] = round(float(p.mean()) * 100, 2) if p.size else float("nan")
            res[f"AR_{lbl}"] = round(float(r.mean()) * 100, 2) if r.size else float("nan")
    return res
