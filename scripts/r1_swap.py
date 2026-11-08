"""Revision R1 (reviewer comment 03), post hoc: direct attribution of the INT8 loss to boxes vs scores.
For YOLO11n s0 (val), FP32 vs W8A8 fake-quant (percentile 99.99, 512 random train calib images, same as G2):
  exempt sets E in {none, dfl, p3, dfl+p3}; for each E three detectors are evaluated on the same pre-NMS predictions:
    q      : INT8 boxes + INT8 scores
    qbox   : INT8 boxes + FP32 scores   (only the box regression is quantized)
    qcls   : FP32 boxes + INT8 scores   (only the class scores are quantized)
  plus a per-GT IoU analysis at a fixed anchor: the anchor chosen by FP32 (max FP32 score of the GT class among anchors with
  FP32 IoU >= 0.5); IoU with the GT is recorded for FP32 and for each INT8 config at that same anchor.
Runs on CPU (Mac). Usage: python r1_swap.py --imgsz 896 --pt best.pt --out dir
"""
import argparse, copy, json, os, time
import numpy as np, torch
import qlib
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--pt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--threads", type=int, default=5)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
dev = "cpu"
tag = f"y11n_{a.imgsz}_s0"

model = qlib.load_fp32(a.pt, device=dev)
cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
if isinstance(cfg["quant_cfg"], dict):
    cfg["quant_cfg"]["*input_quantizer"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
else:
    for e in cfg["quant_cfg"]:
        if e.get("quantizer_name") == "*input_quantizer":
            e["cfg"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
cfg["algorithm"] = None
qm = mtq.quantize(copy.deepcopy(model), cfg, forward_loop=None)
Q = {n: q for n, q in qm.named_modules() if isinstance(q, TensorQuantizer) and q.is_enabled}
for q in Q.values():
    q.disable_quant(); q.enable_calib()
t0 = time.time()
with torch.no_grad():
    for x in qlib.CalibLoader(qlib.calib_images(512, 0, "random"), a.imgsz, device=dev):
        qm(x)
for n, q in Q.items():
    if n.endswith("weight_quantizer"):
        q.load_calib_amax()
    else:
        q.load_calib_amax(method="percentile", percentile=99.99)
    q.enable_quant(); q.disable_calib()
print("calibrated", round(time.time() - t0), "s", flush=True)
EX = {"none": [], "dfl": ["model.23.dfl.conv"], "p3": ["model.23.cv2.0.2"], "dfl+p3": ["model.23.dfl.conv", "model.23.cv2.0.2"]}
convs = {n.rsplit(".", 1)[0] for n in Q}
for v in EX.values():
    for c in v:
        assert c in convs, c


def set_exempt(ex):
    ex = set(ex)
    for n, q in Q.items():
        (q.disable_quant if n.rsplit(".", 1)[0] in ex else q.enable_quant)()


def raw(m, x):
    y = m(x)
    return (y[0] if isinstance(y, (list, tuple)) else y).float()


def xywh2xyxy(b):
    return torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def iou_1toN(g, B):
    lt = torch.maximum(B[:, :2], g[:2]); rb = torch.minimum(B[:, 2:], g[2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    return inter / ((B[:, 2:] - B[:, :2]).prod(1) + (g[2:] - g[:2]).prod() - inter + 1e-9)


from ultralytics.utils.nms import non_max_suppression
gt = json.load(open(qlib.GT_VAL))
G = {}
for an in gt["annotations"]:
    G.setdefault(an["image_id"], []).append(an)
ims, lbs = qlib.build_cache(a.imgsz, "val")
combos = ["fp32"] + [f"{k}|{c}" for k in EX for c in ("q", "qbox", "qcls")]
dets = {c: [] for c in combos}
iou_rows = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(ims), a.bs):
        chunk, L = ims[i:i + a.bs], lbs[i:i + a.bs]
        x = qlib.to_tensor([l[0] for l in L], dev)
        yf = raw(model, x)
        ys = {"fp32": yf}
        yq = {}
        for k, ex in EX.items():
            set_exempt(ex); yq[k] = raw(qm, x)
            ys[f"{k}|q"] = yq[k]
            ys[f"{k}|qbox"] = torch.cat([yq[k][:, :4], yf[:, 4:]], 1)
            ys[f"{k}|qcls"] = torch.cat([yf[:, :4], yq[k][:, 4:]], 1)
        for c, y in ys.items():
            out = non_max_suppression(y, 0.001, 0.7, multi_label=True, max_det=300, max_time_img=1e4)
            for p, (_, r, left, top), o in zip(chunk, L, out):
                iid = int(os.path.basename(p)[:-4]); o = o.numpy()
                for x1, y1, x2, y2, s, cc in o[:, :6]:
                    dets[c].append({"image_id": iid, "category_id": int(cc) + 1, "score": float(s),
                                    "bbox": [float((x1 - left) / r), float((y1 - top) / r), float((x2 - x1) / r), float((y2 - y1) / r)]})
        # fixed-anchor IoU analysis
        for b, (p, (_, r, left, top)) in enumerate(zip(chunk, L)):
            iid = int(os.path.basename(p)[:-4])
            Bf = xywh2xyxy(yf[b, :4].T)
            Bq = {k: xywh2xyxy(yq[k][b, :4].T) for k in EX}
            for an in G.get(iid, []):
                gx, gy, gw, gh = an["bbox"]
                g = torch.tensor([gx * r + left, gy * r + top, (gx + gw) * r + left, (gy + gh) * r + top])
                iou_f = iou_1toN(g, Bf)
                cand = torch.nonzero(iou_f >= 0.5).flatten()
                row = {"image_id": iid, "size": float(np.sqrt(an["area"])), "matched": int(len(cand) > 0)}
                if len(cand):
                    sc = yf[b, 4 + an["category_id"] - 1, cand]
                    j = int(cand[int(torch.argmax(sc))])
                    row["iou_fp32"] = float(iou_f[j]); row["score_fp32"] = float(yf[b, 4 + an["category_id"] - 1, j])
                    for k in EX:
                        row[f"iou_{k}"] = float(iou_1toN(g, Bq[k][j:j + 1])[0])
                        row[f"score_{k}"] = float(yq[k][b, 4 + an["category_id"] - 1, j])
                iou_rows.append(row)
        if i % (a.bs * 10) == 0:
            print(i, len(ims), round(time.time() - t0), "s", flush=True)
res = []
for c in combos:
    met = qlib.coco_eval(dets[c], "val")
    res.append({"model": tag, "combo": c, **met})
    print(c, met["AP"], met["AP50"], met["AP75"], met["AP_S"], flush=True)
json.dump(res, open(f"{a.out}/{tag}_swap_metrics.json", "w"), indent=1)
json.dump(iou_rows, open(f"{a.out}/{tag}_anchor_iou.json", "w"))
print("DONE", round(time.time() - t0), "s")
