"""G2: strong PTQ baselines (ModelOpt fake-quant W8 per-channel / A8 per-tensor), TT100K45 val.
- calibration set: random | smallrich (512 train images)
- activation amax method: percentile 99.99 / 99.999 / 99.9 / max (histogram, one pass)
- FP16 exceptions (both weight+input quantizers of a conv disabled), cap 8 convs (<=10% of 88):
  fixed sets E_dfl, E_box1x1+dfl, E_box1x1+cls1x1+dfl; leave-one-out per conv; greedy top-k (k<=8) by LOO gain.
"""
import argparse, copy, json, os, time
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
import qlib

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pt", default="")
ap.add_argument("--name", default="y11n")
ap.add_argument("--calib_mode", default="random")
ap.add_argument("--ncalib", type=int, default=512)
ap.add_argument("--out", required=True)
ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--no_loo", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = f"{a.name}_{a.imgsz}_s{a.seed}"
fout = open(f"{a.out}/{tag}_{a.calib_mode}_g2.jsonl", "a")
cache = qlib.build_cache(a.imgsz, "val")


def ev(m, cfg, extra=None):
    dets, st = qlib.predict_coco(m, a.imgsz, cache=cache)
    met = qlib.coco_eval(dets, "val")
    row = {"model": tag, "imgsz": a.imgsz, "seed": a.seed, "calib_mode": a.calib_mode, "cfg": cfg, **met, **st, **(extra or {}),
           "time": time.strftime("%F %T")}
    fout.write(json.dumps(row) + "\n"); fout.flush()
    print(cfg, met["AP"], met["AP_S"], extra or "", flush=True)
    return met


model = qlib.load_fp32((a.pt or qlib.FP32[(a.imgsz, a.seed)]))
fp = ev(model, "fp32")
cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
if isinstance(cfg["quant_cfg"], dict):  # modelopt <= 0.4x dict format
    cfg["quant_cfg"]["*input_quantizer"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
else:  # list format (modelopt 0.47)
    for e in cfg["quant_cfg"]:
        if e.get("quantizer_name") == "*input_quantizer":
            e["cfg"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
cfg["algorithm"] = None
qm = mtq.quantize(copy.deepcopy(model), cfg, forward_loop=None)
Q = {n: q for n, q in qm.named_modules() if isinstance(q, TensorQuantizer) and q.is_enabled}
for q in Q.values():
    q.disable_quant(); q.enable_calib()
with torch.no_grad():
    for x in qlib.CalibLoader(qlib.calib_images(a.ncalib, 0, a.calib_mode), a.imgsz):
        qm(x)
for n, q in Q.items():
    if n.endswith("weight_quantizer"):
        q.load_calib_amax()
    q.enable_quant(); q.disable_calib()
inputs = {n: q for n, q in Q.items() if n.endswith("input_quantizer")}
convs = sorted({n.rsplit(".", 1)[0] for n in Q if n.endswith("weight_quantizer")})
print("convs:", len(convs), flush=True)
json.dump(convs, open(f"{a.out}/{tag}_convs.json", "w"), indent=0)

amax = {}
for name, kw in [("pct99.99", dict(method="percentile", percentile=99.99)), ("pct99.999", dict(method="percentile", percentile=99.999)),
                 ("pct99.9", dict(method="percentile", percentile=99.9)), ("max", dict(method="percentile", percentile=100.0))]:
    for n, q in inputs.items():
        q.load_calib_amax(**kw)
    qm.cuda()
    amax[name] = {n: q.amax.detach().clone() for n, q in inputs.items()}
    ev(qm, f"full_{name}", {"n_fp16": 0})


def use(name):
    for n, q in inputs.items():
        q.amax = amax[name][n]
    qm.cuda()


def except_convs(ex):
    ex = set(ex)
    for n, q in Q.items():
        (q.disable_quant if n.rsplit(".", 1)[0] in ex else q.enable_quant)()


rows = [json.loads(l) for l in open(f"{a.out}/{tag}_{a.calib_mode}_g2.jsonl")]
best = max([r for r in rows if r["cfg"].startswith("full_") and r["model"] == tag], key=lambda r: r["AP"])["cfg"][5:]
use(best)
print("best amax method:", best, flush=True)
H = f"model.{len(model.model) - 1}."
det = [c for c in convs if c.startswith(H)]
box1 = [c for c in det if c.startswith(H + "cv2.") and c.endswith(".2")]
cls1 = [c for c in det if c.startswith(H + "cv3.") and c.endswith(".2")]
dfl = [c for c in det if "dfl" in c]
print("box1x1", box1, "cls1x1", cls1, "dfl", dfl, flush=True)
FIX = {"E_dfl": dfl, "E_box1x1_dfl": box1 + dfl, "E_box1x1_cls1x1_dfl": box1 + cls1 + dfl,
       "E_allbox_dfl": [c for c in det if c.startswith(H + "cv2.")] + dfl, "E_allhead": det}
for k, ex in FIX.items():
    except_convs(ex); ev(qm, f"{k}_{best}", {"n_fp16": len(ex), "fp16_convs": ex})
if not a.no_loo:
    full_ap = [r for r in rows if r["cfg"] == f"full_{best}"][0]["AP"]
    gain = {}
    for cv in convs:
        except_convs([cv]); m = ev(qm, f"loo_{cv}_{best}", {"n_fp16": 1})
        gain[cv] = m["AP"] - full_ap
    rank = sorted(convs, key=lambda c: -gain[c])
    json.dump({"rank": rank, "gain": gain}, open(f"{a.out}/{tag}_{a.calib_mode}_loo_rank.json", "w"), indent=0)
    for k in range(1, a.kmax + 1):
        except_convs(rank[:k]); ev(qm, f"greedy{k}_{best}", {"n_fp16": k, "fp16_convs": rank[:k]})
print("DONE", flush=True)
