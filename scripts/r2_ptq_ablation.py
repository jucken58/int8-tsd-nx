"""Revision R2 control analyses (CPU, fake quant, validation; no configuration selection):
(a) E7 exception set with max ranges vs percentile-99.99 ranges (512 random letterboxed train images, BN fused),
    per seed -> explains why the QAT set-up without fine-tuning (E7 + max on augmented batches) beats the best PTQ.
(b) seed 0 only: weight vs input quantizer of the DFL conv and of the P3 box-output conv, starting from full INT8 (pct 99.99).
Usage: python r2_ptq_ablation.py --imgsz 896 --seed 0 --pt best.pt --out dir
"""
import argparse, copy, json, os, time
import torch
import qlib
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--pt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=5)
a = ap.parse_args()
torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True)
dev = "cpu"; tag = f"y11n_{a.imgsz}_s{a.seed}"
model = qlib.load_fp32(a.pt, device=dev)
cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
for e in (cfg["quant_cfg"] if isinstance(cfg["quant_cfg"], list) else []):
    if e.get("quantizer_name") == "*input_quantizer":
        e["cfg"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
if isinstance(cfg["quant_cfg"], dict):
    cfg["quant_cfg"]["*input_quantizer"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
cfg["algorithm"] = None
qm = mtq.quantize(copy.deepcopy(model), cfg, forward_loop=None)
Q = {n: q for n, q in qm.named_modules() if isinstance(q, TensorQuantizer) and q.is_enabled}
for q in Q.values():
    q.disable_quant(); q.enable_calib()
with torch.no_grad():
    for x in qlib.CalibLoader(qlib.calib_images(512, 0, "random"), a.imgsz, device=dev):
        qm(x)
for n, q in Q.items():
    if n.endswith("weight_quantizer"):
        q.load_calib_amax()
    q.enable_quant(); q.disable_calib()
inputs = {n: q for n, q in Q.items() if n.endswith("input_quantizer")}
amax = {}
for name, pct in [("pct99.99", 99.99), ("max", 100.0)]:
    for n, q in inputs.items():
        q.load_calib_amax(method="percentile", percentile=pct)
    amax[name] = {n: q.amax.detach().clone() for n, q in inputs.items()}
E7 = ["model.23.dfl.conv"] + [f"model.23.cv{b}.{i}.2" for b in (2, 3) for i in range(3)]
DFL, P3 = "model.23.dfl.conv", "model.23.cv2.0.2"
convs = {n.rsplit(".", 1)[0] for n in Q}
assert all(c in convs for c in E7), [c for c in E7 if c not in convs]
cache = qlib.build_cache(a.imgsz, "val")


def setup(rng, off):  # off: set of quantizer names to disable
    for n, q in inputs.items():
        q.amax = amax[rng][n]
    for n, q in Q.items():
        (q.disable_quant if n in off else q.enable_quant)()


def ev(cfgname, rng, off):
    setup(rng, off)
    dets, st = qlib.predict_coco(qm, a.imgsz, device=dev, cache=cache, bs=8)
    met = qlib.coco_eval(dets, "val")
    row = {"model": tag, "imgsz": a.imgsz, "seed": a.seed, "cfg": cfgname, **met, "time": time.strftime("%F %T")}
    open(f"{a.out}/r2_ptq_ablation.jsonl", "a").write(json.dumps(row) + "\n"); print(cfgname, met["AP"], met["AP_S"], flush=True)


both = lambda c: {f"{c}.weight_quantizer", f"{c}.input_quantizer"}
e7off = set().union(*[both(c) for c in E7])
ev("E7_max", "max", e7off)
ev("E7_pct99.99", "pct99.99", e7off)
if a.seed == 0:
    ev("full_pct99.99", "pct99.99", set())
    for c, lab in [(DFL, "dfl"), (P3, "p3")]:
        ev(f"{lab}_weight_fp", "pct99.99", {f"{c}.weight_quantizer"})
        ev(f"{lab}_input_fp", "pct99.99", {f"{c}.input_quantizer"})
        ev(f"{lab}_both_fp", "pct99.99", both(c))
    print("DFL input amax pct99.99 / max:", float(amax["pct99.99"][f"{DFL}.input_quantizer"]), float(amax["max"][f"{DFL}.input_quantizer"]), flush=True)
print("DONE")
