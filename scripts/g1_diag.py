"""G1: INT8 fake-quant diagnosis (ModelOpt, W8 per-channel / A8 per-tensor) on TT100K45 val.
One histogram calibration pass per model; activation amax recomputed per method from the same histograms.
Usage: python g1_diag.py --imgsz 896 --seed 0 --out ../out/g1 [--smoke]
"""
import argparse, copy, json, os, re, time
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
import qlib

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pt", default="")
ap.add_argument("--name", default="y11n")
ap.add_argument("--ncalib", type=int, default=512)
ap.add_argument("--calib_mode", default="random")
ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--no_sens", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = f"{a.name}_{a.imgsz}_s{a.seed}"
rows_path = f"{a.out}/{tag}_{a.calib_mode}.jsonl"
fout = open(rows_path, "a")
torch.backends.cudnn.benchmark = True


def log(cfg, met, extra=None):
    row = {"model": tag, "imgsz": a.imgsz, "seed": a.seed, "calib_mode": a.calib_mode, "ncalib": a.ncalib,
           "cfg": cfg, **met, **(extra or {}), "time": time.strftime("%F %T")}
    fout.write(json.dumps(row) + "\n"); fout.flush()
    print(cfg, {k: met[k] for k in ("AP", "AP_S", "AP_lt32", "AP_ge64")}, extra or "", flush=True)


cache = qlib.build_cache(a.imgsz, "val")
if a.smoke:
    cache = (cache[0][:64], cache[1][:64])


def evaluate(model, cfg, extra=None, save=False):
    dets, st = qlib.predict_coco(model, a.imgsz, cache=cache)
    met = qlib.coco_eval(dets, "val") if not a.smoke else {"AP": -1, "AP_S": -1, "AP_lt32": -1, "AP_ge64": -1}
    if save:
        json.dump(dets, open(f"{a.out}/{tag}_{cfg}_dets.json", "w"))
    log(cfg, met, {**st, **(extra or {})})
    return met


pt = (a.pt or qlib.FP32[(a.imgsz, a.seed)])
model = qlib.load_fp32(pt)
HEAD = len(model.model) - 1
evaluate(model, "fp32", save=True)

cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
cfg["quant_cfg"]["*input_quantizer"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
cfg["algorithm"] = None  # calibration done manually below
paths = qlib.calib_images(a.ncalib if not a.smoke else 32, seed=0, mode=a.calib_mode)
loader = qlib.CalibLoader(paths, a.imgsz)


def fwd_loop(m):
    for x in loader:
        m(x)


qm = mtq.quantize(copy.deepcopy(model), cfg, forward_loop=None)  # insert quantizers only
Q = {n: m for n, m in qm.named_modules() if isinstance(m, TensorQuantizer)}
print("quantizers:", len(Q), list(Q)[:6], flush=True)
json.dump(sorted(Q), open(f"{a.out}/{tag}_quantizer_names.json", "w"), indent=0)
# manual calibration: weights max (per channel), inputs histogram
for n, q in Q.items():
    if q.is_enabled:
        q.disable_quant(); q.enable_calib()
with torch.no_grad():
    fwd_loop(qm)
for n, q in Q.items():
    if q.is_enabled:
        if n.endswith("weight_quantizer"):
            q.load_calib_amax()
        q.enable_quant(); q.disable_calib()
inputs = {n: q for n, q in Q.items() if n.endswith("input_quantizer") and q.is_enabled}
weights = {n: q for n, q in Q.items() if n.endswith("weight_quantizer") and q.is_enabled}
print("enabled inputs/weights:", len(inputs), len(weights), flush=True)

METHODS = [("max", {"method": "percentile", "percentile": 100.0}), ("pct99.999", {"method": "percentile", "percentile": 99.999}),
           ("pct99.99", {"method": "percentile", "percentile": 99.99}), ("pct99.9", {"method": "percentile", "percentile": 99.9}),
           ("entropy", {"method": "entropy"}), ("mse", {"method": "mse"})]
if a.smoke:
    METHODS = METHODS[:2] + METHODS[4:5]
if os.environ.get("G1_METHODS"):  # e.g. skip entropy: ModelOpt entropy overflows int32 on large (1280-px) histograms
    METHODS = [m for m in METHODS if m[0] in os.environ["G1_METHODS"].split(",")]
amax_sets = {}
for name, kw in METHODS:
    t0 = time.time()
    for n, q in inputs.items():
        q.load_calib_amax(**kw)
    qm.cuda()  # histogram amax buffers are created on CPU
    amax_sets[name] = {n: q.amax.detach().clone() for n, q in inputs.items()}
    evaluate(qm, f"int8_full_{name}", {"calib_s": round(time.time() - t0, 1)})


def set_amax(name):
    for n, q in inputs.items():
        q.amax = amax_sets[name][n]
    qm.cuda()


def group_of(n):
    m = re.match(r"model\.(\d+)\.(.*)", n)
    i, rest = int(m.group(1)), m.group(2)
    if i == HEAD:  # Detect
        if rest.startswith("dfl"):
            return "det.dfl"
        br = rest.split(".")[0]  # cv2 (box) / cv3 (cls)
        return f"det.{br}"
    return f"m{i:02d}"


groups = sorted({group_of(n) for n in list(inputs) + list(weights)})
json.dump({g: [n for n in list(inputs) + list(weights) if group_of(n) == g] for g in groups},
          open(f"{a.out}/{tag}_groups.json", "w"), indent=0)


def enable_only(sel):
    for n, q in {**inputs, **weights}.items():
        (q.enable_quant if sel(n) else q.disable_quant)()


# select best activation method on val by AP (selection on val is allowed by prereg)
rows = [json.loads(l) for l in open(rows_path)]
cand = [r for r in rows if r["cfg"].startswith("int8_full_") and r["model"] == tag]
best = max(cand, key=lambda r: r["AP"])["cfg"].replace("int8_full_", "") if not a.smoke else "entropy"
set_amax(best)
print("best method:", best, flush=True)
enable_only(lambda n: n.endswith("weight_quantizer")); evaluate(qm, f"w8_only_{best}")
enable_only(lambda n: n.endswith("input_quantizer")); evaluate(qm, f"a8_only_{best}")
enable_only(lambda n: True); evaluate(qm, f"int8_full_{best}_dets", save=True)
if not a.no_sens:
    for g in (groups if not a.smoke else groups[-3:]):
        enable_only(lambda n: group_of(n) == g); evaluate(qm, f"only_{g}_{best}")
        enable_only(lambda n: group_of(n) != g); evaluate(qm, f"allbut_{g}_{best}")
    # whole detect head kept FP
    enable_only(lambda n: not group_of(n).startswith("det.")); evaluate(qm, f"allbut_det_{best}")
torch.save({n: v.cpu() for n, v in amax_sets[best].items()}, f"{a.out}/{tag}_{a.calib_mode}_amax_{best}.pt")
print("DONE", flush=True)
