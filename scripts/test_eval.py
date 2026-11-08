"""One-time TEST evaluation of the frozen direction-C configurations (preregistration §2: test opened once, after freezing).
Frozen configs (decided on val, 2026-10-31): FP32 original, PTQ-E7 (per-seed activation method as selected on val in g4_ptq_v2),
PTQ-greedy8 (pct99.99, s0 v2 LOO ranking), official QAT (lr0 2e-4, 20 ep); YOLO11n at 896/1024, seeds 0-4.
Calibration identical to val runs (512 random train images, seed 0). Each config is evaluated exactly once on test.
"""
import argparse, copy, json, os, time
import torch
import qlib

os.environ["ALLOW_TEST"] = "1"
ap = argparse.ArgumentParser(); ap.add_argument("--imgsz", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
E7_METHOD = {896: {0: 99.99, 1: 99.99, 2: 99.99, 3: 99.9, 4: 99.999}, 1024: {0: 99.99, 1: 99.99, 2: 99.999, 3: 99.999, 4: 99.99}}[a.imgsz]
RANK = json.load(open(f"./work/qatc/out/g2v2/y11n_{a.imgsz}_s0_random_loo_rank.json"))["rank"][:8]
E7 = [f"model.23.cv2.{i}.2" for i in range(3)] + [f"model.23.cv3.{i}.2" for i in range(3)] + ["model.23.dfl.conv"]
cache = qlib.build_cache(a.imgsz, "test")
done = set()
if os.path.exists(a.out):
    done = {(r["seed"], r["cfg"]) for r in map(json.loads, open(a.out))}


def ev(m, seed, cfg, extra=None):
    if (seed, cfg) in done:
        print("skip (already evaluated once)", seed, cfg); return
    t0 = time.time()
    dets, st = qlib.predict_coco(m, a.imgsz, split="test", cache=cache)
    met = qlib.coco_eval(dets, "test")
    row = {"imgsz": a.imgsz, "seed": seed, "cfg": cfg, "split": "test", **met, **st, **(extra or {}), "wall_s": round(time.time() - t0, 1), "time": time.strftime("%F %T")}
    open(a.out, "a").write(json.dumps(row) + "\n")
    print(seed, cfg, met["AP"], met["AP_S"], flush=True)


def ptq(pt):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer
    base = qlib.load_fp32(pt)
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for e in cfg["quant_cfg"]:
        if e.get("quantizer_name") == "*input_quantizer":
            e["cfg"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
    cfg["algorithm"] = None
    qm = mtq.quantize(base, cfg, forward_loop=None)
    Q = {k: q for k, q in qm.named_modules() if isinstance(q, TensorQuantizer) and q.is_enabled}
    for q in Q.values(): q.disable_quant(); q.enable_calib()
    with torch.no_grad():
        for x in qlib.CalibLoader(qlib.calib_images(512, 0, "random"), a.imgsz): qm(x)
    for k, q in Q.items():
        if k.endswith("weight_quantizer"): q.load_calib_amax()
        q.enable_quant(); q.disable_calib()
    return qm, Q


def set_cfg(qm, Q, pct, exc):
    for k, q in Q.items():
        if k.endswith("input_quantizer"): q.load_calib_amax(method="percentile", percentile=pct)
    qm.cuda()
    ex = set(exc)
    for k, q in Q.items():
        (q.disable_quant if k.rsplit(".", 1)[0] in ex else q.enable_quant)()


for sd in range(5):
    pt = qlib.FP32[(a.imgsz, sd)]
    ev(qlib.load_fp32(pt), sd, "fp32")
    qat = f"./work/qatc/runs/sweep/q{a.imgsz}_s{sd}_lr0.0002_e20/weights/best.pt"
    ev(qlib.load_fp32(qat), sd, "qat_lr2e-4_e20", {"ckpt": qat})
    qm, Q = ptq(pt)
    set_cfg(qm, Q, E7_METHOD[sd], E7); ev(qm, sd, "ptq_E7", {"pct": E7_METHOD[sd]})
    set_cfg(qm, Q, 99.99, RANK); ev(qm, sd, "ptq_greedy8", {"pct": 99.99, "fp16_convs": RANK})
    del qm, Q; torch.cuda.empty_cache()
print("DONE")
