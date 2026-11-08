"""Export static b1 ONNX files for the NX (TensorRT 8.5): fp32 (plain), ptq_{full,E7,greedy8} (explicit Q/DQ from ModelOpt
fake-quant, percentile-99.99, 512 random train images, same as G2), qat (Ultralytics QAT checkpoint, Q/DQ).
Also re-evaluates each exported model in PyTorch (protocol §4) so NX accuracy can be checked against it.
Usage: python export_onnx.py --imgsz 896 --seed 0 --kinds fp32,ptq_full,ptq_E7,ptq_greedy8,qat --qat_ckpt <best.pt> --out ../out/nx_onnx
"""
import argparse, copy, json, os, shutil, hashlib
import torch
import qlib

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kinds", default="fp32,ptq_full,ptq_E7,ptq_greedy8,qat"); ap.add_argument("--qat_ckpt", default="")
ap.add_argument("--rank", default="")
ap.add_argument("--nq_regex", default="", help="extra modules kept un-quantized (TensorRT 8.5 deployment constraint)"); ap.add_argument("--out", required=True); ap.add_argument("--tagsuffix", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from ultralytics import YOLO
src = qlib.FP32[(a.imgsz, a.seed)]
cache = qlib.build_cache(a.imgsz, "val")
log = open(f"{a.out}/export_log.jsonl", "a")


def export(yolo, name):
    d = f"{a.out}/{name}"; os.makedirs(d, exist_ok=True)
    p = yolo.export(format="onnx", imgsz=a.imgsz, batch=1, opset=17, dynamic=False, simplify=True)
    dst = f"{d}/model.onnx"; shutil.move(p, dst)
    return dst, hashlib.sha256(open(dst, "rb").read()).hexdigest()


def evaluate(m):
    dets, st = qlib.predict_coco(m, a.imgsz, cache=cache)
    return qlib.coco_eval(dets, "val")


def ptq_model(exc):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer
    base = qlib.load_fp32(src)
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
        q.load_calib_amax() if k.endswith("weight_quantizer") else q.load_calib_amax(method="percentile", percentile=99.99)
        q.enable_quant(); q.disable_calib()
    qm.cuda()
    ex = set(exc)
    for k, q in Q.items():
        if k.rsplit(".", 1)[0] in ex: q.disable()
    return qm


import re
NQ = re.compile(a.nq_regex) if a.nq_regex else None


def drop_nq(m):
    """Disable fake-quant on modules matching --nq_regex; returns the list of affected module names."""
    if NQ is None:
        return []
    from modelopt.torch.quantization.nn import TensorQuantizer
    hit = sorted({k.rsplit(".", 1)[0] for k, q in m.named_modules() if isinstance(q, TensorQuantizer) and NQ.search(k.rsplit(".", 1)[0])})
    for k, q in m.named_modules():
        if isinstance(q, TensorQuantizer) and k.rsplit(".", 1)[0] in hit:
            q.disable()
    return hit


convs_det = None
for kind in a.kinds.split(","):
    name = f"y11n_{a.imgsz}_s{a.seed}_{kind}{a.tagsuffix}"
    hit = []
    if kind == "fp32":
        y = YOLO(src); met = evaluate(qlib.load_fp32(src))
    elif kind == "qat":
        y = YOLO(a.qat_ckpt); hit = drop_nq(y.model)
        mq = qlib.load_fp32(a.qat_ckpt); drop_nq(mq); met = evaluate(mq)
    else:
        sub = kind.split("_", 1)[1]
        if sub == "full": exc = []
        elif sub == "E7":
            exc = [f"model.23.cv2.{i}.2" for i in range(3)] + [f"model.23.cv3.{i}.2" for i in range(3)] + ["model.23.dfl.conv"]
        elif sub == "greedy8": exc = json.load(open(a.rank))["rank"][:8]
        qm = ptq_model(exc); hit = drop_nq(qm); met = evaluate(qm)
        y = YOLO(src); y.model = qm
    path, sha = export(y, name)
    row = {"name": name, "kind": kind, "imgsz": a.imgsz, "seed": a.seed, "onnx": path, "sha256": sha, "torch_val": met, "nq_regex": a.nq_regex, "nq_modules": hit}
    log.write(json.dumps(row) + "\n"); log.flush()
    print(name, met["AP"], met["AP_S"], os.path.getsize(path), flush=True)
print("DONE")
