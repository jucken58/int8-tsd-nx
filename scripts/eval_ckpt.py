"""Evaluate a (QAT or FP32) Ultralytics checkpoint on val/test with the direction-C protocol (qlib).
Optional --trt: export ONNX (Q/DQ for QAT checkpoints), build TRT10 engine (INT8+FP16 flags), evaluate it, dump layer precisions.
"""
import argparse, json, os, shutil, time
import qlib

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="./work/qatc/out/eval")
ap.add_argument("--trt", action="store_true")
ap.add_argument("--save_dets", action="store_true")
a = ap.parse_args()
assert a.split == "val" or os.environ.get("ALLOW_TEST") == "1", "test is opened only in the final step"
os.makedirs(a.out, exist_ok=True)
cache = qlib.build_cache(a.imgsz, a.split)
m = qlib.load_fp32(a.ckpt)
from ultralytics.utils.torch_utils import is_qat
rows = []


def log(cfg, met, st, extra=None):
    row = {"tag": a.tag, "ckpt": a.ckpt, "imgsz": a.imgsz, "split": a.split, "cfg": cfg, "is_qat": bool(is_qat(m)), **met, **st,
           **(extra or {}), "time": time.strftime("%F %T")}
    open(f"{a.out}/eval_{a.split}.jsonl", "a").write(json.dumps(row) + "\n")
    print(cfg, {k: met[k] for k in ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")}, flush=True)


dets, st = qlib.predict_coco(m, a.imgsz, split=a.split, cache=cache)
met = qlib.coco_eval(dets, a.split)
dets2, _ = qlib.predict_coco(m, a.imgsz, split=a.split, cache=cache)  # guard: transient wrong evals were seen under GPU sharing
met2 = qlib.coco_eval(dets2, a.split)
log("torch_fakequant" if is_qat(m) else "torch_fp32", met, st, {"AP_rep2": met2["AP"], "rep_consistent": abs(met["AP"] - met2["AP"]) < 0.05})
if a.save_dets:
    json.dump(dets, open(f"{a.out}/{a.tag}_{a.split}_torch_dets.json", "w"))
if a.trt:
    import trtlib
    from ultralytics import YOLO
    d = f"{a.out}/{a.tag}_export"; os.makedirs(d, exist_ok=True)
    local = f"{d}/model.pt"; shutil.copy(a.ckpt, local)
    onnx = YOLO(local).export(format="onnx", imgsz=a.imgsz, batch=1, opset=17, dynamic=False)
    eng = f"{d}/model_int8.engine"
    bs = trtlib.build(onnx, eng, int8=bool(is_qat(m)), fp16=True)
    lp = trtlib.layer_precisions(eng); json.dump(lp, open(f"{d}/layers.json", "w"), indent=0)
    n_int8 = sum(1 for L in lp if any("Int8" in str(o) for o in L["out"]))
    dets, st = qlib.predict_coco(trtlib.Runner(eng), a.imgsz, split=a.split, cache=cache)
    log("trt10_engine", qlib.coco_eval(dets, a.split), st, {"build_s": bs, "n_layers": len(lp), "n_int8_out_layers": n_int8})
print("DONE")
