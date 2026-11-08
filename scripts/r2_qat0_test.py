"""Revision R2: one-time TEST evaluation of the 'QAT set-up without fine-tuning' checkpoints from r1_qat0.py (CPU).
Frozen configuration, evaluated once; no selection. Usage: ALLOW_TEST=1 python r2_qat0_test.py --ckpt last.pt --imgsz 896 --seed 0 --out f.jsonl
"""
import argparse, json, os, time
import torch
import qlib

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=5)
a = ap.parse_args()
assert os.environ.get("ALLOW_TEST") == "1"
torch.set_num_threads(a.threads)
m = qlib.load_fp32(a.ckpt, device="cpu")
from ultralytics.utils.torch_utils import is_qat
assert is_qat(m)
dets, st = qlib.predict_coco(m, a.imgsz, split="test", device="cpu", bs=8)
met = qlib.coco_eval(dets, "test")
row = {"tag": f"qat0_{a.imgsz}_s{a.seed}", "imgsz": a.imgsz, "seed": a.seed, "split": "test", **met, "time": time.strftime("%F %T")}
open(a.out, "a").write(json.dumps(row) + "\n"); print(row, flush=True)
