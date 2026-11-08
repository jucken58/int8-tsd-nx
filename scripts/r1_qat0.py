"""Revision R1 (reviewer comment 05), post hoc: 'QAT at step 0' control.
Runs Ultralytics' own QAT setup (quantize=8: prepare_qat = INT8_DEFAULT max calibration on 8 augmented train batches,
head output convs + DFL left in float, i.e. the E7 exception set) with the same arguments as qat_train.py, then stops
before the first optimisation step, saves the checkpoint and evaluates it with the direction-C protocol.
QAT minus this control isolates the effect of the QAT fine-tuning itself (same exceptions, same initial ranges).
CPU (Mac). Usage: python r1_qat0.py --imgsz 896 --seed 0 --pt best.pt --data TT100K45_local.yaml --out dir
"""
import argparse, json, os, time
import torch
import qlib

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--pt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=5)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
from ultralytics import YOLO


class Stop(Exception):
    pass


def stop_after_setup(trainer):
    trainer.epoch = 0  # not yet defined before the first epoch
    if not hasattr(trainer, 'fitness'): trainer.fitness = None
    trainer.save_model()
    raise Stop()


name = f"qat0_{a.imgsz}_s{a.seed}"
m = YOLO(a.pt)
m.add_callback("on_pretrain_routine_end", stop_after_setup)
kw = dict(data=a.data, imgsz=a.imgsz, epochs=20, batch=32, optimizer="SGD", lr0=2e-4, lrf=0.1, momentum=0.937, weight_decay=5e-4,
          warmup_epochs=0.0, close_mosaic=5, fliplr=0.0, seed=a.seed, deterministic=True, workers=4, cache=False, device="cpu",
          project=a.out, name=name, exist_ok=True, plots=False, amp=False, patience=1000, quantize=8, val=False)
t0 = time.time()
try:
    m.train(**kw)
except Stop:
    pass
ck = f"{a.out}/{name}/weights/last.pt"
assert os.path.exists(ck), ck
print("setup+calib", round(time.time() - t0), "s", flush=True)
model = qlib.load_fp32(ck, device="cpu")
from ultralytics.utils.torch_utils import is_qat
assert is_qat(model)
cache = qlib.build_cache(a.imgsz, "val")
dets, st = qlib.predict_coco(model, a.imgsz, split="val", device="cpu", cache=cache, bs=8)
met = qlib.coco_eval(dets, "val")
row = {"tag": name, "imgsz": a.imgsz, "seed": a.seed, "cfg": "qat_step0_E7_max8aug", **met, **st, "time": time.strftime("%F %T")}
open(f"{a.out}/qat0_val.jsonl", "a").write(json.dumps(row) + "\n")
print(row, flush=True)
