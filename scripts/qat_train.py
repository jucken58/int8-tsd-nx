"""QAT fine-tuning from an existing FP32 YOLO11n checkpoint with Ultralytics' built-in ModelOpt QAT (quantize=8).
Built-in behaviour (ultralytics 8.4.166, utils/torch_utils.prepare_qat): INT8_DEFAULT_CFG (W8 per-channel, A8 per-tensor,
max calibration on 8 train batches), BN kept unfused during training, head output convs + DFL kept float.
Usage: python qat_train.py --imgsz 896 --seed 0 --lr0 1e-3 --epochs 20 --name q896_s0_lr1e-3_e20 [--fraction 1.0]
"""
import argparse, json, os, shutil, time
import qlib

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr0", type=float, default=1e-3)
ap.add_argument("--lrf", type=float, default=0.1)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--close_mosaic", type=int, default=-1, help="-1 -> max(1, epochs//4)")
ap.add_argument("--warmup", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--quantize", default="8", help="8 for QAT, 'none' for an FP32 fine-tune control")
ap.add_argument("--name", required=True)
ap.add_argument("--project", default="./work/qatc/runs/qat")
ap.add_argument("--cache", default="ram")
ap.add_argument("--variant", default="base", choices=["base", "c1", "c2", "c3"])
ap.add_argument("--lam", type=float, default=2.0)
ap.add_argument("--dis", type=float, default=1.0)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

src = qlib.FP32[(a.imgsz, a.seed)]
os.makedirs("./work/qatc/runs/init", exist_ok=True)
init = f"./work/qatc/runs/init/y11n_{a.imgsz}_s{a.seed}.pt"
if not os.path.exists(init):
    shutil.copy(src, init)  # never train from a path inside edgets/

from ultralytics import YOLO, settings
settings.update({"weights_dir": "./work/edgets/weights"} if os.path.isdir("./work/edgets/weights") else {})
kw = dict(data=f"{qlib.DATA}/TT100K45.yaml", imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, optimizer="SGD",
          lr0=a.lr0, lrf=a.lrf, momentum=0.937, weight_decay=5e-4, warmup_epochs=a.warmup,
          close_mosaic=(max(1, a.epochs // 4) if a.close_mosaic < 0 else a.close_mosaic), fliplr=0.0,
          seed=a.seed, deterministic=True, workers=a.workers, cache=(False if a.cache == "False" else a.cache),
          fraction=a.fraction, project=a.project, name=a.name, exist_ok=False, plots=False, amp=True, patience=1000)
if a.quantize != "none":
    kw["quantize"] = int(a.quantize)
if a.variant == "c2":
    kw["mosaic"] = 0.0
model = YOLO(init)
if a.variant == "c1":
    import functools, ultralytics.engine.trainer as T, g5_variants
    T.prepare_qat = functools.partial(g5_variants.prepare_qat_percentile, imgsz=a.imgsz)
if a.variant == "c3":
    import g5_variants
    os.makedirs(f"{a.project}/{a.name}", exist_ok=True)
    model.add_callback("on_train_start", lambda tr: g5_variants.attach_distill(tr, init, a.lam, a.dis, f"{a.project}/{a.name}/distill_log.jsonl"))
meta = {"init": init, "src": src, "args": vars(a), "train_kw": kw, "start": time.strftime("%F %T")}
t0 = time.time()
model.train(**kw)
meta["wall_s"] = round(time.time() - t0, 1)
json.dump(meta, open(f"{a.project}/{a.name}/qat_meta.json", "w"), indent=1)
print("DONE", meta["wall_s"])
