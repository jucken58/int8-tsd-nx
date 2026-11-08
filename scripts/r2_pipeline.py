"""Revision R2: single-stream pipeline timing on the NX with per-frame records and two pre-processing variants.
--pre cpu : CPU letterbox (cv2 INTER_LINEAR + pad) -> pinned upload -> GPU normalisation   (same as r1_pipeline.py)
--pre gpu : pinned upload of the full 2048x2048 uint8 frame -> GPU bilinear resize (torch, no antialias) + pad -> normalisation
Engine and post-processing are identical to r1_pipeline.py. Per-frame stage times are written to <out>.frames.json.
"""
import argparse, json, time
import cv2, numpy as np, tensorrt as trt, torch, torch.nn.functional as F
import qlib
from ultralytics.utils.nms import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True); ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pre", default="cpu", choices=["cpu", "gpu"])
ap.add_argument("--n", type=int, default=150); ap.add_argument("--reps", type=int, default=4); ap.add_argument("--warm", type=int, default=50)
ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args()
LOG = trt.Logger(trt.Logger.WARNING)
eng = trt.Runtime(LOG).deserialize_cuda_engine(open(a.engine, "rb").read()); ctx = eng.create_execution_context()
ii = [i for i in range(eng.num_bindings) if eng.binding_is_input(i)][0]
oi = [i for i in range(eng.num_bindings) if not eng.binding_is_input(i)][0]
idt = torch.float16 if eng.get_binding_dtype(ii) == trt.float16 else torch.float32
odt = torch.float16 if eng.get_binding_dtype(oi) == trt.float16 else torch.float32
xin = torch.empty(tuple(eng.get_binding_shape(ii)), dtype=idt, device="cuda")
y = torch.empty(tuple(eng.get_binding_shape(oi)), dtype=odt, device="cuda")
bind = [0] * eng.num_bindings; bind[ii] = int(xin.data_ptr()); bind[oi] = int(y.data_ptr())
S = a.imgsz
paths = qlib.split_images("val")[:a.n]
frames = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths]
H, W = frames[0].shape[:2]
assert all(f.shape[:2] == (H, W) for f in frames)
r = min(S / H, S / W); nh, nw = round(H * r), round(W * r); top, left = (S - nh) // 2, (S - nw) // 2
pin_s = torch.empty((S, S, 3), dtype=torch.uint8).pin_memory()
pin_f = torch.empty((H, W, 3), dtype=torch.uint8).pin_memory()
canvas = torch.full((1, 3, S, S), 114 / 255.0, dtype=idt, device="cuda")
sync = torch.cuda.synchronize


def pre_cpu(im):
    lb, _, _, _ = qlib.letterbox(im, S)
    pin_s.numpy()[...] = lb
    g = pin_s.to("cuda", non_blocking=True)
    xin.copy_(g.flip(-1).permute(2, 0, 1).unsqueeze(0).to(idt).div_(255.0))


def pre_gpu(im):
    pin_f.numpy()[...] = im
    g = pin_f.to("cuda", non_blocking=True).flip(-1).permute(2, 0, 1).unsqueeze(0).to(idt).div_(255.0)
    rs = F.interpolate(g, size=(nh, nw), mode="bilinear", align_corners=False)
    xin.copy_(canvas); xin[:, :, top:top + nh, left:left + nw] = rs


pre = pre_cpu if a.pre == "cpu" else pre_gpu


def run(im):
    t0 = time.perf_counter(); pre(im); sync(); t1 = time.perf_counter()
    ctx.execute_v2(bind); sync(); t2 = time.perf_counter()
    out = non_max_suppression(y.float(), a.conf, 0.7, multi_label=False, max_det=300, max_time_img=1e4)[0]
    b = out[:, :4].clone(); b[:, [0, 2]] = (b[:, [0, 2]] - left) / r; b[:, [1, 3]] = (b[:, [1, 3]] - top) / r
    res = torch.cat([b, out[:, 4:6]], 1).cpu().numpy(); t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2, len(res)


for k in range(a.warm):
    run(frames[k % len(frames)])
rec = []; ts = time.time(); tp0 = time.perf_counter()
for rep in range(a.reps):
    for im in frames:
        rec.append(run(im))
tp1 = time.perf_counter(); te = time.time()
R = np.array(rec); tot = R[:, :3].sum(1) * 1000
q = lambda v, p: float(np.percentile(v, p))
row = {"tag": a.tag, "pre": a.pre, "engine": a.engine, "imgsz": S, "conf": a.conf, "frames": len(rec), "wall_start": ts, "wall_end": te,
       "fps": len(rec) / (tp1 - tp0), "dets_mean": float(R[:, 3].mean()),
       "miss_33ms": float((tot > 1000 / 30).mean()), "total_ms_mean": float(tot.mean()), "total_ms_p50": q(tot, 50),
       "total_ms_p95": q(tot, 95), "total_ms_p99": q(tot, 99), "total_ms_max": float(tot.max())}
for j, n in enumerate(["pre", "infer", "post"]):
    v = R[:, j] * 1000; row[f"{n}_ms_mean"] = float(v.mean()); row[f"{n}_ms_p50"] = q(v, 50); row[f"{n}_ms_p95"] = q(v, 95)
open(a.out, "a").write(json.dumps(row) + "\n")
json.dump({"tag": a.tag, "pre": a.pre, "ms": R[:, :3].round(6).tolist(), "dets": R[:, 3].astype(int).tolist()},
          open(a.out + f".{a.tag}.{a.pre}.frames.json", "w"))
print(json.dumps(row), flush=True)
