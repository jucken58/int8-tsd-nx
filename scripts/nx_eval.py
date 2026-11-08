"""Evaluate a TensorRT 8.5 engine on the NX with the direction-C protocol (qlib §4). Bindings API (TRT 8.5)."""
import argparse, json, os, time
import tensorrt as trt, torch
import qlib

ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True); ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--split", default="val"); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert a.split == "val" or os.environ.get("ALLOW_TEST") == "1"
LOG = trt.Logger(trt.Logger.WARNING)
eng = trt.Runtime(LOG).deserialize_cuda_engine(open(a.engine, "rb").read()); ctx = eng.create_execution_context()
names = [eng.get_binding_name(i) for i in range(eng.num_bindings)]
oi = [i for i in range(eng.num_bindings) if not eng.binding_is_input(i)][0]
odt = torch.float16 if eng.get_binding_dtype(oi) == trt.float16 else torch.float32
y = torch.empty(tuple(eng.get_binding_shape(oi)), dtype=odt, device="cuda")


def fwd(x):
    outs = []
    for i in range(x.shape[0]):
        xi = x[i:i + 1].contiguous()
        b = [0] * eng.num_bindings
        for j in range(eng.num_bindings):
            b[j] = int(xi.data_ptr()) if eng.binding_is_input(j) else int(y.data_ptr())
        ctx.execute_v2(b); outs.append(y.float().clone())
    return torch.cat(outs)


t0 = time.time()
dets, st = qlib.predict_coco(fwd, a.imgsz, split=a.split, bs=8)
met = qlib.coco_eval(dets, a.split)
row = {"tag": a.tag, "engine": a.engine, "imgsz": a.imgsz, "split": a.split, "trt": trt.__version__, **met, **st, "wall_s": round(time.time() - t0, 1)}
open(a.out, "a").write(json.dumps(row) + "\n")
print(json.dumps(row))
