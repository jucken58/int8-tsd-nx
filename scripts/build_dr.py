"""S1 (post-hoc): TensorRT 8.5 implicit INT8 with externally supplied per-tensor dynamic ranges (no calibrator),
plus optional FP16 constraints on a few layers. Usage:
python3 build_dr.py --onnx model.onnx --ranges ranges.json --pct 99.99 --fp16 "<regex>" --out engine.plan
"""
import argparse, json, re, time
import tensorrt as trt

ap = argparse.ArgumentParser(); ap.add_argument("--onnx", required=True); ap.add_argument("--ranges", required=True)
ap.add_argument("--pct", default="99.99"); ap.add_argument("--fp16", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--list-layers", action="store_true")
ap.add_argument("--fp16-inputs", action="store_true", help="also make the producers of the constrained layers' inputs emit FP16 (matches fake-quant exceptions, where the conv input is not quantized)")
a = ap.parse_args()
LOG = trt.Logger(trt.Logger.WARNING)
b = trt.Builder(LOG); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); p = trt.OnnxParser(net, LOG)
assert p.parse_from_file(a.onnx), [p.get_error(i) for i in range(p.num_errors)]
if a.list_layers:
    for i in range(net.num_layers):
        L = net.get_layer(i); print(i, L.type, L.name)
    raise SystemExit
R = json.load(open(a.ranges))["ranges"]
cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048 << 20)
cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_flag(trt.BuilderFlag.INT8)
inp = net.get_input(0); inp.set_dynamic_range(-1.0, 1.0)
n_set, missing = 0, []
for i in range(net.num_layers):
    L = net.get_layer(i)
    for k in range(L.num_outputs):
        t = L.get_output(k)
        if t.dtype not in (trt.float32, trt.float16):
            continue
        if t.name in R:
            r = max(float(R[t.name][a.pct]), 1e-6); t.set_dynamic_range(-r, r); n_set += 1
        else:
            missing.append((L.name, t.name))
rx = re.compile(a.fp16) if a.fp16 else None
fp16_layers = []
if rx:
    cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    for i in range(net.num_layers):
        L = net.get_layer(i)
        if rx.search(L.name):
            L.precision = trt.float16
            for k in range(L.num_outputs): L.set_output_type(k, trt.float16)
            fp16_layers.append(L.name)
    if a.fp16_inputs:
        prod = {}
        for i in range(net.num_layers):
            L = net.get_layer(i)
            for k in range(L.num_outputs): prod[L.get_output(k).name] = (L, k)
        for name in list(fp16_layers):
            L = [net.get_layer(i) for i in range(net.num_layers) if net.get_layer(i).name == name][0]
            for j in range(L.num_inputs):
                t = L.get_input(j)
                if t is not None and t.name in prod and t.dtype in (trt.float32, trt.float16):
                    P, k = prod[t.name]; P.set_output_type(k, trt.float16); fp16_layers.append("in<-" + P.name)
t0 = time.time(); ser = b.build_serialized_network(net, cfg); assert ser is not None, "build failed"
open(a.out, "wb").write(ser)
info = {"onnx": a.onnx, "pct": a.pct, "n_ranges_set": n_set, "n_missing": len(missing), "missing_sample": missing[:15],
        "fp16_regex": a.fp16, "fp16_layers": fp16_layers, "build_s": round(time.time() - t0, 1), "trt": trt.__version__}
json.dump(info, open(a.out + ".build.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "missing_sample"}))
