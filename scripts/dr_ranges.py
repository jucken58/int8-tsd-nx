"""S1 (post-hoc): per-tensor |x| percentile dynamic ranges for TensorRT implicit INT8 (set_dynamic_range).
Runs the FP32 ONNX in onnxruntime (CPU) on the 512 calibration train images (same list as G1/G2), two passes:
abs-max, then a 4096-bin histogram of |x| on [0, absmax]; range = percentile(|x|, pct).
Usage: python dr_ranges.py --onnx model.onnx --imgsz 896 --imgdir <train images> --list calib512.txt --out ranges.json
"""
import argparse, json, os, time
import numpy as np, onnx, onnxruntime as ort, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--onnx", required=True); ap.add_argument("--imgsz", type=int, required=True)
ap.add_argument("--imgdir", required=True); ap.add_argument("--list", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pcts", default="99.99,99.999,100"); ap.add_argument("--bins", type=int, default=4096)
a = ap.parse_args()
m = onnx.load(a.onnx)
m = onnx.shape_inference.infer_shapes(m)
ftypes = {vi.name: vi.type.tensor_type.elem_type for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
names = []
for node in m.graph.node:
    for o in node.output:
        if o and ftypes.get(o) == onnx.TensorProto.FLOAT and o not in [x.name for x in m.graph.output]:
            names.append(o)
for nm in names:
    m.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
so = ort.SessionOptions(); so.intra_op_num_threads = 8
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
outs = [o.name for o in sess.get_outputs()]
inp = sess.get_inputs()[0].name
files = [os.path.join(a.imgdir, os.path.basename(l.strip())) for l in open(a.list) if l.strip()]


def load(p):
    im = cv2.imread(p); h, w = im.shape[:2]; S = a.imgsz; r = min(S / h, S / w); nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w): im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((S, S, 3), 114, np.uint8); t, l = (S - nh) // 2, (S - nw) // 2; out[t:t + nh, l:l + nw] = im
    return np.ascontiguousarray(out[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


t0 = time.time()
amax = {o: 0.0 for o in outs}
for p in files:
    for o, v in zip(outs, sess.run(outs, {inp: load(p)})):
        amax[o] = max(amax[o], float(np.abs(v).max()))
hist = {o: np.zeros(a.bins, np.int64) for o in outs}
for p in files:
    for o, v in zip(outs, sess.run(outs, {inp: load(p)})):
        if amax[o] > 0:
            hist[o] += np.histogram(np.abs(v), bins=a.bins, range=(0, amax[o]))[0]
res = {"onnx": os.path.basename(a.onnx), "imgsz": a.imgsz, "n_images": len(files), "input": inp, "ranges": {}}
for o in outs:
    c = np.cumsum(hist[o]); tot = c[-1] if c[-1] else 1
    rr = {}
    for q in map(float, a.pcts.split(",")):
        if q >= 100: rr[str(q)] = amax[o]; continue
        k = int(np.searchsorted(c, q / 100 * tot)); rr[str(q)] = (k + 1) / a.bins * amax[o]
    res["ranges"][o] = rr
json.dump(res, open(a.out, "w"))
print("tensors", len(outs), "images", len(files), "s", round(time.time() - t0, 1))
