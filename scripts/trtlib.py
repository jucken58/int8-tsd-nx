"""TensorRT 10 helpers: build an engine from ONNX (explicit Q/DQ or plain FP16) and run it on torch CUDA tensors."""
import json, time
import tensorrt as trt, torch

LOG = trt.Logger(trt.Logger.WARNING)


def build(onnx_path, eng_path, int8=False, fp16=True):
    b = trt.Builder(LOG); net = b.create_network(0); p = trt.OnnxParser(net, LOG)
    assert p.parse(open(onnx_path, "rb").read()), [p.get_error(i) for i in range(p.num_errors)]
    cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    t0 = time.time(); ser = b.build_serialized_network(net, cfg); assert ser is not None
    open(eng_path, "wb").write(ser)
    return round(time.time() - t0, 1)


def layer_precisions(eng_path):
    eng = trt.Runtime(LOG).deserialize_cuda_engine(open(eng_path, "rb").read())
    insp = eng.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    out = []
    for L in info["Layers"]:
        if isinstance(L, dict):
            out.append({"name": L.get("Name"), "type": L.get("LayerType"),
                        "in": [i.get("Format/Datatype") for i in L.get("Inputs", [])],
                        "out": [o.get("Format/Datatype") for o in L.get("Outputs", [])]})
    return out


class Runner:
    def __init__(self, path):
        self.eng = trt.Runtime(LOG).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.eng.create_execution_context()
        names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inp = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.out = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        dt = torch.float16 if self.eng.get_tensor_dtype(self.out) == trt.float16 else torch.float32
        self.y = torch.empty(tuple(self.eng.get_tensor_shape(self.out)), dtype=dt, device="cuda")
        self.s = torch.cuda.Stream()

    def __call__(self, x):
        outs = []
        for i in range(x.shape[0]):
            xi = x[i:i + 1].contiguous()
            self.ctx.set_tensor_address(self.inp, int(xi.data_ptr()))
            self.ctx.set_tensor_address(self.out, int(self.y.data_ptr()))
            torch.cuda.current_stream().synchronize()
            self.ctx.execute_async_v3(self.s.cuda_stream); self.s.synchronize()
            outs.append(self.y.float().clone())
        return torch.cat(outs)
