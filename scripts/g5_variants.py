"""G5 variants for Ultralytics built-in QAT (preregistration §8.1).
C1: percentile-99.99 histogram calibration on 512 letterboxed random train images (replaces official max/mosaic init).
C3: small-object-weighted output distillation from the frozen FP32 teacher (DFL KL + cls BCE, teacher-score weighted,
    P3 anchors x lam), added to the detection loss.
(C2 = mosaic=0 is a plain training argument.)
"""
import copy, json, os
import torch, torch.nn.functional as F
import qlib


def prepare_qat_percentile(model, dataloader, preprocess, batches=8, imgsz=896, n=512, pct=99.99):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    hist = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
    if isinstance(cfg["quant_cfg"], dict):
        cfg["quant_cfg"]["*input_quantizer"] = hist
    else:
        for e in cfg["quant_cfg"]:
            if e.get("quantizer_name") == "*input_quantizer":
                e["cfg"] = hist
    cfg["algorithm"] = None
    training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model = mtq.quantize(model, cfg, forward_loop=None)
        Q = {k: q for k, q in model.named_modules() if isinstance(q, TensorQuantizer) and q.is_enabled}
        for q in Q.values():
            q.disable_quant(); q.enable_calib()
        for x in qlib.CalibLoader(qlib.calib_images(n, 0, "random"), imgsz, device=str(dev)):
            model(x)
        for k, q in Q.items():
            if k.endswith("weight_quantizer"):
                q.load_calib_amax()
            else:
                q.load_calib_amax(method="percentile", percentile=pct)
            q.enable_quant(); q.disable_calib()
        model.to(dev)
        head = f"model.{len(model.model) - 1}."  # same float exceptions as the official prepare_qat
        mtq.disable_quantizer(model, lambda k: k.startswith(head) and (".conv." not in k or ".dfl." in k))
    model.train(training)
    print(f"[C1] percentile-{pct} init on {n} letterboxed train images, {len(Q)} quantizers", flush=True)
    return model


class DistillCriterion:
    """Wraps the detection criterion; adds dis * L_distill * batch_size to the box component."""

    def __init__(self, base, teacher, lam=2.0, dis=1.0, log_path=None):
        self.base, self.teacher, self.lam, self.dis, self.log_path, self.step = base, teacher, lam, dis, log_path, 0
        self.reg_max = getattr(base, "reg_max", 16)

    def __getattr__(self, k):  # delegate attributes (e.g. .hyp, .update) to the base criterion
        return getattr(self.__dict__["base"], k)

    def __call__(self, preds, batch):
        loss, items = self.base(preds, batch)
        p = preds[1] if isinstance(preds, (list, tuple)) else preds
        with torch.no_grad():
            tp = self.teacher(batch["img"])
            tp = tp[1] if isinstance(tp, (list, tuple)) else tp
        sb, ss = p["boxes"].float(), p["scores"].float()      # (B, 4*R, N), (B, nc, N)
        tb, ts = tp["boxes"].float(), tp["scores"].float()
        B, _, N = sb.shape
        R = self.reg_max
        ls = F.log_softmax(sb.view(B, 4, R, N), dim=2)
        lt = F.log_softmax(tb.view(B, 4, R, N), dim=2)
        kl = (lt.exp() * (lt - ls)).sum(2).mean(1)            # (B, N)
        bce = F.binary_cross_entropy_with_logits(ss, ts.sigmoid(), reduction="none").mean(1)  # (B, N)
        w = ts.sigmoid().amax(1)                              # teacher max class score (B, N)
        n3 = p["feats"][0].shape[-2] * p["feats"][0].shape[-1]
        lvl = torch.ones(N, device=w.device); lvl[:n3] = self.lam
        w = w * lvl
        ld = ((kl + bce) * w).sum() / (w.sum() + 1e-9)
        loss = loss.clone()
        loss[0] = loss[0] + self.dis * ld * B
        self.step += 1
        if self.log_path and self.step % 50 == 0:
            open(self.log_path, "a").write(json.dumps({"step": self.step, "distill": float(ld), "kl": float((kl * w).sum() / w.sum()),
                                                        "bce": float((bce * w).sum() / w.sum())}) + "\n")
        return loss, items


def attach_distill(trainer, teacher_pt, lam, dis, log_path):
    from ultralytics import YOLO
    teacher = YOLO(teacher_pt).model.float().to(trainer.device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    m = trainer.model
    m = m.module if hasattr(m, "module") else m
    base = m.criterion if getattr(m, "criterion", None) is not None else m.init_criterion()
    m.criterion = DistillCriterion(base, teacher, lam=lam, dis=dis, log_path=log_path)
    print(f"[C3] distillation attached: lam={lam} dis={dis} teacher={teacher_pt}", flush=True)
