# INT8 quantization of small traffic sign detectors on the Jetson Xavier NX

Code and result tables for the manuscript

> **INT8 Quantization of Small Traffic Sign Detectors: Localization Error and End-to-End Performance on the Jetson Xavier NX**
> Keyuan Tang, Chuanli Liu, Kui Cheng, Sanxiu Jiao, Chaoyang Zhang (submitted to the *Journal of Real-Time Image Processing*).

The repository holds the minimum needed to check the reported numbers and to rerun the experiments. It does **not** contain the dataset, model weights or TensorRT engines.

## Contents

```
scripts/              quantization, QAT, evaluation, TensorRT and Jetson benchmarking scripts
results/
  diagnosis/          range rules, weight-/activation-only INT8, group sensitivity (Fig. 1, Table S7)
  ptq/                leave-one-out ranking and floating-point sets, seed 0 and per-seed PTQ (Table 1)
  qat/                QAT sweep, five-seed validation and one-time test results, QAT variants (Table 2, S2, S3)
  controls/           output exchange, fixed-anchor IoU, QAT step 0, E7 with max ranges, quantizer ablation (Fig. 2, Tables 2, S4, S9)
  generality/         cross-detector summary (Table S8, Fig. S1)
  nx/                 Jetson Xavier NX engine-level and end-to-end results (Tables 3, 4, S5, S6, Fig. 4)
  nx_ranges/          per-tensor 99.99th-percentile dynamic ranges used for implicit INT8, and the 512 calibration image IDs
docs/analysis_plan_2026-10-23_zh.md   dated analysis plan (in Chinese), written before the experiments
```

## Data and models

- **Dataset.** TT100K (Zhu et al., CVPR 2016) is available from its authors at https://cg.cs.tsinghua.edu.cn/traffic-sign/. We use the 45 classes with more than 100 instances (TT100K45); the split lists are published with our earlier study at https://github.com/jucken58/EdgeTS-YOLO.
- **FP32 weights.** YOLO11n trained at 896 and 1024 px (seeds 0–4) come from that earlier study; they are available from the corresponding author on request.

## Environment

| Purpose | Software |
|---|---|
| Fake quantization (PTQ, controls) | Python 3.12, PyTorch 2.8, Ultralytics 8.4.166, NVIDIA ModelOpt 0.37 (PTQ) / 0.47 (QAT), pycocotools |
| Tensor ranges | onnxruntime (CPU) |
| Jetson | Jetson Xavier NX, L4T R35.6.5, TensorRT 8.5.2.2, power mode 8 with `jetson_clocks` |

Scripts read the dataset and COCO ground truth from environment variables:

```
export QDATA=/path/to/TT100K45          # YOLO-format images/ and labels/
export QGT_VAL=/path/to/gt_val.json     # COCO ground truth built from the YOLO labels
export QGT_TEST=/path/to/gt_test.json
```

Default working paths in the scripts (`./work/...`) can be edited at the top of each file.

## Main scripts

| Step | Script |
|---|---|
| Shared quantization and evaluation library | `qlib.py` |
| Loss diagnosis (range rules, group sensitivity) | `g1_diag.py` |
| PTQ with layers kept in floating point (LOO ranking, top-k, E7) | `g2_ptq.py` |
| Ultralytics built-in QAT and QAT variants | `qat_train.py`, `g5_variants.py` |
| Evaluation of checkpoints, one-time test evaluation | `eval_ckpt.py`, `test_eval.py` |
| Output exchange and fixed-anchor IoU | `r1_swap.py` |
| QAT step-0 control (validation / test) | `r1_qat0.py`, `r2_qat0_test.py` |
| E7 with max ranges, DFL/P3 quantizer ablation | `r2_ptq_ablation.py` |
| ONNX export, per-tensor ranges | `export_onnx.py`, `dr_ranges.py` |
| TensorRT implicit INT8 with set ranges and FP16 constraints (DR0/DR2/DR7, dr2i/dr7i) | `build_dr.py`, `trtlib.py` |
| NX accuracy and `trtexec` benchmarking | `nx_eval.py`, `nx_bench.sh`, `nx_bench_dr.sh` |
| NX end-to-end pipeline timing (CPU or GPU pre-processing, per-frame records) | `r2_pipeline.py` |

## Notes

- Evaluation uses confidence 0.001, NMS IoU 0.7, multi-label NMS and at most 300 detections; the end-to-end pipeline uses deployment settings (confidence 0.25, single label). Ultralytics' NMS time limit is disabled (`max_time_img=1e4`) because it silently drops detections on a shared GPU.
- Each test configuration was evaluated once. All selection was done on the validation split.

## License

Code: GNU AGPL-3.0 (the scripts build on Ultralytics, which is AGPL-3.0). Result tables: CC BY 4.0.
