# INT8 quantization of small traffic sign detectors on the Jetson Xavier NX

Code and result tables for the manuscript

> **INT8 Quantization of Small Traffic Sign Detectors: Localization Error and End-to-End Performance on the Jetson Xavier NX**
>
> Keyuan Tang, Chuanli Liu, Kui Cheng, Sanxiu Jiao, Chaoyang Zhang

This README follows the manuscript prepared for the *Journal of Signal Processing Systems* (JSPS). Figure and table references below refer to that version and its Online Resource 1.

The study examines INT8 quantization for embedded vision deployment: localization error, layer-level precision choices, post-training quantization (PTQ) and quantization-aware training (QAT), and accuracy-latency-energy trade-offs on the Jetson Xavier NX. Engine-level measurements and end-to-end pipeline measurements are reported separately.

The repository provides experimental scripts, result tables, tensor ranges and the dated analysis plan. It does **not** contain the dataset, model weights or TensorRT engines; reproducing the experiments requires those resources and the software environment below.

## Contents

```text
scripts/              quantization, QAT, evaluation, TensorRT and Jetson benchmarking scripts
results/
  diagnosis/          range rules, weight-/activation-only INT8, group sensitivity (Fig. 1, Table S7)
  ptq/                leave-one-out ranking and floating-point sets, seed 0 and per-seed PTQ (Table 1)
  qat/                QAT sweep, five-seed validation and test results, QAT variants (Table 3, Fig. 3, Tables S2-S3)
  controls/           output exchange, fixed-anchor IoU, QAT step 0, E7 with max ranges, quantizer ablation (Tables 2-3, Fig. 2, Table S4)
  generality/         cross-detector summary (Table S8, Fig. S1)
  nx/                 Jetson Xavier NX engine-level and end-to-end results (Tables 4-5, Fig. 4, Tables S5-S6)
  nx_ranges/          per-tensor 99.99th-percentile dynamic ranges used for implicit INT8, and the 512 calibration image IDs
docs/analysis_plan_2026-10-23_zh.md   original analysis plan and dated amendments (in Chinese)
```

## Data and models

- **Dataset.** TT100K (Zhu et al., CVPR 2016) is available from its authors at https://cg.cs.tsinghua.edu.cn/traffic-sign/. We use the 45 classes with more than 100 instances (TT100K45); the split lists are available at https://github.com/jucken58/EdgeTS-YOLO.
- **FP32 weights.** YOLO11n trained at 896 and 1024 px (seeds 0-4) comes from a related study using the same split. The weights are available from the corresponding author on request. FP32 accuracy and FP16 timing baselines were re-measured under the present study's protocol; the quantization experiments and associated results belong to the present study.

## Environment

| Purpose | Software |
|---|---|
| Fake quantization (PTQ, controls) | Python 3.12, PyTorch 2.8, Ultralytics 8.4.166, NVIDIA ModelOpt 0.37 (PTQ) / 0.47 (QAT), pycocotools |
| Tensor ranges | onnxruntime (CPU) |
| Jetson | Jetson Xavier NX, L4T R35.6.5, TensorRT 8.5.2.2, power mode 8 with `jetson_clocks` |

Scripts read the dataset and COCO ground truth from environment variables:

```sh
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
| NX end-to-end pipeline timing (CPU or GPU pre-processing) | `r2_pipeline.py` |

## Interpretation and measurement scope

- Accuracy evaluation uses confidence 0.001, NMS IoU 0.7, multi-label NMS and at most 300 detections; the end-to-end pipeline uses deployment settings (confidence 0.25, single label). Ultralytics' NMS time limit is disabled (`max_time_img=1e4`) because it silently drops detections on a shared GPU.
- Each test configuration was evaluated once. Configuration selection was performed on the validation split. The exploratory implicit INT8 variants are identified as such in the manuscript.
- The original analysis plan predates the main experiments of this study and records motivating evidence from the related project. Later amendments are dated within the same document; post-hoc controls and exploratory deployment variants are distinguished in the manuscript.
- End-to-end timing covers a serial, single-stream pipeline using pre-decoded frames in memory. JPEG decoding, camera arrivals and queueing are excluded. Mean throughput above 30 frames/s does not imply that every frame meets a 33.3 ms processing-time budget.
- End-to-end result tables summarize runs, including frame-time percentiles and the share of frames above 33.3 ms; these summaries should not be treated as raw per-frame traces.
- Board energy per frame includes idle consumption. Comparisons at a fixed input resolution and comparisons that increase input resolution answer different deployment questions.

## License

Code: GNU AGPL-3.0 (the scripts build on Ultralytics, which is AGPL-3.0). Result tables: CC BY 4.0.
