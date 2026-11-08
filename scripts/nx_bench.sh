#!/bin/bash
# Direction C NX benchmark (prereg §8.3, aligned with PLOS bench_fp16_formal.sh timing protocol).
# Usage: bash nx_bench.sh <root with <name>/model.onnx> <name> [<name> ...]
# name suffix decides the build: *_fp32 -> --fp16 ; *_entropy2 -> build_trt.py implicit INT8 ; others (Q/DQ) -> --int8 --fp16
set -u
source $HOME/edgets_env.sh
ROOT=$1; shift
S=$HOME/edgets/qatc/scripts
export QDATA=$HOME/edgets/data/TT100K45 QGT_VAL=$HOME/edgets/data/TT100K45/gt_val.json QGT_TEST=$HOME/edgets/data/TT100K45/gt_test.json
trap 'sudo -n tegrastats --stop >/dev/null 2>&1' EXIT
sudo -n nvpmodel -m 8 >/dev/null 2>&1; sudo -n jetson_clocks
if [ "$(systemctl is-active gdm3)" = "active" ]; then echo "ABORT gdm3 active"; exit 1; fi
for N in "$@"; do
  W=$ROOT/$N; cd $W || continue
  IMG=$(echo $N | sed -E 's/^y11n_([0-9]+)_.*/\1/')
  { date '+%F %T'; sudo -n nvpmodel -q | tail -1; systemctl is-active gdm3; top -bn2 -d 3 | grep -i 'cpu(s)' | tail -1; } > env_at_start.txt 2>&1
  t0=$(date +%s)
  if [ ! -f engine.plan ]; then
    case $N in
      *_fp32) trtexec --onnx=model.onnx --saveEngine=engine.plan --fp16 --memPoolSize=workspace:2048 --profilingVerbosity=detailed > build.log 2>&1 ;;
      *_entropy2) python3 $S/build_trt_any.py --onnx model.onnx --out engine --precision int8 --calib-algo entropy2 --calib-list $HOME/edgets/qatc/calib512_${IMG}.txt --cache calib_entropy2.cache > build.log 2>&1; [ -f engine.engine ] && mv engine.engine engine.plan ;;
      *) trtexec --onnx=model.onnx --saveEngine=engine.plan --int8 --fp16 --memPoolSize=workspace:2048 --profilingVerbosity=detailed > build.log 2>&1 ;;
    esac
  fi
  echo "BUILD $N rc=$? build_s=$(( $(date +%s)-t0 )) bytes=$(stat -c %s engine.plan 2>/dev/null)"
  [ -f engine.plan ] || continue
  trtexec --loadEngine=engine.plan --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=layers.json --iterations=10 > layerinfo.log 2>&1
  for k in 1 2 3; do
    sleep 60
    sudo -n tegrastats --interval 100 --logfile tegrastats_run$k.log --start
    trtexec --loadEngine=engine.plan --warmUp=3000 --iterations=1000 --avgRuns=1000 --useSpinWait > run$k.log 2>&1; echo "RUN $N #$k rc=$?"
    sudo -n tegrastats --stop; sleep 1
  done
  python3 $S/nx_eval.py --engine engine.plan --imgsz $IMG --tag $N --out $ROOT/nx_val.jsonl > eval.log 2>&1; echo "EVAL $N rc=$?"
done
echo "BENCH_DONE $(date '+%F %T')"
