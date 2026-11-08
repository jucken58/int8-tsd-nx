#!/bin/bash
# S1 benchmark: same timing/eval protocol as nx_bench.sh; engines built by build_dr.py.
# Usage: bash nx_bench_dr.sh <root> <name> [...]   name = y11n_<S>_s0_dr{0,2,7}
set -u
source $HOME/edgets_env.sh
ROOT=$1; shift
S=$HOME/edgets/qatc/scripts
export QDATA=$HOME/edgets/data/TT100K45 QGT_VAL=$HOME/edgets/data/TT100K45/gt_val.json QGT_TEST=$HOME/edgets/data/TT100K45/gt_test.json
trap 'sudo -n tegrastats --stop >/dev/null 2>&1' EXIT
sudo -n nvpmodel -m 8 >/dev/null 2>&1; sudo -n jetson_clocks
if [ "$(systemctl is-active gdm3)" = "active" ]; then echo "ABORT gdm3 active"; exit 1; fi
DR2='^/model\.23/dfl/conv/Conv$|^/model\.23/cv2\.0/cv2\.0\.2/Conv$'
DR7='^/model\.23/dfl/conv/Conv$|^/model\.23/cv[23]\.[012]/cv[23]\.[012]\.2/Conv$'
for N in "$@"; do
  W=$ROOT/$N; mkdir -p $W; cd $W
  IMG=$(echo $N | sed -E 's/^y11n_([0-9]+)_.*/\1/'); V=${N##*_}
  SEED=$(echo $N | sed -E 's/^y11n_[0-9]+_s([0-9]+)_.*/\1/')
  cp -n $ROOT/y11n_${IMG}_s${SEED}_fp32/model.onnx model.onnx
  RJ=$HOME/edgets/qatc/ranges_${IMG}_s${SEED}.json; [ -f $RJ ] || RJ=$HOME/edgets/qatc/ranges_${IMG}.json  # s0 ranges were saved without the seed tag
  EXTRA=""
  case $V in dr0) RX="";; dr2) RX=$DR2;; dr7) RX=$DR7;; dr2i) RX=$DR2; EXTRA="--fp16-inputs";; dr7i) RX=$DR7; EXTRA="--fp16-inputs";; esac
  { date '+%F %T'; sudo -n nvpmodel -q | tail -1; systemctl is-active gdm3; top -bn2 -d 3 | grep -i 'cpu(s)' | tail -1; } > env_at_start.txt 2>&1
  t0=$(date +%s)
  [ -f engine.plan ] || python3 $S/build_dr.py --onnx model.onnx --ranges $RJ --pct 99.99 --fp16 "$RX" $EXTRA --out engine.plan > build.log 2>&1
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
