#!/bin/bash
# A/B speed test at 8 GB available memory: bench/ab_8gb.sh NAME "ARGS_A" "ARGS_B"
# Runs A, B, A, B (32 generated tokens each) under memhog, then checks every routing trace
# against the established baseline (output must not change).
set -u
cd /d/ClaudeCode/project/large_model
NAME=$1; A=$2; B=$3
OUT=bench/ab/$NAME
mkdir -p $OUT
M=E:/models/qwen3-235b/Q4_0/Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf
IDS=$(cat bench/traces/ids_zh.txt)
rm -f $OUT/stop.flag
trap 'touch $OUT/stop.flag' EXIT
bin/memhog.exe 8 $OUT/stop.flag > $OUT/memhog.out 2> $OUT/memhog.err &
for i in $(seq 1 600); do grep -q READY $OUT/memhog.out 2>/dev/null && break; sleep 1; done
echo "memhog: $(cat $OUT/memhog.out)"
for r in 1 2; do
    for v in A B; do
        args=$A; [ $v = B ] && args=$B
        tag=${v}_$r
        QMS_PROF=1 bin/qms.exe run $M --ids $IDS --gen 33 --top 1 --mem-gb 7.5 --direct --ctx 256 $args \
            --trace $OUT/$tag.trace > $OUT/$tag.out 2> $OUT/$tag.err
        dec=$(grep -oE 'decode: .*s/token\)' $OUT/$tag.err)
        prof=$(grep -oE 'fetch [0-9.]+ moe [0-9.]+' $OUT/$tag.err)
        echo "$tag [$args]  $dec  | $prof ms/token"
    done
done
touch $OUT/stop.flag
python -I - "$OUT" <<'PYEOF'
import sys, hashlib, glob
out = sys.argv[1]
base = hashlib.md5(open("bench/traces/pf_0_0.bin", "rb").read()).hexdigest()
for f in sorted(glob.glob(out + "/*.trace")):
    h = hashlib.md5(open(f, "rb").read()).hexdigest()
    print(f"  trace {f.split('/')[-1]:12s} {'== baseline' if h == base else '!= BASELINE ' + h[:12]}")
PYEOF
