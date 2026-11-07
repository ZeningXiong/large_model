#!/bin/bash
# Round-robin speed test at 8 GB available memory:
#   bench/rr_8gb.sh NAME ROUNDS "label1|args1" "label2|args2" ...
# Every config runs once per round, in the same order, under one memhog session, so slow
# drift (thermals, background activity) hits all configs alike. Each run: 32 tokens.
set -u
cd /d/ClaudeCode/project/large_model
NAME=$1; ROUNDS=$2; shift 2
OUT=bench/ab/$NAME
mkdir -p $OUT
M=E:/models/qwen3-235b/Q4_0/Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf
IDS=$(cat bench/traces/ids_zh.txt)
rm -f $OUT/stop.flag
trap 'touch $OUT/stop.flag' EXIT
bin/memhog.exe 8 $OUT/stop.flag > $OUT/memhog.out 2> $OUT/memhog.err &
for i in $(seq 1 600); do grep -q READY $OUT/memhog.out 2>/dev/null && break; sleep 1; done
echo "memhog: $(cat $OUT/memhog.out)"
for r in $(seq 1 $ROUNDS); do
    for cfg in "$@"; do
        label=${cfg%%|*}; args=${cfg#*|}
        tag=${label}_$r
        QMS_PROF=1 bin/qms.exe run $M --ids $IDS --gen 33 --top 1 --mem-gb 7.5 --direct --ctx 256 $args \
            --trace $OUT/$tag.trace > $OUT/$tag.out 2> $OUT/$tag.err
        dec=$(grep -oE '\([0-9.]+ s/token\)' $OUT/$tag.err | tr -d '()')
        prof=$(grep -oE 'fetch [0-9.]+ moe [0-9.]+' $OUT/$tag.err)
        pf=$(grep -oE '\([0-9.]+% precision\)' $OUT/$tag.err)
        rd=$(grep -oE 'decode\)   : [0-9.]+ GB/token' $OUT/$tag.err | grep -oE '[0-9.]+ GB/token')
        echo "$tag  $dec  | $prof ms/token | experts read $rd $pf  [$args]"
    done
done
touch $OUT/stop.flag
python -I - "$OUT" <<'PYEOF'
import sys, hashlib, glob, re, statistics as st
out = sys.argv[1]
base = hashlib.md5(open("bench/traces/pf_0_0.bin", "rb").read()).hexdigest()
bad = [f for f in sorted(glob.glob(out + "/*.trace")) if hashlib.md5(open(f, "rb").read()).hexdigest() != base]
print(f"traces: {len(glob.glob(out + '/*.trace'))} runs, {'all == baseline' if not bad else 'MISMATCH: ' + ' '.join(bad)}")
runs = {}
for f in sorted(glob.glob(out + "/*.err")):
    m = re.search(r"\(([\d.]+) s/token\)", open(f, encoding="utf-8", errors="replace").read())
    if m:
        label = f.replace("\\", "/").split("/")[-1].rsplit("_", 1)[0]
        runs.setdefault(label, []).append(float(m.group(1)))
for label, v in runs.items():
    print(f"  {label:10s} runs {' '.join(f'{x:.3f}' for x in v)}  mean {st.mean(v):.3f} s/token")
PYEOF
