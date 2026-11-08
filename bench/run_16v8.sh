#!/bin/bash
# 16 GB vs 8 GB available memory, qms only (Qwen3-235B-A22B Q4_0).
# Each run: memhog locks everything above the target, qms runs with 95% of it, memhog is released.
set -u
cd /d/ClaudeCode/project/large_model
OUT=bench/16v8
mkdir -p $OUT
M=E:/models/qwen3-235b/Q4_0/Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf
IDS=$(cat bench/traces/ids_zh.txt)
trap 'touch $OUT/stop.flag' EXIT

run_qms() {   # run_qms <tag> <target_gb> <mem_gb>
    echo "== $1  ($(date +%T))"
    rm -f $OUT/stop.flag $OUT/memhog.out
    bin/memhog.exe $2 $OUT/stop.flag > $OUT/memhog.out 2>> $OUT/memhog.err &
    for i in $(seq 1 600); do grep -q READY $OUT/memhog.out 2>/dev/null && break; sleep 1; done
    echo "   memhog: $(cat $OUT/memhog.out)"
    QMS_PROF=1 bin/peakrun.exe $OUT/$1.peak bin/qms.exe run $M --ids $IDS --gen 33 --top 1 --mem-gb $3 --direct \
        --ctx 256 --trace $OUT/$1.trace > $OUT/$1.out 2> $OUT/$1.err
    grep -E 'attention pinned|expert cache  |TOTAL|prompt:|decode:|expert cache:|experts streamed \(decode|profile|PEAK|REFUSING|FATAL' $OUT/$1.err
    touch $OUT/stop.flag
    sleep 8
    while tasklist 2>/dev/null | grep -qi memhog; do sleep 2; done
}

for r in 1 2; do
    run_qms qms8_$r 8 7.5
    run_qms qms16_$r 16 15
done
python -I - <<'PYEOF'
import glob, hashlib, re, statistics as st
base = hashlib.md5(open("bench/traces/pf_0_0.bin", "rb").read()).hexdigest()
for f in sorted(glob.glob("bench/16v8/*.trace")):
    ok = hashlib.md5(open(f, "rb").read()).hexdigest() == base
    print(f"trace {f.replace(chr(92), '/').split('/')[-1]:14s} {'== baseline' if ok else '!= BASELINE'}")
runs = {}
for f in sorted(glob.glob("bench/16v8/*.err")):
    m = re.search(r"\(([\d.]+) s/token\)", open(f, encoding="utf-8", errors="replace").read())
    if m:
        runs.setdefault(f.replace(chr(92), "/").split("/")[-1].rsplit("_", 1)[0], []).append(float(m.group(1)))
for k, v in runs.items():
    print(f"{k:6s} runs {' '.join(f'{x:.3f}' for x in v)}  mean {st.mean(v):.3f} s/token")
PYEOF
echo "done $(date +%T)"
