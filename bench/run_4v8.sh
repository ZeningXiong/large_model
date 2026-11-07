#!/bin/bash
# 4 GB vs 8 GB available memory, Qwen3-235B-A22B Q4_0.
# Each run: memhog locks everything above the target, the engine runs, memhog is released.
# qms gets 95% of the target (its own startup rule); llama.cpp gets the whole target.
set -u
cd /d/ClaudeCode/project/large_model
OUT=bench/4v8
mkdir -p $OUT
M=E:/models/qwen3-235b/Q4_0/Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf
IDS=$(cat bench/traces/ids_zh.txt)
cleanup() { touch $OUT/stop.flag; taskkill //IM typeperf.exe //F >/dev/null 2>&1; }
trap cleanup EXIT

typeperf "\PhysicalDisk(_Total)\Disk Read Bytes/sec" "\Memory\Available MBytes" "\Processor(_Total)\% Processor Time" \
    -si 2 -o $OUT/perf.csv -y >/dev/null 2>&1 &

hold() {   # hold <target_gb>: start memhog and wait until memory is in place
    rm -f $OUT/stop.flag $OUT/memhog.out
    bin/memhog.exe $1 $OUT/stop.flag > $OUT/memhog.out 2>> $OUT/memhog.err &
    for i in $(seq 1 600); do grep -q READY $OUT/memhog.out 2>/dev/null && break; sleep 1; done
    echo "   memhog: $(cat $OUT/memhog.out)"
}
release() { touch $OUT/stop.flag; sleep 8; while tasklist 2>/dev/null | grep -qi memhog; do sleep 2; done; }

run_qms() {   # run_qms <tag> <target_gb> <mem_gb>
    echo "== $1  ($(date +%T))"
    hold $2
    bin/peakrun.exe $OUT/$1.peak bin/qms.exe run $M --ids $IDS --gen 17 --top 1 --mem-gb $3 --direct --ctx 256 \
        --trace $OUT/$1.trace > $OUT/$1.out 2> $OUT/$1.err
    grep -E 'attention pinned|TOTAL|prompt:|decode:|trunk streamed|experts streamed \(decode|PEAK|REFUSING|FATAL' $OUT/$1.err
    cat $OUT/$1.peak
    release
}
run_llama() {   # run_llama <tag> <target_gb>
    echo "== $1  ($(date +%T))"
    hold $2
    bin/peakrun.exe $OUT/$1.peak E:/llama.cpp/llama-completion.exe -m $M -f bench/traces/p_zh.txt \
        -n 16 --temp 0 --ignore-eos -no-cnv --no-repack -t 12 -c 512 > $OUT/$1.out 2> $OUT/$1.err
    grep -E 'prompt eval time|eval time' $OUT/$1.err | sed 's/^.*common_perf_print: *//'
    cat $OUT/$1.peak
    release
}

for r in 1 2; do
    run_qms qms8_$r 8 7.5
    run_qms qms4_$r 4 3.75
    [ "${QMS_ONLY:-0}" = 1 ] || run_llama llama4_$r 4
done
echo "done $(date +%T)"
