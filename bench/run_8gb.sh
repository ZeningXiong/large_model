#!/bin/bash
# 8 GB comparison: qms vs llama.cpp on Qwen3-235B-A22B Q4_0, both with ~8 GB of RAM available.
# memhog holds everything above 8 GB; runs alternate qms / llama.cpp, two each.
set -u
cd /d/ClaudeCode/project/large_model
OUT=bench/8gb
mkdir -p $OUT
M=E:/models/qwen3-235b/Q4_0/Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf
STOP=$OUT/stop.flag
rm -f $STOP
cleanup() { touch $STOP; taskkill //IM typeperf.exe //F >/dev/null 2>&1; }
trap cleanup EXIT

bin/memhog.exe 8 $STOP > $OUT/memhog.out 2> $OUT/memhog.err &
for i in $(seq 1 600); do grep -q READY $OUT/memhog.out 2>/dev/null && break; sleep 1; done
echo "memhog: $(cat $OUT/memhog.out)"

typeperf "\PhysicalDisk(_Total)\Disk Read Bytes/sec" "\Memory\Available MBytes" "\Processor(_Total)\% Processor Time" \
    -si 2 -o $OUT/perf.csv -y >/dev/null 2>&1 &

IDS=$(cat bench/traces/ids_zh.txt)
for r in 1 2; do
    echo "== qms run $r  ($(date +%T))"
    bin/peakrun.exe $OUT/qms_$r.peak bin/qms.exe run $M --ids $IDS --gen 17 --top 1 \
        --stream-experts --direct --cache-gb 2 --ctx 256 > $OUT/qms_$r.out 2> $OUT/qms_$r.err
    grep -E 'TOTAL|prompt:|decode:|expert cache:|experts streamed \(decode|PEAK' $OUT/qms_$r.err
    cat $OUT/qms_$r.peak
    echo "== llama.cpp run $r  ($(date +%T))"
    bin/peakrun.exe $OUT/llama_$r.peak E:/llama.cpp/llama-completion.exe -m $M -f bench/traces/p_zh.txt \
        -n 16 --temp 0 --ignore-eos -no-cnv --no-repack -t 12 -c 512 > $OUT/llama_$r.out 2> $OUT/llama_$r.err
    grep -E 'load time|prompt eval time|eval time' $OUT/llama_$r.err | sed 's/^.*common_perf_print: *//'
    cat $OUT/llama_$r.peak
done
echo "done $(date +%T)"
