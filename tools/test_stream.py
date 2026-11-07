"""Streamed experts must give bit-identical logits to resident experts.

    python -I tools/test_stream.py MODEL.gguf --ids 1,2,3 --gen 8 [--exact]
"""
import argparse
import filecmp
import re
import subprocess
import sys
import tempfile
from pathlib import Path

QMS = str(Path(__file__).resolve().parent.parent / "bin" / "qms.exe")


def run(model, ids, gen, extra, dump):
    cmd = [QMS, "run", model, "--ids", ids, "--gen", str(gen), "--top", "5",
           "--dump-prompt-logits", dump] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"qms failed ({r.returncode}): {' '.join(extra)}")
    tops = re.findall(r"^step .*$", r.stdout, re.M)
    gen_ids = re.search(r"^ids: (.*)$", r.stdout, re.M).group(1)
    return tops, gen_ids, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--stream-args", default="", help="extra flags for the streamed run")
    a = ap.parse_args()
    extra = ["--exact"] if a.exact else []
    with tempfile.TemporaryDirectory() as d:
        d1, d2 = str(Path(d) / "res.bin"), str(Path(d) / "str.bin")
        t1, g1, _ = run(a.model, a.ids, a.gen, extra, d1)
        t2, g2, log = run(a.model, a.ids, a.gen, extra + ["--stream-experts"] + a.stream_args.split(), d2)
        same_logits = filecmp.cmp(d1, d2, shallow=False)
    same_steps = t1 == t2
    same_ids = g1 == g2
    for line in log.splitlines():
        if "experts streamed" in line or "PEAK" in line or "expert cache" in line:
            print("   ", line.strip())
    ok = same_logits and same_steps and same_ids
    print(f"  prompt logits bit-identical: {same_logits}; per-step top-5 identical: {same_steps}; "
          f"generated ids identical: {same_ids}  -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
