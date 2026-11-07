"""Compare the C engine (bin/qms.exe) against the numpy reference.

Checks, on one prompt:
  1. prompt logits at EVERY position, full vocabulary: max |logprob_C - logprob_ref|
  2. greedy generation: incremental (KV cache) vs full recompute vs reference greedy
Both sides use fp32 activations on the same dequantized weights, so agreement should be
tight (~1e-4). A router near-tie (k-th vs k+1-th expert probability closer than --router-tie)
is reported, because there the two sides may legitimately pick different experts.

    python -I tools/cmp_ref.py MODEL.gguf --ids 1,2,3 --gen 8
"""
import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_qwen3moe import Model, log_softmax  # noqa: E402

QMS = str(Path(__file__).resolve().parent.parent / "bin" / "qms.exe")


FAST = False   # set by --fast: compare the int8 fast path instead of the exact fp32 path


def run_qms(model, ids, gen, mode, dump=None, ctx=1024):
    cmd = [QMS, "run", model, "--ids", ",".join(map(str, ids)), "--gen", str(gen),
           "--top", "5", "--mode", mode, "--ctx", str(ctx)]
    if not FAST:
        cmd.append("--exact")
    if dump:
        cmd += ["--dump-prompt-logits", dump]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"qms failed ({r.returncode})")
    out_ids = [int(x) for x in re.search(r"^ids: (.*)$", r.stdout, re.M).group(1).split(",")]
    return out_ids, time.time() - t0, r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--router-tie", type=float, default=1e-4)
    ap.add_argument("--skip-recompute", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="test the int8 fast path (looser: activation rounding moves logits)")
    a = ap.parse_args()
    global FAST
    FAST = a.fast
    prompt = [int(s) for s in a.ids.split(",") if s]
    fails = 0

    with tempfile.TemporaryDirectory() as d:
        dump = str(Path(d) / "logits.bin")
        ids_inc, t_inc, log = run_qms(a.model, prompt, a.gen, "inc", dump)
        V = None
        c_logits = np.fromfile(dump, dtype=np.float32)
    print(f"qms inc      : {t_inc:6.1f}s  | {log.splitlines()[0] if log else ''}")
    if not a.skip_recompute:
        ids_rec, t_rec, _ = run_qms(a.model, prompt, a.gen, "recompute")
        print(f"qms recompute: {t_rec:6.1f}s")

    m = Model(a.model, verbose=False)
    V = m.vocab
    c_lp = log_softmax(c_logits.reshape(len(prompt), V))
    gaps = []
    t0 = time.time()
    r_lp = log_softmax(m.forward(prompt, gaps=gaps))
    print(f"reference    : {time.time() - t0:6.1f}s")
    min_gap = float(np.min(np.stack(gaps)))

    d = np.abs(c_lp - r_lp)
    worst = float(d.max())
    per_pos = d.max(axis=1)
    # the same metric cmp_llama.py uses for llama.cpp: |d| over the reference top-10 tokens
    top10 = np.argsort(-r_lp, axis=1)[:, :10]
    d10 = np.take_along_axis(d, top10, axis=1)
    worst10, med10 = float(d10.max()), float(np.median(d10.max(axis=1)))
    gate = worst10 if FAST else worst
    ok1 = gate <= a.tol or min_gap < a.router_tie
    fails += not ok1
    print(f"[1] prompt logprobs, {len(prompt)} positions x {V} vocab: max|d| {worst:.2e} "
          f"(per position: {' '.join(f'{x:.1e}' for x in per_pos)})  min router gap {min_gap:.1e}")
    print(f"    over reference top-10: worst {worst10:.2e}, median of per-position worst {med10:.2e}"
          f"  -> {'PASS' if ok1 else 'FAIL'} (gate: {'top-10' if FAST else 'full vocab'} <= {a.tol})")

    # reference greedy, full recompute per step
    ref_ids = list(prompt)
    for _ in range(a.gen):
        ref_ids.append(int(np.argmax(m.forward(ref_ids)[-1])))
    same_ref = ids_inc == ref_ids
    print(f"[2] greedy inc vs reference : {'IDENTICAL' if same_ref else 'DIFFERENT'}"
          + ("  (informational in --fast mode)" if FAST else ""))
    if not same_ref:
        print("    inc:", ids_inc[len(prompt):], "\n    ref:", ref_ids[len(prompt):])
    fails += (not same_ref) and not FAST
    if not a.skip_recompute:
        same_rec = ids_inc == ids_rec
        print(f"[3] greedy inc vs recompute : {'IDENTICAL' if same_rec else 'DIFFERENT'}")
        fails += not same_rec
    print("VERDICT:", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
