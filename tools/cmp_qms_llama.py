"""Compare qms against llama.cpp (llama-server) on the same GGUF and prompt.

Both engines generate greedily; at every step we compare the greedy id and the log-probs
of the tokens that appear in BOTH engines' top-10 lists. Steps after the first divergence
are reported but not judged (the two engines then see different contexts).

    python -I tools/cmp_qms_llama.py MODEL.gguf --ids 785,6722,315,9625,374 --gen 8 \
        [--qms-args "--stream-experts --ctx 512"] [--save out.json]
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_llama import llama_generate  # noqa: E402

QMS = str(Path(__file__).resolve().parent.parent / "bin" / "qms.exe")


def qms_generate(model, ids, gen, extra):
    cmd = [QMS, "run", model, "--ids", ",".join(map(str, ids)), "--gen", str(gen), "--top", "10"]
    cmd += shlex.split(extra)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"qms failed ({r.returncode})")
    steps = []
    for line in re.findall(r"^step \d+ top\d+: (.*)$", r.stdout, re.M):
        pairs = [p.split(":") for p in line.split()]
        steps.append([(int(i), float(l)) for i, l in pairs])
    return steps, time.time() - t0, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--qms-args", default="")
    ap.add_argument("--tol", type=float, default=0.6)
    ap.add_argument("--logit-tie", type=float, default=0.1)
    ap.add_argument("--port", type=int, default=18932)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    prompt = [int(s) for s in a.ids.split(",") if s]

    q_steps, q_time, q_log = qms_generate(a.model, prompt, a.gen, a.qms_args)
    print(f"qms      : {len(q_steps)} steps in {q_time:.1f}s")
    for line in q_log.splitlines():
        if any(k in line for k in ("TOTAL", "prompt:", "decode:", "experts streamed", "PEAK", "loaded")):
            print("   ", line.strip())
    t0 = time.time()
    l_steps = llama_generate("E:/llama.cpp/llama-server.exe", a.model, prompt, a.gen, 10, a.port, 12, 7200)
    print(f"llama.cpp: {len(l_steps)} steps in {time.time() - t0:.1f}s")

    diverged = False
    bad = 0
    worst = 0.0
    for i, (qs, ls) in enumerate(zip(q_steps, l_steps)):
        q_top, l_top = qs[0][0], ls["id"]
        qd, ld = dict(qs), dict(ls["top"])
        common = sorted(set(qd) & set(ld))
        d = max((abs(qd[t] - ld[t]) for t in common), default=float("nan"))
        gap = qs[0][1] - qs[1][1] if len(qs) > 1 else 99.0
        same = q_top == l_top
        note = ""
        if diverged:
            note = "(after divergence: not judged)"
        else:
            worst = max(worst, d)
            ok = d <= a.tol and (same or gap < a.logit_tie)
            if not ok:
                bad += 1
                note = "<-- UNEXPLAINED"
            elif not same:
                note = "(logit near-tie)"
            if not same:
                diverged = True
        print(f"  step {i}: qms {q_top:>6}  llama {l_top:>6}  {'==' if same else '!='}  "
              f"top-10 overlap {len(common):2d}  max|dlogp| {d:.3f}  qms top1-top2 {gap:.3f} {note}")
    print(f"worst |dlogprob| before divergence {worst:.3f}, unexplained {bad}")
    print("VERDICT:", "PASS" if bad == 0 else "FAIL")
    if a.save:
        Path(a.save).write_text(json.dumps({"qms": q_steps, "llama": l_steps}, indent=1))
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
