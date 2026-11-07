"""Compare the numpy reference against llama.cpp (llama-server) on the same GGUF.

For a prompt of token ids, llama-server greedily generates N tokens and reports the top-k
log-probabilities at every step. The reference recomputes the same sequence and we compare:
  * greedy token ids, step by step
  * |logprob_ref - logprob_llama| over llama's top-k tokens at every step
Differences are expected to be small but non-zero: llama.cpp quantizes activations to 8 bits
for its Q4 matmuls, the reference does not.

    python -I tools/cmp_llama.py MODEL.gguf --ids 9707,11,1879 --gen 8
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_qwen3moe import Model, log_softmax  # noqa: E402


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def llama_generate(server, model, ids, n_gen, n_probs, port, threads, timeout):
    cmd = [server, "-m", model, "--port", str(port), "--host", "127.0.0.1", "-t", str(threads),
           "-c", "2048", "-np", "1", "--no-repack", "--no-webui"]
    log = open(Path(model).with_suffix(".server.log"), "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{port}"
        t0 = time.time()
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited; see " + log.name)
            if time.time() - t0 > timeout:
                raise TimeoutError("llama-server did not become ready")
            time.sleep(1)
        res = post(base + "/completion", {
            "prompt": ids, "n_predict": n_gen, "temperature": 0.0, "top_k": 1,
            "n_probs": n_probs, "cache_prompt": False, "return_tokens": True,
            "post_sampling_probs": False, "ignore_eos": True,
        }, timeout)
    finally:
        proc.terminate()
        proc.wait(30)
        log.close()
    steps = []
    for st in res["completion_probabilities"]:
        tops = st.get("top_logprobs") or st.get("probs")
        steps.append({
            "id": st["id"],
            "top": [(t["id"], t["logprob"] if "logprob" in t else float(np.log(t["prob"])))
                    for t in tops],
        })
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--server", default="E:/llama.cpp/llama-server.exe")
    ap.add_argument("--port", type=int, default=18931)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tol", type=float, default=0.05, help="max |dlogprob| over top-k to PASS")
    ap.add_argument("--router-tie", type=float, default=1e-3,
                    help="router k vs k+1 prob margin below which an expert flip is expected")
    ap.add_argument("--logit-tie", type=float, default=0.1,
                    help="top1-top2 logprob gap below which a greedy flip is expected")
    a = ap.parse_args()

    prompt = [int(s) for s in a.ids.split(",") if s]
    t0 = time.time()
    steps = llama_generate(a.server, a.model, prompt, a.gen, a.top, a.port, a.threads, a.timeout)
    print(f"llama.cpp: {len(steps)} steps in {time.time() - t0:.1f}s")
    gen = [s["id"] for s in steps]

    m = Model(a.model)
    t0 = time.time()
    gaps: list = []
    logits = m.forward(prompt + gen[:-1], gaps=gaps)   # one pass covers every generation step
    print(f"reference: forward over {len(prompt) + len(gen) - 1} tokens in {time.time() - t0:.1f}s")
    lp = log_softmax(logits[len(prompt) - 1:])
    # smallest router selection margin over all layers at positions <= p: a near-tie anywhere
    # upstream can legitimately flip an expert and move every later logit.
    gap_pos = np.min(np.stack(gaps), axis=0)                    # [T]
    gap_upto = np.minimum.accumulate(gap_pos)

    worst = 0.0
    agree = 0
    bad = 0
    diffs = []
    for i, st in enumerate(steps):
        ref_top = int(np.argmax(lp[i]))
        d = max(abs(lp[i][tid] - l) for tid, l in st["top"])
        diffs.append(d)
        worst = max(worst, d)
        ok = ref_top == st["id"]
        agree += ok
        srt = np.sort(lp[i])
        gap = srt[-1] - srt[-2]
        rgap = float(gap_upto[len(prompt) - 1 + i])
        tie = rgap < a.router_tie
        good = (d <= a.tol or tie) and (ok or gap < a.logit_tie or tie)
        bad += not good
        note = "" if good else "   <-- UNEXPLAINED"
        if good and (d > a.tol or not ok):
            note = f"   (router near-tie {rgap:.1e})" if tie else "   (logit near-tie)"
        print(f"  step {i:2d}: llama {st['id']:>6}  ref {ref_top:>6}  {'==' if ok else '!='}  "
              f"max|dlogp| {d:.4f}  top1-top2 {gap:.3f}  min router gap {rgap:.1e}{note}")
    print(f"greedy agreement {agree}/{len(steps)}, |dlogprob| median {np.median(diffs):.4f} "
          f"worst {worst:.4f}, unexplained steps {bad}")
    verdict = bad == 0
    print("VERDICT:", "PASS" if verdict else "FAIL")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
