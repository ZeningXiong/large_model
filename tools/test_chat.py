"""Multi-turn chat check (tiny model, greedy):
  1. the whole conversation decodes to exactly the ChatML text of the messages and replies
  2. every reply, generated incrementally with the KV cache carried across turns, equals what
     a from-scratch recompute (qms run --mode recompute) generates from the same prefix

    python -I tools/test_chat.py MODEL.gguf
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QMS = str(ROOT / "bin" / "qms.exe")
MESSAGES = ["Hello! Who are you?", "用一句话介绍你自己。", "And 2+2?"]
SYSTEM = "You are a test assistant."
MAX_NEW = 10


def main():
    model = sys.argv[1]
    with tempfile.TemporaryDirectory() as d:
        dump = Path(d) / "ids.txt"
        r = subprocess.run([QMS, "chat", model, "--greedy", "--max-new", str(MAX_NEW), "--ctx", "512",
                            "--system", SYSTEM, "--dump-ids", str(dump)],
                           input=("\n".join(MESSAGES) + "\n\n").encode("utf-8"), capture_output=True)
        if r.returncode != 0:
            print(r.stderr.decode(errors="replace")[-1500:])
            raise SystemExit("chat failed")
        ids = [int(x) for x in dump.read_text().strip().split(",")]
        out = Path(d) / "conv.bin"
        subprocess.run([QMS, "detok", model, ",".join(map(str, ids)), str(out)], check=True)
        conv = out.read_bytes().decode("utf-8", errors="replace")

    # 1. template
    parts = re.split(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", conv, flags=re.S)
    replies = parts[1::2]
    expected = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for i, (msg, rep) in enumerate(zip(MESSAGES, replies)):
        expected += ("\n" if i else "") + f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n{rep}<|im_end|>"
    ok_template = conv == expected and len(replies) == len(MESSAGES)
    print(f"[1] {len(ids)} context ids, {len(replies)} replies; decodes to exact ChatML: {ok_template}")
    if not ok_template:
        print("    got:     ", repr(conv[:400]))
        print("    expected:", repr(expected[:400]))

    # 2. each reply vs from-scratch recompute of its prefix
    end_id = ids[-1]
    starts, i = [], 0
    marker = None
    # find reply spans: tokens after "<|im_start|>assistant\n" up to the next <|im_end|>
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "m.txt"
        f.write_bytes(b"<|im_start|>assistant\n")
        marker = [int(x) for x in subprocess.run([QMS, "tok", model, str(f)], capture_output=True,
                                                 check=True).stdout.decode().strip().split(",")]
    for j in range(len(ids) - len(marker)):
        if ids[j:j + len(marker)] == marker:
            s = j + len(marker)
            e = ids.index(end_id, s)
            starts.append((s, e))
    ok_all = True
    for k, (s, e) in enumerate(starts):
        r = subprocess.run([QMS, "run", model, "--ids", ",".join(map(str, ids[:s])), "--gen", str(e - s + 1),
                            "--top", "1", "--mode", "recompute"], capture_output=True, check=True)
        gen = [int(x) for x in re.search(r"^ids: (.*)$", r.stdout.decode(), re.M).group(1).split(",")][s:]
        # a reply cut at --max-new got its <|im_end|> from the chat loop, not from the model
        natural_end = e - s < MAX_NEW
        want = ids[s:e] + ([end_id] if natural_end else [])
        same = gen[: len(want)] == want
        ok_all &= same
        print(f"[2] reply {k + 1}: {e - s} tokens ({'ended by model' if natural_end else 'cut at --max-new'}), "
              f"equals from-scratch recompute: {same}")
        if not same:
            print("    chat     :", want, "\n    recompute:", gen[: len(want)])
    verdict = ok_template and ok_all and len(starts) == len(MESSAGES)
    print("CHAT", "PASSED" if verdict else "FAILED")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
