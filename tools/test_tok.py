"""Tokenizer conformance: qms tok vs llama-tokenize on a hand-made corpus plus random strings,
and decode(encode(text)) == text byte for byte.

    python -I tools/test_tok.py MODEL.gguf [--random 300]
"""
import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QMS = str(ROOT / "bin" / "qms.exe")
LLAMA_TOK = "E:/llama.cpp/llama-tokenize.exe"

CORPUS = [
    "Hello world", "Hello, world!", "  leading spaces", "trailing spaces   ", "a  b   c    d",
    "line1\nline2\n\nline3\n\n\n", "windows\r\nline\r\n", "\t\ttabbed\tthing", "x  \n  y", " \n \n ",
    "I'm here, you're there, they've gone, he'll come, we'd go. IT'S DON'T Y'ALL",
    "'s 'S 're 'RE 'll", "rock'n'roll", "12345", "3.14159 and 2,718", "year 2026, 10:30pm", "a1b2c3",
    "x^2 + y^2 = z^2", "!!!???...", "(parentheses) [brackets] {braces}", "@user #tag $100 %50 &amp;",
    "https://example.com/path?q=1&b=2#frag", '{"key": [1, 2, 3], "nested": {"a": null}}',
    "def f(x):\n    return x * 2\n\n\nclass A:\n\tpass\n",
    "for (int i = 0; i < n; i++) { sum += a[i]; }",
    "法国的首都是巴黎。", "请用三段话介绍长城的历史，以及它在今天对中国旅游业的意义。",
    "中文English混合text，数字123和标点！", "日本語のテキストとカタカナ", "한국어 텍스트입니다",
    "Ελληνικά κείμενο", "Русский текст, ёжик", "العربية النص", "हिन्दी पाठ", "ภาษาไทย",
    "café naïve résumé", "e\u0301 combining accent", "Straße ß", "ＡＢＣ１２３　全角", "١٢٣ ٣٤٥ arabic digits",
    "😀 👍🏽 👨‍👩‍👧 🇨🇳", "a\u00a0nbsp\u00a0b", "ideographic\u3000space", "zero\u200bwidth",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n", "<|im_start|><|im_end|>",
    "<|im_star not special", "<|endoftext|>", "text<|im_end|>more", "<think>\nreasoning\n</think>",
    "", " ", "\n", "a", "'", "''", "' s", "...\n", "\r", "\u2028sep\u2029para",
]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"failed: {cmd}\n{r.stderr.decode(errors='replace')[-500:]}")
    return r.stdout.decode("utf-8", errors="replace")


def ids_llama(model, path):
    out = run([LLAMA_TOK, "-m", model, "-f", path, "--ids", "--log-disable"]).strip().splitlines()[-1]
    return [int(x) for x in out.strip("[] ").split(",") if x.strip()]


def ids_qms(model, path):
    out = run([QMS, "tok", model, path]).strip()
    return [int(x) for x in out.split(",") if x]


def random_strings(n, seed=7):
    rng = random.Random(seed)
    pools = ["abcxyzABCXYZ", "0123456789", " ", "  ", "\n", "\r\n", "\t", ".,!?;:'\"()-_/",
             "的是不了人我在有他这", "あいうアイウ", "한국", "éßñü", "😀👍", "\u00a0\u3000", "'s", "'re",
             "<|im_end|>", "<|"]
    out = []
    for _ in range(n):
        out.append("".join(rng.choice(pools) for _ in range(rng.randint(1, 40))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--random", type=int, default=300)
    a = ap.parse_args()
    cases = CORPUS + random_strings(a.random)
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        src, back = Path(d) / "in.txt", Path(d) / "out.bin"
        for i, text in enumerate(cases):
            src.write_bytes(text.encode("utf-8"))
            ref = ids_llama(a.model, str(src))
            got = ids_qms(a.model, str(src))
            ok_ids = ref == got
            run([QMS, "detok", a.model, ",".join(map(str, got)) or "", str(back)]) if got else back.write_bytes(b"")
            ok_rt = back.read_bytes() == text.encode("utf-8")
            if not (ok_ids and ok_rt):
                fails += 1
                print(f"FAIL #{i} {text!r}\n   llama {ref}\n   qms   {got}\n   roundtrip ok: {ok_rt}")
    print(f"{len(cases)} strings ({len(CORPUS)} hand-made + {a.random} random): "
          f"{len(cases) - fails} identical ids + exact round trip, {fails} failures")
    print("TOKENIZER", "PASSED" if fails == 0 else "FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
