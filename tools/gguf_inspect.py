"""Minimal GGUF header inspector (no dependencies). Usage: python -I gguf_inspect.py <file.gguf>"""
import struct, sys, collections

GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K",
        11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def rstr(f):
    return f.read(rd(f, "Q")).decode("utf-8", "replace")


def rval(f, t):
    if t in SCALAR:
        return rd(f, SCALAR[t])
    if t == 8:
        return rstr(f)
    if t == 9:
        et, n = rd(f, "I"), rd(f, "Q")
        if et in SCALAR:
            f.seek(struct.calcsize("<" + SCALAR[et]) * n, 1)
        else:
            for _ in range(n):
                rval(f, et)
        return f"<array n={n}>"
    raise ValueError(t)


with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    ver, nt, nkv = rd(f, "I"), rd(f, "Q"), rd(f, "Q")
    kv = {}
    for _ in range(nkv):
        k = rstr(f)
        kv[k] = rval(f, rd(f, "I"))
    tens = []
    for _ in range(nt):
        name = rstr(f)
        nd = rd(f, "I")
        dims = [rd(f, "Q") for _ in range(nd)]
        tens.append((name, dims, rd(f, "I"), rd(f, "Q")))
    align = kv.get("general.alignment", 32)
    data_start = (f.tell() + align - 1) // align * align

print(f"version {ver}, tensors {nt}, kv {nkv}, alignment {align}, data_start {data_start} (mod 4096 = {data_start % 4096})")
for k, v in kv.items():
    if not str(v).startswith("<array") or "tokenizer" not in k:
        if any(s in k for s in ("qwen3moe.", "general.name", "split", "file_type", "tokenizer.ggml.model", "chat_template")) and "chat_template" not in k:
            print(f"  {k} = {v}")
types = collections.Counter()
for name, dims, t, off in tens:
    types[(name.split(".")[-2] if name.startswith("blk.") else name, GGML.get(t, t))] += 1
print("tensor kinds:")
for (k, t), n in sorted(types.items()):
    print(f"  {k:20s} {t:6s} x{n}")
print("sample expert tensors:")
for name, dims, t, off in tens:
    if name.startswith("blk.0.") or (name.startswith("blk.1.ffn") and "exps" in name):
        print(f"  {name:28s} dims={dims} type={GGML.get(t, t)} abs_off={data_start + off} mod4096={(data_start + off) % 4096}")
