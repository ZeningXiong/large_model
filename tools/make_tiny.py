"""Write a tiny random Qwen3-MoE GGUF (F32), then quantize with llama-quantize.

The tokenizer is copied verbatim from a real Qwen3 GGUF so llama.cpp can load the file;
everything else is small random weights with a fixed seed. Layer sizes are multiples of 32
so every matrix is quantizable to Q4_0.

    python -I tools/make_tiny.py --src D:/models/qwen3-30b/...Q4_0.gguf --out tests/fixtures/tiny
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import gguf

ARCH = "qwen3moe"
CFG = dict(n_layer=4, n_embd=256, n_head=8, n_head_kv=2, head_dim=32,
           n_expert=16, n_used=4, n_ff_exp=64, rope_base=1_000_000.0, eps=1e-6)


def copy_tokenizer(src: gguf.GGUFReader, w: gguf.GGUFWriter) -> int:
    vocab = 0
    for key, fld in src.fields.items():
        if not key.startswith("tokenizer."):
            continue
        if key == "tokenizer.chat_template":
            w.add_string(key, fld.contents())
            continue
        vt = fld.types[0]
        val = fld.contents()
        if vt == gguf.GGUFValueType.ARRAY:
            et = fld.types[-1]
            if key == "tokenizer.ggml.tokens":
                vocab = len(val)
            w.add_key_value(key, val, vt, sub_type=et)
        else:
            w.add_key_value(key, val, vt)
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="real Qwen3 GGUF to copy the tokenizer from")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--quantize", default="E:/llama.cpp/llama-quantize.exe")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    c = CFG
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f32_path = out / "tiny-f32.gguf"
    rng = np.random.default_rng(a.seed)

    w = gguf.GGUFWriter(str(f32_path), ARCH)
    w.add_name("qms-tiny-qwen3moe")
    w.add_block_count(c["n_layer"])
    w.add_context_length(4096)
    w.add_embedding_length(c["n_embd"])
    w.add_feed_forward_length(c["n_ff_exp"] * c["n_used"])
    w.add_head_count(c["n_head"])
    w.add_head_count_kv(c["n_head_kv"])
    w.add_key_length(c["head_dim"])
    w.add_value_length(c["head_dim"])
    w.add_rope_freq_base(c["rope_base"])
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_expert_count(c["n_expert"])
    w.add_expert_used_count(c["n_used"])
    w.add_expert_feed_forward_length(c["n_ff_exp"])
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    vocab = copy_tokenizer(gguf.GGUFReader(a.src), w)

    E, H, Hk, D, X, F = (c["n_embd"], c["n_head"], c["n_head_kv"], c["head_dim"],
                         c["n_expert"], c["n_ff_exp"])

    def rnd(*shape, std):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    def norm_w(n):  # around 1, so a sign or indexing bug in the norm changes the output
        return (1.0 + 0.2 * rng.standard_normal(n)).astype(np.float32)

    w.add_tensor("token_embd.weight", rnd(vocab, E, std=1.0))
    for L in range(c["n_layer"]):
        p = f"blk.{L}."
        w.add_tensor(p + "attn_norm.weight", norm_w(E))
        w.add_tensor(p + "attn_q.weight", rnd(H * D, E, std=E ** -0.5))
        w.add_tensor(p + "attn_k.weight", rnd(Hk * D, E, std=E ** -0.5))
        w.add_tensor(p + "attn_v.weight", rnd(Hk * D, E, std=E ** -0.5))
        w.add_tensor(p + "attn_q_norm.weight", norm_w(D))
        w.add_tensor(p + "attn_k_norm.weight", norm_w(D))
        w.add_tensor(p + "attn_output.weight", rnd(E, H * D, std=(H * D) ** -0.5))
        w.add_tensor(p + "ffn_norm.weight", norm_w(E))
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(X, E, std=2.0 * E ** -0.5))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(X, F, E, std=E ** -0.5))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(X, F, E, std=E ** -0.5))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(X, E, F, std=F ** -0.5))
    w.add_tensor("output_norm.weight", norm_w(E))
    w.add_tensor("output.weight", rnd(vocab, E, std=0.25))   # logits std ~4: peaked, few ties

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {f32_path} ({f32_path.stat().st_size / 1e6:.1f} MB, vocab {vocab})")

    q_path = out / "tiny-q4_0.gguf"
    r = subprocess.run([a.quantize, str(f32_path), str(q_path), "Q4_0"],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("llama-quantize failed")
    types = {}
    for t in gguf.GGUFReader(str(q_path)).tensors:
        key = (t.name.split(".")[-2] if t.name.startswith("blk.") else t.name, t.tensor_type.name)
        types[key] = types.get(key, 0) + 1
    print(f"wrote {q_path} ({q_path.stat().st_size / 1e6:.1f} MB)")
    for (k, ty), n in sorted(types.items()):
        print(f"  {k:20s} {ty:5s} x{n}")


if __name__ == "__main__":
    main()
