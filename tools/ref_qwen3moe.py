"""Reference Qwen3-MoE forward pass in numpy, reading weights straight from GGUF.

Purpose: an independent, readable oracle for the C engine. Weights are dequantized with
gguf-py's own dequantizers (not ours), all math is float32 unless noted, and the whole
sequence is recomputed on every call (no KV cache) so there is nothing stateful to get wrong.

Usage as a library:
    m = Model("path/to/model-00001-of-00003.gguf")
    logits = m.forward([token ids])            # [T, vocab] float32
CLI:
    python -I tools/ref_qwen3moe.py MODEL.gguf --ids 1,2,3 [--layers N] [--top 10]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import gguf


def shard_paths(path: str) -> list[str]:
    """Expand 'x-00001-of-00003.gguf' into all shards; a single file stays as is."""
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(3))
    return [f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


class Model:
    def __init__(self, path: str, verbose: bool = True):
        self.readers = [gguf.GGUFReader(p) for p in shard_paths(path)]
        self.t = {t.name: t for r in self.readers for t in r.tensors}
        f = self.readers[0].fields
        arch = f["general.architecture"].contents()
        if arch != "qwen3moe":
            raise ValueError(f"unsupported architecture {arch!r}")

        def kv(key, default=None):
            fld = f.get(f"{arch}.{key}")
            if fld is None:
                if default is None:
                    raise KeyError(f"missing GGUF key {arch}.{key}")
                return default
            return fld.contents()

        self.n_layer = int(kv("block_count"))
        self.n_embd = int(kv("embedding_length"))
        self.n_head = int(kv("attention.head_count"))
        self.n_head_kv = int(kv("attention.head_count_kv"))
        self.head_dim = int(kv("attention.key_length"))
        self.n_expert = int(kv("expert_count"))
        self.n_used = int(kv("expert_used_count"))
        self.eps = float(kv("attention.layer_norm_rms_epsilon"))
        self.rope_base = float(kv("rope.freq_base"))
        self.vocab = int(self.t["token_embd.weight"].shape[1])
        self._small: dict[str, np.ndarray] = {}
        if verbose:
            print(f"[ref] {len(self.readers)} shard(s), {len(self.t)} tensors: layers={self.n_layer} "
                  f"embd={self.n_embd} heads={self.n_head}/{self.n_head_kv}x{self.head_dim} "
                  f"experts={self.n_used}/{self.n_expert} rope={self.rope_base:g} eps={self.eps:g} "
                  f"vocab={self.vocab}", file=sys.stderr)

    # ---------------------------------------------------------------- weights ----
    def deq(self, name: str, index=None) -> np.ndarray:
        """Dequantize a whole tensor (or one slice of its outermost dim) to float32."""
        t = self.t[name]
        data = t.data if index is None else t.data[index]
        if t.tensor_type in (gguf.GGMLQuantizationType.F32,):
            return np.asarray(data, dtype=np.float32)
        return gguf.quants.dequantize(np.asarray(data), t.tensor_type).astype(np.float32, copy=False)

    def small(self, name: str) -> np.ndarray:
        """Cache norms and routers: tiny, used every token."""
        if name not in self._small:
            self._small[name] = self.deq(name)
        return self._small[name]

    def embed_rows(self, ids) -> np.ndarray:
        t = self.t["token_embd.weight"]
        rows = np.asarray(t.data[np.asarray(ids)])
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            return rows.astype(np.float32)
        return gguf.quants.dequantize(rows, t.tensor_type).astype(np.float32, copy=False)

    # ---------------------------------------------------------------- ops ----
    def rmsnorm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x64 = x.astype(np.float64)
        inv = 1.0 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + self.eps)
        return (x64 * inv).astype(np.float32) * w

    def rope_neox(self, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """x: [T, H, D]. NEOX layout: element i pairs with i + D/2."""
        d = x.shape[-1]
        half = d // 2
        inv_freq = self.rope_base ** (-np.arange(half, dtype=np.float64) * 2.0 / d)
        ang = pos[:, None].astype(np.float64) * inv_freq[None, :]          # [T, half]
        cos = np.cos(ang)[:, None, :].astype(np.float32)
        sin = np.sin(ang)[:, None, :].astype(np.float32)
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def attention(self, L: int, x: np.ndarray, pos: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        H, Hk, D = self.n_head, self.n_head_kv, self.head_dim
        p = f"blk.{L}."
        q = (x @ self.deq(p + "attn_q.weight").T).reshape(T, H, D)
        k = (x @ self.deq(p + "attn_k.weight").T).reshape(T, Hk, D)
        v = (x @ self.deq(p + "attn_v.weight").T).reshape(T, Hk, D)
        q = self.rmsnorm(q, self.small(p + "attn_q_norm.weight"))       # per-head QK-norm
        k = self.rmsnorm(k, self.small(p + "attn_k_norm.weight"))
        q = self.rope_neox(q, pos)
        k = self.rope_neox(k, pos)

        group = H // Hk
        k = np.repeat(k, group, axis=1)                                   # GQA: share kv heads
        v = np.repeat(v, group, axis=1)
        scores = np.einsum("thd,shd->hts", q, k).astype(np.float64) / np.sqrt(D)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, mask] = -np.inf
        scores -= scores.max(axis=-1, keepdims=True)
        prob = np.exp(scores)
        prob /= prob.sum(axis=-1, keepdims=True)
        out = np.einsum("hts,shd->thd", prob.astype(np.float32), v).reshape(T, H * D)
        return out @ self.deq(p + "attn_output.weight").T

    def moe(self, L: int, x: np.ndarray, trace: list | None = None,
            gaps: list | None = None) -> np.ndarray:
        p = f"blk.{L}."
        logits = x @ self.small(p + "ffn_gate_inp.weight").T              # [T, n_expert]
        l64 = logits.astype(np.float64)
        prob = np.exp(l64 - l64.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)                          # softmax over all experts
        if gaps is not None:   # selection margin: k-th minus (k+1)-th probability, per position
            s = -np.sort(-prob, axis=-1)
            gaps.append(s[:, self.n_used - 1] - s[:, self.n_used])
        out = np.zeros_like(x)
        cache: dict[int, tuple] = {}
        for t in range(x.shape[0]):
            top = np.argsort(-prob[t], kind="stable")[: self.n_used]
            w = prob[t, top] / prob[t, top].sum()                         # renormalise the top-k
            if trace is not None:
                trace.append((L, t, top.tolist(), w.tolist()))
            for e, we in zip(top.tolist(), w.tolist()):
                if e not in cache:
                    cache[e] = (self.deq(p + "ffn_gate_exps.weight", e),
                                self.deq(p + "ffn_up_exps.weight", e),
                                self.deq(p + "ffn_down_exps.weight", e))
                g, u, d = cache[e]
                a = x[t] @ g.T
                a = a / (1.0 + np.exp(-a)) * (x[t] @ u.T)                 # SwiGLU: silu(gate) * up
                out[t] += np.float32(we) * (a @ d.T)
        return out

    # ---------------------------------------------------------------- model ----
    def forward(self, ids, layers: int | None = None, hidden_out: list | None = None,
                trace: list | None = None, gaps: list | None = None,
                verbose: bool = False) -> np.ndarray:
        ids = list(ids)
        pos = np.arange(len(ids))
        h = self.embed_rows(ids)
        n = self.n_layer if layers is None else layers
        for L in range(n):
            t0 = time.time()
            p = f"blk.{L}."
            h = h + self.attention(L, self.rmsnorm(h, self.small(p + "attn_norm.weight")), pos)
            h = h + self.moe(L, self.rmsnorm(h, self.small(p + "ffn_norm.weight")), trace, gaps)
            if hidden_out is not None:
                hidden_out.append(h.copy())
            if verbose:
                print(f"[ref] layer {L:3d}  |h|max {np.abs(h).max():9.3f}  {time.time() - t0:6.2f}s",
                      file=sys.stderr)
        hn = self.rmsnorm(h, self.small("output_norm.weight"))
        out_name = "output.weight" if "output.weight" in self.t else "token_embd.weight"
        return hn @ self.deq(out_name).T


def log_softmax(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    m = x64.max(axis=-1, keepdims=True)
    return x64 - m - np.log(np.exp(x64 - m).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to generate (full recompute)")
    a = ap.parse_args()
    m = Model(a.model)
    ids = [int(s) for s in a.ids.split(",") if s]
    for step in range(max(1, a.gen)):
        t0 = time.time()
        lp = log_softmax(m.forward(ids, layers=a.layers, verbose=(step == 0))[-1])
        top = np.argsort(-lp)[: a.top]
        print(f"step {step} ({time.time() - t0:.1f}s) top{a.top}: " +
              " ".join(f"{i}:{lp[i]:.4f}" for i in top))
        if a.gen:
            ids.append(int(top[0]))
    if a.gen:
        print("generated ids:", ",".join(map(str, ids)))


if __name__ == "__main__":
    main()
