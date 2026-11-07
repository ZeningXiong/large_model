"""Bit-exact check of qms dequantization against gguf-py on real tensors of every type.

    python -I tools/test_deq.py MODEL.gguf [MODEL2.gguf ...]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_qwen3moe import shard_paths  # noqa: E402

QMS = str(Path(__file__).resolve().parent.parent / "bin" / "qms.exe")


def check(model: str) -> int:
    tensors = {t.name: t for p in shard_paths(model) for t in gguf.GGUFReader(p).tensors}
    picked = {}
    for name, t in tensors.items():   # first tensor of each (type, ndim) plus the output head
        key = (t.tensor_type.name, len(t.shape))
        if key not in picked or name == "output.weight":
            picked[key] = name
    fails = 0
    for (tname, nd), name in sorted(picked.items()):
        t = tensors[name]
        rows = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1
        r0 = max(0, rows // 2 - 2)
        nr = min(4, rows - r0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rows.bin"
            subprocess.run([QMS, "deq", model, name, str(r0), str(nr), str(out)],
                           check=True, capture_output=True)
            got = np.fromfile(out, dtype=np.float32).reshape(nr, -1)
        flat = t.data.reshape(rows, -1) if len(t.shape) > 1 else t.data.reshape(1, -1)
        rows_raw = np.asarray(flat[r0:r0 + nr])
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            ref = rows_raw.astype(np.float32)
        else:
            ref = gguf.quants.dequantize(rows_raw, t.tensor_type).astype(np.float32)
        same = np.array_equal(got.view(np.uint32), ref.view(np.uint32))
        fails += not same
        print(f"  {'PASS' if same else 'FAIL'}  {tname:5s} {name:32s} rows {r0}..{r0 + nr - 1}"
              + ("" if same else f"  max|d| {np.abs(got - ref).max():.3e}"))
    return fails


if __name__ == "__main__":
    total = 0
    for mpath in sys.argv[1:]:
        print(Path(mpath).name)
        total += check(mpath)
    print("DEQUANT", "PASSED (bit-exact)" if total == 0 else f"FAILED ({total})")
    sys.exit(1 if total else 0)
