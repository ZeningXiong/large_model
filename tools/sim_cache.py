"""Replay expert-request traces against cache policies (offline, no model needed).

Trace records: {int32 step, int16 layer, int16 expert}; step 1 is the prompt (prefill),
steps >= 2 are decode tokens. Requests of one (step, layer) arrive together, exactly like
the engine's fetch_group: they cannot evict each other.

Policies, at several cache sizes:
  lru        what the engine does today
  lfu        online: evict the least-used-so-far (counts from this run only)
  pin+lru    a fixed hot set (by frequency in OTHER traces: leave-one-out) never evicted,
             the remaining slots LRU. pin fraction of capacity is a parameter.
  belady     optimal offline (evict the expert whose next use is farthest): an upper bound

    python -I tools/sim_cache.py bench/traces/trace_en.bin bench/traces/trace_zh.bin ...
"""
import argparse
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

REC = np.dtype([("step", "<i4"), ("layer", "<i2"), ("expert", "<i2")])
SLOT_MB = 11.03          # one 235B expert slot (gate+up+down, unbuffered windows)
N_EXPERT = 128


def load(path):
    r = np.fromfile(path, dtype=REC)
    keys = r["layer"].astype(np.int64) * N_EXPERT + r["expert"]
    groups = []           # list of (step, [keys]) in request order
    cur, cur_id = [], None
    for st, ly, k in zip(r["step"], r["layer"], keys):
        gid = (int(st), int(ly))
        if gid != cur_id and cur:
            groups.append((cur_id[0], cur))
            cur = []
        cur_id = gid
        cur.append(int(k))
    if cur:
        groups.append((cur_id[0], cur))
    return groups


def simulate(groups, cap, policy, pinned=frozenset()):
    """Returns (decode_hits, decode_requests). Prefill requests also flow through the cache."""
    pinned = set(list(pinned)[:cap])
    lru = OrderedDict()                 # key -> None, most recent last
    count = Counter()
    hits = reqs = 0
    free = cap - len(pinned)
    decay_half = float(policy.split(":")[1]) if policy.startswith("lfu-decay") else 0.0
    dscore = {}

    def score_now(x, step):
        s0, t0 = dscore.get(x, (0.0, step))
        return s0 * 0.5 ** ((step - t0) / decay_half)
    # Belady needs the next-use position of every request
    if policy == "belady":
        flat = [k for _, g in groups for k in g]
        nxt = [0] * len(flat)
        last = {}
        for i in range(len(flat) - 1, -1, -1):
            nxt[i] = last.get(flat[i], 1 << 60)
            last[flat[i]] = i
        next_use = {}
        pos = 0
    for step, g in groups:
        gset = set(g)
        for k in g:
            count[k] += 1
            if decay_half:   # decayed frequency: score halves every decay_half tokens
                s0, t0 = dscore.get(k, (0.0, step))
                dscore[k] = (s0 * 0.5 ** ((step - t0) / decay_half) + 1.0, step)
            hit = k in pinned or k in lru
            if step >= 2:
                reqs += 1
                hits += hit
            if policy == "belady":
                next_use[k] = nxt[pos]
                pos += 1
            if k in pinned:
                continue
            if hit:
                lru.move_to_end(k)
                continue
            if free <= 0:
                continue          # zero LRU slots: nothing to insert into
            if len(lru) >= free:
                cands = [x for x in lru if x not in gset]
                if not cands:
                    continue      # group larger than cache: stream through
                if policy == "lru" or policy == "pin+lru":
                    victim = cands[0]
                elif policy == "lfu":
                    victim = min(cands, key=lambda x: (count[x], 0))
                elif policy.startswith("lfu-decay"):
                    victim = min(cands, key=lambda x: score_now(x, step))
                else:             # belady, with bypass: never cache what is needed later than all
                    victim = max(cands, key=lambda x: next_use.get(x, 1 << 60))
                    if next_use[k] >= next_use.get(victim, 1 << 60):
                        continue
                del lru[victim]
            lru[k] = None
    return hits, reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--gb", default="1,2,4,8,16")
    ap.add_argument("--pin-frac", default="0.5,0.9")
    a = ap.parse_args()
    traces = {Path(p).stem: load(p) for p in a.traces}
    sizes = [float(x) for x in a.gb.split(",")]
    fracs = [float(x) for x in a.pin_frac.split(",")]

    for name, groups in traces.items():
        dec = [g for s, g in groups if s >= 2]
        nreq = sum(len(g) for g in dec)
        distinct = len({k for g in dec for k in g})
        print(f"\n== {name}: {len(dec)} decode layer-groups, {nreq} requests, {distinct} distinct experts "
              f"({distinct * SLOT_MB / 1e3:.1f} GB to hold them all)")
        # frequency from the OTHER traces only
        freq = Counter()
        for other, og in traces.items():
            if other != name:
                for s, g in og:
                    if s >= 2:
                        freq.update(g)
        hot = [k for k, _ in freq.most_common()]
        header = f"{'cache':>7} {'slots':>6} {'lru':>7} {'lfu':>7}"
        header += "".join(f" {'pin' + format(f, '.0%'):>8}" for f in fracs) + f" {'belady':>7}"
        print(header)
        for gb in sizes:
            cap = int(gb * 1000 / SLOT_MB)
            row = [simulate(groups, cap, "lru"), simulate(groups, cap, "lfu")]
            for f in fracs:
                row.append(simulate(groups, cap, "pin+lru", frozenset(hot[: int(cap * f)])))
            row.append(simulate(groups, cap, "belady"))
            print(f"{gb:6.0f}G {cap:6d} " + " ".join(f"{100 * h / max(r, 1):7.1f}" for h, r in row))
    print("\nhit rates in % of decode expert requests; pin sets learned on the other traces only")


if __name__ == "__main__":
    main()
