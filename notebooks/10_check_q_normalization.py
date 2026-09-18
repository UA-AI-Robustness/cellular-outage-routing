# notebooks/10_check_q_normalization.py
import osmnx as ox, numpy as np, math
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","q_dwell","time","length","s_mean"): d[a]=float(d[a])

# --- 1. the NORMALIZED q we currently use ---
q = np.array([d["q"] for _,_,d in G.edges(data=True)])
print("=== normalized q (current) ===")
print(f"  min={q.min():.4f}  p25={np.percentile(q,25):.4f}  median={np.median(q):.4f}"
      f"  p75={np.percentile(q,75):.4f}  p95={np.percentile(q,95):.4f}  max={q.max():.4f}")
print(f"  fraction q==0: {(q==0).mean()*100:.1f}%")

# --- 2. reconstruct the RAW rate (before /q_max) to see the true spread ---
# r_raw was Shannon on s_mean; q_raw = r_raw * covered_frac / max(1,N(T)).
# We didn't store q_raw, but we can inspect the RATE DISTRIBUTION directly from s_mean.
def shannon(s_mean_db, B):
    if not math.isfinite(s_mean_db): return 0.0
    return B*math.log2(1+10**(s_mean_db/10))
B = cfg.signal.bandwidth_hz
r = np.array([shannon(d["s_mean"], B) for _,_,d in G.edges(data=True)])
print("\n=== raw Shannon rate R_raw (bits/s), before load & normalization ===")
print(f"  min={r.min():.3e}  median={np.median(r):.3e}  p95={np.percentile(r,95):.3e}  max={r.max():.3e}")
print(f"  max / median ratio: {r.max()/max(np.median(r),1):.1f}x   <-- if huge, max-normalization squashes everything")

# --- 3. how skewed is it? a few big edges dominating? ---
print(f"\n  top 1% of edges have rate >= {np.percentile(r,99):.3e}")
print(f"  a max-normalization divides ALL edges by {r.max():.3e}")
print(f"  so a median edge becomes {np.median(r)/r.max():.4f} of max  <-- this is why median q is tiny")

# --- 4. what would a PERCENTILE normalization give? (divide by p95, clip to 1) ---
p95 = np.percentile(r, 95)
q_p95 = np.clip(r / p95, 0, 1)
print("\n=== if we normalized by p95 instead of max ===")
print(f"  median={np.median(q_p95):.3f}  p25={np.percentile(q_p95,25):.3f}  p75={np.percentile(q_p95,75):.3f}")
print(f"  fraction at 0: {(q_p95==0).mean()*100:.1f}%")