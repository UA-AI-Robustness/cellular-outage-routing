# notebooks/22_check_frac.py
import osmnx as ox, numpy as np
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
cfg=load_config(); G=ox.load_graphml(cache_path(cfg)); _coerce_types(G)
d1=[]; d2=[]
for _,_,d in G.edges(data=True):
    d1.append(float(d["dead_fraction"])); d2.append(float(d["frac_below_cover"]))
d1=np.array(d1); d2=np.array(d2)
print(f"max |dead_fraction - frac_below_cover| = {np.abs(d1-d2).max():.6f}  (should be ~0)")
for name in ("frac_below_cover","frac_below_call","frac_below_upload"):
    v=np.array([float(d[name]) for _,_,d in G.edges(data=True)])
    print(f"{name}: mean below-frac={v.mean():.3f}  edges fully-below={(v>=0.999).mean()*100:.1f}%")