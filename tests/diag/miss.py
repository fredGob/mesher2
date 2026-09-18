import sys, gmsh, numpy as np, math
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import step2sc8r as m
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(sys.argv[1]); gmsh.model.occ.synchronize()
vol = gmsh.model.getEntities(3)[0][1]
ana = m.analyse(gmsh, [vol], 1.0)
A = ana["area"]; info = ana["info"]; s0, s1 = map(set, ana["sides"])
tot = sum(A.values())
print(f"t={ana['t']:.3f} bornes [{ana['t_lo']:.2f},{ana['t_hi']:.2f}]  aire totale {tot:.0f}")
print(f"côté0 {sum(A[f] for f in s0):.0f} ({len(s0)} faces)  côté1 {sum(A[f] for f in s1):.0f} ({len(s1)} faces)  choisi {m.choose_side(ana)}")
other = [f for f in A if f not in s0 | s1]
cat = Counter()
for f in other:
    d, c, p = info[f]
    k = "inf" if not math.isfinite(d) else ("cos<0.8" if c <= 0.8 else ("trop mince" if d < ana["t_lo"] else "trop épais"))
    cat[k] += A[f]
print("hors peau par cause (aire):", {k: round(v) for k, v in cat.items()})
big = sorted(other, key=lambda f: -A[f])[:12]
for f in big:
    d, c, p = info[f]
    print(f"  face {f} {gmsh.model.getType(2,f)[:8]} aire={A[f]:.0f} d={d:.2f} cos={c:.2f} vis-à-vis={p} {'(côté0)' if p in s0 else '(côté1)' if p in s1 else ''}")
