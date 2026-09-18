import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from run_tests import read_inp
nodes, elsets = read_inp(sys.argv[1]); A_side, V_cad = float(sys.argv[2]), float(sys.argv[3])
ids = {n: i for i, n in enumerate(nodes)}; X = np.array(list(nodes.values()))
H = np.array([[ids[a] for a in h] for hs in elsets.values() for h in hs])
P = X[H[:, :4]]
area = 0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1)
th = np.linalg.norm(X[H[:, 4:]] - X[H[:, :4]], axis=2).mean(1)
print(f"aire maillée {area.sum():.0f} vs aire côté CAO {A_side:.0f} ({area.sum()/A_side-1:+.2%})")
print(f"épaisseur moyenne pondérée maillage {np.sum(area*th)/area.sum():.3f} vs V_CAO/A_côté {V_cad/A_side:.3f}")
print("histogramme épaisseurs (aire %):", {round(k,1): round(100*v/area.sum(),1) for k, v in sorted(__import__('collections').Counter(dict()).items())})
b = np.histogram(th, bins=[0,2,3,3.3,3.6,4,5,6,7,8,9,10,20], weights=area)
for lo, hi, w in zip(b[1][:-1], b[1][1:], b[0]): print(f"  [{lo:4.1f},{hi:4.1f}[ : {100*w/area.sum():5.1f} %")
