import sys, math, numpy as np
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1])); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from run_tests import read_inp
from step2sc8r import scaled_jacobians, edge_counts
nodes, elsets = read_inp(sys.argv[1])
ids = {n: i for i, n in enumerate(nodes)}; X = np.array(list(nodes.values()))
H = np.array([[ids[a] for a in h] for hs in elsets.values() for h in hs])
sj = scaled_jacobians(X, H); Q = H[:, :4]; ec = edge_counts(Q)
bnd = {i for e, c in ec.items() if c == 1 for i in e}
val = Counter(Q.ravel().tolist())
for k in np.argsort(sj)[:8]:
    r = Q[k].tolist(); P = X[r]
    nb = sum(ec[tuple(sorted((r[a], r[(a+1)%4])))] == 1 for a in range(4))
    ang = [round(math.degrees(math.acos(np.clip(np.dot(P[(a+1)%4]-P[a], P[(a-1)%4]-P[a]) / np.linalg.norm(P[(a+1)%4]-P[a]) / np.linalg.norm(P[(a-1)%4]-P[a]), -1, 1)))) for a in range(4)]
    L = [round(float(np.linalg.norm(P[(a+1)%4]-P[a])), 1) for a in range(4)]
    print(f"jac {sj[k]:.2f} bords={nb} angles={ang} côtés={L} valences={[val[i] for i in r]} surbord={[i in bnd for i in r]} @ {np.round(P.mean(0)).astype(int)}")
