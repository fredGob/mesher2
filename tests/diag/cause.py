import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1])); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from run_tests import read_inp
from step2sc8r import scaled_jacobians
nodes, elsets = read_inp(sys.argv[1])
ids = {n: i for i, n in enumerate(nodes)}; X = np.array(list(nodes.values()))
H = np.array([[ids[a] for a in h] for hs in elsets.values() for h in hs])
sj = scaled_jacobians(X, H)
def quad_q(P):   # jacobien normalisé 2D min aux 4 coins
    n = np.cross(P[:, 1] - P[:, 0], P[:, 3] - P[:, 0]); n /= np.linalg.norm(n, axis=1)[:, None]
    q = np.full(len(P), np.inf)
    for c in range(4):
        a, b = P[:, (c + 1) % 4] - P[:, c], P[:, (c - 1) % 4] - P[:, c]
        q = np.minimum(q, np.einsum("ij,ij->i", np.cross(a, b), n) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))
    return q
q_bot = quad_q(X[H[:, :4]])
th = np.linalg.norm(X[H[:, 4:]] - X[H[:, :4]], axis=2)       # épaisseur aux 4 coins
L = np.linalg.norm(X[H[:, 1]] - X[H[:, 0]], axis=1)
bad = np.argsort(sj)[:15]
print(" jac3D  quad2D  ép.min  ép.max  arête   centre")
for k in bad:
    print(f" {sj[k]:5.2f}  {q_bot[k]:5.2f}   {th[k].min():6.2f} {th[k].max():6.2f} {L[k]:6.2f}   {np.round(X[H[k]].mean(0)).astype(int)}")
print(f"\n{len(H)} hexas ; jac3D<0.3 : {(sj<0.3).sum()} ; dont quad 2D<0.3 : {((sj<0.3)&(q_bot<0.3)).sum()} ; dont rapport ép. max/min>1.5 : {((sj<0.3)&(th.max(1)/th.min(1)>1.5)).sum()}")
