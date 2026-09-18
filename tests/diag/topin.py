import sys, time, gmsh, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from run_tests import read_inp
nodes, elsets = read_inp(sys.argv[2])
X = np.array(list(nodes.values())); nn = len(X) // 2
B, T = X[:nn], X[nn:]; D = T - B; d = np.linalg.norm(D, axis=1); U = D / d[:, None]
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(sys.argv[1]); gmsh.model.occ.synchronize()
vol = gmsh.model.getEntities(3)[0][1]
rng = np.random.default_rng(0); idx = rng.choice(nn, int(sys.argv[3]), replace=False)
t0 = time.time(); res = []
e = 0.08
for i in idx:
    a = gmsh.model.isInside(3, vol, list(B[i] + e * U[i]))            # juste sous la peau de départ : dedans ?
    b = gmsh.model.isInside(3, vol, list(T[i] - e * U[i]))            # juste sous le noeud du haut : dedans ?
    c = gmsh.model.isInside(3, vol, list(T[i] + e * U[i]))            # juste au-dessus : dehors ?
    res.append((a, b, c))
res = np.array(res, bool)
print(f"{len(idx)} noeuds en {time.time()-t0:.0f}s ; base dedans {res[:,0].mean():.0%} ; haut-e dedans {res[:,1].mean():.0%} ; haut+e dehors {(~res[:,2]).mean():.0%}")
bad = idx[res[:, 2]]
print("noeuds dont le haut dépasse (dans la matière au-delà) :", len(bad), "épaisseurs", np.round(d[bad][:10], 2))
bad2 = idx[~res[:, 1]]
print("noeuds dont le haut est hors matière :", len(bad2), "épaisseurs", np.round(d[bad2][:10], 2), "centres", np.round(B[bad2][:3]).astype(int).tolist())
bad3 = idx[~res[:, 0]]
print("base hors matière :", len(bad3))
for i in bad3[:8]:
    mids = [gmsh.model.isInside(3, vol, list(B[i] + s * U[i])) for s in np.linspace(0.05, d[i] - 0.05, 8)]
    print(f"  noeud {i} d={d[i]:.2f} @ {np.round(B[i]).astype(int)}  le long de l'hexa (dedans ?) : {''.join('X' if x else '.' for x in mids)}")
