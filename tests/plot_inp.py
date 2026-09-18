"""Contrôle visuel : python tests/plot_inp.py fichier.inp [sortie.png]"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from run_tests import read_inp

inp = sys.argv[1]
png = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".inp", ".png")
nodes, elsets = read_inp(inp)
ids = {n: i for i, n in enumerate(nodes)}; X = np.array(list(nodes.values()))
fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(projection="3d")
colors = plt.cm.Pastel1.colors
for k, (name, hexas) in enumerate(elsets.items()):
    polys = [X[[ids[h[j]] for j in f]] for h in hexas
             for f in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]]
    ax.add_collection3d(Poly3DCollection(polys, fc=colors[k % 9], ec="k", lw=.25))
lo, hi = X.min(0), X.max(0); c, r = (lo + hi) / 2, (hi - lo).max() / 2
ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
ax.set_box_aspect((1, 1, 1)); ax.view_init(*(float(a) for a in sys.argv[3:5])) if len(sys.argv) > 4 else ax.view_init(35, -60)
plt.savefig(png, dpi=85, bbox_inches="tight"); print(png)
