"""Contrôle visuel d'un INP réel : vue en projection sur les axes principaux
de la pièce, faces des hexas colorées par jacobien normalisé (vert = 1,
rouge = 0), croix bleue sur le pire élément, avec un zoom dessus.

    python tests/plot_jac.py fichier.inp [prefixe_sortie] [demi_largeur_zoom]

Sorties : <prefixe>_xy.png (vue principale), <prefixe>_xz.png (vue de côté),
<prefixe>_worst.png (zoom sur le pire élément)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from run_tests import read_inp
from step2sc8r import scaled_jacobians

inp = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".inp", "")
zoom = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0

nodes, elsets = read_inp(inp)
ids = {n: i for i, n in enumerate(nodes)}
X = np.array(list(nodes.values()))
H = np.array([[ids[a] for a in h] for hs in elsets.values() for h in hs])
jac = scaled_jacobians(X, H)
print(f"{len(H)} hexas, jacobien min {jac.min():.3f}, {(jac < 0.3).sum()} sous 0.3")

C = X[H].mean(1); mu = C.mean(0)
_, _, Vt = np.linalg.svd(C - mu, full_matrices=False)          # axes principaux
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
polys = np.concatenate([(X[H[:, list(f)]] - mu) @ Vt.T for f in FACES])
col = plt.cm.RdYlGn(np.clip(np.tile(jac, 6), 0, 1))
worst = (C[jac.argmin()] - mu) @ Vt.T


def draw(fname, axes, xlim=None, ylim=None):
    P = polys[:, :, axes]
    o = np.argsort(polys[:, :, 3 - sum(axes)].mean(1))            # tri par profondeur
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.add_collection(PolyCollection(P[o], facecolors=col[o], edgecolors="k", linewidths=0.15))
    ax.set_aspect("equal"); ax.autoscale()
    if xlim:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.plot(worst[axes[0]], worst[axes[1]], "b+", ms=20, mew=2)
    plt.savefig(fname, dpi=90, bbox_inches="tight"); plt.close(); print(fname)


draw(out + "_xy.png", [0, 1])
draw(out + "_xz.png", [0, 2])
draw(out + "_worst.png", [0, 1], (worst[0] - zoom, worst[0] + zoom), (worst[1] - zoom, worst[1] + zoom))
