"""Contrôle indépendant sur un lot réel : volume du maillage vs volume CAO.

    python tests/check_volumes.py dossier_step dossier_inp

Pour chaque ELSET, compare la somme des volumes des hexas au volume du solide
CAO correspondant (même numérotation _S<k> que step2sc8r.py)."""
import sys
from pathlib import Path
import numpy as np
import gmsh
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tests import read_inp, hex_volume

step_dir, inp_dir = Path(sys.argv[1]), Path(sys.argv[2])
for step in sorted(p for p in step_dir.iterdir() if p.suffix.lower() in (".step", ".stp")):
    inp = inp_dir / (step.stem + ".inp")
    if not inp.exists():
        print(f"{step.name:32s} pas d'INP"); continue
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(step)); gmsh.model.occ.synchronize()
    cad = [gmsh.model.occ.getMass(3, v) for _, v in gmsh.model.getEntities(3)]
    gmsh.finalize()
    nodes, elsets = read_inp(inp)
    ids = {n: i for i, n in enumerate(nodes)}; X = np.array(list(nodes.values()))
    for k, vcad in enumerate(cad, start=1):
        name = f"{step.stem}_S{k}".upper()
        if name not in elsets:
            print(f"{name:32s} absent"); continue
        v = np.array([hex_volume(X, [ids[n] for n in h]) for h in elsets[name]])
        print(f"{name:32s} hexas={len(v):7d} volume maillage/CAO = {v.sum() / vcad - 1:+.2%}  hexa min = {v.min():.3g}")
