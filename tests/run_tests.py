"""Tests synthétiques : python tests/run_tests.py  (code retour 0 = tout passe)

Pour chaque STEP de tests/steps : maillage, puis contrôles indépendants du
mailleur : volume maillage == volume CAO, jacobien > 0, statut attendu."""
import subprocess, sys
from pathlib import Path
import numpy as np
import gmsh

ROOT = Path(__file__).resolve().parent.parent
STEPS, OUT = ROOT / "tests" / "steps", ROOT / "tests" / "out"
EXPECT_FAIL = {"MASSIVE_S1", "RIBBED_S1"}   # pièces qui doivent être rejetées
EXPECT_IN = {"TINY_HOLES_S1": "trous_bouches=4", "POCKET_S1": "[1.2..1.5]"}   # détail attendu
VOL_TOL = 0.01                         # 1 % (facettisation des arcs)
VOL_TOL_CASE = {"CURVED_PANEL_S1": 0.002}   # sans reprojection des noeuds sur la CAO : -0.34 %


def read_inp(path):
    nodes, elsets, mode, cur = {}, {}, None, None
    for line in open(path):
        if line.startswith("**"):
            continue
        if line.startswith("*"):
            up = line.upper()
            mode = "N" if up.startswith("*NODE") else "E" if up.startswith("*ELEMENT") else None
            if mode == "E":
                assert "TYPE=SC8R" in up
                cur = line.strip().split("ELSET=")[1]; elsets[cur] = []
            continue
        v = line.split(",")
        if mode == "N":
            nodes[int(v[0])] = [float(x) for x in v[1:4]]
        elif mode == "E":
            elsets[cur].append([int(x) for x in v[1:9]])
    return nodes, elsets


def hex_volume(X, h):
    p = X[h]; c = p.mean(0); vol = 0.0
    for f in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
        q = p[list(f)]; m = q.mean(0)
        for a in range(4):
            vol += np.dot(np.cross(q[a] - c, q[(a + 1) % 4] - c), m - c) / 6
    return vol


def main():
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_test_steps.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "step2sc8r.py"), str(STEPS), str(OUT), "--jobs", "2"])
    report = {r.split(";")[1]: r.strip().split(";") for r in open(OUT / "rapport.csv").readlines()[1:]}
    failures = []
    for step in sorted(STEPS.glob("*.step")):
        gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(step)); gmsh.model.occ.synchronize()
        cad = [gmsh.model.occ.getMass(3, v) for _, v in gmsh.model.getEntities(3)]
        gmsh.finalize()
        inp = OUT / (step.stem + ".inp")
        nodes, elsets = read_inp(inp) if inp.exists() else ({}, {})
        ids = {n: i for i, n in enumerate(nodes)}
        X = np.array(list(nodes.values())) if nodes else None
        for k, vcad in enumerate(cad, start=1):
            name = f"{step.stem}_S{k}".upper()
            status = report.get(name, ["", "", "ABSENT", ""])[2]
            if name in EXPECT_FAIL:
                ok, msg = status == "ECHEC" and name not in elsets, f"rejet attendu, statut={status}"
            elif name not in elsets:
                ok, msg = False, f"pas de maillage ({report.get(name, ['', '', '', '?'])[3]})"
            else:
                vols = np.array([hex_volume(X, [ids[n] for n in h]) for h in elsets[name]])
                err = abs(vols.sum() - vcad) / vcad
                ok = status == "OK" and err < VOL_TOL_CASE.get(name, VOL_TOL) and vols.min() > 0
                ok = ok and EXPECT_IN.get(name, "") in report[name][3]
                msg = f"statut={status} err_volume={err:.2%} vol_min={vols.min():.3g} | {report[name][3]}"
            print(("PASS " if ok else "FAIL ") + f"{name:16s} {msg}")
            if not ok:
                failures.append(name)
    # reprise après plantage natif de gmsh : le lot continue, la pièce est remaillée
    # sans la stratégie qui a planté
    crash_dir = OUT / "crash"
    env = dict(__import__("os").environ, STEP2SC8R_CRASH="A-quasi-structure")
    one = crash_dir / "in"; one.mkdir(parents=True, exist_ok=True)
    (one / "sharp_U.step").write_bytes((STEPS / "sharp_U.step").read_bytes())
    subprocess.run([sys.executable, str(ROOT / "step2sc8r.py"), str(one), str(crash_dir), "--jobs", "1"],
                   env=env, capture_output=True, timeout=300)
    rep = (crash_dir / "rapport.csv").read_text() if (crash_dir / "rapport.csv").exists() else ""
    ok = ";OK;" in rep and "exclues_apres_plantage=A-quasi-structure" in rep
    print(("PASS " if ok else "FAIL ") + f"{'REPRISE_CRASH':16s} {rep.strip().splitlines()[-1][:110] if rep else 'pas de rapport'}")
    if not ok:
        failures.append("REPRISE_CRASH")
    print(f"\n{'TOUT PASSE' if not failures else 'ECHECS : ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
