"""Génère les STEP synthétiques de test dans tests/steps/ (gmsh seul)."""
import math, sys
from pathlib import Path
import gmsh

OUT = Path(__file__).parent / "steps"


def profile_extrude(occ, pts, arcs, width):
    """pts : liste de (x, y) ; arcs : {i: (cx, cy)} = l'arête i->i+1 est un arc."""
    p = [occ.addPoint(x, y, 0) for x, y in pts]
    cv = []
    for i in range(len(p)):
        j = (i + 1) % len(p)
        if i in arcs:
            cv.append(occ.addCircleArc(p[i], occ.addPoint(*arcs[i], 0), p[j]))
        else:
            cv.append(occ.addLine(p[i], p[j]))
    s = occ.addPlaneSurface([occ.addCurveLoop(cv)])
    return [e for e in occ.extrude([(2, s)], 0, 0, width) if e[0] == 3]


def L_bend(occ, t=2.0, R=3.0, L1=60, L2=40, W=50):
    pts = [(0, 0), (L1, 0), (L1 + R + t, R + t), (L1 + R + t, R + t + L2),
           (L1 + R, R + t + L2), (L1 + R, R + t), (L1, t), (0, t)]
    return profile_extrude(occ, pts, {1: (L1, R + t), 5: (L1, R + t)}, W)


def case_L_holes():            # pli rayonné + 2 trous + 2e tôle plate dans le même fichier
    occ = gmsh.model.occ
    vol = L_bend(occ)
    h1 = occ.addCylinder(25, -5, 25, 0, 20, 0, 5)
    h2 = occ.addCylinder(60, 30, 15, 30, 0, 0, 4)
    occ.cut(vol, [(3, h1), (3, h2)])
    occ.addBox(0, -30, 0, 80, 1.5, 40)


def case_sharp_U():            # U à arêtes vives (pas de rayon de pliage)
    t = 1.6
    pts = [(0, 0), (50, 0), (50, 30), (50 - t, 30), (50 - t, t), (t, t), (t, 30), (0, 30)]
    profile_extrude(gmsh.model.occ, pts, {}, 40)


def case_perforated():         # plaque avec grille de trous + trou trop près du bord
    occ = gmsh.model.occ
    plate = [(3, occ.addBox(0, 0, 0, 120, 80, 1.2))]
    tools = [(3, occ.addCylinder(20 + 27 * i, 20 + 20 * j, -1, 0, 0, 5, 3))
             for i in range(4) for j in range(3)]
    tools.append((3, occ.addCylinder(116, 40, -1, 0, 0, 5, 2.5)))
    occ.cut(plate, tools)


def case_disc():               # disque : contour extérieur circulaire + trou central
    occ = gmsh.model.occ
    occ.cut([(3, occ.addCylinder(0, 0, 0, 0, 0, 2, 40))], [(3, occ.addCylinder(0, 0, -1, 0, 0, 4, 6))])


def case_massive():            # bloc massif : DOIT être rejeté proprement
    gmsh.model.occ.addBox(0, 0, 0, 30, 20, 25)


def case_tiny_holes():         # plaque avec 4 trous plus petits que l'élément : bouchés
    occ = gmsh.model.occ
    plate = [(3, occ.addBox(0, 0, 0, 60, 40, 1.2))]
    occ.cut(plate, [(3, occ.addCylinder(15 + 30 * i, 12 + 16 * j, -1, 0, 0, 5, 0.6))
                    for i in range(2) for j in range(2)])


def case_pocket():             # tôle usinée chimiquement : poche de 0.3 sur une face (épaisseur variable)
    occ = gmsh.model.occ
    plate = [(3, occ.addBox(0, 0, 0, 80, 50, 1.5))]
    occ.cut(plate, [(3, occ.addBox(15, 10, 1.2, 50, 30, 1))])


def case_ribbed():             # plaque avec nervure (jonction en T) : DOIT être rejetée
    occ = gmsh.model.occ
    occ.fuse([(3, occ.addBox(0, 0, 0, 60, 40, 2))], [(3, occ.addBox(0, 19, 0, 60, 2, 20))])


def case_curved_panel():       # peau de fuselage : surface BSpline à double courbure, épaisseur 3
    occ = gmsh.model.occ
    n, L, W, R1, R2, t = 7, 400.0, 300.0, 900.0, 1500.0, 3.0

    def sheet(off):             # grille de pôles d'une calotte (R1 x R2) décalée de off
        pts = []
        for j in range(n):
            for i in range(n):
                x, y = L * (i / (n - 1) - 0.5), W * (j / (n - 1) - 0.5)
                z = x * x / (2 * R1) + y * y / (2 * R2) + off
                pts.append(occ.addPoint(x, y, z))
        return occ.addBSplineSurface(pts, n)
    s0, s1 = sheet(0.0), sheet(t)
    # solide fermé : les 2 calottes + 4 bandes de tranche
    occ.synchronize()
    b0 = [abs(c) for _, c in gmsh.model.getBoundary([(2, s0)], oriented=False)]
    b1 = [abs(c) for _, c in gmsh.model.getBoundary([(2, s1)], oriented=False)]
    sides = []
    for c0, c1 in zip(b0, b1):
        sides += [x[1] for x in occ.addThruSections([occ.addWire([c0]), occ.addWire([c1])],
                                                     makeSolid=False, makeRuled=True) if x[0] == 2]
    sl = occ.addSurfaceLoop([s0, s1] + sides, sewing=True)
    occ.addVolume([sl])


CASES = {"L_holes": case_L_holes, "sharp_U": case_sharp_U, "perforated": case_perforated,
         "disc": case_disc, "massive": case_massive, "tiny_holes": case_tiny_holes,
         "pocket": case_pocket, "ribbed": case_ribbed, "curved_panel": case_curved_panel}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in CASES.items():
        gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
        fn(); gmsh.model.occ.synchronize()
        gmsh.write(str(OUT / f"{name}.step")); gmsh.finalize()
    print(f"{len(CASES)} STEP écrits dans {OUT}")
