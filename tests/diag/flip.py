import sys, gmsh, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import step2sc8r as m
pts = np.array([[9931, -1108, 2949], [8930, -952, 2921], [6897, -981, 2557], [9371, -1148, 2882], [9593, -1010, 2964]], float)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(sys.argv[1]); gmsh.model.occ.synchronize()
vol = gmsh.model.getEntities(3)[0][1]
ana = m.analyse(gmsh, [vol], 1.0); side = m.choose_side(ana); skin = ana["sides"][side]; other = ana["sides"][1 - side]
allf = list(ana["area"])
for p in pts:
    best = None
    for f in allf:
        q = np.asarray(gmsh.model.getClosestPoint(2, f, list(p))[0])
        dd = np.linalg.norm(q - p)
        if best is None or dd < best[0]: best = (dd, f, q)
    dd, f, q = best
    n = np.asarray(gmsh.model.getNormal(f, gmsh.model.getParametrization(2, f, list(q))))
    grid, t = ana["grid"], ana["t"]
    s = m.face_orientation(gmsh, f, np.array([q]), np.array([n]), grid, t) if False else None
    ins_minus = gmsh.model.isInside(3, vol, list(q - 0.3 * n)); ins_plus = gmsh.model.isInside(3, vol, list(q + 0.3 * n))
    info = ana["info"][f]
    print(f"pt {p.astype(int)} -> face {f} dist {dd:.2f} {'PEAU MAILLEE' if f in skin else ('autre peau' if f in other else 'hors peau')} "
          f"aire {ana['area'][f]:.0f} probe d={info[0]:.2f} cos={info[1]:.2f} vis-à-vis {info[2]} ; getNormal sortante ? {ins_minus and not ins_plus} (dedans -n:{ins_minus} +n:{ins_plus})")
