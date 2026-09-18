import sys, gmsh
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import step2sc8r as m
m.STRATEGIES[:] = [s for s in m.STRATEGIES if s[0][0] in sys.argv[2]]
m.VERBOSE = True
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(sys.argv[1]); gmsh.model.occ.synchronize()
vol = gmsh.model.getEntities(3)[0][1]
p = m.mesh_part(gmsh, vol, -1, 1, 0.3, True, 1.0, 0.5, (), 1.0)
print("jac", round(p["jac"], 3), "volume", f"{p['dv']:+.2%}", "elems", len(p["hexas"]), "sous0.3", p["n_bad"], "coins", p["n_drop"], "essais", p["tries"])
if len(sys.argv) > 3: m.write_inp(sys.argv[3], [dict(name="P", **p)])
