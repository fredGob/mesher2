#!/usr/bin/env python3
"""
step2sc8r.py : dossier de STEP (pièces minces : tôles pliées, peaux et cadres
               usinés, épaisseur localement variable)  ->  dossier d'INP Abaqus en SC8R.

Principe (par solide) :
  0. nettoyage OCC optionnel (petites arêtes / faces), sur une copie, gardé
     seulement si le volume est conservé
  1. triangulation de la surface du solide (gmsh) avec normales CAO sortantes,
     contrôlée par le théorème de la divergence (volume triangulé == volume CAO)
  2. épaisseur locale mesurée par lancer de rayons (grille numpy) : pour
     chaque face, distance de sortie de la matière le long de la normale
     intérieure et face en vis-à-vis
  3. faces de peau = distance ~ épaisseur et vis-à-vis parallèle ; les 2 côtés
     sont séparés par 2-coloration (adjacence = même côté, vis-à-vis = côté
     opposé). Conflit -> nervures / jonction en T / massif -> rejet explicite
  4. couronnes réglées autour des trous circulaires (partition OCC locale)
  5. maillage 100% quads du côté le plus simple (transfinite auto + blossom +
     subdivision) avec cascade de rattrapage si la qualité est insuffisante
  6. extrusion nodale : chaque noeud est projeté sur l'autre peau par un rayon
     le long de la normale CAO (distance affinée sur la surface CAO exacte)
     -> hexa d'épaisseur locale exacte, onglet automatique sur arête vive
  7. écriture INP (SC8R, direction d'empilement = épaisseur, garantie)

Usage : python step2sc8r.py dossier_step dossier_inp --size 4 --layers 1 --jobs 8
Dépendances : gmsh, numpy
"""
import argparse, math, os, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import multiprocessing
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # 1 thread par process : le parallélisme est par fichier
import numpy as np

VERBOSE = False
_STATE = None                           # tube vers le process parent (étape en cours)


def state(msg):
    """Signale l'étape en cours au parent : en cas de plantage natif de gmsh,
    il sait quelle stratégie éviter à la relance."""
    if _STATE is not None:
        try:
            _STATE.send(("etat", msg))
        except Exception:
            pass


def log(msg):
    if VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------- triangulation
def outer_faces(gmsh, vols):
    """Faces externes des volumes. combined=True : les faces internes (partagées
    par 2 volumes d'une même pièce partitionnée) s'annulent. Sur ces faces,
    getNormal de gmsh est la normale SORTANTE du solide (vérifié par le
    contrôle de fermeture de la triangulation)."""
    return sorted({abs(tag) for _, tag in gmsh.model.getBoundary(
        [(3, v) for v in vols], combined=True, oriented=True)})


def triangulate(gmsh, faces, size, volume):
    """Triangule les faces (taille ~size, bornée pour ne pas exploser sur les
    rayons de pliage). Renvoie (T (m,3,3), fid (m,), nout (m,3), aire (m,)),
    triangles orientés comme getNormal = normale sortante.
    Contrôle : volume fermé (divergence) == volume CAO."""
    o = gmsh.option.setNumber
    gmsh.model.mesh.clear()
    gmsh.model.mesh.removeConstraints()
    gmsh.model.setVisibility(gmsh.model.getEntities(), 0)
    gmsh.model.setVisibility([(2, f) for f in faces], 1, recursive=True)
    o("Mesh.MeshOnlyVisible", 1)
    o("Mesh.MeshSizeMin", size / 3.0); o("Mesh.MeshSizeMax", size)
    o("Mesh.MeshSizeFromCurvature", 8)      # 8 segments par tour : la distance est affinée sur la CAO ensuite
    o("Mesh.MeshSizeExtendFromBoundary", 0)
    o("Mesh.Algorithm", 6); o("Mesh.RecombineAll", 0); o("Mesh.SubdivisionAlgorithm", 0); o("Mesh.Smoothing", 0)
    gmsh.model.mesh.generate(2)
    T, fid = [], []
    for f in faces:
        tags, coords, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True)
        X = np.asarray(coords).reshape(-1, 3)
        idx = dict(zip(tags.tolist(), range(len(tags))))
        etypes, _, enodes = gmsh.model.mesh.getElements(2, f)
        for et, en in zip(etypes, enodes):
            if et != 2:
                continue
            T.append(X[[[idx[a] for a in e] for e in np.asarray(en).reshape(-1, 3).tolist()]])
            fid += [f] * len(T[-1])
    if not T:
        raise ValueError("triangulation vide")
    T, fid = np.concatenate(T), np.array(fid)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    vdiv = np.einsum("ij,ij->", T.mean(1), n) / 6.0
    if abs(vdiv - volume) > 0.03 * volume:
        raise ValueError(f"triangulation non fermée (volume {vdiv:.4g} vs CAO {volume:.4g})")
    a2 = np.linalg.norm(n, axis=1)
    return T, fid, n / np.maximum(a2, 1e-30)[:, None], a2 / 2.0


class TriGrid:
    """Grille uniforme sur une triangulation, pour des lancers de rayons courts
    (longueur ~ épaisseur) : Möller-Trumbore vectorisé sur les candidats."""

    def __init__(self, T, nout, fid, cell):
        self.T, self.n, self.fid, self.cell = T, nout, fid, float(cell)
        self.e1, self.e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        self.lo = T.min(axis=(0, 1)) - cell
        lo_c = np.floor((T.min(1) - self.lo) / cell).astype(np.int64)
        hi_c = np.floor((T.max(1) - self.lo) / cell).astype(np.int64)
        self.shape = hi_c.max(0) + 2
        ext = hi_c - lo_c
        keys, idx = [], []
        for i in range(int(ext[:, 0].max()) + 1):
            for j in range(int(ext[:, 1].max()) + 1):
                for k in range(int(ext[:, 2].max()) + 1):
                    m = (ext[:, 0] >= i) & (ext[:, 1] >= j) & (ext[:, 2] >= k)
                    keys.append(self.key(lo_c[m] + [i, j, k])); idx.append(np.nonzero(m)[0])
        keys, idx = np.concatenate(keys), np.concatenate(idx)
        o = np.argsort(keys, kind="stable")
        self.idx = idx[o]
        self.uniq, self.start = np.unique(keys[o], return_index=True)
        self.end = np.append(self.start[1:], len(keys))

    def key(self, c):
        return (c[..., 0] * self.shape[1] + c[..., 1]) * self.shape[2] + c[..., 2]

    def candidates(self, pmin, pmax):
        a = np.clip(np.floor((pmin - self.lo) / self.cell).astype(np.int64), 0, self.shape - 1)
        b = np.clip(np.floor((pmax - self.lo) / self.cell).astype(np.int64), 0, self.shape - 1)
        cells = np.array([[i, j, k] for i in range(a[0], b[0] + 1) for j in range(a[1], b[1] + 1)
                          for k in range(a[2], b[2] + 1)])
        ks = self.key(cells)
        pos = np.searchsorted(self.uniq, ks)
        ok = (pos < len(self.uniq)) & (self.uniq[np.minimum(pos, len(self.uniq) - 1)] == ks)
        if not ok.any():
            return np.empty(0, int)
        return np.unique(np.concatenate([self.idx[self.start[p]:self.end[p]] for p in pos[ok]]))

    def first_hit(self, P, D, dmax, eps, exit_only=True):
        """Pour chaque rayon p + s*d (eps < s <= dmax) : (distance, triangle) ou (inf, -1).
        Seules les traversées sortantes comptent (d . normale sortante > 0)."""
        dist = np.full(len(P), np.inf); hit = np.full(len(P), -1)
        for r, (p, d) in enumerate(zip(P, D)):
            q = p + dmax * d
            c = self.candidates(np.minimum(p, q), np.maximum(p, q))
            if len(c) == 0:
                continue
            e1, e2, v0 = self.e1[c], self.e2[c], self.T[c, 0]
            h = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, h)
            ok = np.abs(det) > 1e-14
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            s = p - v0
            u = np.einsum("ij,ij->i", s, h) * inv
            qv = np.cross(s, e1)
            v = (qv @ d) * inv
            t = np.einsum("ij,ij->i", qv, e2) * inv
            m = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > eps) & (t <= dmax)
            if exit_only:                       # sortie de la matière seulement (ignore sa propre corde)
                m &= (self.n[c] @ d) > 0
            if m.any():
                i = np.argmin(np.where(m, t, np.inf))
                dist[r] = t[i]; hit[r] = c[i]
        return dist, hit


def refine_on_cad(gmsh, P, D, dist, hit, fid):
    """Affine la distance d'impact sur la surface CAO exacte de la face touchée
    (projection du point d'impact triangulé sur la surface)."""
    out = dist.copy()
    ok = np.nonzero(hit >= 0)[0]
    for f in np.unique(fid[hit[ok]]):
        sel = ok[fid[hit[ok]] == f]
        q = P[sel] + dist[sel, None] * D[sel]
        try:
            qc = np.asarray(gmsh.model.getClosestPoint(2, f, q.ravel())[0]).reshape(-1, 3)
        except Exception:
            continue
        d2 = np.einsum("ij,ij->i", qc - P[sel], D[sel])
        good = np.abs(d2 - dist[sel]) < 0.3 * dist[sel]   # projection cohérente
        out[sel[good]] = d2[good]
    return out


# ------------------------------------------------------- analyse de la pièce
def probe_faces(gmsh, grid, faces, tri_area, dmax, eps, nsamp=8):
    """Pour chaque face : distance médiane de sortie de la matière le long de
    la normale intérieure, cosinus avec la face en vis-à-vis, face en vis-à-vis."""
    P, N, F = [], [], []
    for f in faces:
        idx = np.nonzero(grid.fid == f)[0]
        idx = idx[np.argsort(-tri_area[idx])[:nsamp]]
        p, n = grid.T[idx].mean(1), grid.n[idx]
        try:                                    # points et normales exacts sur la CAO
            pc, uv = gmsh.model.getClosestPoint(2, f, p.ravel())
            pc = np.asarray(pc).reshape(-1, 3)
            nc = np.asarray(gmsh.model.getNormal(f, uv)).reshape(-1, 3)
            good = np.linalg.norm(pc - p, axis=1) < 0.5 * dmax
            p[good], n[good] = pc[good], nc[good]
        except Exception:
            pass
        P.append(p); N.append(n); F += [f] * len(idx)
    P, N, F = np.concatenate(P), np.concatenate(N), np.array(F)
    dist, hit = grid.first_hit(P, -N, dmax, eps)
    dist = refine_on_cad(gmsh, P, -N, dist, hit, grid.fid)
    cos = np.where(hit >= 0, -np.einsum("ij,ij->i", N, grid.n[np.maximum(hit, 0)]), 0.0)
    info = {}
    for f in faces:
        m = F == f
        ok = np.isfinite(dist[m])
        if ok.sum() * 2 < m.sum():
            info[f] = (math.inf, 0.0, None)          # sortie au-delà de dmax : face épaisse
            continue
        pair = Counter(grid.fid[hit[m][ok]].tolist()).most_common(1)[0][0]
        info[f] = (float(np.median(dist[m][ok])), float(np.median(cos[m][ok])), pair)
    return info


def two_color(skin, edges, pair, area):
    """Sépare les faces de peau en 2 côtés : faces adjacentes = même côté,
    faces en vis-à-vis = côtés opposés. Les faces en conflit (petites tranches
    étroites vues comme peau...) sont retirées si leur aire est négligeable."""
    skin = set(skin)
    for _ in range(6):
        e2f = defaultdict(list)
        for f in skin:
            for e in edges[f]:
                e2f[e].append(f)
        color, bad = {}, set()
        for f0 in sorted(skin, key=lambda f: -area[f]):
            if f0 in color:
                continue
            # nouvelle composante (ex. fond de poche isolé par ses parois) : couleur
            # déduite du vis-à-vis s'il est déjà coloré
            color[f0] = 1 - color[pair[f0]] if pair[f0] in color else 0
            stack = [f0]
            while stack:
                f = stack.pop()
                nb = [(g, color[f]) for e in edges[f] for g in e2f[e] if g != f]
                if pair[f] in skin:
                    nb.append((pair[f], 1 - color[f]))
                for g, c in nb:
                    if g not in color:
                        color[g] = c; stack.append(g)
                    elif color[g] != c:
                        bad.add(min(f, g, key=lambda x: area[x]))
        if not bad:
            return [[f for f in skin if color[f] == c] for c in (0, 1)], 0.0
        a_bad = sum(area[f] for f in bad)
        if a_bad > 0.02 * sum(area[f] for f in skin):
            return None, a_bad
        skin -= bad
    return None, a_bad


def analyse(gmsh, vols, tvar, t_ref=None):
    """Classe les faces d'UNE pièce (éventuellement partitionnée) : épaisseur
    de référence, faces de peau des 2 côtés, triangulation + grille pour les
    rayons. Lève ValueError avec un message clair si ce n'est pas une pièce mince."""
    occ = gmsh.model.occ
    volume = sum(occ.getMass(3, v) for v in vols)
    faces = outer_faces(gmsh, vols)
    area = {f: occ.getMass(2, f) for f in faces}
    A = sum(area.values())
    t_est = t_ref or 2.0 * volume / A                     # V ~ A_peau * t / 2 * 2
    size = 3.0 * t_est
    t0 = time.time()
    T, fid, nout, ta = triangulate(gmsh, faces, size, volume)
    log(f"  triangulation : {len(faces)} faces, {len(T)} triangles, {time.time() - t0:.1f}s")
    dmax = 4.0 * t_est
    grid = TriGrid(T, nout, fid, max(size, dmax))
    t0 = time.time()
    info = probe_faces(gmsh, grid, faces, ta, dmax, 0.02 * t_est)
    log(f"  sondage épaisseur : {time.time() - t0:.1f}s")
    if t_ref is None:                                     # médiane pondérée par l'aire
        cand = [f for f in faces if math.isfinite(info[f][0]) and info[f][1] > 0.8]
        if not cand:
            raise ValueError("aucune face avec une face parallèle en vis-à-vis : pas une pièce mince")
        d = np.array([info[f][0] for f in cand]); a = np.array([area[f] for f in cand])
        o = np.argsort(d); cum = np.cumsum(a[o])
        t_ref = float(d[o][np.searchsorted(cum, cum[-1] / 2.0)])
    t_lo, t_hi = t_ref / (1.0 + tvar), t_ref * (1.0 + tvar)
    skin = [f for f in faces if info[f][1] > 0.8 and t_lo <= info[f][0] <= t_hi]
    thick = [f for f in faces if info[f][1] > 0.8 and info[f][0] > t_hi and area[f] > 4 * t_ref ** 2]
    a_thick = sum(area[f] for f in thick)
    edges = {f: {abs(e) for _, e in gmsh.model.getBoundary([(2, f)], combined=False, oriented=False)}
             for f in faces}
    pair = {f: info[f][2] for f in faces}
    sides, a_bad = two_color(skin, edges, pair, area)
    if sides is None:
        raise ValueError(f"peaux incompatibles (faces en conflit : {a_bad:.0f} mm2) : "
                         "nervures, jonctions en T ou pièce massive")
    a_side = [sum(area[f] for f in s) for s in sides]
    if sum(a_side) < 0.6 * A:
        raise ValueError(f"faces de peau = {100 * sum(a_side) / A:.0f} % de la surface "
                         f"(épaisseur hors [{t_lo:.3g}, {t_hi:.3g}] sur {a_thick:.0f} mm2) : pas une pièce mince")
    if min(a_side) < 0.5 * max(a_side):
        raise ValueError(f"les deux peaux ont des aires trop différentes ({a_side[0]:.0f} / {a_side[1]:.0f} mm2)")
    d_skin = [info[f][0] for f in skin]
    return dict(t=t_ref, t_lo=t_lo, t_hi=t_hi, t_min=min(d_skin), t_max=max(d_skin), sides=sides,
                area=area, grid=grid, info=info, volume=volume, size=size)


def choose_side(ana):
    """Côté à mailler : le moins découpé (à égalité, le plus grand)."""
    s0, s1 = ana["sides"]
    a0, a1 = (sum(ana["area"][f] for f in s) for s in (s0, s1))
    return 0 if (len(s0), -a0) <= (len(s1), -a1) else 1


def match_side(ana_ref, faces_ref, ana):
    """Indice du côté de `ana` (pièce partitionnée) qui coïncide géométriquement
    avec les faces `faces_ref` de la pièce d'origine."""
    ref, t, g = set(faces_ref), ana_ref["t"], ana_ref["grid"]
    score = []
    for faces in ana["sides"]:
        idx = np.nonzero(np.isin(ana["grid"].fid, faces))[0]
        idx = idx[:: max(1, len(idx) // 200)]
        P, N = ana["grid"].T[idx].mean(1), ana["grid"].n[idx]
        _, hit = g.first_hit(P - 0.3 * t * N, N, 0.6 * t, 0.0)
        hit = hit[hit >= 0]
        score.append(np.mean([g.fid[i] in ref for i in hit]) if len(hit) else 0.0)
    if max(score) < 0.5:
        raise ValueError("côté de maillage introuvable après partition")
    return int(np.argmax(score))


# ------------------------------------------------------- couronnes de trous
def inside_any(gmsh, vols, p):
    return any(gmsh.model.isInside(3, v, list(p)) for v in vols)


def find_holes(gmsh, vol, skin, ana, h, r_min):
    """Trous circulaires débouchants (rayon >= r_min, les plus petits seront
    bouchés) dans les faces PLANES de la peau, avec la place nécessaire pour
    une couronne de largeur w."""
    occ, holes, t = gmsh.model.occ, [], ana["t"]
    for f in skin:
        if gmsh.model.getType(2, f) != "Plane":
            continue
        _, loops = occ.getCurveLoops(f)
        for loop in loops:
            loop = [abs(int(e)) for e in loop]
            if not all(gmsh.model.getType(1, e) == "Circle" for e in loop):
                continue
            lengths = np.array([occ.getMass(1, e) for e in loop])
            r = lengths.sum() / (2 * math.pi)
            if r < r_min:
                continue
            c = sum(np.array(occ.getCenterOfMass(1, e)) * l for e, l in zip(loop, lengths)) / lengths.sum()
            pt = gmsh.model.getBoundary([(1, loop[0])], oriented=False, combined=False)[0][1]
            p0 = np.array(gmsh.model.getValue(0, pt, []))
            if abs(np.linalg.norm(p0 - c) - r) > 1e-3 * r:     # arcs non concentriques
                continue
            n = np.array(gmsh.model.getNormal(f, gmsh.model.getParametrization(2, f, list(p0))))
            d = -n                                             # vers la matière
            u = (p0 - c) / r
            v = np.cross(n, u)
            w = max(min(max(r, h), 2 * h), 1.5 * t)
            ring = lambda k, a: c + (r + k * w) * (math.cos(a) * u + math.sin(a) * v)
            angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
            depth = 0.5 * min(ana["info"][f][0], t)
            if not inside_any(gmsh, [vol], ring(0.5, 0.3) + depth * d):   # contour extérieur, pas un trou
                continue
            # place libre : le cylindre r+1.4w doit être plein de matière
            if not all(inside_any(gmsh, [vol], ring(1.4, a) + depth * d) for a in angles):
                continue
            holes.append(dict(c=c, r=r, w=w, u=u, v=v, d=d))
    # couronnes qui se chevauchent entre elles : on garde la première
    kept = []
    for hl in holes:
        if all(np.linalg.norm(hl["c"] - k["c"]) > 1.2 * (hl["r"] + hl["w"] + k["r"] + k["w"]) for k in kept):
            kept.append(hl)
    return kept


def partition_rings(gmsh, vol, holes, ana):
    """Copie le solide et le découpe en couronnes à 4 quadrants autour des
    trous. L'original reste intact (stratégies de repli). Retourne les volumes."""
    occ, tools = gmsh.model.occ, []
    for hl in holes:
        c, r, R, u, v = hl["c"], hl["r"], hl["r"] + hl["w"], hl["u"], hl["v"]
        ang = np.arange(4) * math.pi / 2                     # 1er rayon sur le sommet du cercle CAO
        cp = occ.addPoint(*c)
        pin = [occ.addPoint(*(c + r * (math.cos(a) * u + math.sin(a) * v))) for a in ang]
        pout = [occ.addPoint(*(c + R * (math.cos(a) * u + math.sin(a) * v))) for a in ang]
        curves = [occ.addLine(pin[i], pout[i]) for i in range(4)]
        curves += [occ.addCircleArc(pout[i], cp, pout[(i + 1) % 4]) for i in range(4)]
        ext = occ.extrude([(1, x) for x in curves], *(hl["d"] * 1.3 * ana["t_hi"]))
        tools += [x for x in ext if x[0] == 2]
    copy = occ.copy([(3, vol)])
    out, _ = occ.fragment(copy, tools)
    occ.synchronize()
    return sorted({v for dim, v in out if dim == 3})


# ---------------------------------------------------------- bouchage trous
def loop_size(gmsh, curves):
    """Diagonale de la boîte englobante d'une boucle de courbes."""
    bb = np.array([gmsh.model.getBoundingBox(1, abs(int(c))) for c in curves])
    return float(np.linalg.norm(bb[:, 3:].max(0) - bb[:, :3].min(0)))


def fill_small_holes(gmsh, skin, r_max):
    """Recrée chaque face de peau sans ses boucles intérieures de diamètre
    < 2 r_max, sur la même surface CAO (addPlaneSurface pour un plan,
    addTrimmedSurface sinon). Les faces originales restent en place (le solide
    est intact). Renvoie (nouvelle peau, nb trous bouchés)."""
    occ, out, n_fill = gmsh.model.occ, [], 0
    for f in skin:
        try:
            loops, curves = occ.getCurveLoops(f)
        except Exception:
            out.append(f); continue
        if len(loops) < 2:
            out.append(f); continue
        sizes = [loop_size(gmsh, c) for c in curves]
        outer = int(np.argmax(sizes))
        keep = [int(l) for k, l in enumerate(loops) if k == outer or sizes[k] >= 2 * r_max]
        if len(keep) == len(loops):
            out.append(f); continue
        nf = None
        try:
            if gmsh.model.getType(2, f) == "Plane":
                nf = occ.addPlaneSurface(keep)
            else:                               # même surface support, fils 3D projetés
                nf = occ.addTrimmedSurface(f, keep, wire3D=True)
            occ.synchronize()
            if abs(abs(occ.getMass(2, nf)) - abs(occ.getMass(2, f))) > 0.5 * abs(occ.getMass(2, f)):
                raise ValueError("aire incohérente")
            out.append(nf); n_fill += len(loops) - len(keep)
        except Exception:                       # face recréée invalide : on garde l'originale
            out.append(f)
            if nf is not None:
                try:
                    occ.remove([(2, nf)], recursive=False); occ.synchronize()
                except Exception:
                    pass
    return out, n_fill


# ------------------------------------------------------------------ maillage
def median_edge(gmsh, skin):
    """Longueur médiane des arêtes des quads de la peau maillée."""
    L = []
    for f in skin:
        tags, coords, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True)
        X = np.asarray(coords).reshape(-1, 3)
        idx = dict(zip(tags.tolist(), range(len(tags))))
        for et, en in zip(*gmsh.model.mesh.getElements(2, f)[::2]):
            if et != 3:
                continue
            q = np.asarray(en).reshape(-1, 4).tolist()
            P = X[[[idx[a] for a in e] for e in q]]
            L += np.linalg.norm(P[:, 1] - P[:, 0], axis=1).tolist()
    return float(np.median(L)) if L else 0.0


def mesh_skin(gmsh, skin, h, transfinite, algo):
    """Maille la peau en 100% quads de taille cible h. La taille réelle est
    mesurée et corrigée une fois (les algos ne respectent la consigne qu'à un
    facteur près : subdivision, quasi-structuré avec scaling 0.75...)."""
    # l'algo 11 met en cache sa carte de taille (champ + vue 'guiding_field') :
    # sans purge, la taille du 1er maillage est réutilisée pour tous les suivants
    # ORDRE IMPOSÉ (sinon segfault gmsh 4.15 au generate suivant) : champs retirés,
    # champ de fond désactivé, puis vues retirées
    fields = gmsh.model.mesh.field.list()
    if len(fields):
        for fl in fields:
            gmsh.model.mesh.field.remove(fl)
        gmsh.model.mesh.field.setAsBackgroundMesh(0)
    for v in gmsh.view.getTags():
        gmsh.view.remove(v)
    subdivide = algo != 11                  # l'algo 11 fait sa propre subdivision
    size = 2.0 * h                          # les deux voies divisent la taille par ~2
    o = gmsh.option.setNumber
    gmsh.model.setVisibility(gmsh.model.getEntities(), 0)
    gmsh.model.setVisibility([(2, f) for f in skin], 1, recursive=True)
    o("Mesh.MeshOnlyVisible", 1)
    o("Mesh.MeshSizeFromCurvature", 0)
    o("Mesh.MeshSizeExtendFromBoundary", 0)
    o("Mesh.Algorithm", algo)
    o("Mesh.RecombineAll", 1)
    o("Mesh.RecombinationAlgorithm", 1)     # blossom (le "full-quad" 3 plante si nb impair de segments)
    o("Mesh.SubdivisionAlgorithm", 1 if subdivide else 0)   # 1 = garantit 100% quads
    o("Mesh.Smoothing", 10)
    for _ in range(2):
        gmsh.model.mesh.clear()
        gmsh.model.mesh.removeConstraints()
        o("Mesh.MeshSizeMin", size); o("Mesh.MeshSizeMax", size)
        if transfinite:                     # plis, couronnes, faces à 4 côtés -> maillage réglé
            gmsh.model.mesh.setTransfiniteAutomatic([(2, f) for f in skin],
                                                    cornerAngle=2.0, recombine=True)
        gmsh.model.mesh.generate(2)
        L = median_edge(gmsh, skin)
        if L <= 0 or abs(L - h) < 0.15 * h:
            break
        size *= h / L                       # correction proportionnelle, une seule fois
    gmsh.model.mesh.optimize("Laplace2D", niter=5)


def project_skin_nodes(gmsh, skin):
    """Reprojette chaque noeud de la peau maillée sur SA face (noeuds
    intérieurs) ou SON arête CAO (noeuds de bord). Indispensable avec l'algo 11 :
    il maille une triangulation de fond et laisse ses noeuds jusqu'à plusieurs
    mm hors des surfaces BSpline -> épaisseur sous-estimée. Renvoie l'écart max corrigé."""
    moved = 0.0
    ents = [(2, f) for f in skin]
    ents += sorted({(1, abs(c)) for f in skin for _, c in gmsh.model.getBoundary(
        [(2, f)], combined=False, oriented=False)})
    for dim, tag in ents:
        tags, coords, _ = gmsh.model.mesh.getNodes(dim, tag)
        if len(tags) == 0:
            continue
        X = np.asarray(coords).reshape(-1, 3)
        try:
            Y, par = gmsh.model.getClosestPoint(dim, tag, X.ravel())
        except Exception:
            continue
        Y = np.asarray(Y).reshape(-1, 3); par = np.asarray(par).reshape(len(X), -1)
        dist = np.linalg.norm(Y - X, axis=1)
        moved = max(moved, float(dist.max()))
        for tg, y, p, dd in zip(tags.tolist(), Y, par, dist):
            if dd > 1e-9:
                gmsh.model.mesh.setNode(tg, list(y), list(p))
    return moved


def face_orientation(gmsh, f, coords, n, grid, t):
    """+1 si getNormal(f) sort de la matière, -1 sinon (faces recréées ou
    partitionnées) : comparaison avec la triangulation d'origine par un rayon
    court traversant la surface aux noeuds intérieurs de la face."""
    inner = gmsh.model.mesh.getNodes(2, f)[0]
    if len(inner) == 0:
        return 1.0
    pos = dict(zip(gmsh.model.mesh.getNodes(2, f, includeBoundary=True)[0].tolist(), range(len(coords))))
    sel = [pos[tg] for tg in inner.tolist()[:: max(1, len(inner) // 12)]]
    P, N = coords[sel], n[sel]
    _, hit = grid.first_hit(P - 0.3 * t * N, N, 0.6 * t, 0.0, exit_only=False)
    ok = hit >= 0
    if not ok.any():
        raise ValueError(f"face {f} introuvable dans la triangulation")
    s = np.einsum("ij,ij->i", grid.n[hit[ok]], N[ok])
    return 1.0 if s.sum() >= 0 else -1.0


def collect_skin(gmsh, skin, grid, t):
    """Peau maillée -> tableaux numpy : xyz (n,3), N normales sortantes (n,3),
    Q quads (m,4) orientés selon N, ent[i] = (dim, tag) de l'entité CAO qui
    porte le noeud i (0 = sommet, fixe ; 1 = arête ; 2 = face)."""
    ids, xyz, nsum, quads = {}, [], {}, []
    for f in skin:
        tags, coords, _ = gmsh.model.mesh.getNodes(2, f, includeBoundary=True)
        coords = np.asarray(coords).reshape(-1, 3)
        uv = gmsh.model.getParametrization(2, f, coords.ravel())
        n = np.asarray(gmsh.model.getNormal(f, uv)).reshape(-1, 3)
        n *= face_orientation(gmsh, f, coords, n, grid, t)             # sortante
        for tg, x, nn in zip(tags.tolist(), coords, n):
            if tg not in ids:
                ids[tg] = len(xyz); xyz.append(x); nsum[tg] = nn.copy()
            else:
                nsum[tg] += nn
        etypes, _, enodes = gmsh.model.mesh.getElements(2, f)
        for et, en in zip(etypes, enodes):
            if et != 3:
                raise ValueError("élément non-quad dans la peau")
            quads += np.asarray(en).reshape(-1, 4).tolist()
    if not quads:
        raise ValueError("peau non maillée")
    xyz = np.array(xyz)
    Q = np.array([[ids[tg] for tg in q] for q in quads])
    N = np.zeros_like(xyz)
    for tg, i in ids.items():
        N[i] = nsum[tg] / max(np.linalg.norm(nsum[tg]), 1e-30)
    ent = [(0, 0)] * len(xyz)                              # défaut : sommet CAO, fixe
    for dim, tags in ((2, skin), (1, sorted({abs(c) for f in skin for _, c in gmsh.model.getBoundary(
            [(2, f)], combined=False, oriented=False)}))):
        for tag in tags:
            for tg in gmsh.model.mesh.getNodes(dim, tag)[0].tolist():
                if tg in ids and ent[ids[tg]] == (0, 0):
                    ent[ids[tg]] = (dim, tag)
    # orientation de chaque quad selon la normale sortante (par sa diagonale :
    # robuste même pour un quad très déformé)
    d = np.cross(xyz[Q[:, 2]] - xyz[Q[:, 0]], xyz[Q[:, 3]] - xyz[Q[:, 1]])
    flip = np.einsum("ij,ij->i", d, N[Q].mean(1)) < 0
    Q[flip] = Q[flip][:, ::-1]
    return xyz, N, Q, ent


def quad_quality(X, Q, N):
    """Jacobien normalisé 2D min aux 4 coins de chaque quad, par rapport à la
    normale sortante moyenne de ses noeuds. 1 = carré, <= 0 = retourné. Sur
    les pièces réelles il est égal au jacobien 3D de l'hexa (vérifié)."""
    P = X[Q]
    n = N[Q].mean(1)
    q = np.full(len(Q), np.inf)
    for c in range(4):
        a, b = P[:, (c + 1) % 4] - P[:, c], P[:, (c - 1) % 4] - P[:, c]
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        q = np.minimum(q, np.einsum("ij,ij->i", np.cross(a, b), n) / np.maximum(den, 1e-30))
    return q


def edge_counts(Q):
    ec = Counter()
    for q in Q.tolist():
        for a in range(4):
            i, j = q[a], q[(a + 1) % 4]
            ec[(min(i, j), max(i, j))] += 1
    return ec


def drop_flat_corner_quads(xyz, N, Q, q_bad, h):
    """Défauts n°1 des STEP réels, sur le contour de la pièce : un mauvais quad
    de bord dont 2 arêtes de bord se rejoignent à plus de 150° (sommet CAO
    parasite sur un bord presque droit), ou qui porte une arête de bord très
    courte (micro-arête CAO < 0,15 h). Aucun lissage ne peut le corriger : on le
    supprime (petit coin de matière, aire < 0,5 h², contrôlé ensuite par le
    volume). Refusé si le contour ne reste pas une courbe simple (chaque noeud
    doit garder 0 ou 2 arêtes de bord). Renvoie (Q, nb supprimés)."""
    dropped = 0
    for _ in range(4):
        q = quad_quality(xyz, Q, N)
        ec = edge_counts(Q)
        nbnd = Counter()                                   # arêtes de bord par noeud
        for (a, b), c in ec.items():
            if c == 1:
                nbnd[a] += 1; nbnd[b] += 1
        keep = np.ones(len(Q), bool)
        touched = set()
        for k in np.argsort(q):
            if q[k] >= q_bad:
                break
            r = Q[k].tolist()
            if touched & set(r):                           # un seul quad retiré par zone et par passe
                continue
            edges = [(min(r[a], r[(a + 1) % 4]), max(r[a], r[(a + 1) % 4])) for a in range(4)]
            bedges = [e for e in edges if ec[e] == 1]
            if not bedges:
                continue
            P = xyz[r]
            if 0.5 * np.linalg.norm(np.cross(P[2] - P[0], P[3] - P[1])) > 0.5 * h * h:
                continue
            short = min(np.linalg.norm(xyz[a] - xyz[b]) for a, b in bedges) < 0.15 * h
            flat = False
            for a in range(4):
                v, p, n_ = r[a], r[(a - 1) % 4], r[(a + 1) % 4]
                if ec[(min(p, v), max(p, v))] == 1 and ec[(min(v, n_), max(v, n_))] == 1:
                    u, w = xyz[p] - xyz[v], xyz[n_] - xyz[v]
                    c = np.dot(u, w) / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-30)
                    flat |= c < math.cos(math.radians(150))
            if not (short or flat):
                continue
            ok = True
            for v in r:        # après retrait : ses arêtes de bord disparaissent, ses arêtes internes deviennent bord
                gone = sum(1 for e in edges if v in e and ec[e] == 1)
                new = sum(1 for e in edges if v in e and ec[e] == 2)
                if nbnd[v] - gone + new not in (0, 2):
                    ok = False
            if not ok:
                continue
            keep[k] = False; touched |= set(r); dropped += 1
        if keep.all():
            break
        Q = Q[keep]
    return Q, dropped


def smooth_quads(gmsh, xyz, N, Q, ent, q_target=0.4, sweeps=40):
    """Lissage intelligent : autour des quads de qualité < q_target, chaque
    noeud non fixe va vers le barycentre de ses voisins, reprojeté sur SA face
    ou SON arête CAO ; gardé seulement si le pire quad voisin s'améliore."""
    q = quad_quality(xyz, Q, N)
    node_quads, nbrs = defaultdict(list), defaultdict(set)
    for k, r in enumerate(Q.tolist()):
        for a in range(4):
            node_quads[r[a]].append(k)
            nbrs[r[a]].update((r[(a + 1) % 4], r[(a + 3) % 4]))
    for _ in range(sweeps):
        bad = np.nonzero(q < q_target)[0]
        if len(bad) == 0:
            break
        cand = {i for k in bad for i in Q[k]}
        cand |= {j for i in list(cand) for j in nbrs[i]}
        moved = False
        for i in cand:
            dim, tag = ent[i]
            if dim == 0:
                continue
            ks = node_quads[i]
            old, qold = xyz[i].copy(), q[ks].min()
            target = xyz[list(nbrs[i])].mean(0)
            for w in (1.0, 0.5, 0.25):
                try:
                    p = np.asarray(gmsh.model.getClosestPoint(dim, tag, list(old + w * (target - old)))[0])
                except Exception:
                    break
                xyz[i] = p
                qn = quad_quality(xyz, Q[ks], N)
                if qn.min() > qold + 1e-3:
                    q[ks] = qn; moved = True
                    break
                xyz[i] = old
        if not moved:
            break
    return xyz


def extrude_to_hexa(gmsh, skin, ana, layers, h=None):
    """Noeuds + hexas à partir de la peau maillée : nettoyage des quads (coins
    plats du contour, lissage), puis chaque noeud est projeté sur l'autre peau
    le long de la normale CAO sortante (moyennée aux arêtes)."""
    grid, t = ana["grid"], ana["t"]
    xyz, N, Q, ent = collect_skin(gmsh, skin, grid, t)
    n_drop = 0
    if h is not None:
        Q, n_drop = drop_flat_corner_quads(xyz, N, Q, 0.3, h)
        xyz = smooth_quads(gmsh, xyz, N, Q, ent)
        used = np.unique(Q)                           # noeuds orphelins retirés
        if len(used) < len(xyz):
            new = -np.ones(len(xyz), int); new[used] = np.arange(len(used))
            xyz, N, Q = xyz[used], N[used], new[Q]
    nn = len(xyz)
    # bord du patch (arêtes vues par un seul quad) : origine des rayons rentrée
    # dans la face pour éviter les rayons rasants le long des tranches
    ecount = edge_counts(Q)
    adj = defaultdict(set)
    for (i, j) in ecount:
        adj[i].add(j); adj[j].add(i)
    bnd = {i for e, c in ecount.items() if c == 1 for i in e}
    cent = xyz[Q].mean(1)
    acc = np.zeros_like(xyz); cnt = np.zeros(nn)
    np.add.at(acc, Q.ravel(), np.repeat(cent, 4, axis=0)); np.add.at(cnt, Q.ravel(), 1)
    origin = xyz.copy()
    for i in bnd:
        e = acc[i] / cnt[i] - xyz[i]
        e -= np.dot(e, N[i]) * N[i]
        ne = np.linalg.norm(e)
        if ne > 0:
            origin[i] += min(0.3 * t, 0.3 * ne) * e / ne
    d, hit = grid.first_hit(origin, -N, 3.0 * ana["t_hi"], 0.05 * t)
    d = refine_on_cad(gmsh, origin, -N, d, hit, grid.fid)
    d[(d < 0.5 * ana["t_lo"]) | (d > 3.0 * ana["t_hi"])] = np.inf
    # noeuds sans mesure (rayon rasant, coin) : moyenne des voisins mesurés
    for _ in range(20):
        miss = np.nonzero(~np.isfinite(d))[0]
        if len(miss) == 0:
            break
        new = {}
        for i in miss:
            vals = [d[j] for j in adj[i] if np.isfinite(d[j])]
            if vals:
                new[i] = float(np.mean(vals))
        if not new:
            break
        for i, v in new.items():
            d[i] = v
    n_miss = int((~np.isfinite(d)).sum())
    d[~np.isfinite(d)] = t
    off = -N * d[:, None]
    nodes = np.vstack([xyz + off * (k / layers) for k in range(layers + 1)])
    hexas = []
    for i in Q.tolist():
        p = xyz[i]
        nq = np.cross(p[1] - p[0], p[3] - p[0])
        if np.dot(nq, off[i[0]]) < 0:       # face 1-2-3-4 doit "regarder" vers 5-6-7-8
            i = i[::-1]
        for k in range(layers):
            hexas.append([j + k * nn for j in i] + [j + (k + 1) * nn for j in i])
    return nodes, np.array(hexas), dict(d_min=float(d.min()), d_max=float(d.max()), n_miss=n_miss,
                                        n_drop=n_drop)


def scaled_jacobians(nodes, hexas):
    """Contrôle rapide : jacobien aux 8 coins, normalisé (1 = cube parfait),
    minimum par élément."""
    nb = [(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)]
    worst = np.full(len(hexas), np.inf)
    for c, (a, b, d) in enumerate(nb):
        p0 = nodes[hexas[:, c]]
        e1, e2, e3 = (nodes[hexas[:, a]] - p0, nodes[hexas[:, b]] - p0, nodes[hexas[:, d]] - p0)
        det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
        det /= (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1))
        worst = np.minimum(worst, det)
    return worst


def hexa_volumes(nodes, hexas):
    """Volume de chaque hexa (6 faces gauches découpées en triangles autour de
    leur centre, tétraèdres vers le centre de l'élément)."""
    P = nodes[hexas]
    c = P.mean(1)
    vol = np.zeros(len(hexas))
    for f in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
        q = P[:, list(f)]
        m = q.mean(1)
        for a in range(4):
            vol += np.einsum("ij,ij->i", np.cross(q[:, a] - c, q[:, (a + 1) % 4] - c), m - c) / 6.0
    return vol


# -------------------------------------------------------------------- export
def write_inp(path, parts):
    with open(path, "w") as fh:
        fh.write("*HEADING\n genere par step2sc8r.py\n")
        n0 = 0
        fh.write("*NODE\n")
        for p in parts:
            p["n0"] = n0
            for i, x in enumerate(p["nodes"], start=n0 + 1):
                fh.write(f"{i}, {x[0]:.8g}, {x[1]:.8g}, {x[2]:.8g}\n")
            n0 += len(p["nodes"])
        e0 = 0
        for p in parts:
            fh.write(f"** epaisseur detectee = {p['t']:.6g} (min {p['t_min']:.6g}, max {p['t_max']:.6g})\n")
            fh.write(f"*ELEMENT, TYPE=SC8R, ELSET={p['name']}\n")
            for i, h in enumerate(p["hexas"] + p["n0"] + 1, start=e0 + 1):
                fh.write(f"{i}, " + ", ".join(map(str, h)) + "\n")
            e0 += len(p["hexas"])


# ------------------------------------------------------------------ pilotage
# cascade de rattrapage : (nom, couronnes, transfinite, algo gmsh, facteur de taille)
# algo 11 = quasi-structuré (meilleur sur les pièces réelles), 8 = frontal-Delaunay
# pour quads, 6 = frontal-Delaunay. Les couronnes/plis/faces à 4 côtés sont en
# transfinite dans tous les cas sauf D.
STRATEGIES = [
    ("A-quasi-structure",  True,  True,  11, 1.0),
    ("B-frontal-quads",    True,  True,  8,  1.0),
    ("C-quasi-fin",        True,  True,  11, 1 / 1.5),
    ("E-frontal-fin",      True,  True,  8,  1 / 1.5),
    ("D-libre-fin",        False, False, 6,  1 / 1.5),
]


def mesh_part(gmsh, vol, size, layers, jac_ok, use_rings, tvar, fill, skip=(), vol_tol=0.02):
    """Maille un solide en essayant les stratégies dans l'ordre ; s'arrête dès
    que jac_min >= jac_ok, sinon garde le meilleur essai."""
    ana = analyse(gmsh, [vol], tvar)
    t = ana["t"]
    skin0 = ana["sides"][choose_side(ana)]
    h = size if size > 0 else 3.0 * t
    r_max = fill * h
    ringed, ana_r, skin_r, holes, best, errors, done, tries = None, None, None, [], None, [], set(), []
    filled = {}                                              # peau -> (peau sans petits trous, nb)
    for name, rings, transfinite, algo, k in STRATEGIES:
        if name in skip:
            tries.append(f"{name[0]}:plante")
            continue
        state(name)
        if os.environ.get("STEP2SC8R_CRASH") == name:     # crochet de test : plantage natif simulé
            os.abort()
        try:
            if rings and use_rings and ringed is None:
                holes = find_holes(gmsh, vol, skin0, ana, h, r_max)
                if holes:
                    ringed = partition_rings(gmsh, vol, holes, ana)
                    ana_r = analyse(gmsh, ringed, tvar, t_ref=t)
                    skin_r = ana_r["sides"][match_side(ana, skin0, ana_r)]
                else:
                    ringed = False
            with_rings = bool(rings and use_rings and ringed)
            key = (with_rings, transfinite, algo, k)
            if key in done:                                  # pas de trou -> A == C
                continue
            done.add(key)
            skin, a = (skin_r, ana_r) if with_rings else (skin0, ana)
            if r_max > 0:
                if id(skin) not in filled:
                    filled[id(skin)] = fill_small_holes(gmsh, skin, r_max)
                skin, n_fill = filled[id(skin)]
            else:
                n_fill = 0
            t0 = time.time()
            mesh_skin(gmsh, skin, h * k, transfinite, algo)
            moved = project_skin_nodes(gmsh, skin)
            log(f"  {name} : maillage peau {time.time() - t0:.1f}s, reprojection CAO max {moved:.3g}")
            t0 = time.time()
            nodes, hexas, ext = extrude_to_hexa(gmsh, skin, a, layers, h * k)
            sj = scaled_jacobians(nodes, hexas)
            k = int(sj.argmin()); jac = float(sj[k])
            dv = float(hexa_volumes(nodes, hexas).sum() / ana["volume"] - 1.0)
            log(f"  {name} : extrusion {time.time() - t0:.1f}s, {len(hexas)} hexas, jac_min={jac:.2f}, "
                f"volume {dv:+.2%}")
            if abs(dv) > vol_tol:                # contrôle indépendant : jamais de maillage faux silencieux
                tries.append(f"{name[0]}:vol{dv:+.1%}")
                raise ValueError(f"volume maille {dv:+.1%} vs CAO (tolerance {vol_tol:.0%})")
            tries.append(f"{name[0]}:{jac:.2f}")
            if best is None or jac > best["jac"]:
                best = dict(t=t, t_min=ana["t_min"], t_max=ana["t_max"], nodes=nodes, hexas=hexas,
                            jac=jac, where=nodes[hexas[k]].mean(0), n_bad=int((sj < jac_ok).sum()),
                            strategy=name, rings=len(holes) if with_rings else 0, n_miss=ext["n_miss"], dv=dv,
                            n_drop=ext["n_drop"],
                            n_fill=n_fill)
            if jac >= jac_ok:
                break
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            log(f"  {name} : ECHEC {exc}")
    if best is None:
        raise ValueError(" / ".join(errors[:2]) + (f" (essais={','.join(tries)})" if tries else ""))
    best["tries"] = ",".join(tries)
    return best


def import_step(gmsh, step, clean):
    """Importe le STEP ; si clean > 0, nettoyage OCC (arêtes/faces < clean x
    épaisseur estimée) sur TOUT le modèle (healShapes renumérote tout), gardé
    seulement si chaque volume est conservé (< 0.1 %), sinon ré-import brut."""
    occ = gmsh.model.occ

    def load():
        gmsh.clear(); occ.importShapes(str(step)); occ.synchronize()
        vols = [v for _, v in gmsh.model.getEntities(3)]
        if not vols:                                        # STEP surfacique/abîmé : réparation
            occ.healShapes(sewFaces=True, makeSolids=True); occ.synchronize()
            vols = [v for _, v in gmsh.model.getEntities(3)]
        return vols

    vols = load()
    if clean <= 0 or not vols:
        return vols, ""
    v0 = sorted(occ.getMass(3, v) for v in vols)
    nf0 = len(gmsh.model.getEntities(2))
    t_est = min(2.0 * occ.getMass(3, v) / sum(occ.getMass(2, f) for _, f in gmsh.model.getBoundary(
        [(3, v)], combined=False, oriented=False)) for v in vols)
    try:
        occ.healShapes(tolerance=clean * t_est, fixDegenerated=True, fixSmallEdges=True,
                       fixSmallFaces=True, sewFaces=False, makeSolids=False)
        occ.synchronize()
        new = [v for _, v in gmsh.model.getEntities(3)]
        v1 = sorted(occ.getMass(3, v) for v in new)
        if len(v1) == len(v0) and all(abs(a - b) < 1e-3 * a for a, b in zip(v0, v1)):
            return new, f" nettoyage={nf0}->{len(gmsh.model.getEntities(2))} faces"
        note = " nettoyage refuse (volume modifie)"
    except Exception as exc:
        note = f" nettoyage impossible ({str(exc).strip()[:40]})"
    return load(), note


def process_file(args):
    """Maille un fichier STEP (dans un process dédié). Renvoie les lignes du rapport."""
    global VERBOSE
    step, out_dir, size, layers, jac_ok, jac_fail, use_rings, tvar, clean, fill, vol_tol, VERBOSE, skip = args
    import gmsh
    rows, parts = [], []
    log(f"{step.name} : début" + (f" (stratégies exclues : {','.join(sorted(skip))})" if skip else ""))
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        state("import")
        vols, note = import_step(gmsh, step, clean)
        log(f"{step.name} : {len(vols)} solide(s){note}")
        if skip:
            note += f" exclues_apres_plantage={','.join(sorted(skip))}"
        if not vols:
            rows.append((step.name, "-", "ECHEC", "aucun solide dans le STEP"))
        for k, vol in enumerate(vols, start=1):
            name = f"{step.stem}_S{k}".upper().replace("-", "_").replace(" ", "_")
            t0 = time.time()
            state(f"analyse {name}")
            try:
                p = mesh_part(gmsh, vol, size, layers, jac_ok, use_rings, tvar, fill, skip, vol_tol)
                # sous jac_fail : hexas quasi dégénérés, refusés par le solveur -> pas d'INP
                status = "OK" if p["jac"] >= jac_ok else ("MEDIOCRE" if p["jac"] >= jac_fail else "ECHEC")
                if status != "ECHEC":
                    parts.append(dict(name=name, **p))
                w = p["where"]
                rows.append((step.name, name, status,
                            f"t={p['t']:.4g} [{p['t_min']:.3g}..{p['t_max']:.3g}] elems={len(p['hexas'])} "
                            f"volume={p['dv']:+.2%} jac_min={p['jac']:.2f} @({w[0]:.0f},{w[1]:.0f},{w[2]:.0f}) "
                            f"elems_sous_{jac_ok:g}={p['n_bad']} trous_bouches={p['n_fill']} coins_supprimes={p['n_drop']} "
                            f"strategie={p['strategy']} couronnes={p['rings']} essais={p['tries']} "
                            f"noeuds_sans_mesure={p['n_miss']} duree={time.time() - t0:.0f}s{note}"))
            except Exception as exc:                        # une pièce KO ne bloque pas le lot
                rows.append((step.name, name, "ECHEC", f"{exc} duree={time.time() - t0:.0f}s{note}"))
            log(f"{step.name} : {rows[-1][1]} {rows[-1][2]} {rows[-1][3]}")
        state("ecriture")
        if parts:
            write_inp(Path(out_dir) / (step.stem + ".inp"), parts)
    except Exception:
        rows.append((step.name, "-", "ECHEC", traceback.format_exc(limit=1).strip()))
    finally:
        gmsh.finalize()
    return rows


def _worker(args, conn):
    """Point d'entrée du process fils : renvoie les lignes du rapport par le tube."""
    global _STATE
    _STATE = conn
    try:
        rows = process_file(args)
    except BaseException:
        rows = [(args[0].name, "-", "ECHEC", traceback.format_exc(limit=1).strip())]
    conn.send(("fin", rows))
    conn.close()


def run_batch(jobs, n_jobs, timeout):
    """Ordonnanceur : 1 process par fichier, au plus n_jobs en parallèle.
    Un plantage natif de gmsh (segfault) ne bloque pas le lot : le fichier est
    relancé sans la stratégie qui a planté ; un timeout donne un ECHEC."""
    ctx = multiprocessing.get_context("spawn")     # pas de fork après init des threads numpy/gmsh
    todo = [(j, frozenset()) for j in jobs]
    running, results = [], {}
    while todo or running:
        while todo and len(running) < n_jobs:
            job, skip = todo.pop(0)
            parent, child = ctx.Pipe(duplex=False)
            pr = ctx.Process(target=_worker, args=(job + (skip,), child), daemon=True)
            pr.start(); child.close()
            running.append(dict(job=job, skip=skip, proc=pr, conn=parent, t0=time.time(), etat="demarrage"))
        time.sleep(0.2)
        for r in list(running):
            done = None
            try:
                while r["conn"].poll():
                    kind, val = r["conn"].recv()
                    if kind == "etat":
                        r["etat"] = val
                    else:
                        done = val
            except (EOFError, OSError):
                pass
            name = r["job"][0].name
            if done is not None:
                results[r["job"][0]] = done
            elif not r["proc"].is_alive():
                code = r["proc"].exitcode
                strat = [s[0] for s in STRATEGIES]
                if r["etat"] in strat and len(r["skip"]) + 1 < len(strat):
                    log(f"{name} : PLANTAGE gmsh (code {code}) dans {r['etat']} -> relance sans elle")
                    todo.append((r["job"], r["skip"] | {r["etat"]}))
                else:
                    results[r["job"][0]] = [(name, "-", "ECHEC",
                                             f"plantage gmsh (code {code}) pendant : {r['etat']}")]
            elif time.time() - r["t0"] > timeout:
                r["proc"].kill()
                results[r["job"][0]] = [(name, "-", "ECHEC",
                                         f"timeout {timeout:.0f}s depasse pendant : {r['etat']}")]
            else:
                continue
            r["proc"].join(5); r["conn"].close(); running.remove(r)
    return [results[j[0]] for j in jobs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step_dir"); ap.add_argument("inp_dir")
    ap.add_argument("--size", type=float, default=-1, help="taille d'élément (défaut : 3 x épaisseur)")
    ap.add_argument("--layers", type=int, default=1, help="nb d'éléments dans l'épaisseur")
    ap.add_argument("--jobs", type=int, default=4, help="process en parallèle (1 par fichier)")
    ap.add_argument("--jac-min", type=float, default=0.3, help="jacobien normalisé mini accepté sans rattrapage")
    ap.add_argument("--jac-fail", type=float, default=0.05,
                    help="sous ce jacobien normalisé : ECHEC, pas d'INP (entre les deux : MEDIOCRE)")
    ap.add_argument("--no-rings", action="store_true", help="désactive les couronnes autour des trous")
    ap.add_argument("--t-var", type=float, default=1.0,
                    help="variation d'épaisseur locale admise : peau si t/(1+v) <= t_loc <= t*(1+v)")
    ap.add_argument("--clean", type=float, default=0,
                    help="nettoyage OCC des arêtes/faces < clean x épaisseur (ex. 0.02 ; 0 = désactivé, "
                         "gardé seulement si le volume est conservé)")
    ap.add_argument("--fill-holes", type=float, default=0.5,
                    help="bouche les trous de rayon < fill x taille d'élément (0 = désactivé)")
    ap.add_argument("--timeout", type=float, default=1800, help="durée max par fichier STEP (s)")
    ap.add_argument("--vol-tol", type=float, default=0.02,
                    help="écart max volume maillé / volume CAO (au-delà : essai rejeté)")
    ap.add_argument("--verbose", action="store_true", help="chronométrage des étapes sur stderr")
    a = ap.parse_args()
    global VERBOSE
    VERBOSE = a.verbose
    out = Path(a.inp_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.step_dir).iterdir() if p.suffix.lower() in (".step", ".stp"))
    jobs = [(f, out, a.size, a.layers, a.jac_min, a.jac_fail, not a.no_rings, a.t_var, a.clean, a.fill_holes,
             a.vol_tol, a.verbose) for f in files]
    results = run_batch(jobs, a.jobs, a.timeout)
    rows = [r for res in results for r in res]
    with open(out / "rapport.csv", "w") as fh:
        fh.write("fichier;piece;statut;detail\n")
        for r in rows:
            fh.write(";".join(r) + "\n"); print(*r, sep=" | ")
    ko = sum(r[2] == "ECHEC" for r in rows)
    print(f"\n{len(rows) - ko}/{len(rows)} pièces maillées -> {out}")
    return 1 if ko else 0


if __name__ == "__main__":
    sys.exit(main())
