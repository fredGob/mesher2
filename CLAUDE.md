# CLAUDE.md — step2sc8r

Mailleur batch local : **dossier de STEP (pièces minces : tôles pliées, peaux et cadres usinés à épaisseur localement variable) → dossier d'INP Abaqus en hexa SC8R**.

## Objectif et périmètre

- Entrée : un dossier de `.step/.stp`, chaque fichier contenant un ou plusieurs solides minces (épaisseur constante ou localement variable : usinage chimique, poches). Les jonctions en T (nervures) sont rejetées.
- Sortie : un `.inp` par STEP (un `ELSET` par solide, `*NODE` + `*ELEMENT, TYPE=SC8R`, épaisseur min/max en commentaire) + `rapport.csv`.
- **Hors périmètre** : liaisons entre tôles (fixations, contact, tie), matériaux, sections, chargements → faits dans la mise en données, ailleurs. Ne pas les ajouter ici.
- Pas de LLM ni de modèle local dans la boucle : tout est géométrique et déterministe.

## Contraintes d'environnement

- 100 % local / hors ligne. Poste verrouillé, proxy d'entreprise : **dépendances = `gmsh` + `numpy` uniquement** pour l'outil (`matplotlib` toléré pour `tests/plot_inp.py` et `tests/plot_jac.py`). Ne pas ajouter de dépendance sans demander.
- Un seul fichier `step2sc8r.py`, copiable tel quel sur un poste. Pas de packaging.
- Python ≥ 3.9. Commentaires et messages en français.

## Commandes

```bash
source myenv/bin/activate            # venv local : gmsh 4.15.2, numpy, matplotlib (outils de test)
python step2sc8r.py <dossier_step> <dossier_inp> [--size H] [--layers N] [--jobs J] [--jac-min 0.3] [--jac-fail 0.05]
       [--no-rings] [--t-var 1.0] [--fill-holes 0.5] [--clean 0] [--timeout 1800] [--vol-tol 0.02] [--verbose]
python tests/run_tests.py            # génère les STEP synthétiques, maille, contrôle. Code retour 0 = OK
python tests/plot_inp.py x.inp x.png [elev azim]   # contrôle visuel 3D
python tests/plot_jac.py x.inp prefixe [zoom]      # carte du jacobien + zoom sur le pire élément (pièces réelles)
python tests/check_volumes.py stps_test results   # contrôle indépendant volume maillage / CAO sur un lot réel
python step2sc8r.py stps_test results --jobs 5 --verbose   # banc sur les pièces réelles (jamais commitées)
```

`--size` par défaut = 3 × épaisseur de référence (par pièce). `--verbose` chronomètre chaque étape sur stderr.

## Pipeline (par solide)

1. `import_step` : import OCC ; `--clean` optionnel (healShapes sur tout le modèle, gardé seulement si chaque volume est conservé à 0,1 %). Désactivé par défaut : échoue sur les STEP réels.
2. `analyse` : triangulation de la surface (`triangulate`, contrôlée par divergence : volume triangulé = volume CAO à 3 %), grille de lancer de rayons (`TriGrid`), puis `probe_faces` : pour chaque face, distance de sortie de la matière le long de la normale intérieure (points projetés sur la CAO, distance affinée par `refine_on_cad`) et face en vis-à-vis. Épaisseur de référence `t` = médiane pondérée par l'aire. Peau = faces avec vis-à-vis parallèle (cos > 0,8) et distance dans `[t/(1+v), t(1+v)]` (`--t-var v`).
3. `two_color` : les faces de peau sont séparées en 2 côtés (adjacence = même côté, vis-à-vis = côté opposé ; une composante isolée, ex. fond de poche, prend la couleur opposée à son vis-à-vis). Conflit > 2 % de l'aire → rejet (nervure, T, massif). Peau < 60 % de la surface, ou côtés d'aires trop différentes → rejet.
4. `choose_side` : on maille le côté le moins découpé. `find_holes` + `partition_rings` : couronnes à 4 quadrants autour des trous circulaires des faces planes (copie du solide, fragment local) ; `match_side` retrouve le côté après partition.
5. `fill_small_holes` : trous de rayon < `--fill-holes` × h bouchés (face recréée sur la même surface CAO : `addPlaneSurface` pour un plan, `addTrimmedSurface(wire3D=True)` sinon ; le solide n'est pas modifié).
6. `mesh_skin` : quads 100 %, taille mesurée (`median_edge`) et corrigée une fois. Purge des champs/vues avant chaque essai (voir pièges). Puis `project_skin_nodes` : chaque nœud est reprojeté sur sa face ou son arête CAO (indispensable avec l'algo 11).
7. `extrude_to_hexa` : chaque nœud est projeté sur l'autre peau par un rayon le long de la normale CAO sortante moyennée (`face_orientation` vérifie le sens sur les faces recréées) → épaisseur locale exacte, onglet automatique. Nœuds de bord : origine du rayon rentrée de 0,3 t dans la face. Nœud sans impact → moyenne des voisins (compté dans `noeuds_sans_mesure`).
8. Statut : OK si jac_min ≥ `--jac-min`, MEDIOCRE entre `--jac-fail` et `--jac-min` (INP écrit, à vérifier), ECHEC en dessous (pas d'INP). Contrôle de volume dans l'outil (`hexa_volumes`) : un essai dont le volume s'écarte de la CAO de plus de `--vol-tol` (2 %) est rejeté. Puis `scaled_jacobians` et cascade `STRATEGIES` (A quasi-structuré → B frontal quads → C quasi fin → E frontal fin → D libre fin) : arrêt dès `jac_min ≥ --jac-min`, sinon meilleur essai. Le rapport donne `essais=`, la position du pire élément `@(x,y,z)` et `elems_sous_0.3`.
9. `write_inp`.

Parallélisme : `run_batch` = 1 process `spawn` par fichier STEP, au plus `--jobs`. Chaque fils signale son étape par un tube ; s'il meurt (segfault gmsh), le fichier est relancé **sans la stratégie qui a planté** (`exclues_apres_plantage=` dans le rapport) ; au-delà de `--timeout` → ECHEC. Crochet de test : `STEP2SC8R_CRASH=<nom de stratégie>` provoque un `os.abort()`.

## Invariants — à ne jamais casser

- **Jamais de maillage faux silencieux** : en cas de doute (épaisseur hors tolérance `--t-var`, nervure ou T, massif, côté matière ambigu) → `ECHEC` avec un message clair dans `rapport.csv`, pas d'INP pour cette pièce.
- Une pièce en échec ne bloque ni le fichier ni le lot.
- SC8R : nœuds 1-4 sur la peau de départ, 5-8 en face, face 1-2-3-4 orientée vers 5-8 → direction d'empilement = épaisseur, jacobien > 0.
- 100 % hexa (une peau contenant un non-quad lève une erreur).
- Volume du maillage ≈ volume CAO (< 1 %) : c'est le contrôle indépendant de `run_tests.py`.
- Un plantage natif de gmsh ne doit jamais figer le lot (test `REPRISE_CRASH`).
- Pas de `fragment` global sur un assemblage (coût) : uniquement la partition locale d'UNE pièce.

## Méthode de travail

- **Spec d'abord, puis cas synthétique, puis code.** Toute nouvelle feature géométrique (soyage, bord tombé, trou oblong, poche…) commence par un `case_xxx()` dans `tests/make_test_steps.py` et son attendu dans `run_tests.py` (`EXPECT_FAIL` si rejet voulu).
- Ne pas passer au composant suivant tant que `python tests/run_tests.py` n'affiche pas `TOUT PASSE`.
- Un bug trouvé sur une vraie pièce → le reproduire par un cas synthétique minimal (les STEP réels ne sortent pas de l'entreprise, ne pas en demander ni en committer).
- Modifs petites et ciblées ; garder les fonctions pures `(gmsh, tags, params) → résultat`.

## Pièges gmsh déjà rencontrés (4.15.2)

- `Geometry.OCCSewFaces=1` à l'import **détruit les solides** (même avec `OCCMakeSolids`). Utiliser `occ.healShapes(sewFaces=True, makeSolids=True)` seulement si aucun volume n'est lu.
- `Mesh.RecombinationAlgorithm=3` (full-quad) plante : « 1D mesh cannot be divided by 2 ». Rester sur 1 (blossom) + `SubdivisionAlgorithm=1`.
- Le champ `BoundaryLayer` ne marche pas sur un modèle 3D (« curve adjacent to 2 surfaces ») → d'où la partition OCC pour les couronnes.
- `getBoundary` sur une courbe fermée ou plusieurs volumes : toujours `combined=False`, sinon les entités partagées s'annulent.
- Couture d'un cylindre complet : compte 2 fois dans le périmètre, 1 fois dans l'adjacence.
- `mesh.clear()` ne retire pas les contraintes transfinite → `mesh.removeConstraints()` entre deux essais.
- Algo 11 (quasi-structuré) : le meilleur sur les pièces réelles, mais peut sortir des éléments retournés sur petits trous → toujours derrière le contrôle du jacobien. Il fait sa propre subdivision : consigne 2h pour obtenir ~h (et `median_edge` corrige).
- Algo 11 **laisse ses nœuds hors de la surface CAO** (il maille une triangulation de fond) : jusqu'à 4,4 mm sur la peau BSpline de part_201 → épaisseur mesurée trop faible, −12 % de volume, statut OK à tort. D'où `project_skin_nodes` et le contrôle de volume dans l'outil. Cas synthétique : `curved_panel` (tolérance 0,2 %).
- Algo 11 **met en cache sa carte de taille** (champ + vue `guiding_field`) : sans purge, tous les maillages suivants reprennent la taille du premier.
- Purge dans cet ordre exact : `field.remove` de tous les champs, `field.setAsBackgroundMesh(0)`, puis `view.remove`. Retirer le champ sans désactiver le fond → **segfault** au `generate` suivant ; retirer la vue d'abord → « View with index 0 does not exist ».
- `getNormal` sur une face d'un solide OCC = normale **sortante**, quel que soit le signe de `getBoundary(oriented=True)` (ne pas s'en servir pour orienter).
- `healShapes` renumérote TOUT le modèle, même appliqué à une copie ; « Could not fix wire » sur les STEP réels.
- `addTrimmedSurface` exige `wire3D=True` avec des boucles 3D ; échoue sur les plans réels → `addPlaneSurface`.
- `multiprocessing` en `fork` après import de numpy/gmsh : workers bloqués (futex) → `spawn` + `OMP_NUM_THREADS=1`. Un worker d'un `Pool` qui segfault fige le `Pool` pour toujours → d'où `run_batch`.
- Compound 2D (`setCompound(2, peau)`) : échoue sur les pièces réelles (algo 11 : « GlobalBackgroundMesh: failed to import mesh », algos 6/8 : nœuds non reclassés). Compound 1D sur des arêtes tangentes : n'enlève pas le nœud d'un sommet partagé avec une tranche. Remaillage via `classifySurfaces`/`createGeometry` : quads retournés dans les plis. Pistes testées et abandonnées (18/09/2026).
- Lancer de rayons depuis une face courbe : le rayon peut retoucher sa propre face triangulée (corde) → ne compter que les traversées **sortantes** (d · n > 0).
- Le 1er rayon d'une couronne doit passer par le sommet CAO du cercle du trou, sinon quadrant à 5 côtés.

## Limites connues / pistes

- Rejetés : pièces avec nervures / jonctions en T (ex. panneau à poches raidies), massifs, épaisseur hors `[t/(1+v), t(1+v)]`.
- Le côté maillé est projeté sur l'autre : sur une poche d'usinage, la marche est lissée dans l'épaisseur (hexas d'épaisseur variable), pas représentée par une arête.
- Trous sur faces non planes et trous oblongs : pas de couronne (maillage libre). Petits trous bouchés (`--fill-holes`), signalés dans le rapport.
- Congés, micro-faces et marches de 0,1 mm des STEP réels : pas de defeaturing géométrique, ils contraignent le maillage (jac_min ≈ 0,1 sur part_000 et part_145).
- Temps : ~1 à 2 min par pièce de 4 m (triangulation 10 s, maillage peau 60–80 s par stratégie).
- Pas de gradation de taille, pas de cache pour les pièces répétées.

## Journal des pièces réelles (`stps_test/`)

| Pièce | Nature | 17/09 (v1) | 18/09 matin | 18/09 midi (nettoyage peau) |
|---|---|---|---|---|
| part_000 | profil plié t=4,34 | MEDIOCRE jac 0,12 | MEDIOCRE jac 0,12 | **OK jac 0,36**, volume −0,23 %, 5 coins supprimés |
| part_093 | panneau à poches + nervures | ECHEC 25 peaux | ECHEC voulu (T) | ECHEC voulu (T) |
| part_145 | cadre courbe usiné t 3,4→10,4 | ECHEC | MEDIOCRE jac 0,07 | MEDIOCRE jac 0,26 (1 élément < 0,3), volume −1,94 % |
| part_201 | peau fuselage 4 m, t 3,0→6,9 | ECHEC | OK jac 0,34 | (non remesurée après nettoyage) |
| part_349 | raidisseur 3,8 m t 1,14→2,01 | ECHEC | ECHEC jac 0,00, vol +2 % | (non remesurée) — cause du +2 % trouvée, voir ci-dessous |

Mesures « midi » faites avec `tests/diag/one2.py <step> ABCDE <sortie.inp>` (toutes les stratégies, une pièce).

### Où j'en suis (18/09/2026, arrêt pour changement de poste)

Fait :
- Nettoyage de la peau maillée avant extrusion (`collect_skin` → `drop_flat_corner_quads` → `smooth_quads` dans `extrude_to_hexa`) : supprime les quads de contour dégénérés par les micro-arêtes CAO (coin plat > 150° ou arête de bord < 0,15 h), contour gardé simple, puis lissage projeté sur la CAO. Compté dans le rapport (`coins_supprimes=`). `run_tests.py` : TOUT PASSE.
- Diagnostic : 100 % des mauvais hexas de 000/145 venaient du quad de peau (pas de l'extrusion) ; classés par `tests/diag/badinp.py`.
- Essayé et abandonné : `occ.defeature` des petites faces (ne retire que 3-4 faces sur 11-23, jac pire : 0,01-0,03).

**Prochaine étape (en cours) — part_349, +2 % de volume** : `tests/diag/topin.py` (isInside sur la CAO) montre qu'environ 2 % des nœuds de peau ont une colonne d'hexa **entièrement hors matière** (épaisseur ~1,14, emplacements ex. (9931,-1108,2949), (8930,-952,2921), (6897,-981,2557)). Donc normale inversée ou face de l'autre peau intégrée au côté maillé. Lancer `tests/diag/flip.py stps_test/part_349_V5313914720200.stp` pour identifier la face CAO et sa classification, puis corriger (probablement `face_orientation` ou `two_color`). Faire un cas synthétique qui reproduit avant de corriger. Ajouter aussi un contrôle dans l'outil : colonne d'hexa hors matière → rejet.

Ensuite : relancer le lot complet (`python step2sc8r.py stps_test results --jobs 5 --verbose`), `tests/check_volumes.py stps_test results`, remettre ce tableau à jour.

Pistes restantes :
1. part_145 −1,9 % de volume : aire maillée exacte, déficit géométrique non localisé (coins de la section ?). Même méthode `topin.py`.
2. Le dernier mauvais quad de 145 (losange jac 0,26) : sommets CAO fixes voisins d'un coin supprimé.

Scripts de diagnostic (`tests/diag/`) : `one2.py` (une pièce, stratégies choisies, écrit l'INP), `badinp.py` (pires quads : angles, côtés, bord), `topin.py` (colonnes d'hexa dans/hors matière), `thk.py` (aire et épaisseur moyenne vs CAO), `miss.py` (faces hors peau et pourquoi), `cause.py` (défaut 2D vs extrusion), `flip.py` (face CAO sous des points donnés).
