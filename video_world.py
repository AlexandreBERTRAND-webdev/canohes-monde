"""Canohès façon jeu vidéo : construit le monde 3D en couleurs (photo aérienne IGN drapée sur le relief,
bâtiments à leur vraie hauteur, arbres, lampadaires) et la trajectoire de la voiture et de la caméra.
Sortie : out/video/world/ (world.bin, world.json, ortho_*.jpg)."""
import json, math, os, sys, time, urllib.parse
import numpy as np
import shapely, shapely.affinity, shapely.ops
from shapely.geometry import shape, Polygon, LineString, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
import mapbox_earcut as earcut
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from rasterio import features
from rasterio.transform import from_origin
from skimage.feature import peak_local_max
import compas_cgal.straight_skeleton_2 as sk
import networkx as nx
import manifold3d as mf3
from manifold3d import Manifold, Mesh, CrossSection
from PIL import Image
import io

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("DONNEES") or os.path.join(HERE, "data")
GAME = os.environ.get("GAME") == "1"
BLOC = os.environ.get("BLOC") == "1"           # bloc de 2 km du monde ouvert : raccords exacts avec les voisins, photo légère          # monde complet pour le jeu : chaussée et trottoirs partout
OUT = os.environ.get("SORTIE") or os.path.join(HERE, "out", "game", "raw") if GAME else os.path.join(HERE, "out", "video", "world")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import fetch  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)
BB = json.load(open(os.path.join(D, "v_bbox.json")))["bbox"]          # carré de données (Canohès : 1 900 m de côté)
COTE = int(round(BB[2] - BB[0]))                                       # côté du carré de données, en m
XO, YO, ZO = BB[0], BB[1], 50.0
NT = int(os.environ.get("NT", "4"))                                   # NT x NT dalles (Canohès : 4 x 4 dalles de 476 m)
TS, PAD = (COTE / NT if BLOC else (COTE + 4) / NT), 24.0
WORLD = NT * TS
FPS = 30
rng = np.random.default_rng(11)


def loc(g):
    return shapely.transform(g, lambda c: c - np.array([XO, YO]))


def load(name):
    return json.load(open(os.path.join(D, name + ".geojson")))["features"]


# ==== relief ====
mnt = np.load(os.path.join(D, "v_mnt.npy")).astype(float)
MARGE = int(json.load(open(os.path.join(D, "v_bbox.json"))).get("marge", 0))     # marge de relief autour du bloc (m), pour un lissage identique des deux côtés d'un bord
mns = np.load(os.path.join(D, "v_mns.npy")).astype(float)
mnt[~np.isfinite(mnt) | (mnt < -50)] = 0.0                              # mer et trous du relevé (le bord de mer renvoie des valeurs aberrantes)
mns[~np.isfinite(mns) | (mns < -50)] = 0.0
if MARGE:
    mnt_s = ndi.gaussian_filter(mnt, 1.5)[MARGE:-MARGE, MARGE:-MARGE]
    mnt, mns = mnt[MARGE:-MARGE, MARGE:-MARGE], mns[2 * MARGE:-2 * MARGE, 2 * MARGE:-2 * MARGE]
if mnt.min() <= 51:                                                      # altitude de base : 50 m à Canohès, sinon 5 m sous le point le plus bas
    ZO = float(np.floor(mnt.min()) - 5)
mnh = mns - ndi.zoom(mnt, 2, order=1)
if not MARGE:
    mnt_s = ndi.gaussian_filter(mnt, 1.5)
GS = COTE / round(COTE / 7.0) if BLOC else 7.0                         # en bloc : des nœuds de relief exactement sur les bords
gn = int(round(WORLD / GS)) + 1 if BLOC else int(WORLD / GS) + 1
gxs = np.arange(gn) * GS
gX, gY = np.meshgrid(gxs, gxs)
Zg = ndi.map_coordinates(mnt_s, [np.clip(COTE - gY - 0.5, 0, COTE - 1), np.clip(gX - 0.5, 0, COTE - 1)], order=1, mode="nearest") - ZO
_gi = RegularGridInterpolator((gxs, gxs), Zg, bounds_error=False, fill_value=None)
zg = lambda x, y: float(_gi((y, x)))
zgv = lambda P: _gi(np.column_stack([P[:, 1], P[:, 0]]))
log("relief", Zg.min().round(1), Zg.max().round(1))

# ==== photo aérienne IGN ====
MOS = os.path.join(D, "v_ortho_mos.npy")
if BLOC and not os.path.exists(MOS):
    # une seule photo à 1 m par pixel : couleur des toits et des arbres, teinte du sol
    p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS="ORTHOIMAGERY.ORTHOPHOTOS", STYLES="", CRS="EPSG:2154",
             BBOX=f"{XO},{YO},{XO + COTE},{YO + COTE}", WIDTH=COTE, HEIGHT=COTE, FORMAT="image/jpeg")
    im = Image.open(io.BytesIO(fetch.get(fetch.WMSR + "?" + urllib.parse.urlencode(p)))).convert("RGB")
    im.resize((160, 160), Image.LANCZOS).save(os.path.join(OUT, "teinte.jpg"), quality=86)
    np.save(MOS, np.asarray(im.resize((int(WORLD / 0.5),) * 2, Image.BILINEAR)))
    log("photo aérienne légère téléchargée")
elif not all(os.path.exists(os.path.join(OUT, f"ortho_{i}_{j}.jpg")) for i in range(NT) for j in range(NT)) or not os.path.exists(MOS):
    mos = np.zeros((int(WORLD / 0.5), int(WORLD / 0.5), 3), np.uint8)
    for i in range(NT):
        for j in range(NT):
            x0, y0 = XO + i * TS - PAD, YO + j * TS - PAD
            ext = TS + 2 * PAD
            px = int(ext / 0.2)
            p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS="ORTHOIMAGERY.ORTHOPHOTOS", STYLES="", CRS="EPSG:2154",
                     BBOX=f"{x0},{y0},{x0 + ext},{y0 + ext}", WIDTH=px, HEIGHT=px, FORMAT="image/jpeg")
            raw = fetch.get(fetch.WMSR + "?" + urllib.parse.urlencode(p))
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            im.save(os.path.join(OUT, f"ortho_{i}_{j}.jpg"), quality=88)
            inner = im.crop((int(PAD / 0.2), int(PAD / 0.2), int((PAD + TS) / 0.2), int((PAD + TS) / 0.2))).resize((int(TS / 0.5),) * 2, Image.BILINEAR)
            a = np.asarray(inner)
            r0 = mos.shape[0] - (j + 1) * a.shape[0]
            mos[r0:r0 + a.shape[0], i * a.shape[1]:(i + 1) * a.shape[1]] = a
    np.save(MOS, mos)
    log("photos aériennes téléchargées")
mos = np.load(MOS)


def ortho_rgb(x, y, k=1):
    r, c = int(mos.shape[0] - y / 0.5), int(x / 0.5)
    r, c = min(max(r, k), mos.shape[0] - k - 1), min(max(c, k), mos.shape[1] - k - 1)
    return mos[r - k:r + k + 1, c - k:c + k + 1].reshape(-1, 3).mean(axis=0) / 255.0


# ==== réseau de rues ====
ABR = {"R": "Rue", "AV": "Avenue", "BD": "Boulevard", "CHE": "Chemin", "ALL": "Allée", "IMP": "Impasse", "PL": "Place", "RTE": "Route",
       "TRA": "Traverse", "PAS": "Passage", "LOT": "Lotissement", "SQ": "Square", "RPT": "Rond-point", "CRS": "Cours", "CAR": "Carrefour"}
DRIVE = ("Route à 1 chaussée", "Route à 2 chaussées", "Rond-point", "Bretelle", "Type autoroutier")
G = nx.DiGraph()
road_polys, lamps_src, mini, road_lines, cyc_lines, named = [], [], [], [], [], []
ponts_src, plat_src, axes_src = [], [], []                            # ... ; axes : (ligne, largeur, nature, sens) pour glissières et terre-plein                                          # tronçons en pont ; tronçons au sol (pour aplanir la chaussée)
wbox = box(0, 0, WORLD, WORLD)
for f in load("troncon_de_route"):
    p = f["properties"]
    g = loc(shape(f["geometry"]))
    if not g.intersects(wbox) or p.get("etat_de_l_objet") != "En service":
        continue
    w = float(p.get("largeur_de_chaussee") or 4.0)
    libre = p.get("acces_vehicule_leger") in ("Libre", "A péage")          # sinon : piste cyclable, voie piétonne, accès réservé
    for ln in (g.geoms if g.geom_type == "MultiLineString" else [g]):
        c = np.asarray(ln.coords)[:, :2]
        if p["nature"] in DRIVE and libre:
            road_polys.append(LineString(c).buffer(max(w, 3.0) / 2 + 0.3, cap_style=1))
            _np = str(p.get("position_par_rapport_au_sol") or "0")
            niveau = int(_np) if _np.lstrip("-").isdigit() else 0                 # « Gué ou radier » et autres valeurs : au sol
            if niveau > 0:
                ponts_src.append((c, w))
            elif niveau == 0:
                plat_src.append((c, w))
            if niveau == 0:
                road_lines.append((LineString(c), w))
                axes_src.append((LineString(c), w, p["nature"], p.get("sens_de_circulation") or "Double sens", bool(p.get("urbain"))))
            mini.append((np.round(c, 1).tolist(), 2 if w >= 5.5 else 1))
            if p.get("prive"):
                continue
            a, b = tuple(np.round(c[0], 0)), tuple(np.round(c[-1], 0))
            if a == b:
                continue
            name = p.get("nom_voie_ban_gauche") or p.get("nom_voie_ban_droite") or ""
            if not name:
                ab = (p.get("nom_collaboratif_gauche") or p.get("nom_collaboratif_droite") or "").split(" ", 1)
                if len(ab) == 2:
                    name = ABR.get(ab[0], ab[0]) + " " + ab[1]
            named.append((np.round(c, 1).tolist(), 2 if w >= 5.5 else 1, name))
            sens = p.get("sens_de_circulation") or "Double sens"
            two = sens == "Double sens"
            kw = dict(length=float(LineString(c).length), name=name, width=w, two=two)
            if sens in ("Double sens", "Sens direct"):
                G.add_edge(a, b, pts=c, **kw)
            if sens in ("Double sens", "Sens inverse"):
                G.add_edge(b, a, pts=c[::-1], **kw)
            if p.get("urbain") and w >= 4.0 and p["nature"] == "Route à 1 chaussée" and niveau == 0:
                lamps_src.append((c, w))
        elif p["nature"] in DRIVE and p.get("nature_de_la_restriction") == "Piste cyclable":
            cyc_lines.append(LineString(c)); mini.append((np.round(c, 1).tolist(), 0))
        elif p["nature"] in ("Route empierrée", "Chemin"):
            mini.append((np.round(c, 1).tolist(), 0))
roads = unary_union(road_polys)
# chaussée à plat : le relief sous chaque route prend la hauteur de son axe, lissée sur une vingtaine de mètres
# (sans cela, les nœuds de la grille de 7 m tombent dans les fossés et la route ondule comme un chemin de terre)
from scipy.spatial import cKDTree
_ax, _az, _aw = [], [], []
for c, w in plat_src:
    ln = LineString(c)
    if ln.length < 1:
        continue
    ss = np.linspace(0, ln.length, max(2, int(ln.length / 2) + 1))
    P = np.array([ln.interpolate(t).coords[0] for t in ss])
    zz = ndi.map_coordinates(mnt_s, [np.clip(COTE - P[:, 1] - 0.5, 0, COTE - 1), np.clip(P[:, 0] - 0.5, 0, COTE - 1)], order=1, mode="nearest") - ZO
    if len(zz) > 4:
        zz = ndi.gaussian_filter1d(zz, 4, mode="nearest")
    _ax.append(P); _az.append(zz); _aw.append(np.full(len(P), w))
if _ax:
    _ax, _az, _aw = np.vstack(_ax), np.concatenate(_az), np.concatenate(_aw)
    dd, ii = cKDTree(_ax).query(np.column_stack([gX.ravel(), gY.ravel()]), distance_upper_bound=20)
    ok = np.isfinite(dd)
    ok[ok] = dd[ok] <= _aw[ii[ok]] / 2 + 3.5
    Zf = Zg.ravel().copy(); Zf[ok] = _az[ii[ok]]
    Zg = Zf.reshape(Zg.shape)
    _gi = RegularGridInterpolator((gxs, gxs), Zg, bounds_error=False, fill_value=None)
    log("chaussée aplanie :", int(ok.sum()), "nœuds du relief recalés sur l'axe des routes")
log("rues:", G.number_of_edges(), "tronçons carrossables")
nodes = np.array(list(G.nodes))


def nearest(X, Y):
    d = np.hypot(nodes[:, 0] - (X - XO), nodes[:, 1] - (Y - YO))
    return tuple(nodes[int(np.argmin(d))])


# ==== itinéraire : lotissements, mairie, écoles et stade, centre ancien, église, retour ====
WAY = json.load(open(os.path.join(HERE, "video_route.json")))["points"]
_dans = lambda X, Y: XO + 20 < X < XO + COTE - 20 and YO + 20 < Y < YO + COTE - 20
if not all(_dans(*w_) for w_ in WAY):
    # carré sans le trajet vidéo de Canohès : trajet de secours, du centre vers le point le plus éloigné du plus grand réseau connexe, puis retour
    _cc = np.array(list(max(nx.strongly_connected_components(G), key=len))) if G.number_of_nodes() > 1 else np.zeros((0, 2))
if not all(_dans(*w_) for w_ in WAY) and len(_cc) < 2:                    # bloc sans route (mer, montagne) : trajet fictif au centre
    G.add_edge((COTE / 2 - 40, COTE / 2), (COTE / 2 + 40, COTE / 2), pts=np.array([[COTE / 2 - 40, COTE / 2], [COTE / 2 + 40, COTE / 2]]), length=80.0, name="", width=4.0, two=True)
    G.add_edge((COTE / 2 + 40, COTE / 2), (COTE / 2 - 40, COTE / 2), pts=np.array([[COTE / 2 + 40, COTE / 2], [COTE / 2 - 40, COTE / 2]]), length=80.0, name="", width=4.0, two=True)
    nodes = np.array(list(G.nodes))
    _cc = nodes
if not all(_dans(*w_) for w_ in WAY):
    _c0 = _cc[int(np.argmin(np.hypot(_cc[:, 0] - COTE / 2, _cc[:, 1] - COTE / 2)))]
    _c1 = _cc[int(np.argmax(np.hypot(_cc[:, 0] - _c0[0], _cc[:, 1] - _c0[1])))]
    WAY = [[_c0[0] + XO, _c0[1] + YO], [_c1[0] + XO, _c1[1] + YO], [_c0[0] + XO, _c0[1] + YO]]
used = {}


def wfun(u, v, d):
    k = frozenset((u, v))
    return d["length"] * (1 + 4.0 * used.get(k, 0)) * (1.6 if d["width"] < 3.5 else 1.0)


route_pts, route_names, route_off = [], [], []
cur = nearest(*WAY[0])
for W in WAY[1:]:
    tgt = nearest(*W)
    if tgt == cur:
        continue
    try:
        path = nx.shortest_path(G, cur, tgt, weight=wfun)      # respecte les sens uniques
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        continue                                               # réseau en morceaux (bloc de bord) : on saute cette étape
    for u, v in zip(path[:-1], path[1:]):
        d = G.edges[u, v]
        used[frozenset((u, v))] = used.get(frozenset((u, v)), 0) + 1
        off = min(d["width"] / 4, 1.5) if (d["two"] and d["width"] >= 4.5) else 0.0     # on roule à droite
        for q in d["pts"][(1 if route_pts else 0):]:
            route_pts.append(q); route_names.append(d["name"]); route_off.append(off)
    cur = tgt
if len(route_pts) < 2:                                         # aucun trajet possible : trajet fictif au centre du bloc
    route_pts, route_names, route_off = [(COTE / 2 - 40, COTE / 2), (COTE / 2 + 40, COTE / 2)], ["", ""], [0.0, 0.0]
route = LineString(route_pts)
log(f"itinéraire: {route.length:.0f} m")
# rééchantillonnage tous les 0,5 m puis lissage des virages
STEP = 0.5
ns = int(route.length / STEP)
R = np.array([route.interpolate(k * STEP).coords[0] for k in range(ns + 1)])
cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(np.array(route_pts), axis=0).T))])
nm_idx = np.searchsorted(cum, np.arange(ns + 1) * STEP, side="right") - 1
names_s = [route_names[min(max(i, 0), len(route_names) - 1)] for i in nm_idx]
off_s = ndi.gaussian_filter1d(np.array([route_off[min(max(i, 0), len(route_off) - 1)] for i in nm_idx], float), 14.0, mode="nearest")
R0 = np.column_stack([ndi.gaussian_filter1d(R[:, 0], 4.0, mode="nearest"), ndi.gaussian_filter1d(R[:, 1], 4.0, mode="nearest")])
tx, ty = np.gradient(R0[:, 0]), np.gradient(R0[:, 1])
tn = np.maximum(np.hypot(tx, ty), 1e-9)
R = R + np.column_stack([ty / tn, -tx / tn]) * off_s[:, None]
Rs = np.column_stack([ndi.gaussian_filter1d(R[:, 0], 5.0, mode="nearest"), ndi.gaussian_filter1d(R[:, 1], 5.0, mode="nearest")])
route_s = LineString(Rs)
corridor = route_s.buffer(2.9)

# ==== bâtiments ====
log("bâtiments")
clear = unary_union([roads.buffer(-0.6), corridor])
raw = []
for f in load("batiment"):
    p = f["properties"]
    if p.get("etat_de_l_objet") not in (None, "En service"):
        continue
    g = loc(shape(f["geometry"]))
    if not (wbox.contains(g.centroid) if BLOC else g.within(wbox.buffer(-6))):
        continue
    for part in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
        q = Polygon(part.exterior, part.interiors).buffer(0).simplify(0.25)
        q = q.difference(clear)
        if q.geom_type == "MultiPolygon":
            q = max(q.geoms, key=lambda s: s.area)
        q = q.buffer(-0.35, join_style=2).buffer(0.35, join_style=2)
        if q.geom_type == "MultiPolygon":
            q = max(q.geoms, key=lambda s: s.area)
        if q.is_empty or q.area < 12 or q.geom_type != "Polygon":
            continue
        q = Polygon(q.exterior, [h for h in q.interiors if Polygon(h).area > 20]).simplify(0.15)
        if q.is_valid and q.area >= 12:
            raw.append(dict(g=q, p=p))
B = raw
tr = from_origin(0, COTE, 0.5, 0.5)
lab = features.rasterize([(b["g"], i + 1) for i, b in enumerate(B)], out_shape=mnh.shape, transform=tr, fill=0, dtype="int32")
objs = ndi.find_objects(lab, max_label=len(B))
for i, b in enumerate(B):
    eave = top = None
    sl = objs[i]
    if sl is not None:
        m = lab[sl] == i + 1
        me = ndi.binary_erosion(m, iterations=2) if m.sum() > 60 else m
        v = mnh[sl][me if me.sum() >= 12 else m]
        if v.size >= 12:
            eave, top = float(np.percentile(v, 15)), float(np.percentile(v, 95))
    bd = b["p"].get("hauteur")
    if eave is None or eave < 2.0:
        eave = bd if (bd and bd >= 2.0) else 3.0
        top = max(top or 0, eave)
    b["eave"], b["rise"] = float(np.clip(eave, 2.3, 30)), float(max(0, top - eave))
    b["levels"] = max(1, int(round(b["eave"] / 2.9)))
log(len(B), "bâtiments")


def hs(x, y):
    r, c = int((COTE - y) / 0.5), int(x / 0.5)
    if 1 <= r < 2 * COTE - 1 and 1 <= c < 2 * COTE - 1:
        return float(np.median(mnh[r - 1:r + 2, c - 1:c + 2]))
    return 0.0


def _ec(pts, ends):
    return earcut.triangulate_float64(np.asarray(pts, np.float64), np.asarray(ends, np.uint32)).reshape(-1, 3)


def skeleton_top(poly, slope, cap):
    """Renvoie (sommets xy, hauteur t, triangles du toit, arêtes de mur orientées)."""
    poly = orient(poly, 1.0)
    ext = [(x, y, 0.0) for x, y in list(poly.exterior.coords)[:-1]]
    holes = [[(x, y, 0.0) for x, y in list(h.coords)[:-1]] for h in poly.interiors]
    V, ids, E, T = (sk.interior_straight_skeleton_with_holes(ext, holes, as_graph=False) if holes
                    else sk.interior_straight_skeleton(ext, as_graph=False))
    V = np.asarray(V, float)[:, :2]
    idm = {int(i): k for k, i in enumerate(np.asarray(ids).ravel())}
    E = [(idm[int(a)], idm[int(b)]) for a, b in np.asarray(E)]
    T = np.asarray(T).ravel()
    bset = {frozenset(e) for e, t in zip(E, T) if t == 2}
    key = {(V[i, 0], V[i, 1]): i for i in range(len(V))}
    faces = [f for f in polygonize([LineString([V[a], V[b]]) for a, b in E]) if poly.contains(f.representative_point())]
    if len(faces) != len(bset):
        raise ValueError("faces")
    t = np.zeros(len(V)); tris, walls = [], []
    for f in faces:
        ring = [key[c] for c in list(orient(f, 1.0).exterior.coords)[:-1]]
        n = len(ring)
        be = [(ring[k], ring[(k + 1) % n]) for k in range(n) if frozenset((ring[k], ring[(k + 1) % n])) in bset]
        if len(be) != 1:
            raise ValueError("arête")
        i, j = be[0]
        d = V[j] - V[i]; L = np.hypot(*d)
        nrm = np.array([-d[1], d[0]]) / max(L, 1e-9)
        for k in ring:
            t[k] = max(t[k], max(0.0, float(np.dot(V[k] - V[i], nrm))))
        tris += [[ring[a], ring[b], ring[c]] for a, b, c in _ec(V[ring], [n])]
        walls.append((i, j))
    return V, np.minimum(t * slope, cap), np.array(tris), walls


PAL = np.array([[0.93, 0.87, 0.74], [0.96, 0.94, 0.88], [0.90, 0.76, 0.58], [0.94, 0.82, 0.64], [0.88, 0.68, 0.54], [0.97, 0.96, 0.93],
                [0.83, 0.74, 0.60], [0.95, 0.86, 0.60], [0.92, 0.72, 0.62], [0.86, 0.80, 0.72], [0.98, 0.90, 0.78]])
# style régional (styles/regions.json) : couleurs de façades, volets, clôtures, d'après le département du bloc
_STY = json.load(open(os.path.join(HERE, "styles", "regions.json")))
DEP = str(json.load(open(os.path.join(D, "v_bbox.json"))).get("dep", "66"))
STYLE = next((k_ for k_, v_ in _STY.items() if not k_.startswith("_") and DEP in v_.get("departements", [])), "defaut")
if BLOC:
    PAL = np.array(_STY[STYLE]["facades"])
VOLETS = _STY[STYLE]["volets"][rng.integers(len(_STY[STYLE]["volets"]))]
CLO = np.cumsum(_STY[STYLE]["clotures"]) / sum(_STY[STYLE]["clotures"])


class Acc:
    """Accumulateur de maillage par dalle."""
    def __init__(self, attrs):
        self.attrs = attrs
        self.d = {}

    def add(self, x, y, tris_by_attr):
        k = (min(NT - 1, max(0, int(x // TS))), min(NT - 1, max(0, int(y // TS))))
        t = self.d.setdefault(k, {a: [] for a in self.attrs})
        for a in self.attrs:
            t[a].append(np.asarray(tris_by_attr[a], np.float32))

    def out(self):
        return {k: {a: np.concatenate(v[a]) for a in self.attrs} for k, v in self.d.items() if v["pos"]}


roofs, wallsW, plain = Acc(["pos", "col"] if BLOC else ["pos"]), Acc(["pos", "uv", "col"]), Acc(["pos", "col"])
gtree = STRtree([b["g"] for b in B])
SLOPE, CAP = 0.36, 3.2
nfail = 0
for bi, b in enumerate(B):
    g = orient(b["g"], 1.0)
    c = g.representative_point()
    pts = np.asarray(g.exterior.coords)[:-1]
    zs = zgv(pts)
    z_ref = float(np.median(zs)); z_bot = float(zs.min()) - 0.8
    z_wall = z_ref + b["eave"]
    tint = PAL[rng.integers(len(PAL))] * rng.uniform(0.94, 1.03)
    rc_ = ortho_rgb(c.x, c.y, 3)
    tuile = rc_[0] > rc_[1] * 1.12 and rc_[0] > rc_[2] * 1.22
    hangar = b["p"].get("nature") in ("Industriel, agricole ou commercial", "Serre", "Tribune") or (not tuile and g.area > 450)
    if hangar:
        tint = np.array([0.80, 0.82, 0.83]) * rng.uniform(0.85, 1.05)          # bardage clair, sans fenêtres
    elif not tuile:
        tint = np.array([[0.97, 0.97, 0.95], [0.93, 0.93, 0.92], [0.96, 0.93, 0.86]][rng.integers(3)]) * rng.uniform(0.96, 1.02)   # toit plat : maison récente, enduit blanc
    b["z_ref"], b["z_wall"] = z_ref, z_wall
    flat = b["rise"] < 0.7 or b["p"].get("nature") == "Serre"
    rt, wl, gab = None, None, []
    try:
        if not flat:
            q = pts
            ridge = None
            if len(q) == 4 and not g.interiors and g.convex_hull.area / g.area < 1.02:
                e = b["eave"]; ris = max(hs(c.x, c.y) - e, b["rise"] * 0.7)
                ctr = q.mean(axis=0)
                mids = [(q[k] + q[(k + 1) % 4]) / 2 for k in range(4)]
                hi = [hs(*(mk + (ctr - mk) / max(np.hypot(*(ctr - mk)), 1e-6) * 1.2)) > e + 0.6 * ris for mk in mids]
                L = [np.hypot(*(q[(k + 1) % 4] - q[k])) for k in range(4)]
                if hi[0] and hi[2] and not (hi[1] and hi[3]):
                    ridge = 0
                elif hi[1] and hi[3] and not (hi[0] and hi[2]):
                    ridge = 1
                elif any(hi):
                    ridge = 0 if (L[0] + L[2]) < (L[1] + L[3]) else 1
            if ridge is not None:
                qq = np.roll(q, -ridge, axis=0)
                m1, m3 = (qq[0] + qq[1]) / 2, (qq[2] + qq[3]) / 2
                rise = min((np.hypot(*(qq[1] - qq[0])) + np.hypot(*(qq[3] - qq[2]))) / 4 * SLOPE * 1.25, CAP)
                P3 = lambda p, z: [p[0], p[1], z]
                rt = np.array([[P3(qq[1], z_wall), P3(qq[2], z_wall), P3(m3, z_wall + rise)], [P3(qq[1], z_wall), P3(m3, z_wall + rise), P3(m1, z_wall + rise)],
                               [P3(qq[3], z_wall), P3(qq[0], z_wall), P3(m1, z_wall + rise)], [P3(qq[3], z_wall), P3(m1, z_wall + rise), P3(m3, z_wall + rise)]])
                gab = [[P3(qq[0], z_wall), P3(qq[1], z_wall), P3(m1, z_wall + rise)], [P3(qq[2], z_wall), P3(qq[3], z_wall), P3(m3, z_wall + rise)]]
                wl = [(qq[k], qq[(k + 1) % 4]) for k in range(4)]
            else:
                V, t, tris, walls = skeleton_top(g, SLOPE, CAP)
                V3 = np.column_stack([V, z_wall + t])
                rt = V3[tris]
                wl = [(V[i], V[j]) for i, j in walls]
    except Exception:
        nfail += 1
        rt = None
    if rt is None:
        rings = [np.asarray(g.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in g.interiors]
        flatp = np.vstack(rings)
        tr_ = _ec(flatp, np.cumsum([len(r) for r in rings]))
        rt = np.column_stack([flatp, np.full(len(flatp), z_wall + 0.25)])[tr_]
        wl = [(r[k], r[(k + 1) % len(r)]) for r in rings for k in range(len(r))]
        z_wall += 0.25
    if BLOC:                                                        # toit coloré d'après la photo (pas de texture)
        rv = rt.reshape(-1, 3); roofs.add(c.x, c.y, dict(pos=rv, col=np.tile(np.clip(rc_ * 1.06, 0, 1), (len(rv), 1))))
    else:
        roofs.add(c.x, c.y, dict(pos=rt.reshape(-1, 3)))
    wp, wu, wc, pp, pc = [], [], [], [], []
    for a_, b_ in wl:
        L = float(np.hypot(*(b_ - a_)))
        if L < 0.05:
            continue
        quad = [[a_[0], a_[1], z_bot], [b_[0], b_[1], z_bot], [b_[0], b_[1], z_wall], [a_[0], a_[1], z_bot], [b_[0], b_[1], z_wall], [a_[0], a_[1], z_wall]]
        if L >= 2.6 and not hangar:
            n = max(1, round(L / 3.1))
            v0 = (z_bot - z_ref) / b["eave"] * b["levels"]
            wp += quad; wu += [[0, v0], [n, v0], [n, b["levels"]], [0, v0], [n, b["levels"]], [0, b["levels"]]]; wc += [tint] * 6
        else:
            pp += quad; pc += [tint] * 6
    for tri in gab:
        pp += tri; pc += [tint] * 3
    if wp:
        wallsW.add(c.x, c.y, dict(pos=wp, uv=wu, col=wc))
    if pp:
        plain.add(c.x, c.y, dict(pos=pp, col=pc))
log("toits faits, replis:", nfail)

# ==== arbres (repérés au LiDAR, couleur prise sur la photo) ====
bmask = ndi.binary_dilation(lab > 0, iterations=3)
veg = ndi.binary_opening((mnh > 2.2) & ~bmask, structure=np.ones((3, 3)))
l3, n3 = ndi.label(veg)
veg = np.isin(l3, 1 + np.nonzero(ndi.sum(veg, l3, range(1, n3 + 1)) >= 16)[0])
sm = ndi.gaussian_filter(np.where(veg, mnh, 0.0), 1.6)
dt = ndi.distance_transform_edt(veg) * 0.5
pk = peak_local_max(sm, min_distance=7, threshold_abs=2.8, labels=veg.astype(int), exclude_border=False)
if BLOC and len(pk) > 14000:                                             # forêts : plafond d'arbres par bloc, on garde les plus hauts
    pk = pk[np.argsort(-sm[pk[:, 0], pk[:, 1]])[:14000]]
ICO_V = np.array([[-1, 1.618, 0], [1, 1.618, 0], [-1, -1.618, 0], [1, -1.618, 0], [0, -1, 1.618], [0, 1, 1.618], [0, -1, -1.618], [0, 1, -1.618],
                  [1.618, 0, -1], [1.618, 0, 1], [-1.618, 0, -1], [-1.618, 0, 1]]) / 1.902
ICO_F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
trees = Acc(["pos", "col"])
tree_list = []
roads_large = roads.buffer(0.9)
shapely.prepare(roads_large)


def prism_tris(cx, cy, z0, z1, r0, r1, n=5, rot=0.0):
    out = []
    for k in range(n):
        a0, a1 = rot + 2 * math.pi * k / n, rot + 2 * math.pi * (k + 1) / n
        p0, p1 = [cx + r0 * math.cos(a0), cy + r0 * math.sin(a0), z0], [cx + r0 * math.cos(a1), cy + r0 * math.sin(a1), z0]
        q0, q1 = [cx + r1 * math.cos(a0), cy + r1 * math.sin(a0), z1], [cx + r1 * math.cos(a1), cy + r1 * math.sin(a1), z1]
        out += [p0, p1, q1, p0, q1, q0]
    return out


for r, c in pk:
    x, y = (c + 0.5) * 0.5, COTE - (r + 0.5) * 0.5
    if not ((0 <= x < COTE and 0 <= y < COTE) if BLOC else (8 < x < COTE - 8 and 8 < y < COTE - 8)) or corridor.distance(Point(x, y)) < 1.2 or roads_large.contains(Point(x, y)):
        continue
    h = float(np.clip(mnh[max(0, r - 2):r + 3, max(0, c - 2):c + 3].max(), 3.0, 22.0))
    cr = float(np.clip(dt[r, c] * 1.15, 1.2, 4.6)) * rng.uniform(0.9, 1.1)
    z = zg(x, y)
    col = ortho_rgb(x, y, 2)
    green = col[1] > col[0] * 1.02 and col[1] > col[2] * 1.05
    base = np.array([0.22, 0.36, 0.16]) if not green else np.clip(col * np.array([0.8, 1.0, 0.75]), 0.08, 0.6)
    base = base * rng.uniform(0.85, 1.12)
    P, C = [], []
    trunk = np.array([0.33, 0.25, 0.18])
    rot = rng.uniform(0, 6.28)
    if h > 6.5 and cr < 1.9:                      # cyprès : cône sombre
        tt = prism_tris(x, y, z - 0.3, z + h, cr * 0.75, 0.12, 5, rot)
        P += tt; C += [base * 0.72] * len(tt)
    else:
        th = h * (0.5 if (h > 9 and rng.random() < 0.3) else 0.34)
        tt = prism_tris(x, y, z - 0.3, z + th, 0.2 + h * 0.014, 0.14 + h * 0.009, 3, rot)
        P += tt; C += [trunk] * len(tt)
        for (ox, oy, sc, zc) in [(0, 0, 1.0, 0.0)] + ([(rng.uniform(-0.5, 0.5) * cr, rng.uniform(-0.5, 0.5) * cr, 0.7, 0.25)] if (cr > 3.2 and route_s.distance(Point(x, y)) < 120) else []):
            rz = (h - th) * 0.62 * sc
            ctr = np.array([x + ox, y + oy, z + th + (h - th) * (0.45 + zc * 0.5)])
            Vv = ICO_V * np.array([cr * sc, cr * sc, max(rz, 0.9)]) + ctr
            sh = 0.78 + 0.3 * (ICO_V[:, 2] + 1) / 2
            for f in ICO_F:
                P += Vv[f].tolist(); C += [base * sh[f].mean()] * 3
    trees.add(x, y, dict(pos=P, col=C))
    tree_list.append((x, y))
log(len(tree_list), "arbres")

# ==== lampadaires (posés automatiquement le long des rues, ce ne sont pas leurs vraies positions) ====
lamps = Acc(["pos", "col"])
nl = 0


def box_tris(c, s):
    x0, y0, z0 = c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2
    x1, y1, z1 = c[0] + s[0] / 2, c[1] + s[1] / 2, c[2] + s[2] / 2
    v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    f = [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [4, 5, 6], [4, 6, 7], [0, 2, 1], [0, 3, 2]]
    return [v[i] for t in f for i in t]


bun = unary_union([b["g"] for b in B]).buffer(0.4)
shapely.prepare(bun)
side = 1
for c, w in lamps_src:
    ln = LineString(c)
    for s in np.arange(12.0, ln.length - 6, 46.0 if GAME else 34.0):
        p = np.array(ln.interpolate(s).coords[0]); q = np.array(ln.interpolate(s + 1).coords[0])
        d = (q - p) / max(np.hypot(*(q - p)), 1e-6)
        nrm = np.array([-d[1], d[0]]) * side
        side = -side
        pos = p + nrm * (w / 2 + 1.1)
        if bun.contains(Point(pos)) or not ((0 <= pos[0] < COTE and 0 <= pos[1] < COTE) if BLOC else (5 < pos[0] < COTE - 5 and 5 < pos[1] < COTE - 5)) or (not GAME and route_s.distance(Point(pos)) > 220):
            continue
        z = zg(*pos)
        P = box_tris([pos[0], pos[1], z + 3.2], [0.14, 0.14, 6.4]) + box_tris([pos[0] - nrm[0] * 0.7, pos[1] - nrm[1] * 0.7, z + 6.4], [0.1 + abs(nrm[0]) * 1.4, 0.1 + abs(nrm[1]) * 1.4, 0.1]) \
            + box_tris([pos[0] - nrm[0] * 1.3, pos[1] - nrm[1] * 1.3, z + 6.3], [0.45, 0.45, 0.14])
        C = [[0.2, 0.22, 0.24]] * 72 + [[0.98, 0.95, 0.8]] * 36
        lamps.add(pos[0], pos[1], dict(pos=P, col=C)); nl += 1
log(nl, "lampadaires")

# ==== chaussée, trottoirs et ligne axiale redessinés au net près de l'itinéraire ====
log("chaussée et trottoirs")
over = Acc(["pos", "col"])


def add_mesh(acc, tri, col):
    """tri : (N,3,3) ; col : (N,3) couleur par triangle."""
    tri = np.asarray(tri)
    if tri.size == 0:                                                      # rien à ajouter (bloc sans clôture, en mer...)
        return
    cx_, cy_ = tri[:, :, 0].mean(axis=1), tri[:, :, 1].mean(axis=1)
    ti, tj = np.clip((cx_ // TS).astype(int), 0, NT - 1), np.clip((cy_ // TS).astype(int), 0, NT - 1)
    for i in range(NT):
        for j in range(NT):
            m = (ti == i) & (tj == j)
            if m.any():
                t = acc.d.setdefault((i, j), {a: [] for a in acc.attrs})
                t["pos"].append(tri[m].reshape(-1, 3).astype(np.float32))
                t["col"].append(np.repeat(col[m], 3, axis=0).astype(np.float32))


def hfield(dz):
    ny, nx_ = Zg.shape
    top = np.column_stack([gX.ravel(), gY.ravel(), (Zg + dz).ravel()])
    idx = np.arange(nx_ * ny).reshape(ny, nx_)
    a, b_, c_, d_ = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = [np.column_stack([a, b_, c_]), np.column_stack([a, c_, d_])]
    per = np.concatenate([idx[0, :-1], idx[:-1, -1], idx[-1, :0:-1], idx[:0:-1, 0]])
    n = len(per); bot = top[per].copy(); bot[:, 2] = -30.0
    bi = nx_ * ny + np.arange(n); ctr = nx_ * ny + n
    V = np.vstack([top, bot, [[WORLD / 2, WORLD / 2, -30.0]]])
    nxt = np.roll(np.arange(n), -1)
    F += [np.column_stack([per, bi, bi[nxt]]), np.column_stack([per, bi[nxt], per[nxt]]), np.column_stack([bi[nxt], bi, np.full(n, ctr)])]
    return Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=np.vstack(F).astype(np.uint32)))


def drape(poly, dz, color, jitter=0.0):
    cs = []
    for q in (poly.geoms if hasattr(poly, "geoms") else [poly]):
        if q.is_empty or q.geom_type != "Polygon" or q.area < 0.02:
            continue
        q = orient(q, 1.0)
        cs.append(np.asarray(q.exterior.coords)[:-1]); cs += [np.asarray(h.coords)[:-1] for h in q.interiors]
    if not cs:
        return
    pr = Manifold.extrude(CrossSection(cs, mf3.FillRule.Positive), 200.0).translate((0, 0, -100))
    me = (hfield(dz) ^ pr).to_mesh()
    V = np.asarray(me.vert_properties)[:, :3]; F = np.asarray(me.tri_verts)
    keep = (V[:, 2] > zgv(V[:, :2]) - 0.6)[F].all(axis=1)
    tri = V[F[keep]]
    col = np.tile(np.asarray(color, float), (len(tri), 1)) * (1 + rng.uniform(-jitter, jitter, (len(tri), 1)))
    add_mesh(over, tri, col)


zone = wbox if BLOC else wbox.buffer(-4) if GAME else route_s.buffer(140)
chaussee = unary_union([ln.buffer(max(w, 3.0) / 2 + 0.1, cap_style=1) for ln, w in road_lines if ln.intersects(zone)]).intersection(zone)
chaussee = chaussee.union(corridor.buffer(-0.6)).buffer(0.4).buffer(-0.4)
bfoot = unary_union([b["g"] for b in B])
RAPIDE = ("Type autoroutier", "Bretelle")
_urb = [ln.buffer(max(w, 3.0) / 2 + 0.1, cap_style=1) for ln, w, nat, _, urb in axes_src if urb and nat not in RAPIDE and ln.intersects(zone)]
_rap = [ln.buffer(w / 2 + 4.5, cap_style=2) for ln, w, nat, _, urb in axes_src if (nat in RAPIDE or not urb) and ln.intersects(zone)]
# trottoirs : seulement en agglomération (tronçons « urbains » de la BD TOPO), jamais sur les autoroutes, bretelles et routes de campagne
trottoir = (unary_union(_urb).buffer(1.5, join_style=1) if _urb else Polygon()).difference(chaussee).difference(bfoot.buffer(0.05))
if _rap:
    trottoir = trottoir.difference(unary_union(_rap))
trottoir = trottoir.buffer(-0.15).buffer(0.15)
tirets = []
for ln, w in road_lines:
    if w >= 5.0 and ln.intersects(zone):
        for s0 in np.arange(1.5, ln.length - 3, 6.0):
            seg = shapely.ops.substring(ln, s0, s0 + 3.0)
            tirets.append(seg.buffer(0.07, cap_style=2))
piste = unary_union([ln.buffer(1.3, cap_style=2) for ln in cyc_lines if ln.intersects(zone)] or [Polygon()]).intersection(zone).difference(chaussee.buffer(0.3))
trottoir = trottoir.difference(piste.buffer(0.1))
drape(piste.difference(bfoot.buffer(0.2)), 0.06, [0.45, 0.50, 0.44], 0.03)          # pistes cyclables : bande à part, on n'y roule pas
drape(chaussee, 0.05, [0.27, 0.28, 0.30], 0.05)
drape(trottoir, 0.17, [0.70, 0.68, 0.64], 0.03)
if tirets:
    drape(unary_union(tirets).intersection(chaussee.buffer(-0.3)), 0.075, [0.92, 0.92, 0.9])
log("chaussée:", round(chaussee.area), "m2, trottoirs:", round(trottoir.area), "m2")

# ==== ponts : tablier à sa hauteur réelle (relevé laser de surface), garde-corps, piles ====
def quads(A, Bq, C, Dq):
    """Quadrilatères A-B-C-D (tableaux (n,3)) en triangles."""
    return np.concatenate([np.stack([A, Bq, C], 1), np.stack([A, C, Dq], 1)])


ponts_out = []


def z_tablier(P):
    """Hauteur réelle du tablier le long de l'axe : relevé laser de surface (médiane locale, sans voitures ni lampadaires),
    lissée ; si le laser ne voit pas de tablier, interpolation entre les deux extrémités au sol."""
    sol_ = zgv(P)
    v = []
    for x_, y_ in P:
        r_, c_ = int((COTE - y_) / 0.5), int(x_ / 0.5)
        win = mns[max(0, r_ - 2):r_ + 3, max(0, c_ - 2):c_ + 3]
        v.append(float(np.median(win)) - ZO if win.size else np.nan)
    v = np.array(v)
    lin = sol_[0] + (sol_[-1] - sol_[0]) * np.linspace(0, 1, len(P))
    if not np.isfinite(v).all() or np.nanmax(v - sol_) < 1.5:
        return lin + 0.05
    v = ndi.median_filter(v, size=7, mode="nearest")
    v = ndi.gaussian_filter1d(v, 2.0, mode="nearest")
    v[0], v[-1] = sol_[0], sol_[-1]                                       # les extrémités rejoignent le sol (culées)
    k = max(1, min(len(v) // 4, 3))
    v[:k + 1] = np.linspace(sol_[0], v[k], k + 1); v[-k - 1:] = np.linspace(v[-k - 1], sol_[-1], k + 1)
    return np.maximum(v, sol_) + 0.05


for c_p, w in ponts_src:
    ln = LineString(c_p)
    if ln.length < 6:
        continue
    n = max(3, int(ln.length / 2) + 1)
    ss = np.linspace(0, ln.length, n)
    P = np.array([ln.interpolate(t).coords[0] for t in ss])
    Z = z_tablier(P)
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9
    N_ = np.column_stack([-T[:, 1], T[:, 0]])
    hw = max(w, 3.0) / 2 + 0.6
    L3 = lambda off, dz: np.column_stack([P + N_ * off, Z + dz])
    tri, col = [], []
    def put_(t_, c_):
        tri.append(t_); col.append(np.tile(c_, (len(t_), 1)))
    put_(quads(L3(-hw, 0)[:-1], L3(hw, 0)[:-1], L3(hw, 0)[1:], L3(-hw, 0)[1:]), [0.27, 0.28, 0.30])           # chaussée du tablier
    for sg in (-1, 1):
        put_(quads(L3(sg * hw, 0)[:-1], L3(sg * hw, -0.9)[:-1], L3(sg * hw, -0.9)[1:], L3(sg * hw, 0)[1:]), [0.66, 0.64, 0.6])   # rive du tablier
        for o0, o1, z0, z1 in ((hw, hw, 0, 1.05), (hw - 0.25, hw - 0.25, 0, 1.05), (hw - 0.25, hw, 1.05, 1.05)):     # garde-corps (muret béton)
            put_(quads(L3(sg * o0, z0)[:-1], L3(sg * o1, z1)[:-1], L3(sg * o1, z1)[1:], L3(sg * o0, z0)[1:]), [0.78, 0.77, 0.74])
    put_(quads(L3(hw, -0.9)[:-1], L3(-hw, -0.9)[:-1], L3(-hw, -0.9)[1:], L3(hw, -0.9)[1:]), [0.55, 0.54, 0.52])       # dessous
    for t in np.arange(9.0, ln.length - 8, 16.0):                                                                     # piles
        k = int(round(t / (ln.length / (n - 1))))
        z0 = zg(*P[k])
        if Z[k] - 0.9 - z0 < 2.0:
            continue
        ang = math.atan2(T[k, 1], T[k, 0])
        for sg in (-1, 1):
            cx_, cy_ = P[k] + N_[k] * sg * (hw - 1.0)
            pt = np.array(prism_tris(cx_, cy_, z0 - 0.3, Z[k] - 0.9, 0.55, 0.55, 8, ang)).reshape(-1, 3, 3)
            put_(pt, [0.62, 0.6, 0.57])
    add_mesh(plain, np.concatenate(tri).astype(np.float64), np.concatenate(col))
    ponts_out.append([np.round(np.column_stack([P, Z]), 2).tolist(), round(hw - 0.3, 2)])
log("ponts :", len(ponts_out), "tabliers")

# ==== autoroutes et voies rapides : terre-plein central en béton (bord gauche des chaussées à sens unique), glissières métalliques à droite ====
barrieres = []                                                             # emprises au sol, pour les collisions
def ruban(ln, off, h0, h1, ep, c_, tri, col):
    """Muret le long de ln, décalé de off (positif = à gauche), du dessus h1 au-dessus du sol (h0 = bas)."""
    n = max(2, int(ln.length / 3) + 1)
    P = np.array([ln.interpolate(t).coords[0] for t in np.linspace(0, ln.length, n)])
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9
    N_ = np.column_stack([-T[:, 1], T[:, 0]])
    Z = zgv(P)
    L3 = lambda o, h: np.column_stack([P + N_ * o, Z + h])
    for o0, o1, z0, z1 in ((off - ep / 2, off - ep / 2, h0, h1), (off + ep / 2, off + ep / 2, h0, h1), (off - ep / 2, off + ep / 2, h1, h1)):
        tri.append(quads(L3(o0, z0)[:-1], L3(o1, z1)[:-1], L3(o1, z1)[1:], L3(o0, z0)[1:]))
        col.append(np.tile(c_, (len(tri[-1]), 1)))
    return LineString(P + N_ * off)
_bt, _bc = [], []
for ln, w, nat, sens, _ in axes_src:
    if nat not in ("Type autoroutier", "Bretelle", "Route à 2 chaussées") or ln.length < 4:
        continue
    l_ = ln if sens != "Sens inverse" else LineString(np.asarray(ln.coords)[::-1])
    hw = w / 2 + 0.7
    if nat in ("Type autoroutier", "Route à 2 chaussées") and sens != "Double sens":
        g_ = ruban(l_, hw + 0.5, -0.3, 0.9, 0.6, [0.66, 0.65, 0.62], _bt, _bc)            # muret béton du terre-plein central (GBA)
        ruban(l_, hw + 0.5, 0.88, 0.92, 0.24, [0.85, 0.84, 0.8], _bt, _bc)                  # arête claire du muret
        barrieres.append(g_.buffer(0.35))
    if nat in ("Type autoroutier", "Bretelle"):
        g_ = ruban(l_, -(hw + 0.6), 0.45, 0.8, 0.12, [0.8, 0.82, 0.85], _bt, _bc)        # lisse de la glissière métallique (galvanisée)
        for t in np.arange(0, l_.length, 4.0):                                              # poteaux tous les 4 m
            q_ = np.array(g_.interpolate(t).coords[0]); z_ = zg(*q_)
            _bt.append(np.array(box_tris([q_[0], q_[1], z_ + 0.35], [0.1, 0.1, 0.7])).reshape(-1, 3, 3)); _bc.append(np.tile([0.55, 0.57, 0.6], (12, 1)))
        barrieres.append(g_.buffer(0.3))
if _bt:
    add_mesh(plain, np.concatenate(_bt).astype(np.float64), np.concatenate(_bc))
log("terre-pleins et glissières :", len(barrieres), "tronçons")

# ==== péages (BD TOPO, équipement de transport « Péage ») : auvent, cabines, barrières ====
peages = 0
if os.path.exists(os.path.join(D, "equipement_de_transport.geojson")):
    for f in load("equipement_de_transport"):
        if f["properties"].get("nature") != "Péage":
            continue
        zp = loc(shape(f["geometry"])).buffer(0)
        if zp.is_empty or not zp.intersects(wbox.buffer(-20)):
            continue
        dirs = []
        for ln, w in road_lines:
            seg = ln.intersection(zp)
            if not seg.is_empty and seg.length > 5:
                for q in (seg.geoms if hasattr(seg, "geoms") else [seg]):
                    a_, b_ = np.asarray(q.coords)[0], np.asarray(q.coords)[-1]
                    v_ = (b_ - a_) / (np.hypot(*(b_ - a_)) + 1e-9)
                    dirs.append(v_ if (v_ @ (dirs[0] if dirs else v_)) >= 0 else -v_)
        if not dirs:
            continue
        TP = np.mean(dirs, axis=0); TP /= np.hypot(*TP); N0 = np.array([-TP[1], TP[0]])
        c0 = np.array(zp.centroid.coords[0])
        cut = LineString([c0 - N0 * 400, c0 + N0 * 400]).intersection(zp)
        if cut.is_empty:
            continue
        cut = max(cut.geoms, key=lambda q: q.length) if hasattr(cut, "geoms") else cut
        a_, b_ = np.asarray(cut.coords)[0], np.asarray(cut.coords)[-1]
        L_ = np.hypot(*(b_ - a_)); mid = (a_ + b_) / 2; z0 = zg(*mid)
        ang = math.atan2(N0[1], N0[0])
        tri, col = [], []
        def boite(cx_, cy_, zc, sx, sy, sz, c_):
            v = np.array(box_tris([0, 0, 0], [sx, sy, sz])).reshape(-1, 3, 3)
            R_ = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            v[:, :, :2] = v[:, :, :2] @ R_.T + [cx_, cy_]; v[:, :, 2] += zc
            tri.append(v); col.append(np.tile(c_, (len(v), 1)))
        # auvent sur poteaux, au-dessus de toute la largeur de la gare de péage
        boite(mid[0], mid[1], z0 + 6.6, L_ + 4, 12, 0.7, [0.93, 0.93, 0.95])
        boite(mid[0], mid[1], z0 + 6.2, L_ + 4.2, 12.2, 0.12, [0.15, 0.32, 0.62])
        nb = max(2, int(L_ / 4.2))
        for k in range(nb + 1):
            u = -L_ / 2 + k * L_ / nb
            px, py = mid + N0 * u
            boite(px, py, z0 + 3.2, 0.35, 0.35, 6.4, [0.7, 0.7, 0.72])                                      # poteau
            if 0 < k < nb + 1 and k < nb:
                cx_, cy_ = mid + N0 * (u + L_ / nb / 2)
            if k < nb:
                ix, iy = mid + N0 * u
                boite(ix, iy, z0 + 0.15, 1.1, 7.0, 0.3, [0.85, 0.82, 0.2])                                  # îlot jaune et blanc
                boite(ix, iy, z0 + 1.6, 1.0, 2.4, 2.6, [0.9, 0.9, 0.88])                                    # cabine
                boite(ix, iy, z0 + 2.3, 1.02, 2.0, 0.9, [0.16, 0.22, 0.28])                                 # vitres de la cabine
                boite(ix, iy, z0 + 3.3, 0.5, 0.3, 0.3, [0.1, 0.75, 0.25] if k % 3 else [0.85, 0.12, 0.1])   # feu vert ou rouge
                bx, by = mid + N0 * (u + 1.6) + TP * 1.0
                boite(bx, by, z0 + 1.0, 2.6, 0.12, 0.12, [0.95, 0.2, 0.15])                                 # barrière
        v = np.concatenate(tri)
        # orientation : box_tris est tourné de ang (axe x de la boîte le long de N0)
        add_mesh(plain, v.astype(np.float64), np.concatenate(col))
        peages += 1
log("péages :", peages)

if GAME:
    log("clôtures et mobilier")
    # clôtures de jardins : limites réelles des parcelles bâties (cadastre) ; hauteur et aspect choisis au hasard
    lims = []
    for f in load("v_cadastre"):
        g_ = loc(shape(f["geometry"])).buffer(0)
        if g_.is_empty or g_.area > 3500 or g_.area < 60 or not g_.intersects(bfoot):
            continue
        for q in (g_.geoms if hasattr(g_, "geoms") else [g_]):
            lims.append(q.exterior)
    lim = unary_union(lims).difference(chaussee.buffer(1.2)).difference(bfoot.buffer(0.5)).intersection(wbox if BLOC else wbox.buffer(-6))
    lim = shapely.ops.linemerge(lim) if lim.geom_type == "MultiLineString" else lim
    wt, wc, nclo = [], [], 0
    for ln in (lim.geoms if hasattr(lim, "geoms") else [lim]):
        if ln.length < 2.5:
            continue
        k = rng.random()
        _c1, _c2 = (CLO[0], CLO[1]) if BLOC else (0.5, 0.8)
        hh, col_ = (1.5, [0.90, 0.87, 0.80]) if k < _c1 else (1.8, [0.20, 0.36, 0.17]) if k < _c2 else (1.1, [0.72, 0.70, 0.67])
        col_ = np.array(col_) * rng.uniform(0.9, 1.06)
        pts_ = np.array([ln.interpolate(d_).coords[0] for d_ in np.linspace(0, ln.length, max(2, int(ln.length / 4) + 1))])
        zz = zgv(pts_)
        for i_ in range(len(pts_) - 1):
            a_, b_ = pts_[i_], pts_[i_ + 1]
            wt += [[[a_[0], a_[1], zz[i_] - 0.3], [b_[0], b_[1], zz[i_ + 1] - 0.3], [b_[0], b_[1], zz[i_ + 1] + hh]],
                   [[a_[0], a_[1], zz[i_] - 0.3], [b_[0], b_[1], zz[i_ + 1] + hh], [a_[0], a_[1], zz[i_] + hh]]]
            wc += [col_, col_ * 0.96]
        nclo += ln.length
    add_mesh(plain, np.array(wt, np.float32), np.array(wc, np.float32))
    # mobilier urbain réel (OpenStreetMap) : feux, arrêts de bus, bancs, panneaux, conteneurs, passages piétons
    import xml.etree.ElementTree as ET
    from pyproj import Transformer
    TR = Transformer.from_crs(4326, 2154, always_xy=True)
    rtree = STRtree([ln for ln, _ in road_lines] or [Point(0, 0)])

    def bord(x, y, extra=0.9):
        """Renvoie (point sur le bord droit de la rue la plus proche, tangente, largeur, point sur l'axe)."""
        if not road_lines:                                                     # bloc sans route : on pose sur place
            p_ = np.array([x, y], float)
            return p_, np.array([1.0, 0.0]), 4.0, p_
        i_ = int(rtree.nearest(Point(x, y)))
        ln, w_ = road_lines[i_]
        s_ = ln.project(Point(x, y)); p_ = np.array(ln.interpolate(s_).coords[0]); q_ = np.array(ln.interpolate(min(s_ + 1, ln.length)).coords[0])
        if np.hypot(*(q_ - p_)) < 1e-6:
            q_ = p_ + np.array([1.0, 0.0])
        t_ = (q_ - p_) / np.hypot(*(q_ - p_))
        return p_ + np.array([t_[1], -t_[0]]) * (w_ / 2 + extra), t_, w_, p_

    P_, C_, zebra, cnt = [], [], [], {}
    for nd in ET.parse(os.path.join(D, "osm.xml")).getroot().iter("node"):
        tg = {t.get("k"): t.get("v") for t in nd.findall("tag")}
        if not tg:
            continue
        X_, Y_ = TR.transform(float(nd.get("lon")), float(nd.get("lat")))
        x, y = X_ - XO, Y_ - YO
        if not (10 < x < COTE - 10 and 10 < y < COTE - 10):
            continue
        hw, am = tg.get("highway"), tg.get("amenity")
        kind = None
        if hw == "traffic_signals" or tg.get("crossing") == "traffic_signals":
            kind = "feu"
            (px, py), t_, w_, ax = bord(x, y)
            z = zg(px, py)
            P_ += box_tris([px, py, z + 1.7], [0.16, 0.16, 3.4]) + box_tris([px, py, z + 3.85], [0.34, 0.34, 0.95])
            C_ += [[0.16, 0.17, 0.18]] * 72
            for kz, cl in ((4.14, [0.95, 0.12, 0.1]), (3.85, [1.0, 0.62, 0.1]), (3.56, [0.15, 0.85, 0.3])):
                P_ += box_tris([px, py, z + kz], [0.4, 0.4, 0.2]); C_ += [cl] * 36
        if hw in ("stop", "give_way"):
            kind = "panneau"
            (px, py), t_, w_, ax = bord(x, y)
            z = zg(px, py)
            P_ += box_tris([px, py, z + 1.1], [0.08, 0.08, 2.2]) + box_tris([px, py, z + 2.35], [0.12 + abs(t_[1]) * 0.6, 0.12 + abs(t_[0]) * 0.6, 0.7])
            C_ += [[0.6, 0.62, 0.64]] * 36 + [[0.85, 0.1, 0.1] if hw == "stop" else [0.95, 0.95, 0.95]] * 36
        if hw == "crossing":
            kind = "passage"
            _, t_, w_, ax = bord(x, y)
            n_ = np.array([-t_[1], t_[0]])
            for off_ in np.arange(-w_ / 2 + 0.4, w_ / 2 - 0.3, 1.0):
                c0 = ax + n_ * off_
                zebra.append(Polygon([c0 - t_ * 1.6 - n_ * 0.25, c0 + t_ * 1.6 - n_ * 0.25, c0 + t_ * 1.6 + n_ * 0.25, c0 - t_ * 1.6 + n_ * 0.25]))
        if hw == "bus_stop" and not bfoot.buffer(1).contains(Point(x, y)):
            kind = "arrêt de bus"
            (px, py), t_, w_, ax = bord(x, y, 1.6)
            z = zg(px, py); n_ = np.array([t_[1], -t_[0]])
            for sg in (-1.4, 1.4):
                q_ = np.array([px, py]) + t_ * sg + n_ * 0.6
                P_ += box_tris([q_[0], q_[1], z + 1.2], [0.1, 0.1, 2.4]); C_ += [[0.25, 0.3, 0.35]] * 36
            P_ += box_tris([px, py, z + 2.45], [0.3 + abs(t_[0]) * 3.0 + abs(n_[0]) * 1.5, 0.3 + abs(t_[1]) * 3.0 + abs(n_[1]) * 1.5, 0.1]); C_ += [[0.2, 0.45, 0.7]] * 36
        if am == "bench" and not bfoot.contains(Point(x, y)):
            kind = "banc"
            z = zg(x, y)
            P_ += box_tris([x, y, z + 0.42], [1.6, 0.45, 0.1]) + box_tris([x, y + 0.2, z + 0.75], [1.6, 0.08, 0.45]); C_ += [[0.45, 0.3, 0.18]] * 72
        if am in ("recycling", "waste_disposal") and not bfoot.contains(Point(x, y)) and not roads_large.contains(Point(x, y)):
            kind = "conteneur"
            z = zg(x, y)
            P_ += box_tris([x, y, z + 0.8], [1.4, 1.4, 1.6]); C_ += [[0.18, 0.5, 0.3]] * 36
        if tg.get("barrier") == "bollard":
            kind = "borne"
            z = zg(x, y)
            P_ += box_tris([x, y, z + 0.45], [0.18, 0.18, 0.9]); C_ += [[0.3, 0.3, 0.32]] * 36
        if kind:
            cnt[kind] = cnt.get(kind, 0) + 1
    if P_:
        add_mesh(lamps, np.array(P_, np.float32).reshape(-1, 3, 3), np.array(C_, np.float32)[::3])
    if zebra:
        drape(unary_union(zebra).intersection(chaussee), 0.085, [0.94, 0.94, 0.92])
    log(f"clôtures: {nclo / 1000:.1f} km ; mobilier OSM: {cnt}")

# ==== terrain par dalle ====
terr = {}
cpt = int(TS / GS)
for i in range(NT):
    for j in range(NT):
        sx, sy = slice(i * cpt, (i + 1) * cpt + 1), slice(j * cpt, (j + 1) * cpt + 1)
        X, Y, Z = gX[sy, sx], gY[sy, sx], Zg[sy, sx]
        n = cpt + 1
        idx = np.arange(n * n).reshape(n, n)
        a, b_, c_, d_ = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        F = np.vstack([np.column_stack([a, b_, c_]), np.column_stack([a, c_, d_])])
        Vt = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        terr[(i, j)] = dict(pos=Vt[F].reshape(-1, 3))

# ==== alentours : photo aérienne large (2 m par pixel) posée à plat autour du village ====
OEXT = 6400.0
OJ = os.path.join(OUT, "ortho_outer.jpg")
if not BLOC and not os.path.exists(OJ):
    x0, y0 = XO + WORLD / 2 - OEXT / 2, YO + WORLD / 2 - OEXT / 2
    pq = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS="ORTHOIMAGERY.ORTHOPHOTOS", STYLES="", CRS="EPSG:2154",
              BBOX=f"{x0},{y0},{x0 + OEXT},{y0 + OEXT}", WIDTH=3200, HEIGHT=3200, FORMAT="image/jpeg")
    Image.open(io.BytesIO(fetch.get(fetch.WMSR + "?" + urllib.parse.urlencode(pq)))).convert("RGB").save(OJ, quality=88)
zedge = float(np.concatenate([Zg[0], Zg[-1], Zg[:, 0], Zg[:, -1]]).mean())
tq = np.linspace(0, WORLD, 69)
per = np.vstack([np.column_stack([tq[:-1], np.zeros(68)]), np.column_stack([np.full(68, WORLD), tq[:-1]]),
                 np.column_stack([tq[::-1][:-1], np.full(68, WORLD)]), np.column_stack([np.zeros(68), tq[::-1][:-1]])])
ctr2 = np.array([WORLD / 2, WORLD / 2])
rows = []
for sc_, zz in ((1.0, None), (1.22, zedge), (OEXT / WORLD, zedge)):
    q = ctr2 + (per - ctr2) * sc_
    rows.append(np.column_stack([q, zgv(per) if zz is None else np.full(len(q), zz)]))
otri = []
n_ = len(per)
for k in range(2):
    for i in range(n_):
        a_, b_, c_, d_ = rows[k][i], rows[k][(i + 1) % n_], rows[k + 1][(i + 1) % n_], rows[k + 1][i]
        otri += [a_, b_, c_, a_, c_, d_]
otri = np.array(otri, np.float32)
ouv = (otri[:, :2] - (ctr2 - OEXT / 2)) / OEXT
if BLOC:                                                                 # pas d'alentours : les blocs voisins les remplacent
    otri, ouv = np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)

# ==== trajectoire de la voiture et caméra, image par image ====
log("trajectoire")
dx, dy = np.gradient(Rs[:, 0], STEP), np.gradient(Rs[:, 1], STEP)
ddx, ddy = np.gradient(dx, STEP), np.gradient(dy, STEP)
kap = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-6)
kap = ndi.maximum_filter1d(ndi.gaussian_filter1d(kap, 6), 30)
VMAX = 13.6
vlim = np.minimum(VMAX, np.sqrt(7.0 / np.maximum(kap, 1e-4)))
vlim = np.maximum(ndi.gaussian_filter1d(ndi.minimum_filter1d(vlim, 40), 24), 7.5)
SMAX = (len(Rs) - 1) * STEP
T_INTRO, T_OUT = 8.0, 11.0
s, v, S = 0.0, 0.0, []
while s < SMAX - 0.5:
    vt = float(np.interp(s, np.arange(len(vlim)) * STEP, vlim))
    brake = math.sqrt(max(0.0, 2 * 4.0 * (SMAX - s)))            # freinage final
    vt = min(vt, brake + 0.3)
    v += np.clip(vt - v, -6.0 / FPS, 3.0 / FPS)
    s += v / FPS
    S.append((s, v))
n_drive = len(S)
n_in, n_out = int(T_INTRO * FPS), int(T_OUT * FPS)
NF = n_in + n_drive + n_out
s_f = np.concatenate([np.zeros(n_in), [a for a, _ in S], np.full(n_out, S[-1][0])])
v_f = np.concatenate([np.zeros(n_in), [b_ for _, b_ in S], np.zeros(n_out)])
si = np.arange(len(Rs)) * STEP
car = np.column_stack([np.interp(s_f, si, Rs[:, 0]), np.interp(s_f, si, Rs[:, 1])])
hx, hy = np.interp(s_f, si, dx), np.interp(s_f, si, dy)
hn = np.hypot(hx, hy); hx, hy = hx / hn, hy / hn
carz = zgv(car)
carz = ndi.gaussian_filter1d(carz, 3)
# cap de la caméra : en retard sur la voiture
chx, chy = hx.copy(), hy.copy()
for k in range(1, NF):
    a = 0.055
    chx[k] = chx[k - 1] + (hx[k] - chx[k - 1]) * a; chy[k] = chy[k - 1] + (hy[k] - chy[k - 1]) * a
cn = np.hypot(chx, chy); chx, chy = chx / cn, chy / cn
tt = np.arange(NF) / FPS
TT = NF / FPS
LM = {k_: v_ for k_, v_ in json.load(open(os.path.join(HERE, "video_route.json")))["reperes"].items() if _dans(*v_)} or {"Centre": WAY[0]}


def t_near(X, Y):
    d = np.hypot(car[:, 0] - (X - XO), car[:, 1] - (Y - YO))
    k = int(np.argmin(d))
    return k / FPS, float(d[k])


CH = (math.log(7.5), 3.1, 0.0, 6.0)        # caméra de poursuite : log(distance), hauteur, angle, visée en avant
keys = [(0.0, (math.log(620), 330.0, 2.4, 0.0)), (T_INTRO + 1.5, CH)]
drones = []
for nm, (X, Y) in LM.items():
    t0, dmin = t_near(X, Y)
    if dmin < 90 and T_INTRO + 14 < t0 < TT - T_OUT - 12:
        drones.append((t0, nm))
drones.sort()
last = T_INTRO + 1.5
sgn = 1
for t0, nm in drones:
    a, b_ = t0 - 9.0, t0 + 9.0
    if a - 3.5 < last + 6:
        continue
    keys += [(a - 3.5, CH), (a, (math.log(62), 46.0, 0.55 * sgn, 0.0)), (b_, (math.log(74), 58.0, -0.5 * sgn, 0.0)), (b_ + 3.5, CH)]
    last = b_ + 3.5; sgn = -sgn
    log(f"  vue drone autour de {nm} à {t0:.0f} s")
keys += [(TT - T_OUT - 0.5, CH), (TT - 5.0, (math.log(330), 250.0, 1.5, 0.0)), (TT, (math.log(420), 330.0, 2.1, 0.0))]
keys.sort()
kt = np.array([k[0] for k in keys]); kv = np.array([k[1] for k in keys])
ss = lambda u: u * u * u * (u * (u * 6 - 15) + 10)
par = np.zeros((NF, 4))
for k in range(NF):
    j = int(np.clip(np.searchsorted(kt, tt[k], side="right") - 1, 0, len(kt) - 2))
    u = ss(float(np.clip((tt[k] - kt[j]) / max(kt[j + 1] - kt[j], 1e-6), 0, 1)))
    par[k] = kv[j] * (1 - u) + kv[j + 1] * u
par = ndi.gaussian_filter1d(par, 4, axis=0, mode="nearest")
Dc, Hc, Yaw, Fwd = np.exp(par[:, 0]), par[:, 1], par[:, 2], par[:, 3]
# en poursuite, la caméra ne doit pas entrer dans une maison : on la rapproche si besoin
btree = STRtree([b["g"].buffer(0.9) for b in B])
for k in range(NF):
    if Hc[k] > 10:
        continue
    for it in range(10):
        cx_, cy_ = car[k, 0] - chx[k] * Dc[k], car[k, 1] - chy[k] * Dc[k]
        if len(btree.query(Point(cx_, cy_), predicate="intersects")) == 0 or Dc[k] < 3.2:
            break
        Dc[k] *= 0.86
low = Hc < 10
Dm = ndi.minimum_filter1d(Dc, 21)
Dc = np.where(low, ndi.gaussian_filter1d(Dm, 7), Dc)
ca, sa = np.cos(Yaw), np.sin(Yaw)
bx, by = chx * ca - chy * sa, chx * sa + chy * ca
cam = np.column_stack([car[:, 0] - bx * Dc, car[:, 1] - by * Dc, carz + Hc])
cam[:, 2] = np.maximum(cam[:, 2], zgv(cam[:, :2]) + 1.6)
look = np.column_stack([car[:, 0] + chx * Fwd, car[:, 1] + chy * Fwd, carz + 1.1])
names_f = [names_s[min(int(q / STEP), len(names_s) - 1)] for q in s_f]
uniq = sorted(set(names_f))
print("noms:", {n: names_f.count(n) for n in uniq})
log(f"vidéo: {NF} images, {TT:.1f} s, conduite {n_drive / FPS:.0f} s, vitesse moyenne {np.mean([b_ for _, b_ in S]) * 3.6:.0f} km/h")

# ==== export ====
blobs, man, off = [], {}, 0


def put(arr, dtype=np.float32):
    global off
    b_ = np.ascontiguousarray(arr, dtype).tobytes()
    blobs.append(b_)
    r = [off, len(b_) // np.dtype(dtype).itemsize]
    off += len(b_)
    return r


tiles = {}
for i in range(NT):
    for j in range(NT):
        t = {}
        t["terrain"] = dict(pos=put(terr[(i, j)]["pos"]))
        for nm, acc in (("roofs", roofs), ("walls", wallsW), ("plain", plain), ("trees", trees), ("lamps", lamps), ("over", over)):
            o = acc.out().get((i, j))
            if o:
                t[nm] = {a: put(o[a]) for a in o}
        tiles[f"{i}_{j}"] = t
frames = np.column_stack([car, carz, hx, hy, cam, look, v_f * 3.6, [uniq.index(n) for n in names_f]])
man = dict(outer=dict(pos=put(otri), uv=put(ouv)), world=WORLD, ts=TS, pad=PAD, nt=NT, fps=FPS, nf=NF, tiles=tiles, frames=put(frames), ncol=frames.shape[1], names=uniq,
           t_intro=T_INTRO, t_out=T_OUT, zmean=float(Zg.mean()), zedge=float(np.concatenate([Zg[0], Zg[-1], Zg[:, 0], Zg[:, -1]]).mean()),
           ground=[float(v_) for v_ in mos.reshape(-1, 3)[::97].mean(axis=0) / 255.0], mini=mini)
open(os.path.join(OUT, "world.bin"), "wb").write(b"".join(blobs))
json.dump(man, open(os.path.join(OUT, "world.json"), "w"))
ntri = sum(v_[1] // 3 for t in tiles.values() for m in t.values() for a, v_ in m.items() if a == "pos") // 3
log(f"export: {off / 1e6:.0f} Mo, {ntri} triangles")
if GAME:
    occ_m = lab > 0
    if barrieres:                                                          # terre-pleins et glissières : la voiture ne les traverse pas
        occ_m |= features.rasterize([(g_, 1) for g_ in barrieres], out_shape=lab.shape, transform=tr, fill=0, dtype="uint8") > 0
    occ = occ_m.reshape(COTE, 2, COTE, 2).any(axis=(1, 3))                 # emprise des bâtiments, 1 m par case, ligne 0 = nord
    np.savez_compressed(os.path.join(OUT, "extra.npz"), Zg=Zg.astype(np.float32), occ=occ)
    # commerces (OpenStreetMap) : une enseigne sur la façade côté rue, au-dessus du rez-de-chaussée
    PUB_AM = {"townhall", "school", "kindergarten", "college", "library"}          # lieux publics : nom affiché quand on passe devant
    COM_AM = {"restaurant", "cafe", "bar", "pub", "fast_food", "ice_cream", "pharmacy", "bank", "post_office", "fuel", "doctors", "dentist", "car_wash"}
    osm_r = ET.parse(os.path.join(D, "osm.xml")).getroot()
    osm_n = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in osm_r.iter("node")}
    btree, bgeo = STRtree([b["g"] for b in B] or [Point(0, 0)]), [b["g"] for b in B]
    shops = []
    for e in list(osm_r.iter("node")) + list(osm_r.iter("way")):
        tg = {t.get("k"): t.get("v") for t in e.findall("tag")}
        typ = tg.get("shop") or (tg.get("amenity") if tg.get("amenity") in COM_AM | PUB_AM else None) or tg.get("craft")
        if not typ or not tg.get("name"):
            continue
        if e.tag == "node":
            lo_, la_ = osm_n[e.get("id")]
        else:
            pts_ = [osm_n[n.get("ref")] for n in e.findall("nd") if n.get("ref") in osm_n]
            if not pts_:
                continue
            lo_, la_ = np.mean(pts_, axis=0)
        X_, Y_ = TR.transform(lo_, la_)
        x, y = X_ - XO, Y_ - YO
        if not (10 < x < COTE - 10 and 10 < y < COTE - 10):
            continue
        pt = Point(x, y)
        _, _, _, ax_ = bord(x, y)                                             # point sur l'axe de la rue la plus proche
        k_ = btree.nearest(pt) if bgeo else None
        bg_ = bgeo[int(k_)] if k_ is not None else None
        if bg_ is not None and bg_.distance(pt) < 25:
            ring = bg_.exterior if bg_.geom_type == "Polygon" else max(bg_.geoms, key=lambda q: q.area).exterior
            f_ = np.array(ring.interpolate(ring.project(Point(ax_))).coords[0])   # point de la façade le plus proche de la rue
            n_ = np.asarray(ax_) - f_
            if np.hypot(*n_) < 0.5:
                continue
            n_ = n_ / np.hypot(*n_)
            hb = B[k_]["eave"]
            sx, sy = f_ + n_ * 0.12
            z_ = zg(sx, sy) + min(3.3, max(2.2, hb - 0.7))
            shops.append([round(float(sx), 2), round(float(sy), 2), round(float(z_), 2), round(float(math.atan2(n_[1], n_[0])), 3), tg["name"], typ, 1, tg.get("brand") or ""])
        else:                                                                  # pas de bâtiment : totem au bord de la rue
            n_ = np.asarray(ax_) - np.array([x, y]); d_ = np.hypot(*n_)
            n_ = n_ / d_ if d_ > 0.5 else np.array([1.0, 0.0])
            shops.append([round(x, 2), round(y, 2), round(zg(x, y) + 1.6, 2), round(float(math.atan2(n_[1], n_[0])), 3), tg["name"], typ, 0, tg.get("brand") or ""])
    # deux enseignes au même endroit (plusieurs commerces à une adresse) : on les écarte le long de la façade
    for k_, s_ in enumerate(shops):
        voisins = sum(1 for q_ in shops[:k_] if math.hypot(q_[0] - s_[0], q_[1] - s_[1]) < 2.8)
        if voisins:
            dec = 3.0 * ((voisins + 1) // 2) * (1 if voisins % 2 else -1)
            s_[0] = round(s_[0] - math.sin(s_[3]) * dec, 2); s_[1] = round(s_[1] + math.cos(s_[3]) * dec, 2)
    log(f"commerces et lieux publics : {len(shops)} ({sum(s_[6] for s_ in shops)} sur une façade)")

    # panneaux de direction : entrées de ronds-points et carrefours de routes numérotées (réseau principal sur 15 km)
    RG = nx.Graph()
    for f in load("r_routes"):
        p_ = f["properties"]
        if p_.get("acces_vehicule_leger") not in (None, "Libre") or p_.get("etat_de_l_objet") not in (None, "En service"):
            continue
        g_ = shape(f["geometry"])
        for ln in (g_.geoms if hasattr(g_, "geoms") else [g_]):
            cs = [(round(c_[0] - XO, 1), round(c_[1] - YO, 1)) for c_ in ln.coords]
            if len(cs) < 2 or cs[0] == cs[-1]:
                continue
            RG.add_edge(cs[0], cs[-1], w=LineString(cs).length, rp=p_.get("nature") == "Rond-point", num=p_.get("cpx_numero") or "",
                        cl=p_.get("cpx_classement_administratif") or "", larg=float(p_.get("largeur_de_chaussee") or 6.0), pts=cs)
    villes = []
    for f in load("r_villes"):
        g_ = shape(f["geometry"])
        c_ = g_.representative_point()
        villes.append((f["properties"]["toponyme"], loc(g_), (c_.x - XO, c_.y - YO)))
    vn = list(RG.nodes)
    vxy = np.array(vn)
    v_node = [vn[int(np.argmin(np.hypot(vxy[:, 0] - c_[0], vxy[:, 1] - c_[1])))] for _, _, c_ in villes]
    classee = lambda d_: any(k in d_["cl"] for k in ("Départementale", "Nationale", "Autoroute"))

    def cap_arete(u, v):
        """Direction de départ de l'arête u -> v, mesurée 25 m après u."""
        pts_ = RG[u][v]["pts"]
        ln_ = LineString(pts_ if pts_[0] == u else pts_[::-1])
        q_ = ln_.interpolate(min(25.0, ln_.length))
        return math.atan2(q_.y - u[1], q_.x - u[0])

    groupes, vus = [], set()
    rp_sub = nx.Graph([(u, v) for u, v, d_ in RG.edges(data=True) if d_["rp"]])
    for comp in nx.connected_components(rp_sub):
        groupes.append(set(comp)); vus |= comp
    for n_ in RG.nodes:
        if n_ in vus or RG.degree(n_) < 3:
            continue
        nums = {d_["num"] for _, _, d_ in RG.edges(n_, data=True) if classee(d_) and d_["num"]}
        if len(nums) >= 2:
            groupes.append({n_})
    signs = []
    for grp in groupes:
        cx_, cy_ = np.mean(list(grp), axis=0)
        if not (40 < cx_ < COTE - 40 and 40 < cy_ < COTE - 40):
            continue
        branches = [(u, v) for u in grp for v in RG.neighbors(u) if v not in grp]
        if len(branches) < 3:
            continue
        H_ = RG.copy(); H_.remove_edges_from([(u, v) for u in grp for v in RG.neighbors(u) if v in grp]); H_.add_node("S")
        for u in grp:
            H_.add_edge("S", u, w=0.0)
        dist_, chem = nx.single_source_dijkstra(H_, "S", weight="w")
        dest = {b_: [] for b_ in branches}
        for (nom, poly, _), vnod in zip(villes, v_node):
            if vnod not in chem or poly.contains(Point(cx_, cy_)) or len(chem[vnod]) < 3:
                continue
            b_ = (chem[vnod][1], chem[vnod][2])
            if b_ in dest:
                dest[b_].append((dist_[vnod], nom))
        for (u, a_) in branches:                                         # un panneau par voie d'arrivée
            d_ = RG[u][a_]
            pts_ = d_["pts"] if d_["pts"][0] == u else d_["pts"][::-1]
            ln_ = LineString(pts_)
            if ln_.length < 30:
                continue
            p0 = np.array(ln_.interpolate(18.0).coords[0]); p1 = np.array(ln_.interpolate(17.0).coords[0])
            dv = (p1 - p0) / np.hypot(*(p1 - p0))                         # sens de marche vers le carrefour
            pos = p0 + np.array([dv[1], -dv[0]]) * (d_["larg"] / 2 + 1.8)    # bord droit
            if not (10 < pos[0] < COTE - 10 and 10 < pos[1] < COTE - 10):
                continue
            hd = math.atan2(dv[1], dv[0])
            lignes = []
            for (u2, v2) in branches:
                if (u2, v2) == (u, a_) or not dest[(u2, v2)]:
                    continue
                rel = (math.degrees(cap_arete(u2, v2) - hd) + 540) % 360 - 180
                if abs(rel) > 150:                                         # demi-tour (autre chaussée d'une route à 2 voies) : pas affiché
                    continue
                vv = sorted(dest[(u2, v2)])[:2]
                lignes.append([round(rel), RG[u2][v2]["num"], [[n2, max(1, round(d2 / 1000))] for d2, n2 in vv]])
            if lignes:
                lignes.sort(key=lambda l_: -l_[0])                          # de gauche à droite
                signs.append([round(float(pos[0]), 2), round(float(pos[1]), 2), round(zg(*pos), 2), round(math.atan2(-dv[1], -dv[0]), 3), lignes[:3]])
    # panneaux d'agglomération (EB10 entrée, EB20 sortie au dos) : là où une route du réseau principal franchit la limite d'une commune
    agglo = []
    for (u, v, d_) in RG.edges(data=True):
        ln_ = LineString(d_["pts"])
        for nom, poly, _ in villes:
            fr = ln_.intersection(poly.boundary)
            if fr.is_empty:
                continue
            for q in (fr.geoms if hasattr(fr, "geoms") else [fr]):
                if q.geom_type != "Point" or not (15 < q.x < COTE - 15 and 15 < q.y < COTE - 15):
                    continue
                t_ = ln_.project(q)
                a0 = np.array(ln_.interpolate(max(0, t_ - 3)).coords[0]); a1 = np.array(ln_.interpolate(min(ln_.length, t_ + 3)).coords[0])
                dv = (a1 - a0) / (np.hypot(*(a1 - a0)) + 1e-9)
                if poly.contains(Point(*(np.array([q.x, q.y]) - dv * 4))):
                    dv = -dv                                                  # dv : sens de l'entrée dans la commune
                pos = np.array([q.x, q.y]) + np.array([dv[1], -dv[0]]) * (d_["larg"] / 2 + 1.6)
                if all(math.hypot(pos[0] - g_[0], pos[1] - g_[1]) > 25 for g_ in agglo):
                    agglo.append([round(float(pos[0]), 2), round(float(pos[1]), 2), round(zg(*pos), 2), round(math.atan2(-dv[1], -dv[0]), 3), nom])
    log(f"panneaux d'agglomération : {len(agglo)}")
    signs.sort(key=lambda s_: -len(s_[4]))
    garde = []
    for s_ in signs:                                                      # deux panneaux à moins de 6 m : on garde le plus complet
        if all(math.hypot(s_[0] - q_[0], s_[1] - q_[1]) > 6 for q_ in garde):
            garde.append(s_)
    signs = garde
    log(f"panneaux de direction : {len(signs)} sur {len(groupes)} carrefours du réseau principal")
    json.dump(dict(named=named, gs=GS, world=WORLD, zedge=float(zedge), cote=COTE, zo=ZO, style=STYLE, volets=VOLETS, villes=[[n_, np.round(np.asarray((q_.exterior if q_.geom_type == "Polygon" else max(q_.geoms, key=lambda r_: r_.area).exterior).simplify(4).coords), 1).tolist()] for n_, p_, _ in villes for q_ in [p_.intersection(wbox.buffer(5))] if not q_.is_empty and q_.geom_type in ("Polygon", "MultiPolygon")], shops=shops, signs=signs, ponts=ponts_out, agglo=agglo, xo=XO, yo=YO,
                   mairie=([LM["Mairie"][0] - XO, LM["Mairie"][1] - YO] if "Mairie" in LM else next(([s_[0], s_[1]] for s_ in shops if s_[5] == "townhall"), [COTE / 2, COTE / 2]))), open(os.path.join(OUT, "extra.json"), "w"))
# plan de contrôle
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(mos[::4, ::4], extent=(0, WORLD, 0, WORLD))
ax.plot(Rs[:, 0], Rs[:, 1], "y-", lw=2)
ax.plot(Rs[0, 0], Rs[0, 1], "go", ms=10); ax.plot(Rs[-1, 0], Rs[-1, 1], "ro", ms=10)
for nm, (X, Y) in LM.items():
    ax.plot(X - XO, Y - YO, "c*", ms=14); ax.text(X - XO + 10, Y - YO, nm, color="w")
ax.set_xlim(200, 1700); ax.set_ylim(300, 1700); ax.axis("off")
os.makedirs(os.path.join(HERE, "out", "video"), exist_ok=True)
fig.savefig(os.path.join(HERE, "out", "video", "route.png"), dpi=70, bbox_inches="tight")
log("terminé")
