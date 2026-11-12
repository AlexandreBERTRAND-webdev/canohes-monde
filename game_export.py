"""Compacte le monde 3D (out/game/raw) pour le jeu en ligne : out/game/site/ (index.html, monde.json, photos aériennes allégées)."""
import base64, gzip, json, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("SORTIE") or os.path.join(HERE, "out", "game", "raw")
SITE = os.path.join(os.path.dirname(RAW), "site")
os.makedirs(SITE, exist_ok=True)
M = json.load(open(os.path.join(RAW, "world.json")))
X = json.load(open(os.path.join(RAW, "extra.json")))
E = np.load(os.path.join(RAW, "extra.npz"))
buf = open(os.path.join(RAW, "world.bin"), "rb").read()
arr = lambda r: np.frombuffer(buf, np.float32, count=r[1], offset=r[0])
W, H2, SC = M["world"], M["world"] / 2, 0.03
blob = bytearray()


def put(a):
    if len(blob) % 2:
        blob.append(0)
    off = len(blob)
    blob.extend(a.tobytes())
    return [off, int(a.size)]


tiles, ntri = {}, 0
for key, t in M["tiles"].items():
    o = {}
    for nm in ("roofs", "walls", "plain", "trees", "lamps", "over"):
        if nm not in t:
            continue
        pos = arr(t[nm]["pos"]).reshape(-1, 3)
        q = np.column_stack([pos[:, 0] - H2, pos[:, 2], -(pos[:, 1] - H2)]) / SC
        e = dict(pos=put(np.clip(np.round(q), -32768, 32767).astype("<i2")))
        if "uv" in t[nm]:
            e["uv"] = put(np.clip(np.round(arr(t[nm]["uv"]) * 64), -32768, 32767).astype("<i2"))
        if "col" in t[nm]:
            e["col"] = put(np.clip(np.round(arr(t[nm]["col"]) * 255), 0, 255).astype("u1"))
        o[nm] = e
        ntri += len(pos) // 3
    tiles[key] = o
Zg = E["Zg"]
HSC = 100 if Zg.max() < 640 else 20                                    # hauteurs en cm (5 cm en montagne, au-delà de 640 m de dénivelé)
hts = put(np.clip(np.round(Zg * HSC), 0, 65535).astype("<u2"))
occ = put(np.packbits(E["occ"], axis=1))
# départ : sur la rue la plus proche de la nouvelle mairie
mx, my = X.get("mairie", [686223 - 685450.0, 6172821 - 6171750.0])
best = None
for pts, k, name in X["named"]:
    p = np.asarray(pts)
    for a, b in zip(p[:-1], p[1:]):
        d = b - a; L = float(np.hypot(*d))
        if L < 2:
            continue
        u = float(np.clip(np.dot([mx - a[0], my - a[1]], d) / L / L, 0.1, 0.9))
        q = a + d * u
        dist = float(np.hypot(q[0] - mx, q[1] - my))
        if (k == 2 or not any(k_ == 2 for _, k_, _ in X["named"])) and (best is None or dist < best[0]):
            nrm = np.array([d[1], -d[0]]) / L
            best = (dist, (q + nrm * 1.4).tolist(), float(np.arctan2(d[1], d[0])), name)
names = sorted({n for _, _, n in X["named"] if n})
roads = [[pts, k, (names.index(n) if n else -1)] for pts, k, n in X["named"]]
man = dict(world=W, sc=SC, ts=M["ts"], pad=M["pad"], nt=M["nt"], gs=X["gs"], gn=int(Zg.shape[0]), zedge=X["zedge"], tiles=tiles, hts=hts, occ=occ,
           occn=X.get("cote", 1900), **({"hsc": HSC} if HSC != 100 else {}), roads=roads, names=names, ground=M["ground"], start=dict(pos=best[1], cap=best[2], rue=best[3]), mairie=[mx, my], shops=X.get("shops", []), signs=X.get("signs", []), ponts=X.get("ponts", []), agglo=X.get("agglo", []))
gz = gzip.compress(bytes(blob), 9)
json.dump(dict(man=man, gz=base64.b64encode(gz).decode("ascii")), open(os.path.join(SITE, "monde.json"), "w"))
for i in range(M["nt"]):
    for j in range(M["nt"]):
        Image.open(os.path.join(RAW, f"ortho_{i}_{j}.jpg")).resize((1024, 1024), Image.LANCZOS).save(os.path.join(SITE, f"sol_{i}_{j}.jpg"), quality=80)
Image.open(os.path.join(RAW, "ortho_outer.jpg")).resize((1600, 1600), Image.LANCZOS).save(os.path.join(SITE, "sol_alentours.jpg"), quality=78)
tpl = open(os.path.join(HERE, "game.html"), encoding="utf-8").read()
imap = {"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js", "fflate": "https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js"}}
open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(tpl.replace("__IMPORTMAP__", json.dumps(imap)))
imap = {"imports": {"three": "/nm/three/build/three.module.js", "fflate": "/nm/fflate/esm/browser.js"}}
open(os.path.join(SITE, "index_local.html"), "w", encoding="utf-8").write(tpl.replace("__IMPORTMAP__", json.dumps(imap)))
tot = sum(os.path.getsize(os.path.join(SITE, f)) for f in os.listdir(SITE) if not f.startswith("index_local"))
print(f"{ntri} triangles (hors terrain), géométrie {len(blob) / 1e6:.1f} Mo -> {len(gz) / 1e6:.1f} Mo compressés, site {tot / 1e6:.1f} Mo, départ {best[3]} à {best[0]:.0f} m de la mairie")
