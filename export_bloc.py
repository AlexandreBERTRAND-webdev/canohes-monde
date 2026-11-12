"""Compacte un bloc de 2 km pour le monde ouvert : <sortie>/<ix>_<iy>/m.json (manifeste), g.bin (géométrie gzip), t.jpg (teinte du sol).
Usage : SORTIE=<dossier raw du générateur> python export_bloc.py <dossier des blocs> ix iy"""
import gzip, json, os, shutil, sys
import numpy as np

RAW = os.environ["SORTIE"]
DEST = os.path.join(sys.argv[1], f"{sys.argv[2]}_{sys.argv[3]}")
os.makedirs(DEST, exist_ok=True)
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
        if nm not in t or "pos" not in t[nm]:
            continue
        pos = arr(t[nm]["pos"]).reshape(-1, 3)
        if not len(pos):
            continue
        g = np.column_stack([pos[:, 0] - H2, pos[:, 2], -(pos[:, 1] - H2)])           # repère three.js du bloc, centré
        e = {"pos": put(np.clip(np.round(g / SC), -32767, 32767).astype("<i2"))}
        if "uv" in t[nm]:
            e["uv"] = put(np.clip(np.round(arr(t[nm]["uv"]) * 64), -32767, 32767).astype("<i2"))
        if "col" in t[nm]:
            e["col"] = put(np.clip(np.round(arr(t[nm]["col"]) * 255), 0, 255).astype("u1"))
        o[nm] = e
        ntri += len(pos) // 3
    tiles[key] = o
Zg = E["Zg"]
HSC = 100 if Zg.max() < 640 else 20
hts = put(np.clip(np.round(Zg * HSC), 0, 65535).astype("<u2"))
occ = put(np.packbits(E["occ"], axis=1))
roads = [[np.round(np.asarray(p), 1).tolist(), k, n] for p, k, n in X["named"]]
man = dict(v=1, world=W, sc=SC, ts=M["ts"], nt=M["nt"], gs=X["gs"], gn=int(Zg.shape[0]), hsc=HSC, zo=X["zo"], occn=X["cote"],
           tiles=tiles, hts=hts, occ=occ, roads=roads, shops=X.get("shops", []), signs=X.get("signs", []), ponts=X.get("ponts", []), agglo=X.get("agglo", []))
json.dump(man, open(os.path.join(DEST, "m.json"), "w"), separators=(",", ":"), ensure_ascii=False)
open(os.path.join(DEST, "g.bin"), "wb").write(gzip.compress(bytes(blob), 9, mtime=0))
t_src = os.path.join(RAW, "teinte.jpg")
if os.path.exists(t_src):
    shutil.copy(t_src, os.path.join(DEST, "t.jpg"))
tot = sum(os.path.getsize(os.path.join(DEST, f)) for f in os.listdir(DEST))
print(f"bloc {sys.argv[2]}_{sys.argv[3]} : {ntri} triangles, {len(blob) / 1e6:.1f} Mo -> {tot / 1e6:.2f} Mo")
