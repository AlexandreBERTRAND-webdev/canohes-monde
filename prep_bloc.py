"""Données d'un bloc (ix, iy) de 2 km : BD TOPO, cadastre, relief laser avec marge, points OpenStreetMap et réseau régional
pris dans les données communes du département. Usage : DONNEES=<dossier du bloc> python prep_bloc.py <dossier commun> ix iy"""
import json, os, sys, time
import numpy as np
import fetch
from pyproj import Transformer
from shapely.geometry import shape, box as sbox
from xml.sax.saxutils import quoteattr

COM, ix, iy = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
COTE, MARGE, MV, MR = 2000, 40, 150, 20000
XO, YO = ix * COTE, iy * COTE
bb = [XO, YO, XO + COTE, YO + COTE]
bbm = [XO - MV, YO - MV, XO + COTE + MV, YO + COTE + MV]
os.makedirs(fetch.D, exist_ok=True)
t0 = time.time()
DEPT = os.environ.get("DEPT", "66")                                       # département du bloc (style régional)
json.dump(dict(bbox=bb, marge=MARGE, dep=DEPT), open(os.path.join(fetch.D, "v_bbox.json"), "w"))
for lay in ("BDTOPO_V3:batiment", "BDTOPO_V3:troncon_de_route", "BDTOPO_V3:equipement_de_transport", "BDTOPO_V3:commune"):
    fetch.wfs(lay, bbox=bbm)
fetch.wfs("CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle", bbox=bbm, name="v_cadastre")
bbr = [XO - MARGE, YO - MARGE, XO + COTE + MARGE, YO + COTE + MARGE]
mnt = fetch.raster("IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", bbr, 1.0, "v_mnt")
mns = fetch.raster("IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", bbr, 0.5, "v_mns")
trou = (mnt < -50) | ~np.isfinite(mnt)                                 # sous -50 m : absence de données (bord de mer), le département n'a rien de plus bas
if trou.mean() > 0.002:                                                    # trou du relevé laser : relief de secours (RGE ALTI)
    sec = fetch.raster("ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", bbr, 1.0, "v_secours")
    mnt[trou] = sec[trou]
    mnt[(mnt < -50) | ~np.isfinite(mnt)] = 0.0                           # mer : niveau zéro
    big = np.repeat(np.repeat(trou, 2, 0), 2, 1)
    mns[big] = np.repeat(np.repeat(mnt, 2, 0), 2, 1)[big]
    np.save(os.path.join(fetch.D, "v_mnt.npy"), mnt); np.save(os.path.join(fetch.D, "v_mns.npy"), mns)
    print(f"relief de secours sur {trou.mean() * 100:.1f} % du bloc")
# OpenStreetMap : points du département dans le cadre du bloc
T = Transformer.from_crs(4326, 2154, always_xy=True)
pts = json.load(open(os.path.join(COM, "osm_points.json")))
xy = np.array(T.transform([p[0] for p in pts], [p[1] for p in pts])).T if pts else np.zeros((0, 2))
ok = (xy[:, 0] > bbm[0]) & (xy[:, 0] < bbm[2]) & (xy[:, 1] > bbm[1]) & (xy[:, 1] < bbm[3]) if len(xy) else []
with open(os.path.join(fetch.D, "osm.xml"), "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm>\n')
    for k, p in enumerate(pts):
        if ok[k]:
            f.write(f'<node id="{k}" lon="{p[0]}" lat="{p[1]}">' + "".join(f"<tag k={quoteattr(a)} v={quoteattr(b)}/>" for a, b in p[2].items()) + "</node>\n")
    f.write("</osm>\n")
# réseau régional et communes à 20 km (panneaux de direction)
zone = sbox(XO - MR, YO - MR, XO + COTE + MR, YO + COTE + MR)
for nom in ("r_routes", "r_villes"):
    fc = json.load(open(os.path.join(COM, nom + ".geojson")))
    fc["features"] = [f for f in fc["features"] if shape(f["geometry"]).intersects(zone)]
    json.dump(fc, open(os.path.join(fetch.D, nom + ".geojson"), "w"))
print(f"données du bloc {ix}_{iy} : {time.time() - t0:.0f} s")
