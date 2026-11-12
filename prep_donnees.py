"""Télécharge dans data/ tout ce dont video_world.py a besoin pour un carré Lambert-93 de 1 900 m.
Usage : python prep_donnees.py XO YO [COTE]   (coin sud-ouest ; Canohès : 685450 6171750)"""
import json, os, sys, time, urllib.parse
from pyproj import Transformer
import fetch

XO, YO = float(sys.argv[1]), float(sys.argv[2])
COTE, MARGE = float(sys.argv[3]) if len(sys.argv) > 3 else 1900.0, 150.0
bb = [XO, YO, XO + COTE, YO + COTE]
bbm = [XO - MARGE, YO - MARGE, XO + COTE + MARGE, YO + COTE + MARGE]
os.makedirs(fetch.D, exist_ok=True)
t0 = time.time()
json.dump(dict(bbox=bb), open(os.path.join(fetch.D, "v_bbox.json"), "w"))
fetch.wfs("BDTOPO_V3:batiment", bbox=bbm)
fetch.wfs("BDTOPO_V3:troncon_de_route", bbox=bbm)
fetch.wfs("CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle", bbox=bbm, name="v_cadastre")
print(f"vecteurs : {time.time() - t0:.1f} s", flush=True)
fetch.raster("IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", bb, 1.0, "v_mnt")
fetch.raster("IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", bb, 0.5, "v_mns")
print(f"laser : {time.time() - t0:.1f} s", flush=True)
T = Transformer.from_crs(2154, 4326, always_xy=True)
lo0, la0 = T.transform(bbm[0], bbm[1]); lo1, la1 = T.transform(bbm[2], bbm[3])
xml = fetch.get("https://api.openstreetmap.org/api/0.6/map?" + urllib.parse.urlencode(dict(bbox=f"{lo0},{la0},{lo1},{la1}")))
open(os.path.join(fetch.D, "osm.xml"), "wb").write(xml)
print(f"OpenStreetMap : {len(xml) // 1024} ko ; total {time.time() - t0:.1f} s", flush=True)
