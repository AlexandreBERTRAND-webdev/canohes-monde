"""Réseau routier principal et communes autour d'un carré, pour les panneaux de direction.
Usage : DONNEES=... python prep_region.py   (lit v_bbox.json, prend 15 km autour)"""
import json, os, time, urllib.parse
import fetch

bb = json.load(open(os.path.join(fetch.D, "v_bbox.json")))["bbox"]
R = 15000
x0, y0, x1, y1 = bb[0] - R, bb[1] - R, bb[2] + R, bb[3] + R
box = f"BBOX(geometrie,{x0},{y0},{x1},{y1},'EPSG:2154')"
t0 = time.time()
fetch.wfs("BDTOPO_V3:troncon_de_route", cql=f"importance IN ('1','2','3','4') AND {box}", name="r_routes")
fetch.wfs("BDTOPO_V3:zone_d_habitation", cql=f"importance = '1' AND nature = 'Lieu-dit habité' AND {box}", name="r_villes")
print(f"région : {time.time() - t0:.1f} s")
