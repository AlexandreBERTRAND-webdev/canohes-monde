"""Données communes à tout le département, téléchargées une fois : réseau routier principal et communes (BD TOPO),
points d'intérêt OpenStreetMap (extrait Geofabrik). Usage : python prep_dept.py <dossier>"""
import json, os, sys, time, urllib.request
import fetch

DEST = sys.argv[1]
os.makedirs(DEST, exist_ok=True)
fetch.D = DEST
dep = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dept", "departement66.geojson")))
from shapely.geometry import shape
x0, y0, x1, y1 = shape(dep["geometry"]).bounds
R = 15000
box = f"BBOX(geometrie,{x0 - R},{y0 - R},{x1 + R},{y1 + R},'EPSG:2154')"
t0 = time.time()
fetch.wfs("BDTOPO_V3:troncon_de_route", cql=f"importance IN ('1','2','3','4') AND {box}", name="r_routes")
fetch.wfs("BDTOPO_V3:zone_d_habitation", cql=f"importance = '1' AND nature = 'Lieu-dit habité' AND {box}", name="r_villes")
print(f"BD TOPO région : {time.time() - t0:.0f} s", flush=True)

# OpenStreetMap : extrait régional Geofabrik, on ne garde que les points utiles au jeu (dans le cadre du département + 3 km)
PBF = os.path.join(DEST, "region.osm.pbf")
if not os.path.exists(PBF):
    urllib.request.urlretrieve("https://download.geofabrik.de/europe/france/languedoc-roussillon-latest.osm.pbf", PBF)
import osmium
from pyproj import Transformer
T = Transformer.from_crs(2154, 4326, always_xy=True)
lo0, la0 = T.transform(x0 - 3000, y0 - 3000); lo1, la1 = T.transform(x1 + 3000, y1 + 3000)
CLES = ("shop", "amenity", "craft", "office", "tourism", "barrier", "traffic_sign", "brand", "name")
PT_HW = {"traffic_signals", "stop", "give_way", "crossing", "bus_stop", "street_lamp", "motorway_junction"}
pts = []


class H(osmium.SimpleHandler):
    def node(self, n):
        t = n.tags
        if not (lo0 < n.location.lon < lo1 and la0 < n.location.lat < la1):
            return
        if t.get("highway") in PT_HW or any(k in t for k in ("shop", "amenity", "craft", "barrier", "traffic_sign")):
            pts.append([round(n.location.lon, 7), round(n.location.lat, 7), {k.k: k.v for k in t}])

    def way(self, w):
        t = w.tags
        if not ("name" in t and any(k in t for k in ("shop", "amenity", "craft"))):
            return
        try:
            ls = [(nd.lon, nd.lat) for nd in w.nodes]
        except osmium.InvalidLocationError:
            return
        lo, la = sum(p[0] for p in ls) / len(ls), sum(p[1] for p in ls) / len(ls)
        if lo0 < lo < lo1 and la0 < la < la1:
            pts.append([round(lo, 7), round(la, 7), {k.k: k.v for k in t}])


H().apply_file(PBF, locations=True, idx="flex_mem")
json.dump(pts, open(os.path.join(DEST, "osm_points.json"), "w"), ensure_ascii=False)
os.remove(PBF)
print(f"OpenStreetMap : {len(pts)} points ; total {time.time() - t0:.0f} s")
