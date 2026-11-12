"""Télécharge les données IGN (Géoplateforme) pour Canohès : BD TOPO (WFS) + altitudes (WMS raster)."""
import json, os, sys, time, urllib.parse, urllib.request
import numpy as np

D = os.environ.get("DONNEES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WFS = "https://data.geopf.fr/wfs/ows"
WMSR = "https://data.geopf.fr/wms-r"


def get(url, tries=7):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa
            print("  retry", i, e, file=sys.stderr)
            time.sleep(min(60, 3 * 2 ** i))                 # attente croissante : on ménage les serveurs de l IGN
    raise RuntimeError(url)


def wfs(layer, bbox=None, cql=None, name=None):
    feats, start = [], 0
    while True:
        p = dict(SERVICE="WFS", VERSION="2.0.0", REQUEST="GetFeature", TYPENAMES=layer,
                 OUTPUTFORMAT="application/json", SRSNAME="EPSG:2154", COUNT=5000, STARTINDEX=start)
        if bbox:
            p["BBOX"] = ",".join(str(v) for v in bbox) + ",EPSG:2154"
        if cql:
            p["CQL_FILTER"] = cql
        js = json.loads(get(WFS + "?" + urllib.parse.urlencode(p)))
        f = js.get("features", [])
        feats += f
        if len(f) < 5000:
            break
        start += 5000
    out = os.path.join(D, (name or layer.split(":")[1]) + ".geojson")
    json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
    print(layer, len(feats))
    return feats


def raster(layer, bbox, res, name):
    """bbox L93 (xmin,ymin,xmax,ymax), res en m/pixel. Tuile par blocs de 2000 px."""
    xmin, ymin, xmax, ymax = bbox
    W, H = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    arr = np.zeros((H, W), np.float32)
    T = 2000
    for j0 in range(0, H, T):
        for i0 in range(0, W, T):
            w, h = min(T, W - i0), min(T, H - j0)
            bx0, bx1 = xmin + i0 * res, xmin + (i0 + w) * res
            by1 = ymax - j0 * res
            by0 = by1 - h * res
            p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS=layer, STYLES="",
                     CRS="EPSG:2154", BBOX=f"{bx0},{by0},{bx1},{by1}", WIDTH=w, HEIGHT=h,
                     FORMAT="image/x-bil;bits=32")
            raw = get(WMSR + "?" + urllib.parse.urlencode(p))
            if len(raw) != w * h * 4:
                raise RuntimeError(f"{layer}: réponse inattendue {len(raw)} octets: {raw[:300]!r}")
            arr[j0:j0 + h, i0:i0 + w] = np.frombuffer(raw, "<f4").reshape(h, w)
    np.save(os.path.join(D, name + ".npy"), arr)
    v = arr[arr > -1000]
    print(layer, arr.shape, "min/max", float(v.min()) if v.size else None, float(v.max()) if v.size else None,
          "nodata", int((arr <= -1000).sum()))
    return arr


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "commune":
        wfs("BDTOPO_V3:commune", cql="code_insee='66038'")
    elif step == "vector":
        bbox = [float(v) for v in sys.argv[2:6]]
        for lay in ["batiment", "troncon_de_route", "zone_de_vegetation", "troncon_de_voie_ferree",
                    "construction_surfacique", "construction_ponctuelle", "cimetiere", "terrain_de_sport",
                    "zone_d_activite_ou_d_interet", "surface_hydrographique", "construction_lineaire",
                    "toponymie_bati", "reservoir", "pylone"]:
            try:
                wfs("BDTOPO_V3:" + lay, bbox=bbox)
            except Exception as e:
                print("ECHEC", lay, e)
    elif step == "raster":
        bbox = [float(v) for v in sys.argv[2:6]]
        raster("IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", bbox, 1.0, "mnt")
        raster("IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93", bbox, 0.5, "mns")
        json.dump(dict(bbox=bbox), open(os.path.join(D, "raster_bbox.json"), "w"))
