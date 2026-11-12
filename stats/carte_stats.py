#!/home/alex/projets/canohes-monde/.venv/bin/python
# -*- coding: utf-8 -*-
"""Carte des balades du jeu « Canohès, balade libre » (demande d'alex du 12/11/2026, « go stats »).

Lit les événements anonymes envoyés par le jeu à Umami (flux.customs-toolbox.com, site « Canohès balade »),
les agrège par bloc de 2 km (grille Lambert-93 : bloc ix_iy = carré de coin sud-ouest ix*2000, iy*2000 en EPSG:2154),
puis produit :
  public/index.html   page autonome : carte Leaflet (tuiles OpenStreetMap), totaux, palmarès des blocs
  public/apercu.png   capture de la carte (navigateur sans écran), pour la fiche de la tour
  donnees.json        les agrégats bruts
et met à jour la fiche « Carte des balades » de la tour Sacha (publier_tour.py), sans mail.

Événements attendus (umami.track) :
  partie {bloc}        au démarrage
  bloc   {bloc}        à chaque entrée dans un nouveau bloc
  km     {bloc, total} à chaque kilomètre parcouru
Passages d'un bloc = parties démarrées dedans + entrées dans le bloc. Km d'un bloc = nombre d'événements km dedans.
Rien de personnel : Umami ne pose pas de cookie, et seules des sommes par bloc sortent d'ici.

Usage : carte_stats.py [--sans-tour] [--sans-apercu]
Les identifiants Umami sont lus dans ~/.config/sacha/umami.identifiants (jamais affichés ni recopiés)."""
import base64
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

from pyproj import Transformer

ICI = Path(__file__).resolve().parent
PUBLIC = ICI / "public"
CACHE = ICI / "cache-jours.json"
DONNEES = ICI / "donnees.json"
VENDOR = ICI / "vendor"
IDENTIFIANTS = Path.home() / ".config/sacha/umami.identifiants"
SITE_ID = "629d4c48-e3dc-4308-9f97-c43316c2f989"   # site « Canohès balade » créé le 12/11/2026
URL_UMAMI = "https://flux.customs-toolbox.com"
PARIS = ZoneInfo("Europe/Paris")
TAILLE_BLOC = 2000
LIMITE_UMAMI = 100        # Umami 3.4 renvoie au plus 100 valeurs par requête event-data/values
BLOC_OK = re.compile(r"^-?\d{1,4}_-?\d{1,5}$")
NAVIGATEUR = Path.home() / ".cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell"
TOUR_PY = "/home/alex/projets/master/sacha/tour/api/venv/bin/python"


def journal(msg):
    print(f"[{datetime.now(PARIS):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ---------- Umami ----------

def _lire_identifiants() -> dict:
    try:
        brut = IDENTIFIANTS.read_text(encoding="utf-8")
    except PermissionError:   # fichier rangé en 600 sous un autre compte : lecture par sudo, sans écho
        brut = subprocess.run(["sudo", "-n", "cat", str(IDENTIFIANTS)], capture_output=True, text=True, check=True).stdout
    vals = {}
    for ligne in brut.splitlines():
        if "=" in ligne and not ligne.lstrip().startswith("#"):
            k, v = ligne.split("=", 1)
            vals[k.strip()] = v.strip()
    return vals


class Umami:
    def __init__(self):
        ident = _lire_identifiants()
        self.base = (ident.get("url") or URL_UMAMI).rstrip("/")
        corps = json.dumps({"username": ident.get("identifiant"), "password": ident.get("mot_de_passe")}).encode()
        del ident
        req = urllib.request.Request(self.base + "/api/auth/login", data=corps, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            self._jeton = json.load(r)["token"]

    def get(self, chemin: str, **params):
        url = f"{self.base}/api/websites/{SITE_ID}/{chemin}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._jeton}", "Accept": "application/json"})
        for essai in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.load(r)
            except Exception:
                if essai == 2:
                    raise
                time.sleep(2 + 3 * essai)

    def site(self):
        req = urllib.request.Request(f"{self.base}/api/websites/{SITE_ID}", headers={"Authorization": f"Bearer {self._jeton}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)

    def valeurs(self, evenement: str, debut_ms: int, fin_ms: int) -> tuple[dict, bool]:
        """Compte par valeur de la propriété « bloc » pour un événement, en coupant la période en deux
        tant qu'Umami atteint sa limite de 100 valeurs. Renvoie (comptes, tronque)."""
        rep = self.get("event-data/values", startAt=debut_ms, endAt=fin_ms, event=evenement, propertyName="bloc",
                       timezone="Europe/Paris")
        lignes = rep if isinstance(rep, list) else rep.get("data", [])
        if len(lignes) < LIMITE_UMAMI:
            return {str(l["value"]): int(l["total"]) for l in lignes}, False
        if fin_ms - debut_ms <= 60_000:
            return {str(l["value"]): int(l["total"]) for l in lignes}, True
        milieu = (debut_ms + fin_ms) // 2
        a, ta = self.valeurs(evenement, debut_ms, milieu)
        b, tb = self.valeurs(evenement, milieu + 1, fin_ms)
        for k, v in b.items():
            a[k] = a.get(k, 0) + v
        return a, ta or tb


# ---------- agrégation ----------

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _jour(u: Umami, debut: datetime, fin: datetime) -> dict:
    d, f = _ms(debut), _ms(fin) - 1
    out, tronque = {}, False
    for ev in ("partie", "bloc", "km"):
        out[ev], t = u.valeurs(ev, d, f)
        tronque = tronque or t
    out["tronque"] = tronque
    return out


def collecter(u: Umami) -> tuple[dict, dict]:
    site = u.site()
    cree = datetime.fromisoformat(site["createdAt"].replace("Z", "+00:00"))
    maintenant = datetime.now(timezone.utc)
    try:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    jours = {}
    jour = cree.astimezone(PARIS).date()
    while True:
        debut = datetime(jour.year, jour.month, jour.day, tzinfo=PARIS)
        if debut > maintenant:
            break
        fin = debut + timedelta(days=1)
        cle = jour.isoformat()
        # un jour fini depuis plus de 2 h ne bouge plus : on le garde en cache
        if cle in cache and fin < maintenant - timedelta(hours=2):
            jours[cle] = cache[cle]
        else:
            jours[cle] = _jour(u, debut, min(fin, maintenant + timedelta(minutes=1)))
            if fin < maintenant - timedelta(hours=2):
                cache[cle] = jours[cle]
        jour += timedelta(days=1)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    sess = u.get("sessions/stats", startAt=_ms(cree - timedelta(minutes=1)), endAt=_ms(maintenant + timedelta(minutes=1)),
                 timezone="Europe/Paris")
    meta = {"cree": cree.isoformat(), "joueurs": int((sess.get("visitors") or {}).get("value") or 0),
            "visites": int((sess.get("visits") or {}).get("value") or 0)}
    return jours, meta


def agreger(jours: dict, meta: dict) -> dict:
    blocs, ignores = {}, {"partie": 0, "bloc": 0, "km": 0}
    parties = km = 0
    tronque = False
    par_jour = []
    for cle in sorted(jours):
        j = jours[cle]
        tronque = tronque or j.get("tronque", False)
        pj = kj = 0
        for ev in ("partie", "bloc", "km"):
            for b, n in (j.get(ev) or {}).items():
                if not BLOC_OK.match(b):          # « test_test » ou valeur inattendue : jamais sur la carte
                    ignores[ev] += n
                    continue
                e = blocs.setdefault(b, {"passages": 0, "km": 0, "parties": 0})
                if ev == "km":
                    e["km"] += n
                    kj += n
                else:
                    e["passages"] += n
                    if ev == "partie":
                        e["parties"] += n
                        pj += n
        parties += pj
        km += kj
        par_jour.append({"jour": cle, "parties": pj, "km": kj})
    return {"genere": datetime.now(PARIS).isoformat(timespec="minutes"), "parties": parties, "km": km,
            "joueurs": meta["joueurs"], "visites": meta["visites"], "depuis": meta["cree"], "blocs": blocs,
            "nb_blocs": len(blocs), "ignores": ignores, "tronque": tronque, "par_jour": par_jour}


def geometrie(blocs: dict) -> list:
    tr = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    out = []
    for b, e in blocs.items():
        ix, iy = (int(x) for x in b.split("_"))
        x0, y0 = ix * TAILLE_BLOC, iy * TAILLE_BLOC
        coins = [(x0, y0), (x0 + TAILLE_BLOC, y0), (x0 + TAILLE_BLOC, y0 + TAILLE_BLOC), (x0, y0 + TAILLE_BLOC)]
        poly = []
        for x, y in coins:
            lon, lat = tr.transform(x, y)
            poly.append([round(lat, 6), round(lon, 6)])
        out.append({"bloc": b, "p": e["passages"], "km": e["km"], "parties": e["parties"], "poly": poly})
    out.sort(key=lambda r: (-r["p"], -r["km"], r["bloc"]))
    return out


# ---------- page HTML ----------

def _n(x) -> str:
    return f"{x:,}".replace(",", " ")


def page(stats: dict, geo: list) -> str:
    gen = datetime.fromisoformat(stats["genere"])
    depuis = datetime.fromisoformat(stats["depuis"]).astimezone(PARIS)
    palmares = "".join(
        f"<tr><td>{i}</td><td>{html.escape(g['bloc'])}</td><td>{_n(g['p'])}</td><td>{_n(g['km'])}</td></tr>"
        for i, g in enumerate(geo[:15], 1)) or '<tr><td colspan="4">Aucun bloc visité pour l\'instant.</td></tr>'
    avert = ""
    if stats["tronque"]:
        avert = '<p class="note">Attention : sur au moins une minute, plus de 100 blocs différents ont été vus et Umami a tronqué la liste.</p>'
    css = (VENDOR / "leaflet.min.css").read_text(encoding="utf-8")
    js = (VENDOR / "leaflet.min.js").read_text(encoding="utf-8")
    donnees = json.dumps(geo, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Carte des balades</title>
<style>{css}</style>
<style>
:root {{
  --fond: #f6f5f1; --surface: #ffffff; --texte: #1d2321; --doux: #5d6763; --bord: #dcdcd4; --accent: #2f6f5e;
  --c1: #fde9b5; --c2: #f9c46b; --c3: #ef8b3b; --c4: #d4512a; --c5: #8f1d21; --trait: #5a1a10;
  --tuiles: none;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --fond: #121615; --surface: #1b211f; --texte: #e8ece9; --doux: #9aa6a1; --bord: #2c3532; --accent: #7cc4ad;
    --c1: #5b4a1f; --c2: #8a6420; --c3: #c07a25; --c4: #e8963a; --c5: #ffd27a; --trait: #ffe6b0;
    --tuiles: invert(1) hue-rotate(180deg) brightness(.9) contrast(.9);
  }}
}}
:root[data-theme="dark"] {{
  --fond: #121615; --surface: #1b211f; --texte: #e8ece9; --doux: #9aa6a1; --bord: #2c3532; --accent: #7cc4ad;
  --c1: #5b4a1f; --c2: #8a6420; --c3: #c07a25; --c4: #e8963a; --c5: #ffd27a; --trait: #ffe6b0;
  --tuiles: invert(1) hue-rotate(180deg) brightness(.9) contrast(.9);
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; background: var(--fond); color: var(--texte);
  font: 16px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 16px; }}
h1 {{ font-size: 1.4rem; margin: 0 0 4px; }}
.sous {{ color: var(--doux); margin: 0 0 16px; font-size: .92rem; }}
.totaux {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }}
.tuile {{ background: var(--surface); border: 1px solid var(--bord); border-radius: 10px; padding: 12px; }}
.tuile b {{ display: block; font-size: 1.6rem; font-variant-numeric: tabular-nums; }}
.tuile span {{ color: var(--doux); font-size: .85rem; }}
#carte {{ height: min(70vh, 620px); border-radius: 10px; border: 1px solid var(--bord); background: var(--surface); }}
.leaflet-tile-pane {{ filter: var(--tuiles); }}
.legende {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 10px 0 20px; font-size: .85rem; color: var(--doux); }}
.legende i {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 4px; vertical-align: -2px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--bord); border-radius: 10px; overflow: hidden; }}
th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--bord); font-variant-numeric: tabular-nums; }}
th {{ color: var(--doux); font-weight: 600; font-size: .85rem; }}
.note {{ color: var(--doux); font-size: .85rem; }}
body.apercu main {{ max-width: none; padding: 0; }}
body.apercu .masque-apercu {{ display: none; }}
body.apercu #carte {{ height: 100vh; border: 0; border-radius: 0; }}
</style>
</head>
<body>
<main>
<div class="masque-apercu">
<h1>Carte des balades</h1>
<p class="sous">Canohès, balade libre. Tout ce qui a été parcouru depuis le {depuis:%d/%m/%Y}, mis à jour le {gen:%d/%m/%Y à %H:%M}.</p>
<div class="totaux">
  <div class="tuile"><b>{_n(stats['parties'])}</b><span>parties</span></div>
  <div class="tuile"><b>{_n(stats['joueurs'])}</b><span>joueurs uniques (Umami)</span></div>
  <div class="tuile"><b>{_n(stats['km'])}</b><span>km parcourus</span></div>
  <div class="tuile"><b>{_n(stats['nb_blocs'])}</b><span>blocs différents</span></div>
</div>
</div>
<div id="carte" role="img" aria-label="Carte des blocs de 2 km visités"></div>
<div class="masque-apercu">
<div class="legende" id="legende"></div>
<h2>Palmarès des blocs</h2>
<table><thead><tr><th>#</th><th>Bloc</th><th>Passages</th><th>km</th></tr></thead><tbody>{palmares}</tbody></table>
{avert}
<p class="note">Un bloc est un carré de 2 km sur la grille Lambert-93. Passages : parties démarrées dans le bloc et entrées dans le bloc.
Mesure anonyme, sans cookie (Umami auto-hébergé) ; « joueurs uniques » est l'estimation d'Umami, qui renouvelle son empreinte anonyme chaque mois.</p>
</div>
</main>
<script>{js}</script>
<script>
(function () {{
  var donnees = {donnees};
  var apercu = /[?&]apercu=1/.test(location.search);
  if (apercu) document.body.classList.add("apercu");
  var css = getComputedStyle(document.documentElement);
  var tok = function (n) {{ return css.getPropertyValue(n).trim(); }};
  var seuils = [1, 2, 5, 10, 25];
  var couleurs = ["--c1", "--c2", "--c3", "--c4", "--c5"].map(tok);
  function couleur(p) {{ var i = 0; for (var k = 0; k < seuils.length; k++) if (p >= seuils[k]) i = k; return couleurs[i]; }}
  var carte = L.map("carte", {{ zoomControl: !apercu, attributionControl: true, preferCanvas: true }});
  L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
    maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">contributeurs OpenStreetMap</a>'
  }}).addTo(carte);
  var groupe = L.featureGroup();
  donnees.forEach(function (d) {{
    L.polygon(d.poly, {{ color: tok("--trait"), weight: 1, fillColor: couleur(d.p), fillOpacity: .62 }})
      .bindTooltip("Bloc " + d.bloc + " : " + d.p + " passage" + (d.p > 1 ? "s" : "") + ", " + d.km + " km")
      .addTo(groupe);
  }});
  groupe.addTo(carte);
  if (donnees.length) carte.fitBounds(groupe.getBounds(), {{ padding: [30, 30], maxZoom: 13 }});
  else carte.fitBounds([[42.33, 1.72], [42.92, 3.18]]);   /* Pyrénées-Orientales */
  var leg = document.getElementById("legende");
  if (leg) {{
    var lib = ["1", "2 à 4", "5 à 9", "10 à 24", "25 et plus"];
    leg.innerHTML = "Passages : " + lib.map(function (t, i) {{
      return '<span><i style="background:' + couleurs[i] + '"></i>' + t + "</span>"; }}).join("");
  }}
}})();
</script>
</body>
</html>
"""


# ---------- aperçu PNG ----------

class _Muet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def apercu(sortie: Path) -> bool:
    if not NAVIGATEUR.exists():
        journal("aperçu : navigateur sans écran introuvable, aperçu ignoré")
        return False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Muet, directory=str(PUBLIC)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    brut = sortie.with_suffix(".brut.png")
    try:
        with tempfile.TemporaryDirectory() as profil:
            subprocess.run([str(NAVIGATEUR), "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                            f"--user-data-dir={profil}", "--window-size=900,700", "--virtual-time-budget=15000",
                            f"--screenshot={brut}", f"http://127.0.0.1:{srv.server_port}/index.html?apercu=1"],
                           capture_output=True, timeout=90)
    finally:
        srv.shutdown()
    if not brut.exists():
        journal("aperçu : capture ratée")
        return False
    subprocess.run(["convert", str(brut), "-strip", "-quality", "82", str(sortie.with_suffix(".jpg"))], check=False)
    brut.replace(sortie)
    return True


# ---------- fiche de la tour ----------

def html_fiche(stats: dict, geo: list, image: Path | None) -> str:
    gen = datetime.fromisoformat(stats["genere"])
    depuis = datetime.fromisoformat(stats["depuis"]).astimezone(PARIS)
    lignes = [f"<p>Mise à jour automatique du {gen:%d/%m/%Y à %H:%M}. Données depuis le {depuis:%d/%m/%Y}.</p>",
              "<table><thead><tr><th>Mesure</th><th>Total</th></tr></thead><tbody>",
              f"<tr><td>Parties</td><td>{_n(stats['parties'])}</td></tr>",
              f"<tr><td>Joueurs uniques (estimation Umami)</td><td>{_n(stats['joueurs'])}</td></tr>",
              f"<tr><td>Kilomètres parcourus</td><td>{_n(stats['km'])}</td></tr>",
              f"<tr><td>Blocs différents visités</td><td>{_n(stats['nb_blocs'])}</td></tr>",
              "</tbody></table>"]
    jpg = image.with_suffix(".jpg") if image else None
    if jpg and jpg.exists():
        b64 = base64.b64encode(jpg.read_bytes()).decode()
        lignes.append(f'<h2>Carte</h2><p><img src="data:image/jpeg;base64,{b64}" alt="Carte des blocs visités" width="900"></p>')
        lignes.append("<p>Plus le carré est foncé, plus le bloc a été traversé. Fond de carte : © contributeurs OpenStreetMap.</p>")
    elif not geo:
        lignes.append("<p>Aucune balade enregistrée pour l'instant.</p>")
    lignes.append("<h2>Palmarès des blocs</h2><table><thead><tr><th>#</th><th>Bloc</th><th>Passages</th><th>km</th></tr></thead><tbody>")
    if geo:
        lignes += [f"<tr><td>{i}</td><td>{html.escape(g['bloc'])}</td><td>{_n(g['p'])}</td><td>{_n(g['km'])}</td></tr>"
                   for i, g in enumerate(geo[:20], 1)]
    else:
        lignes.append("<tr><td colspan=\"4\">Aucun bloc visité pour l'instant.</td></tr>")
    lignes.append("</tbody></table>")
    derniers = [j for j in stats["par_jour"] if j["parties"] or j["km"]][-14:]
    if derniers:
        lignes.append("<h2>Par jour</h2><table><thead><tr><th>Jour</th><th>Parties</th><th>km</th></tr></thead><tbody>")
        lignes += [f"<tr><td>{datetime.fromisoformat(j['jour']):%d/%m/%Y}</td><td>{_n(j['parties'])}</td><td>{_n(j['km'])}</td></tr>"
                   for j in reversed(derniers)]
        lignes.append("</tbody></table>")
    if stats["tronque"]:
        lignes.append("<p>Attention : Umami a tronqué une liste (plus de 100 blocs différents en une minute).</p>")
    lignes.append("<p>Un bloc = un carré de 2 km sur la grille Lambert-93. Mesure anonyme, sans cookie (Umami auto-hébergé, "
                  "site « Canohès balade » sur flux.customs-toolbox.com). La page interactive (carte Leaflet zoomable) est "
                  "régénérée en même temps sur le serveur.</p>")
    return "\n".join(lignes)


def publier_tour(stats: dict, contenu: str) -> dict:
    resume = (f"{_n(stats['parties'])} parties, {_n(stats['joueurs'])} joueurs, {_n(stats['km'])} km, "
              f"{_n(stats['nb_blocs'])} blocs visités (mis à jour chaque nuit)")
    corps = json.dumps({"titre": "Canohès balade : carte des balades", "resume": resume, "html": contenu})
    r = subprocess.run([TOUR_PY, str(ICI / "publier_tour.py")], input=corps, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError("publication dans la tour ratée : " + (r.stderr or r.stdout)[-500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    sans_tour = "--sans-tour" in sys.argv
    sans_apercu = "--sans-apercu" in sys.argv
    u = Umami()
    jours, meta = collecter(u)
    stats = agreger(jours, meta)
    geo = geometrie(stats["blocs"])
    PUBLIC.mkdir(exist_ok=True)
    DONNEES.write_text(json.dumps({k: v for k, v in stats.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp = PUBLIC / "index.html.tmp"
    tmp.write_text(page(stats, geo), encoding="utf-8")
    tmp.replace(PUBLIC / "index.html")
    journal(f"totaux : parties={stats['parties']} joueurs={stats['joueurs']} km={stats['km']} blocs={stats['nb_blocs']} "
            f"ignorés={stats['ignores']} tronqué={stats['tronque']}")
    image = None
    if not sans_apercu and apercu(PUBLIC / "apercu.png"):
        image = PUBLIC / "apercu.png"
    if not sans_tour:
        res = publier_tour(stats, html_fiche(stats, geo, image))
        journal(f"tour : {res.get('action')} {res.get('url')}")


if __name__ == "__main__":
    main()
