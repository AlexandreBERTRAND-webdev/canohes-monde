"""Assemble le site du jeu : index.html (monde.html avec l'importmap) et blocs.json (où trouver chaque bloc).
Usage : python outils/assembler_site.py <dossier site> <base des blocs> [<dossier des blocs pour lister>] ou --zones <fichier zones.json>"""
import json, os, sys
SITE, BASE = sys.argv[1], sys.argv[2]
os.makedirs(SITE, exist_ok=True)
ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
imap = {"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js", "fflate": "https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js"}}
open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(open(os.path.join(ICI, "monde.html"), encoding="utf-8").read().replace("__IMPORTMAP__", json.dumps(imap)))
if BASE == "--zones":
    z = json.load(open(sys.argv[3]))
    idx = {"bases": z["bases"], "blocs": z["blocs"]}
else:
    blocs = sorted(d for d in os.listdir(sys.argv[3]) if os.path.exists(os.path.join(sys.argv[3], d, "m.json")))
    idx = {"bases": [BASE], "blocs": {b: 0 for b in blocs}}
json.dump(idx, open(os.path.join(SITE, "blocs.json"), "w"), separators=(",", ":"))
import shutil
shutil.copy(os.path.join(ICI, "monuments", "monuments.json"), os.path.join(SITE, "monuments.json"))
open(os.path.join(SITE, ".nojekyll"), "w").close()
print(len(idx["blocs"]), "blocs listés")
