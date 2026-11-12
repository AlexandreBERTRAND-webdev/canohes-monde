"""Répartit les blocs du département en N zones (bandes ouest-est), une par dépôt GitHub Pages de moins de 1 Go.
Usage : python outils/zones.py N  -> dept/zones.json"""
import json, sys
N = int(sys.argv[1])
blocs = sorted(json.load(open("dept/blocs.json")), key=lambda b: (b[0], b[1]))
par = -(-len(blocs) // N)
zones = {"bases": [f"https://alexandrebertrand-webdev.github.io/canohes-blocs-{k + 1}/" for k in range(N)], "blocs": {}, "listes": [[] for _ in range(N)]}
for i, (ix, iy) in enumerate(blocs):
    k = i // par
    zones["blocs"][f"{ix}_{iy}"] = k
    zones["listes"][k].append(f"{ix}_{iy}")
json.dump(zones, open("dept/zones.json", "w"), separators=(",", ":"))
print([len(l) for l in zones["listes"]], "blocs par zone")
