#!/home/alex/projets/master/sacha/tour/api/venv/bin/python
# -*- coding: utf-8 -*-
"""Crée puis met à jour EN PLACE la fiche « Canohès balade : carte des balades » de la tour Sacha.
Lit sur l'entrée standard {titre, resume, html}. Premier passage : dépôt normal (recherches.deposer, sans mail).
Passages suivants : même fiche (id gardé dans fiche-tour.id), seuls html, resume et misAJourLe changent.
La tour n'a pas d'API de mise à jour (12/11/2026) : on réécrit son fichier data/recherches/<id>.json avec ses
propres fonctions (charger, _vers_html qui nettoie le HTML comme au dépôt), sans toucher au code de la tour."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/home/alex/projets/master/sacha/tour/api")
from tour_app import recherches  # noqa: E402

ID_FICHIER = Path(__file__).resolve().parent / "fiche-tour.id"
d = json.load(sys.stdin)
rid = ID_FICHIER.read_text().strip() if ID_FICHIER.exists() else ""
rec = recherches.charger(rid) if rid else None
if rec is None:
    res = recherches.deposer({
        "titre": d["titre"], "agent": "Claude", "projet": "canohes-jeu", "projets": ["canohes-jeu", "divers-racine"],
        "categorie": "livrable", "resume": d["resume"], "html": d["html"], "nomFichier": "carte-des-balades.html",
        "sourceChemin": "/home/alex/projets/canohes-monde/stats/carte_stats.py", "envoyerEmail": False})
    ID_FICHIER.write_text(res["id"] + "\n")
    res["action"] = "créée"
else:
    rec["html"] = recherches._vers_html(None, d["html"])
    rec["resume"] = d["resume"]
    rec["misAJourLe"] = datetime.now(timezone.utc).isoformat()
    cible = recherches.DOSSIER / f"{rec['id']}.json"
    tmp = cible.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, cible)
    res = {"id": rec["id"], "url": recherches.lien(rec["id"]), "action": "mise à jour"}
print(json.dumps(res, ensure_ascii=False))
