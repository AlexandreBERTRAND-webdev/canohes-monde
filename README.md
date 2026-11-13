# Jumeau numérique de la France, territoire par territoire

Projet personnel, non officiel et sans but lucratif, d'Alexandre Bertrand. Sans lien avec le jumeau numérique national de l'État (JUNN).

Jeu de conduite (et de vol à basse altitude) en 3D dans le navigateur, commencé sur les Pyrénées-Orientales, construit uniquement à partir de données ouvertes. Aucune IA ne tourne pendant le jeu.

## Comment c'est fait

- Le département est découpé en blocs de 2 km sur une grille Lambert-93 (bloc `ix_iy` : coin sud-ouest en `ix*2000, iy*2000`).
- `prep_dept.py` télécharge une fois les données communes : réseau routier principal et communes (BD TOPO), points d'intérêt OpenStreetMap (extrait Geofabrik).
- `prep_bloc.py` télécharge les données d'un bloc : bâtiments, routes, équipements de transport (BD TOPO), parcelles (cadastre), relief laser LiDAR HD avec une marge, photo aérienne légère.
- `video_world.py` (avec `GAME=1 BLOC=1`) construit le bloc : relief, bâtiments et toits, arbres, routes aplanies, trottoirs, ponts, terre-pleins et glissières, péages, clôtures, mobilier, commerces, panneaux de direction et d'agglomération.
- `export_bloc.py` écrit `m.json`, `g.bin` (géométrie compressée) et `t.jpg` (teinte du sol).
- `monde.html` est le jeu : il charge les blocs autour de la voiture et décharge ceux qu'elle quitte.
- Le calcul tourne sur GitHub Actions (`.github/workflows/`), les blocs sont publiés sur GitHub Pages, une zone par dépôt `canohes-blocs-K`.

## Sources et licences

- IGN, Géoplateforme : BD TOPO, LiDAR HD, BD ORTHO, Parcellaire Express. Licence ouverte Etalab 2.0.
- © contributeurs OpenStreetMap, licence ODbL. Extrait régional fourni par Geofabrik.
