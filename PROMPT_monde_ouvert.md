# Mission : transformer « Canohès, balade libre » en monde ouvert généré à la volée

Tu reprends un jeu de conduite en 3D dans le navigateur qui marche déjà sur un village (Canohès, Pyrénées-Orientales). Ta mission : le faire fonctionner sur tout le Roussillon, puis toute la France, en générant le décor par blocs autour de la voiture, sans aucune IA à l'exécution. Uniquement du code et des données ouvertes.

Règles de travail :
- Français direct. Aucun tiret long ni demi-cadratin dans les textes, le code, les commentaires.
- Tu tranches toi-même, tu annonces tes choix, tu avances.
- À chaque étape, tu montres une preuve (capture, mesure) et tu dis ce que tu n'as pas vérifié.
- Tu ne casses pas la version actuelle : elle reste en ligne tant que la nouvelle n'est pas validée.

## 1. Ce qui existe (dossier source fourni)

| Fichier | Rôle |
|---|---|
| `fetch.py` | Télécharge les données IGN : `wfs(couche, bbox)` vers GeoJSON, `raster(couche, bbox, résolution, nom)` vers `.npy`. |
| `video_world.py` | Construit le monde 3D d'un carré de 1,9 km. Avec `GAME=1`, sort `out/game/raw/` : `world.bin` + `world.json` (maillages par dalle), `extra.npz` (grille de hauteurs, emprise des bâtiments), `extra.json` (rues nommées). |
| `game_export.py` | Compacte pour le navigateur : positions sur 16 bits (pas de 3 cm), gzip, base64 dans `monde.json`, photos aériennes réduites en `sol_i_j.jpg`. |
| `game.html` | Le jeu (three.js 0.160 par CDN, un seul fichier) : chargement, voiture, physique, caméra, mini-carte, nom de rue, commandes clavier et tactiles. |
| `video_route.json` | Sert seulement au mode vidéo, à ignorer pour le jeu. |

Chiffres mesurés sur une machine à 2 processeurs, pour le carré de 1,9 km de Canohès :
- 3 449 bâtiments, 7 633 arbres, 118 km de clôtures, 905 000 triangles.
- Construction : environ 55 s, téléchargements compris (dont 20 s de photos aériennes).
- Site final : 12,6 Mo (5 Mo de géométrie compressée, le reste en photos).

## 2. Sources de données (gratuites, sans clé)

IGN Géoplateforme, projection Lambert-93 (EPSG:2154) partout :
- WFS `https://data.geopf.fr/wfs/ows` avec `SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES=...&SRSNAME=EPSG:2154&BBOX=xmin,ymin,xmax,ymax,EPSG:2154&OUTPUTFORMAT=application/json&COUNT=5000&STARTINDEX=...`
  - `BDTOPO_V3:batiment` : emprise, `hauteur`, `nombre_d_etages`, `nature`.
  - `BDTOPO_V3:troncon_de_route` : `nature`, `largeur_de_chaussee`, `sens_de_circulation`, `acces_vehicule_leger`, `nature_de_la_restriction`, `position_par_rapport_au_sol`, `importance`, `urbain`, `prive`, `nom_voie_ban_gauche`, `nom_collaboratif_gauche`, `cpx_numero`, `cpx_classement_administratif`.
  - `BDTOPO_V3:zone_d_habitation` : `toponyme`, `importance` (1 et 2 = bourgs et villages), `nature`.
  - `BDTOPO_V3:zone_de_vegetation`, `terrain_de_sport`, `cimetiere`, `construction_surfacique`, `construction_lineaire`, `surface_hydrographique`, `troncon_de_voie_ferree`.
  - `CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle` : limites de parcelles (clôtures).
- WMS raster `https://data.geopf.fr/wms-r` avec `REQUEST=GetMap&CRS=EPSG:2154&BBOX=...&WIDTH=&HEIGHT=` (5 000 px maximum par requête) :
  - Sol au laser, 1 m : `IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93`, format `image/x-bil;bits=32`.
  - Surface au laser, 0,5 m : `IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93`. Hauteur des objets = surface moins sol.
  - Relief de secours partout en France : `ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES`.
  - Photo aérienne : `ORTHOIMAGERY.ORTHOPHOTOS`, format `image/jpeg`, 20 cm par pixel.
- OpenStreetMap : `https://api.openstreetmap.org/api/0.6/map?bbox=lonmin,latmin,lonmax,latmax` (XML). Utilisé pour `highway=traffic_signals|stop|give_way|crossing|bus_stop`, `amenity=bench|recycling`, `barrier=bollard`. Pour un service en production, préférer un extrait Geofabrik téléchargé une fois plutôt que cette API.

À vérifier avant de s'y fier : la couverture du laser (LiDAR HD) sur tout le département, et la présence des échangeurs d'autoroute dans la BD TOPO.

## 3. Ce que fait déjà le générateur (à conserver)

- Relief : grille de hauteurs de 7 m, lissée, photo aérienne drapée dessus.
- Bâtiments : emprise IGN, hauteur d'égout = 15e centile du laser dans l'emprise, toit plat si l'écart entre 95e et 15e centile est sous 0,7 m. Quadrilatères : toit à deux pans, sens du faîtage lu dans le laser. Autres formes : toit à pans par squelette droit (`compas_cgal.straight_skeleton_2`). Toits texturés par la photo aérienne. Murs avec une texture de fenêtre à volets répétée, sauf les hangars.
- Type de bâtiment d'après la couleur du toit sur la photo : tuiles (enduit chaud), toit plat (blanc), grand toit gris ou bâtiment d'activité (bardage sans fenêtres).
- Arbres : sommets locaux du laser hors bâtiments et hors chaussée, couleur prise sur la photo, cyprès en cône si haut et étroit.
- Chaussée, trottoirs (1,5 m), ligne axiale en tirets si largeur de 5 m ou plus, pistes cyclables à part : polygones drapés sur le relief par intersection `manifold3d` (grille de hauteurs décalée de 5 à 17 cm).
- Clôtures : limites des parcelles bâties du cadastre, moins la chaussée et les bâtiments. Type tiré au sort (mur clair, haie, muret).
- Mobilier OpenStreetMap posé sur le bord droit de la rue la plus proche. Lampadaires tous les 46 m sur les rues urbaines.
- Collisions : image binaire des bâtiments à 1 m par case.

Pièges déjà rencontrés, à ne pas refaire :
- La BD TOPO classe les pistes cyclables en « Route à 1 chaussée ». Ne garder pour la conduite que `acces_vehicule_leger = "Libre"`.
- Respecter `sens_de_circulation` (Double sens, Sens direct, Sens inverse).
- Les fichiers `.bin` sont refusés par certains hébergeurs : la géométrie est en gzip puis base64 dans du JSON. Sur Cloudflare tu peux revenir à du binaire brut, plus léger.
- La voiture se coinçait dans les murs : la correction (pas de rotation dans un mur, glissement, sortie automatique) est dans `game.html`, fonction `avance`.

## 4. Architecture cible

Le générateur est en Python avec des bibliothèques lourdes (`manifold3d`, `compas_cgal`, `rasterio`, `shapely`, `scipy`, `scikit-image`, `numpy`, `pyproj`, `networkx`, `Pillow`, `mapbox_earcut`). Il ne peut pas tourner dans un Worker Cloudflare. Donc trois morceaux :

1. **Générateur : un service Python en conteneur sur Railway.** API HTTP minimale :
   - `GET /bloc/{ix}/{iy}` : renvoie le bloc s'il est en cache, sinon lance son calcul et répond 202.
   - `POST /autour` avec position, cap et vitesse : met en file les blocs manquants, ceux devant la voiture en premier.
   - Une file de travail, un nombre de calculs simultanés limité, pas deux calculs du même bloc.
2. **Cache : Cloudflare R2.** Un objet par bloc (géométrie + photos + métadonnées), plus le cache des téléchargements IGN. Un bloc calculé sert à tous les joueurs.
3. **Jeu : Cloudflare Pages** (site statique) + un petit Worker qui sert les blocs depuis R2 et relaie les demandes au générateur. Le navigateur ne parle qu'à Cloudflare.

Tu peux proposer mieux si tu le justifies, mais le jeu doit rester servi par Cloudflare et le calcul ne doit pas être sur le VPS.

## 5. Les blocs

- Grille fixe en Lambert-93 : bloc `(ix, iy)` = carré de 2 000 m dont le coin sud-ouest est `(ix*2000, iy*2000)`.
- Mesure attendue : un bloc de 2 000 m coûte environ 16 fois un bloc de 500 m, soit de l'ordre d'une minute sur 2 processeurs. À 130 km/h on traverse 2 000 m en 55 s. Donc :
  - soit tu calcules deux blocs d'avance dans la direction du déplacement,
  - soit tu gardes des sous-blocs de 500 m calculés en parallèle et regroupés par 2 000 m,
  - et dans tous les cas tu fournis d'abord un **bloc provisoire** (relief + photo aérienne, quelques secondes) remplacé ensuite par le bloc complet, sans à-coup ni saut de la voiture.
- Raccords : la grille de hauteurs utilise des nœuds communs sur les bords des blocs. Routes, clôtures et bâtiments sont coupés exactement à la limite, ou attribués entièrement au bloc qui contient leur centre (bâtiments, arbres, mobilier).
- Côté jeu : charger les blocs à moins de 2 blocs, décharger au-delà de 3, libérer la mémoire graphique (`geometry.dispose()`, `texture.dispose()`).
- Les coordonnées du jeu doivent rester petites : origine flottante recalée sur le bloc courant, sinon la précision se dégrade loin de l'origine.

## 6. Ponts et tunnels (rien n'existe, priorité haute)

Aujourd'hui toutes les routes sont plaquées au sol, ce qui donne un résultat faux au pont entre Toulouges et Canohès.
- `position_par_rapport_au_sol` : 0 au sol, positif sur un pont ou viaduc, négatif en tunnel. Voir aussi `BDTOPO_V3:construction_surfacique` (nature Pont) pour l'emprise du tablier.
- Pont : tablier dont la hauteur est interpolée entre les altitudes de ses deux extrémités au sol, épaisseur visible, garde-corps des deux côtés, piles ou culées, rampes d'accès qui rejoignent le relief en douceur.
- Ce qui passe dessous (route, voie ferrée, cours d'eau) reste praticable et visible.
- Conduite : la voiture suit la hauteur de la route sur laquelle elle roule, pas celle du terrain. Il faut donc savoir à quel niveau elle est quand deux routes se superposent.
- Tunnel : au minimum une tête de tunnel propre et une chaussée qui ne traverse pas le relief.
- Test obligatoire : le pont entre Toulouges et Canohès, vu de dessus et de dessous.

## 7. Panneaux de direction

Où en poser :
- À chaque rond-point et à chaque grosse intersection hors agglomération ou sur les grands axes (routes d'`importance` 1 à 3, ou classées départementales, nationales, autoroutes).
- Jamais dans les petites rues à l'intérieur d'un village.

Quoi écrire :
- **Routes normales :** sur chaque branche de l'intersection, les deux prochains villages dans cette direction, avec la distance en km. Méthode : graphe routier (`networkx`), on part dans la branche et on suit le plus court chemin vers les lieux habités d'importance 1 et 2 ; on retient les deux premiers atteints par cette branche.
- **Entrée d'autoroute et autoroute :** pas le prochain village, mais la prochaine sortie, dans chaque sens, avec son nom, son numéro et la distance, plus une grande ville plus lointaine. Source à vérifier dans la BD TOPO ; à défaut, OpenStreetMap (`highway=motorway_junction`, étiquettes `ref` et `name`).
- **Entrée et sortie d'agglomération :** panneau avec le nom de la commune (sortie : nom barré).
- Numéro de la route (`cpx_numero`) en cartouche : rouge pour une nationale, jaune pour une départementale, bleu pour une autoroute.

Comment les dessiner :
- Panneaux français : flèche blanche à texte noir pour les directions locales, fond vert pour les grands itinéraires, fond bleu pour l'autoroute.
- Texte dessiné dans un canvas puis utilisé comme texture, lisible depuis la voiture à 30 m.
- Posés sur le bord droit, hors chaussée, tournés vers la voie d'arrivée.
- Le calcul des destinations dépasse le bloc courant : préparer une fois pour toutes le graphe routier du département et la liste des lieux habités, et les interroger à la génération.

## 8. Signalisation, mobilier, clôtures

- Garder tout ce qui existe (section 3).
- Ajouter : marquage au sol des stops et cédez-le-passage, îlots et anneau central des ronds-points, glissières sur les grands axes, panneaux de limitation de vitesse si OpenStreetMap les donne.
- Les clôtures et murets deviennent des obstacles, avec une ouverture devant chaque maison côté rue pour ne pas enfermer la voiture.
- Style : Roussillon pour l'instant. Mets tous les réglages de style (couleurs de façades, pente des toits, types de clôtures, essences d'arbres) dans un fichier par région, pour en ajouter d'autres sans toucher au code.

## 9. Bon voisinage avec l'IGN

- Leurs serveurs sont gratuits et pas faits pour un jeu : mettre en cache tout ce qui est téléchargé, limiter le rythme des requêtes, réessayer avec attente en cas d'erreur.
- Si l'IGN ne répond pas : servir le bloc provisoire et réessayer plus tard, sans bloquer le joueur.
- Afficher la mention des sources dans le jeu : IGN (BD TOPO, LiDAR HD, BD ORTHO), cadastre, contributeurs OpenStreetMap.

## 10. Ordre de travail et preuves attendues

1. Mettre le générateur en fonction `construire_bloc(ix, iy)` sans rien changer au résultat, et mesurer son temps pour 500 m et 2 000 m.
2. Chargement par blocs dans le jeu avec quatre blocs autour de Canohès calculés à la main. Preuve : passer d'un bloc à l'autre sans couture visible.
3. Service sur Railway + cache R2 + Worker. Preuve : rouler de Canohès à Toulouges sans décor manquant à 50 puis 90 km/h.
4. Ponts. Preuve : captures du pont Toulouges-Canohès.
5. Panneaux. Preuve : captures de trois ronds-points et d'une entrée d'autoroute, avec les destinations vérifiées sur une vraie carte.
6. Clôtures en obstacles, style par région.
7. Mesures finales : temps par bloc, mémoire sur téléphone avec 9 blocs chargés, images par seconde, coût mensuel estimé de Railway et de R2.

Avant de commencer, dis-moi en dix lignes ton plan, ce que tu comptes vérifier en premier, et le coût mensuel que tu anticipes.
