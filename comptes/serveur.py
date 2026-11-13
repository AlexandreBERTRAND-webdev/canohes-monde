"""Comptes des joueurs de « Balade en Roussillon » : inscription par code reçu par e-mail, trajets, statistiques, classement.
Jeu gratuit et non commercial : l'e-mail ne sert qu'à se connecter et aux messages liés au jeu.
Lancement : python3 serveur.py (écoute 127.0.0.1:9172, derrière nginx sous /jeu-api/)."""
import hashlib, json, os, re, secrets, sqlite3, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ICI = os.path.dirname(os.path.abspath(__file__))
DONNEES = os.path.join(ICI, "data")
os.makedirs(DONNEES, exist_ok=True)
BASE = os.path.join(DONNEES, "comptes.sqlite")
SEL_F = os.path.join(DONNEES, "sel")
if not os.path.exists(SEL_F):
    with open(os.open(SEL_F, os.O_WRONLY | os.O_CREAT, 0o600), "w") as f:
        f.write(secrets.token_hex(32))
SEL = open(SEL_F).read().strip()
CLE_RESEND = os.path.expanduser("~/.config/sacha/resend.key")
ORIGINES = {"https://alexandrebertrand-webdev.github.io", "https://canohes-balade.pages.dev", "https://bertrand.al"}
EXPEDITEUR = "Jumeau numérique de la France <jeu@bertrand.al>"
verrou = threading.Lock()


def db():
    c = sqlite3.connect(BASE, timeout=10)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS joueurs (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, pseudo TEXT UNIQUE NOT NULL, cree REAL, vu REAL);
    CREATE TABLE IF NOT EXISTS codes (email TEXT PRIMARY KEY, pseudo TEXT, empreinte TEXT, expire REAL, essais INTEGER DEFAULT 0, envois TEXT DEFAULT '[]');
    CREATE TABLE IF NOT EXISTS sessions (empreinte TEXT PRIMARY KEY, joueur INTEGER, cree REAL);
    CREATE TABLE IF NOT EXISTS trajets (id TEXT PRIMARY KEY, joueur INTEGER, debut REAL, fin REAL, km REAL DEFAULT 0, points TEXT DEFAULT '[]', blocs TEXT DEFAULT '[]');
    CREATE INDEX IF NOT EXISTS i_traj_joueur ON trajets(joueur);
    CREATE INDEX IF NOT EXISTS i_traj_fin ON trajets(fin);
    """)

h = lambda s: hashlib.sha256((SEL + s).encode()).hexdigest()
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[a-z]{2,24}$", re.I)
PSEUDO = re.compile(r"^[\w\- ]{3,20}$")
limites = {}


def trop(cle, n, fenetre):
    """Limite simple en mémoire : au plus n appels par fenêtre (s)."""
    t = time.time()
    with verrou:
        l = [x for x in limites.get(cle, []) if t - x < fenetre]
        if len(l) >= n:
            limites[cle] = l
            return True
        limites[cle] = l + [t]
    return False


def envoyer_code(email, code):
    texte = (f"Bonjour,\n\nVoici ton code pour le Jumeau numérique de la France : {code}\n\nIl est valable 15 minutes. "
             "Si tu n'as rien demandé, tu peux ignorer ce message.\n\n"
             "Le Jumeau numérique de la France est un projet personnel, gratuit et non commercial, créé par Alexandre Bertrand. Ce n'est pas un service de l'État. "
             "Pas de publicité, pas de spam : tu ne recevras que des messages liés au jeu.\n"
             "Une question technique : alexandre@bertrand.al\n")
    corps = json.dumps({"from": EXPEDITEUR, "to": [email], "subject": f"Ton code Jumeau numérique de la France : {code}", "text": texte,
                        "reply_to": "alexandre@bertrand.al"}).encode()
    req = urllib.request.Request("https://api.resend.com/emails", data=corps, method="POST",
                                 headers={"Authorization": "Bearer " + open(CLE_RESEND).read().strip(), "Content-Type": "application/json", "User-Agent": "balade-en-roussillon/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status in (200, 201)


def simplifier(pts, pas=25.0):
    out = []
    for p in pts:
        if not out or (p[0] - out[-1][0]) ** 2 + (p[1] - out[-1][1]) ** 2 >= pas * pas:
            out.append([round(p[0]), round(p[1])])
    return out


class H(BaseHTTPRequestHandler):
    server_version = "balade"

    def log_message(self, *a):
        pass

    def entetes(self, code=200, ctype="application/json"):
        self.send_response(code)
        o = self.headers.get("Origin", "")
        if o in ORIGINES:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def rep(self, obj, code=200):
        self.entetes(code)
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode())

    def do_OPTIONS(self):
        self.entetes(204)

    def corps(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 400_000:
            return None
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return None

    def ip(self):
        return self.headers.get("X-Real-IP") or self.client_address[0]

    def joueur(self, c):
        a = self.headers.get("Authorization", "")
        if not a.startswith("Bearer "):
            return None
        r = c.execute("SELECT j.* FROM sessions s JOIN joueurs j ON j.id = s.joueur WHERE s.empreinte = ?", (h(a[7:].strip()),)).fetchone()
        if r:
            c.execute("UPDATE joueurs SET vu = ? WHERE id = ?", (time.time(), r["id"]))
        return r

    def do_GET(self):
        chemin, _, qs = self.path.partition("?")
        q = dict(x.split("=", 1) for x in qs.split("&") if "=" in x)
        with db() as c:
            if chemin == "/classement":
                rows = c.execute("""SELECT j.pseudo, ROUND(SUM(t.km), 1) km, COUNT(t.id) parties FROM joueurs j JOIN trajets t ON t.joueur = j.id
                                    GROUP BY j.id ORDER BY km DESC LIMIT 30""").fetchall()
                tot = c.execute("SELECT COUNT(DISTINCT joueur) n, ROUND(COALESCE(SUM(km), 0), 1) km, COUNT(*) parties FROM trajets").fetchone()
                inscrits = c.execute("SELECT COUNT(*) FROM joueurs").fetchone()[0]
                return self.rep({"joueurs": [dict(r) for r in rows], "total": dict(tot), "inscrits": inscrits})
            j = self.joueur(c)
            if not j:
                return self.rep({"erreur": "connexion requise"}, 401)
            if chemin == "/moi":
                rows = c.execute("SELECT id, debut, fin, km, blocs FROM trajets WHERE joueur = ? ORDER BY fin DESC", (j["id"],)).fetchall()
                blocs = set()
                for r in rows:
                    blocs.update(json.loads(r["blocs"]))
                return self.rep({"pseudo": j["pseudo"], "inscrit": j["cree"], "km": round(sum(r["km"] for r in rows), 1), "parties": len(rows),
                                 "blocs": len(blocs), "minutes": round(sum(max(0, r["fin"] - r["debut"]) for r in rows) / 60),
                                 "trajets": [{"debut": r["debut"], "km": round(r["km"], 1)} for r in rows[:20]]})
            if chemin == "/carte":                                   # routes de tous les joueurs (récentes), pour la carte et la mini-carte
                x0, y0, x1, y1 = (float(q.get(k, d)) for k, d in (("x0", -1e9), ("y0", -1e9), ("x1", 1e9), ("y1", 1e9)))
                rows = c.execute("SELECT t.points, j.pseudo, t.joueur = ? moi FROM trajets t JOIN joueurs j ON j.id = t.joueur ORDER BY t.fin DESC LIMIT 400", (j["id"],)).fetchall()
                out = []
                for r in rows:
                    pts = json.loads(r["points"])
                    if pts and any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in pts):
                        out.append({"pseudo": r["pseudo"], "moi": bool(r["moi"]), "points": simplifier(pts, 60 if x1 - x0 > 20000 else 25)})
                return self.rep({"routes": out})
        self.rep({"erreur": "inconnu"}, 404)

    def do_POST(self):
        d = self.corps()
        if d is None:
            return self.rep({"erreur": "requête invalide"}, 400)
        with db() as c:
            if self.path == "/code":
                email, pseudo = str(d.get("email", "")).strip().lower(), str(d.get("pseudo", "")).strip()
                if not EMAIL.match(email):
                    return self.rep({"erreur": "Adresse e-mail invalide."}, 400)
                if trop("ip" + self.ip(), 8, 3600) or trop("em" + email, 3, 900):
                    return self.rep({"erreur": "Trop de demandes, réessaie dans quelques minutes."}, 429)
                existe = c.execute("SELECT pseudo FROM joueurs WHERE email = ?", (email,)).fetchone()
                if not existe:
                    if not PSEUDO.match(pseudo):
                        return self.rep({"erreur": "Choisis un pseudo de 3 à 20 caractères (lettres, chiffres, tiret)."}, 400)
                    if c.execute("SELECT 1 FROM joueurs WHERE lower(pseudo) = lower(?)", (pseudo,)).fetchone():
                        return self.rep({"erreur": "Ce pseudo est déjà pris."}, 409)
                code = f"{secrets.randbelow(10 ** 6):06d}"
                c.execute("INSERT OR REPLACE INTO codes (email, pseudo, empreinte, expire, essais) VALUES (?, ?, ?, ?, 0)",
                          (email, existe["pseudo"] if existe else pseudo, h(email + code), time.time() + 900))
                try:
                    envoyer_code(email, code)
                except Exception as e:
                    print("envoi du code impossible :", repr(e)[:300], flush=True)
                    return self.rep({"erreur": "Envoi du mail impossible pour le moment."}, 502)
                return self.rep({"ok": True, "nouveau": not existe})
            if self.path == "/connexion":
                email, code = str(d.get("email", "")).strip().lower(), re.sub(r"\D", "", str(d.get("code", "")))
                r = c.execute("SELECT * FROM codes WHERE email = ?", (email,)).fetchone()
                if not r or r["expire"] < time.time() or r["essais"] >= 5:
                    return self.rep({"erreur": "Code expiré, redemande un code."}, 400)
                if not secrets.compare_digest(r["empreinte"], h(email + code)):
                    c.execute("UPDATE codes SET essais = essais + 1 WHERE email = ?", (email,))
                    return self.rep({"erreur": "Code incorrect."}, 400)
                c.execute("DELETE FROM codes WHERE email = ?", (email,))
                j = c.execute("SELECT * FROM joueurs WHERE email = ?", (email,)).fetchone()
                if not j:
                    if c.execute("SELECT 1 FROM joueurs WHERE lower(pseudo) = lower(?)", (r["pseudo"],)).fetchone():
                        return self.rep({"erreur": "Ce pseudo vient d'être pris, choisis-en un autre."}, 409)
                    c.execute("INSERT INTO joueurs (email, pseudo, cree, vu) VALUES (?, ?, ?, ?)", (email, r["pseudo"], time.time(), time.time()))
                    j = c.execute("SELECT * FROM joueurs WHERE email = ?", (email,)).fetchone()
                jeton = secrets.token_urlsafe(32)
                c.execute("INSERT INTO sessions VALUES (?, ?, ?)", (h(jeton), j["id"], time.time()))
                return self.rep({"jeton": jeton, "pseudo": j["pseudo"]})
            j = self.joueur(c)
            if not j:
                return self.rep({"erreur": "connexion requise"}, 401)
            if self.path == "/trajet":
                tid = re.sub(r"[^\w-]", "", str(d.get("id", "")))[:40]
                pts = [[float(p[0]), float(p[1])] for p in (d.get("points") or [])[:2000] if isinstance(p, list) and len(p) == 2]
                if not tid or trop("tr" + str(j["id"]), 30, 60):
                    return self.rep({"erreur": "refusé"}, 429)
                r = c.execute("SELECT * FROM trajets WHERE id = ?", (tid,)).fetchone()
                if r and r["joueur"] != j["id"]:
                    return self.rep({"erreur": "refusé"}, 403)
                anc = json.loads(r["points"]) if r else []
                tout = (anc + pts)[-20000:]
                blocs = sorted({f"{int(p[0] // 2000)}_{int(p[1] // 2000)}" for p in tout})
                km = min(float(d.get("km") or 0), 2000.0)
                if r:
                    c.execute("UPDATE trajets SET fin = ?, km = MAX(km, ?), points = ?, blocs = ? WHERE id = ?", (time.time(), km, json.dumps(tout), json.dumps(blocs), tid))
                else:
                    c.execute("INSERT INTO trajets VALUES (?, ?, ?, ?, ?, ?, ?)", (tid, j["id"], time.time(), time.time(), km, json.dumps(tout), json.dumps(blocs)))
                return self.rep({"ok": True})
            if self.path == "/supprimer":                            # droit à l'effacement : compte, sessions et trajets
                c.execute("DELETE FROM trajets WHERE joueur = ?", (j["id"],))
                c.execute("DELETE FROM sessions WHERE joueur = ?", (j["id"],))
                c.execute("DELETE FROM joueurs WHERE id = ?", (j["id"],))
                return self.rep({"ok": True})
        self.rep({"erreur": "inconnu"}, 404)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 9172), H).serve_forever()
