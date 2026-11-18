"""Rechnet die vorgerechneten Kostenzahlen der App und schreibt `svw_kosten.json` (und `svw_iat8.json`).

Aufruf (Python des Test-venv, aus dem Repo-Ordner):
    python tools/svw_make_kosten.py --schreibe --lang     alles rechnen (Ring n = 8, 12, 16; rund 1 bis 3 Minuten) und die JSON-Datei schreiben
    python tools/svw_make_kosten.py --check               n = 8, 12, Torus 4×4 und Takt neu rechnen und mit der Datei vergleichen (relativ 1e-6; t_rel und τ des Gipfelindikators 1e-4)
    python tools/svw_make_kosten.py --check --lang        zusätzlich n = 16
    python tools/svw_make_kosten.py --iat8                Heat-bath-τ auf dem 8×8-Torus und die Eichung am 4×4-Torus (Messung, schreibt svw_iat8.json)

Was gerechnet wird (alles Platzaktualisierungen bzw. Platzentscheidungen, keine Sekunden):
  * τ_f = 1 + 2 Σ ρ_k(f) einer stationär gestarteten Heat-bath-Kette, exakt aus der Fundamentalmatrix Z = (I − P + 1πᵀ)⁻¹: σ² = 2⟨f̃, Z f̃⟩_π − ⟨f̃, f̃⟩_π,
    τ = σ² / Var f, für f = Teilchenzahl und f = Gipfelindikator I (1 für s > 0, 1/2 für s = 0, 0 für s < 0; s = belegte gerade − belegte ungerade Plätze).
  * t_mix(1/4) = erstes t mit d(t) ≤ 1/4. Für n ≤ 12 ist d(t) = max über alle Starts (Matrixpotenzen); für n = 16 nur vom Start „alle geraden Plätze“
    (Vektoriteration; für n ≤ 12 ist das der schlechteste Start, das prüft der Test).
  * Kopplung aus der Vergangenheit: erwartete Kosten Σ_k T_k S(T_(k−1)) mit Rückschau T_k = 8, 16, 32, ... und S = Überlebenswahrscheinlichkeit der Paarkette
    aus oberer und unterer Extremkette (gemeinsamer Platz, gemeinsame Münze). Die Paarkette ist unter Drehungen um zwei Plätze und Spiegelungen zusammengefasst
    (Zustände sind Bahnen), das ist exakt, weil diese Symmetrien Plätze und Münzen verwandeln und die Extremketten festlassen. Kosten in Platzentscheidungen =
    2 × Schritte der Paarkette (beide Ketten aktualisieren je Schritt einen Platz). n = 16 mit λ = 20 und 100: nicht gerechnet (die Überlebenswahrscheinlichkeit
    ist nach 40 000 Schritten noch nicht unter 1e-15).
  * Torus 4×4 (743 Zustände): τ beider Funktionen für sieben λ, dazu t_rel = 1/(1 − zweitgrößter Eigenwert).
  * Takt: t_mix(1/4) des Heat-bath durch t_mix(1/4) des Einzelplatz-Metropolis auf dem 12-Ring (alle Starts), gegen (1 + λ) / max(1, λ).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import svw_diag as G  # noqa: E402
import svw_direct as D  # noqa: E402
from svw_zufall import Strom  # noqa: E402

KOSTEN = ROOT / "svw_kosten.json"
IAT8 = ROOT / "svw_iat8.json"
RING_N = (8, 12, 16)
RING_LAM = (1.0, 5.0, 20.0, 100.0)
TORUS_LAM = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
TAKT_LAM = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
T0 = 8
TOL = 1e-15


# ----------------------------------------------------------------------------------------------------------------------------------------
# Raum, Matrizen, exakte Größen
# ----------------------------------------------------------------------------------------------------------------------------------------
class Raum:
    """Zulässige Belegungen (Hard-Core) eines Graphen mit höchstens 16 Plätzen; Bitmasken aufsteigend."""

    def __init__(self, nachbarn, gerade):
        self.V = len(nachbarn)
        self.nb = np.array(nachbarn, dtype=np.int64)
        m = np.arange(1 << self.V, dtype=np.int64)
        schlecht = np.zeros(len(m), dtype=bool)
        for v in range(self.V):
            schlecht |= (((m >> v) & 1) == 1) & ((m & self.nb[v]) != 0)
        self.s = m[~schlecht]
        self.N = len(self.s)
        self.k = D.popcount(self.s)
        self.gerade = int(gerade)
        self.ungerade = ((1 << self.V) - 1) ^ self.gerade

    def idx(self, x):
        return np.searchsorted(self.s, x)

    def pi(self, lam):
        w = np.power(float(lam), self.k.astype(float))
        return w / w.sum()

    def heatbath(self, lam):
        p = lam / (1.0 + lam)
        P = np.zeros((self.N, self.N))
        z = np.arange(self.N)
        for v in range(self.V):
            bit = 1 << v
            frei = (self.s & self.nb[v]) == 0
            P[z, self.idx(self.s & ~bit)] += (1.0 - np.where(frei, p, 0.0)) / self.V
            f = z[frei]
            P[f, self.idx(self.s[f] | bit)] += p / self.V
        return P

    def metropolis(self, lam):
        P = np.zeros((self.N, self.N))
        z = np.arange(self.N)
        for v in range(self.V):
            bit = 1 << v
            belegt = (self.s & bit) != 0
            frei = (~belegt) & ((self.s & self.nb[v]) == 0)
            blockiert = (~belegt) & (~frei)
            ab = min(1.0, 1.0 / lam)
            auf = min(1.0, lam)
            b = z[belegt]
            P[b, self.idx(self.s[b] ^ bit)] += ab / self.V
            P[b, b] += (1.0 - ab) / self.V
            f = z[frei]
            P[f, self.idx(self.s[f] ^ bit)] += auf / self.V
            P[f, f] += (1.0 - auf) / self.V
            P[z[blockiert], z[blockiert]] += 1.0 / self.V
        return P

    def gipfel(self):
        """Gipfelindikator I: 1 für s > 0, 1/2 für s = 0, 0 für s < 0 (s = belegte gerade − belegte ungerade Plätze)."""
        s = D.popcount(self.s & self.gerade) - D.popcount(self.s & self.ungerade)
        return np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0))


def ring_raum(n):
    nb = [(1 << ((v + 1) % n)) | (1 << ((v - 1) % n)) for v in range(n)]
    return Raum(nb, sum(1 << v for v in range(0, n, 2)))


def torus_raum(a, b):
    nb = []
    for i in range(a):
        for j in range(b):
            m = 0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                m |= 1 << (((i + di) % a) * b + (j + dj) % b)
            nb.append(m)
    gerade = sum(1 << (i * b + j) for i in range(a) for j in range(b) if (i + j) % 2 == 0)
    return Raum(nb, gerade)


def tau_fundamental(P, pi, f, lu=None):
    """τ aus der Fundamentalmatrix; liefert τ und die LU-Zerlegung (für weitere Funktionen derselben Kette)."""
    if lu is None:
        lu = sla.lu_factor(np.eye(len(pi)) - P + np.outer(np.ones(len(pi)), pi))
    ft = f - float(pi @ f)
    var = float(pi @ (ft * ft))
    g = sla.lu_solve(lu, ft)
    return (2.0 * float(pi @ (ft * g)) - var) / var, lu


def t_rel(P, pi):
    d = np.sqrt(pi)
    s = (d[:, None] * P) / d[None, :]
    ev = np.sort(np.linalg.eigvalsh(0.5 * (s + s.T)))[::-1]
    return float(1.0 / (1.0 - ev[1]))


def _d_max(M, pi):
    return float(0.5 * np.abs(M - pi).sum(axis=1).max())


def tmix_alle_starts(P, pi, eps=0.25):
    """t_mix(eps) = erstes t mit max_x TV(P^t(x, ·), π) ≤ eps, durch Quadrieren und bitweises Heraufzählen (d(t) fällt monoton)."""
    pw = [P]
    while _d_max(pw[-1], pi) > eps:
        pw.append(pw[-1] @ pw[-1])
        if len(pw) > 40:
            raise RuntimeError("t_mix über 2^40")
    M = np.eye(len(pi))
    t = 0
    for j in range(len(pw) - 2, -1, -1):
        M2 = M @ pw[j]
        if _d_max(M2, pi) > eps:
            M, t = M2, t + (1 << j)
    return t + 1


def tmix_start(P_sparse, pi, start, eps=0.25, tmax=2_000_000):
    """t_mix(eps) nur vom Zustand `start` (Vektoriteration)."""
    v = np.zeros(len(pi))
    v[start] = 1.0
    PT = P_sparse.T.tocsr()
    for t in range(1, tmax + 1):
        v = PT @ v
        if 0.5 * np.abs(v - pi).sum() <= eps:
            return t
    raise RuntimeError("t_mix nicht erreicht")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Kopplung aus der Vergangenheit: Paarkette der beiden Extremketten
# ----------------------------------------------------------------------------------------------------------------------------------------
def _symmetrien(n):
    """Drehungen um zwei Plätze und Spiegelungen v -> 2j − v als Permutationen der Plätze (erhalten die Parität, also die Rollen von oben und unten)."""
    perms = []
    for j in range(n // 2):
        perms.append([(v + 2 * j) % n for v in range(n)])
        perms.append([(2 * j - v) % n for v in range(n)])
    return perms


def paarkette(n):
    """Erreichbare Paar-Bahnen (obere, untere Extremkette) ab (alle geraden, alle ungeraden Plätze). Rückgabe: Quelle, Ziel (−1 = zusammengetroffen), Münze, Zahl."""
    if n % 2:
        raise ValueError("gerades n")
    gerade = sum(1 << v for v in range(0, n, 2))
    ungerade = sum(1 << v for v in range(1, n, 2))
    nb = [(1 << ((v + 1) % n)) | (1 << ((v - 1) % n)) for v in range(n)]
    tabellen = []
    for perm in _symmetrien(n):
        lo = np.zeros(256, dtype=np.int64)
        hi = np.zeros(256, dtype=np.int64)
        for byte in range(256):
            for t in range(8):
                if (byte >> t) & 1:
                    if t < n:
                        lo[byte] |= 1 << perm[t]
                    if t + 8 < n:
                        hi[byte] |= 1 << perm[t + 8]
        tabellen.append((lo.tolist(), hi.tolist()))

    def kanonisch(a, b):
        best = None
        for lo, hi in tabellen:
            key = ((lo[a & 255] | hi[(a >> 8) & 255]) << n) | (lo[b & 255] | hi[(b >> 8) & 255])
            if best is None or key < best:
                best = key
        return best >> n, best & ((1 << n) - 1)

    start = kanonisch(gerade, ungerade)
    nummer = {start: 0}
    liste = [start]
    quelle, ziel, muenze = [], [], []
    i = 0
    while i < len(liste):
        a, b = liste[i]
        for v in range(n):
            bit = 1 << v
            for m in (0, 1):
                na = (a & ~bit) | (bit if (m and not a & nb[v]) else 0)
                nbb = (b & ~bit) | (bit if (m and not b & nb[v]) else 0)
                if na == nbb:
                    j = -1
                else:
                    key = kanonisch(na, nbb)
                    if key not in nummer:
                        nummer[key] = len(liste)
                        liste.append(key)
                    j = nummer[key]
                quelle.append(i)
                ziel.append(j)
                muenze.append(m)
        i += 1
    return np.array(quelle), np.array(ziel), np.array(muenze), len(liste)


def kopplung_kosten(kette, n, lam, schrittgrenze=40000):
    """Erwartete Kosten (Schritte der Paarkette) eines Laufs mit Rückschau 8, 16, 32, ...; None, wenn die Überlebenswahrscheinlichkeit die Grenze nicht unterschreitet."""
    quelle, ziel, muenze, m = kette
    p = lam / (1.0 + lam)
    gew = (1.0 / n) * np.where(muenze == 1, p, 1.0 - p)
    bleibt = ziel >= 0
    Q = sp.coo_matrix((gew[bleibt], (quelle[bleibt], ziel[bleibt])), shape=(m, m)).tocsr()
    kosten = float(T0)
    if m <= 3000:
        Pw = np.linalg.matrix_power(Q.toarray(), T0)
        S = float(Pw[0].sum())
        T = T0
        while S >= TOL:
            kosten += 2 * T * S
            Pw = Pw @ Pw
            T *= 2
            S = float(Pw[0].sum())
            if T > 1 << 40:
                return None
        return kosten
    v = np.zeros(m)
    v[0] = 1.0
    QT = Q.T.tocsr()
    t, T = 0, T0
    while True:
        while t < T:
            v = QT @ v
            t += 1
            if t > schrittgrenze:
                return None
        S = float(v.sum())
        if S < TOL:
            return kosten
        kosten += 2 * T * S
        T *= 2


# ----------------------------------------------------------------------------------------------------------------------------------------
# Zellen
# ----------------------------------------------------------------------------------------------------------------------------------------
def ring_zellen(n):
    raum = ring_raum(n)
    f_m = raum.k.astype(float)
    f_i = raum.gipfel()
    kette = paarkette(n)
    zellen = []
    for lam in RING_LAM:
        P = raum.heatbath(lam)
        pi = raum.pi(lam)
        tau_m, lu = tau_fundamental(P, pi, f_m)
        tau_i, _ = tau_fundamental(P, pi, f_i, lu)
        start_g = int(raum.idx(raum.gerade))
        t_gerade = tmix_start(sp.csr_matrix(P), pi, start_g)
        t_mix = tmix_alle_starts(P, pi) if n <= 12 else t_gerade
        schritte = kopplung_kosten(kette, n, lam)
        zellen.append({
            "n": n, "lam": lam, "zustaende": int(raum.N), "paarketten_zustaende": int(kette[3]), "t_mix": int(t_mix), "t_mix_start_gerade": int(t_gerade),
            "tau_teilchen": tau_m, "tau_gipfel": tau_i, "direkt": n,
            "kopplung_schritte": schritte, "kopplung_platz": None if schritte is None else 2.0 * schritte,
        })
    return zellen


def torus_zellen(lams=TORUS_LAM):
    raum = torus_raum(4, 4)
    f_m = raum.k.astype(float)
    f_i = raum.gipfel()
    zellen = []
    for lam in lams:
        P = raum.heatbath(lam)
        pi = raum.pi(lam)
        tau_m, lu = tau_fundamental(P, pi, f_m)
        tau_i, _ = tau_fundamental(P, pi, f_i, lu)
        zellen.append({"lam": lam, "zustaende": int(raum.N), "tau_teilchen": tau_m, "tau_gipfel": tau_i, "t_rel": t_rel(P, pi), "direkt": 16})
    return zellen


def takt_zellen(n=12, lams=TAKT_LAM):
    raum = ring_raum(n)
    zellen = []
    for lam in lams:
        pi = raum.pi(lam)
        t_hb = tmix_alle_starts(raum.heatbath(lam), pi)
        t_me = tmix_alle_starts(raum.metropolis(lam), pi)
        zellen.append({"n": n, "lam": lam, "t_mix_heatbath": int(t_hb), "t_mix_metropolis": int(t_me), "verhaeltnis": t_hb / t_me, "takt": (1 + lam) / max(1.0, lam)})
    return zellen


FELDER = {
    "einheit": "Platzentscheidungen (Heat-bath: Platzaktualisierungen = Schritte; direkt: n; Kopplung: 2 × Schritte der Paarkette)",
    "t_mix": "t_mix(1/4) in Schritten; n <= 12: Maximum über alle Starts, n = 16: Start alle geraden Plätze; exakt (Matrixpotenzen bzw. Vektoriteration)",
    "tau_teilchen": "Schritte je effektiv unabhängige Probe der Teilchenzahl, stationärer Start, exakt aus der Fundamentalmatrix",
    "tau_gipfel": "Schritte je effektiv unabhängige Probe des Gipfelindikators I, stationärer Start, exakt aus der Fundamentalmatrix",
    "kopplung_schritte": "erwartete Kosten eines Laufs, Rückschau 8, 16, 32, ..., exakt über die zusammengefasste Paarkette; null = nicht gerechnet",
    "t_rel": "1 / (1 − zweitgrößter Eigenwert) der Heat-bath-Matrix, Torus 4×4",
}


def rechne(lang):
    ring = []
    for n in RING_N:
        if n == 16 and not lang:
            continue
        t = time.time()
        ring += ring_zellen(n)
        print(f"Ring n = {n}: {time.time() - t:.1f} s", flush=True)
    torus = torus_zellen()
    takt = takt_zellen()
    return {"felder": FELDER, "ring": ring, "torus44": torus, "takt": takt}


LOCKER = {"t_rel": 1e-4, "tau_gipfel": 1e-4}   # bei sehr kleiner Spektrallücke (Torus 4×4, λ = 100: 1,75e-10) verliert die Rechnung Stellen, auf Linux gemessen 6e-6 Abweichung


def vergleiche(neu, alt, pfad="", rel=1e-6, schluessel=None):
    """Liste der Abweichungen zwischen zwei verschachtelten Strukturen (Zahlen relativ, sonst exakt; für die Schlüssel in LOCKER eine lockerere Grenze)."""
    fehler = []
    if isinstance(neu, dict):
        for k in neu:
            if k == "felder":
                continue
            if k not in alt:
                fehler.append(f"{pfad}/{k} fehlt in der Datei")
            else:
                fehler += vergleiche(neu[k], alt[k], f"{pfad}/{k}", rel, k)
    elif isinstance(neu, list):
        if len(neu) != len(alt):
            fehler.append(f"{pfad}: Länge {len(neu)} gegen {len(alt)}")
        else:
            for i, (a, b) in enumerate(zip(neu, alt)):
                fehler += vergleiche(a, b, f"{pfad}[{i}]", rel, schluessel)
    elif isinstance(neu, (int, float)) and isinstance(alt, (int, float)) and not isinstance(neu, bool):
        grenze = max(rel, LOCKER.get(schluessel, 0.0))
        if abs(neu - alt) > grenze * max(abs(neu), abs(alt), 1e-300):
            fehler.append(f"{pfad}: {neu} gegen {alt}")
    elif neu != alt:
        fehler.append(f"{pfad}: {neu!r} gegen {alt!r}")
    return fehler


def check(lang):
    """Rechnet neu und vergleicht mit der Datei; n = 16 nur mit `lang`. Rückgabe: Liste der Abweichungen (leer = in Ordnung)."""
    alt = json.loads(KOSTEN.read_text(encoding="utf-8"))
    neu = rechne(lang)
    if not lang:
        alt = dict(alt)
        alt["ring"] = [z for z in alt["ring"] if z["n"] != 16]
    return vergleiche(neu, alt)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Heat-bath auf dem Torus (Simulation): τ der Teilchenzahl, geeicht am 4×4-Torus
# ----------------------------------------------------------------------------------------------------------------------------------------
def simuliere_torus(a, b, lam, ketten, schritte, seed):
    """Heat-bath auf dem a×b-Torus, `ketten` Ketten parallel, Start leer; Rückgabe: Teilchenzahl nach jedem Schritt (ketten, schritte), int16."""
    V = a * b
    nbr = np.zeros((V, 4), dtype=np.int64)
    for i in range(a):
        for j in range(b):
            nbr[i * b + j] = [((i + 1) % a) * b + j, ((i - 1) % a) * b + j, i * b + (j + 1) % b, i * b + (j - 1) % b]
    strom = Strom(seed)
    plaetze = strom.unter(V, schritte * ketten).reshape(schritte, ketten)
    muenzen = strom.gleichverteilt(schritte * ketten).reshape(schritte, ketten)
    X = np.zeros((ketten, V), dtype=np.uint8)
    p = lam / (1.0 + lam)
    z = np.arange(ketten)
    anzahl = np.zeros(ketten, dtype=np.int64)
    aus = np.empty((ketten, schritte), dtype=np.int16)
    for t in range(schritte):
        v = plaetze[t]
        belegt_nachbarn = X[z[:, None], nbr[v]].sum(axis=1)
        neu = ((muenzen[t] < p) & (belegt_nachbarn == 0)).astype(np.uint8)
        anzahl += neu.astype(np.int64) - X[z, v]
        X[z, v] = neu
        aus[:, t] = anzahl
    return aus


def tau_schaetzung(reihen):
    """τ = (Ketten × Schritte) / ESS (Geyer-Folge, ohne Rangnormierung)."""
    reihen = reihen.astype(float)
    return reihen.size / G.ess_mehrketten(reihen)


def iat8():
    ergebnis = {"einheit": "Platzaktualisierungen je effektiv unabhängige Teilchenzahl-Probe (Heat-bath)", "schaetzer": "ESS nach Vehtari u. a. (Geyer-Folge), τ = Ketten × Schritte / ESS",
                "eichung_4x4": [], "torus8x8": []}
    raum = torus_raum(4, 4)
    for lam, seed in ((1.0, 30001), (2.0, 30002)):
        exakt = tau_fundamental(raum.heatbath(lam), raum.pi(lam), raum.k.astype(float))[0]
        reihen = simuliere_torus(4, 4, lam, 8, 100000, seed)[:, 10000:]
        geschaetzt = tau_schaetzung(reihen)
        ergebnis["eichung_4x4"].append({"lam": lam, "ketten": 8, "schritte": 100000, "verworfen": 10000, "zufalls_seed": seed, "tau_exakt": exakt,
                                        "tau_geschaetzt": geschaetzt, "verhaeltnis": geschaetzt / exakt})
        print("4x4", lam, exakt, geschaetzt, flush=True)
    for lam, seeds in ((1.0, (30003, 30005, 30007, 30009, 30011)), (5.0, (30004, 30006, 30008, 30010, 30012))):
        taus, dichten = [], []
        for seed in seeds:
            t = time.time()
            reihen = simuliere_torus(8, 8, lam, 8, 200000, seed)[:, 20000:]
            ess = G.ess_mehrketten(reihen.astype(float))
            taus.append(reihen.size / ess)
            dichten.append(float(reihen.mean() / 64))
            print("8x8", lam, seed, taus[-1], f"{time.time() - t:.0f} s", flush=True)
        ergebnis["torus8x8"].append({"lam": lam, "ketten": 8, "schritte": 200000, "verworfen": 20000, "zufalls_seeds": list(seeds), "taus": taus,
                                     "tau": float(np.median(taus)), "tau_min": min(taus), "tau_max": max(taus), "dichte_kette_mittel": float(np.mean(dichten)),
                                     "dichte_exakt": D.dichte_exakt(D.torus_streifen(8), 8, lam), "direkt": 64})
    IAT8.write_text(json.dumps(ergebnis, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schreibe", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lang", action="store_true")
    ap.add_argument("--iat8", action="store_true")
    a = ap.parse_args()
    if a.iat8:
        iat8()
        return 0
    if a.schreibe:
        if not a.lang:
            ap.error("--schreibe braucht --lang (alle Zellen)")
        KOSTEN.write_text(json.dumps(rechne(True), indent=1, ensure_ascii=False), encoding="utf-8")
        print("geschrieben:", KOSTEN)
        return 0
    if a.check:
        fehler = check(a.lang)
        print("OK" if not fehler else "\n".join(fehler))
        return 1 if fehler else 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
