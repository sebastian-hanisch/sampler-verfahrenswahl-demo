"""Diagnostik gegen Richtigkeit: Hard-Core-Ring, ein richtiger und ein falscher Kern, Split-R̂, Bulk-ESS und die exakte Vorprobe π P = π.

Der richtige Kern ist der Heat-bath (Platz gleichverteilt, belegt mit Wahrscheinlichkeit λ/(1+λ), wenn beide Nachbarn leer sind, sonst leer). Der falsche Kern
vergisst die Annahmeregel: er wählt einen Platz gleichverteilt, kehrt die Belegung um und nimmt JEDEN zulässigen Vorschlag an. Seine Übergangsmatrix ist symmetrisch,
sein Gleichgewicht ist die Gleichverteilung auf den zulässigen Belegungen, nicht π ∝ λ^k.

R̂ und ESS folgen Vehtari u. a. (2021): Ketten halbiert (Split), klassisches R̂ aus Zwischen- und Innerhalb-Varianz; Bulk-ESS aus rangnormierten halbierten Ketten mit
der Geyer-Folge der Autokorrelations-Paare. Die Formeln stehen ausführlicher in `posterior-sampling-demo`; hier nur eine kleine Fassung für ganzzahlige Reihen."""

from statistics import NormalDist

import numpy as np

import svw_direct as D
from svw_zufall import Strom

KERNE = ("richtig", "falsch")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Zustandsraum des Rings und Übergangsmatrizen (exakt)
# ----------------------------------------------------------------------------------------------------------------------------------------
class Ring:
    """Alle zulässigen Belegungen des n-Rings als Bitmasken (aufsteigend); n mindestens 3 (kleinere Ringe haben doppelte Kanten)."""

    def __init__(self, n: int):
        if not 3 <= n <= 16:
            raise ValueError("Ring: n von 3 bis 16")
        self.n = n
        self.nachbarn = np.array([(1 << ((v + 1) % n)) | (1 << ((v - 1) % n)) for v in range(n)], dtype=np.int64)
        self.zustaende = D.ring_zeilen(n)
        self.k = D.popcount(self.zustaende)
        self.gerade = sum(1 << v for v in range(0, n, 2))

    @property
    def zahl(self) -> int:
        return len(self.zustaende)

    def index(self, masken) -> np.ndarray:
        return np.searchsorted(self.zustaende, masken)

    def pi(self, lam: float) -> np.ndarray:
        if lam < 0:
            raise ValueError("λ darf nicht negativ sein")
        w = np.power(float(lam), self.k.astype(float))
        return w / w.sum()


def kern_matrix(ring: Ring, lam: float, kern: str) -> np.ndarray:
    """Dichte Übergangsmatrix des Kerns (Zeile = von, Spalte = nach)."""
    if kern not in KERNE:
        raise ValueError(f"Kern: {KERNE}")
    n, nz = ring.n, ring.zahl
    p = lam / (1.0 + lam)
    z = np.arange(nz)
    m = np.zeros((nz, nz))
    for v in range(n):
        bit = 1 << v
        belegt = (ring.zustaende & bit) != 0
        frei = (ring.zustaende & ring.nachbarn[v]) == 0
        if kern == "richtig":
            leer = ring.index(ring.zustaende & ~bit)
            q = np.where(frei, p, 0.0)
            m[z, leer] += (1.0 - q) / n
            f = z[frei]
            m[f, ring.index(ring.zustaende[frei] | bit)] += p / n
        else:
            geht = belegt | frei                       # umkehren ist zulässig: ein belegter Platz wird leer, ein freier Platz belegt
            ziel = ring.index(ring.zustaende ^ bit)
            m[z[geht], ziel[geht]] += 1.0 / n
            m[z[~geht], z[~geht]] += 1.0 / n
    return m


def vorprobe(ring: Ring, lam: float, kern: str) -> dict:
    """Exakte Prüfung π P = π: größte Abweichung, TV-Abstand von π P zu π und größte Abweichung der Detailed Balance."""
    pi = ring.pi(lam)
    m = kern_matrix(ring, lam, kern)
    fluss = pi[:, None] * m
    return {"max_abw": float(np.abs(pi @ m - pi).max()), "tv": float(0.5 * np.abs(pi @ m - pi).sum()), "db": float(np.abs(fluss - fluss.T).max()),
            "pi_max": float(pi.max())}


# ----------------------------------------------------------------------------------------------------------------------------------------
# Ketten
# ----------------------------------------------------------------------------------------------------------------------------------------
def simuliere_ring(ring: Ring, lam: float, kern: str, ketten: int, schritte: int, seed: int, start=None) -> np.ndarray:
    """`ketten` Ketten parallel, `schritte` Schritte; Rückgabe: Teilchenzahl nach jedem Schritt (ketten, schritte), int8.

    Start: Voreinstellung zwei von vier Ketten leer und zwei „alle geraden Plätze“ (überstreut). Plätze und Münzen stammen aus demselben Strom
    (zuerst alle Plätze, dann alle Münzen), beide Kerne bekommen bei gleichem Zufalls-Seed also dieselben Zufallszahlen."""
    if kern not in KERNE:
        raise ValueError(f"Kern: {KERNE}")
    n = ring.n
    strom = Strom(seed)
    plaetze = strom.unter(n, schritte * ketten).reshape(schritte, ketten)
    muenzen = strom.gleichverteilt(schritte * ketten).reshape(schritte, ketten)
    if start is None:
        x = np.where(np.arange(ketten) % 4 < 2, 0, ring.gerade).astype(np.int64)
    else:
        x = np.asarray(start, dtype=np.int64).copy()
    p = lam / (1.0 + lam)
    pop = D.popcount(np.arange(1 << n))
    ausgabe = np.empty((ketten, schritte), dtype=np.int8)
    for t in range(schritte):
        v = plaetze[t]
        bit = np.left_shift(np.int64(1), v)
        frei = (x & ring.nachbarn[v]) == 0
        if kern == "richtig":
            kopf = muenzen[t] < p
            x = (x & ~bit) | np.where(frei & kopf, bit, 0)
        else:
            belegt = (x & bit) != 0
            x = np.where(belegt, x & ~bit, np.where(frei, x | bit, x))
        ausgabe[:, t] = pop[x]
    return ausgabe


# ----------------------------------------------------------------------------------------------------------------------------------------
# R̂ und ESS (Vehtari u. a. 2021, kleine Fassung)
# ----------------------------------------------------------------------------------------------------------------------------------------
def halbiere(ketten: np.ndarray) -> np.ndarray:
    """Jede Kette in eine erste und eine zweite Hälfte teilen (bei ungerader Länge fällt der mittlere Wert weg)."""
    m, n = ketten.shape
    h = n // 2
    return np.concatenate([ketten[:, :h], ketten[:, n - h:]], axis=0)


def rhat_klassisch(ketten: np.ndarray) -> float:
    m, n = ketten.shape
    if m < 2 or n < 2:
        raise ValueError("mindestens zwei Ketten mit zwei Werten")
    b = n * ketten.mean(axis=1).var(ddof=1)
    w = ketten.var(axis=1, ddof=1).mean()
    if w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def split_rhat(ketten: np.ndarray) -> float:
    return rhat_klassisch(halbiere(ketten))


def _autokov(x: np.ndarray) -> np.ndarray:
    n = len(x)
    xc = x - x.mean()
    groesse = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, groesse)
    return np.fft.irfft(f * np.conj(f), groesse)[:n] / n


def ess_mehrketten(ketten: np.ndarray) -> float:
    """Effektive Stichprobengröße über alle Ketten (Geyer-Folge, Obergrenze der Zeit nach Stan: tau >= 1/log10(S))."""
    m, n = ketten.shape
    acov = np.array([_autokov(c) for c in ketten])
    w = ketten.var(axis=1, ddof=1).mean()
    b = n * ketten.mean(axis=1).var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * w + b / n
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    paare = n // 2
    summe = rho[0:2 * paare:2] + rho[1:2 * paare:2]
    negativ = np.where(summe < 0)[0]
    stopp = negativ[0] if len(negativ) else paare
    stopp = max(stopp, 1)
    summe = np.minimum.accumulate(summe[:stopp])
    tau = -1.0 + 2.0 * summe.sum()
    s = m * n
    tau = max(tau, 1.0 / np.log10(s))
    return float(s / tau)


def rang_normiert(ketten: np.ndarray) -> np.ndarray:
    """Durchschnittsränge über alle Werte, dann z = Φ^-1((r − 3/8) / (S + 1/4)); Gleichstände (ganzzahlige Reihen) bekommen denselben Rang."""
    werte, rueck, anzahl = np.unique(ketten.ravel(), return_inverse=True, return_counts=True)
    rang = np.cumsum(anzahl) - (anzahl - 1) / 2.0
    s = ketten.size
    nd = NormalDist()
    z = np.array([nd.inv_cdf((r - 0.375) / (s + 0.25)) for r in rang])
    return z[rueck].reshape(ketten.shape)


def ess_bulk(ketten: np.ndarray) -> float:
    return ess_mehrketten(rang_normiert(halbiere(ketten)))


# ----------------------------------------------------------------------------------------------------------------------------------------
# Das Experiment „Wer täuscht wen?“
# ----------------------------------------------------------------------------------------------------------------------------------------
def pruefe_wiederholungen(n: int, lam: float, kern: str, wiederholungen: int, schritte: int, verworfen: int, seed: int, start: str = "gestreut") -> dict:
    """Je Wiederholung vier Ketten × `schritte` Schritte, die ersten `verworfen` Schritte entfallen; Funktion: Teilchenzahl.

    Rückgabe: je Wiederholung Split-R̂, Bulk-ESS, Mittel, Standardfehler (Standardabweichung / √ESS der Ketten ohne Rangnormierung), z = (Mittel − exakt)/SF,
    dazu das exakte Mittel und die Zählungen (R̂ < 1,01; ESS ≥ 400; beides). `start`: „gestreut“ (überstreut: zwei Ketten leer, zwei „alle geraden Plätze“) oder „leer“ (alle leer)."""
    if schritte <= verworfen + 3 or wiederholungen < 1:
        raise ValueError("zu wenige Schritte oder Wiederholungen")
    ring = Ring(n)
    exakt = float(ring.pi(lam) @ ring.k)
    if start not in ("gestreut", "leer"):
        raise ValueError("Start: gestreut oder leer")
    reihen = simuliere_ring(ring, lam, kern, 4 * wiederholungen, schritte, seed, start=np.zeros(4 * wiederholungen, dtype=np.int64) if start == "leer" else None)
    rh, es, mw, sf = [], [], [], []
    for r in range(wiederholungen):
        ch = reihen[4 * r:4 * r + 4, verworfen:].astype(float)
        rh.append(split_rhat(ch))
        es.append(ess_bulk(ch))
        mw.append(ch.mean())
        sf.append(ch.std(ddof=1) / np.sqrt(ess_mehrketten(ch)))
    rh, es, mw, sf = map(np.array, (rh, es, mw, sf))
    z = (mw - exakt) / sf
    gut = (rh < 1.01) & (es >= 400)
    return {"rhat": rh, "ess": es, "mittel": mw, "sf": sf, "z": z, "exakt": exakt, "n_rhat": int((rh < 1.01).sum()), "n_ess": int((es >= 400).sum()),
            "n_beides": int(gut.sum()), "wiederholungen": wiederholungen, "mittel_abs_z": float(np.abs(z).mean()),
            "mittel_abs_z_beides": float(np.abs(z[gut]).mean()) if gut.any() else float("nan"),
            "n_ueber3": int((np.abs(z) > 3).sum()), "mittel_mittel": float(mw.mean())}
