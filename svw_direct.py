"""Direktsampler über die Transfermatrix für den Hard-Core-Ring und den Hard-Core-Torus (Stellplätze, keine zwei benachbarten belegt, Gewicht λ^k).

Das Ziel ist π(x) = λ^|x| / Z auf den zulässigen Belegungen x. Ein Streifen aus L Zeilen mit periodischem Rand lässt sich Zeile für Zeile ziehen: die
Zeile j+1 hängt nur von der Zeile j und der Zeile 0 ab (Vorwärts-Rückwärts-Verfahren mit Potenzen der Transfermatrix T[a][b] = [a und b vertragen sich] λ^|b|).
Bei Zeilenbreite 1 (nur zwei Zeilenzustände: leer, belegt) ist der Streifen der Hard-Core-Ring selbst.

    Z = Spur T^L
    Zeile 0:      P(r0 = a)                 = (T^L)[a][a] / Z
    Zeile j+1:    P(r(j+1) = b | rj = a, r0) = T[a][b] (T^(L-j-1))[b][r0] / (T^(L-j))[a][r0]

Die Wahrscheinlichkeit jeder Folge von Zeilen ist das Produkt dieser Faktoren und teleskopiert zu λ^|x| / Z. Gezählt wird in Platzentscheidungen (je Zeile eine
Entscheidung über alle Plätze der Zeile); es gibt kein Einbrennen. Die Breite des Streifens bestimmt die Größe der Matrix (Zeilenzustände = Lucas-Zahl L_w),
nicht die Zahl der Zustände des ganzen Torus.

Die Gleitkomma-Fassung ist vektorisiert (viele Proben gleichzeitig); die Bruch-Fassung (`sequenz_wahrscheinlichkeit`) rechnet dieselben Formeln mit `fractions` und
dient den Tests als exakte Wahrscheinlichkeit jeder Belegung."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

# Lucas-Zahlen: L_0 = 2, L_1 = 1, L_2 = 3, L_3 = 4, L_4 = 7 (OEIS A000032); L_w ist die Zahl der unabhängigen Mengen des w-Rings.


def lucas(w: int) -> int:
    a, b = 2, 1
    for _ in range(w):
        a, b = b, a + b
    return a


def zeilen_zustaende(w: int) -> int:
    """Zeilenzustände eines Streifens der Breite w (periodisch): unabhängige Mengen des w-Rings."""
    if w < 1:
        raise ValueError("Breite mindestens 1")
    return lucas(w)


def vertraegliche_paare(w: int) -> int:
    """Zahl der Zeilenpaare (a, b) mit a und b gleichzeitig zulässig übereinander: Spur M^w mit der 3×3-Matrix M des Rings aus den Zeichen leer, A, B.

    Herleitung: ein Paar legt auf jeden Platz einen von drei Zuständen (leer, nur a belegt = A, nur b belegt = B); gleiche Buchstaben dürfen auf dem Ring nie
    benachbart sein, leer verträgt sich mit allem."""
    if w < 1:
        raise ValueError("Breite mindestens 1")
    m = [[1, 1, 1], [1, 0, 1], [1, 1, 0]]
    r = [[int(i == j) for j in range(3)] for i in range(3)]
    for _ in range(w):
        r = [[sum(r[i][t] * m[t][j] for t in range(3)) for j in range(3)] for i in range(3)]
    return sum(r[i][i] for i in range(3))


def dichte_matrix_mb(w: int) -> float:
    """Speicher der dichten Matrix (Zeilenzustände zum Quadrat, 8 Byte je Eintrag) in Megabyte (1 MB = 10^6 Byte)."""
    z = zeilen_zustaende(w)
    return z * z * 8 / 1e6


def popcount(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = a.copy()
    c = np.zeros(a.shape, dtype=np.int64)
    while b.any():
        c += b & 1
        b >>= 1
    return c


def ring_zeilen(w: int) -> np.ndarray:
    """Alle Belegungen einer Zeile der Breite w (Bitmasken), in der keine zwei auf dem w-Ring benachbarten Plätze belegt sind (aufsteigend)."""
    if not 1 <= w <= 16:
        raise ValueError("Zeilen werden nur für Breite 1 bis 16 aufgezählt")
    voll = (1 << w) - 1
    m = np.arange(1 << w, dtype=np.int64)
    nachbarn = ((m << 1) | (m >> (w - 1))) & voll
    return m[(m & nachbarn) == 0]


@dataclass(frozen=True)
class Streifen:
    """Zeilenzustände eines Streifens: Zahl belegter Plätze je Zeilenzustand und welche Zeilen übereinander passen."""

    k: np.ndarray            # belegte Plätze je Zeilenzustand
    vertraeglich: np.ndarray  # bool[a, b]
    zeilen: np.ndarray | None = None   # Bitmaske der Zeile (nur Torus)
    breite: int = 1


def ring_streifen() -> Streifen:
    """Ring: Zeilenbreite 1 (Plätze nacheinander); Zustand 0 = leer, 1 = belegt, zwei belegte Nachbarn sind verboten."""
    return Streifen(np.array([0, 1]), np.array([[True, True], [True, False]]), np.array([0, 1]), 1)


def torus_streifen(w: int) -> Streifen:
    """Torus der Breite w: Zeilenzustände sind die zulässigen Zeilen, zwei Zeilen passen, wenn kein Platz in beiden belegt ist."""
    if w < 3:
        raise ValueError("Ein Torus braucht mindestens Breite 3 (kleinere haben doppelte Kanten)")
    z = ring_zeilen(w)
    return Streifen(popcount(z), (z[:, None] & z[None, :]) == 0, z, w)


def symmetrische_matrix(s: Streifen, lam: float) -> np.ndarray:
    """Symmetrische Fassung T[a][b] = [a ~ b] λ^((|a|+|b|)/2), durch den größten Eigenwert geteilt (Potenzen bleiben im Zahlenbereich).

    Die Faktoren λ^(|a|/2) kürzen sich in jedem geschlossenen Zug zu λ^|x|; das Teilen durch eine Konstante ändert keine der bedingten Wahrscheinlichkeiten oben (Zähler und Nenner enthalten gleich viele Faktoren)."""
    if lam < 0:
        raise ValueError("λ darf nicht negativ sein")
    sw = np.power(float(lam), s.k / 2.0)
    t = sw[:, None] * s.vertraeglich * sw[None, :]
    mu = float(np.linalg.eigvalsh(t).max())
    return t / mu


def potenzen(t: np.ndarray, anzahl: int) -> list[np.ndarray]:
    """[T^0, T^1, ..., T^anzahl]."""
    pw = [np.eye(len(t))]
    for _ in range(anzahl):
        pw.append(pw[-1] @ t)
    return pw


def ziehe_zeilen(s: Streifen, laenge: int, lam: float, proben: int, strom) -> np.ndarray:
    """`proben` exakte Ziehungen eines Streifens aus `laenge` Zeilen (periodisch); Rückgabe: Zeilenzustände (proben, laenge).

    `strom` liefert gleichverteilte Zahlen (`svw_zufall.Strom.gleichverteilt`); verbraucht werden proben * laenge Zahlen. Je Zeile eine Entscheidung."""
    if laenge < 1 or proben < 1:
        raise ValueError("Zeilenzahl und Zahl der Proben mindestens 1")
    t = symmetrische_matrix(s, lam)
    n = len(t)
    pw = potenzen(t, laenge)
    u = strom.gleichverteilt(proben * laenge).reshape(laenge, proben)
    d0 = np.clip(np.diag(pw[laenge]), 0.0, None)
    cdf0 = np.cumsum(d0)
    r0 = np.minimum(np.searchsorted(cdf0 / cdf0[-1], u[0], side="right"), n - 1)
    out = np.empty((proben, laenge), dtype=np.int64)
    out[:, 0] = r0
    cur = r0
    for j in range(laenge - 1):
        w = np.clip(t[cur, :] * pw[laenge - j - 1][:, r0].T, 0.0, None)
        cdf = np.cumsum(w, axis=1)
        ziel = u[j + 1] * cdf[:, -1]
        nxt = np.argmax(cdf >= ziel[:, None], axis=1)
        out[:, j + 1] = nxt
        cur = nxt
    return out


def belegung(s: Streifen, zeilen: np.ndarray) -> np.ndarray:
    """Zahl der belegten Plätze je Probe."""
    return s.k[zeilen].sum(axis=1)


def gitter(s: Streifen, zeilen_probe: np.ndarray) -> np.ndarray:
    """Eine Ziehung als 0/1-Gitter (Zeilen untereinander), nur für den Torus (Zeilen als Bitmasken)."""
    if s.zeilen is None:
        raise ValueError("kein Gitter ohne Bitmasken")
    masken = s.zeilen[zeilen_probe]
    return np.array([[(int(m) >> j) & 1 for j in range(s.breite)] for m in masken], dtype=np.int8)


def dichte_exakt(s: Streifen, laenge: int, lam: float) -> float:
    """Erwarteter Anteil belegter Plätze, exakt aus den Eigenwerten der symmetrischen Transfermatrix (anderer Weg als die Ziehung).

    E[Zahl belegter Plätze einer Zeile] = Σ_i μ_i^L (u_iᵀ K u_i) / Σ_i μ_i^L mit K = diag(k); durch die Breite geteilt."""
    t = symmetrische_matrix(s, lam)
    mu, u = np.linalg.eigh(t)
    rel = mu ** laenge
    z = rel.sum()
    ek = float((rel * ((s.k[:, None] * u * u).sum(axis=0))).sum() / z)
    return ek / s.breite


@lru_cache(maxsize=16)
def _bruch_potenzen(k: tuple, vertraeglich: tuple, laenge: int, lam: Fraction):
    """Matrix T (Brüche, ohne Normierung) und ihre Potenzen T^0 .. T^L; zwischengespeichert, weil jede Folge dieselben Potenzen braucht."""
    n = len(k)
    t = [[(lam ** k[b]) if vertraeglich[a][b] else Fraction(0) for b in range(n)] for a in range(n)]

    def mul(x, y):
        return [[sum((x[i][m] * y[m][j] for m in range(n)), Fraction(0)) for j in range(n)] for i in range(n)]

    pw = [[[Fraction(int(i == j)) for j in range(n)] for i in range(n)]]
    for _ in range(laenge):
        pw.append(mul(pw[-1], t))
    return t, pw


def _bruch_args(s: Streifen, laenge: int, lam):
    return tuple(int(x) for x in s.k), tuple(tuple(bool(x) for x in z) for z in s.vertraeglich), laenge, Fraction(lam)


def sequenz_wahrscheinlichkeit(s: Streifen, laenge: int, lam: Fraction, folge) -> Fraction:
    """Wahrscheinlichkeit, mit der der Direktsampler die Zeilenfolge (r_0, ..., r_(L-1)) zieht, als Bruch (T ohne Normierung, λ als Bruch)."""
    t, pw = _bruch_potenzen(*_bruch_args(s, laenge, lam))
    n = len(t)
    z = sum(pw[laenge][a][a] for a in range(n))
    r0 = folge[0]
    p = pw[laenge][r0][r0] / z
    for j in range(laenge - 1):
        a, b = folge[j], folge[j + 1]
        nenner = pw[laenge - j][a][r0]
        if nenner == 0:
            return Fraction(0)
        p *= t[a][b] * pw[laenge - j - 1][b][r0] / nenner
    return p


def zustandssumme(s: Streifen, laenge: int, lam: Fraction) -> Fraction:
    """Z = Spur T^L als Bruch."""
    _, pw = _bruch_potenzen(*_bruch_args(s, laenge, lam))
    return sum(pw[laenge][a][a] for a in range(len(pw[laenge])))


def torus_zustaende(w: int):
    """Zahl der Belegungen des w×w-Torus bei λ = 1: Spur (T01)^w mit der 0/1-Matrix der verträglichen Zeilen.

    Exakt (Ganzzahl) bis Breite 8, als Gleitkommazahl bis Breite 12, darüber nicht berechnet (None)."""
    if w < 3 or w > 12:
        return None
    z = ring_zeilen(w)
    if w <= 8:
        m = ((z[:, None] & z[None, :]) == 0).astype(object)
        r = np.identity(len(z), dtype=object)
        for _ in range(w):
            r = r.dot(m)
        return int(sum(r[i][i] for i in range(len(z))))
    mu = np.linalg.eigvalsh(((z[:, None] & z[None, :]) == 0).astype(float))
    return float((mu ** w).sum())


def reichweite(w: int) -> dict:
    """Kennzahlen eines Streifens der Breite w: Zeilenzustände, verträgliche Paare, dichte Matrix, Zahl der Torus-Zustände (falls berechnet)."""
    zz = zeilen_zustaende(w)
    paare = vertraegliche_paare(w)
    return {"w": w, "zeilen": zz, "paare": paare, "paare_je_zeile": paare / zz, "mb": dichte_matrix_mb(w), "zustaende": torus_zustaende(w)}


def probe_gegen_exakt(w: int, lam: float, proben: int, strom) -> dict:
    """Direktproben des w×w-Torus gegen die exakte Belegungsdichte: Mittel, Standardfehler (nan bei einer Probe), Abweichung in Standardfehlern, eine Probe als Gitter."""
    s = torus_streifen(w)
    zeilen = ziehe_zeilen(s, w, float(lam), proben, strom)
    anteil = belegung(s, zeilen) / (w * w)
    rho = dichte_exakt(s, w, float(lam))
    sf = float(anteil.std(ddof=1) / np.sqrt(proben)) if proben > 1 else float("nan")
    z = float((anteil.mean() - rho) / sf) if sf > 0 else 0.0
    return {"lam": lam, "proben": proben, "rho_exakt": rho, "rho_probe": float(anteil.mean()), "sf": sf, "z": z, "gitter": gitter(s, zeilen[0])}
