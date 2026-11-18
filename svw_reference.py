"""Exakte Bruchrechnung (Python-`fractions`) für den kleinen Hard-Core-Ring: das Handbeispiel der App und die Orakel der Tests.

Alles hier rechnet mit Brüchen und ohne numpy; es ist der Rechenweg, an dem sich die Gleitkomma-Fassungen der App messen lassen. Gedacht für kleine n (n = 4:
sieben Zustände); größere n rechnen mit Brüchen sehr lange.

Modell: n Stellplätze im Kreis, keine zwei benachbarten belegt, Gewicht λ^|x| (ausgedachtes Modell). Heat-bath: Platz gleichverteilt, belegt mit Wahrscheinlichkeit
λ/(1+λ), wenn beide Nachbarn leer sind, sonst leer."""

from fractions import Fraction as F


def nachbarn(n: int) -> list[int]:
    return [(1 << ((v + 1) % n)) | (1 << ((v - 1) % n)) for v in range(n)]


def zustaende(n: int) -> list[int]:
    nb = nachbarn(n)
    return [m for m in range(1 << n) if all(not ((m >> v) & 1 and (m & nb[v])) for v in range(n))]


def _mul(a, b):
    return [[sum((a[i][t] * b[t][j] for t in range(len(b))), F(0)) for j in range(len(b[0]))] for i in range(len(a))]


def _vec_mat(v, a):
    return [sum((v[i] * a[i][j] for i in range(len(v))), F(0)) for j in range(len(a[0]))]


def _pot(a, t):
    r = [[F(int(i == j)) for j in range(len(a))] for i in range(len(a))]
    while t:
        if t & 1:
            r = _mul(r, a)
        a = _mul(a, a)
        t >>= 1
    return r


def loese(a, b):
    """Gaußsches Verfahren in Brüchen: a x = b."""
    n = len(a)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        pv = m[c][c]
        m[c] = [x / pv for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


class Ring:
    """Ring mit n Plätzen und Gewicht λ (Bruch): Zustände, π, Heat-bath-Matrix, TV-Abstand, τ und Kosten der Kopplung aus der Vergangenheit, alles exakt."""

    def __init__(self, n: int, lam):
        self.n = n
        self.lam = F(lam)
        self.nb = nachbarn(n)
        self.s = zustaende(n)
        self.k = [bin(m).count("1") for m in self.s]
        self.gewicht = [self.lam ** k for k in self.k]
        self.z = sum(self.gewicht)
        self.pi = [w / self.z for w in self.gewicht]
        self.hb = self._heatbath()

    def idx(self, m: int) -> int:
        return self.s.index(m)

    def _heatbath(self):
        p = self.lam / (1 + self.lam)
        N = len(self.s)
        P = [[F(0)] * N for _ in range(N)]
        for i, x in enumerate(self.s):
            for v in range(self.n):
                bit = 1 << v
                leer = self.idx(x & ~bit)
                if (x & self.nb[v]) == 0:
                    P[i][leer] += (1 - p) / self.n
                    P[i][self.idx(x | bit)] += p / self.n
                else:
                    P[i][leer] += F(1, self.n)
        return P

    def stationaer(self) -> bool:
        return _vec_mat(self.pi, self.hb) == self.pi

    def detailed_balance(self) -> bool:
        N = len(self.s)
        return all(self.pi[i] * self.hb[i][j] == self.pi[j] * self.hb[j][i] for i in range(N) for j in range(N))

    def tv_je_start(self, pt):
        return [sum(abs(pt[i][j] - self.pi[j]) for j in range(len(self.s))) / 2 for i in range(len(self.s))]

    def d(self, t: int) -> F:
        """d(t) = max über alle Starts von TV(P^t(x, ·), π)."""
        return max(self.tv_je_start(_pot(self.hb, t)))

    def t_mix(self, eps=F(1, 4), tmax=2000) -> int:
        """Erstes t mit d(t) <= eps (Matrixpotenzen in Brüchen, t schrittweise)."""
        pt = [[F(int(i == j)) for j in range(len(self.s))] for i in range(len(self.s))]
        for t in range(tmax + 1):
            if max(self.tv_je_start(pt)) <= eps:
                return t
            pt = _mul(pt, self.hb)
        raise RuntimeError("t_mix nicht erreicht")

    def tau(self, f) -> F:
        """τ = 1 + 2 Σ ρ_k exakt: g = (I − P + 1πᵀ)⁻¹ f̃, σ² = 2⟨f̃, g⟩_π − ⟨f̃, f̃⟩_π, τ = σ² / Var f."""
        N = len(self.s)
        ef = sum(p * x for p, x in zip(self.pi, f))
        ft = [x - ef for x in f]
        var = sum(p * x * x for p, x in zip(self.pi, ft))
        a = [[F(int(i == j)) - self.hb[i][j] + self.pi[j] for j in range(N)] for i in range(N)]
        g = loese(a, ft)
        return (2 * sum(p * x * y for p, x, y in zip(self.pi, ft, g)) - var) / var

    def teilchenzahl(self):
        return [F(k) for k in self.k]

    def gipfelindikator(self):
        """I = 1 für s > 0, 1/2 für s = 0, 0 für s < 0 mit s = belegte gerade − belegte ungerade Plätze."""
        gerade = sum(1 << v for v in range(0, self.n, 2))
        ungerade = sum(1 << v for v in range(1, self.n, 2))
        s = [bin(x & gerade).count("1") - bin(x & ungerade).count("1") for x in self.s]
        return [F(1) if v > 0 else (F(1, 2) if v == 0 else F(0)) for v in s]

    # --- Kopplung aus der Vergangenheit -------------------------------------------------------------------------------------------
    def schritt(self, x: int, v: int, muenze: int) -> int:
        bit = 1 << v
        return (x & ~bit) | (bit if (muenze and not x & self.nb[v]) else 0)

    def paarkette(self):
        """Erreichbare Paare (obere Kette = alle geraden Plätze, untere = alle ungeraden), Matrix Q ohne den Zustand „zusammengetroffen“."""
        gerade = sum(1 << v for v in range(0, self.n, 2))
        ungerade = sum(1 << v for v in range(1, self.n, 2))
        p = self.lam / (1 + self.lam)
        nummer = {(gerade, ungerade): 0}
        liste = [(gerade, ungerade)]
        zeilen = []
        i = 0
        while i < len(liste):
            o, u = liste[i]
            zeile = {}
            for v in range(self.n):
                for m in (0, 1):
                    w = F(1, self.n) * (p if m else 1 - p)
                    no, nu = self.schritt(o, v, m), self.schritt(u, v, m)
                    if no == nu:
                        j = -1
                    else:
                        if (no, nu) not in nummer:
                            nummer[(no, nu)] = len(liste)
                            liste.append((no, nu))
                        j = nummer[(no, nu)]
                    zeile[j] = zeile.get(j, F(0)) + w
            zeilen.append(zeile)
            i += 1
        q = [[zeile.get(j, F(0)) for j in range(len(liste))] for zeile in zeilen]
        return liste, q

    def erwartete_treffzeit(self) -> F:
        """E[τ]: Erwartete Zahl der Schritte, bis sich die beiden Extremketten bei gemeinsamem Platz und gemeinsamer Münze treffen: (I − Q) e = 1."""
        liste, q = self.paarkette()
        m = len(liste)
        a = [[F(int(i == j)) - q[i][j] for j in range(m)] for i in range(m)]
        return loese(a, [F(1)] * m)[0]

    def kopplung_kosten(self, t0: int = 8) -> F:
        """Erwartete Kosten eines Laufs mit Rückschau t0, 2 t0, 4 t0, ...: Σ_k T_k S(T_(k−1)), S(T) = Wahrscheinlichkeit, dass die Extremketten nach T Schritten noch
        verschieden sind. Gerechnet, bis S unter 1e-15 fällt (der Rest ist kleiner als 1e-12 Schritte)."""
        liste, q = self.paarkette()
        m = len(liste)
        v0 = [F(1)] + [F(0)] * (m - 1)
        pw = _pot(q, t0)
        s = sum(_vec_mat(v0, pw))
        kosten = F(t0)
        t = t0
        while s >= F(1, 10 ** 15):
            kosten += 2 * t * s
            pw = _mul(pw, pw)
            t *= 2
            s = sum(_vec_mat(v0, pw))
        return kosten

    def kopplungslauf(self, zuege, rueckschau: int):
        """Ein Lauf von Hand: `zuege` = {Zeit: (Platz, Münze)} für Zeiten −T..−1; beide Extremketten starten bei −T und bekommen dieselben Zufallszahlen.
        Rückgabe (obere, untere Kette) bei Zeit 0."""
        o = sum(1 << v for v in range(0, self.n, 2))
        u = sum(1 << v for v in range(1, self.n, 2))
        for t in range(-rueckschau, 0):
            v, m = zuege[t]
            o, u = self.schritt(o, v, m), self.schritt(u, v, m)
        return o, u


def direkt_wahrscheinlichkeit(n: int, lam, bits) -> F:
    """Wahrscheinlichkeit, die der Direktsampler (Transfermatrix 2×2) der Belegung `bits` (Liste aus 0/1, Länge n) gibt; unzulässige Belegungen bekommen 0."""
    lam = F(lam)
    a = [[F(1), lam], [F(1), F(0)]]
    pw = [[[F(int(i == j)) for j in range(2)] for i in range(2)]]
    for _ in range(n):
        pw.append(_mul(pw[-1], a))
    z = pw[n][0][0] + pw[n][1][1]
    x0 = bits[0]
    pr = pw[n][x0][x0] / z
    for i in range(n - 1):
        s, t = bits[i], bits[i + 1]
        nenner = pw[n - i][s][x0]
        if nenner == 0:
            return F(0)
        pr *= a[s][t] * pw[n - i - 1][t][x0] / nenner
    # Schluss: Platz n-1 gegen Platz 0 (Ring) steckt in (A^1)[t][x0] des letzten Faktors
    return pr


def gamma_poisson(form: F, rate: F, summe: int, perioden: int):
    """Konjugater Posterior einer Poisson-Rate: Prior Gamma(form, rate), `perioden` Beobachtungen mit Summe `summe` -> Gamma(form + summe, rate + perioden).

    Rückgabe (form, rate, Mittel als Bruch, Standardabweichung als float = √form / rate)."""
    a, b = F(form) + summe, F(rate) + perioden
    return a, b, a / b, float(a) ** 0.5 / float(b)
