"""Orakel-Tests: Übergangsmatrizen, τ, Mischzeit und Kosten der Kopplung aus der Vergangenheit gegen einen anderen Rechenweg.

Rechenwege der Prüfung (nicht die der App):
  * exakte Bruchrechnung (`svw_reference`) für den Ring n = 4: Heat-bath-Matrix, d(t), t_mix, τ, erwartete Treffzeit und Kosten der Kopplung;
  * τ auf drei Wegen: Fundamentalmatrix (`tools/svw_make_kosten.py`), Spektralzerlegung der symmetrisierten Matrix und Summe der Autokorrelationen;
  * zwei verschieden gebaute Heat-bath-Matrizen (`tools/svw_make_kosten.py` und `svw_diag`) gegeneinander;
  * Simulation der Kopplung aus der Vergangenheit mit eigenem Code (numpy-Generator) gegen die exakten erwarteten Kosten, mit Band (4 Standardfehler);
  * Wiederholung der Messung am 4×4-Torus mit einer Aufzählung aller 743 Zustände von Hand.
Die Orakel selbst sind am Handbeispiel (Ring n = 4, λ = 2) geprüft: `tests/test_reference.py`."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import svw_cost as C
import svw_diag as G
import svw_reference as R
import svw_make_kosten as K


# --- Heat-bath-Matrix: zwei Bauweisen und die Brüche -----------------------------------------------------------------------------------------

def test_heatbath_matrix_tool_gleich_bruchrechnung_n4():
    ring = R.Ring(4, 2)
    raum = K.ring_raum(4)
    assert list(raum.s) == ring.s
    P = raum.heatbath(2.0)
    exakt = np.array([[float(x) for x in z] for z in ring.hb])
    assert np.abs(P - exakt).max() < 1e-15


@pytest.mark.parametrize("n, lam", [(6, 2.0), (8, 5.0), (10, 0.5)])
def test_heatbath_matrix_tool_gleich_svw_diag(n, lam):
    assert np.abs(K.ring_raum(n).heatbath(lam) - G.kern_matrix(G.Ring(n), lam, "richtig")).max() < 1e-15


def test_heatbath_ist_umkehrbar_und_pi_stationaer():
    for n, lam in ((6, 2.0), (8, 5.0)):
        raum = K.ring_raum(n)
        P, pi = raum.heatbath(lam), raum.pi(lam)
        assert np.abs(pi @ P - pi).max() < 1e-14
        fluss = pi[:, None] * P
        assert np.abs(fluss - fluss.T).max() < 1e-15


# --- τ auf drei Wegen -----------------------------------------------------------------------------------------------------------------------

def tau_spektral(P, pi, f):
    d = np.sqrt(pi)
    s = (d[:, None] * P) / d[None, :]
    ev, u = np.linalg.eigh(0.5 * (s + s.T))
    ft = f - pi @ f
    var = pi @ (ft * ft)
    c = (d * ft) @ u
    tau = 1.0
    for j in range(len(ev)):
        if ev[j] > 1 - 1e-9:
            continue
        tau += 2.0 * (c[j] ** 2 / var) * ev[j] / (1.0 - ev[j])
    return float(tau)


def tau_summe(P, pi, f, schritte):
    ft = f - pi @ f
    var = pi @ (ft * ft)
    v = ft.copy()
    tau = 1.0
    for _ in range(schritte):
        v = P @ v
        tau += 2.0 * float(pi @ (ft * v)) / var
    return tau


@pytest.mark.parametrize("n, lam, schritte", [(6, 2.0, 4000), (6, 5.0, 20000)])
def test_tau_teilchenzahl_und_gipfel_auf_drei_wegen(n, lam, schritte):
    raum = K.ring_raum(n)
    P, pi = raum.heatbath(lam), raum.pi(lam)
    for f in (raum.k.astype(float), raum.gipfel()):
        t1, _ = K.tau_fundamental(P, pi, f)
        assert t1 == pytest.approx(tau_spektral(P, pi, f), rel=1e-8)
        assert t1 == pytest.approx(tau_summe(P, pi, f, schritte), rel=1e-6)


def test_tau_n8_lambda5_fundamentalmatrix_gleich_spektralzerlegung():
    raum = K.ring_raum(8)
    P, pi = raum.heatbath(5.0), raum.pi(5.0)
    for f in (raum.k.astype(float), raum.gipfel()):
        assert K.tau_fundamental(P, pi, f)[0] == pytest.approx(tau_spektral(P, pi, f), rel=1e-8)


def test_tau_n4_tool_gleich_bruchrechnung():
    ring = R.Ring(4, 2)
    raum = K.ring_raum(4)
    P, pi = raum.heatbath(2.0), raum.pi(2.0)
    assert K.tau_fundamental(P, pi, raum.k.astype(float))[0] == pytest.approx(float(F(1195, 221)), rel=1e-12)
    assert K.tau_fundamental(P, pi, raum.gipfel())[0] == pytest.approx(53.0, rel=1e-12)
    assert float(ring.tau(ring.gipfelindikator())) == 53.0


def test_eigenschaft_tau_aendert_sich_nicht_bei_verschiebung_und_skalierung_der_funktion():
    raum = K.ring_raum(8)
    P, pi = raum.heatbath(3.0), raum.pi(3.0)
    f = raum.k.astype(float)
    t0 = K.tau_fundamental(P, pi, f)[0]
    for c, s in ((5.0, 1.0), (0.0, -2.5), (-3.0, 7.0)):
        assert K.tau_fundamental(P, pi, s * f + c)[0] == pytest.approx(t0, rel=1e-9)


def test_eigenschaft_drehung_der_plaetze_aendert_tau_der_teilchenzahl_nicht():
    """Die Teilchenzahl ist drehinvariant: eine Heat-bath-Matrix mit umbenannten Plätzen hat dasselbe τ (hier: Spiegelung der Plätze)."""
    n = 8
    raum = K.ring_raum(n)
    P, pi = raum.heatbath(5.0), raum.pi(5.0)
    spiegel = np.array([sum(((int(x) >> v) & 1) << ((n - 1 - v) % n) for v in range(n)) for x in raum.s])
    perm = raum.idx(spiegel)
    P2 = P[np.ix_(perm, perm)]
    f = raum.k.astype(float)
    assert K.tau_fundamental(P2, pi[perm], f[perm])[0] == pytest.approx(K.tau_fundamental(P, pi, f)[0], rel=1e-9)


# --- Mischzeit -------------------------------------------------------------------------------------------------------------------------------

def test_mischzeit_n4_tool_gleich_bruchrechnung_und_start_gerade_ist_der_schlechteste():
    raum = K.ring_raum(4)
    P, pi = raum.heatbath(2.0), raum.pi(2.0)
    assert K.tmix_alle_starts(P, pi) == R.Ring(4, 2).t_mix() == 22
    import scipy.sparse as sp
    for n, lam in ((4, 1.0), (4, 2.0), (6, 1.0), (6, 5.0), (8, 2.0), (8, 20.0)):
        raum = K.ring_raum(n)
        P, pi = raum.heatbath(lam), raum.pi(lam)
        assert K.tmix_alle_starts(P, pi) == K.tmix_start(sp.csr_matrix(P), pi, int(raum.idx(raum.gerade))), (n, lam)


def test_mischzeit_tool_gleich_schrittweiser_vektoriteration_ohne_quadrieren():
    """Anderer Weg: t_mix schrittweise aus P^t (alle Starts), ohne Quadrieren und Heraufzählen."""
    raum = K.ring_raum(6)
    P, pi = raum.heatbath(2.0), raum.pi(2.0)
    M, t = np.eye(len(pi)), 0
    while 0.5 * np.abs(M - pi).sum(axis=1).max() > 0.25:
        M, t = M @ P, t + 1
    assert K.tmix_alle_starts(P, pi) == t


# --- Kopplung aus der Vergangenheit ---------------------------------------------------------------------------------------------------------

def test_kopplungskosten_n4_tool_gleich_bruchrechnung():
    ring = R.Ring(4, 2)
    kette = K.paarkette(4)
    assert kette[3] <= len(ring.paarkette()[0]) == 19                         # die Bahnenkette hat höchstens so viele Zustände wie die 19 Paare der vollen Paarkette
    assert K.kopplung_kosten(kette, 4, 2.0) == pytest.approx(float(ring.kopplung_kosten(8)), rel=1e-9)


def cftp_simulation(n, lam, laeufe, rng, t0=8):
    """Eigene Simulation: Rückschau T0, 2 T0, ... mit wiederverwendeten Zufallszahlen; Rückgabe Ausgabe-Masken und Kosten (Schritte) je Lauf."""
    gerade = sum(1 << v for v in range(0, n, 2))
    ungerade = sum(1 << v for v in range(1, n, 2))
    nb = [(1 << ((v + 1) % n)) | (1 << ((v - 1) % n)) for v in range(n)]
    p = lam / (1 + lam)
    ausgaben, kosten = [], []
    for _ in range(laeufe):
        plaetze, muenzen = [], []                      # Index 0 = Zeitpunkt −1, Index k = Zeitpunkt −(k+1): die Zufallszahlen der späteren Zeitpunkte bleiben
        T, summe = t0, 0
        while True:
            while len(plaetze) < T:
                plaetze.append(int(rng.integers(0, n)))
                muenzen.append(rng.random() < p)
            o, u = gerade, ungerade
            for k in range(T - 1, -1, -1):
                v, m = plaetze[k], muenzen[k]
                bit = 1 << v
                o = (o & ~bit) | (bit if (m and not o & nb[v]) else 0)
                u = (u & ~bit) | (bit if (m and not u & nb[v]) else 0)
            summe += T
            if o == u:
                ausgaben.append(o)
                kosten.append(summe)
                break
            T *= 2
    return np.array(ausgaben), np.array(kosten)


def test_kopplungskosten_exakt_gegen_simulation_im_band_n8_lambda5():
    exakt = K.kopplung_kosten(K.paarkette(8), 8, 5.0)
    rng = np.random.default_rng(8123)
    aus, kosten = cftp_simulation(8, 5.0, 3000, rng)
    se = kosten.std(ddof=1) / math.sqrt(len(kosten))
    assert abs(kosten.mean() - exakt) < 4 * se
    raum = K.ring_raum(8)
    pi = raum.pi(5.0)
    haeufig = np.bincount(raum.idx(aus), minlength=raum.N) / len(aus)
    tv = 0.5 * np.abs(haeufig - pi).sum()
    rauschen = max(0.5 * np.abs(rng.multinomial(len(aus), pi) / len(aus) - pi).sum() for _ in range(20))
    assert tv <= 1.3 * rauschen                                           # die Ausgabe ist π-verteilt (bis auf Stichprobenrauschen)


def volle_paarkette_kosten(n, lam, t0=8):
    """Eigener Weg ohne Symmetriezusammenfassung: alle erreichbaren Paare (obere, untere Extremkette) als dichte Gleitkommamatrix, Kosten Σ_k T_k S(T_(k−1))."""
    gerade = sum(1 << v for v in range(0, n, 2))
    ungerade = sum(1 << v for v in range(1, n, 2))
    nb = [(1 << ((v + 1) % n)) | (1 << ((v - 1) % n)) for v in range(n)]
    p = lam / (1 + lam)
    nummer, liste, eintraege = {(gerade, ungerade): 0}, [(gerade, ungerade)], []
    i = 0
    while i < len(liste):
        o, u = liste[i]
        for v in range(n):
            for m in (0, 1):
                bit = 1 << v
                no = (o & ~bit) | (bit if (m and not o & nb[v]) else 0)
                nu = (u & ~bit) | (bit if (m and not u & nb[v]) else 0)
                if no != nu:
                    if (no, nu) not in nummer:
                        nummer[(no, nu)] = len(liste)
                        liste.append((no, nu))
                    eintraege.append((i, nummer[(no, nu)], (p if m else 1 - p) / n))
        i += 1
    Q = np.zeros((len(liste), len(liste)))
    for a_, b_, w in eintraege:
        Q[a_, b_] += w
    pw = np.linalg.matrix_power(Q, t0)
    s_, t, kosten = pw[0].sum(), t0, float(t0)
    while s_ >= 1e-15:
        kosten += 2 * t * s_
        pw, t = pw @ pw, 2 * t
        s_ = pw[0].sum()
    return kosten, len(liste)


@pytest.mark.parametrize("n, lam", [(6, 1.0), (8, 5.0)])
def test_paarkette_zusammengefasst_gleich_voller_paarkette(n, lam):
    """Die Bahnenkette (Symmetriezusammenfassung) liefert dieselben Kosten wie die volle Paarkette ohne Zusammenfassung; für n = 8 hat die volle Kette mehr Zustände."""
    voll, m_voll = volle_paarkette_kosten(n, lam)
    kette = K.paarkette(n)
    assert kette[3] < m_voll
    assert K.kopplung_kosten(kette, n, lam) == pytest.approx(voll, rel=1e-9)


# --- Torus 4×4: Aufzählung von Hand ----------------------------------------------------------------------------------------------------------

def torus_4x4_von_hand(lam):
    """Heat-bath-Matrix des 4×4-Torus durch Aufzählen aller 743 Belegungen (eigene Schleifen, ohne `Raum`); Rückgabe P, π, k, gerade-minus-ungerade."""
    nb = {}
    for i in range(4):
        for j in range(4):
            nb[4 * i + j] = [4 * ((i + 1) % 4) + j, 4 * ((i - 1) % 4) + j, 4 * i + (j + 1) % 4, 4 * i + (j - 1) % 4]
    zust = []
    for m in range(1 << 16):
        if all(not ((m >> v) & 1 and any((m >> u) & 1 for u in nb[v])) for v in range(16)):
            zust.append(m)
    index = {m: i for i, m in enumerate(zust)}
    p = lam / (1 + lam)
    P = np.zeros((len(zust), len(zust)))
    for i, m in enumerate(zust):
        for v in range(16):
            frei = not any((m >> u) & 1 for u in nb[v])
            leer = m & ~(1 << v)
            if frei:
                P[i, index[leer]] += (1 - p) / 16
                P[i, index[m | (1 << v)]] += p / 16
            else:
                P[i, index[leer]] += 1 / 16
    k = np.array([bin(m).count("1") for m in zust], dtype=float)
    schach = [v for v in range(16) if ((v // 4) + (v % 4)) % 2 == 0]
    s = np.array([sum((m >> v) & 1 for v in schach) - (bin(m).count("1") - sum((m >> v) & 1 for v in schach)) for m in zust])
    gipfel = np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0))
    w = lam ** k
    return P, w / w.sum(), k, gipfel


def test_torus_4x4_tau_eingefroren_gleich_aufzaehlung_von_hand_und_spektral():
    for lam in (2.0, 10.0):
        P, pi, k, gipfel = torus_4x4_von_hand(lam)
        assert len(pi) == 743 and np.abs(pi @ P - pi).max() < 1e-12
        z = C.torus_zelle(lam)
        assert z["tau_teilchen"] == pytest.approx(tau_spektral(P, pi, k), rel=1e-6)
        if lam == 2.0:                                                    # bei größerem λ überspringt die Spektralsumme (Lücke unter 1e-9) die langsame Mode des Gipfels
            assert z["tau_gipfel"] == pytest.approx(tau_spektral(P, pi, gipfel), rel=1e-6)


# --- ESS-Schätzer an einer exakten Größe geeicht (Eichung am 4×4-Torus) -----------------------------------------------------------------------

def test_ess_schaetzer_am_4x4_torus_gegen_exaktes_tau():
    """Die Schätzung τ = Ketten × Schritte / ESS auf dem 4×4-Torus (λ = 1; 8 Ketten × 100 000 Schritte) liegt im Band 0,85 bis 1,15 um das exakte τ."""
    z = C.torus_zelle(1.0)
    reihen = K.simuliere_torus(4, 4, 1.0, 8, 100000, 30001)[:, 10000:]
    ratio = K.tau_schaetzung(reihen) / z["tau_teilchen"]
    assert 0.85 < ratio < 1.15
