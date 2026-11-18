"""Direktsampler über die Transfermatrix: Zählungen, exakte Wahrscheinlichkeit jeder Belegung (Brüche), Ziehungen gegen die exakte Verteilung, Randfälle."""

from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

import svw_direct as D
from svw_zufall import Strom

LUCAS_OEIS_A000032 = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349, 15127]


def zulaessig(bits, n):
    return all(not (bits[v] and bits[(v + 1) % n]) for v in range(n))


def torus_aufzaehlung(a, b):
    """Alle Belegungen des a×b-Torus durch Aufzählen aller Bitmasken (anderer Weg als die Transfermatrix)."""
    nb = []
    for i in range(a):
        for j in range(b):
            m = 0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                m |= 1 << (((i + di) % a) * b + (j + dj) % b)
            nb.append(m)
    ms = np.arange(1 << (a * b), dtype=np.int64)
    ok = np.ones(len(ms), dtype=bool)
    for v in range(a * b):
        ok &= ~((((ms >> v) & 1) == 1) & ((ms & nb[v]) != 0))
    return ms[ok]


def test_lucas_zahlen_gegen_oeis():
    assert [D.lucas(w) for w in range(len(LUCAS_OEIS_A000032))] == LUCAS_OEIS_A000032


def test_zeilenzustaende_gleich_aufzaehlung_und_lucas():
    for w in range(1, 15):
        assert len(D.ring_zeilen(w)) == D.zeilen_zustaende(w) == LUCAS_OEIS_A000032[w], w
    assert D.zeilen_zustaende(16) == 2207 and D.zeilen_zustaende(24) == 103682


def test_vertraegliche_paare_gleich_aufzaehlung():
    for w in range(1, 13):
        z = D.ring_zeilen(w)
        assert int(((z[:, None] & z[None, :]) == 0).sum()) == D.vertraegliche_paare(w), w
    assert [D.vertraegliche_paare(w) for w in (4, 8, 12, 16)] == [35, 1155, 39203, 1331715]


def test_dichte_matrix_mb_und_reichweite():
    assert D.dichte_matrix_mb(16) == pytest.approx(38.97, abs=0.005)
    assert D.dichte_matrix_mb(20) == pytest.approx(1830.6, abs=0.1)
    assert D.dichte_matrix_mb(24) == pytest.approx(85999.7, abs=0.1)
    r = D.reichweite(8)
    assert r["zeilen"] == 47 and r["paare"] == 1155 and r["paare_je_zeile"] == pytest.approx(24.57, abs=0.005)


def test_torus_zustaende_gegen_aufzaehlung():
    assert D.torus_zustaende(4) == len(torus_aufzaehlung(4, 4)) == 743
    assert D.torus_zustaende(8) == 213256442503
    assert D.torus_zustaende(12) == pytest.approx(3.0497e25, rel=1e-4)
    assert D.torus_zustaende(13) is None and D.torus_zustaende(2) is None


@pytest.mark.parametrize("n, lam", [(4, 2), (5, 3), (6, 2), (7, 2), (8, 3), (10, 2)])
def test_ring_direktsampler_jede_belegung_gleich_lambda_hoch_k_durch_z(n, lam):
    """Die Wahrscheinlichkeit des Samplers für jede der 2^n Bitfolgen ist λ^|x|/Z (zulässig) bzw. 0, Summe 1; Z aus der Aufzählung."""
    s = D.ring_streifen()
    folgen = {m: [(m >> v) & 1 for v in range(n)] for m in range(1 << n)}
    z_auf = sum(F(lam) ** sum(b) for b in folgen.values() if zulaessig(b, n))
    assert D.zustandssumme(s, n, F(lam)) == z_auf
    summe = F(0)
    for b in folgen.values():
        p = D.sequenz_wahrscheinlichkeit(s, n, F(lam), b)
        summe += p
        assert p == (F(lam) ** sum(b) / z_auf if zulaessig(b, n) else 0)
    assert summe == 1


@pytest.mark.parametrize("lam", [2, 3])
def test_torus_4x4_direktsampler_jede_belegung(lam):
    s = D.torus_streifen(4)
    zeilen = [int(x) for x in s.zeilen]
    belegungen = torus_aufzaehlung(4, 4)
    z_auf = sum(F(lam) ** bin(int(m)).count("1") for m in belegungen)
    assert D.zustandssumme(s, 4, F(lam)) == z_auf
    summe, gezaehlt = F(0), 0
    for folge in product(range(len(zeilen)), repeat=4):
        p = D.sequenz_wahrscheinlichkeit(s, 4, F(lam), folge)
        if p == 0:
            continue
        m = sum(zeilen[r] << (4 * i) for i, r in enumerate(folge))
        assert p == F(lam) ** bin(m).count("1") / z_auf
        summe += p
        gezaehlt += 1
    assert summe == 1 and gezaehlt == len(belegungen) == 743


def _z_je_zustand(haeufigkeit, erwartet, b):
    """Größter |z|-Wert über alle Zustände (z = (relative Häufigkeit − p) / sqrt(p (1 − p) / B))."""
    return float(np.abs((haeufigkeit - erwartet) / np.sqrt(erwartet * (1 - erwartet) / b)).max())


def test_ziehungen_ring_gegen_exakte_verteilung_im_band():
    n, lam, b = 8, 3.0, 40000
    s = D.ring_streifen()
    zeilen = D.ziehe_zeilen(s, n, lam, b, Strom(11))
    masken = (zeilen * (1 << np.arange(n))).sum(axis=1)                # Zeilenzustand 0/1 = Platz leer/belegt
    alle = np.array([m for m in range(1 << n) if zulaessig([(m >> v) & 1 for v in range(n)], n)])
    gew = np.array([lam ** bin(int(m)).count("1") for m in alle])
    p = gew / gew.sum()
    h = np.array([(masken == m).sum() for m in alle]) / b
    assert len(alle) == 47 and (h > 0).all()
    assert _z_je_zustand(h, p, b) < 4.5                                 # 47 Zustände, 4,5 Standardfehler


def test_ziehungen_torus_4x4_gegen_exakte_verteilung_im_band():
    lam, b = 2.0, 60000
    s = D.torus_streifen(4)
    zeilen = D.ziehe_zeilen(s, 4, lam, b, Strom(5))
    masken = sum(s.zeilen[zeilen[:, i]] << (4 * i) for i in range(4))
    belegungen = torus_aufzaehlung(4, 4)
    gew = np.array([lam ** bin(int(m)).count("1") for m in belegungen])
    p = gew / gew.sum()
    h = np.array([(masken == m).sum() for m in belegungen]) / b
    assert len(belegungen) == 743
    kenn = b * p >= 5                                                   # nur Zustände mit erwarteter Anzahl ab 5
    assert _z_je_zustand(h[kenn], p[kenn], b) < 4.8


def test_dichte_exakt_gegen_aufzaehlung_4x4_und_ring():
    """Der Eigenwertweg der exakten Dichte gleich der Summe über alle Belegungen (4×4: 743 Zustände; Ring n = 8: 47)."""
    belegungen = torus_aufzaehlung(4, 4)
    k = np.array([bin(int(m)).count("1") for m in belegungen])
    for lam in (0.5, 1.0, 3.0, 20.0):
        w = lam ** k.astype(float)
        assert D.dichte_exakt(D.torus_streifen(4), 4, lam) == pytest.approx(float((k * w).sum() / w.sum() / 16), rel=1e-12)
    ring = [sum(b[v] << v for v in range(8)) for b in product((0, 1), repeat=8) if zulaessig(list(b), 8)]
    kr = np.array([bin(m).count("1") for m in ring])
    w = 5.0 ** kr
    assert D.dichte_exakt(D.ring_streifen(), 8, 5.0) == pytest.approx(float((kr * w).sum() / w.sum() / 8), rel=1e-12)


def test_dichte_ueber_proben_8x8_und_12x12_gegen_transfermatrix():
    """7 Zellen (8×8: λ = 1, 5, 20, 100 mit 20 000 Proben; 12×12: λ = 1, 5, 20 mit 8 000 Proben): die Direktproben treffen die exakte Dichte innerhalb von 4 Standardfehlern."""
    zellen = [(8, 1, 20000), (8, 5, 20000), (8, 20, 20000), (8, 100, 20000), (12, 1, 8000), (12, 5, 8000), (12, 20, 8000)]
    zs = []
    for i, (w, lam, b) in enumerate(zellen):
        s = D.torus_streifen(w)
        anteil = D.belegung(s, D.ziehe_zeilen(s, w, float(lam), b, Strom(20003 + i))) / (w * w)
        zs.append((anteil.mean() - D.dichte_exakt(s, w, float(lam))) / (anteil.std(ddof=1) / np.sqrt(b)))
    assert max(abs(z) for z in zs) < 4
    assert sum(abs(z) < 4 for z in zs) == 7


def test_8x8_dichte_bei_lambda_1_gleich_bekannter_wert():
    assert D.dichte_exakt(D.torus_streifen(8), 8, 1.0) == pytest.approx(0.2270, abs=6e-5)
    assert D.dichte_exakt(D.torus_streifen(12), 12, 1.0) == pytest.approx(0.2266, abs=6e-5)


def test_gitter_ist_zulaessig():
    s = D.torus_streifen(8)
    g = D.gitter(s, D.ziehe_zeilen(s, 8, 5.0, 5, Strom(3))[0])
    assert g.shape == (8, 8)
    r = np.roll
    assert not ((g == 1) & ((r(g, 1, 0) == 1) | (r(g, -1, 0) == 1) | (r(g, 1, 1) == 1) | (r(g, -1, 1) == 1))).any()      # keine zwei Nachbarn belegt, auch über den Rand


# --- Eigenschaften ----------------------------------------------------------------------------------------------------------------------------

def test_eigenschaft_summe_der_wahrscheinlichkeiten_ist_eins_fuer_zufaellige_instanzen():
    rng = np.random.default_rng(2026)
    for _ in range(12):
        n = int(rng.integers(3, 8))
        lam = F(int(rng.integers(1, 7)), int(rng.integers(1, 4)))
        s = D.ring_streifen()
        assert sum(D.sequenz_wahrscheinlichkeit(s, n, lam, [(m >> v) & 1 for v in range(n)]) for m in range(1 << n)) == 1


def test_eigenschaft_dichte_waechst_mit_lambda():
    for w in (4, 6, 8):
        d = [D.dichte_exakt(D.torus_streifen(w), w, lam) for lam in (0.5, 1, 2, 5, 20, 100)]
        assert all(a < b for a, b in zip(d, d[1:]))
        assert d[-1] < 0.5                                              # höchstens die Hälfte der Plätze (Schachbrett)


def test_eigenschaft_ziehung_haengt_nur_am_zufalls_seed():
    s = D.torus_streifen(6)
    a = D.ziehe_zeilen(s, 6, 3.0, 50, Strom(8))
    assert np.array_equal(a, D.ziehe_zeilen(s, 6, 3.0, 50, Strom(8)))
    assert not np.array_equal(a, D.ziehe_zeilen(s, 6, 3.0, 50, Strom(9)))


# --- Randfälle --------------------------------------------------------------------------------------------------------------------------------

def test_randfaelle_breite_und_proben():
    with pytest.raises(ValueError):
        D.torus_streifen(2)                                             # doppelte Kanten
    with pytest.raises(ValueError):
        D.ring_zeilen(0)
    with pytest.raises(ValueError):
        D.zeilen_zustaende(0)
    with pytest.raises(ValueError):
        D.ziehe_zeilen(D.torus_streifen(3), 3, 1.0, 0, Strom(1))
    with pytest.raises(ValueError):
        D.symmetrische_matrix(D.torus_streifen(3), -1.0)
    assert D.torus_streifen(3).k.tolist() == [0, 1, 1, 1]               # Breite 3: leer und drei Einzelbelegungen


def test_randfall_eine_probe_und_zufalls_seed_0():
    s = D.torus_streifen(4)
    z = D.ziehe_zeilen(s, 4, 2.0, 1, Strom(0))
    assert z.shape == (1, 4)


def test_randfall_lambda_0_ist_der_leere_torus_und_sehr_grosses_lambda_ueberlaeuft_nicht():
    s = D.torus_streifen(4)
    z = D.ziehe_zeilen(s, 4, 0.0, 30, Strom(2))
    assert (D.belegung(s, z) == 0).all() and D.dichte_exakt(s, 4, 0.0) == 0.0
    g = D.ziehe_zeilen(s, 4, 1e8, 200, Strom(2))
    assert np.isfinite(D.dichte_exakt(s, 4, 1e8))
    assert D.dichte_exakt(s, 4, 1e8) == pytest.approx(0.5, abs=1e-6)    # Schachbrett: nur zwei Maximalbelegungen
    assert set(D.belegung(s, g).tolist()) == {8}


def test_randfall_kleinste_ringe():
    """n = 3 (4 Zustände); n = 2: Ring mit Doppelkante (die Transfermatrix zählt 3 Zustände, es ist kein Hard-Core-Ring); n = 1: nur der leere Zustand."""
    s = D.ring_streifen()
    assert D.zustandssumme(s, 3, F(2)) == 1 + 3 * 2
    assert D.zustandssumme(s, 2, F(2)) == 1 + 2 * 2
    assert D.zustandssumme(s, 1, F(2)) == 1
