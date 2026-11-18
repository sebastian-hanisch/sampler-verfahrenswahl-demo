"""Diagnostik: R̂ und ESS gegen eine einfache, andere Umsetzung (und scipy), Kerne des Rings, exakte Vorprobe, Wiederholungsexperiment."""

import numpy as np
import pytest

import svw_diag as G
import svw_reference as R

stats = pytest.importorskip("scipy.stats")


# --- R̂ und ESS gegen eine einfache Umsetzung ---------------------------------------------------------------------------------------------------

def ar1(m, n, phi, rng):
    x = np.zeros((m, n))
    x[:, 0] = rng.normal(size=m) / np.sqrt(1 - phi ** 2)
    e = rng.normal(size=(m, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def rhat_einfach(ketten):
    """R̂ aus der Definition mit Schleifen: Mittel je Kette, Zwischen- und Innerhalb-Varianz."""
    m, n = ketten.shape
    mittel = [sum(c) / n for c in ketten]
    gesamt = sum(mittel) / m
    b = n / (m - 1) * sum((mu - gesamt) ** 2 for mu in mittel)
    w = sum(sum((x - mu) ** 2 for x in c) / (n - 1) for c, mu in zip(ketten, mittel)) / m
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def ess_einfach(ketten):
    """ESS nach Geyer mit der Autokovarianz per Faltung (np.correlate), ohne FFT; gleiche Konvention (Stan: tau >= 1/log10(S))."""
    m, n = ketten.shape
    acov = np.array([np.correlate(c - c.mean(), c - c.mean(), "full")[n - 1:] / n for c in ketten])
    w = np.mean([c.var(ddof=1) for c in ketten])
    b = n * np.var([c.mean() for c in ketten], ddof=1)
    vp = (n - 1) / n * w + b / n
    rho = 1 - (w - acov.mean(axis=0)) / vp
    rho[0] = 1.0
    summe, t = [], 0
    while 2 * t + 1 < n:
        p = rho[2 * t] + rho[2 * t + 1]
        if p < 0:
            break
        summe.append(p if not summe else min(p, summe[-1]))
        t += 1
    tau = max(-1 + 2 * sum(summe), 1 / np.log10(m * n))
    return m * n / tau


def test_rhat_gleich_definition_und_halbierung():
    rng = np.random.default_rng(1)
    ch = ar1(4, 400, 0.5, rng)
    assert G.rhat_klassisch(ch) == pytest.approx(rhat_einfach(ch), rel=1e-12)
    assert G.split_rhat(ch) == pytest.approx(rhat_einfach(np.concatenate([ch[:, :200], ch[:, 200:]])), rel=1e-12)


def test_rhat_zeigt_getrennte_ketten_und_ist_nahe_1_bei_gleichen():
    rng = np.random.default_rng(2)
    gleich = rng.normal(size=(4, 2000))
    assert G.split_rhat(gleich) < 1.01
    getrennt = gleich + np.array([[0], [0], [5], [5]])
    assert G.split_rhat(getrennt) > 1.5


def test_ess_gleich_einfacher_umsetzung_und_bekanntem_ar1_tau():
    rng = np.random.default_rng(3)
    ch = ar1(4, 3000, 0.9, rng)
    assert G.ess_mehrketten(ch) == pytest.approx(ess_einfach(ch), rel=1e-9)
    tau_exakt = (1 + 0.9) / (1 - 0.9)                                    # 19 für AR(1) mit φ = 0,9
    assert 0.8 < (4 * 3000 / G.ess_mehrketten(ch)) / tau_exakt < 1.25     # Band um die exakte Zeit


def test_rangnormierung_gleich_scipy_auch_bei_gleichstaenden():
    rng = np.random.default_rng(4)
    ch = rng.integers(0, 7, size=(4, 500)).astype(float)                  # ganzzahlig: viele Gleichstände
    s = ch.size
    erwartet = stats.norm.ppf((stats.rankdata(ch.ravel(), method="average") - 0.375) / (s + 0.25)).reshape(ch.shape)
    assert np.abs(G.rang_normiert(ch) - erwartet).max() < 1e-12


def test_ess_bulk_ist_gleich_ess_der_rangnormierten_halbierten_ketten():
    rng = np.random.default_rng(5)
    ch = ar1(4, 1000, 0.7, rng)
    halb = np.concatenate([ch[:, :500], ch[:, 500:]])
    z = stats.norm.ppf((stats.rankdata(halb.ravel(), method="average") - 0.375) / (halb.size + 0.25)).reshape(halb.shape)
    assert G.ess_bulk(ch) == pytest.approx(ess_einfach(z), rel=1e-9)


# --- Kerne des Rings -------------------------------------------------------------------------------------------------------------------------

def test_richtiger_kern_pi_stationaer_falscher_kern_gleichverteilung_stationaer_und_symmetrisch():
    ring = G.Ring(12)
    pi = ring.pi(3.0)
    mr = G.kern_matrix(ring, 3.0, "richtig")
    mf = G.kern_matrix(ring, 3.0, "falsch")
    assert np.allclose(mr.sum(1), 1) and np.allclose(mf.sum(1), 1)
    assert np.abs(pi @ mr - pi).max() < 1e-12
    assert np.allclose(mf, mf.T)
    gleich = np.ones(ring.zahl) / ring.zahl
    assert np.abs(gleich @ mf - gleich).max() < 1e-14
    assert np.abs(pi @ mf - pi).max() > 5e-3


def test_vorprobe_findet_den_falschen_kern_und_laesst_den_richtigen_durch():
    ring = G.Ring(12)
    f, r = G.vorprobe(ring, 3.0, "falsch"), G.vorprobe(ring, 3.0, "richtig")
    assert ring.zahl == 322
    assert f["max_abw"] == pytest.approx(0.0109, abs=5e-5) and f["tv"] == pytest.approx(0.1189, abs=5e-5)
    assert f["max_abw"] > 0.1 * f["pi_max"] and f["db"] > 1e-3
    assert r["max_abw"] < 1e-12 and r["tv"] < 1e-12 and r["db"] < 1e-12


def test_exakter_mittelwert_der_teilchenzahl_ring_12_lambda_3_und_gleichverteilung():
    ring = G.Ring(12)
    assert float(ring.pi(3.0) @ ring.k) == pytest.approx(4.3395, abs=5e-5)
    assert float(ring.k.mean()) == pytest.approx(3.3168, abs=5e-5)         # Gleichverteilung auf den 322 Zuständen


def test_kontrollzelle_lambda_1_beide_kerne_haben_dasselbe_gleichgewicht():
    """Bei λ = 1 ist π die Gleichverteilung: auch der Kern ohne Annahmeregel hat sie als Gleichgewicht; die Vorprobe sieht keinen Unterschied."""
    ring = G.Ring(12)
    assert G.vorprobe(ring, 1.0, "falsch")["max_abw"] < 1e-14
    assert G.vorprobe(ring, 1.0, "richtig")["max_abw"] < 1e-14


def test_kern_n4_lambda2_von_hand_gegen_bruchrechnung_und_falschen_kern_zeile():
    ring = G.Ring(4)
    ref = R.Ring(4, 2)
    assert np.abs(G.kern_matrix(ring, 2.0, "richtig") - np.array([[float(x) for x in z] for z in ref.hb])).max() < 1e-15
    mf = G.kern_matrix(ring, 2.0, "falsch")
    i0 = int(ring.index(0))
    assert mf[i0, i0] == 0 and mf[i0, int(ring.index(1))] == pytest.approx(0.25)     # vom leeren Ring: jeder der 4 Plätze (1/4) wird belegt, nichts bleibt
    i5 = int(ring.index(5))
    assert mf[i5, int(ring.index(4))] == pytest.approx(0.25) and mf[i5, int(ring.index(1))] == pytest.approx(0.25)
    assert mf[i5, i5] == pytest.approx(0.5)                                              # Plätze 1 und 3 sind blockiert: bleibt mit 2/4


def test_ketten_gleichgewicht_im_band_beide_kerne():
    """Lange Ketten laufen gegen ihr Gleichgewicht: richtiger Kern gegen π, falscher gegen die Gleichverteilung (n = 6, λ = 3; Verteilung der Teilchenzahl)."""
    ring = G.Ring(6)
    for kern, gew in (("richtig", ring.pi(3.0)), ("falsch", np.ones(ring.zahl) / ring.zahl)):
        reihen = G.simuliere_ring(ring, 3.0, kern, 8, 60000, 77)[:, 2000:]
        erwartet = np.array([gew[ring.k == k].sum() for k in range(4)])
        h = np.array([(reihen == k).mean() for k in range(4)])
        assert np.abs(h - erwartet).max() < 0.01, kern


# --- Wiederholungsexperiment -----------------------------------------------------------------------------------------------------------------

def test_pruefe_wiederholungen_ist_bei_gleichem_zufalls_seed_reproduzierbar():
    a = G.pruefe_wiederholungen(12, 3.0, "richtig", 6, 1500, 500, 5)
    b = G.pruefe_wiederholungen(12, 3.0, "richtig", 6, 1500, 500, 5)
    c = G.pruefe_wiederholungen(12, 3.0, "richtig", 6, 1500, 500, 6)
    assert np.array_equal(a["mittel"], b["mittel"]) and not np.array_equal(a["mittel"], c["mittel"])


def test_pruefe_wiederholungen_zaehlungen_sind_konsistent():
    d = G.pruefe_wiederholungen(12, 3.0, "falsch", 10, 2000, 500, 9)
    assert d["n_beides"] <= min(d["n_rhat"], d["n_ess"]) and len(d["rhat"]) == 10
    assert d["n_ueber3"] == int((np.abs(d["z"]) > 3).sum())


def test_falscher_kern_besteht_die_diagnose_und_liegt_daneben_richtiger_kern_nicht():
    """20 Wiederholungen: falscher Kern in allen 20 mit Split-R̂ < 1,01 und ESS ≥ 400, aber das Mittel ist mehr als 10 Standardfehler daneben; richtiger Kern im Mittel unter 2."""
    f = G.pruefe_wiederholungen(12, 3.0, "falsch", 20, 4000, 1000, 20004)
    r = G.pruefe_wiederholungen(12, 3.0, "richtig", 20, 4000, 1000, 20004)
    assert f["n_beides"] == 20 and f["mittel_abs_z"] > 10 and f["mittel_mittel"] < 3.5
    assert r["mittel_abs_z"] < 2 and abs(r["mittel_mittel"] - r["exakt"]) < 0.1


def test_randfaelle_wiederholungen_und_ring():
    with pytest.raises(ValueError):
        G.pruefe_wiederholungen(12, 3.0, "richtig", 0, 4000, 1000, 1)
    with pytest.raises(ValueError):
        G.pruefe_wiederholungen(12, 3.0, "richtig", 5, 1003, 1000, 1)       # zu wenige Schritte nach dem Einbrennen
    with pytest.raises(ValueError):
        G.pruefe_wiederholungen(12, 3.0, "sonstwas", 5, 2000, 500, 1)
    with pytest.raises(ValueError):
        G.Ring(2)                                                            # Ring mit Doppelkante ist kein Hard-Core-Ring
    assert G.Ring(3).zahl == 4                                               # n = 3: leer und drei Einzelbelegungen
    with pytest.raises(ValueError):
        G.rhat_klassisch(np.ones((1, 10)))
    d = G.pruefe_wiederholungen(12, 3.0, "richtig", 1, 1200, 200, 0)         # eine Wiederholung, Zufalls-Seed 0
    assert d["wiederholungen"] == 1
