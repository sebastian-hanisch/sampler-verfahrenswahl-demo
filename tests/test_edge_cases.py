"""Randfälle der Kernroutinen, die in den übrigen Dateien nicht stehen (DEMO-PLAYBOOK §12); Randfälle der Zeilenzustände, Ziehung, Ring, Kerne und Verdikte
stehen jeweils in `test_direct.py`, `test_diag.py`, `test_tree.py` und `test_verdicts.py`."""

import math

import numpy as np
import pytest

import svw_constants as C
import svw_cost as K
import svw_direct as D
import svw_diag as G
import svw_evaluation as ev
import svw_format as F
from svw_tree import Aufgabe, waehle
from svw_zufall import Strom


def test_eine_zeile_ziehen_und_dichte_mit_einer_zeile():
    """Ein Streifen aus nur einer Zeile (die Zeile liegt neben sich selbst): zulässig ist nur die leere Zeile."""
    s = D.torus_streifen(3)
    z = D.ziehe_zeilen(s, 1, 2.0, 10, Strom(1))
    assert z.shape == (10, 1) and (D.belegung(s, z) == 0).all()
    assert D.dichte_exakt(s, 1, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_probe_gegen_exakt_mit_einer_probe_hat_keinen_standardfehler_aber_keine_abweichung():
    r = D.probe_gegen_exakt(4, 2, 1, Strom(0))                              # B = 1, Zufalls-Seed 0
    assert math.isnan(r["sf"]) and r["z"] == 0.0 and r["gitter"].shape == (4, 4)
    r2 = D.probe_gegen_exakt(4, 2, 2, Strom(0))
    assert r2["sf"] >= 0 and not math.isnan(r2["sf"])


def test_probe_gegen_exakt_ist_ueber_den_zufalls_seed_reproduzierbar_und_enthaelt_alle_felder():
    a, b = D.probe_gegen_exakt(6, 5, 50, Strom(3)), D.probe_gegen_exakt(6, 5, 50, Strom(3))
    assert a["rho_probe"] == b["rho_probe"] and np.array_equal(a["gitter"], b["gitter"])
    assert {"lam", "proben", "rho_exakt", "rho_probe", "sf", "z", "gitter"} <= set(a)


def test_eine_kette_mit_einem_schritt_und_ring_n3():
    ring = G.Ring(3)
    reihen = G.simuliere_ring(ring, 1.0, "richtig", 1, 1, 5)
    assert reihen.shape == (1, 1) and 0 <= reihen[0, 0] <= 1                 # n = 3: höchstens ein Platz belegt
    assert G.vorprobe(ring, 2.0, "richtig")["max_abw"] < 1e-14


def test_ring_n16_aufzaehlung_stimmt_mit_den_zeilenzustaenden_ueberein():
    assert G.Ring(16).zahl == 2207


def test_erlaubte_werte_der_regler_liegen_innerhalb_der_grenzen_und_der_raster():
    assert C.BREITE_MIN <= C.DEFAULT_BREITE <= C.BREITE_MAX and C.DIM_MIN <= C.DEFAULT_DIM <= C.DIM_MAX
    assert C.DEFAULT_KOST_N in C.KOST_N and C.DEFAULT_KOST_LAM in C.KOST_LAM and C.DEFAULT_LAM_DICHTE in C.LAM_DICHTE
    assert C.LIVE_BREITE_MAX <= 12 and C.PROBEN_MIN <= C.DEFAULT_PROBEN <= C.PROBEN_MAX and C.REP_MIN <= C.DEFAULT_REP <= C.REP_MAX
    assert C.ZEILEN_MAX == D.zeilen_zustaende(16)
    assert C.DIAG_SCHRITTE - C.DIAG_VERWORFEN > 3


def test_jede_zelle_des_rasters_mit_ausnahme_des_nicht_gerechneten_hat_einen_balken():
    for n in C.KOST_N:
        for lam in C.KOST_LAM:
            b = K.balken("Ring", n, lam, "Teilchenzahl")
            assert (b is None) == (lam not in (1, 5, 20, 100))
    assert all(K.balken("Torus 4×4", 16, lam, f) is not None for lam in C.KOST_LAM for f in C.KOST_FUNKTIONEN)


def test_formatfunktionen_mit_grenzwerten():
    assert F.de(0.0, 0) == "0" and F.de(999.9996, 3) == "1 000,000" and F.zehnerpotenz(1e10) == "1,0·10¹⁰" and F.zehnerpotenz(-3e5, 1) == "−3,0·10⁵"
    assert F.gross(999999, 1) == "999 999" and F.gross(1e6, 1) == "1,0·10⁶"
    assert F.mb(10.0) == "10 MB" and F.mb(999.0) == "999 MB" and F.mb(1000.0) == "1,0 GB"


def test_verdikte_bei_extremen_aufgaben_ohne_fehler():
    for a in (Aufgabe(ziel="torus", breite=C.BREITE_MIN), Aufgabe(ziel="torus", breite=C.BREITE_MAX, kopplung_ok=True), Aufgabe(ziel="stetig", dim=C.DIM_MAX, gradient=True),
              Aufgabe(ziel="stetig", dim=1, gradient=True, kovarianz=True), Aufgabe(ziel="gauss", dim=100)):
        kind, text = ev.verdict_weg(a, waehle(a), None)
        assert kind in ("success", "info", "warning") and text
