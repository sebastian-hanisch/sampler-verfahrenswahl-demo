"""Entscheidungsweg `waehle`: acht von Hand gesetzte Aufgaben, Eigenschaften über viele Aufgaben, Schwelle und Randfälle."""

import itertools

import pytest

import svw_constants as C
import svw_table as TB
import svw_texts as T
from svw_tree import Aufgabe, DIAGNOSE_BASIS, REPO, waehle


def test_acht_aufgaben_von_hand():
    a1 = waehle(Aufgabe(ziel="konjugat"))
    assert a1.weg == "direkt" and a1.diagnose == () and a1.tabellenzeilen == (1,) and REPO["exakt"] in a1.nachbarn
    a2 = waehle(Aufgabe(ziel="ring"))
    assert a2.weg == "direkt" and a2.zeilenzustaende == 2 and a2.diagnose == ()
    a3 = waehle(Aufgabe(ziel="torus", breite=8))
    assert a3.weg == "direkt" and a3.zeilenzustaende == 47 and a3.diagnose == ()                 # 47 Zeilenzustände: direkt
    a4 = waehle(Aufgabe(ziel="torus", breite=24, gipfel="mehrere"))
    assert a4.weg == "abtasten" and a4.zeilenzustaende == 103682 and "Tempering" in a4.kern
    assert "Starts in allen bekannten Gipfeln" in a4.diagnose and REPO["tempering"] in a4.nachbarn
    a5 = waehle(Aufgabe(ziel="gauss", dim=50))
    assert a5.weg == "direkt" and a5.nachbarn == () and a5.tabellenzeilen == (4,)
    a6 = waehle(Aufgabe(ziel="stetig", dim=48, gipfel="ein"))
    assert a6.weg == "abtasten" and "isotrop" in a6.kern and REPO["posterior"] in a6.nachbarn
    a7 = waehle(Aufgabe(ziel="stetig", dim=20, gradient=True, gipfel="ein"))
    assert a7.kern.startswith("HMC") and "zwei Teile" in a7.kern and 11 in a7.tabellenzeilen
    a8 = waehle(Aufgabe(ziel="torus", breite=30, kopplung_ok=True, gipfel="unbekannt"))
    assert REPO["kopplung"] in a8.nachbarn and REPO["diagnose"] in a8.nachbarn and "Kopplung aus der Vergangenheit" in a8.kern


def test_direkte_wege_haben_nie_eine_diagnoseliste_jeder_abtastweg_mindestens_fuenf_punkte():
    for ziel, gipfel, breite, dim, grad, kov, kop in itertools.product(C.ZIELE, C.GIPFEL, (3, 8, 16, 17, 40), (1, 10, 100), (False, True), (False, True), (False, True)):
        w = waehle(Aufgabe(ziel=ziel, breite=breite, dim=dim, gradient=grad, kovarianz=kov, gipfel=gipfel, kopplung_ok=kop))
        if w.weg == "direkt":
            assert w.diagnose == () and w.kern is None and w.zeilenzustaende in (None, 2) or ziel == "torus"
        else:
            assert len(w.diagnose) >= 5 and set(DIAGNOSE_BASIS) <= set(w.diagnose) and w.kern


def test_schwelle_2207_genau_dann_direkt_wenn_zeilenzustaende_bis_zur_grenze():
    for breite in range(C.BREITE_MIN, C.BREITE_MAX + 1):
        w = waehle(Aufgabe(ziel="torus", breite=breite))
        assert (w.weg == "direkt") == (w.zeilenzustaende <= C.ZEILEN_MAX), breite
    assert waehle(Aufgabe(ziel="torus", breite=16)).weg == "direkt" and waehle(Aufgabe(ziel="torus", breite=16)).zeilenzustaende == 2207
    assert waehle(Aufgabe(ziel="torus", breite=17)).weg == "abtasten" and waehle(Aufgabe(ziel="torus", breite=17)).zeilenzustaende == 3571


def test_stetig_wird_immer_abgetastet_ring_und_konjugat_nie():
    for dim in (1, 2, 50, 100):
        assert waehle(Aufgabe(ziel="stetig", dim=dim)).weg == "abtasten"
    assert all(waehle(Aufgabe(ziel=z, gipfel=g)).weg == "direkt" for z in ("ring", "konjugat", "gauss") for g in C.GIPFEL)


def test_dimension_1_mit_gradient_nimmt_den_zufallsschritt_nicht_hmc():
    w = waehle(Aufgabe(ziel="stetig", dim=1, gradient=True))
    assert not w.kern.startswith("HMC") and "Zufallsschritt" in w.kern


def test_bekannte_kovarianz_ohne_gradient_nimmt_zufallsschritt_mit_kovarianz():
    w = waehle(Aufgabe(ziel="stetig", dim=20, kovarianz=True))
    assert "Kovarianz als Vorschlag" in w.kern and "isotrop" not in w.kern


def test_fragen_pfad_ist_vollstaendig_und_hebt_den_erreichten_weg_hervor():
    direkt = waehle(Aufgabe(ziel="konjugat"))
    assert [f.nr for f in direkt.fragen] == [0, 1, 2, 3, 4] and [f.erreicht for f in direkt.fragen] == [True, False, False, False, False]
    mcmc = waehle(Aufgabe(ziel="stetig", dim=5))
    assert all(f.erreicht for f in mcmc.fragen) and mcmc.fragen[0].antwort.startswith("nein")


def test_tabellenzeilen_und_nachbarn_verweisen_auf_vorhandenes():
    nummern = {z["nr"] for z in TB.zeilen()}
    namen = {n for n, _ in T.NACHBARN}
    for ziel, gipfel, kop, grad in itertools.product(C.ZIELE, C.GIPFEL, (False, True), (False, True)):
        w = waehle(Aufgabe(ziel=ziel, breite=24, dim=20, gradient=grad, gipfel=gipfel, kopplung_ok=kop))
        assert set(w.tabellenzeilen) <= nummern
        assert set(w.nachbarn) <= namen, set(w.nachbarn) - namen
        assert len(set(w.nachbarn)) == len(w.nachbarn)


def test_randfaelle_ungueltige_aufgaben_geben_eine_meldung():
    with pytest.raises(ValueError):
        waehle(Aufgabe(ziel="unbekannt"))
    with pytest.raises(ValueError):
        waehle(Aufgabe(ziel="ring", gipfel="viele"))
    with pytest.raises(ValueError):
        waehle(Aufgabe(ziel="torus", breite=2))
    with pytest.raises(ValueError):
        waehle(Aufgabe(ziel="stetig", dim=0))
    waehle(Aufgabe(ziel="torus", breite=C.BREITE_MAX))              # obere Grenze inklusive
    waehle(Aufgabe(ziel="torus", breite=C.BREITE_MIN))              # untere Grenze inklusive
