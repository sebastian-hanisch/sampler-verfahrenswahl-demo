"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text, jede Art und jede Schwelle steht hier."""

import numpy as np

import svw_evaluation as ev
import svw_cost as K
import svw_direct as D
from svw_format import de, ganz, gross, mb, zehnerpotenz
from svw_tree import Aufgabe, waehle


# --- Zahlenformat ----------------------------------------------------------------------------------------------------------------------------

def test_zahlenformat_dezimalkomma_tausender_und_zehnerpotenz():
    assert de(60.56, 1) == "60,6" and de(2073, 0) == "2 073" and de(-1.234, 2) == "−1,23" and de(0.0, 1) == "0,0" and de(-0.004, 2) == "0,00"
    assert ganz(1999.6) == "2 000" and ganz(228826127) == "228 826 127"
    assert zehnerpotenz(1.14e10) == "1,1·10¹⁰" and zehnerpotenz(9.96e5, 1) == "1,0·10⁶" and zehnerpotenz(2.5e-18, 1) == "2,5·10⁻¹⁸" and zehnerpotenz(0) == "0"
    assert gross(584.4, 1) == "584,4" and gross(6408.6, 1) == "6 409" and gross(1.14e10, 1) == "1,1·10¹⁰" and gross(124355.4, 0) == "124 355"
    assert mb(38.97) == "39 MB" and mb(1830.6) == "1,8 GB" and mb(85999.7) == "86 GB" and mb(0.0177) == "0,018 MB" and mb(0.83) == "0,83 MB"


# --- Kernfrage 1: Weg -------------------------------------------------------------------------------------------------------------------------

TORUS8 = {1.0: {"lam": 1.0, "tau": 144.8, "tau_min": 127.3, "tau_max": 145.9, "ketten": 8, "schritte": 200000, "zufalls_seeds": [1, 2, 3, 4, 5]},
          5.0: {"lam": 5.0, "tau": 1069.0, "tau_min": 648.1, "tau_max": 1360.9, "ketten": 8, "schritte": 200000, "zufalls_seeds": [1, 2, 3, 4, 5]}}


def test_verdict_weg_konjugat_gauss_und_ring_sind_erfolge_ohne_diagnose():
    for ziel, snippet in (("konjugat", "Konjugater Prior"), ("gauss", "Zerlegung der Kovarianz"), ("ring", "2 Zeilenzustände")):
        a = Aufgabe(ziel=ziel)
        kind, text = ev.verdict_weg(a, waehle(a))
        assert kind == "success" and snippet in text and "irekt ziehen" in text


def test_verdict_weg_torus_8x8_nennt_zeilenzustaende_grenze_plaetze_und_kettenkosten():
    a = Aufgabe(ziel="torus", breite=8)
    kind, text = ev.verdict_weg(a, waehle(a), TORUS8)
    assert kind == "success"
    assert "Torus 8×8: 47 Zeilenzustände, unter der Grenze 2 207" in text and "64 Platzentscheidungen je Probe" in text and "kein Einbrennen, keine Diagnostik" in text
    assert "im Median über 5 Zufalls-Seeds 145 (λ = 1) bis 1 069 (λ = 5)" in text and "Spannen 127 bis 146 und 648 bis 1 361" in text and "8 Ketten × 200 000 Schritte" in text


def test_verdict_weg_torus_ohne_messung_der_kette_sagt_nicht_gemessen():
    a = Aufgabe(ziel="torus", breite=10)
    kind, text = ev.verdict_weg(a, waehle(a), TORUS8)
    assert kind == "success" and "Torus 10×10: 123 Zeilenzustände" in text and "nicht gemessen (gemessen: Breite 8)" in text
    assert "Spannen" not in text


def test_verdict_weg_torus_24_ueber_der_grenze_ist_eine_warnung_ohne_kostenmessung():
    a = Aufgabe(ziel="torus", breite=24, gipfel="mehrere")
    kind, text = ev.verdict_weg(a, waehle(a))
    assert kind == "warning" and "Torus 24×24: 103 682 Zeilenzustände, über der Grenze 2 207" in text and "abtasten" in text
    assert "Tempering" in text and "Starts in allen bekannten Gipfeln" in text and "die Kosten sind nicht gemessen" in text
    a = Aufgabe(ziel="torus", breite=24, gipfel="unbekannt")
    assert "R̂ sieht einen nicht gefundenen Gipfel nicht" in ev.verdict_weg(a, waehle(a))[1]


def test_verdict_weg_stetig_nennt_den_kern_und_dass_nicht_umgerechnet_wird():
    a = Aufgabe(ziel="stetig", dim=48, gipfel="ein")
    kind, text = ev.verdict_weg(a, waehle(a))
    assert kind == "info" and "Dimension 48" in text and "isotroper Vorschlag" in text and "nicht in Platzentscheidungen umrechnen" in text


def test_verdict_weg_grenze_breite_16_ist_noch_direkt_breite_17_nicht():
    assert ev.verdict_weg(Aufgabe(ziel="torus", breite=16), waehle(Aufgabe(ziel="torus", breite=16)))[0] == "success"
    assert ev.verdict_weg(Aufgabe(ziel="torus", breite=17), waehle(Aufgabe(ziel="torus", breite=17)))[0] == "warning"


# --- Kernfrage 2: Kosten ----------------------------------------------------------------------------------------------------------------------

RING_Z = {"zellen": 12, "kette_teurer_als_direkt_teilchen": 12, "kopplung_gerechnet": 10, "kopplung_teurer_als_gipfel": 10}
TORUS_Z = {"zellen": 7, "kette_teurer_als_direkt_teilchen": 7}


def test_verdict_kosten_ring_n16_lambda5_ist_der_anker():
    b = K.balken("Ring", 16, 5, "Teilchenzahl")
    kind, text = ev.verdict_kosten(b, RING_Z, TORUS_Z)
    assert kind == "info"
    for snippet in ("Hard-Core-Ring n = 16, λ = 5: direkt 16 Platzentscheidungen", "Heat-bath 60,6 je effektiv unabhängiger Teilchenzahl-Probe", "584,4 je Gipfel-Probe",
                    "plus einmalig 402 Schritte Einbrennen", "Kopplung aus der Vergangenheit 2 073 Schritte = 4 146 Platzentscheidungen", "(5,16 · t_mix(1/4))",
                    "In 12 von 12 Zellen kostet die Kette mehr als der Direktsampler", "in 10 von 10 gerechneten Zellen die Kopplung mehr als eine effektiv unabhängige Gipfel-Probe"):
        assert snippet in text, snippet


def test_verdict_kosten_ring_ohne_gerechnete_kopplung_und_nicht_gerechnete_zelle():
    kind, text = ev.verdict_kosten(K.balken("Ring", 16, 20, "Teilchenzahl"), RING_Z, TORUS_Z)
    assert kind == "info" and "Kopplung aus der Vergangenheit: für diese Zelle nicht gerechnet" in text and "Schritte =" not in text and "Heat-bath 872,8" in text.replace("872,9", "872,8")
    kind, text = ev.verdict_kosten(None, RING_Z, TORUS_Z)
    assert kind == "info" and "Für diese Zelle ist nichts gerechnet" in text


def test_verdict_kosten_torus_nennt_beide_funktionen_in_derselben_kette():
    kind, text = ev.verdict_kosten(K.balken("Torus 4×4", 16, 100, "Teilchenzahl"), RING_Z, TORUS_Z)
    assert kind == "info"
    assert "Torus 4×4, λ = 100: direkt 16 Platzentscheidungen" in text and "τ der Teilchenzahl 34,7" in text and "1,1·10¹⁰" in text
    assert "die effektive Stichprobengröße gilt je Funktion" in text and "in 7 von 7 Zellen" in text


def test_verdict_kosten_zahl_der_zellen_kommt_aus_den_daten():
    z = dict(RING_Z, kette_teurer_als_direkt_teilchen=11)
    assert "In 11 von 12 Zellen" in ev.verdict_kosten(K.balken("Ring", 8, 1, "Teilchenzahl"), z, TORUS_Z)[1]


# --- Kernfrage 3: Reichweite ------------------------------------------------------------------------------------------------------------------

def _probe(z):
    return {"lam": 1, "proben": 20000, "rho_exakt": 0.22701, "rho_probe": 0.22673, "sf": 0.00026, "z": z}


def test_verdict_reichweite_8x8_ist_der_anker():
    kind, text = ev.verdict_reichweite(8, D.reichweite(8), _probe(-1.07))
    assert kind == "success"
    for snippet in ("Torus 8×8: 213 256 442 503 Zustände, aber nur 47 Zeilenzustände", "20 000 Direktproben treffen die Belegungsdichte 0,22701 mit 0,22673 ± 0,00026",
                    "(λ = 1, Abweichung −1,07 Standardfehler)", "Unter der Grenze 2 207 (Breite 16): exakt ziehbar", "eine Breite, nicht die Zahl der Zustände"):
        assert snippet in text, snippet


def test_verdict_reichweite_schwelle_der_abweichung_4_standardfehler_inklusive():
    assert ev.verdict_reichweite(8, D.reichweite(8), _probe(4.0))[0] == "success"
    assert ev.verdict_reichweite(8, D.reichweite(8), _probe(-4.01))[0] == "warning"


def test_verdict_reichweite_breiten_ohne_ziehung():
    kind, text = ev.verdict_reichweite(20, D.reichweite(20), None)
    assert kind == "info" and "Zeilenzustände (dichte Matrix 1,8 GB)" in text and "15 127 Zeilenzustände" in text and "hier nicht gezogen und nicht gemessen" in text
    assert "nicht widerlegt" in text and "nur bis Breite 12" in text
    kind, text = ev.verdict_reichweite(24, D.reichweite(24), None)
    assert "103 682 Zeilenzustände" in text and "86 GB" in text and "Zahl der Zustände hier nicht berechnet" in text
    assert "Unter der Grenze" in ev.verdict_reichweite(16, D.reichweite(16), None)[1]       # Breite 16: 2 207, genau an der Grenze, inklusive


# --- Kernfrage 4: Diagnostik ------------------------------------------------------------------------------------------------------------------

def _diag(beides, rep, mittel, z, ueber3):
    return {"wiederholungen": rep, "n_beides": beides, "mittel_mittel": mittel, "exakt": 4.3395, "mittel_abs_z": z, "n_ueber3": ueber3}


VOR_FALSCH = {"max_abw": 0.0109174}
VOR_RICHTIG = {"max_abw": 1.7e-18}


def test_verdict_diagnose_falscher_kern_ist_eine_warnung_mit_der_vorprobe():
    kind, text = ev.verdict_diagnose("falsch", _diag(200, 200, 3.320, 40.1, 200), VOR_FALSCH, VOR_RICHTIG)
    assert kind == "warning"
    for snippet in ("Falscher Kern (Annahmeregel vergessen)", "200 von 200 Wiederholungen bestehen Split-R̂ < 1,01 und ESS ≥ 400", "3,320 (exakt 4,340)",
                    "im Mittel 40,1 Standardfehler daneben", "(200 von 200 Wiederholungen über 3 Standardfehler)", "max |π P − π| = 0,0109", "1,7·10⁻¹⁸",
                    "kein Gegenfall in 200 Wiederholungen, nicht Wahrscheinlichkeit 1"):
        assert snippet in text, snippet


def test_verdict_diagnose_ohne_alle_bestanden_steht_kein_gegenfallsatz():
    kind, text = ev.verdict_diagnose("falsch", _diag(48, 50, 3.3, 40.0, 50), VOR_FALSCH, VOR_RICHTIG)
    assert kind == "warning" and "48 von 50" in text and "kein Gegenfall" not in text


def test_verdict_diagnose_richtiger_kern_ist_ein_erfolg_ohne_gueteaussage():
    kind, text = ev.verdict_diagnose("richtig", _diag(129, 200, 4.339, 0.8, 2), {"max_abw": 1.7e-18}, VOR_RICHTIG)
    assert kind == "success"
    assert "Richtiger Kern (Heat-bath)" in text and "129 von 200" in text and "im Mittel 0,8 Standardfehler daneben" in text and "(2 von 200 Wiederholungen über 3 Standardfehler)" in text
    assert "kein Gütevergleich der Kerne" in text and "mischt schneller" in text


def test_verdict_tabelle_nennt_den_gezaehlten_satz():
    kind, text = ev.verdict_tabelle("18 Zeilen: 14 gemessen, 1 Quelle, 3 nicht gemessen")
    assert kind == "info" and text.startswith("18 Zeilen: 14 gemessen, 1 Quelle, 3 nicht gemessen;")


def test_lam_text():
    assert ev.lam_text(5.0) == "5" and ev.lam_text(100) == "100" and ev.lam_text(0.5) == "0,5" and np.isclose(float(ev.lam_text(1.0)), 1)
