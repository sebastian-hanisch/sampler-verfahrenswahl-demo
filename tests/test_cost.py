"""Eingefrorene Kosten (`svw_kosten.json`, `svw_iat8.json`): Vergleich mit einer früheren Messung derselben Größen (anderer Code), Neurechnung kleiner Zellen und Zählungen."""

import json

import pytest

import svw_cost as C
import svw_make_kosten as K

# frühere Messung (anderer Code, andere Zufallszahlen für die Simulation): n, λ -> t_mix(1/4), τ Teilchenzahl, τ Gipfelindikator, Kopplung in Schritten (None = nicht gerechnet)
FRUEHER = {
    (8, 1): (31, 10.18, 46.5, 142.6), (8, 5): (153, 33.99, 318.2, 756.1), (8, 20): (1403, 192.1, 3554, 6409), (8, 100): (29147, 1140, 81490, 124355),
    (12, 1): (54, 15.46, 66.8, 258.3), (12, 5): (267, 50.73, 450.8, 1371), (12, 20): (2276, 546.9, 4971, 11127), (12, 100): (44647, 4724, 118883, 196787),
    (16, 1): (79, 20.92, 87.0, 386.8), (16, 5): (402, 60.56, 584.4, 2073), (16, 20): (3304, 872.9, 6305, None), (16, 100): (61220, 10855, 153504, None),
}
FRUEHER_TORUS_TAU = {1: (39.5, 277), 2: (67.7, 1236), 5: (82.2, 2.05e4), 10: (66.1, 2.94e5), 20: (49.8, 5.78e6), 50: (38.5, 4.03e8), 100: (34.7, 1.14e10)}


def test_zellen_gleich_fruehere_messung():
    assert len(C.ring_zellen()) == 12
    for (n, lam), (tm, t1, t2, kop) in FRUEHER.items():
        z = C.ring_zelle(n, lam)
        assert z["t_mix"] == tm and z["t_mix_start_gerade"] == tm
        assert z["tau_teilchen"] == pytest.approx(t1, rel=3e-3) and z["tau_gipfel"] == pytest.approx(t2, rel=3e-3)
        assert z["direkt"] == n
        if kop is None:
            assert z["kopplung_schritte"] is None and z["kopplung_platz"] is None
        else:
            assert z["kopplung_schritte"] == pytest.approx(kop, rel=2e-3) and z["kopplung_platz"] == pytest.approx(2 * z["kopplung_schritte"], rel=1e-12)


def test_torus_zellen_gleich_fruehere_messung():
    for lam, (t1, t2) in FRUEHER_TORUS_TAU.items():
        z = C.torus_zelle(lam)
        assert z["tau_teilchen"] == pytest.approx(t1, rel=3e-3) and z["tau_gipfel"] == pytest.approx(t2, rel=3e-3) and z["direkt"] == 16
    assert C.torus_zelle(100)["tau_teilchen"] == pytest.approx(34.68, abs=0.005) and C.torus_zelle(100)["tau_gipfel"] == pytest.approx(1.14e10, rel=5e-3)


def test_zaehlungen_ring_und_torus():
    z = C.ring_zaehlungen()
    assert z["zellen"] == 12 and z["kette_teurer_als_direkt_teilchen"] == 12 and z["kette_teurer_als_direkt_gipfel"] == 12
    assert z["kopplung_gerechnet"] == 10 and z["kopplung_teurer_als_gipfel"] == 10
    lo, hi = z["tau_teilchen_je_n"]
    assert (lo, hi) == (pytest.approx(1.27, abs=0.005), pytest.approx(678.5, abs=0.5))
    lo, hi = z["kopplung_je_tmix"]
    assert (lo, hi) == (pytest.approx(4.27, abs=0.005), pytest.approx(5.16, abs=0.005))
    lo, hi = z["kopplung_je_gipfel"]
    assert (lo, hi) == (pytest.approx(3.05, abs=0.01), pytest.approx(8.897, abs=0.005))          # Kehrwerte von 0,328 und 0,112
    lo, hi = z["kopplung_je_teilchen"]
    assert (lo, hi) == (pytest.approx(28.0, abs=0.5), pytest.approx(218.1, abs=0.05))
    lo, hi = z["gipfel_je_teilchen"]
    assert (lo, hi) == (pytest.approx(4.2, abs=0.05), pytest.approx(71.5, abs=0.05))
    t = C.torus_zaehlungen()
    assert t["zellen"] == 7 and t["kette_teurer_als_direkt_teilchen"] == 7 and t["lam_max_tau_teilchen"] == 5.0
    assert t["tau_teilchen_je_direkt"][0] == pytest.approx(2.17, abs=0.005) and t["tau_teilchen_je_direkt"][1] == pytest.approx(5.14, abs=0.005)


def test_der_beste_start_fuer_t_mix_ist_alle_geraden_plaetze_in_den_zellen_bis_n12():
    zellen = [z for z in C.ring_zellen() if z["n"] <= 12]
    assert len(zellen) == 8
    for z in zellen:                                    # für n <= 12 ist t_mix das Maximum über alle Starts, t_mix_start_gerade nur der Start „alle geraden Plätze“
        assert z["t_mix"] == z["t_mix_start_gerade"]


def test_neurechnung_ring_n8_und_torus_stichprobe_gleich_datei():
    neu = K.ring_zellen(8)
    assert K.vergleiche({"ring": neu}, {"ring": [z for z in C.ring_zellen() if z["n"] == 8]}) == []
    torus = K.torus_zellen((1.0, 100.0))
    alt = [z for z in C.laden()["torus44"] if z["lam"] in (1.0, 100.0)]
    assert K.vergleiche({"t": torus}, {"t": alt}) == []
    takt = K.takt_zellen(12, (1.0, 100.0))
    assert K.vergleiche({"t": takt}, {"t": [z for z in C.takt_zellen() if z["lam"] in (1.0, 100.0)]}) == []


def test_neurechnung_ring_n12_gleich_datei():
    assert K.vergleiche({"ring": K.ring_zellen(12)}, {"ring": [z for z in C.ring_zellen() if z["n"] == 12]}) == []


def test_vergleiche_findet_abweichungen():
    assert K.vergleiche({"a": [1.0, {"b": 2}]}, {"a": [1.0, {"b": 2}]}) == []
    assert K.vergleiche({"a": 1.0}, {"a": 1.01}) != []
    assert K.vergleiche({"a": [1, 2]}, {"a": [1]}) != []
    assert K.vergleiche({"a": 1}, {}) != []


def test_takt_verhaeltnis_nahe_dem_takt_alle_8_zellen():
    zellen = C.takt_zellen()
    assert [z["lam"] for z in zellen] == [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    assert [round(z["verhaeltnis"], 2) for z in zellen] == [1.52, 2.00, 1.50, 1.20, 1.10, 1.05, 1.02, 1.01]
    assert max(abs(z["verhaeltnis"] - z["takt"]) for z in zellen) < 0.021      # größte Abweichung 0,02 bei λ = 0,5


def test_iat8_aufbau_und_eichung_am_4x4_torus():
    d = C.iat8()
    assert [z["lam"] for z in d["torus8x8"]] == [1.0, 5.0] and all(len(z["taus"]) == len(z["zufalls_seeds"]) == 5 for z in d["torus8x8"])
    for z in d["torus8x8"]:
        assert z["tau_min"] == min(z["taus"]) and z["tau_max"] == max(z["taus"]) and z["tau_min"] <= z["tau"] <= z["tau_max"]
        assert z["direkt"] == 64 and abs(z["dichte_kette_mittel"] - z["dichte_exakt"]) < 0.01
    for e in d["eichung_4x4"]:
        assert 0.85 < e["verhaeltnis"] < 1.15 and e["tau_exakt"] == pytest.approx(C.torus_zelle(e["lam"])["tau_teilchen"], rel=1e-9)


def test_felder_nennen_die_einheit_und_jede_zelle_hat_alle_felder():
    d = json.loads((C.ROOT / "svw_kosten.json").read_text(encoding="utf-8"))
    assert "Platzentscheidungen" in d["felder"]["einheit"]
    for z in d["ring"]:
        assert {"n", "lam", "t_mix", "tau_teilchen", "tau_gipfel", "direkt", "kopplung_schritte", "kopplung_platz"} <= set(z)


def test_balken_und_randfaelle():
    b = C.balken("Ring", 16, 5, "Teilchenzahl")
    assert b["direkt"] == 16 and b["kette"] == pytest.approx(60.56, abs=0.005) and b["kopplung"] == pytest.approx(4146.0, abs=0.5) and b["einbrennen"] == 402
    assert C.balken("Ring", 16, 5, "Gipfelindikator")["kette"] == pytest.approx(584.4, abs=0.05)
    assert C.balken("Ring", 16, 2, "Teilchenzahl") is None and C.balken("Ring", 12, 10, "Teilchenzahl") is None       # λ nicht gerechnet
    assert C.balken("Ring", 16, 20, "Teilchenzahl")["kopplung"] is None                                                # Kopplung nicht gerechnet
    t = C.balken("Torus 4×4", 16, 100, "Gipfelindikator")
    assert t["kopplung"] is None and t["einbrennen"] is None and t["kette"] == pytest.approx(1.14e10, rel=5e-3)
    with pytest.raises(ValueError):
        C.balken("Kegel", 16, 5, "Teilchenzahl")
    with pytest.raises(ValueError):
        C.balken("Ring", 16, 5, "Länge")
