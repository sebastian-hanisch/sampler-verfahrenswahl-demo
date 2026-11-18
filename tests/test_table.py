"""Entscheidungstabelle: Aufbau, gezählter Belegungsgrad, Abgleichwerkzeug und die Zeilen, die hier nachgerechnet sind."""

import json

import pytest

import svw_cost as C
import svw_reference as R
import svw_table as TB
import svw_texts as T
import svw_check_table as CT


def test_aufbau_achtzehn_zeilen_mit_allen_feldern():
    zeilen = TB.zeilen()
    assert [z["nr"] for z in zeilen] == list(range(1, 19))
    for z in zeilen:
        assert {"frage", "antwort", "zahl", "nachbar", "herkunft", "status", "stand", "pruefzahlen"} <= set(z)
        assert z["status"] in TB.STATUS and z["stand"] in TB.STAND
        assert z["zahl"] and z["frage"] and z["antwort"]


def test_belegungsgrad_aus_der_tabelle_gezaehlt():
    c = TB.zaehle()
    assert c == {"gemessen": 14, "Quelle": 1, "nicht gemessen": 3, "zeilen": 18}
    assert TB.belegungssatz() == "18 Zeilen: 14 gemessen, 1 Quelle, 3 nicht gemessen"
    assert [z["nr"] for z in TB.zeilen() if z["status"] == "Quelle"] == [12]
    assert [z["nr"] for z in TB.zeilen() if z["status"] == "nicht gemessen"] == [4, 17, 18]


def test_status_und_stand_passen_zusammen():
    for z in TB.zeilen():
        if z["status"] == "Quelle":
            assert z["stand"] == "Quelle"
        if z["status"] == "nicht gemessen":
            assert z["stand"] == "–" and z["pruefzahlen"] == []
        if z["nachbar"] == "":
            assert z["pruefzahlen"] == []                      # ohne Nachbar-Demo gibt es nichts abzugleichen
        if z["stand"] == "aus Nachbar-Demo übernommen":
            assert z["nachbar"] and z["pruefzahlen"]           # übernommene Zahlen brauchen die Demo und eine Prüfzahl


def test_nachbarn_der_tabelle_stehen_in_der_nachbarliste_der_app():
    namen = {n for n, _ in T.NACHBARN}
    for z in TB.zeilen():
        if z["nachbar"]:
            assert z["nachbar"] in namen


def test_keine_planungswoerter_in_den_tabellentexten():
    text = json.dumps(TB.zeilen(), ensure_ascii=False)
    for wort in ("Stück", "Linie", "Vormessung", "Phase 0", "Kantenabgleich", "mc_v"):
        assert wort not in text, wort


# --- Zeilen, die hier nachgerechnet sind ------------------------------------------------------------------------------------------------------

def test_zeile_1_gamma_poisson_posterior():
    a, b, mittel, sd = R.gamma_poisson(2, 1, 32, 20)
    z = TB.zeile(1)["zahl"]
    assert (a, b) == (34, 21) and "Gamma(34, 21)" in z
    assert f"{float(mittel):.5f}".replace(".", ",") in z and f"{sd:.5f}".replace(".", ",") in z


def test_zeile_3_zeilenzustaende_und_grenze():
    import svw_direct as D
    z = TB.zeile(3)["zahl"]
    assert f"{D.zeilen_zustaende(8)} Zeilenzustände" in z and f"12×12: {D.zeilen_zustaende(12)}" in z and "2 207" in z
    assert D.zeilen_zustaende(16) == 2207 and D.dichte_matrix_mb(16) == pytest.approx(39, abs=0.1)


def test_zeile_5_takt_verhaeltnisse_stehen_in_der_tabelle():
    z = TB.zeile(5)["zahl"]
    for zelle in C.takt_zellen():
        assert f"{zelle['verhaeltnis']:.2f}".replace(".", ",") in z


def test_zeile_7_und_8_zahlen_aus_den_kosten():
    assert f"{C.ring_zelle(16, 100)['t_mix']:,}".replace(",", " ") in TB.zeile(7)["zahl"]
    lo, hi = C.ring_zaehlungen()["kopplung_je_tmix"]
    z = TB.zeile(8)["zahl"]
    assert f"{lo:.2f}".replace(".", ",") in z and f"{hi:.2f}".replace(".", ",") in z and "10 Zellen" in z


def test_zeile_15_torus_tau_je_funktion():
    z = TB.zeile(15)["zahl"]
    t = C.torus_zelle(100)
    assert f"{t['tau_teilchen']:.1f}".replace(".", ",") in z and "1,1·10¹⁰" in z and t["tau_gipfel"] == pytest.approx(1.1e10, rel=0.05)


# --- Abgleichwerkzeug ------------------------------------------------------------------------------------------------------------------------

def test_abgleich_mit_gelesenen_readmes_abgeglichen_fehlt_und_nicht_abgeglichen():
    zeilen = [{"nr": 1, "nachbar": "a-demo", "pruefzahlen": ["1,61905", "Gamma(34, 21)"]}, {"nr": 2, "nachbar": "b-demo", "pruefzahlen": ["71 585"]},
              {"nr": 3, "nachbar": "c-demo", "pruefzahlen": ["0,222"]}, {"nr": 4, "nachbar": "", "pruefzahlen": []}, {"nr": 5, "nachbar": "d-demo", "pruefzahlen": []}]
    readmes = {"a-demo": "Mittel 1,61905 bei Gamma(34,  21).", "b-demo": "nur 71 584 Schritte", "c-demo": None}
    erg = CT.pruefe(zeilen, None, leser=readmes.get)
    assert [(e["nr"], e["ergebnis"]) for e in erg] == [(1, "abgeglichen"), (2, "FEHLT"), (3, "nicht abgeglichen")]       # Zeilen ohne Nachbar oder Prüfzahl fehlen
    assert erg[1]["fehlt"] == ["71 585"]


def test_abgleich_liest_ordner(tmp_path):
    (tmp_path / "x-demo").mkdir()
    (tmp_path / "x-demo" / "README.md").write_text("Wert 0,325 und 29,1", encoding="utf-8")
    zeilen = [{"nr": 1, "nachbar": "x-demo", "pruefzahlen": ["0,325", "29,1"]}, {"nr": 2, "nachbar": "y-demo", "pruefzahlen": ["1"]}]
    erg = CT.pruefe(zeilen, tmp_path)
    assert [e["ergebnis"] for e in erg] == ["abgeglichen", "nicht abgeglichen"]


def test_abgleich_zaehlt_nur_ganze_zahlen():
    """Eine Prüfzahl als Teilzeichenkette einer anderen Zahl gilt nicht als abgeglichen („53“ in „1 053“, „0,222“ in „0,2225“, „16“ in „0,16“)."""
    assert not CT.kommt_vor("53", "Wert 1 053 und 153 und 0,53")
    assert CT.kommt_vor("53", "Wert 53 und mehr") and CT.kommt_vor("53", "von 12 bis 53.") and CT.kommt_vor("53", "(53)")
    assert not CT.kommt_vor("0,222", "ESS 0,2225 je Schritt") and CT.kommt_vor("0,222", "ESS 0,222 je Schritt")
    assert not CT.kommt_vor("16", "Verhältnis 0,16 und 0,83") and CT.kommt_vor("16", "Verhältnis 1,6 und 16-fach")
    assert CT.kommt_vor("71 585", "nur 71 585 Schritte") and not CT.kommt_vor("71 585", "nur 171 585 Schritte")
    assert CT.kommt_vor("99 / 49 / 9 / 0", "Kette 99 / 49 / 9 / 0 (ohne Wechsel") and not CT.kommt_vor("99 / 49 / 9 / 0", "Kette 99 / 49 / 9 / 05")
    assert CT.kommt_vor("Gamma(34, 21)", "bei Gamma(34,  21).")


def test_abgleich_ignoriert_angaben_zur_ersten_rechnung():
    """Eine überholte Zahl, die nur als „erste Rechnung“ im Text steht, ist nicht abgeglichen (Fehltreffer der Tabellenzeilen 9, 10, 11 und 14)."""
    text = "ESS 0,205 bei d = 1 (die erste Rechnung nannte 0,222); der Wert 19,8 (erste Rechnung 4,2 und 19,3). Täuschend 191 / 186 / 194 (erste Rechnung: 192 / 186 / 189); Ring 51 bis 57."
    assert CT.kommt_vor("0,205", text) and CT.kommt_vor("19,8", text) and CT.kommt_vor("191 / 186 / 194", text) and CT.kommt_vor("51 bis 57", text)
    assert not CT.kommt_vor("0,222", text) and not CT.kommt_vor("19,3", text) and not CT.kommt_vor("192", text) and not CT.kommt_vor("189", text)
    assert CT.kommt_vor("186", text)                                       # steht auch in der aktuellen Angabe 191 / 186 / 194
    assert CT.kommt_vor("4,43", "Die erste Rechnung nannte 4,02. Mit dem Raster sind es 4,43 bis 5,16.")      # Satzende beendet die Angabe zur ersten Rechnung
    zeilen = [{"nr": 11, "nachbar": "a-demo", "pruefzahlen": ["19,8", "19,3"]}]
    erg = CT.pruefe(zeilen, None, leser={"a-demo": text}.get)
    assert erg[0]["ergebnis"] == "FEHLT" and erg[0]["fehlt"] == ["19,3"]


def test_abgleich_der_echten_tabelle_meldet_keinen_fehler_ohne_nachbar_readmes(tmp_path):
    zeilen = json.loads((TB.ROOT / "svw_table.json").read_text(encoding="utf-8"))["zeilen"]
    erg = CT.pruefe(zeilen, tmp_path)                                      # leerer Ordner: keine README liegt vor
    assert erg and all(e["ergebnis"] == "nicht abgeglichen" for e in erg)
    assert {e["nr"] for e in erg} == {z["nr"] for z in zeilen if z["nachbar"] and z["pruefzahlen"]}
