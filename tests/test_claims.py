"""Claims-Tests: jede Zahl der README wird hier nachgerechnet (Ganzzahlen und Brüche exakt, Zufallsproben mit dem Zufalls-Seed der README als Band).

Die Tests bilden die Zahlen aus den Messdaten mit demselben Zahlenformat wie die App und prüfen, dass der Text in der README steht; so kann weder die README noch die Messung
unbemerkt abweichen. Messungen, die nicht im Test stehen (Heat-bath auf dem 8×8-Torus), nennt die README ausdrücklich als „Messung dieser README“."""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import svw_cost as K
import svw_diag as G
import svw_direct as D
import svw_table as TB
from svw_format import de, ganz, gross, mb, zehnerpotenz
from svw_zufall import Strom

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def zeile(*zellen) -> str:
    return "| " + " | ".join(zellen) + " |"


def test_kosten_tabelle_und_lesarten():
    for z in K.ring_zellen():
        kp = ("nicht gerechnet", "–") if z["kopplung_schritte"] is None else (gross(z["kopplung_schritte"], 1), gross(z["kopplung_platz"], 1))
        erwartet = zeile(str(z["n"]), ganz(z["lam"]), ganz(z["t_mix"]), gross(z["tau_teilchen"], 1), gross(z["tau_gipfel"], 1), str(z["direkt"]), *kp)
        assert erwartet in README, erwartet
    c = K.ring_zaehlungen()
    assert c["zellen"] == 12 and c["kette_teurer_als_direkt_teilchen"] == 12 and c["kopplung_gerechnet"] == 10 and c["kopplung_teurer_als_gipfel"] == 10
    for satz in ("in 12 von 12 Zellen weniger als die Kette", "in 10 von 10 gerechneten Zellen mehr Platzentscheidungen als",
                 f"dem {de(c['tau_teilchen_je_n'][0], 2)}-Fachen (n = 8, λ = 1) und dem {ganz(c['tau_teilchen_je_n'][1])}-Fachen (n = 16, λ = 100)",
                 f"dem {de(c['tau_gipfel_je_n'][0], 1)}- und dem {ganz(c['tau_gipfel_je_n'][1])}-Fachen",
                 f"das {de(c['kopplung_je_tmix'][0], 2)}- bis {de(c['kopplung_je_tmix'][1], 2)}-Fache von t_mix(1/4)",
                 f"(das {de(c['kopplung_je_gipfel'][0], 2)}- bis {de(c['kopplung_je_gipfel'][1], 2)}-Fache)",
                 f"das {de(c['kopplung_je_teilchen'][0], 1)}- bis {de(c['kopplung_je_teilchen'][1], 1)}-Fache",
                 f"um das {de(c['gipfel_je_teilchen'][0], 1)}- bis {de(c['gipfel_je_teilchen'][1], 1)}-Fache"):
        assert satz in README, satz
    groesste = max(K.ring_zellen(), key=lambda z: z["tau_gipfel"] / z["tau_teilchen"])
    assert (groesste["n"], groesste["lam"]) == (8, 100.0)                    # „am größten bei n = 8, λ = 100“


def test_torus_tabelle():
    for z in K.laden()["torus44"]:
        erwartet = zeile(ganz(z["lam"]), gross(z["tau_teilchen"], 1), gross(z["tau_gipfel"], 1), gross(z["t_rel"], 1))
        assert erwartet in README, erwartet
    t = K.torus_zaehlungen()
    assert t["kette_teurer_als_direkt_teilchen"] == 7 and t["lam_max_tau_teilchen"] == 5.0
    assert f"das {de(t['tau_teilchen_je_direkt'][0], 1)}- bis {de(t['tau_teilchen_je_direkt'][1], 1)}-Fache" in README
    z = K.torus_zelle(100)
    assert f"τ = {de(z['tau_teilchen'], 1)} für die Teilchenzahl, τ = {zehnerpotenz(z['tau_gipfel'], 1)} für den Gipfelindikator" in README
    assert "4 Zeilenentscheidungen = 16 Platzentscheidungen" in README and D.reichweite(4)["paare_je_zeile"] == 5.0
    assert "rund 20 gelesene" in README and 4 * D.reichweite(4)["paare_je_zeile"] == 20.0


def test_takt_tabelle():
    z = K.takt_zellen()
    assert " / ".join(de(x["verhaeltnis"], 2) for x in z) + " für λ = 0,5 / 1 / 2 / 5 / 10 / 20 / 50 / 100" in README
    assert "t_mix Heat-bath " + " / ".join(ganz(x["t_mix_heatbath"]) for x in z) in README
    assert "Metropolis " + " / ".join(ganz(x["t_mix_metropolis"]) for x in z) in README
    assert max(abs(x["verhaeltnis"] - x["takt"]) for x in z) == pytest.approx(0.02, abs=1e-9) and "größte Abweichung ist 0,02 (bei λ = 0,5)" in README


def test_reichweite_tabelle():
    for w in (4, 8, 12, 16, 20, 24):
        r = D.reichweite(w)
        zu = r["zustaende"]
        zus = "nicht berechnet" if zu is None else (ganz(zu) if isinstance(zu, int) else zehnerpotenz(zu, 1))
        assert zeile(str(w), ganz(r["zeilen"]), gross(r["paare"], 1, ab=1e6) if r["paare"] >= 1e7 else ganz(r["paare"]), de(r["paare_je_zeile"], 1), mb(r["mb"]), zus) in README, w
    assert "2,1·10¹¹ Zustände, aber nur 47 Zeilenzustände" in README and D.torus_zustaende(8) == 213256442503 and zehnerpotenz(D.torus_zustaende(8), 1) == "2,1·10¹¹"
    assert "(322) ist ebenfalls exakt ziehbar" in README and D.zeilen_zustaende(12) == 322
    assert D.reichweite(8)["paare_je_zeile"] == pytest.approx(24.57, abs=0.005) and "24,6" in README and 8 * D.reichweite(8)["paare_je_zeile"] == pytest.approx(197, abs=0.5)


def test_direktproben_sieben_zellen():
    zellen = [(8, 1, 20000), (8, 5, 20000), (8, 20, 20000), (8, 100, 20000), (12, 1, 8000), (12, 5, 8000), (12, 20, 8000)]
    zs, erste = [], None
    for i, (w, lam, b) in enumerate(zellen):
        r = D.probe_gegen_exakt(w, lam, b, Strom(20003 + i))
        zs.append(r["z"])
        erste = erste or r
    assert max(abs(z) for z in zs) == pytest.approx(1.16, abs=0.02) and int(np.argmax(np.abs(zs))) == 1        # 8×8, λ = 5
    assert all(abs(z) < 4 for z in zs)
    assert f"{de(erste['rho_probe'], 5)} ± {de(erste['sf'], 5)} gegen exakt {de(erste['rho_exakt'], 5)}" in README
    assert "größte Abweichung ist 1,16 Standardfehler" in README and "20003 bis 20009" in README


def test_diagnose_200_wiederholungen():
    def lauf(kern, schritte=4000, verworfen=1000, seed=20004, start="gestreut"):
        return G.pruefe_wiederholungen(12, 3.0, kern, 200, schritte, verworfen, seed, start)

    f, r = lauf("falsch"), lauf("richtig")
    assert (f["n_rhat"], f["n_ess"], f["n_beides"], f["n_ueber3"]) == pytest.approx((200, 200, 200, 200), abs=3)
    assert f["mittel_mittel"] == pytest.approx(3.318, abs=0.005) and f["mittel_abs_z"] == pytest.approx(40.3, abs=1.0)
    assert (r["n_rhat"], r["n_ess"], r["n_beides"], r["n_ueber3"]) == pytest.approx((136, 176, 129, 2), abs=3)
    assert r["mittel_mittel"] == pytest.approx(4.339, abs=0.005) and r["mittel_abs_z"] == pytest.approx(0.75, abs=0.3)
    assert f["exakt"] == pytest.approx(4.3395, abs=5e-5)
    # Variation: anderer Zufalls-Seed, kürzere und längere Ketten, Start alle leer (Playbook 11a Nr. 3: der Satz hängt nicht an einem Seed oder einer Länge)
    f2, r2 = lauf("falsch", seed=20005), lauf("richtig", seed=20005)
    assert f2["n_beides"] >= 197 and f2["mittel_abs_z"] == pytest.approx(40.3, abs=1.0)
    assert (r2["n_rhat"], r2["n_ess"], r2["n_beides"]) == pytest.approx((142, 182, 136), abs=3)
    f3, r3 = lauf("falsch", 2000, 500), lauf("richtig", 2000, 500)
    assert f3["n_beides"] == pytest.approx(199, abs=2) and f3["mittel_abs_z"] == pytest.approx(28.3, abs=1.0)
    assert r3["n_rhat"] == pytest.approx(50, abs=5) and r3["n_ess"] <= 6 and r3["n_beides"] == 0
    f4, r4 = lauf("falsch", 8000, 2000), lauf("richtig", 8000, 2000)
    assert f4["n_beides"] >= 197 and f4["mittel_abs_z"] == pytest.approx(56.9, abs=1.5) and r4["n_beides"] == pytest.approx(197, abs=3)
    f5, r5 = lauf("falsch", start="leer"), lauf("richtig", start="leer")
    assert f5["n_beides"] >= 197 and (r5["n_rhat"], r5["n_ess"], r5["n_beides"]) == pytest.approx((136, 176, 129), abs=3)
    for text in ("| falsch (Annahmeregel vergessen) | 200 von 200 | 200 von 200 | 200 von 200 | 3,318 | 40,3 | 200 von 200 |",
                 "| richtig (Heat-bath) | 136 von 200 | 176 von 200 | 129 von 200 | 4,339 | 0,75 | 2 von 200 |",
                 "falsch 200 / 200 / 200, 40,3 Standardfehler, richtig 142 / 182 / 136", "falsch beides in 199 von 200 (28,3 Standardfehler), richtig Split-R̂ in 50 von 200, ESS ≥ 400 nur in 3 von 200",
                 "falsch 200 von 200\n(56,9 Standardfehler), richtig 197 von 200", "dieselben Zahlen wie bei überstreutem Start"):
        assert " ".join(text.split()) in " ".join(README.split()), text


def test_vorprobe_und_mittelwerte():
    ring = G.Ring(12)
    f, r = G.vorprobe(ring, 3.0, "falsch"), G.vorprobe(ring, 3.0, "richtig")
    pi = ring.pi(3.0)
    assert f["max_abw"] == pytest.approx(0.0109, abs=5e-5) and f["tv"] == pytest.approx(0.1189, abs=5e-5) and f["db"] == pytest.approx(0.0018, abs=5e-5)
    assert f["max_abw"] / f["pi_max"] == pytest.approx(0.33, abs=0.005) and f["pi_max"] == pytest.approx(0.0328, abs=5e-5) and r["max_abw"] < 1e-12
    assert float(pi @ ring.k) == pytest.approx(4.3395, abs=5e-5) and float(np.sqrt(pi @ (ring.k - pi @ ring.k) ** 2)) == pytest.approx(0.8822, abs=5e-5)
    assert float(ring.k.mean()) == pytest.approx(3.3168, abs=5e-5) and ring.zahl == 322
    assert 0.5 * np.abs(np.ones(ring.zahl) / ring.zahl - pi).sum() == pytest.approx(0.3965, abs=5e-5)
    for text in ("max |π P − π| = 0,0109 (33 % des größten Gewichts 0,0328), TV(π P, π) = 0,1189, Abweichung", "der Detailed Balance 0,0018",
                 "(exakt 4,3395, Standardabweichung 0,8822; Gleichverteilung 3,3168)"):
        assert " ".join(text.split()) in " ".join(README.split()), text


def test_beispiel_von_hand_und_gamma_poisson_stehen_in_der_readme():
    import svw_reference as R
    assert "Z = 17, π(leer) = 1/17, π(Einzelbelegung) = 2/17, π({0, 2}) = 4/17" in README and "P(Platz 0 belegt) = 6/17" in README
    assert "t_mix(1/4) = 22; τ der Teilchenzahl 1195/221 = 5,41, τ des Gipfelindikators 53" in README and float(Fraction(1195, 221)) == pytest.approx(5.407, abs=0.001)
    assert "88,04 Schritte = 176,1 Platzentscheidungen" in README and "das 4,00-Fache von t_mix" in README
    a, b, mittel, sd = R.gamma_poisson(2, 1, 32, 20)
    assert "Gamma(34, 21), Mittel 1,61905, Standardabweichung 0,27766" in README and (a, b) == (34, 21)


def test_tabelle_im_readme_stimmt_mit_der_tabelle_der_app():
    assert TB.belegungssatz() in README
    assert "Zeile 1, 3, 5, 15 und 16 hier nachgerechnet" in README and [z["nr"] for z in TB.zeilen() if z["stand"] == "hier gerechnet"] == [1, 3, 5, 15, 16]
    assert [z["nr"] for z in TB.zeilen() if z["stand"] == "teilweise hier gerechnet"] == [7, 8]
    assert [z["nr"] for z in TB.zeilen() if z["stand"] == "aus Nachbar-Demo übernommen"] == [2, 6, 9, 10, 11, 14]
    assert [z["nr"] for z in TB.zeilen() if z["stand"] == "Messreihe, hier nicht neu gerechnet"] == [13]
    assert "die Zeilen 2, 6, 9, 10, 11 und 14 sind aus den READMEs der Nachbar-Demos übernommen" in README and "Zeile 13 stammt aus einer eigenen Messreihe" in README
    assert "die Zeilen 8, 9 und 14 gehören zu Nachbar-Demos, die danach noch korrigiert werden" in README


def test_iat8_zahlen_stehen_in_der_readme_als_messung_nicht_im_test():
    z1, z5 = K.tau_torus8()[1.0], K.tau_torus8()[5.0]
    assert f"λ = 1: Median {ganz(z1['tau'])}, Spanne {ganz(z1['tau_min'])} bis {ganz(z1['tau_max'])}" in README
    assert f"λ = 5: Median {ganz(z5['tau'])}, Spanne {ganz(z5['tau_min'])} bis {ganz(z5['tau_max'])}" in README
    e = K.iat8()["eichung_4x4"]
    assert f"Verhältnis {de(e[0]['verhaeltnis'], 3)} (λ = 1) und {de(e[1]['verhaeltnis'], 3)} (λ = 2)" in README


def test_die_readme_nennt_keine_planungswoerter_ausser_der_ueberschrift():
    text = README.replace("## Befunde und Korrekturen gegenüber dem Plan", "").split("Diese Demo ist Teil des Portfolios")[0]      # die Fußzeile (Reihe) kommt vom Werkzeug
    for wort in ("Stück", "Linie", "Vormessung", "Phase 0", "Kantenabgleich", "mc_v", "geplant", "Wärmebad", "Saatwert", "läuft lokal"):
        assert wort not in text, wort
    assert not re.search(r"\btr(ä|ae)g(t|en)\b", text) and not re.search(r"\bPlan\b", text) and not re.search(r"\bV[123]\b", text)
    assert "Zufalls-Seed" in README and "Zufallsseed" not in README and "Offen" not in README.split("## Ehrliche Grenzen")[0]
