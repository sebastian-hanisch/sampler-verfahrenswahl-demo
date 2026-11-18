"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Die Tests unten stellen sicher, dass wirklich JEDER Regler
(Seitenleiste und Hauptbereich) dort steht: sonst bliebe das Preset trotz Änderung markiert."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import svw_constants as C
import svw_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "select_slider", "number_input", "selectbox", "radio", "toggle", "checkbox", "text_input")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for bereich in (at.sidebar, at.main):
        for art in REGLER_ARTEN:
            try:
                return getattr(bereich, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} steht weder in der Seitenleiste noch im Hauptbereich")


def _schluessel(bereich) -> set:
    return {w.key for art in REGLER_ARTEN for w in getattr(bereich, art) if w.key}


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_steht_in_setting_specs_und_jeder_eintrag_hat_einen_regler():
    at = _app()
    schluessel = _schluessel(at.sidebar) | _schluessel(at.main)
    assert not schluessel - set(P.SETTING_SPECS), f"Regler {sorted(schluessel - set(P.SETTING_SPECS))} fehlen in SETTING_SPECS: ihre Änderung hebt das Preset nicht auf"
    assert not set(P.SETTING_SPECS) - schluessel, f"Einträge ohne Regler: {sorted(set(P.SETTING_SPECS) - schluessel)}"
    assert _schluessel(at.sidebar) == {"ziel", "breite", "dim", "gradient", "kovarianz", "gipfel", "kopplung_ok", "seed"}       # Aufgabe und Zufalls-Seed in der Seitenleiste
    assert _schluessel(at.main) == {"kost_modell", "kost_n", "kost_lam", "kost_f", "proben", "lam_dichte", "diag_rep", "diag_kern"}      # Experimentregler beim Experiment


def _andere_werte(spec, aktuell):
    """Werte, die vom aktuellen abweichen (billigster zuerst: untere Grenze bzw. die anderen Auswahlwerte)."""
    if spec.choices is not None:
        return [c for c in spec.choices if c != aktuell]
    return [v for v in (spec.lo, spec.hi) if v != aktuell]


BASIS = {"kopplung_ok": "Gipfel unbekannt"}                         # Ausgangspreset, in dem der Regler wirksam (nicht deaktiviert) ist; sonst „Posterior in 48 Dimensionen“


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    """Ausgangspunkt ist ein angeklicktes Preset (hervorgehoben, Regler wirksam); der Regler bekommt einen anderen Wert: keine Hervorhebung mehr, erneutes Anklicken markiert wieder."""
    basis = BASIS.get(state_key, "Posterior in 48 Dimensionen")
    at = _app()
    at.sidebar.button(key=f"preset_{basis}").click().run()
    assert _typen(at)[basis] == "primary"
    regler = _regler(at, state_key)
    assert not regler.disabled
    regler.set_value(_andere_werte(P.SETTING_SPECS[state_key], regler.value)[0]).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values(), state_key
    at.sidebar.button(key=f"preset_{basis}").click().run()
    assert _typen(at)[basis] == "primary"


def test_nicht_wirksame_regler_sind_deaktiviert_statt_versteckt():
    """Regler, die zur gewählten Zielart nicht gehören, bleiben sichtbar und sind deaktiviert (kein Zustand eines nicht gezeichneten Reglers)."""
    at = _app()                                           # Standard: Ring
    assert at.sidebar.slider(key="dim").disabled and at.sidebar.toggle(key="gradient").disabled and at.sidebar.toggle(key="kovarianz").disabled
    assert at.sidebar.radio(key="gipfel").disabled and at.sidebar.toggle(key="kopplung_ok").disabled and not at.sidebar.slider(key="breite").disabled
    at.sidebar.selectbox(key="ziel").set_value("stetig").run()
    assert not at.sidebar.slider(key="dim").disabled and not at.sidebar.toggle(key="gradient").disabled and not at.sidebar.radio(key="gipfel").disabled
    assert at.sidebar.toggle(key="kopplung_ok").disabled
    at.sidebar.selectbox(key="ziel").set_value("torus").run()
    at.sidebar.slider(key="breite").set_value(24).run()
    assert not at.sidebar.toggle(key="kopplung_ok").disabled and at.sidebar.slider(key="dim").disabled


def test_permalink_unbekannte_werte_werden_verworfen_und_grenzen_eingehalten():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["z"] = "kegel"
    at.query_params["w"] = "99"
    at.query_params["g"] = "vielleicht"
    at.query_params["kn"] = "13"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["ziel"] == "ring" and at.session_state["breite"] == C.BREITE_MAX
    assert at.session_state["gradient"] is False and at.session_state["kost_n"] == C.DEFAULT_KOST_N
