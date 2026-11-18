"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Slider und jeden Auswahlregler
an seine Grenzen bzw. durch seine Werte und prüft, dass kein Python-Fehler auftritt, insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig identischen Inhalt rendern und kollidieren) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import svw_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app(ziel=None) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    if ziel:
        at.sidebar.selectbox(key="ziel").set_value(ziel).run(timeout=TIMEOUT)      # stetiges Ziel: auch der Regler „Dimension“ ist wirksam
        assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app("stetig").slider]
    assert len(labels) == 4, labels                          # Breite, Dimension, Direktproben, Wiederholungen
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app("stetig")
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("breite", [3, 4, 12, 13, 16, 17, 40])
def test_breiten_um_die_grenzen_laufen(breite):
    """Breite 3 (kleinster Torus), 12 (größte live gezogene), 13 (nur Tabelle), 16 und 17 (Grenze 2 207), 40."""
    at = _fresh_app()
    at.sidebar.slider(key="breite").set_value(breite).run(timeout=TIMEOUT)
    assert not at.exception, (breite, [str(e) for e in at.exception])


def test_jeder_auswahlregler_durch_alle_werte():
    at = _fresh_app()
    for key, werte in (("kost_n", C.KOST_N), ("kost_modell", ("Torus 4×4", "Ring")), ("kost_lam", C.KOST_LAM), ("kost_f", C.KOST_FUNKTIONEN), ("lam_dichte", C.LAM_DICHTE),
                       ("diag_kern", ("falsch", "richtig"))):
        for wert in werte:
            for art in ("radio", "select_slider"):
                try:
                    regler = getattr(at.main, art)(key=key)
                except KeyError:
                    continue
                regler.set_value(wert).run(timeout=TIMEOUT)
                assert not at.exception, (key, wert, [str(e) for e in at.exception])
    for ziel in C.ZIELE:
        at.sidebar.selectbox(key="ziel").set_value(ziel).run(timeout=TIMEOUT)
        assert not at.exception, (ziel, [str(e) for e in at.exception])
