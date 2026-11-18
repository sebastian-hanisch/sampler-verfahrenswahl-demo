"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, und je Preset die erwartete Verdikt-Zeile in der App."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import svw_constants as C
import svw_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def test_jedes_preset_hat_einen_hilfetext_und_nur_bekannte_regler():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_sind_erlaubt():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.choices is not None:
                assert wert in spec.choices, (name, key)
            if spec.lo is not None:
                assert spec.lo <= wert <= spec.hi, (name, key)


def test_das_erste_preset_ist_die_standardeinstellung_und_alle_presets_unterscheiden_sich():
    assert C.PRESETS[next(iter(C.PRESETS))] == {}
    voll = [tuple(sorted({k: p.get(k, s.default) for k, s in P.SETTING_SPECS.items()}.items(), key=lambda kv: kv[0])) for p in C.PRESETS.values()]
    assert len(set(voll)) == len(C.PRESETS)


def test_standardwerte_liegen_in_den_grenzen():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        if spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi, key


def test_url_parameter_sind_eindeutig():
    urls = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(set(urls)) == len(urls)


def test_wahr_caster():
    assert P._wahr("true") is True and P._wahr("False") is False and P._wahr("1") is True
    with pytest.raises(ValueError):
        P._wahr("vielleicht")


# Erwartete Verdikt-Zeile je Preset (Anfang des Textes der ersten Verdikt-Meldung der Kernfrage 1): ein Preset ohne Effekt meldet keinen Effekt.
ERWARTET = {
    "Ring n = 16, λ = 5: direkt oder Kette?": "Hard-Core-Ring: Die Transfermatrix hat 2 Zeilenzustände",
    "Gamma–Poisson: kein Sampler": "Konjugater Prior",
    "Torus 8×8: noch exakt": "Torus 8×8: 47 Zeilenzustände, unter der Grenze 2 207",
    "Torus 24×24 kalt: abtasten": "Torus 24×24: 103 682 Zeilenzustände, über der Grenze 2 207",
    "Posterior in 48 Dimensionen": "Stetiges Ziel, Dimension 48",
    "HMC mit Gradient (20-d)": "Stetiges Ziel, Dimension 20",
    "Gipfel unbekannt": "Torus 24×24: 103 682 Zeilenzustände, über der Grenze 2 207",
    "Falscher Kern (Annahmeregel vergessen)": "Hard-Core-Ring: Die Transfermatrix hat 2 Zeilenzustände",
}


def _meldungen(at):
    return [e.value for art in ("success", "info", "warning") for e in getattr(at, art)]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_zeigt_die_erwartete_verdikt_zeile(name):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    texte = _meldungen(at)
    assert any(t.startswith(ERWARTET[name]) or ERWARTET[name] in t for t in texte), (name, texte[:3])
    if name == "HMC mit Gradient (20-d)":
        assert any("HMC" in t and "zwei Teile" in t for t in texte + [at.markdown[i].value for i in range(len(at.markdown))])
    if name == "Falscher Kern (Annahmeregel vergessen)":
        assert any(t.startswith("**Falscher Kern (Annahmeregel vergessen):**") for t in texte)
    if name == "Gipfel unbekannt":
        assert any("Kopplung aus der Vergangenheit" in m.value for m in at.markdown) and any("R̂ sieht einen nicht gefundenen Gipfel nicht" in t for t in texte)
    if name == "Torus 24×24 kalt: abtasten":
        assert any("Tempering" in t for t in texte)
