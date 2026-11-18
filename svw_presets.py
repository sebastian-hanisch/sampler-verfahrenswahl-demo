"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import svw_constants as C


def _wahr(text: str) -> bool:
    if text.lower() in ("1", "true", "ja"):
        return True
    if text.lower() in ("0", "false", "nein"):
        return False
    raise ValueError(text)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type | object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None   # erlaubte Werte (Auswahlregler); ein Wert außerhalb wird aus der Adresszeile verworfen


SETTING_SPECS = {
    # Seitenleiste: die Aufgabe
    "ziel": SettingSpec("z", str, "ring", choices=C.ZIELE),
    "breite": SettingSpec("w", int, C.DEFAULT_BREITE, C.BREITE_MIN, C.BREITE_MAX),
    "dim": SettingSpec("d", int, C.DEFAULT_DIM, C.DIM_MIN, C.DIM_MAX),
    "gradient": SettingSpec("g", _wahr, False, choices=(True, False)),
    "kovarianz": SettingSpec("kv", _wahr, False, choices=(True, False)),
    "gipfel": SettingSpec("gi", str, "unbekannt", choices=C.GIPFEL),
    "kopplung_ok": SettingSpec("kp", _wahr, False, choices=(True, False)),
    "seed": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    # Hauptbereich: Experiment 2
    "kost_modell": SettingSpec("km", str, "Ring", choices=C.KOST_MODELLE),
    "kost_n": SettingSpec("kn", int, C.DEFAULT_KOST_N, choices=C.KOST_N),
    "kost_lam": SettingSpec("kl", int, C.DEFAULT_KOST_LAM, choices=C.KOST_LAM),
    "kost_f": SettingSpec("kf", str, "Teilchenzahl", choices=C.KOST_FUNKTIONEN),
    # Experiment 3
    "proben": SettingSpec("b", int, C.DEFAULT_PROBEN, C.PROBEN_MIN, C.PROBEN_MAX),
    "lam_dichte": SettingSpec("ld", int, C.DEFAULT_LAM_DICHTE, choices=C.LAM_DICHTE),
    # Experiment 4
    "diag_rep": SettingSpec("r", int, C.DEFAULT_REP, C.REP_MIN, C.REP_MAX),
    "diag_kern": SettingSpec("k", str, "richtig", choices=("richtig", "falsch")),
}
PRESET_KEYS = {k: k for k in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt; unbekannte Auswahlwerte werden verworfen."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None and value not in spec.choices:
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler auf den Wert des Presets setzen, nicht genannte Regler auf ihren Standardwert."""
    preset = C.PRESETS[name]
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = preset.get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(k, spec.default) == preset.get(k, spec.default) for k, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed"] = random.randint(0, C.SEED_MAX)
