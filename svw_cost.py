"""Vorgerechnete Kosten je effektiv unabhängige Probe (Datei `svw_kosten.json`, erzeugt von `tools/svw_make_kosten.py`) und die gemeinsame Einheit.

Gemeinsame Einheit: Platzentscheidungen. Eine Platzaktualisierung der Kette zählt 1, ein entschiedener Platz des Direktsamplers 1, ein Schritt der Paarkette der
Kopplung aus der Vergangenheit 2 (beide Extremketten aktualisieren je einen Platz). Die Einheit gilt nur für diskrete Ziele; stetige Ziele zählen Auswertungen
von Dichte oder Gradient, und diese lassen sich nicht in Platzentscheidungen umrechnen."""

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RING_N = (8, 12, 16)
RING_LAM = (1.0, 5.0, 20.0, 100.0)
TORUS_LAM = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
FUNKTIONEN = ("Teilchenzahl", "Gipfelindikator")

EINHEITEN = [
    ("Kette (Heat-bath, Metropolis)", "Schritt = eine Platzaktualisierung", "1"),
    ("Direktsampler (Transfermatrix)", "ein entschiedener Platz (je Zeile eine Entscheidung über alle Plätze der Zeile)", "1"),
    ("Kopplung aus der Vergangenheit", "Schritt der Paarkette = ein Schritt beider Extremketten", "2"),
    ("Tempering", "Aktualisierungen aller Replikas zusammen", "1 (das Budget wird auf die Replikas geteilt)"),
    ("Koordinationsspiele (Nachbar-Demos)", "Aktualisierungen je Betrieb", "Betriebe × Aktualisierungen (eigene Folgerung aus der Beschreibung der Einheit)"),
    ("stetige Ziele", "Auswertungen von Dichte oder Gradient", "nicht umrechenbar: ein Gradient kostet je nach Modell mehr als eine Dichte"),
]


@lru_cache(maxsize=1)
def laden() -> dict:
    return json.loads((ROOT / "svw_kosten.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def iat8() -> dict:
    return json.loads((ROOT / "svw_iat8.json").read_text(encoding="utf-8"))


def ring_zelle(n: int, lam: float) -> dict | None:
    for z in laden()["ring"]:
        if z["n"] == n and z["lam"] == float(lam):
            return z
    return None


def torus_zelle(lam: float) -> dict | None:
    for z in laden()["torus44"]:
        if z["lam"] == float(lam):
            return z
    return None


def tau(zelle: dict, funktion: str) -> float:
    if funktion not in FUNKTIONEN:
        raise ValueError(f"Funktion: {FUNKTIONEN}")
    return zelle["tau_teilchen"] if funktion == "Teilchenzahl" else zelle["tau_gipfel"]


def balken(modell: str, n: int, lam: float, funktion: str) -> dict | None:
    """Die Kosten der drei Wege in Platzentscheidungen für eine Zelle; None, wenn die Zelle nicht gerechnet ist.

    modell: "Ring" (n aus 8, 12, 16; λ aus 1, 5, 20, 100) oder "Torus 4×4" (n wird ignoriert; λ aus sieben Werten).
    Rückgabe: direkt, kette (τ der gewählten Funktion), einbrennen (t_mix, nur Ring), kopplung (Platzentscheidungen oder None)."""
    if modell == "Ring":
        z = ring_zelle(n, lam)
        if z is None:
            return None
        return {"modell": modell, "n": n, "lam": float(lam), "funktion": funktion, "direkt": z["direkt"], "kette": tau(z, funktion), "kette_teilchen": z["tau_teilchen"],
                "kette_gipfel": z["tau_gipfel"], "einbrennen": z["t_mix"], "kopplung": z["kopplung_platz"], "kopplung_schritte": z["kopplung_schritte"]}
    if modell == "Torus 4×4":
        z = torus_zelle(lam)
        if z is None:
            return None
        return {"modell": modell, "n": 16, "lam": float(lam), "funktion": funktion, "direkt": z["direkt"], "kette": tau(z, funktion), "kette_teilchen": z["tau_teilchen"],
                "kette_gipfel": z["tau_gipfel"], "einbrennen": None, "kopplung": None, "kopplung_schritte": None}
    raise ValueError("Modell: Ring oder Torus 4×4")


def ring_zellen() -> list[dict]:
    return laden()["ring"]


def ring_zaehlungen() -> dict:
    """Zählungen über alle Ring-Zellen (n ∈ {8, 12, 16} × λ ∈ {1, 5, 20, 100}) für die Verdikte und die README."""
    zellen = ring_zellen()
    gerechnet = [z for z in zellen if z["kopplung_schritte"] is not None]
    return {
        "zellen": len(zellen),
        "kette_teurer_als_direkt_teilchen": sum(z["tau_teilchen"] > z["direkt"] for z in zellen),
        "kette_teurer_als_direkt_gipfel": sum(z["tau_gipfel"] > z["direkt"] for z in zellen),
        "tau_teilchen_je_n": (min(z["tau_teilchen"] / z["n"] for z in zellen), max(z["tau_teilchen"] / z["n"] for z in zellen)),
        "tau_gipfel_je_n": (min(z["tau_gipfel"] / z["n"] for z in zellen), max(z["tau_gipfel"] / z["n"] for z in zellen)),
        "kopplung_gerechnet": len(gerechnet),
        "kopplung_je_tmix": (min(z["kopplung_schritte"] / z["t_mix"] for z in gerechnet), max(z["kopplung_schritte"] / z["t_mix"] for z in gerechnet)),
        "kopplung_teurer_als_gipfel": sum(z["kopplung_platz"] > z["tau_gipfel"] for z in gerechnet),
        "kopplung_je_gipfel": (min(z["kopplung_platz"] / z["tau_gipfel"] for z in gerechnet), max(z["kopplung_platz"] / z["tau_gipfel"] for z in gerechnet)),
        "kopplung_je_teilchen": (min(z["kopplung_platz"] / z["tau_teilchen"] for z in gerechnet), max(z["kopplung_platz"] / z["tau_teilchen"] for z in gerechnet)),
        "gipfel_je_teilchen": (min(z["tau_gipfel"] / z["tau_teilchen"] for z in zellen), max(z["tau_gipfel"] / z["tau_teilchen"] for z in zellen)),
    }


def torus_zaehlungen() -> dict:
    zellen = laden()["torus44"]
    return {
        "zellen": len(zellen),
        "kette_teurer_als_direkt_teilchen": sum(z["tau_teilchen"] > z["direkt"] for z in zellen),
        "tau_teilchen_je_direkt": (min(z["tau_teilchen"] / z["direkt"] for z in zellen), max(z["tau_teilchen"] / z["direkt"] for z in zellen)),
        "tau_teilchen": (min(z["tau_teilchen"] for z in zellen), max(z["tau_teilchen"] for z in zellen)),
        "lam_max_tau_teilchen": max(zellen, key=lambda z: z["tau_teilchen"])["lam"],
    }


def takt_zellen() -> list[dict]:
    return laden()["takt"]


def tau_torus8() -> dict:
    """Heat-bath-τ der Teilchenzahl auf dem 8×8-Torus (Messung, `svw_iat8.json`), je λ."""
    return {z["lam"]: z for z in iat8()["torus8x8"]}
