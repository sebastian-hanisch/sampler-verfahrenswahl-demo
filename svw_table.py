"""Entscheidungstabelle mit Herkunft und Status je Zeile (Daten in `svw_table.json`).

Jede Zeile nennt die Frage, die Antwort, die Zahl mit ihrer Voraussetzung, die Demo, aus der die Zahl stammt, den Status (gemessen, Quelle, nicht gemessen) und
den Stand der Prüfung in dieser Demo:
  „hier gerechnet“               die Zahl ist in dieser Demo nachgerechnet und von einem Test geprüft;
  „teilweise hier gerechnet“     ein Teil der Zahlen ist hier nachgerechnet, der Rest steht so in der Nachbar-Demo;
  „aus Nachbar-Demo übernommen“  die Zahl steht in der README der genannten Demo und ist hier nicht neu gerechnet;
  „Messreihe, hier nicht neu gerechnet“  die Zahl stammt aus einer eigenen Messreihe, die in keiner Nachbar-Demo steht;
  „Quelle“                       Wortlaut der Quelle gelesen, nicht gemessen;
  „–“                            keine Zahl.
„Gemessen“ heißt: in diesem Modell, bei diesen Parametern, in der genannten Demo."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS = ("gemessen", "Quelle", "nicht gemessen")
STAND = ("hier gerechnet", "teilweise hier gerechnet", "aus Nachbar-Demo übernommen", "Messreihe, hier nicht neu gerechnet", "Quelle", "–")


def zeilen() -> list[dict]:
    return json.loads((ROOT / "svw_table.json").read_text(encoding="utf-8"))["zeilen"]


def zeile(nr: int) -> dict:
    for z in zeilen():
        if z["nr"] == nr:
            return z
    raise KeyError(nr)


def zaehle(rows=None) -> dict:
    """Zählung der Status in der Tabelle (aus den Zeilen, nicht aus dem Gedächtnis)."""
    rows = zeilen() if rows is None else rows
    out = {s: sum(1 for z in rows if z["status"] == s) for s in STATUS}
    out["zeilen"] = len(rows)
    return out


def belegungssatz(rows=None) -> str:
    c = zaehle(rows)
    return f"{c['zeilen']} Zeilen: {c['gemessen']} gemessen, {c['Quelle']} Quelle, {c['nicht gemessen']} nicht gemessen"
