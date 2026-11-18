"""Gleicht die Zahlen der Entscheidungstabelle (`svw_table.json`) mit den READMEs der Nachbar-Demos ab.

Aufruf (Python des Test-venv, aus dem Repo-Ordner):
    python tools/svw_check_table.py [--ordner ..]           liest <ordner>/<demo>/README.md der Nachbar-Demos (Voreinstellung: der Elternordner dieses Repos)
    python tools/svw_check_table.py --github                liest stattdessen `gh api repos/sebastian-hanisch/<demo>/readme` (braucht gh und Netz)

Je Zeile mit Nachbar-Demo (Feld `nachbar`) und Prüfzahlen (Feld `pruefzahlen`): jede Prüfzahl muss im Text der README vorkommen. Für den Vergleich gilt:
  * Leerräume (Leerzeichen, geschützte und schmale Leerzeichen, Zeilenumbrüche) zählen als beliebig viele Leerräume;
  * eine Prüfzahl, die mit einer Ziffer beginnt oder endet, muss als ganze Zahl dastehen: „53“ trifft nicht „1 053“, „0,222“ nicht „0,2225“, „16“ nicht „0,16“;
  * Fundstellen in Angaben zur ersten Rechnung („erste Rechnung 1,7 und 15,6“, bis zur nächsten Klammer, zum Semikolon, zum Zeilenende oder zum Satzende) zählen nicht:
    eine Zahl, die dort nur als überholter Wert steht, ist nicht abgeglichen.
Ergebnis je Zeile: „abgeglichen“, „FEHLT: <Zahlen>“ oder „nicht abgeglichen“ (die README liegt nicht vor).
Der Rückgabewert ist 1, wenn mindestens eine Zeile „FEHLT“ meldet, sonst 0; fehlende READMEs sind kein Fehler, nur ein Stand."""

import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Angabe zur ersten Rechnung: ab dem Stichwort bis zur schließenden Klammer, zum Semikolon, zum Zeilenende oder zum Satzende (Punkt, Leerraum, Großbuchstabe).
_ERSTE_RECHNUNG = re.compile(r"erste[nr]?\s+Rechnung(?:(?!\.\s+[A-ZÄÖÜ])[^)\n;])*")
_ZIFFER_VOR = r"(?<![\d,])(?<!\d\s)"        # nicht Teil einer längeren Zahl (auch nicht hinter einem Tausenderleerraum)
_ZIFFER_NACH = r"(?!\d|,\d)"


def ohne_erste_rechnung(text: str) -> str:
    """Entfernt die Angaben zur ersten Rechnung, damit überholte Zahlen nicht als Beleg zählen."""
    return _ERSTE_RECHNUNG.sub(" ", text)


def muster(pruefzahl: str) -> re.Pattern:
    """Suchmuster einer Prüfzahl: Leerräume beliebig, Zahlen als ganze Zahlen."""
    teile = pruefzahl.split()
    kern = r"\s*".join(re.escape(t) for t in teile)
    if pruefzahl[:1].isdigit():
        kern = _ZIFFER_VOR + kern
    if pruefzahl[-1:].isdigit():
        kern = kern + _ZIFFER_NACH
    return re.compile(kern)


def kommt_vor(pruefzahl: str, text: str) -> bool:
    return muster(pruefzahl).search(ohne_erste_rechnung(text)) is not None


def lies_readme(demo: str, ordner: Path | None, github: bool) -> str | None:
    if github:
        try:
            out = subprocess.run(["gh", "api", f"repos/sebastian-hanisch/{demo}/readme", "--jq", ".content"], capture_output=True, text=True, timeout=60, check=True)
            return base64.b64decode(out.stdout.strip()).decode("utf-8")
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    pfad = ordner / demo / "README.md"
    return pfad.read_text(encoding="utf-8") if pfad.exists() else None


def pruefe(zeilen: list[dict], ordner: Path | None, github: bool = False, leser=None) -> list[dict]:
    """Ergebnis je Zeile: {nr, nachbar, ergebnis, fehlt}; `leser(demo)` ersetzt das Lesen (für Tests)."""
    ergebnisse = []
    for z in zeilen:
        nachbar, zahlen = z.get("nachbar", ""), z.get("pruefzahlen", [])
        if not nachbar or not zahlen:
            continue
        text = leser(nachbar) if leser else lies_readme(nachbar, ordner, github)
        if text is None:
            ergebnisse.append({"nr": z["nr"], "nachbar": nachbar, "ergebnis": "nicht abgeglichen", "fehlt": []})
            continue
        bereinigt = ohne_erste_rechnung(text)
        fehlt = [p for p in zahlen if muster(p).search(bereinigt) is None]
        ergebnisse.append({"nr": z["nr"], "nachbar": nachbar, "ergebnis": "abgeglichen" if not fehlt else "FEHLT", "fehlt": fehlt})
    return ergebnisse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ordner", type=Path, default=ROOT.parent)
    ap.add_argument("--github", action="store_true")
    a = ap.parse_args()
    zeilen = json.loads((ROOT / "svw_table.json").read_text(encoding="utf-8"))["zeilen"]
    erg = pruefe(zeilen, a.ordner, a.github)
    for e in erg:
        print(f"Zeile {e['nr']:>2} ({e['nachbar']}): {e['ergebnis']}" + (": " + ", ".join(e["fehlt"]) if e["fehlt"] else ""))
    zaehl = {k: sum(1 for e in erg if e["ergebnis"] == k) for k in ("abgeglichen", "FEHLT", "nicht abgeglichen")}
    print(f"{len(erg)} Zeilen mit Nachbar-Demo und Prüfzahlen: {zaehl['abgeglichen']} abgeglichen, {zaehl['FEHLT']} mit fehlender Zahl, {zaehl['nicht abgeglichen']} nicht abgeglichen")
    return 1 if zaehl["FEHLT"] else 0


if __name__ == "__main__":
    sys.exit(main())
