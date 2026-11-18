"""Entscheidungsweg: Muss man abtasten, und womit? Eine reine Funktion `waehle(aufgabe)` liefert den durchlaufenen Weg, die Empfehlung, die Diagnoseliste und die
Zeilen der Entscheidungstabelle, die zur Aufgabe gehören.

Die Fragen werden in dieser Reihenfolge gestellt:
  0  Lässt sich π direkt und exakt ziehen (geschlossene Form, Transfermatrix, Zerlegung der Kovarianz)?
  1  Sonst: welcher Raum (endlich oder stetig)?
  2  Welcher Kern (Heat-bath oder Metropolis, Zufallsschritt oder HMC; mit oder ohne Kopplung, mit oder ohne Tempering)?
  3  Welche Diagnostik, und was sieht sie nicht?
  4  Welche Fehlerangabe?

Die Schwellen und Texte sind eine Setzung dieser Demo, keine Messung der Rechenzeit; die Zahlen stehen in der Entscheidungstabelle (`svw_table.json`) mit
Herkunft und Status."""

from dataclasses import dataclass

import svw_constants as C
import svw_direct as D

REPO = {
    "exakt": "exakt-oder-abtasten-demo",
    "kette": "kette-gleichgewicht-demo",
    "metropolis": "metropolis-kleiner-raum-demo",
    "gibbs": "gibbs-glauber-demo",
    "mischzeit": "mischzeit-exakt-demo",
    "kopplung_ketten": "kopplung-ketten-demo",
    "kopplung": "kopplung-vergangenheit-demo",
    "diagnose": "diagnose-taeuscht-demo",
    "tempering": "gipfel-tempering-demo",
    "hmc": "dimension-hmc-demo",
    "posterior": "posterior-sampling-demo",
}

DIAGNOSE_BASIS = (
    "mindestens vier Ketten",
    "überstreute Starts",
    "Split-R̂ < 1,01 und ESS ≥ 400 (Vehtari u. a. 2021); die alte Schwelle 1,1 ist veraltet",
    "exakte Vorprobe π P = π auf einer kleinen Instanz des Modells, wenn es eine gibt",
    "Monte-Carlo-Standardfehler berichten (Flegal u. a.)",
)


@dataclass(frozen=True)
class Aufgabe:
    ziel: str = "ring"
    breite: int = C.DEFAULT_BREITE
    dim: int = C.DEFAULT_DIM
    gradient: bool = False
    kovarianz: bool = False
    gipfel: str = "unbekannt"
    kopplung_ok: bool = False


@dataclass(frozen=True)
class Frage:
    nr: int
    text: str
    antwort: str
    erreicht: bool


@dataclass(frozen=True)
class Weg:
    weg: str                       # "direkt" oder "abtasten"
    verfahren: str
    fragen: tuple
    nachbarn: tuple                # Ordnernamen der Nachbar-Demos
    diagnose: tuple
    tabellenzeilen: tuple          # Nummern der Entscheidungstabelle
    zeilenzustaende: int | None
    kern: str | None


def _fragen(erreicht_bis: int, antworten: dict) -> tuple:
    texte = {
        0: "Lässt sich π direkt und exakt ziehen?",
        1: "Welcher Raum?",
        2: "Welcher Kern?",
        3: "Welche Diagnostik?",
        4: "Welche Fehlerangabe?",
    }
    return tuple(Frage(i, texte[i], antworten.get(i, "entfällt"), i <= erreicht_bis) for i in range(5))


def _pruefe(a: Aufgabe) -> None:
    if a.ziel not in C.ZIELE:
        raise ValueError(f"Zielart: {C.ZIELE}")
    if a.gipfel not in C.GIPFEL:
        raise ValueError(f"Gipfel: {C.GIPFEL}")
    if a.ziel == "torus" and not C.BREITE_MIN <= a.breite <= C.BREITE_MAX:
        raise ValueError(f"Breite: {C.BREITE_MIN} bis {C.BREITE_MAX}")
    if a.ziel in ("gauss", "stetig") and a.dim < 1:
        raise ValueError("Dimension mindestens 1")


def waehle(a: Aufgabe) -> Weg:
    """Weg für die Aufgabe `a` (siehe Modulkopf)."""
    _pruefe(a)
    if a.ziel == "konjugat":
        return Weg("direkt", "direkt ziehen: der Posterior hat eine geschlossene Form (Gamma)", _fragen(0, {0: "ja: konjugater Prior, Posterior in geschlossener Form"}),
                   (REPO["exakt"],), (), (1,), None, None)
    if a.ziel == "gauss":
        return Weg("direkt", "direkt ziehen: Komponente wählen, dann Normalverteilung über die Zerlegung der Kovarianz (Lehrbuch, hier nicht gemessen)",
                   _fragen(0, {0: "ja: Gauß-Ziel, Zerlegung der Kovarianz"}), (), (), (4,), None, None)
    if a.ziel in ("ring", "torus"):
        zeilen = 2 if a.ziel == "ring" else D.zeilen_zustaende(a.breite)
        if zeilen <= C.ZEILEN_MAX:
            name = "Hard-Core-Ring" if a.ziel == "ring" else f"Torus der Breite {a.breite}"
            return Weg("direkt", f"direkt ziehen über die Transfermatrix ({name}: {zeilen} Zeilenzustände, Grenze {C.ZEILEN_MAX})",
                       _fragen(0, {0: f"ja: {zeilen} Zeilenzustände ≤ {C.ZEILEN_MAX}"}), (), (), (3,), zeilen, None)
        return _abtasten(a, zeilen)
    return _abtasten(a, None)


def _abtasten(a: Aufgabe, zeilen) -> Weg:
    diag = list(DIAGNOSE_BASIS)
    nachbarn = []
    tab = [13, 16]
    if a.ziel == "torus":
        raum = f"endlich: Torus der Breite {a.breite}, {zeilen} Zeilenzustände (über der Grenze {C.ZEILEN_MAX})"
        kern = "Heat-bath oder Metropolis (gleiche Verteilung, anderer Takt)"
        nachbarn += [REPO["metropolis"], REPO["gibbs"], REPO["mischzeit"]]
        tab += [3, 5, 7, 15]
        if a.kopplung_ok:
            kern += "; exakte Proben mit Kopplung aus der Vergangenheit möglich (im geraden Ring 4,27 bis 5,16 t_mix(1/4) Schritte je Probe, Experiment 2; für den Torus nicht gemessen)"
            nachbarn.append(REPO["kopplung"])
            tab.append(8)
        antwort0 = f"nein: {zeilen} Zeilenzustände > {C.ZEILEN_MAX}"
        antwort1 = raum
    else:
        raum = f"stetig, Dimension {a.dim}, keine geschlossene Form"
        nachbarn += [REPO["posterior"], REPO["hmc"]]
        tab += [10, 12]
        if a.dim == 1:
            kern = "Zufallsschritt-Metropolis mit Schrittweite 2,38/√d"
        elif a.gradient:
            kern = ("HMC (Vorbehalt: der Gewinn hat zwei Teile, Kenntnis der Kovarianz und Gradient in der Dimension; "
                    "gegen den Zufallsschritt mit bekannter Kovarianz vergleichen)")
            tab.append(11)
        elif a.kovarianz:
            kern = "Zufallsschritt-Metropolis mit der Kovarianz als Vorschlag, Schrittweite 2,38/√d"
        else:
            kern = "Zufallsschritt-Metropolis, isotroper Vorschlag, Schrittweite 2,38/√d"
        antwort0 = "nein: stetiges Ziel ohne geschlossene Form"
        antwort1 = raum
    if a.gipfel == "mehrere":
        kern += "; Tempering"
        nachbarn.append(REPO["tempering"])
        tab.append(9)
        diag.append("Starts in allen bekannten Gipfeln")
    elif a.gipfel == "unbekannt":
        diag.append("Starts breit streuen; R̂ sieht einen nicht gefundenen Gipfel nicht")
        nachbarn.append(REPO["diagnose"])
        tab.append(14)
    else:
        diag.append("ein Gipfel ist hier eine Annahme, keine Messung")
    antworten = {0: antwort0, 1: antwort1, 2: kern, 3: "; ".join(diag[:3]) + " …", 4: "Standardfehler aus der effektiven Stichprobengröße der gewählten Funktion"}
    return Weg("abtasten", kern, _fragen(4, antworten), tuple(dict.fromkeys(nachbarn)), tuple(diag), tuple(sorted(set(tab))), zeilen, kern)
