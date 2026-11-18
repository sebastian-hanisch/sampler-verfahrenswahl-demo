"""Verdikte der App als reine Funktionen: jede gibt (Art, Text) zurück, die App ruft `getattr(st, Art)(Text)`; `tests/test_verdicts.py` prüft jede Zahl im Text.

Alle Zahlen in den Texten stammen aus den übergebenen Messwerten (eingefrorene Zahlen aus `svw_kosten.json` und `svw_iat8.json`, Live-Rechnungen der App), nie aus
dem Gedächtnis. Zeilen- und Zellzählungen werden aus den Daten gebildet."""

import svw_constants as C
from svw_format import de, ganz, gross, mb, zehnerpotenz
from svw_tree import Aufgabe, Weg


def lam_text(lam: float) -> str:
    return de(lam, 0) if float(lam).is_integer() else de(lam, 1)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: der Weg
# ----------------------------------------------------------------------------------------------------------------------------------------
def verdict_weg(a: Aufgabe, w: Weg, torus8: dict | None = None) -> tuple[str, str]:
    """Verdikt der Frage „Muss man überhaupt abtasten?“. `torus8`: Heat-bath-τ auf dem 8×8-Torus je λ (Messung), nur für Breite 8."""
    if a.ziel == "konjugat":
        return "success", ("Konjugater Prior: Der Posterior steht in geschlossener Form da (Gamma–Poisson). **Direkt ziehen oder den Mittelwert ablesen, kein Sampler, kein Einbrennen, "
                           "keine Diagnostik.** Die Zahlen des Beispiels stehen in Zeile 1 der Tabelle.")
    if a.ziel == "gauss":
        return "success", ("Gauß-Ziel oder Gaußmischung: Komponente wählen, dann Normalverteilung über die Zerlegung der Kovarianz, **direkt ziehen**. "
                           "Das ist Lehrbuchwissen und hier nicht gemessen.")
    if a.ziel == "ring":
        return "success", ("Hard-Core-Ring: Die Transfermatrix hat 2 Zeilenzustände, **direkt ziehen**: n Platzentscheidungen je Probe, kein Einbrennen, keine Diagnostik. "
                           "Was die Kette im Vergleich kostet, zeigt Experiment 2.")
    plaetze = a.breite * a.breite
    if w.weg == "direkt":
        text = (f"Torus {a.breite}×{a.breite}: {ganz(w.zeilenzustaende)} Zeilenzustände, unter der Grenze {ganz(C.ZEILEN_MAX)}: **direkt ziehen** (Transfermatrix), "
                f"{plaetze} Platzentscheidungen je Probe, kein Einbrennen, keine Diagnostik.")
        if a.breite == 8 and torus8:
            lams = sorted(torus8)
            text += (f" Die Heat-bath-Kette braucht je effektiv unabhängiger Teilchenzahl-Probe im Median über {len(torus8[lams[0]]['zufalls_seeds'])} Zufalls-Seeds "
                     f"{ganz(torus8[lams[0]]['tau'])} (λ = {lam_text(lams[0])}) bis {ganz(torus8[lams[-1]]['tau'])} (λ = {lam_text(lams[-1])}) Platzaktualisierungen "
                     f"(Spannen {ganz(torus8[lams[0]]['tau_min'])} bis {ganz(torus8[lams[0]]['tau_max'])} und {ganz(torus8[lams[-1]]['tau_min'])} bis "
                     f"{ganz(torus8[lams[-1]]['tau_max'])}; je {torus8[lams[0]]['ketten']} Ketten × {ganz(torus8[lams[0]]['schritte'])} Schritte).")
        else:
            text += " Für diese Breite ist die Kette hier nicht gemessen (gemessen: Breite 8)."
        return "success", text
    if a.ziel == "torus":
        text = (f"Torus {a.breite}×{a.breite}: {ganz(w.zeilenzustaende)} Zeilenzustände, über der Grenze {ganz(C.ZEILEN_MAX)}: **abtasten** (Heat-bath oder Metropolis). ")
        if a.gipfel == "mehrere":
            text += "Mehrere Gipfel: Tempering; Diagnose: mindestens vier Ketten, Starts in allen bekannten Gipfeln, Split-R̂ < 1,01 und ESS ≥ 400. "
        elif a.gipfel == "unbekannt":
            text += "Gipfelzahl unbekannt: Starts breit streuen, R̂ sieht einen nicht gefundenen Gipfel nicht. "
        text += "**Für diese Größe gibt es keine Messung dieser Demo, die Kosten sind nicht gemessen.**"
        return "warning", text
    return "info", (f"Stetiges Ziel, Dimension {a.dim}: **abtasten**, {w.kern}. Die Kosten zählen Auswertungen von Dichte oder Gradient und lassen sich nicht in "
                    "Platzentscheidungen umrechnen; die Zahlen stehen in den Zeilen 10 bis 12 der Tabelle (aus der Nachbar-Demo, hier nicht gemessen).")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Kosten je effektiv unabhängige Probe
# ----------------------------------------------------------------------------------------------------------------------------------------
def verdict_kosten(b: dict | None, ring_z: dict, torus_z: dict) -> tuple[str, str]:
    """Verdikt der Kostenfrage für eine Zelle `b` (aus `svw_cost.balken`, None = nicht gerechnet); `ring_z` und `torus_z` sind die Zählungen über alle Zellen."""
    if b is None:
        return "info", "Für diese Zelle ist nichts gerechnet (Ring: λ = 1, 5, 20, 100 bei n = 8, 12, 16; Torus 4×4: sieben λ-Werte)."
    lam = lam_text(b["lam"])
    if b["modell"] == "Ring":
        n = b["n"]
        text = (f"Hard-Core-Ring n = {n}, λ = {lam}: direkt {n} Platzentscheidungen; Heat-bath {de(b['kette_teilchen'], 1)} je effektiv unabhängiger Teilchenzahl-Probe und "
                f"{gross(b['kette_gipfel'], 1)} je Gipfel-Probe (plus einmalig {ganz(b['einbrennen'])} Schritte Einbrennen); ")
        if b["kopplung"] is None:
            text += "Kopplung aus der Vergangenheit: für diese Zelle nicht gerechnet. "
        else:
            text += (f"Kopplung aus der Vergangenheit {gross(b['kopplung_schritte'], 0)} Schritte = {gross(b['kopplung'], 0)} Platzentscheidungen je exakter Probe "
                     f"({de(b['kopplung_schritte'] / b['einbrennen'], 2)} · t_mix(1/4)). ")
        text += (f"Über alle Zellen (n ∈ {{8, 12, 16}} × λ ∈ {{1, 5, 20, 100}}, Heat-bath): In {ring_z['kette_teurer_als_direkt_teilchen']} von {ring_z['zellen']} Zellen kostet "
                 f"die Kette mehr als der Direktsampler, in {ring_z['kopplung_teurer_als_gipfel']} von {ring_z['kopplung_gerechnet']} gerechneten Zellen die Kopplung mehr als eine "
                 "effektiv unabhängige Gipfel-Probe der Kette.")
        return "info", text
    text = (f"Torus 4×4, λ = {lam}: direkt {b['direkt']} Platzentscheidungen; Heat-bath τ der Teilchenzahl {de(b['kette_teilchen'], 1)}, "
            f"τ des Gipfelindikators {gross(b['kette_gipfel'], 1)} in derselben Kette: **die effektive Stichprobengröße gilt je Funktion.** "
            f"Die Kette kostet für die Teilchenzahl in {torus_z['kette_teurer_als_direkt_teilchen']} von {torus_z['zellen']} Zellen mehr als der Direktsampler.")
    return "info", text


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Reichweite des exakten Ziehens
# ----------------------------------------------------------------------------------------------------------------------------------------
def verdict_reichweite(w: int, reach: dict, probe: dict | None) -> tuple[str, str]:
    """Verdikt der Reichweitenfrage für die Breite w. `reach`: `svw_direct.reichweite(w)`; `probe`: Direktprobe (lam, proben, rho_exakt, rho_probe, sf, z) oder None."""
    zu = reach["zustaende"]
    zus = "nicht berechnet" if zu is None else (ganz(zu) if isinstance(zu, int) else zehnerpotenz(zu, 1))
    if zu is None:
        text = f"Torus {w}×{w}: Zahl der Zustände hier nicht berechnet, aber nur {ganz(reach['zeilen'])} Zeilenzustände (dichte Matrix {mb(reach['mb'])}). "
    else:
        text = f"Torus {w}×{w}: {zus} Zustände, aber nur {ganz(reach['zeilen'])} Zeilenzustände (dichte Matrix {mb(reach['mb'])}). "
    kind = "info"
    if probe is not None:
        text += (f"{ganz(probe['proben'])} Direktproben treffen die Belegungsdichte {de(probe['rho_exakt'], 5)} mit {de(probe['rho_probe'], 5)} ± {de(probe['sf'], 5)} "
                 f"(λ = {lam_text(probe['lam'])}, Abweichung {de(probe['z'], 2)} Standardfehler). ")
        kind = "success" if abs(probe["z"]) <= 4 else "warning"
    else:
        text += f"Für diese Breite zieht die App nicht live (nur bis Breite {C.LIVE_BREITE_MAX}); die Größe der Matrix steht in der Tabelle. "
    if reach["zeilen"] <= C.ZEILEN_MAX:
        text += f"Unter der Grenze {ganz(C.ZEILEN_MAX)} (Breite 16): exakt ziehbar. "
    else:
        text += (f"Über der Grenze {ganz(C.ZEILEN_MAX)}: **hier nicht gezogen und nicht gemessen**; das Abtasten ist damit nicht widerlegt, aber die dicht gespeicherte Matrix "
                 "wird groß. ")
    text += "Die Grenze des exakten Ziehens ist hier eine Breite, nicht die Zahl der Zustände (gerechnet für diese Modellfamilie, für andere Modelle nicht)."
    return kind, text


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Diagnostik gegen Richtigkeit
# ----------------------------------------------------------------------------------------------------------------------------------------
def verdict_diagnose(kern: str, d: dict, vor: dict, vor_richtig: dict) -> tuple[str, str]:
    """Verdikt der Diagnosefrage: `d` aus `svw_diag.pruefe_wiederholungen`, `vor`/`vor_richtig` aus `svw_diag.vorprobe` des gewählten bzw. des richtigen Kerns."""
    reps = d["wiederholungen"]
    kopf = (f"{d['n_beides']} von {reps} Wiederholungen bestehen Split-R̂ < {de(C.RHAT_SCHWELLE, 2)} und ESS ≥ {C.ESS_SCHWELLE}, "
            f"das Mittel der Teilchenzahl liegt bei {de(d['mittel_mittel'], 3)} (exakt {de(d['exakt'], 3)}), im Mittel {de(d['mittel_abs_z'], 1)} Standardfehler daneben "
            f"({d['n_ueber3']} von {reps} Wiederholungen über 3 Standardfehler).")
    if kern == "falsch":
        text = ("**Falscher Kern (Annahmeregel vergessen):** " + kopf + f" Die exakte Vorprobe findet ihn: max |π P − π| = {de(vor['max_abw'], 4)}, beim richtigen Kern "
                f"{zehnerpotenz(vor_richtig['max_abw'], 1)}. R̂ vergleicht Ketten untereinander, nicht mit dem Ziel; auf Zielen ohne Orakel gibt es die Vorprobe nicht.")
        if d["n_beides"] == reps:
            text += f" ({reps} von {reps} heißt: kein Gegenfall in {reps} Wiederholungen, nicht Wahrscheinlichkeit 1.)"
        return "warning", text
    text = ("**Richtiger Kern (Heat-bath):** " + kopf + f" Exakte Vorprobe: max |π P − π| = {zehnerpotenz(vor['max_abw'], 1)}. Der Anteil bestandener Wiederholungen ist kein "
            "Gütevergleich der Kerne: der falsche Kern mischt schneller (die Gleichverteilung ist leichter zu erreichen) und besteht R̂ deshalb häufiger.")
    return "success", text


def verdict_tabelle(satz: str) -> tuple[str, str]:
    """Verdikt der Einordnung: der gezählte Belegungsgrad der Tabelle."""
    text = f"{satz}; keine Zahl stammt aus einem anderen Modell als dem genannten."
    return "info", text
