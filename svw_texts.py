"""Texte der App, die auch in der README stehen (ein Test vergleicht beide Fassungen): Einleitung, Modell, Grenzen und die Formeln des 📐-Expanders."""

GITHUB = "https://github.com/sebastian-hanisch/"

NACHBARN = [
    ("exakt-oder-abtasten-demo", "Wann man nicht abtasten muss: Gamma–Poisson und die Gate-Kette"),
    ("kette-gleichgewicht-demo", "Kette und Gleichgewicht: Übergangsmatrix, stationäre Verteilung"),
    ("metropolis-kleiner-raum-demo", "Metropolis im kleinen Raum: Vorschlagsfallen und falsch gebaute Kerne"),
    ("gibbs-glauber-demo", "Heat-bath gegen Metropolis: der Takt"),
    ("mischzeit-exakt-demo", "Exakte Mischzeit und Schranken"),
    ("kopplung-ketten-demo", "Kopplung zweier Ketten und die Kopplungszeit"),
    ("kopplung-vergangenheit-demo", "Exakte Proben durch Kopplung aus der Vergangenheit"),
    ("diagnose-taeuscht-demo", "Wenn R̂ täuscht: alle Ketten im selben Gipfel"),
    ("gipfel-tempering-demo", "Mehrere Gipfel und Tempering"),
    ("dimension-hmc-demo", "Dimension, Zufallsschritt und HMC"),
    ("posterior-sampling-demo", "R̂, ESS, Zufallsschritt und pCN auf einem stetigen Posterior (Inverse Probleme)"),
    ("markov-queue-demo", "Warteschlangen als Markov-Kette (Gate-Kette)"),
    ("monte-carlo-verfahrenswahl-demo", "Verfahrenswahl im Monte Carlo (Frage 0: „Lässt sich die Verteilung nicht ziehen?“)"),
]

EINLEITUNG = """
Gegeben ist ein Ziel π, gesucht sind Proben oder Mittelwerte unter π. Diese Seite fragt der Reihe nach: **Muss man überhaupt abtasten**, oder lässt sich π direkt und exakt
ziehen? Wenn nicht: welcher **Kern**, welche **Diagnostik**, welche **Fehlerangabe**? Gemessen wird in **Platzentscheidungen je effektiv unabhängiger Probe**, nicht in Sekunden.
Alle Aufgaben sind **ausgedachte Modelle** (Stellplätze im Ring und auf dem Torus, eine Gaußmischung, eine Ankunftsrate); es gibt keine reale Datenquelle.

Weiter unten: die Experimente **🔬 Kosten**, **🔬 Reichweite des exakten Ziehens** und **🔬 Wer täuscht wen?**, der Expander **Woher die Zahlen kommen** (Entscheidungstabelle),
**🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Frage 0: Lässt sich π direkt und exakt ziehen?** Geschlossene Form (Gamma–Poisson), Zerlegung der Kovarianz (Gauß), Transfermatrix (Ring, Streifen): dann ist man fertig,
   ohne Einbrennen und ohne Diagnostik.
2. **Frage 1 und 2: Raum und Kern.** Endlicher Raum: Heat-bath oder Metropolis (gleiche Verteilung, anderer Takt), auf Wunsch mit Kopplung aus der Vergangenheit für exakte Proben
   oder mit Tempering gegen mehrere Gipfel. Stetiger Raum: Zufallsschritt oder HMC.
3. **Frage 3 und 4: Diagnostik und Fehlerangabe.** Mindestens vier Ketten, überstreute Starts, Split-R̂ < 1,01 und ESS ≥ 400, eine exakte Vorprobe auf einer kleinen Instanz, wenn es
   eine gibt, und den Monte-Carlo-Standardfehler.
4. **Warum die Einheit zählt:** Die Zahlen der Nachbar-Demos zählen Schritte, Aktualisierungen, Läufe oder Auswertungen. Eine Tabelle, die sie nebeneinanderstellt, braucht eine
   gemeinsame Einheit; hier ist es die Platzentscheidung (Umrechnung im Expander „Woher die Zahlen kommen“).

**Ein Beispiel von Hand: Hard-Core-Ring mit 4 Plätzen, λ = 2** (7 Belegungen). Z = 1 + 4·2 + 2·4 = 17. Der Direktsampler zieht in 4 Platzentscheidungen; P(Platz 0 belegt) = 6/17 und
die Belegung {0, 2} entsteht mit 6/17 · 1 · 2/3 · 1 = 4/17 = π({0, 2}). Der Heat-bath braucht einmalig t_mix(1/4) = 22 Schritte Einbrennen und danach τ = 1195/221 ≈ 5,4 Schritte je
effektiv unabhängiger Teilchenzahl-Probe, aber τ = 53 Schritte je Probe des Gipfelindikators. Die Kopplung aus der Vergangenheit kostet im Mittel 88,0 Schritte der Paarkette = 176,1
Platzentscheidungen je exakter Probe. Reihenfolge: 4 < 5,4 < 53 < 176.
"""

MODELL = """
Alle Aufgaben sind ausgedacht: Stellplätze im Ring und auf dem Torus (keine zwei benachbarten belegt, Gewicht λ je belegtem Platz), eine Gaußmischung und eine Ankunftsrate mit erfundenen
Zahlen. Es gibt keine reale Datenquelle. Gemessen wird in Platzentscheidungen je effektiv unabhängiger Probe, nicht in Sekunden. Die Tabelle gilt für diese Modelle und diese Parameter;
was sie für ein echtes Ziel heißt, wird nicht behauptet. Alle hier vorgeführten Ziele lassen sich direkt ziehen (das ist der Grund, warum es ein exaktes Orakel gibt); ob Abtasten bessere
Betriebsentscheidungen liefert, ist in den gelesenen Quellen nicht belegt.
"""

GRENZEN_TABELLE = """
| Annahme | Was passiert, wenn sie verletzt ist | Wo steht mehr |
|---|---|---|
| **Das Verfahren ist vorgegeben und es gibt ein Orakel.** | Auf einem echten Ziel gibt es kein Orakel, nur Diagnostik; sie prüft Ketten untereinander, nicht das Ziel (Experiment 4). | `diagnose-taeuscht-demo`, `metropolis-kleiner-raum-demo` |
| **Alle vorgeführten Ziele lassen sich direkt ziehen.** | Das ist gewollt, denn nur dort gibt es ein exaktes Orakel; die Seite führt also Sampler an Zielen vor, die man nicht abtasten müsste. | Zeile 4 und 17 der Tabelle |
| **Ausgedachte Modelle, Heat-bath, diese Funktionen.** | Die Zahlen gelten für Hard-Core-Ring (n = 8, 12, 16), Torus 4×4 und 8×8 bis 12×12; ein zweites Modell ist nicht gemessen. | Experimente 2 und 3 |
| **Kosten in Platzentscheidungen.** | Keine Sekunden: eine Zeilenentscheidung des Direktsamplers liest im Mittel 24,6 Matrixeinträge (47 Zeilenzustände), ein Heat-bath-Schritt 4 Nachbarn; bei λ = 1 liegt die Zahl der gelesenen Matrixeinträge (ohne das Vorrechnen der Matrixpotenzen) in derselben Größenordnung wie die der Heat-bath-Kette. | Experiment 3 |
| **Grenze 2 207 Zeilenzustände.** | Eine Setzung aus dem Speicher der dichten Matrix (39 MB), keine Messung der Rechenzeit; jenseits der Breite 20 (dicht gespeichert) ist das exakte Ziehen hier nicht gemessen, und das Abtasten ist damit nicht widerlegt. Dünn besetzte Speicherung würde die Grenze verschieben (nicht gemessen). | Experiment 3 |
| **Kopplung aus der Vergangenheit: Heat-bath, gerades n, Rückschau ab 8.** | Für Metropolis (die Standardfunktion ist nicht monoton) und ungerades n nicht neu gerechnet; für n = 16 mit λ = 20 und 100 nicht gerechnet. | `kopplung-vergangenheit-demo` |
| **„Kopplung“ heißt hier zwei Ketten.** | In Monte Carlo heißt „Kopplung“ gemeinsame Zufallszahlen (`gekoppelte-ziehungen-demo`); das ist etwas anderes. | `kopplung-ketten-demo` |
| **Bessere Betriebsentscheidungen durch Abtasten.** | In den gelesenen Quellen nicht belegt und hier nicht untersucht. | Tabellenzeile 17 |
"""

FORMELN = r"""
**Lesehilfe.** Gegeben ist ein Ziel $\pi$ auf einem Raum. Gesucht sind Proben aus $\pi$ oder Mittelwerte unter $\pi$ mit einer Fehlerangabe. Die Seite vergleicht, was ein Direktsampler,
eine Markov-Kette und die Kopplung aus der Vergangenheit je Probe kosten, und prüft, was die Diagnostik sieht.

| Symbol | Bedeutung |
|---|---|
| $n$, $w$ | Plätze des Rings; Breite des Streifens bzw. Torus |
| $\lambda$ | Gewicht je belegtem Platz |
| $\lvert x\rvert$ | Zahl belegter Plätze der Belegung $x$ |
| $Z$ | Zustandssumme $\sum_x \lambda^{\lvert x\rvert}$ |
| $T$ | Transfermatrix, Zeilen-Übergänge des Streifens |
| $\tau_f$ | Schritte je effektiv unabhängiger Probe der Funktion $f$ |
| $t_{\mathrm{mix}}$ | TV-Mischzeit $t_{\mathrm{mix}}(1/4)$ mit $d(t)=\max_x \mathrm{TV}(P^t(x,\cdot),\pi)$ |
| $\hat R$, ESS | Split-R-Dach und effektive Stichprobengröße (Formeln: `posterior-sampling-demo`) |

**Ziel (Hard-Core, ausgedachtes Modell).**

$$\pi(x)=\frac{\lambda^{\lvert x\rvert}}{Z},\qquad x\in\{0,1\}^V,\; x_u x_v=0\ \text{für jede Kante }(u,v).$$

**Direktsampler über die Transfermatrix** (Streifen aus $L$ Zeilen, periodisch). Mit $T[a][b]=[a\ \text{und}\ b\ \text{vertragen sich}]\,\lambda^{\lvert b\rvert}$ gilt $Z=\operatorname{Spur}T^L$.

$$P(r_0=a)=\frac{(T^L)[a][a]}{Z}$$

$$P(r_{j+1}=b\mid r_j=a,\,r_0)=\frac{T[a][b]\,(T^{L-j-1})[b][r_0]}{(T^{L-j})[a][r_0]}$$

Das Produkt dieser Faktoren ist $\lambda^{\lvert x\rvert}/Z$ für jede zulässige Belegung. Kosten: eine Entscheidung je Zeile, kein Einbrennen. Die Zahl der Zeilenzustände ist die Lucas-Zahl $L_w$
(unabhängige Mengen des $w$-Rings).

**Kosten der Kette je effektiv unabhängige Probe** (stationärer Start; das Einbrennen $t_{\mathrm{mix}}$ kommt einmalig hinzu). Mit der Fundamentalmatrix $Z_P=(I-P+\mathbf 1\pi^{\top})^{-1}$ und $\tilde f=f-\mathbb E_\pi f$:

$$\sigma^2=2\,\langle\tilde f,Z_P\tilde f\rangle_\pi-\langle\tilde f,\tilde f\rangle_\pi,\qquad \tau_f=\frac{\sigma^2}{\operatorname{Var}_\pi f}.$$

**Kosten der Kopplung aus der Vergangenheit.** Rückschau $T_k=8\cdot 2^{k-1}$; $S(T)$ ist die Wahrscheinlichkeit, dass die beiden Extremketten nach $T$ Schritten noch verschieden sind.

$$\mathbb E[\text{Kosten}]=T_1+\sum_{k\ge 2}T_k\,S(T_{k-1}),\qquad \text{Platzentscheidungen}=2\times\text{Schritte der Paarkette}.$$

**Falscher Kern.** Wählt man einen Platz gleichverteilt, kehrt die Belegung um und nimmt jeden zulässigen Vorschlag an, ist die Matrix symmetrisch und die Gleichverteilung stationär, nicht $\pi$.
Die exakte Vorprobe prüft

$$\max_x\,\lvert(\pi P)(x)-\pi(x)\rvert .$$

**Die Entscheidungsfunktion ist eine Skizze, kein Modell.** Sie wählt nach Zielart, Zeilenzustandszahl, Dimension, Gradient, Kenntnis der Kovarianz und Gipfelzahl; ihre Schwelle (2 207 Zeilenzustände)
ist eine Setzung aus dem Speicher, die Texte sind Regeln dieser Seite. Die Zahlen der Tabelle sind gemessen oder aus Quellen gelesen und nennen ihren Status.

**Literatur** (nur so zitiert, wie gelesen): Levin, Peres, Wilmer: *Markov Chains and Mixing Times* (Satz 12.4, Beispiele zu monotonen Ketten und zur Kopplung aus der Vergangenheit; als Wiedergabe in diesem Buch);
Vehtari u. a. (2021), arXiv 1903.08008 (R̂ < 1,01, mindestens vier Ketten, ESS ≥ 400; Abschnitt 3.2: die ESS gilt je Funktion); Flegal, Haran, Jones (2008), arXiv math/0703746
(Monte-Carlo-Standardfehler berichten); Talts u. a., arXiv 1804.06788 (Simulation-Based Calibration; nur der Abstract gelesen, nicht gemessen).
"""
