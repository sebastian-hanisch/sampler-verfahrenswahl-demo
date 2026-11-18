# Welcher Sampler, welche Diagnostik? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-sampler-verfahrenswahl-demo.streamlit.app/)** -->

Letzte Demo der Konzepte-Reihe **MCMC: Verteilungen abtasten** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie ist der **Zusammenfluss** der
übrigen Demos der Reihe: kein neues Verfahren, sondern die Entscheidung, **welches Verfahren man nimmt, was eine Probe kostet und wer prüft, dass das Ziel stimmt**, mit Zahlen.
Die Nachbar-Demos [exakt-oder-abtasten-demo](https://github.com/sebastian-hanisch/exakt-oder-abtasten-demo), [kette-gleichgewicht-demo](https://github.com/sebastian-hanisch/kette-gleichgewicht-demo),
[metropolis-kleiner-raum-demo](https://github.com/sebastian-hanisch/metropolis-kleiner-raum-demo), [gibbs-glauber-demo](https://github.com/sebastian-hanisch/gibbs-glauber-demo),
[mischzeit-exakt-demo](https://github.com/sebastian-hanisch/mischzeit-exakt-demo), [kopplung-ketten-demo](https://github.com/sebastian-hanisch/kopplung-ketten-demo),
[kopplung-vergangenheit-demo](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo), [diagnose-taeuscht-demo](https://github.com/sebastian-hanisch/diagnose-taeuscht-demo),
[gipfel-tempering-demo](https://github.com/sebastian-hanisch/gipfel-tempering-demo) und [dimension-hmc-demo](https://github.com/sebastian-hanisch/dimension-hmc-demo) messen je ein Verfahren an
einer Welt mit bekannter Antwort. Diese Demo hebt zwei ihrer Annahmen auf: **das Verfahren ist nicht vorgegeben** (man muss wählen) und **auf dem Ziel, das man wirklich hat, gibt es kein
Orakel**, nur Diagnostik. Anschluss nach außen: Frage 0 von [monte-carlo-verfahrenswahl-demo](https://github.com/sebastian-hanisch/monte-carlo-verfahrenswahl-demo) („Lässt sich die Verteilung nicht
ziehen?“) endet bei MCMC; diese Demo ist die Antwort darauf. Für R̂, ESS, Zufallsschritt und pCN auf stetigen Zielen siehe [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo);
die Formeln stehen dort und werden hier nicht wiederholt.

## Warum dieses Problem

Wer eine Verteilung abtasten will, fragt meist sofort „welcher Sampler?“. Die erste Frage ist eine andere: **Muss man überhaupt abtasten?** Auf einem Hard-Core-Ring und einem Torus der Breite 8
kostet eine exakt gezogene, unabhängige Probe weniger Platzentscheidungen als eine effektiv unabhängige Probe der Heat-bath-Kette, und der Direktsampler braucht weder Einbrennen noch Diagnostik.
Wo man abtasten muss, gibt es kein Orakel, und die Diagnostik prüft nur, ob die Ketten untereinander übereinstimmen. Die Demo führt vier Kernfragen mit je einer Zahl vor:
(1) Muss man abtasten, und womit? (2) Was kostet eine unabhängige Probe, in welcher Einheit? (3) Wie weit reicht das exakte Ziehen? (4) Merkt die Diagnostik, wenn das Ziel falsch ist (Live-Variante)?

## Modell

Alle Aufgaben sind **ausgedacht**: Stellplätze im Ring und auf dem Torus (keine zwei benachbarten belegt, Gewicht λ je belegtem Platz), eine Gaußmischung und eine Ankunftsrate mit erfundenen Zahlen.
Es gibt keine reale Datenquelle. Gemessen wird in **Platzentscheidungen je effektiv unabhängiger Probe**, nicht in Sekunden. Die Tabelle gilt für diese Modelle und diese Parameter; was sie für
ein echtes Ziel heißt, wird nicht behauptet. Alle hier vorgeführten Ziele lassen sich direkt ziehen (das ist der Grund, warum es ein exaktes Orakel gibt); ob Abtasten bessere
Betriebsentscheidungen liefert, ist in den gelesenen Quellen nicht belegt. Der Zufall der Läufe kommt nur aus dem ganzzahligen SplitMix64 (`svw_rng.py`, Vorlage aus dem Website-Repo; `svw_zufall.py` zieht
dieselbe Folge blockweise); nur der Knopf für einen neuen Zufalls-Seed nutzt Python-`random`. Der Zufalls-Seed heißt „Zufalls-Seed“.

**Konventionen.** „Kopplung“ heißt hier die Kopplung zweier Ketten (Kopplungszeit, Kopplung aus der Vergangenheit); in Monte Carlo heißt „Kopplung“ gemeinsame Zufallszahlen
([gekoppelte-ziehungen-demo](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo)): das ist nicht dasselbe. „Gibbs-Sampler“ ist der Heat-bath-Kern, dessen Gleichgewicht das Gibbs-Maß
exp(−βΦ)/Z ist. t_mix ist die TV-Mischzeit t_mix(1/4) mit d(t) = max_x TV(Pᵗ(x, ·), π) (Levin/Peres/Wilmer), nicht d̄. Diagnostik heute: Split-R̂ < 1,01, mindestens vier Ketten, ESS ≥ 400
(Vehtari u. a. 2021); die Schwelle 1,1 ist veraltet.

**Zähleinheiten.** Eine Platzaktualisierung der Kette zählt 1, ein entschiedener Platz des Direktsamplers 1, ein Schritt der Paarkette der Kopplung aus der Vergangenheit 2 (beide Extremketten
aktualisieren je einen Platz), beim Tempering zählen die Aktualisierungen aller Replikas zusammen. Die Zahlen der Koordinationsspiele zählen Aktualisierungen je Betrieb (Betriebe mal diese Zahl
ist eine eigene Folgerung aus der Beschreibung der Einheit). Stetige Ziele zählen Auswertungen von Dichte oder Gradient und lassen sich **nicht** in Platzentscheidungen umrechnen, weil ein Gradient
je nach Modell mehr kostet als eine Dichte. Die App rechnet nie über diese Grenze.

## Methodik

- **Direktsampler über die Transfermatrix** (`svw_direct.py`): T[a][b] = [a und b vertragen sich] λ^|b|, Z = Spur T^L; Zeile 0 aus (T^L)[a][a]/Z, Zeile j+1 aus
  T[a][b](T^(L−j−1))[b][r₀]/(T^(L−j))[a][r₀]. Ein Streifen der Breite w hat so viele Zeilenzustände wie der w-Ring unabhängige Mengen hat (Lucas-Zahl L_w); die Zahl der verträglichen
  Zeilenpaare ist die Spur M^w einer 3×3-Matrix (eigene Herleitung: jeder Platz ist leer, A oder B, gleiche Buchstaben nie benachbart). Die Bruch-Fassung (`fractions`) gibt jeder Belegung ihre
  exakte Wahrscheinlichkeit.
- **Kosten der Kette** (`tools/svw_make_kosten.py`, vorgerechnet in `svw_kosten.json`): τ_f = 1 + 2 Σ ρ_k(f) exakt aus der Fundamentalmatrix Z = (I − P + 1πᵀ)⁻¹ für die Teilchenzahl und den
  Gipfelindikator I (1, wenn mehr gerade als ungerade Plätze belegt sind, 1/2 bei Gleichstand, sonst 0); t_mix(1/4) aus Matrixpotenzen bzw. (n = 16) von der Belegung „alle geraden Plätze“.
- **Kopplung aus der Vergangenheit** (nur Heat-bath, gerades n): erwartete Kosten Σ_k T_k S(T_(k−1)) mit Rückschau 8, 16, 32, … und der Überlebenswahrscheinlichkeit S der Paarkette aus oberer
  und unterer Extremkette, die unter Drehung um zwei Plätze und Spiegelung zusammengefasst ist. Der Satz von Propp/Wilson (die Ausgabe ist π-verteilt) steht nur als Wiedergabe in Levin/Peres/Wilmer;
  hier ist die Ausgabe in einer Simulation gegen π geprüft, nicht bewiesen.
- **Diagnostik** (`svw_diag.py`): Split-R̂ und Bulk-ESS in einer kleinen Fassung nach Vehtari u. a. (2021), Kern richtig (Heat-bath) oder falsch (Annahmeregel vergessen: jeden zulässigen Vorschlag
  annehmen), exakte Vorprobe max |π P − π| auf dem 322-Zustands-Ring.
- **Entscheidungsweg** (`svw_tree.py`): eine reine Funktion `waehle(aufgabe)` mit den Fragen 0 bis 4. Die Schwelle 2 207 Zeilenzustände (Breite 16, dichte Matrix 39 MB) ist eine **Setzung** aus dem
  Speicher, keine Messung der Rechenzeit. Der Formelblock der App nennt die Entscheidungsfunktion ausdrücklich eine Skizze.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für diese ausgedachten Modelle, die Heat-bath-Kette und die Funktionen Teilchenzahl und Gipfelindikator. „Test“ nennt die Datei, die die Zahl prüft; „Messung dieser README“
heißt: steht in einer Datei des Repos, wird aber vom Test nur teilweise geprüft.

**Kernfrage 2: Kosten je effektiv unabhängiger Probe, Ring (exakt aus der Übergangsmatrix, Platzentscheidungen, Heat-bath, stationärer Start; das Einbrennen t_mix(1/4) kommt einmalig dazu).**

| n | λ | t_mix(1/4) | τ Teilchenzahl | τ Gipfelindikator | direkt | Kopplung (Schritte der Paarkette) | Kopplung (Platzentscheidungen) |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 31 | 10,2 | 46,5 | 8 | 142,6 | 285,2 |
| 8 | 5 | 153 | 34,0 | 318,2 | 8 | 756,1 | 1 512 |
| 8 | 20 | 1 403 | 192,1 | 3 554 | 8 | 6 409 | 12 817 |
| 8 | 100 | 29 147 | 1 140 | 81 490 | 8 | 124 355 | 248 711 |
| 12 | 1 | 54 | 15,5 | 66,8 | 12 | 258,3 | 516,6 |
| 12 | 5 | 267 | 50,7 | 450,8 | 12 | 1 371 | 2 743 |
| 12 | 20 | 2 276 | 546,9 | 4 971 | 12 | 11 127 | 22 254 |
| 12 | 100 | 44 647 | 4 724 | 118 883 | 12 | 196 787 | 393 575 |
| 16 | 1 | 79 | 20,9 | 87,0 | 16 | 386,8 | 773,6 |
| 16 | 5 | 402 | 60,6 | 584,4 | 16 | 2 073 | 4 146 |
| 16 | 20 | 3 304 | 872,8 | 6 305 | 16 | nicht gerechnet | – |
| 16 | 100 | 61 220 | 10 855 | 153 504 | 16 | nicht gerechnet | – |

Lesarten (nur für diese 12 bzw. 10 Zellen; Test `test_claims.py::test_kosten_tabelle_und_lesarten` und `test_cost.py`):
(1) **Der Direktsampler kostet in 12 von 12 Zellen weniger als die Kette**: τ der Teilchenzahl liegt zwischen dem 1,27-Fachen (n = 8, λ = 1) und dem 678-Fachen (n = 16, λ = 100) von n,
τ des Gipfelindikators zwischen dem 5,4- und dem 10 186-Fachen; das Einbrennen kommt bei der Kette noch dazu.
(2) **Die Kopplung aus der Vergangenheit kostet das 4,27- bis 5,16-Fache von t_mix(1/4) in Schritten** (10 Zellen, Rückschau ab 8) und **in 10 von 10 gerechneten Zellen mehr Platzentscheidungen als
eine effektiv unabhängige Gipfel-Probe der Kette** (das 3,05- bis 8,90-Fache), gegenüber der Teilchenzahl das 28,0- bis 218,1-Fache. „Unverzerrt“ (jede Probe exakt π-verteilt) heißt nicht „billig“:
der Preis ist die Exaktheit der Einzelprobe. Die Nachbar-Demo kopplung-vergangenheit-demo misst dieselbe Frage in Schritten mit einem anderen Zeilensatz (Rückschau ab 1, 28 Zeilen:
4,02 bis 5,16 t_mix; Rückschau ab 8: 3,27 bis 5,14 t_mix); die Spannen widersprechen sich nicht, sie zählen andere Zellen.
(3) **Die Funktion entscheidet** („die ESS gilt je Funktion“, Vehtari u. a. 2021, Abschnitt 3.2): in derselben Kette bei demselben λ liegt τ des Gipfelindikators um das 4,2- bis 71,5-Fache über
τ der Teilchenzahl (12 Zellen; am größten bei n = 8, λ = 100).

**Torus 4×4 (743 Zustände), exakt.** Der Direktsampler braucht 4 Zeilenentscheidungen = 16 Platzentscheidungen, unabhängig von λ (im Mittel 5,0 verträgliche Nachfolger je Zeile, rund 20 gelesene
Matrixeinträge je Probe).

| λ | τ Teilchenzahl | τ Gipfelindikator (Schachbrett) | t_rel |
|---|---|---|---|
| 1 | 39,5 | 277,4 | 155,7 |
| 2 | 67,7 | 1 236 | 649,4 |
| 5 | 82,2 | 20 532 | 10 319 |
| 10 | 66,1 | 293 580 | 146 857 |
| 20 | 49,8 | 5,8·10⁶ | 2,9·10⁶ |
| 50 | 38,5 | 4,0·10⁸ | 2,0·10⁸ |
| 100 | 34,7 | 1,1·10¹⁰ | 5,7·10⁹ |

τ der Teilchenzahl ist **nicht monoton** in λ (Maximum bei λ = 5; bei großem λ ist die Teilchenzahl fast konstant) und liegt in allen 7 Zellen über den 16 Platzentscheidungen des
Direktsamplers (das 2,2- bis 5,1-Fache). **Dieselbe Kette bei λ = 100: τ = 34,7 für die Teilchenzahl, τ = 1,1·10¹⁰ für den Gipfelindikator.** Die ESS der einen Größe sagt nichts über die der anderen.

**Heat-bath gegen Metropolis (Ring n = 12, alle Starts, t_mix(1/4)).** Verhältnis Heat-bath durch Metropolis: 1,52 / 2,00 / 1,50 / 1,20 / 1,10 / 1,05 / 1,02 / 1,01 für λ = 0,5 / 1 / 2 / 5 / 10 / 20 / 50 / 100
(t_mix Heat-bath 38 / 54 / 93 / 267 / 727 / 2 276 / 11 925 / 44 647, Metropolis 25 / 27 / 62 / 222 / 661 / 2 168 / 11 691 / 44 204). Das Verhältnis liegt nahe am Takt (1 + λ)/max(1, λ), die
größte Abweichung ist 0,02 (bei λ = 0,5). Der Beweis des Takts gehört zu gibbs-glauber-demo.

**Kernfrage 3: Wie weit reicht das exakte Ziehen?** Die Zahl der Zeilenzustände (Lucas-Zahl), der verträglichen Zeilenpaare und der dichte Speicher (8 Byte je Eintrag; Test `test_claims.py::test_reichweite_tabelle`):

| Breite w | Zeilenzustände | verträgliche Paare | Paare je Zeile | dichte Matrix | Zustände des w×w-Torus (λ = 1) |
|---|---|---|---|---|---|
| 4 | 7 | 35 | 5,0 | 0,0004 MB | 743 |
| 8 | 47 | 1 155 | 24,6 | 0,018 MB | 213 256 442 503 |
| 12 | 322 | 39 203 | 121,7 | 0,83 MB | 3,0·10²⁵ |
| 16 | 2 207 | 1 331 715 | 603,4 | 39 MB | nicht berechnet |
| 20 | 15 127 | 4,5·10⁷ | 2 990,6 | 1,8 GB | nicht berechnet |
| 24 | 103 682 | 1,5·10⁹ | 14 822,2 | 86 GB | nicht berechnet |

Die Grenze des exakten Ziehens ist hier eine **Breite, nicht die Zahl der Zustände**: der 8×8-Torus hat 2,1·10¹¹ Zustände, aber nur 47 Zeilenzustände, und 12×12 (322) ist ebenfalls exakt ziehbar.
**Direktproben gegen die exakte Belegungsdichte** (Eigenwerte der Transfermatrix, anderer Weg als die Ziehung; Zufalls-Seeds 20003 bis 20009, Test `test_direct.py`): 8×8 mit 20 000 Proben bei
λ = 1 / 5 / 20 / 100 und 12×12 mit 8 000 Proben bei λ = 1 / 5 / 20 treffen die Dichte in 7 von 7 Zellen innerhalb von 4 Standardfehlern; die größte Abweichung ist 1,16 Standardfehler
(8×8, λ = 5; z. B. λ = 1: 0,22690 ± 0,00026 gegen exakt 0,22701).
**Heat-bath auf dem 8×8-Torus** (Messung dieser README, `svw_iat8.json`, 8 Ketten × 200 000 Schritte, erste 20 000 verworfen, je λ fünf Zufalls-Seeds; τ der Teilchenzahl in
Platzaktualisierungen): λ = 1: Median 137, Spanne 127 bis 146; λ = 5: Median 1 263, Spanne 648 bis 1 361, gegen 64 Platzentscheidungen des Direktsamplers (fünf Zufalls-Seeds sind wenig: mit mehr Seeds kann die Spanne breiter werden, der Median ist die belastbarere Zahl). Der Schätzer (ESS nach Vehtari u. a.,
τ = Ketten × Schritte / ESS) ist am 4×4-Torus gegen das exakte τ geeicht: Verhältnis 0,986 (λ = 1) und 0,985 (λ = 2); der Test prüft diese Eichung (Band 0,85 bis 1,15), die 8×8-Zahlen selbst nicht.
**Gezählt sind Entscheidungen, keine Sekunden.** Bei λ = 1 liegt die Zahl der gelesenen Matrixeinträge (ohne das Vorrechnen der Matrixpotenzen) in derselben Größenordnung wie die der Heat-bath-Kette (eine Zeilenentscheidung des Direktsamplers liest im Mittel 24,6
Matrixeinträge, 8 Zeilen rund 197 Einträge je Probe; ein Heat-bath-Schritt liest 4 Nachbarn); der Vorteil des Direktsamplers ist die Unabhängigkeit von λ, nicht ein Vorsprung bei λ = 1.
Jenseits der Breite 20 (dicht gespeichert) ist das exakte Ziehen hier nicht gemessen und das Abtasten damit nicht widerlegt.

**Kernfrage 4 (Live-Variante): Wer täuscht wen?** Ring n = 12 (322 Zustände), λ = 3, Funktion Teilchenzahl (exakt 4,3395, Standardabweichung 0,8822; Gleichverteilung 3,3168), 4 Ketten × 4 000 Schritte
je Wiederholung, die ersten 1 000 verworfen, Start überstreut (zwei Ketten leer, zwei „alle geraden Plätze“), 200 Wiederholungen, Zufalls-Seed 20004 (Test `test_claims.py::test_diagnose_200_wiederholungen`,
Zählungen als Band ±3):

| Kern | Split-R̂ < 1,01 | Bulk-ESS ≥ 400 | beides | Mittel der Teilchenzahl | Abweichung in Standardfehlern (Mittel) | über 3 Standardfehler |
|---|---|---|---|---|---|---|
| falsch (Annahmeregel vergessen) | 200 von 200 | 200 von 200 | 200 von 200 | 3,318 | 40,3 | 200 von 200 |
| richtig (Heat-bath) | 136 von 200 | 176 von 200 | 129 von 200 | 4,339 | 0,75 | 2 von 200 |

**Der falsche Kern besteht die Diagnostik in jeder Wiederholung und liegt im Mittel 40 Standardfehler neben dem exakten Mittelwert.** „200 von 200“ heißt „kein Gegenfall in 200 Wiederholungen“, nicht
Wahrscheinlichkeit 1. Die **exakte Vorprobe** π P = π auf dem 322-Zustands-Raum findet den falschen Kern: max |π P − π| = 0,0109 (33 % des größten Gewichts 0,0328), TV(π P, π) = 0,1189, Abweichung
der Detailed Balance 0,0018, beim Heat-bath höchstens 1e-12. Das ist kein Vorteil der Diagnostik, sondern ein Argument für kleine exakt lösbare Instanzen des Modells; auf Zielen ohne Orakel gibt es
die Vorprobe nicht. Der Anteil bestandener Wiederholungen ist kein Gütevergleich der Kerne: der falsche Kern mischt schneller (die Gleichverteilung ist leichter zu erreichen) und besteht R̂
häufiger. **Variation, damit der Satz nicht an einem Seed hängt:** anderer Zufalls-Seed (20005): falsch 200 / 200 / 200, 40,3 Standardfehler, richtig 142 / 182 / 136; kürzere Ketten (4 × 2 000, 500
verworfen): falsch beides in 199 von 200 (28,3 Standardfehler), richtig Split-R̂ in 50 von 200, ESS ≥ 400 nur in 3 von 200, beides in 0 von 200; längere Ketten (4 × 8 000, 2 000 verworfen): falsch 200 von 200
(56,9 Standardfehler), richtig 197 von 200; alle Ketten leer gestartet: dieselben Zahlen wie bei überstreutem Start. Ursachen und Rezepte falsch gebauter Kerne gehören zu metropolis-kleiner-raum-demo.

**Entscheidungstabelle (Expander „Woher die Zahlen kommen“, `svw_table.json`).** 18 Zeilen: 14 gemessen, 1 Quelle, 3 nicht gemessen (aus der Tabelle gezählt, `tests/test_table.py`). Von den 18 Zeilen sind
Zeile 1, 3, 5, 15 und 16 hier nachgerechnet, die Zeilen 7 und 8 teilweise (Zeile 7: 61 220; Zeile 8: 4,27 bis 5,16); die Zeilen 2, 6, 9, 10, 11 und 14 sind aus den READMEs der Nachbar-Demos übernommen
und hier nicht neu gerechnet; Zeile 13 stammt aus einer eigenen Messreihe vor dem Bau der Demos (je 200 Wiederholungen), die in keiner Nachbar-Demo steht, und ist hier ebenfalls nicht neu gerechnet. **Die Nachbar-Demos waren beim Bau dieser Demo selbst noch im Bau; die übernommenen Zahlen stammten aus ihren ersten Rechnungen.** Die Zeilen 8, 9, 10, 11 und 14 sind deshalb am 2026-11-18 auf die damals vorliegenden READMEs gesetzt worden (die Zeilen 2 und 6 stimmten schon); die Zeilen 8, 9 und 14 gehören zu Nachbar-Demos, die danach noch korrigiert werden, und sind nach deren Abschluss nochmals mit `tools/svw_check_table.py` zu prüfen. `tools/svw_check_table.py` zeigt den Stand je Zeile („abgeglichen“, „FEHLT“, „nicht abgeglichen“); es sucht die Prüfzahlen als ganze Zahlen und zählt Fundstellen in Zeilen mit „erste Rechnung“ nicht mit, damit eine überholte Zahl nicht als abgeglichen gilt. Zeile 12 (Skalierungen d^(1/3), d^(1/4)) ist eine Quelle (Diffusionsgrenzwerte unter Annahmen), keine Messung.

**Beispiel von Hand (Ring n = 4, λ = 2, Brüche; Test `tests/test_reference.py`).** 7 Belegungen, Z = 17, π(leer) = 1/17, π(Einzelbelegung) = 2/17, π({0, 2}) = 4/17; P(Platz 0 belegt) = 6/17;
t_mix(1/4) = 22; τ der Teilchenzahl 1195/221 = 5,41, τ des Gipfelindikators 53; Kopplung aus der Vergangenheit im Mittel 88,04 Schritte = 176,1 Platzentscheidungen (das 4,00-Fache von t_mix);
Reihenfolge der Kosten je Probe: 4 < 5,4 < 53 < 176. Gamma–Poisson (Prior Gamma(2, 1), 20 Beobachtungen mit Summe 32): Posterior Gamma(34, 21), Mittel 1,61905, Standardabweichung 0,27766.

| Zahl | Test |
|---|---|
| Zeilenzustände, Paare, Speicher, Zustandszahlen der Tabelle (Kernfrage 3) | `test_claims.py::test_reichweite_tabelle` (exakt), `test_direct.py` |
| Kosten-Tabelle Ring und alle Lesarten (Kernfrage 2) | `test_claims.py::test_kosten_tabelle_und_lesarten`, `test_cost.py` (gegen eine frühere Messung, relativ 3e-3; n = 8 und 12 neu gerechnet, relativ 1e-6, t_rel und τ des Gipfelindikators 1e-4) |
| Torus 4×4 | `test_claims.py::test_torus_tabelle`, `test_oracle_sampler_verfahrenswahl.py` (Aufzählung aller 743 Zustände, Spektralweg) |
| Takt | `test_claims.py::test_takt_tabelle` (exakt aus den Mischzeiten), `test_cost.py` |
| Direktproben 7 Zellen, größte Abweichung 1,16 | `test_claims.py::test_direktproben_sieben_zellen` (Band ±0,02), `test_direct.py` (alle 7 unter 4 Standardfehlern) |
| Heat-bath 8×8 (Median, Spannen), Eichung | Messung dieser README (`svw_iat8.json`); Eichung am 4×4-Torus in `test_oracle_sampler_verfahrenswahl.py` und `test_cost.py` |
| Wer täuscht wen: Zählungen, Mittel, Abweichungen, Variation | `test_claims.py::test_diagnose_200_wiederholungen` (Zählungen ±3, Abweichung ±1), `test_diag.py` |
| Vorprobe 0,0109 / 0,1189 / 0,0018, Mittelwerte 4,3395 und 3,3168 | `test_claims.py::test_vorprobe_und_mittelwerte` (Band ±5e-5), `test_diag.py` |
| Beispiel von Hand (Ring n = 4) und Gamma–Poisson | `test_reference.py` (Brüche exakt) |
| 18 / 14 / 1 / 3 der Entscheidungstabelle | `test_table.py` (aus der Tabelle gezählt) |
| Zahl der Tests in dieser README | `readme_test_counts.py` im Website-Repo |

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen aus der ersten Rechnung wurden im Bau mit dem Zufallsgenerator der Demo (SplitMix64) und frisch geschriebenem Code neu gemessen; wo sie sich verschoben, gilt die neue Messung.

- **Heat-bath auf dem 8×8-Torus, λ = 5:** die frühere Einzelmessung nannte τ = 1 069. Mit fünf Zufalls-Seeds streut τ zwischen 648 und 1 361 (Median 1 263); ein einzelner Seed reicht für diese
  Zahl nicht (Faktor 2 zwischen den Seeds), 1 069 liegt innerhalb der Spanne. Bei λ = 1 stimmen beide: Median 137 (Spanne 127 bis 146) gegen die frühere Einzelmessung 145.
- **Kopplung gegenüber der Teilchenzahl:** das 28,0- bis 218,1-Fache (die frühere Angabe „28 bis 217“ war gerundet; der größte Wert ist 218,1).
- **Heat-bath gegen Metropolis:** die Verhältnisse der Mischzeiten sind dem Takt (1 + λ)/max(1, λ) **nahe, nicht gleich** (größte Abweichung 0,02 bei λ = 0,5: 1,52 gegen 1,5).
- **Direktproben:** die größte Abweichung über die 7 Zellen ist 1,16 Standardfehler (frühere Rechnung: 1,8 bei 12×12, λ = 5); andere Zufalls-Seeds, kein Widerspruch.
- **Wer täuscht wen:** richtiger Kern mit Zufalls-Seed 20004: Split-R̂ in 136, ESS in 176, beides in 129 von 200 (frühere Rechnung 140 / 177 / 135); falscher Kern unverändert 200 von 200, 40,3 statt
  40,1 Standardfehler. Neu: die Zählung „200 von 200“ gilt für Ketten ab 4 × 4 000 Schritten; bei 4 × 2 000 (500 verworfen) besteht der falsche Kern in 199 von 200, der richtige in 0 von 200.
- **Reichweite:** `svw_reach.json` (eingefrorene Zahlen) entfällt: Zeilenzustände, Paare und Speicher werden in Millisekunden live gerechnet und im Test gegen eine Aufzählung geprüft.
- **Regler „Weg“ in Kernfrage 2** entfällt: der Balken zeigt alle drei Wege zugleich, damit kein Regler nur eine Hervorhebung ändert.
- **Zeile 9 der Tabelle** nennt für die Gaußmischung auch d = 10 („60 gegen 0“); ob die Nachbar-Demo gipfel-tempering-demo diesen Wert führt, ist beim Abgleich zu prüfen.

## Ehrliche Grenzen

- **Ausgedachte Modelle, keine reale Datenquelle**; die Zahlen gelten für den Hard-Core-Ring (n = 8, 12, 16), den Torus 4×4 und 8×8 bis 12×12, den Heat-bath und die Funktionen Teilchenzahl und
  Gipfelindikator. Ein zweites Modell (etwa der Ising-Kreis) ist **nicht gemessen**; die Übertragung auf andere Modelle wird nicht behauptet.
- **Alle vorgeführten Ziele lassen sich direkt ziehen.** Die Demos dieser Reihe führen ihre Sampler an Zielen vor, die man nicht abtasten müsste, weil nur dort ein exaktes Orakel existiert.
- **Keine Sekunden gemessen**, nur Entscheidungen und gelesene Matrixeinträge. Die Schwelle 2 207 Zeilenzustände ist eine Setzung aus dem Speicher der dichten Matrix; sie wurde nicht auf Streamlit Cloud
  nachgemessen. Dünn besetzte Speicherung würde sie verschieben (nicht gemessen).
- **Kopplung aus der Vergangenheit** nur für den Heat-bath und gerades n, Rückschau 8, 16, 32, …; für n = 16 mit λ = 20 und 100 nicht gerechnet (die Überlebenswahrscheinlichkeit ist nach 40 000 Schritten
  noch nicht unter 1e-15). Metropolis (die Standardfunktion ist nicht monoton) und ungerades n sind nicht neu gerechnet. Für n = 16 gilt t_mix(1/4) vom Start „alle geraden Plätze“; für n ≤ 12
  ist das der schlechteste Start (8 von 8 Zellen).
- **Heat-bath auf 8×8:** nur die Teilchenzahl, nur λ = 1 und 5, je fünf Zufalls-Seeds; τ des Gipfelindikators dort nicht gemessen.
- **Gauß-Ziele und Gaußmischungen** direkt zu ziehen ist Lehrbuchwissen und hier nicht gemessen; eine Schwelle der Dimension, ab der man abtasten muss, setzt die Demo nicht.
- **Zahlen aus den Nachbar-Demos** (Tabellenzeilen 2, 6, 9, 10, 11, 13, 14 und Teile von 7 und 8) sind übernommen und hier nicht neu gerechnet; ihr Abgleich mit den fertigen READMEs steht aus und kann Zahlen ändern.
- **Quellen:** gelesen (Wortlaut) sind Levin/Peres/Wilmer (nur als Wiedergabe der Sätze und Beispiele), Vehtari u. a. 2021, Flegal/Haran/Jones, Roberts/Gelman/Gilks 1997, Roberts/Rosenthal 1998 und
  Beskos u. a. (jeweils Zusammenfassung und Sätze); von Talts u. a. (Simulation-Based Calibration) nur der Abstract, **nicht gemessen**. Nicht gelesen und nicht zitiert: Norris, das Handbook of MCMC
  (Buch), Robert/Casella, Sinclair/Jerrum 1989, Besag 1974, Gelman/Rubin 1992, Propp/Wilson im Original (nur nach der Wiedergabe in Levin/Peres/Wilmer).
- **Neuheit:** nichts hier ist neu. Entscheidungswege für MCMC stehen in jedem Lehrbuch, der Direktsampler über die Transfermatrix ist das bekannte Vorwärts-Rückwärts-Verfahren; neu ist nur die
  Vorführbarkeit mit einer Zähleinheit und exakten Orakeln.
- **Bessere Betriebsentscheidungen durch Abtasten:** in den gelesenen Quellen nicht belegt und hier nicht untersucht. Plattform: Zahlen unter Windows, Python 3.14; Ganzzahlen und Brüche sind plattformfest, exakte τ und t_rel stimmen
  zwischen Windows und Linux auf 1e-6 (bei sehr kleiner Spektrallücke, Torus 4×4 mit λ = 100, auf 1e-5), Zufallsanteile stehen als Band.

## Verifikation

- **Kern:** Direktsampler in Brüchen gegen λ^|x|/Z für jede der 2ⁿ Bitfolgen (Ring n = 4 bis 10) und jede der 743 Belegungen des 4×4-Torus; Ziehungen gegen die exakte Verteilung im Band; exakte Dichte
  gegen Aufzählung (`tests/test_direct.py`).
- **Orakel:** `tests/test_oracle_sampler_verfahrenswahl.py` prüft τ auf drei Wegen (Fundamentalmatrix, Spektralzerlegung, Summe der Autokorrelationen), zwei verschieden gebaute Heat-bath-Matrizen
  gegeneinander und gegen Brüche, t_mix gegen Brüche und schrittweise Iteration, die erwarteten Kosten der Kopplung gegen Brüche, gegen die volle Paarkette ohne Symmetriezusammenfassung und gegen
  eine Simulation (Band 4 Standardfehler), den Torus 4×4 gegen eine Aufzählung von Hand.
- **Handbeispiel:** `tests/test_reference.py` (Ring n = 4, Brüche). **Diagnostik:** `tests/test_diag.py` (R̂ und ESS gegen eine einfache Umsetzung und scipy; Kerne; Vorprobe).
- **Entscheidungsweg und Tabelle:** `tests/test_tree.py`, `tests/test_table.py`; **Verdikte** als reine Funktionen: `tests/test_verdicts.py`; je Preset die erwartete Verdikt-Zeile: `tests/test_presets.py`.
- **Eingefrorene Kosten:** `tests/test_cost.py` (Vergleich mit einer früheren Messung, Neurechnung), `tools/svw_make_kosten.py --check`.
- **Zufall:** `tests/test_rng.py` (Referenzfolge des SplitMix64, Block gleich Skalar).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler, Breiten um die Grenzen), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines
  Reglers hebt die Markierung auf), Footer und Laufzeit-Importe.
- **Randfälle:** Breite 3, Breite 2 (Meldung), λ = 0 und sehr großes λ, eine Probe, Zufalls-Seed 0, Ring n = 2 und 3, genau an der Grenze 2 207 / 3 571, Aufgabe ohne gültige Zielart (Meldung).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `svw_constants.py` | Regler-Grenzen, Raster, Farben, Presets |
| `svw_cost.py`, `svw_kosten.json`, `svw_iat8.json` | vorgerechnete Kosten, Einheiten, Zählungen |
| `svw_diag.py` | Kerne, Split-R̂, Bulk-ESS, exakte Vorprobe, Wiederholungsexperiment |
| `svw_direct.py` | Transfermatrix, Direktsampler, Zeilenzustände, Reichweite |
| `svw_evaluation.py` | Verdikte als reine Funktionen |
| `svw_format.py` | Zahlenformat (Dezimalkomma, Tausendertrenner) |
| `svw_layout.py` | Fester Kopf (Stylesheet) |
| `svw_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `svw_reference.py` | Exakte Bruchrechnung für den kleinen Ring (Handbeispiel, Orakel) |
| `svw_rng.py`, `svw_zufall.py` | SplitMix64 (Vorlage) und derselbe Strom blockweise |
| `svw_table.py`, `svw_table.json` | Entscheidungstabelle mit Herkunft und Status |
| `svw_texts.py` | Texte der App (Einleitung, Grenzen, Formeln) |
| `svw_tree.py` | Entscheidungsweg `waehle` |
| `svw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/svw_make_kosten.py` | rechnet `svw_kosten.json` und `svw_iat8.json`; `--check` rechnet neu und vergleicht |
| `tools/svw_check_table.py` | gleicht die Tabellenzahlen mit den READMEs der Nachbar-Demos ab |
| `tests/` | Handbeispiel, Orakel, Direktsampler, Kosten, Diagnostik, Entscheidungsweg, Tabelle, Claims, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Ein zweites Modell** (Ising-Kreis) als Gegenprobe der Übertragung: nicht gemessen, nicht behauptet.
- **Sekunden und Rechenzeit** als Einheit; **ein PDF-Export** (die Seite ist ein Vergleich von Zahlen, kein Szenario, das man ablegen müsste).
- **Kopplung aus der Vergangenheit** für Metropolis und ungerades n, für n = 16 bei λ = 20 und 100.
- **Direktsampler jenseits der Breite 16 live** (nur die Größe der Matrix ist gezählt) und dünn besetzte Matrizen.
- **Simulation-Based Calibration und die Kopplungsdiagnose** (Jacob u. a., Biswas u. a.) als Diagnostik: nur als Verweis, nicht gemessen.
- **Eine Entscheidungsschwelle für die Dimension** stetiger Ziele und das direkte Ziehen von Gauß-Zielen (Lehrbuch).
- **Aussagen über echte Ziele oder Betriebsentscheidungen.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Suite hat 205 Tests. Kosten neu rechnen: `python tools/svw_make_kosten.py --check` (n = 8, 12, Torus 4×4, Takt; mit `--lang` auch n = 16, rund 5 Minuten).

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html).
