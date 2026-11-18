"""Welcher Sampler, welche Diagnostik? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Muss man überhaupt abtasten, was kostet eine effektiv unabhängige Probe (in einer Einheit), wie weit reicht das exakte Ziehen, und merkt die Diagnostik,
wenn das Ziel falsch ist? Alle Modelle sind ausgedacht (Hard-Core-Ring und -Torus, Gamma–Poisson, Gaußmischung).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import svw_constants as C
import svw_cost as K
import svw_diag as G
import svw_direct as D
import svw_evaluation as ev
import svw_table as TB
import svw_texts as T
from svw_format import de, ganz, gross, mb, zehnerpotenz
from svw_layout import dunkles_design, kopf_css
from svw_presets import SETTING_SPECS, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from svw_tree import Aufgabe, waehle
from svw_visualization import build_beispiel, build_diagnose, build_kosten, build_reichweite
from svw_zufall import Strom

st.set_page_config(page_title="Welcher Sampler, welche Diagnostik? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _direkt_probe(w: int, lam: int, proben: int, seed: int) -> dict:
    return D.probe_gegen_exakt(w, lam, proben, Strom(seed))


@st.cache_data(show_spinner=False)
def _diagnose(kern: str, wiederholungen: int, seed: int) -> dict:
    return G.pruefe_wiederholungen(C.DIAG_N, C.DIAG_LAM, kern, wiederholungen, C.DIAG_SCHRITTE, C.DIAG_VERWORFEN, seed)


@st.cache_data(show_spinner=False)
def _vorprobe(kern: str) -> dict:
    return G.vorprobe(G.Ring(C.DIAG_N), C.DIAG_LAM, kern)


@st.cache_data(show_spinner=False)
def _reichweite_tabelle() -> dict:
    return {w: D.reichweite(w) for w in C.TABELLE_BREITEN}


def _tabelle_zeilen(nummern=None) -> list[dict]:
    zeilen = [z for z in TB.zeilen() if nummern is None or z["nr"] in nummern]
    return [{"Nr": z["nr"], "Frage → Antwort": f"{z['frage']} → {z['antwort']}", "Zahl mit Voraussetzung": z["zahl"], "Herkunft": z["herkunft"], "Status": z["status"],
             "Stand in dieser Demo": z["stand"]} for z in zeilen]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in svw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Welcher Sampler, welche Diagnostik?")
st.markdown(T.EINLEITUNG)

with st.expander("So funktioniert die Verfahrenswahl", expanded=True):
    st.markdown(T.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Aufgabe (Kernfrage 1)**")
    ziel = st.selectbox("Zielart", C.ZIELE, key="ziel", format_func=C.ZIEL_NAMEN.get, help="Was für ein Ziel π ist gegeben? Die Wahl bestimmt den Weg.")
    vorab = waehle(Aufgabe(ziel=st.session_state["ziel"], breite=int(st.session_state["breite"])))
    direkt_gewaehlt = vorab.weg == "direkt"
    st.slider("Breite w des Torus", *bounds("breite"), key="breite", help="Gilt für die Zielart Torus (Kernfrage 1) und für die Reichweite (Experiment 3).")
    st.slider("Dimension d", *bounds("dim"), key="dim", disabled=ziel not in ("gauss", "stetig"), help="Nur für stetige Ziele und Gauß-Ziele.")
    st.toggle("Gradient verfügbar", key="gradient", disabled=ziel != "stetig", help="Nur für stetige Ziele ohne geschlossene Form.")
    st.toggle("Kovarianz bekannt", key="kovarianz", disabled=ziel != "stetig", help="Nur für stetige Ziele ohne geschlossene Form.")
    st.radio("Gipfel", C.GIPFEL, key="gipfel", format_func=C.GIPFEL_NAMEN.get, disabled=direkt_gewaehlt,
             help="Nur wirksam, wenn abgetastet werden muss; direkt gezogene Ziele brauchen keine Gipfelfrage.")
    st.toggle("Kopplung aus der Vergangenheit möglich", key="kopplung_ok", disabled=ziel != "torus" or direkt_gewaehlt,
              help="Nur für endliche Ziele, bei denen abgetastet werden muss und die Kette monoton ist.")
    st.markdown("**Zufall**")
    st.number_input("Zufalls-Seed", *bounds("seed"), key="seed", step=1, help="Gilt für Experiment 3 und 4.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

aufgabe = Aufgabe(ziel=st.session_state["ziel"], breite=int(st.session_state["breite"]), dim=int(st.session_state["dim"]), gradient=bool(st.session_state["gradient"]),
                  kovarianz=bool(st.session_state["kovarianz"]), gipfel=st.session_state["gipfel"], kopplung_ok=bool(st.session_state["kopplung_ok"]))
weg = waehle(aufgabe)
seed = int(st.session_state["seed"])

# --- Kernfrage 1 --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Kernfrage 1: Muss man überhaupt abtasten, und womit?")
m1, m2 = st.columns(2)
m1.metric("Weg", "direkt ziehen" if weg.weg == "direkt" else "abtasten")
m2.metric("Zeilenzustände des Streifens", "–" if weg.zeilenzustaende is None else ganz(weg.zeilenzustaende), delta=f"Grenze {ganz(C.ZEILEN_MAX)}", delta_color="off")
st.markdown("**Der durchlaufene Weg** (▶ = erreicht, · = entfällt):")
pfad = ["| | Frage | Antwort |", "|---|---|---|"]
for f in weg.fragen:
    pfad.append(f"| {'▶' if f.erreicht else '·'} | **{f.nr}. {f.text}** | {f.antwort if f.erreicht else 'entfällt'} |")
st.markdown("\n".join(pfad))
kind, text = ev.verdict_weg(aufgabe, weg, K.tau_torus8())
getattr(st, kind)(text)
if weg.diagnose:
    st.markdown("**Diagnose:** " + "; ".join(weg.diagnose) + ".")
if weg.nachbarn:
    st.markdown("**Mehr dazu:** " + ", ".join(f"[{n}]({T.GITHUB}{n})" for n in weg.nachbarn) + ".")
st.markdown("**Zahlen zu dieser Aufgabe, mit Herkunft und Status** (die ganze Tabelle steht im Expander „Woher die Zahlen kommen“):")
st.dataframe(_tabelle_zeilen(weg.tabellenzeilen), hide_index=True, width="stretch")

st.markdown("---")

# --- Weitere Kernfragen ---------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Weitere Kernfragen")

# Experiment 2 ---------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Was kostet eine unabhängige Probe, in welcher Einheit?")
st.caption("Einheit: Platzentscheidungen je effektiv unabhängiger Probe; Zahlen vorgerechnet (exakt aus der Übergangsmatrix, Heat-bath, stationärer Start, Einbrennen einmalig dazu).")
c1, c2, c3, c4 = st.columns(4)
modell = c1.radio("Modell", C.KOST_MODELLE, key="kost_modell")
c2.select_slider("Plätze n des Rings", C.KOST_N, key="kost_n", disabled=modell != "Ring")
c3.select_slider("λ", C.KOST_LAM, key="kost_lam", help="Gewicht je belegtem Platz. Der Ring ist nur für λ = 1, 5, 20, 100 gerechnet.")
c4.radio("Funktion der Kette", C.KOST_FUNKTIONEN, key="kost_f", help="Die Kette ist dieselbe; die Kosten hängen an der Funktion, deren Mittelwert man schätzt.")
b = K.balken(modell, int(st.session_state["kost_n"]), int(st.session_state["kost_lam"]), st.session_state["kost_f"])
if b is not None:
    st.plotly_chart(build_kosten(b), width="stretch", key="kosten_chart")
    if b["kopplung"] is None and modell == "Ring":
        st.caption("Kopplung aus der Vergangenheit: für diese Zelle nicht gerechnet (n = 16 mit λ = 20 und 100).")
    elif modell != "Ring":
        st.caption("Kopplung aus der Vergangenheit: für den Torus nicht gerechnet.")
kind, text = ev.verdict_kosten(b, K.ring_zaehlungen(), K.torus_zaehlungen())
getattr(st, kind)(text)
if modell == "Ring":
    zeilen = [{"n": z["n"], "λ": ganz(z["lam"]), "t_mix(1/4) einmalig": ganz(z["t_mix"]), "τ Teilchenzahl": gross(z["tau_teilchen"], 1), "τ Gipfelindikator": gross(z["tau_gipfel"], 1),
               "direkt": z["direkt"], "Kopplung (Schritte der Paarkette)": "nicht gerechnet" if z["kopplung_schritte"] is None else gross(z["kopplung_schritte"], 1),
               "Kopplung (Platzentscheidungen)": "nicht gerechnet" if z["kopplung_platz"] is None else gross(z["kopplung_platz"], 1)} for z in K.ring_zellen()]
else:
    zeilen = [{"λ": ganz(z["lam"]), "τ Teilchenzahl": gross(z["tau_teilchen"], 1), "τ Gipfelindikator": gross(z["tau_gipfel"], 1), "direkt": z["direkt"],
               "t_rel": gross(z["t_rel"], 1)} for z in K.laden()["torus44"]]
st.dataframe(zeilen, hide_index=True, width="stretch")
st.caption("**Falle: Die ESS gilt je Funktion** (Vehtari u. a. 2021, Abschnitt 3.2): Eine effektive Stichprobengröße ohne Funktion und Zähleinheit sagt nichts. "
           "Das Einbrennen kommt bei der Kette einmalig dazu, die Kopplung aus der Vergangenheit zählt 2 Platzentscheidungen je Schritt der Paarkette.")
st.markdown("**Heat-bath gegen Metropolis (Ring n = 12):** das Verhältnis der Mischzeiten t_mix(1/4) liegt nahe am Takt (1 + λ)/max(1, λ).")
st.dataframe([{"λ": de(z["lam"], 1 if z["lam"] < 1 else 0), "t_mix Heat-bath": ganz(z["t_mix_heatbath"]), "t_mix Metropolis": ganz(z["t_mix_metropolis"]),
               "Verhältnis": de(z["verhaeltnis"], 2), "Takt (1 + λ)/max(1, λ)": de(z["takt"], 2)} for z in K.takt_zellen()], hide_index=True, width="stretch")

# Experiment 3 ---------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Wie weit reicht das exakte Ziehen?")
st.caption(f"Die Breite w des Torus steht in der Seitenleiste (derzeit {aufgabe.breite}); gezogen wird live bis Breite {C.LIVE_BREITE_MAX}, darüber zeigt die Tabelle nur die Größe der Matrix.")
d1, d2 = st.columns(2)
d1.slider("Zahl der Direktproben B", *bounds("proben"), key="proben")
d2.select_slider("λ der Direktproben", C.LAM_DICHTE, key="lam_dichte")
reach = D.reichweite(aufgabe.breite)
probe = None
if aufgabe.breite <= C.LIVE_BREITE_MAX:
    probe = _direkt_probe(aufgabe.breite, int(st.session_state["lam_dichte"]), int(st.session_state["proben"]), seed)
    p1, p2, p3 = st.columns(3)
    p1.metric("Belegungsdichte exakt", de(probe["rho_exakt"], 5))
    p2.metric("Mittel der Direktproben ± SF", f"{de(probe['rho_probe'], 5)} ± {de(probe['sf'], 5)}")
    p3.metric("Abweichung in Standardfehlern", de(probe["z"], 2), delta=f"{aufgabe.breite * aufgabe.breite} Platzentscheidungen je Probe", delta_color="off")
    st.plotly_chart(build_beispiel(probe["gitter"]), width="stretch", key="beispiel_chart")
    st.caption("Eine der Direktproben (dunkel = belegt).")
tabelle = _reichweite_tabelle()
zeilen_je_breite = {w: r["zeilen"] for w, r in tabelle.items()}
zeilen_je_breite[aufgabe.breite] = reach["zeilen"]
st.plotly_chart(build_reichweite(zeilen_je_breite, aufgabe.breite, C.ZEILEN_MAX), width="stretch", key="reichweite_chart")
kind, text = ev.verdict_reichweite(aufgabe.breite, reach, probe)
getattr(st, kind)(text)
st.dataframe([{"Breite w": w, "Zeilenzustände": ganz(r["zeilen"]), "verträgliche Zeilenpaare": gross(r["paare"], 1, ab=1e12), "Paare je Zeile": de(r["paare_je_zeile"], 1),
               "dichte Matrix": mb(r["mb"]), "Zustände des w×w-Torus (λ = 1)": "nicht berechnet" if r["zustaende"] is None else (ganz(r["zustaende"]) if isinstance(r["zustaende"], int) else zehnerpotenz(r["zustaende"], 2))}
              for w, r in tabelle.items()], hide_index=True, width="stretch")
i8 = K.iat8()
z1, z5 = i8["torus8x8"][0], i8["torus8x8"][-1]
st.markdown(f"**Heat-bath auf dem 8×8-Torus (vorgerechnet, Median über {len(z1['zufalls_seeds'])} Zufalls-Seeds, je {z1['ketten']} Ketten × {ganz(z1['schritte'])} Schritte):** "
            f"τ der Teilchenzahl {ganz(z1['tau'])} Platzaktualisierungen bei λ = {ev.lam_text(z1['lam'])} (Spanne {ganz(z1['tau_min'])} bis {ganz(z1['tau_max'])}) und "
            f"{ganz(z5['tau'])} bei λ = {ev.lam_text(z5['lam'])} (Spanne {ganz(z5['tau_min'])} bis {ganz(z5['tau_max'])}) gegen {z1['direkt']} Platzentscheidungen des Direktsamplers. "
            "Bei λ = 1 liegt die Zahl der gelesenen Matrixeinträge (ohne das Vorrechnen der Matrixpotenzen) in derselben Größenordnung wie die der Heat-bath-Kette; der Vorteil des Direktsamplers ist die Unabhängigkeit von λ.")

# Experiment 4 ---------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4 (Live-Variante): Wer täuscht wen?")
st.caption(f"Hard-Core-Ring n = {C.DIAG_N}, λ = {de(C.DIAG_LAM, 0)}; je Wiederholung {C.DIAG_KETTEN} Ketten × {ganz(C.DIAG_SCHRITTE)} Schritte, die ersten {ganz(C.DIAG_VERWORFEN)} verworfen, "
           "Start überstreut (zwei Ketten leer, zwei „alle geraden Plätze“); Funktion: Teilchenzahl. Ursachen falsch gebauter Kerne: `metropolis-kleiner-raum-demo`.")
e1, e2 = st.columns(2)
e1.slider("Wiederholungen", *bounds("diag_rep"), key="diag_rep")
kern = e2.radio("Kern", ("richtig", "falsch"), key="diag_kern", format_func=C.KERN_NAMEN.get, horizontal=True,
                help="„falsch“ vergisst die Annahmeregel: jeder zulässige Vorschlag wird angenommen.")
with st.spinner("Ketten laufen …"):
    diag = _diagnose(kern, int(st.session_state["diag_rep"]), seed)
vor, vor_richtig = _vorprobe(kern), _vorprobe("richtig")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Split-R̂ < 1,01", f"{diag['n_rhat']} von {diag['wiederholungen']}")
q2.metric("Bulk-ESS ≥ 400", f"{diag['n_ess']} von {diag['wiederholungen']}")
q3.metric("beides", f"{diag['n_beides']} von {diag['wiederholungen']}")
q4.metric("Mittel (exakt " + de(diag["exakt"], 3) + ")", de(diag["mittel_mittel"], 3), delta=f"{de(diag['mittel_abs_z'], 1)} Standardfehler daneben", delta_color="off")
st.plotly_chart(build_diagnose(diag, kern), width="stretch", key="diagnose_chart")
kind, text = ev.verdict_diagnose(kern, diag, vor, vor_richtig)
getattr(st, kind)(text)
st.caption("Die exakte Vorprobe π P = π braucht eine kleine Instanz mit bekannter Antwort; auf Zielen ohne Orakel gibt es sie nicht. "
           "Simulation-Based Calibration (Talts u. a.) wäre ein Weg ohne Orakel; sie ist hier nicht gemessen.")

st.markdown("---")

# --- Einordnung -----------------------------------------------------------------------------------------------------------------------------

with st.expander("Woher die Zahlen kommen: Entscheidungstabelle, Einheiten und Nachbar-Demos"):
    kind, text = ev.verdict_tabelle(TB.belegungssatz())
    getattr(st, kind)(text)
    st.markdown("Status: **gemessen** heißt in diesem Modell bei diesen Parametern, in der genannten Demo; **Quelle** heißt Wortlaut gelesen, nicht gemessen. Die Spalte „Stand in dieser "
                "Demo“ sagt, ob die Zahl hier nachgerechnet ist, so in der README der Nachbar-Demo stand (Stand beim Abgleich, siehe README) oder aus einer eigenen Messreihe stammt, die in keiner Nachbar-Demo steht.")
    st.dataframe(_tabelle_zeilen(), hide_index=True, width="stretch")
    st.markdown("**Umrechnung der Einheiten** (nur innerhalb einer Familie: diskret in Platzentscheidungen, stetig in Auswertungen):")
    st.dataframe([{"Quelle der Zahl": a, "Einheit": e, "in Platzentscheidungen": u} for a, e, u in K.EINHEITEN], hide_index=True, width="stretch")
    st.markdown("**Nachbar-Demos:** " + "; ".join(f"[{n}]({T.GITHUB}{n}) ({t})" for n, t in T.NACHBARN) + ".")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.GRENZEN_TABELLE)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.FORMELN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html)."
)

sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})
