"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Aufgabe (Seitenleiste) ---------------------------------------------------------------------------------------------------------------
ZIELE = ("konjugat", "gauss", "ring", "torus", "stetig")
ZIEL_NAMEN = {
    "konjugat": "Konjugater Prior (Gamma–Poisson)",
    "gauss": "Gauß-Ziel oder Gaußmischung",
    "ring": "Endlich: Hard-Core-Ring",
    "torus": "Endlich: Hard-Core-Torus (Breite w)",
    "stetig": "Stetiger Posterior ohne geschlossene Form (Dimension d)",
}
GIPFEL = ("ein", "mehrere", "unbekannt")
GIPFEL_NAMEN = {"ein": "ein Gipfel (Annahme)", "mehrere": "mehrere bekannte Gipfel", "unbekannt": "Gipfelzahl unbekannt"}
BREITE_MIN, BREITE_MAX, DEFAULT_BREITE = 3, 40, 8
DIM_MIN, DIM_MAX, DEFAULT_DIM = 1, 100, 10
ZEILEN_MAX = 2207          # Grenze der Zeilenzustände für das exakte Ziehen: Lucas-Zahl L_16, dichte Matrix 39 MB (Setzung, keine Messung der Rechenzeit)
SEED_MAX, DEFAULT_SEED = 999999, 20004

# --- Experiment 2: Kosten -----------------------------------------------------------------------------------------------------------------
KOST_MODELLE = ("Ring", "Torus 4×4")
KOST_N = (8, 12, 16)
KOST_LAM = (1, 2, 5, 10, 20, 50, 100)          # Raster für beide Modelle; der Ring ist nur für 1, 5, 20, 100 gerechnet
KOST_FUNKTIONEN = ("Teilchenzahl", "Gipfelindikator")
DEFAULT_KOST_N, DEFAULT_KOST_LAM = 16, 5

# --- Experiment 3: Reichweite -------------------------------------------------------------------------------------------------------------
PROBEN_MIN, PROBEN_MAX, DEFAULT_PROBEN = 20, 500, 200
LIVE_BREITE_MAX = 12       # größte Breite, für die die App live zieht (322 Zeilenzustände)
LAM_DICHTE = (1, 2, 5, 10, 20, 50, 100)
DEFAULT_LAM_DICHTE = 1
TABELLE_BREITEN = (4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 36, 40)

# --- Experiment 4: falscher Kern (Live-Variante) -------------------------------------------------------------------------------------------
DIAG_N, DIAG_LAM = 12, 3.0       # Hard-Core-Ring n = 12 (322 Zustände), λ = 3
DIAG_KETTEN, DIAG_SCHRITTE, DIAG_VERWORFEN = 4, 4000, 1000
REP_MIN, REP_MAX, DEFAULT_REP = 10, 200, 50
KERN_NAMEN = {"richtig": "richtig (Heat-bath)", "falsch": "falsch (Annahmeregel vergessen)"}
RHAT_SCHWELLE, ESS_SCHWELLE = 1.01, 400

COLOR_DIREKT = "#2F6B65"
COLOR_KETTE = "#4c78a8"
COLOR_KOPPLUNG = "#b07aa1"
COLOR_EINBRENNEN = "rgba(150,150,150,0.55)"
COLOR_GRENZE = "#e45756"
COLOR_FALSCH = "#e45756"
COLOR_RICHTIG = "#2F6B65"

# --- Presets ------------------------------------------------------------------------------------------------------------------------------
# Ein Preset nennt nur die Regler, die es ändert; alle übrigen stehen auf ihrem Standardwert (der erste Preset entspricht den Standardwerten).
PRESETS = {
    "Ring n = 16, λ = 5: direkt oder Kette?": {},
    "Gamma–Poisson: kein Sampler": {"ziel": "konjugat"},
    "Torus 8×8: noch exakt": {"ziel": "torus", "breite": 8},
    "Torus 24×24 kalt: abtasten": {"ziel": "torus", "breite": 24, "gipfel": "mehrere"},
    "Posterior in 48 Dimensionen": {"ziel": "stetig", "dim": 48, "gipfel": "ein"},
    "HMC mit Gradient (20-d)": {"ziel": "stetig", "dim": 20, "gradient": True, "gipfel": "ein"},
    "Gipfel unbekannt": {"ziel": "torus", "breite": 24, "kopplung_ok": True, "gipfel": "unbekannt"},
    "Falscher Kern (Annahmeregel vergessen)": {"diag_kern": "falsch"},
}
PRESET_HELP = {
    "Ring n = 16, λ = 5: direkt oder Kette?": "Hard-Core-Ring mit 16 Plätzen, λ = 5: Transfermatrix gegen Heat-bath-Kette (Voreinstellung).",
    "Gamma–Poisson: kein Sampler": "Konjugater Prior: der Posterior steht in geschlossener Form da, ziehen oder Mittelwert ablesen genügt.",
    "Torus 8×8: noch exakt": "2,1·10¹¹ Zustände, aber nur 47 Zeilenzustände: die Transfermatrix zieht exakt.",
    "Torus 24×24 kalt: abtasten": "103 682 Zeilenzustände: über der Grenze des exakten Ziehens, kaltes λ, mehrere Gipfel.",
    "Posterior in 48 Dimensionen": "Stetiges Ziel ohne Gradient und ohne bekannte Kovarianz: Zufallsschritt-Metropolis.",
    "HMC mit Gradient (20-d)": "Stetiges Ziel mit Gradient in 20 Dimensionen: HMC, aber gegen den Zufallsschritt mit Kovarianz nachprüfen.",
    "Gipfel unbekannt": "Große endliche Aufgabe, Kopplung aus der Vergangenheit möglich, Zahl der Gipfel unbekannt.",
    "Falscher Kern (Annahmeregel vergessen)": "Live-Lauf: ein Kern ohne Annahmeregel besteht Split-R̂ und ESS, liegt aber neben dem exakten Mittel.",
}
