"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import svw_constants as C
from svw_format import gross


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_kosten(b: dict):
    """Balken der Kosten je effektiv unabhängiger Probe (Platzentscheidungen, logarithmische Achse); der Balken der Kette gilt für die gewählte Funktion."""
    namen = ["Direktsampler", f"Heat-bath ({b['funktion']})", "Kopplung aus der Vergangenheit"]
    werte = [b["direkt"], b["kette"], b["kopplung"]]
    farben = [C.COLOR_DIREKT, C.COLOR_KETTE, C.COLOR_KOPPLUNG]
    xs, ys, fs, texte = [], [], [], []
    for name, wert, farbe in zip(namen, werte, farben):
        if wert is None:
            continue
        xs.append(wert)
        ys.append(name)
        fs.append(farbe)
        texte.append(gross(wert, 1))
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h", marker_color=fs, text=texte, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.4g} Platzentscheidungen<extra></extra>", showlegend=False))
    if b["einbrennen"] is not None:
        fig.add_trace(go.Scatter(x=[b["einbrennen"]], y=[namen[1]], mode="markers", marker=dict(symbol="diamond", size=13, color=C.COLOR_EINBRENNEN, line=dict(width=1, color="gray")),
                                 name="Einbrennen t_mix(1/4), einmalig", hovertemplate="Einbrennen: %{x} Schritte<extra></extra>"))
    fig.update_xaxes(type="log", title_text="Platzentscheidungen je effektiv unabhängiger Probe (logarithmisch)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.35)
    return _base(fig, 300)


def build_reichweite(zeilen_je_breite: dict, breite: int, grenze: int):
    """Zeilenzustände über der Breite (logarithmische Achse) mit der Grenze des exakten Ziehens; die gewählte Breite ist markiert."""
    ws = sorted(zeilen_je_breite)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ws, y=[zeilen_je_breite[w] for w in ws], mode="lines+markers", line=dict(color=C.COLOR_KETTE, width=3), name="Zeilenzustände",
                             hovertemplate="Breite %{x}: %{y:,} Zeilenzustände<extra></extra>"))
    fig.add_hline(y=grenze, line=dict(color=C.COLOR_GRENZE, dash="dash"), annotation_text=f"Grenze {grenze}", annotation_position="top left")
    if breite in zeilen_je_breite:
        fig.add_trace(go.Scatter(x=[breite], y=[zeilen_je_breite[breite]], mode="markers", marker=dict(size=14, color=C.COLOR_DIREKT, symbol="circle-open", line=dict(width=3)),
                                 name="gewählte Breite", hoverinfo="skip"))
    fig.update_xaxes(title_text="Breite w des Streifens")
    fig.update_yaxes(type="log", title_text="Zeilenzustände (Lucas-Zahl)")
    return _base(fig, 320)


def build_beispiel(gitter: np.ndarray):
    """Eine Direktprobe als Gitter (dunkel = belegt)."""
    fig = go.Figure(go.Heatmap(z=gitter, colorscale=[[0, "rgba(150,150,150,0.25)"], [1, C.COLOR_DIREKT]], showscale=False, xgap=1, ygap=1,
                               hovertemplate="Zeile %{y}, Spalte %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 300, legend=False)


def build_diagnose(d: dict, kern: str):
    """Mittel der Teilchenzahl je Wiederholung (Histogramm) und der exakte Mittelwert als senkrechter Strich."""
    farbe = C.COLOR_FALSCH if kern == "falsch" else C.COLOR_RICHTIG
    fig = go.Figure(go.Histogram(x=d["mittel"], marker_color=farbe, nbinsx=30, name="Mittel je Wiederholung", hovertemplate="Mittel %{x:.2f}: %{y} Wiederholungen<extra></extra>"))
    fig.add_vline(x=d["exakt"], line=dict(color="black", dash="dash"), annotation_text=f"exakt {d['exakt']:.3f}".replace(".", ","), annotation_position="top")
    fig.update_xaxes(title_text="Mittel der Teilchenzahl (vier Ketten, nach dem Einbrennen)")
    fig.update_yaxes(title_text="Wiederholungen")
    return _base(fig, 300, legend=False)
