"""Zahlenformat für Texte: Dezimalkomma, Leerzeichen als Tausendertrenner, Zehnerpotenzen mit hochgestellten Ziffern."""

import math

_HOCH = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def de(x: float, stellen: int = 1) -> str:
    """Dezimalkomma und Tausendertrenner (Leerzeichen): de(2073, 0) = '2 073', de(60.56, 1) = '60,6'."""
    text = f"{abs(x):,.{stellen}f}".replace(",", " ").replace(".", ",")
    return ("−" if x < 0 and float(text.replace(" ", "").replace(",", ".")) != 0 else "") + text


def ganz(x: float) -> str:
    return de(round(x), 0)


def zehnerpotenz(x: float, stellen: int = 1) -> str:
    """1,14e10 -> '1,1·10¹⁰'."""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(abs(m), stellen) >= 10:
        m, e = m / 10, e + 1
    return f"{de(m, stellen)}·10{str(e).translate(_HOCH)}"


def gross(x: float, stellen: int = 1, ab: float = 1e6) -> str:
    """Zahl in Schreibweise nach Größe: ab `ab` als Zehnerpotenz, sonst mit Tausendertrenner (ganzzahlig, wenn über 1000)."""
    if abs(x) >= ab:
        return zehnerpotenz(x, stellen)
    return de(x, 0 if abs(x) >= 1000 else stellen)


def mb(x: float) -> str:
    """Speichergröße aus Megabyte (10^6 Byte): 38,97 -> '39 MB', 1830,6 -> '1,8 GB', 85 999,7 -> '86 GB', 2,8·10^7 -> '28 TB', 8,9·10^9 -> '8,9 PB'."""
    for grenze, einheit in ((1e9, "PB"), (1e6, "TB"), (1e3, "GB")):
        if x >= grenze:
            v = x / grenze
            return f"{de(v, 1 if v < 10 else 0)} {einheit}"
    return f"{de(x, 0 if x >= 10 else (2 if x >= 0.1 else (3 if x >= 0.001 else 4)))} MB"
