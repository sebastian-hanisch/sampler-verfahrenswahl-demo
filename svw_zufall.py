"""Blockweise Zufallszahlen aus demselben SplitMix64 wie `svw_rng.SplitMix64` (gleiche Folge, bitgleich), mit numpy-Feldern statt einer Python-Schleife.

Der Generator ist ein Zähler: die k-te Zahl ist eine feste Mischung von `seed + k * 0x9E3779B97F4A7C15 (mod 2^64)`. Darum lässt sich ein ganzer Block ohne
Schleife berechnen; die Ganzzahl-Arithmetik von numpy (uint64) rechnet modulo 2^64 genauso wie die Maske im Skalar-Generator. Es kommt keine Gleitkomma-Funktion
der C-Bibliothek vor, die Folge ist auf jeder Plattform dieselbe (`tests/test_rng.py` vergleicht Block und Skalar-Generator Zahl für Zahl).

Gebraucht wird das, wo eine App-Rechnung Millionen Ziehungen braucht (Ketten mit 3 000 Schritten in 200 Wiederholungen): eine Python-Schleife mit dem
Skalar-Generator wäre dafür zu langsam."""

import numpy as np

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_MASK = (1 << 64) - 1


class Strom:
    """Folge von 64-Bit-Zahlen zu einem Zufalls-Seed; jeder Aufruf setzt die Folge fort."""

    def __init__(self, seed: int):
        self.seed = int(seed) & _MASK
        self.gezogen = 0

    def u64(self, anzahl: int) -> np.ndarray:
        """Die nächsten `anzahl` Zahlen als uint64-Feld."""
        k = np.arange(self.gezogen + 1, self.gezogen + 1 + anzahl, dtype=np.uint64)
        z = np.uint64(self.seed) + k * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
        self.gezogen += anzahl
        return z

    def gleichverteilt(self, anzahl: int) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit, wie `SplitMix64.random()`."""
        return (self.u64(anzahl) >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    def unter(self, n: int, anzahl: int) -> np.ndarray:
        """Ganzzahlen in 0..n-1 als Rest der Division, wie `SplitMix64.below(n)`."""
        return (self.u64(anzahl) % np.uint64(n)).astype(np.int64)
