"""Handbeispiel von Hand (Hard-Core-Ring, n = 4 Plätze, λ = 2) in Brüchen: jede Zahl ist hier nachgerechnet und steht als Literal im Test.

Von Hand nachrechenbar sind Zustände, Z, π, die Matrixpotenzen A^k des Direktsamplers, d(1), die Zeilen der Heat-bath-Matrix und der Lauf der Kopplung aus der Vergangenheit;
ein Rechner (7×7- und 19×19-Gleichungssystem) braucht es für τ und die erwartete Treffzeit."""

from fractions import Fraction as F

import pytest

import svw_reference as R


@pytest.fixture(scope="module")
def ring():
    return R.Ring(4, 2)


def test_zustaende_gewichte_und_z(ring):
    assert len(ring.s) == 7                                              # Lucas-Zahl L_4
    assert sorted(ring.k) == [0, 1, 1, 1, 1, 2, 2]                       # leer, vier Einzelbelegungen, die Paare {0, 2} und {1, 3}
    assert ring.z == 17                                                  # 1 + 4·2 + 2·4
    assert sum(ring.pi) == 1
    assert ring.pi[ring.idx(0)] == F(1, 17) and ring.pi[ring.idx(1)] == F(2, 17) and ring.pi[ring.idx(5)] == F(4, 17)
    assert ring.pi[ring.idx(5)] + ring.pi[ring.idx(10)] == F(8, 17)      # beide Maximalbelegungen zusammen


def test_direktsampler_jede_belegung_von_hand():
    """Potenzen A^k von Hand: A = [[1,2],[1,0]] -> A^4 = [[11,10],[5,6]], Spur 17; P(Platz 0 belegt) = 6/17; {0, 2}: 6/17 · 1 · 2/3 · 1 = 4/17."""
    ring = R.Ring(4, 2)
    gesamt = F(0)
    platz0 = F(0)
    for m in range(16):
        bits = [(m >> v) & 1 for v in range(4)]
        p = R.direkt_wahrscheinlichkeit(4, 2, bits)
        gesamt += p
        if bits[0]:
            platz0 += p
        if m in ring.s:
            assert p == F(2) ** sum(bits) / 17, m
        else:
            assert p == 0, m
    assert gesamt == 1 and platz0 == F(6, 17)
    assert R.direkt_wahrscheinlichkeit(4, 2, [1, 0, 1, 0]) == F(4, 17)


def test_heatbath_matrix_zeilen_von_hand(ring):
    i0, i5 = ring.idx(0), ring.idx(5)
    assert ring.hb[i0][i0] == F(1, 3) and ring.hb[i0][ring.idx(1)] == F(1, 6)     # leer: Platz v (1/4), Münze 2/3; viermal 1/6, 1/3 bleibt
    assert ring.hb[i5][i5] == F(5, 6) and ring.hb[i5][ring.idx(1)] == F(1, 12) and ring.hb[i5][ring.idx(4)] == F(1, 12)
    assert all(sum(z) == 1 for z in ring.hb)
    assert ring.stationaer() and ring.detailed_balance()


def test_tv_abstand_und_mischzeit(ring):
    assert ring.d(0) == F(16, 17)                                        # 1 − min π (Start „leer“)
    assert ring.d(1) == F(67, 102)                                       # Start {0}: Summe |P − π| = 268/204, halbiert
    start_leer = ring.tv_je_start(ring.hb)[ring.idx(0)]
    assert start_leer == F(24, 51)                                       # vom Start „leer“ nur 24/51
    assert ring.t_mix() == 22
    assert ring.d(21) > F(1, 4) >= ring.d(22)


def test_tau_beider_funktionen_und_treffzeit(ring):
    assert ring.tau(ring.teilchenzahl()) == F(1195, 221)                 # E m = 24/17, Var m = 104/289
    assert ring.tau(ring.gipfelindikator()) == 53                        # E I = 1/2, Var I = 4/17
    liste, q = ring.paarkette()
    assert len(liste) == 19
    assert ring.erwartete_treffzeit() == F(439, 13)


def test_kopplungskosten_exakt(ring):
    kosten = ring.kopplung_kosten(8)
    assert float(kosten) == pytest.approx(88.04, abs=0.005)
    assert 2 * float(kosten) == pytest.approx(176.1, abs=0.05)           # Platzentscheidungen
    assert float(kosten) / 22 == pytest.approx(4.00, abs=0.005)          # in Vielfachen von t_mix(1/4)


def test_ein_lauf_der_kopplung_von_hand(ring):
    zuege = {-4: (2, 0), -3: (3, 0), -2: (1, 0), -1: (0, 1)}             # (Platz, Münze); Münze 1 = belegen, wenn beide Nachbarn leer
    assert ring.kopplungslauf(zuege, 1) == (5, 10)                       # oben {0,2}, unten {1,3}: verschieden
    assert ring.kopplungslauf(zuege, 2) == (5, 8)                        # unten {3}: verschieden
    assert ring.kopplungslauf(zuege, 4) == (1, 1)                        # beide {0}: Ausgabe {0}, Kosten 1 + 2 + 4 = 7


def test_gamma_poisson_von_hand():
    a, b, mittel, sd = R.gamma_poisson(F(2), F(1), 32, 20)               # Gamma(2 + 32, 1 + 20)
    assert (a, b) == (34, 21) and mittel == F(34, 21)
    assert float(mittel) == pytest.approx(1.61905, abs=5e-6) and sd == pytest.approx(0.27766, abs=5e-6)
    a, b, mittel, _ = R.gamma_poisson(F(5), F(3), 0, 0)                  # ohne Daten ist der Posterior der Prior
    assert (a, b, mittel) == (5, 3, F(5, 3))
    assert R.gamma_poisson(F(2), F(1), 0, 20)[:2] == (2, 21)             # Beobachtungssumme 0
