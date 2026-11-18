"""Zufall: SplitMix64 (Vorlage) und der blockweise Strom müssen dieselbe Folge liefern, auf jeder Plattform."""

import numpy as np

from svw_rng import SplitMix64
from svw_zufall import Strom


def test_splitmix64_bekannte_anfangswerte_zufalls_seed_0():
    """Die ersten drei Ausgaben des SplitMix64 mit Zufalls-Seed 0 (veröffentlichte Referenzfolge von Vigna)."""
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_strom_gleich_skalar_generator_zahl_fuer_zahl():
    r = SplitMix64(12345)
    skalar = [r.next_u64() for _ in range(1000)]
    s = Strom(12345)
    block = list(s.u64(300)) + list(s.u64(1)) + list(s.u64(699))      # in Teilen gezogen: die Folge setzt sich fort
    assert [int(x) for x in block] == skalar


def test_gleichverteilt_und_unter_gleich_skalar():
    r = SplitMix64(7)
    assert list(Strom(7).gleichverteilt(50)) == [r.random() for _ in range(50)]
    r = SplitMix64(7)
    assert [int(x) for x in Strom(7).unter(10, 50)] == [r.below(10) for _ in range(50)]


def test_zufalls_seed_0_und_grosse_zufalls_seeds_laufen():
    for seed in (0, 1, 2 ** 63, 2 ** 64 - 1, 2 ** 64 + 5):
        u = Strom(seed).gleichverteilt(100)
        assert u.min() >= 0.0 and u.max() < 1.0
    assert not np.array_equal(Strom(0).u64(5), Strom(1).u64(5))


def test_gleichverteilung_im_band():
    u = Strom(99).gleichverteilt(200000)
    assert abs(u.mean() - 0.5) < 4 * np.sqrt(1 / 12 / 200000)          # 4 Standardfehler
    z = Strom(5).unter(7, 70000)
    anteile = np.bincount(z, minlength=7) / 70000
    assert np.abs(anteile - 1 / 7).max() < 5 * np.sqrt(1 / 7 * 6 / 7 / 70000)
