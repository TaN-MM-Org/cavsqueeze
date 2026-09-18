"""Synchronized-comparison anchors: for two equal ensembles the
differential instability is exactly sqrt(2) times the single-clock
projection noise and the per-clock figure is exactly the single-clock
projection noise (identities of the quadrature combination); the
single-ensemble deviations equal `clock_allan_deviation` bitwise (two
public code paths); a squeezed ensemble improves every reported
figure by exactly its dphi ratio; the 1/sqrt(tau) averaging law is
exact; and non-physical inputs are refused."""
import numpy as np
import pytest

from cavsqueeze import clock_allan_deviation, synchronized_comparison

NU0 = 4.3e14
T_R, T_C, TAU = 0.061, 0.3, 1000.0


def test_equal_ensembles_exact_identities():
    dphi = 3.2e-3
    out = synchronized_comparison(dphi, dphi, NU0, T_R, TAU, T_C)
    single = clock_allan_deviation(dphi, NU0, T_R, TAU, T_C)
    assert out["qpn_a"] == single == out["qpn_b"]
    assert abs(out["differential"] - np.sqrt(2.0) * single) \
        < 1e-18 * single
    assert abs(out["per_clock"] - single) < 1e-15 * single


def test_squeezing_gain_carries_through_exactly():
    """A squeezed ensemble with dphi smaller by a factor s improves
    its own projection noise by exactly s; with both ensembles
    squeezed, every reported figure scales by exactly s."""
    dphi, s = 3.2e-3, 0.63
    base = synchronized_comparison(dphi, dphi, NU0, T_R, TAU, T_C)
    sq = synchronized_comparison(s * dphi, s * dphi, NU0, T_R, TAU,
                                 T_C)
    for key in ("qpn_a", "qpn_b", "differential", "per_clock"):
        assert abs(sq[key] - s * base[key]) < 1e-15 * base[key]


def test_averaging_law_exact():
    dphi = 2.0e-3
    o1 = synchronized_comparison(dphi, dphi, NU0, T_R, 100.0, T_C)
    o2 = synchronized_comparison(dphi, dphi, NU0, T_R, 400.0, T_C)
    assert abs(o1["differential"] / o2["differential"] - 2.0) < 1e-12


def test_unequal_ensembles_quadrature():
    da, db = 2.0e-3, 5.0e-3
    out = synchronized_comparison(da, db, NU0, T_R, TAU, T_C)
    assert abs(out["differential"]
               - np.hypot(out["qpn_a"], out["qpn_b"])) \
        < 1e-18 * out["differential"]
    assert out["qpn_b"] > out["qpn_a"]


def test_refusals():
    with pytest.raises(ValueError, match="dphi_a"):
        synchronized_comparison(-1.0, 1e-3, NU0, T_R, TAU)
    with pytest.raises(ValueError, match="dphi_b"):
        synchronized_comparison(1e-3, np.inf, NU0, T_R, TAU)
    with pytest.raises(ValueError, match="T_cycle"):
        synchronized_comparison(1e-3, 1e-3, NU0, T_R, TAU,
                                T_cycle=0.01)
