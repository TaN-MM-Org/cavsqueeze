"""Metrology projections: identities, coherent-state limits, and consistency
with the solver's own Wineland parameter on a genuinely squeezed state."""
import numpy as np
import pytest

from cavsqueeze import from_hz, homogeneous
from cavsqueeze.cumulant import Rates, wineland_xi2
from cavsqueeze.metrology import (
    clock_allan_deviation,
    magnetometer_sensitivity,
    metrological_gain_db,
    phase_sensitivity,
    squeezing_parameters,
)
from cavsqueeze.protocols import css_x, twist


def _css(N):
    """Equatorial coherent spin state of one homogeneous class."""
    ens = homogeneous(N)
    return css_x(1), ens.n


def _squeezed(N, t):
    """State after cavity-mediated twisting, plus the class populations."""
    p = from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6,
                T=0.02, T2=0.15)
    ens = homogeneous(N)
    rt = Rates.from_params(p, ens)
    return twist(css_x(rt.M), rt, t), rt.n


def test_coherent_state_baselines():
    N = 1e6
    st, n = _css(N)
    m = squeezing_parameters(st, n)
    assert np.isclose(m["xi2_S"], 1.0)
    assert np.isclose(m["xi2_R"], 1.0)
    assert np.isclose(m["contrast"], 1.0)
    assert np.isclose(m["dphi"], 1.0 / np.sqrt(N))
    assert np.isclose(metrological_gain_db(m["xi2_R"]), 0.0)


def test_wineland_kitagawa_ueda_identity():
    """xi_R^2 = xi_S^2 / contrast^2, and both match the solver's own value,
    on a state actually squeezed by the cavity-mediated twist."""
    N = 1e8
    st, n = _squeezed(N, 5e-5)
    m = squeezing_parameters(st, n)
    assert m["xi2_R"] < 1.0                       # genuinely squeezed
    assert np.isclose(m["xi2_R"], m["xi2_S"] / m["contrast"] ** 2, rtol=1e-10)
    xi2_ref, _, vmin_ref, _, Jn_ref = wineland_xi2(st, n)
    assert np.isclose(m["xi2_R"], xi2_ref, rtol=1e-12)
    assert np.isclose(m["dphi"], np.sqrt(vmin_ref) / Jn_ref, rtol=1e-12)
    assert m["dphi"] < 1.0 / np.sqrt(N)           # beats the SQL
    assert metrological_gain_db(m["xi2_R"]) > 0.0


def test_phase_sensitivity_wrapper():
    N = 1e4
    st, n = _css(N)
    assert np.isclose(phase_sensitivity(st, n), 1.0 / np.sqrt(N))


def test_clock_allan_deviation_scalings():
    """SQL value and the standard scalings in tau, N and dead time."""
    N, nu0, T = 1e6, 4.29e14, 0.1        # Sr-clock-like numbers
    dphi = 1.0 / np.sqrt(N)
    s1 = clock_allan_deviation(dphi, nu0, T, tau=1.0)
    assert np.isclose(s1, 1.0 / (2 * np.pi * nu0 * T * np.sqrt(N)))
    # sqrt(tau) averaging
    assert np.isclose(clock_allan_deviation(dphi, nu0, T, tau=4.0), s1 / 2)
    # four times the atoms -> half the instability at fixed xi
    assert np.isclose(clock_allan_deviation(dphi / 2, nu0, T, tau=1.0), s1 / 2)
    # dead time penalty: doubling the cycle costs sqrt(2)
    assert np.isclose(clock_allan_deviation(dphi, nu0, T, tau=1.0, T_cycle=2 * T),
                      s1 * np.sqrt(2))
    with pytest.raises(ValueError):
        clock_allan_deviation(dphi, nu0, T, tau=1.0, T_cycle=T / 2)


def test_magnetometer_sensitivity_scalings():
    gamma = 2 * np.pi * 28.0e9           # electron-like, rad/(s T)
    N, T = 1e10, 1e-3
    dphi = 1.0 / np.sqrt(N)
    b1 = magnetometer_sensitivity(dphi, gamma, T, tau=1.0)
    assert np.isclose(b1, dphi / (gamma * T))
    assert np.isclose(magnetometer_sensitivity(dphi, gamma, T, tau=100.0), b1 / 10)
    # squeezing improves the field sensitivity linearly in dphi
    assert np.isclose(magnetometer_sensitivity(dphi / 3, gamma, T), b1 / 3)


def test_squeezed_state_improves_projected_clock():
    """The projected Allan deviation with the actually squeezed state beats
    the coherent-state projection with the same resources."""
    N = 1e8
    st, n = _squeezed(N, 5e-5)
    dphi_sq = phase_sensitivity(st, n)
    s_sq = clock_allan_deviation(dphi_sq, 4.29e14, 0.1, tau=1.0)
    s_css = clock_allan_deviation(1.0 / np.sqrt(N), 4.29e14, 0.1, tau=1.0)
    assert s_sq < s_css
