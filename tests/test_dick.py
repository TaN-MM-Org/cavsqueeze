"""v1.13 Dick-effect anchors: closed-form Fourier coefficients of the
Ramsey sensitivity function against independent adaptive quadrature (two independent
code paths), the exact continuous-interrogation zero, the exact
rectangular-window ratio, the 1/sqrt(tau) scaling, cutoff-convergence
honesty, and the squeezing-vs-Dick verdict."""
import numpy as np
import pytest

import cavsqueeze as cs


def _quad_coefficients(m_max, T, Tc, tp):
    """Independent path: adaptive quadrature of the sampled
    sensitivity function (segment boundaries passed as break
    points; no shared arithmetic with the closed form)."""
    from scipy.integrate import quad
    pts = [p for p in (tp, tp + T, 2 * tp + T) if 0 < p < Tc]

    def coef(m):
        re, _ = quad(lambda t: cs.ramsey_sensitivity(t, T, Tc, tp)
                     * np.cos(2 * np.pi * m * t / Tc), 0, Tc,
                     points=pts, limit=400)
        im, _ = quad(lambda t: cs.ramsey_sensitivity(t, T, Tc, tp)
                     * np.sin(2 * np.pi * m * t / Tc), 0, Tc,
                     points=pts, limit=400)
        return (re - 1j * im) / Tc

    g0 = coef(0).real
    return g0, np.array([coef(m) for m in range(1, m_max + 1)])


def test_closed_form_coefficients_match_dense_fft():
    for T, Tc, tp in ((0.4, 1.0, 0.0), (0.4, 1.0, 0.05),
                      (0.1, 0.5, 0.01)):
        g0a, gma = cs.dick_fourier_coefficients(15, T, Tc, tp)
        g0b, gmb = _quad_coefficients(15, T, Tc, tp)
        assert abs(g0a - g0b) < 1e-10
        assert np.abs(gma - gmb).max() < 1e-9


def test_continuous_interrogation_has_exactly_zero_dick_noise():
    """Zero dead time, negligible pulses: g = 1 over the whole cycle,
    every harmonic coefficient vanishes, the aliasing is gone."""
    g0, gm = cs.dick_fourier_coefficients(200, 1.0, 1.0, 0.0)
    assert abs(g0 - 1.0) < 1e-15
    assert np.abs(gm).max() < 1e-12
    sig = cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), 1.0,
                                  1.0, 1.0)
    assert sig < 1e-18


def test_rectangular_window_closed_form_ratio():
    """Short-pulse limit: |g_m / g_0| = |sin(pi m eta) / (pi m eta)|
    exactly, eta = T / Tc."""
    T, Tc = 0.3, 1.0
    eta = T / Tc
    g0, gm = cs.dick_fourier_coefficients(25, T, Tc, 0.0)
    m = np.arange(1, 26)
    ref = np.abs(np.sin(np.pi * m * eta) / (np.pi * m * eta))
    assert np.abs(np.abs(gm) / g0 - ref).max() < 1e-12


def test_scaling_convergence_and_psd_helper():
    Sy = cs.power_law_psd(h0=1e-31, h_m1=4e-32)
    s1 = cs.dick_allan_deviation(Sy, 1.0, 0.4, 1.0, tau_pulse=0.02)
    s4 = cs.dick_allan_deviation(Sy, 4.0, 0.4, 1.0, tau_pulse=0.02)
    assert abs(s4 - s1 / 2.0) < 1e-12 * s1          # exact 1/sqrt(tau)
    s_more = cs.dick_allan_deviation(Sy, 1.0, 0.4, 1.0,
                                     tau_pulse=0.02, m_max=40000)
    assert abs(s_more - s1) / s1 < 1e-3             # converged cutoff
    with pytest.raises(ValueError, match="cutoff"):
        # 1/f^2 LO noise with a rectangular window converges slowly
        # enough that a tiny cutoff is refused, not silently truncated
        cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), 1.0,
                                0.4, 1.0, m_max=11, tail_tol=1e-9)
    with pytest.raises(ValueError):
        cs.power_law_psd(h0=-1.0)
    with pytest.raises(ValueError):
        cs.dick_fourier_coefficients(10, 0.5, 0.4)  # Tc < T


def test_squeezing_vs_dick_verdict():
    """The point of the module: with a noisy LO the total is Dick-
    limited and squeezing the atoms does not move it; with a quiet LO
    the projection noise dominates and squeezing helps."""
    dphi_sql, dphi_sq = 1e-2, 3e-3                  # ~10 dB of squeezing
    nu0, T, Tc = 4.29e14, 0.4, 1.0
    noisy = cs.power_law_psd(h0=1e-30)
    quiet = cs.power_law_psd(h0=1e-36)
    r_noisy_sql = cs.total_clock_allan_deviation(dphi_sql, nu0, T, 1.0,
                                                 Tc, S_y=noisy)
    r_noisy_sq = cs.total_clock_allan_deviation(dphi_sq, nu0, T, 1.0,
                                                Tc, S_y=noisy)
    assert r_noisy_sql["dick_limited"] and r_noisy_sq["dick_limited"]
    gain_noisy = r_noisy_sql["total"] / r_noisy_sq["total"]
    r_quiet_sql = cs.total_clock_allan_deviation(dphi_sql, nu0, T, 1.0,
                                                 Tc, S_y=quiet)
    r_quiet_sq = cs.total_clock_allan_deviation(dphi_sq, nu0, T, 1.0,
                                                Tc, S_y=quiet)
    gain_quiet = r_quiet_sql["total"] / r_quiet_sq["total"]
    assert gain_quiet > 3.0                         # squeezing pays off
    assert gain_noisy < 1.1                         # ... and here it cannot
    # the Dick term itself is identical for both dphi values:
    assert r_noisy_sql["dick"] == r_noisy_sq["dick"]
    # no LO model: pure projection noise, matching the metrology module
    r0 = cs.total_clock_allan_deviation(dphi_sql, nu0, T, 1.0, Tc)
    assert r0["dick"] == 0.0 and r0["total"] == r0["qpn"]
    assert abs(r0["qpn"] - cs.clock_allan_deviation(
        dphi_sql, nu0, T, 1.0, T_cycle=Tc)) < 1e-30
