"""1.20: each test here fails on 1.19.0 and passes on 1.20.0 (the bugs and
the numbers before and after are in CHANGELOG.md)."""
import numpy as np
import pytest

import cavsqueeze as cs
from cavsqueeze.cumulant import Rates, collective_moments, evolve
from cavsqueeze.ensemble import Ensemble
from cavsqueeze.protocols import css_x


def test_piqs_reference_has_the_package_dephasing_rate():
    """`exact.dicke_piqs` passed gamma_phi to PIQS unchanged, and PIQS's
    dephasing rate gives each coherence half that decay rate: with dephasing
    alone, <Jx> of 4 spins at t = 1 was 2 exp(-0.35) = 1.409 instead of
    2 exp(-0.7) = 0.993.  Now <Jx> = (N/2) exp(-gamma_phi t) to 1e-6, equal to
    the explicit Lindblad solution `full_hilbert` (L_j = sqrt(gamma_phi/2)
    sigma z_j) to 1e-6 with all terms on, and the cumulant solver agrees with
    PIQS for 40 spins with strong dephasing (gamma_phi t = 0.3 at the end) to
    0.2 dB and 0.02 in the mean spin (0.11 dB and 0.006 measured; with the old
    reference 1.2 dB and 2.0)."""
    pytest.importorskip("qutip")
    from cavsqueeze import cumulant_raw as raw
    from cavsqueeze.exact import dicke_piqs, full_hilbert, xi2_from_moments

    N, g = 4, 0.7
    times = np.array([0.0, 0.5, 1.0, 2.0])
    J = dicke_piqs(N, 0.0, 0.0, 0.0, g, np.pi / 2, times)["J"]
    assert np.allclose(J[:, 0], (N / 2) * np.exp(-g * times), rtol=1e-6, atol=0.0)

    # every term at once, from a tilted coherent state (theta from +z)
    theta, times = 1.1, np.linspace(0.0, 1.5, 4)
    rt = raw.Rates(delta=np.zeros(N), G=np.ones(N), n=np.ones(N), chi1=0.6, Gd=0.3, Gu=0.1,
                   gamma_phi=0.4)
    ex = full_hilbert(rt, [np.sin(theta), 0.0, np.cos(theta)], times)
    J_ex = 0.5 * np.stack([(2 * ex["s"].real).sum(1), (2 * ex["s"].imag).sum(1),
                           ex["z"].sum(1)], axis=1)
    J = dicke_piqs(N, 0.6, 0.3, 0.1, 0.4, theta, times)["J"]
    assert np.max(np.abs(J - J_ex)) < 1e-6
    # theta is measured from +z (2e-8 off with QuTiP 5.0.0, 1e-16 with 5.3)
    assert abs(J[0, 2] - (N / 2) * np.cos(theta)) < 1e-6

    N, chi, Gd, Gu, gphi = 40, 1.0, 0.6, 0.15, 8.0
    times = np.linspace(0, 1.5 / N, 7)
    ref = dicke_piqs(N, chi, Gd, Gu, gphi, theta=np.pi / 2, times=times)
    rtc = raw.Rates(delta=np.zeros(1), G=np.ones(1), n=np.array([N], float), chi1=chi,
                    Gd=Gd, Gu=Gu, gamma_phi=gphi)
    sts = raw.evolve(raw.product_state(1, [1.0, 0.0, 0.0]), rtc, times[-1], t_eval=times)
    for k, st in enumerate(sts):
        assert np.max(np.abs(raw.collective_moments(st, rtc.n)[0] - ref["J"][k])) < 0.02
        if k:
            xi_c = raw.wineland_xi2(st, rtc.n)[0]
            xi_e = xi2_from_moments(ref["J"][k], ref["Cov"][k], N)
            assert abs(10 * np.log10(xi_c / xi_e)) < 0.2, (k, xi_c, xi_e)


def test_twist_untwist_refuses_a_zero_signal_step():
    """The slope is a central difference with steps +/- phi; phi = 0 returned
    a gain of nan (and a RuntimeWarning from NumPy) instead of an error."""
    N = 1e8
    p = cs.from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=0.15)
    ens = cs.homogeneous(N)
    for bad in (0.0, -0.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="phi"):
            cs.twist_untwist(p, ens, 2e-5, phi=bad)
    # valid inputs are unchanged: the gain does not depend on the sign of the step
    a = cs.twist_untwist(p, ens, 2e-5)["gain"][0]
    b = cs.twist_untwist(p, ens, 2e-5, phi=-1e-6)["gain"][0]
    assert np.isfinite(a) and abs(a / b - 1.0) < 1e-6


def _measured(N=1e6):
    """No cavity dynamics, only the continuous measurement of J_z."""
    p = cs.from_hz(g_hz=0.0, kappa_hz=1e4, Delta_hz=30e6)
    return p, Rates.from_params(p, cs.homogeneous(N))


def test_measurement_settings_are_checked():
    """Rates.meas and meas_eta were not checked.  With meas_eta = 1.5 (an
    efficiency above 1) the state ended with Var(J_y) Var(J_z) at 0.833 of
    the smallest value the uncertainty relation allows, <J_x>^2 / 4; a
    negative rate made Var(J_z) grow to 2.5e11 for N = 1e6 spins (N/4 =
    2.5e5); and with half the spins as spectators the measured variance came
    out 2.08e5 where measuring all of J_z gives 1.25e5 (spectators are not
    conditioned).  Each is now refused."""
    N, t = 1e6, 1e-3
    gm = 4.0 / (N * t)                      # Gamma_m N t / 4 = 1
    _, rt = _measured(N)
    for eta in (1.5, 0.0, -1.0):
        rt.meas, rt.meas_eta = gm, eta
        with pytest.raises(ValueError, match="meas_eta"):
            evolve(css_x(1), rt, t)
    rt.meas_eta = 1.0
    for bad in (-gm, np.inf, np.nan):
        rt.meas = bad
        with pytest.raises(ValueError, match="meas"):
            evolve(css_x(1), rt, t)
    p = cs.from_hz(g_hz=0.0, kappa_hz=1e4, Delta_hz=30e6)
    ens = Ensemble(delta=np.zeros(1), weight=np.ones(1), n=np.array([N / 2]),
                   spec_delta=np.array([2 * np.pi * 1e6]), spec_n=np.array([N / 2]))
    rts = Rates.from_params(p, ens)
    rts.meas = gm
    with pytest.raises(ValueError, match="spectator"):
        evolve(css_x(1), rts, t)

    # valid settings are unchanged: the closed form (N/4) / (1 + Gamma_m N t / 4),
    # and the minimal uncertainty product for eta = 1 (twice it for eta = 1/2)
    for eta, product in ((1.0, 1.0), (0.5, 1.5)):
        rt.meas, rt.meas_eta = gm, eta
        J, Cov, _, _ = collective_moments(evolve(css_x(1), rt, t), rt.n)
        assert abs(Cov[2, 2] / (N / 8) - 1.0) < 1e-5
        assert abs(Cov[1, 1] * Cov[2, 2] / (J[0] / 2) ** 2 - product) < 1e-5
    rts.meas = 0.0                          # spectators without a measurement
    evolve(css_x(1), rts, t)
