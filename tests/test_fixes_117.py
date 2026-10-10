"""1.17 regressions: inputs that returned plausible but wrong numbers (or a
silent nan) without an error.  Each test fails on 1.16.0 and passes now;
the 1.16.0 behavior is quoted in each docstring."""
import numpy as np
import pytest

import cavsqueeze as cs
from cavsqueeze.cumulant import Rates


def test_meanfield_refuses_negative_time():
    """1.16.0: evolve_meanfield(..., t=-T2) integrated backwards and
    returned contrast e = 2.718 (an unphysical Bloch vector longer than
    1).  `evolve` already refused negative times; now both do."""
    T2 = 1e-3
    rt = Rates.from_params(cs.from_hz(0.0, 1e4, 3e7, T2=T2), cs.homogeneous(10))
    s0, z0 = np.full(1, 0.5 + 0j), np.zeros(1, complex)
    with pytest.raises(ValueError, match="evolution time"):
        cs.evolve_meanfield(s0, z0, rt, -T2)
    with pytest.raises(ValueError, match="evolution time"):
        cs.evolve_meanfield(s0, z0, rt, np.nan)
    # forward evolution is unchanged: pure dephasing, contrast exp(-t/T2)
    s, _ = cs.evolve_meanfield(s0, z0, rt, T2)
    assert abs(2 * abs(s[0]) - np.exp(-1.0)) < 1e-8
    s, _ = cs.evolve_meanfield(s0, z0, rt, 0.0)
    assert s[0] == s0[0]


def test_oat_closed_form_refuses_fractional_spin_number():
    """1.16.0: oat_closed_form(2.9, 0.3) returned the N = 2 result
    (xi2_S = 0.85056) without a word, and oat_closed_form(100.7, 0.05)
    the N = 100 one.  A whole number given as a float is still fine."""
    with pytest.raises(ValueError, match="whole number"):
        cs.oat_closed_form(2.9, 0.3)
    with pytest.raises(ValueError, match="whole number"):
        cs.oat_closed_form(100.7, 0.05)
    with pytest.raises(ValueError, match="mu"):
        cs.oat_closed_form(10, np.nan)
    assert cs.oat_closed_form(100.0, 0.05) == cs.oat_closed_form(100, 0.05)
    # the N = 2 closed form, by hand: <Jx> = cos(mu/2), Vmin = 1/2 - (1/2)|sin(mu/2)|
    mu = 0.3
    r = cs.oat_closed_form(2, mu)
    assert abs(r["Jx"] - np.cos(mu / 2)) < 1e-15
    assert abs(r["Vmin"] - (0.5 - 0.5 * abs(np.sin(mu / 2)))) < 1e-15
    assert type(r["Jx"]) is float


def test_plan_refuses_fractional_shot_counts():
    """1.16.0: plan_tomography(angles, 9.9, ...) planned 9 shots per
    angle (var_min_sigma 28.7227, the 9-shot value; 10 shots give
    27.0800) without a word."""
    th = np.linspace(0.0, np.pi, 9, endpoint=False)
    with pytest.raises(ValueError, match="whole numbers"):
        cs.plan_tomography(th, 9.9, 90.0, 160.0)
    with pytest.raises(ValueError, match="whole numbers"):
        cs.plan_tomography(th, np.full(9, 2.5), 90.0, 160.0)
    a = cs.plan_tomography(th, 10.0, 90.0, 160.0)["var_min_sigma"]
    b = cs.plan_tomography(th, 10, 90.0, 160.0)["var_min_sigma"]
    assert a == b


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_contrast_sigma_must_be_finite(bad):
    """1.16.0: contrast_sigma = nan (or inf) gave xi2_R_sigma = nan (or
    inf) without an error, in estimate_squeezing and plan_tomography."""
    rng = np.random.default_rng(0)
    th = np.linspace(0.0, np.pi, 6, endpoint=False)
    shots = rng.normal(0.0, 5.0, (6, 100))
    with pytest.raises(ValueError, match="contrast"):
        cs.estimate_squeezing(th, shots, 100, 0.9, contrast_sigma=bad)
    with pytest.raises(ValueError, match="contrast"):
        cs.plan_tomography(th, 10, 90.0, 160.0, N=480, contrast=0.9,
                           contrast_sigma=bad)


@pytest.mark.parametrize("call", [
    lambda: cs.clock_allan_deviation(1e-3, 4.29e14, 0.1, 1.0, T_cycle=np.nan),
    lambda: cs.clock_allan_deviation(1e-3, 4.29e14, 0.1, 1.0, T_cycle=np.inf),
    lambda: cs.magnetometer_sensitivity(1e-3, 1.76e11, 1e-3, 1.0, T_cycle=np.nan),
    lambda: cs.total_clock_allan_deviation(1e-3, 4.29e14, 0.1, 1.0, np.nan),
    lambda: cs.synchronized_comparison(1e-3, 1e-3, 4.29e14, 0.1, 1.0, np.nan),
    lambda: cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), 1.0, 0.4, np.nan),
    lambda: cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), 1.0, 0.4, np.inf),
    lambda: cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), 1.0, np.nan, 1.0),
    lambda: cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), 1.0, 0.4, 1.0,
                                    tau_pulse=np.nan),
    lambda: cs.ramsey_sensitivity([0.1, 0.5], 0.4, np.nan),
    lambda: cs.power_law_psd(h0=np.nan),
    lambda: cs.power_law_psd(h_m1=np.inf),
    lambda: cs.log_uniform_weights(10.0, 0),
    lambda: cs.log_uniform_weights(np.inf, 3),
])
def test_non_finite_timing_and_noise_refused(call):
    """1.16.0 returned nan for each of these, except: inf for an infinite
    clock cycle, [0, 0] from ramsey_sensitivity with a nan cycle, a PSD
    that is inf everywhere for h_m1 = inf, empty arrays for K = 0 weights
    and nan weights for an infinite dynamic range.  Now each raises
    ValueError."""
    with pytest.raises(ValueError):
        call()


def test_finite_cycle_times_unchanged():
    """The checks above leave every accepted input's result unchanged."""
    a = cs.clock_allan_deviation(1e-3, 4.29e14, 0.1, 1.0, T_cycle=0.3)
    assert a == 1e-3 / (2 * np.pi * 4.29e14 * 0.1) * np.sqrt(0.3 / 1.0)
    w, p = cs.log_uniform_weights(100.0, 5)
    assert abs(w.max() / w.min() - 100.0) < 1e-12 and abs(w.mean() - 1.0) < 1e-15
    assert np.all(p == 0.2)
