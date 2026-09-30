"""1.16 anchors for the tomography estimator's weights and error bars.

* The distribution-free identity for the variance of a sample variance,
  checked by enumerating every outcome of a small discrete
  distribution (an exact reference, no sampling).
* The fixed-weight "sandwich" covariance: it reduces to the usual WLS
  covariance when the weights are the true errors (algebraic identity),
  and it matches the Monte-Carlo covariance of the fit when they are not.
* Model weights: the fitted curve is unbiased where sample weights bias
  it low (truth = the generating covariance); at convergence the
  estimator's covariance equals the planner's (two code paths).
* Empirical error bars: with heavy-tailed (contaminated Gaussian) shots
  the Gaussian error bar is too small and the empirical one matches the
  Monte-Carlo scatter.
* The remaining low bias of the smallest variance of a coherent state,
  documented in the README, against its asymptotic value sqrt(pi/3)
  reported error bars (Rayleigh mean of the fitted anisotropy).
"""
import itertools
import warnings

import numpy as np
import pytest

from cavsqueeze import (estimate_squeezing, plan_tomography,
                        sample_variance_sigma, variance_tomography)
from cavsqueeze.measured import _var_s2_factor

VMIN, VMAX, TMIN = 5.0, 40.0, 0.3


def _profile(th):
    return 0.5 * (VMIN + VMAX) - 0.5 * (VMAX - VMIN) * np.cos(
        2.0 * (th - TMIN))


@pytest.mark.parametrize("m", [2, 4, 5])
def test_variance_of_sample_variance_exact_by_enumeration(m):
    """Var(s^2) = sigma^4 [kurtosis/M - (M-3)/(M(M-1))] for any
    distribution: enumerate all 3^M outcomes of a skewed three-point
    distribution and compare with the formula at the exact moments."""
    vals = np.array([-1.0, 0.0, 2.0])
    prob = np.array([0.2, 0.5, 0.3])
    mean = prob @ vals
    var = prob @ (vals - mean) ** 2
    mu4 = prob @ (vals - mean) ** 4
    e1 = e2 = 0.0
    for idx in itertools.product(range(3), repeat=m):
        p = float(np.prod(prob[list(idx)]))
        s2 = float(np.var(vals[list(idx)], ddof=1))
        e1 += p * s2
        e2 += p * s2 * s2
    assert abs(e1 - var) < 1e-12                # s^2 is unbiased
    exact = e2 - e1 ** 2
    formula = var ** 2 * _var_s2_factor(mu4 / var ** 2, m)
    assert abs(formula - exact) < 1e-12 * exact


def test_gaussian_kurtosis_reduces_to_the_old_formula():
    rng = np.random.default_rng(0)
    for m in (2, 3, 10, 57):
        assert abs(_var_s2_factor(3.0, m) - 2.0 / (m - 1)) < 1e-15
        x = rng.normal(0.0, 2.0, m)
        want = np.var(x, ddof=1) * np.sqrt(2.0 / (m - 1))
        assert abs(sample_variance_sigma(x, kurtosis=3.0) - want) \
            < 1e-13 * want


def test_sandwich_covariance_identity_and_monte_carlo():
    th = np.linspace(0.0, np.pi, 7, endpoint=False)
    v = _profile(th)
    sig = 0.1 * v
    usual = variance_tomography(th, v, sig)["cov"]
    same = variance_tomography(th, v, sig, error_sigmas=sig)["cov"]
    assert np.allclose(same, usual, rtol=1e-12, atol=0.0)
    # equal weights, unequal true errors: the fit covariance is the
    # sandwich, not (X^T W X)^-1
    w_sig = np.ones_like(th)
    true_sig = 0.05 * v * (1.0 + th)
    sand = variance_tomography(th, v, w_sig,
                               error_sigmas=true_sig)["cov"]
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(3000):
        fit = variance_tomography(th, v + rng.normal(0.0, true_sig), w_sig)
        draws.append([fit["c"], fit["a"], fit["b"]])
    draws = np.array(draws)
    emp = np.cov(draws.T)
    # Monte-Carlo covariance from 3000 draws: ~ 2.6 % relative scatter
    # on the diagonal
    assert np.allclose(np.diag(emp), np.diag(sand), rtol=0.1)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert abs(emp[i, j] - sand[i, j]) \
            < 0.1 * np.sqrt(sand[i, i] * sand[j, j])
    # and the naive (X^T W X)^-1 is not the fit covariance here
    naive = variance_tomography(th, v, w_sig)["cov"]
    assert not np.allclose(np.diag(naive), np.diag(sand), rtol=0.5)


def test_model_weights_remove_the_weighting_bias():
    """400 seeded experiments, 9 angles x 20 shots: the mean of the
    fitted curve c = (V1 + V2)/2 is biased low by sample weights and
    unbiased (within 3 standard errors) with model weights."""
    k, m, reps = 9, 20, 400
    th = np.linspace(0.0, np.pi, k, endpoint=False)
    sd = np.sqrt(_profile(th))[:, None]
    rng = np.random.default_rng(7)
    c_true = 0.5 * (VMIN + VMAX)
    got = {"sample": [], "model": []}
    for _ in range(reps):
        shots = rng.normal(0.0, 1.0, (k, m)) * sd
        for w in got:
            e = estimate_squeezing(th, shots, 100, 1.0, weighting=w)
            got[w].append(0.5 * (e.V1 + e.V2))
    for w, c in got.items():
        c = np.asarray(c)
        se = c.std(ddof=1) / np.sqrt(reps)
        if w == "model":
            assert abs(c.mean() - c_true) < 3.0 * se, (c.mean(), se)
        else:
            # the 1.15 behaviour: 14.4 % low with this seed
            assert c.mean() < c_true - 10.0 * se
            assert 0.83 < c.mean() / c_true < 0.89


def test_converged_model_fit_equals_the_plan():
    """At convergence the estimator's weights are the planner's
    weights evaluated at the fitted ellipse, so the two covariances
    agree to rounding (including a detection-noise subtraction)."""
    k, m, vdet = 8, 300, 2.5
    th = np.linspace(0.0, np.pi, k, endpoint=False)
    rng = np.random.default_rng(3)
    shots = rng.normal(0.0, 1.0, (k, m)) \
        * np.sqrt(_profile(th) + vdet)[:, None]
    e = estimate_squeezing(th, shots, 100, 1.0, detection_variance=vdet)
    assert e.weighting == "model" and e.n_reweight >= 1
    plan = plan_tomography(th, m, e.var_min, e.var_max, e.theta_min,
                           detection_variance=vdet)
    assert np.allclose(e.cov_cab, plan["cov_cab"], rtol=1e-9, atol=0.0)
    assert abs(e.var_min_sigma - plan["var_min_sigma"]) \
        < 1e-9 * e.var_min_sigma


def test_empirical_error_bars_with_heavy_tailed_shots():
    """Shots with 5 % of outliers at 3x the width (kurtosis 7.65): the
    Gaussian error bar is too small by ~1.8x, the empirical one matches
    the scatter of 200 seeded experiments; the point estimate is the
    same for both."""
    k, m, reps, p, f = 6, 400, 200, 0.05, 3.0
    th = np.linspace(0.0, np.pi, k, endpoint=False)
    sd = np.sqrt(_profile(th))[:, None]
    norm = np.sqrt((1 - p) + p * f * f)
    rng = np.random.default_rng(11)
    vals, sig_g, sig_e = [], [], []
    for _ in range(reps):
        z = rng.normal(0.0, 1.0, (k, m))
        z = np.where(rng.random((k, m)) < p, f * z, z) / norm
        eg = estimate_squeezing(th, z * sd, 100, 1.0)
        ee = estimate_squeezing(th, z * sd, 100, 1.0,
                                shot_noise="empirical")
        assert ee.xi2_S == eg.xi2_S and ee.shot_noise == "empirical"
        vals.append(eg.xi2_S)
        sig_g.append(eg.xi2_S_sigma)
        sig_e.append(ee.xi2_S_sigma)
    scatter = np.std(vals, ddof=1)             # ~5 % statistical error
    assert scatter / np.mean(sig_g) > 1.5
    assert 0.85 < scatter / np.mean(sig_e) < 1.2


def test_coherent_state_minimum_variance_bias():
    """For a coherent state (equal variances) the fitted smallest
    variance is the smaller of two noisy eigenvalues and is biased low.
    With equally spaced angles and Gaussian shots the bias tends to
    sqrt(pi/2) sigma_a and the reported error to sqrt(3/2) sigma_a, a
    ratio sqrt(pi/3) = 1.02; 400 experiments must show it."""
    k, m, reps, n = 9, 400, 400, 100
    th = np.linspace(0.0, np.pi, k, endpoint=False)
    rng = np.random.default_rng(5)
    vm, sg = [], []
    for _ in range(reps):
        e = estimate_squeezing(th, rng.normal(0.0, np.sqrt(n / 4.0),
                                              (k, m)), n, 1.0)
        vm.append(e.var_min)
        sg.append(e.var_min_sigma)
    ratio = (n / 4.0 - np.mean(vm)) / np.mean(sg)
    assert 0.85 < ratio < 1.3, ratio


def test_reweighting_that_does_not_settle_falls_back(monkeypatch):
    import cavsqueeze.measured as meas
    th = np.linspace(0.0, np.pi, 9, endpoint=False)
    shots = np.random.default_rng(2).normal(0.0, 1.0, (9, 20)) \
        * np.sqrt(_profile(th))[:, None]
    assert estimate_squeezing(th, shots, 100, 1.0).n_reweight > 1
    monkeypatch.setattr(meas, "_MAX_REWEIGHT", 1)
    with pytest.warns(RuntimeWarning, match="did not settle"):
        e = estimate_squeezing(th, shots, 100, 1.0)
    ref = estimate_squeezing(th, shots, 100, 1.0, weighting="sample")
    assert e.weighting == "sample" and "did not settle" in e.weighting_fallback
    assert e.xi2_S == ref.xi2_S and e.xi2_S_sigma == ref.xi2_S_sigma


def test_model_weights_never_refuse_what_sample_weights_fit():
    """Three shots per angle: model weighting often cannot proceed. It
    then returns exactly the weighting='sample' result with a warning and
    the reason recorded, so over 200 seeded datasets the default refuses
    only datasets that the sample weights refuse too."""
    th = np.linspace(0.0, np.pi, 6, endpoint=False)
    v = 15.0 - 10.0 * np.cos(2.0 * (th - 0.2))
    rng = np.random.default_rng(1003)
    n_fallback = n_model_refused = n_sample_refused = 0
    for _ in range(200):
        shots = rng.normal(0.0, 1.0, (6, 3)) * np.sqrt(v)[:, None]
        try:
            ref = estimate_squeezing(th, shots, 100, 1.0, weighting="sample")
        except ValueError:
            ref = None
            n_sample_refused += 1
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                e = estimate_squeezing(th, shots, 100, 1.0)
            except ValueError:
                n_model_refused += 1
                assert ref is None          # refused only if sample refuses
                continue
        assert e.weighting_fallback is None or ref is not None
        if e.weighting_fallback is not None:
            n_fallback += 1
            assert e.weighting == "sample" and len(rec) == 1
            assert issubclass(rec[0].category, RuntimeWarning)
            assert e.xi2_S == ref.xi2_S and e.var_min == ref.var_min
        else:
            assert e.weighting == "model" and len(rec) == 0
    assert n_fallback > 20                  # the fallback path is exercised
    assert n_model_refused <= n_sample_refused


def test_new_option_refusals():
    rng = np.random.default_rng(4)
    th = np.linspace(0.0, np.pi, 6, endpoint=False)
    shots = rng.normal(0.0, 3.0, (6, 30))
    with pytest.raises(ValueError, match="weighting"):
        estimate_squeezing(th, shots, 100, 1.0, weighting="median")
    with pytest.raises(ValueError, match="shot_noise"):
        estimate_squeezing(th, shots, 100, 1.0, shot_noise="poisson")
    with pytest.raises(ValueError, match="error_sigmas"):
        variance_tomography(th, np.ones(6), np.ones(6),
                            error_sigmas=-np.ones(6))
    # three shots per angle: the model curve dips below zero, so the
    # default falls back to sample weights with a warning (seed 0 found
    # by scanning, deterministic)
    v = 15.0 - 10.0 * np.cos(2.0 * (th - 0.2))
    few = np.random.default_rng(0).normal(0.0, 1.0, (6, 3)) \
        * np.sqrt(v)[:, None]
    with pytest.warns(RuntimeWarning, match="falling back"):
        e = estimate_squeezing(th, few, 100, 1.0)
    assert "<= 0 at a measured angle" in e.weighting_fallback
    assert e.var_min == estimate_squeezing(th, few, 100, 1.0,
                                           weighting="sample").var_min
    # the dataclass default agrees with the function default
    assert meas_default() == "model"


def meas_default():
    import dataclasses
    from cavsqueeze import SqueezingEstimate
    f = {x.name: x for x in dataclasses.fields(SqueezingEstimate)}
    return f["weighting"].default
