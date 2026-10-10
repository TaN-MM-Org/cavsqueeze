"""1.17: do the measured variances and fringe means follow their exact laws?

* The reported chi2 equals the weighted sum of squared residuals
  recomputed in the test from the shots (an independent calculation).
* With weights that are not the errors, the general form (residuals in
  the pseudo-inverse of their exact covariance) reduces to the usual sum
  when they are, and follows a chi-square law with K - 3 degrees of
  freedom for Gaussian data when they are not (Monte Carlo).
* Seeded tomography and fringe experiments with many shots give uniform
  p-values; angles or phases 5 % off from the recorded ones are flagged.
"""
import numpy as np
import pytest
from scipy import stats

import cavsqueeze as cs
from cavsqueeze.measured import _goodness_of_fit

TH = np.linspace(0.0, np.pi, 9, endpoint=False)


def _var(th):
    return 22.5 - 17.5 * np.cos(2.0 * (th - 0.3))


def test_reported_chi2_is_the_weighted_residual_sum():
    rng = np.random.default_rng(4)
    M = 60
    shots = rng.normal(0.0, 1.0, (TH.size, M)) * np.sqrt(_var(TH))[:, None]
    for weighting in ("model", "sample"):
        e = cs.estimate_squeezing(TH, shots, 100, 1.0, weighting=weighting)
        s2 = shots.var(axis=1, ddof=1)
        fit = e.V1 * np.cos(TH) ** 2 + e.V2 * np.sin(TH) ** 2 + e.C12 * np.sin(2 * TH)
        base = fit if weighting == "model" else s2
        chi2 = np.sum(((s2 - fit) / (base * np.sqrt(2.0 / (M - 1)))) ** 2)
        assert abs(e.chi2 / chi2 - 1) < 1e-9
        assert e.dof == TH.size - 3
        assert abs(e.p_value - stats.chi2.sf(chi2, TH.size - 3)) < 1e-9
    # the fringe: means against A cos + B sin + d, weighted by standard errors
    N, ph = 100, np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rows = [0.45 * N * np.cos(p - 0.2) + rng.normal(0, 5, 50) for p in ph]
    c = cs.estimate_contrast(ph, rows, N)
    m = np.array([r.mean() for r in rows])
    sem = np.array([r.std(ddof=1) / np.sqrt(r.size) for r in rows])
    fit = c.amplitude * np.cos(ph - c.phi0) + c.offset
    chi2 = np.sum(((m - fit) / sem) ** 2)
    assert abs(c.chi2 / chi2 - 1) < 1e-9 and c.dof == 5


def test_general_form_reduces_to_the_usual_sum():
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi, 11)
    X = np.column_stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th)])
    sig = rng.uniform(0.5, 2.0, th.size)
    r = rng.normal(0, 1, th.size)
    w2 = 1 / sig ** 2
    resid = r - X @ np.linalg.solve(X.T @ (X * w2[:, None]), X.T @ (w2 * r))
    a = _goodness_of_fit(X, sig, sig, resid, general=False)
    b = _goodness_of_fit(X, sig, sig, resid, general=True)
    assert abs(a[0] / b[0] - 1) < 1e-10 and a[1] == b[1] == 8


def test_general_form_follows_the_chi_square_law():
    """Gaussian data with errors that are not the fit weights: the
    general statistic is chi-square with K - 3 degrees of freedom."""
    rng = np.random.default_rng(2)
    th = np.linspace(0.0, np.pi, 7, endpoint=False)
    v = _var(th)
    w_sig = np.ones_like(th)                      # equal weights
    err = 0.05 * v * (1.0 + th)                   # unequal true errors
    chi2 = []
    for _ in range(2000):
        y = v + rng.normal(0.0, 1.0, th.size) * err
        chi2.append(cs.variance_tomography(th, y, w_sig, error_sigmas=err)["chi2"])
    chi2 = np.array(chi2)
    assert abs(chi2.mean() - 4) < 3 * np.sqrt(8 / chi2.size)
    assert stats.kstest(stats.chi2.sf(chi2, 4), "uniform").pvalue > 0.01


@pytest.mark.parametrize("noise", ["gaussian", "empirical"])
def test_tomography_p_values_are_uniform(noise):
    rng = np.random.default_rng(5)
    p = []
    for _ in range(400):
        shots = rng.normal(0.0, 1.0, (TH.size, 200)) * np.sqrt(_var(TH))[:, None]
        p.append(cs.estimate_squeezing(TH, shots, 100, 1.0, shot_noise=noise).p_value)
    frac = np.mean(np.array(p) < 0.05)
    assert 0.02 < frac < 0.09


def test_miscalibrated_angles_are_flagged():
    """The rotation actually applied is 5 % larger than the recorded
    angle: with 12 angles of 2000 shots, every one of 20 experiments
    gives p < 0.01, while the same experiments with correct angles do
    not."""
    rng = np.random.default_rng(2)
    th = np.linspace(0.0, np.pi, 12, endpoint=False)
    bad, good = [], []
    for _ in range(20):
        z = rng.normal(0.0, 1.0, (th.size, 2000))
        bad.append(cs.estimate_squeezing(th, z * np.sqrt(_var(1.05 * th))[:, None],
                                         100, 1.0).p_value)
        good.append(cs.estimate_squeezing(th, z * np.sqrt(_var(th))[:, None],
                                          100, 1.0).p_value)
    assert max(bad) < 0.01
    assert np.median(good) > 0.1


def test_fringe_p_values_and_a_phase_error():
    rng = np.random.default_rng(8)
    N = 100
    ph = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p_ok, p_bad = [], []
    for _ in range(400):
        p_ok.append(cs.estimate_contrast(
            ph, [0.45 * N * np.cos(x) + rng.normal(0, 5, 400) for x in ph], N).p_value)
    for _ in range(20):
        p_bad.append(cs.estimate_contrast(
            ph, [0.45 * N * np.cos(1.05 * x) + rng.normal(0, 5, 400) for x in ph], N).p_value)
    assert 0.02 < np.mean(np.array(p_ok) < 0.05) < 0.09
    assert max(p_bad) < 0.01


def test_three_angles_have_no_degrees_of_freedom():
    rng = np.random.default_rng(0)
    th = np.array([0.0, np.pi / 3, 2 * np.pi / 3])
    e = cs.estimate_squeezing(th, rng.normal(0, 5, (3, 50)), 100, 1.0)
    assert e.dof == 0 and e.chi2 == 0.0 and np.isnan(e.p_value)
