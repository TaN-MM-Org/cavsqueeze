"""Planning anchors: the planned covariance IS the estimator's WLS
covariance (two code paths, machine precision); equally spaced angles
give the exactly orthogonal design diag(K, K/2, K/2); the shot-count
inversion is exact and its contrast floor is refused by an exact
argument; seeded Monte Carlo matches the planned error bar; degenerate
angle sets are reported by the planner and refused by the estimator on
the same arithmetic."""
import numpy as np
import pytest

from cavsqueeze import (estimate_squeezing, plan_tomography,
                        shots_for_squeezing, variance_tomography)

VMIN, VMAX, TMIN = 18.0, 40.0, 0.35
ANGLES = np.linspace(0.0, np.pi, 9, endpoint=False)


def _profile(th):
    c = 0.5 * (VMIN + VMAX)
    r = 0.5 * (VMAX - VMIN)
    return c - r * np.cos(2.0 * (th - TMIN))


def test_plan_equals_estimator_covariance_exactly():
    """Feed the estimator the noise-free variance profile with the
    same per-angle sigmas the planner assumes: same X, same weights,
    same matrix -- to machine precision, including var_min_sigma."""
    m = 41
    plan = plan_tomography(ANGLES, m, VMIN, VMAX, TMIN)
    assert plan["identifiable"]
    v = _profile(ANGLES)
    sig = v * np.sqrt(2.0 / (m - 1))
    fit = variance_tomography(ANGLES, v, sig)
    assert np.allclose(plan["cov_cab"], fit["cov"], rtol=0, atol=1e-12)
    assert abs(plan["var_min_sigma"] - fit["var_min_sigma"]) < 1e-12
    assert abs(fit["var_min"] - VMIN) < 1e-9


def test_equally_spaced_angles_are_exactly_orthogonal():
    """K equally spaced angles over half a turn: sum cos(4 th_k) =
    sum sin(4 th_k) = sum cos(2 th_k) = 0 exactly, so with equal
    weights the normal matrix is diag(K, K/2, K/2) and the covariance
    its exact inverse."""
    k, m, v = 9, 25, 30.0
    th = np.linspace(0.0, np.pi, k, endpoint=False)
    plan = plan_tomography(th, m, v, v, 0.0)
    w2 = 1.0 / (v * np.sqrt(2.0 / (m - 1))) ** 2
    want = np.diag([1.0 / (k * w2), 2.0 / (k * w2), 2.0 / (k * w2)])
    assert np.allclose(plan["cov_cab"], want, rtol=1e-12, atol=1e-12)


def test_shots_inversion_exact_and_contrast_floor_refused():
    target = 0.02
    m, plan = shots_for_squeezing(target, ANGLES, VMIN, VMAX, N=200,
                                  contrast=0.9, contrast_sigma=0.004,
                                  theta_min=TMIN)
    assert plan["xi2_R_sigma"] <= target
    if m > 2:
        worse = plan_tomography(ANGLES, m - 1, VMIN, VMAX, TMIN, N=200,
                                contrast=0.9, contrast_sigma=0.004)
        assert worse["xi2_R_sigma"] > target
    # the floor no tomography removes: xi2_R * 2 sigma_C / C
    base = plan_tomography(ANGLES, 2, VMIN, VMAX, TMIN, N=200,
                           contrast=0.9, contrast_sigma=0.004)
    floor = base["xi2_R"] * 2.0 * 0.004 / 0.9
    with pytest.raises(ValueError, match="contrast floor"):
        shots_for_squeezing(0.99 * floor, ANGLES, VMIN, VMAX, N=200,
                            contrast=0.9, contrast_sigma=0.004,
                            theta_min=TMIN)


def test_monte_carlo_matches_planned_sigma():
    """300 seeded synthetic experiments on a Gaussian state with the
    planned covariance: the scatter of the estimated xi2_S must match
    the planner's predicted error bar."""
    n_spins, m = 100, 60
    plan = plan_tomography(ANGLES, m, VMIN, VMAX, TMIN, N=n_spins)
    rng = np.random.default_rng(19)
    v = _profile(ANGLES)
    vals = []
    for _ in range(300):
        shots = [rng.normal(0.0, np.sqrt(vi), m) for vi in v]
        est = estimate_squeezing(ANGLES, shots, n_spins, contrast=1.0)
        vals.append(est.xi2_S)
    emp = np.std(vals, ddof=1)
    assert np.isclose(emp, plan["xi2_S_sigma"], rtol=0.2)


def test_degenerate_angles_reported_and_refused_alike():
    """Two distinct quadratures cannot fix three parameters: the
    planner reports it, and the estimator refuses the same design."""
    th = np.array([0.1, 0.1 + np.pi, 0.6, 0.6 + np.pi])
    plan = plan_tomography(th, 20, VMIN, VMAX)
    assert not plan["identifiable"]
    v = _profile(th)
    with pytest.raises(ValueError, match="degenerate"):
        variance_tomography(th, v, v * 0.05)
    with pytest.raises(ValueError, match="degenerate"):
        shots_for_squeezing(0.1, th, VMIN, VMAX, N=100, contrast=1.0)


def test_input_refusals():
    with pytest.raises(ValueError, match="var_min"):
        plan_tomography(ANGLES, 10, -1.0, 2.0)
    with pytest.raises(ValueError, match="2 shots"):
        plan_tomography(ANGLES, 1, VMIN, VMAX)
    with pytest.raises(ValueError, match="contrast"):
        plan_tomography(ANGLES, 10, VMIN, VMAX, N=100, contrast=1.5)
    with pytest.raises(ValueError, match="positive"):
        shots_for_squeezing(-0.1, ANGLES, VMIN, VMAX, N=100,
                            contrast=1.0)
