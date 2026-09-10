"""Anchors for tomography-based squeezing estimation: exact recovery
of a known covariance (eigenvalues via an independent numpy path), the
Kitagawa-Ueda one-axis-twisting closed form recovered from sampled
shots, the standard quantum limit from a coherent-spin-state sample,
Monte-Carlo compatibility of the reported sigma, the detection-noise
round trip with over-subtraction refusal, and degenerate-design
refusals."""
import numpy as np
import pytest

from cavsqueeze import (estimate_squeezing, metrological_gain_db,
                        oat_closed_form, variance_tomography)


def _cov(V1, V2, C12):
    return np.array([[V1, C12], [C12, V2]])


def _v_theta(cov, th):
    """Variance of the rotated quadrature -- direct conjugation, the
    independent code path."""
    u = np.array([np.cos(th), np.sin(th)])
    return float(u @ cov @ u)


def test_noise_free_fit_recovers_covariance_and_eigenvalues():
    V1, V2, C12 = 2.0, 7.0, -1.5
    cov = _cov(V1, V2, C12)
    th = np.linspace(0.0, np.pi, 9, endpoint=False)
    v = np.array([_v_theta(cov, t) for t in th])
    fit = variance_tomography(th, v, np.full_like(v, 1e-6))
    assert abs(fit["V1"] - V1) < 1e-9
    assert abs(fit["V2"] - V2) < 1e-9
    assert abs(fit["C12"] - C12) < 1e-9
    lo, hi = np.linalg.eigvalsh(cov)
    assert abs(fit["var_min"] - lo) < 1e-9
    assert abs(fit["var_max"] - hi) < 1e-9
    # theta_min is the minimal-variance direction of the same matrix
    assert abs(_v_theta(cov, fit["theta_min"]) - lo) < 1e-9


def test_oat_closed_form_recovered_from_sampled_shots():
    """Sample Gaussian tomography shots from the exact one-axis
    twisting covariance and recover xi2_R within statistical
    tolerance of the closed form."""
    N, mu = 100, 0.05
    oat = oat_closed_form(N, mu)
    a0 = oat["alpha_min"]
    R = np.array([[np.cos(a0), -np.sin(a0)], [np.sin(a0), np.cos(a0)]])
    cov = R @ np.diag([oat["Vmin"], oat["Vmax"]]) @ R.T
    contrast = 2.0 * oat["Jx"] / N
    th = np.linspace(0.0, np.pi, 12, endpoint=False)
    rng = np.random.default_rng(0)
    M = 4000
    shots = [rng.normal(0.0, np.sqrt(_v_theta(cov, t)), M) for t in th]
    est = estimate_squeezing(th, shots, N, contrast)
    assert abs(est.var_min - oat["Vmin"]) < 4.0 * est.var_min_sigma
    assert abs(est.xi2_R - oat["xi2_R"]) < 4.0 * est.xi2_R_sigma
    assert est.xi2_R < 1.0                       # it IS squeezed
    assert metrological_gain_db(est.xi2_R) > 0.0
    # the recovered dip angle matches the closed form (mod pi)
    d = (est.theta_min - a0) % np.pi
    assert min(d, np.pi - d) < 0.05


def test_coherent_spin_state_estimates_the_standard_quantum_limit():
    """Binomial population-difference shots of a CSS: var = N/4 at
    every angle, contrast 1, so xi2_R is compatible with 1."""
    N, M = 400, 3000
    rng = np.random.default_rng(1)
    th = np.linspace(0.0, np.pi, 6, endpoint=False)
    shots = [rng.binomial(N, 0.5, M) - N / 2.0 for _ in th]
    est = estimate_squeezing(th, shots, N, contrast=1.0)
    assert abs(est.xi2_R - 1.0) < 4.0 * est.xi2_R_sigma
    assert abs(est.xi2_S - est.xi2_R) < 1e-12    # contrast exactly 1


def test_monte_carlo_scatter_matches_reported_sigma():
    V1, V2, C12 = 3.0, 9.0, 2.0
    cov = _cov(V1, V2, C12)
    th = np.linspace(0.0, np.pi, 8, endpoint=False)
    N, M = 64, 400
    rng = np.random.default_rng(2)
    vals, sigs = [], []
    for _ in range(150):
        shots = [rng.normal(0.0, np.sqrt(_v_theta(cov, t)), M)
                 for t in th]
        est = estimate_squeezing(th, shots, N, contrast=0.9)
        vals.append(est.xi2_R)
        sigs.append(est.xi2_R_sigma)
    emp = float(np.std(vals))
    rep = float(np.mean(sigs))
    assert 0.6 * rep < emp < 1.6 * rep


def test_detection_noise_round_trip_and_over_subtraction():
    V1, V2, C12 = 2.0, 6.0, 0.5
    cov = _cov(V1, V2, C12)
    vdet = 1.25
    th = np.linspace(0.0, np.pi, 9, endpoint=False)
    rng = np.random.default_rng(3)
    M = 20000
    shots = [rng.normal(0.0, np.sqrt(_v_theta(cov, t) + vdet), M)
             for t in th]
    est = estimate_squeezing(th, shots, 32, contrast=0.95,
                             detection_variance=vdet)
    lo = float(np.linalg.eigvalsh(cov)[0])
    assert abs(est.var_min - lo) < 4.0 * est.var_min_sigma
    assert est.detection_variance == vdet
    with pytest.raises(ValueError, match="over-subtract"):
        estimate_squeezing(th, shots, 32, contrast=0.95,
                           detection_variance=10.0)


def test_refusals_on_degenerate_or_malformed_input():
    rng = np.random.default_rng(4)
    good = [rng.normal(0, 1, 50) for _ in range(4)]
    th4 = np.array([0.0, 0.5, 1.0, 1.5])
    with pytest.raises(ValueError):              # < 3 angles
        estimate_squeezing(th4[:2], good[:2], 10, 1.0)
    with pytest.raises(ValueError, match="degenerate"):
        estimate_squeezing(np.array([0.2, 0.2 + np.pi, 0.2, 0.2]),
                           good, 10, 1.0)
    with pytest.raises(ValueError):              # bad contrast
        estimate_squeezing(th4, good, 10, 0.0)
    with pytest.raises(ValueError):              # bad N
        estimate_squeezing(th4, good, 1, 1.0)
    with pytest.raises(ValueError):              # one shot at an angle
        estimate_squeezing(th4, good[:3] + [np.array([1.0])], 10, 1.0)
    bad = [g.copy() for g in good]
    bad[1][0] = np.nan
    with pytest.raises(ValueError):
        estimate_squeezing(th4, bad, 10, 1.0)
    with pytest.raises(ValueError):              # negative det. var
        estimate_squeezing(th4, good, 10, 1.0, detection_variance=-1)
