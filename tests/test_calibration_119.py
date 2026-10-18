"""1.19: projection-noise calibration of the readout, and the error of the
subtracted detection variance carried into the squeezing estimate.

References used here, none of them the function under test:

* exact sample variances built into the shots (a closed form: the fit must
  return the coefficients that generated them);
* NumPy's `polyfit`, a separate weighted least-squares implementation,
  for the coefficients and their covariance at the same weights;
* the binomial law Var(J_z) = Var(N_up) = N p (1 - p) = N/4 for N
  independent spins with p(up) = 1/2, sampled directly;
* the pulse-jitter law, derived in `_jitter_law` and sampled spin by spin;
* Monte Carlo scatter of the estimates against the reported error bars.
"""
import numpy as np
import pytest

from cavsqueeze import (calibrate_projection_noise, estimate_squeezing,
                        plan_tomography, shots_for_squeezing)

N_CAL = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])


def _binomial_shots(rng, N, m, det_sd):
    """J_z = N_up - N/2 of N independent spins with p(up) = 1/2, plus
    Gaussian detection noise.  Var(N_up) = N p (1 - p) = N/4 (binomial),
    so Var(J_z) = N/4 + det_sd^2."""
    return rng.binomial(int(N), 0.5, m) - N / 2.0 + rng.normal(0.0, det_sd, m)


def _exact_shots(V, m=10):
    """m shots (m even) of +a and -a whose sample variance (ddof=1) is
    exactly V: a^2 m / (m - 1) = V."""
    a = np.sqrt(V * (m - 1) / m)
    return np.tile([a, -a], m // 2)


def test_exact_variances_are_recovered():
    """Variances that follow the law exactly give back its coefficients;
    the N used are large (up to 1e6) to exercise the column scaling."""
    for Nv, coef, quad in [(N_CAL, (4.0, 0.25), False),
                           (N_CAL * 1e3, (300.0, 0.31), False),
                           (N_CAL, (4.0, 0.25, 1e-4), True),
                           (np.geomspace(1e3, 1e6, 7), (50.0, 0.25, 2e-7), True)]:
        V = sum(c * Nv ** k for k, c in enumerate(coef))
        fit = calibrate_projection_noise(Nv, [_exact_shots(v) for v in V],
                                         quadratic=quad)
        assert np.allclose(fit.values, coef, rtol=1e-9, atol=0.0), (fit.values, coef)
        assert fit.chi2 < 1e-12
        assert fit.dof == Nv.size - len(coef)
        assert fit.projection_ratio == pytest.approx(4 * coef[1], rel=1e-9)


@pytest.mark.parametrize("quadratic", [False, True])
def test_matches_numpy_polyfit_at_the_same_weights(quadratic):
    """The coefficients and covariance equal NumPy's polyfit (a separate
    least-squares implementation, with its own column scaling) at the
    weights the fit converged to, and those weights are the fitted curve
    times sqrt(2 / (M - 1)) (the self-consistency the reweighting claims).
    Tolerance: 1e-9 of each error bar for the coefficients (c0 is about
    50 times smaller than its error bar here, so a relative test of c0
    would test rounding of a near-zero number) and 1e-9 relative for the
    covariance; both solutions are direct, with condition numbers below
    1e4 on the scaled design, so rounding is near 1e-12."""
    rng = np.random.default_rng(7)
    shots = [_binomial_shots(rng, n, 800, 3.0) for n in N_CAL]
    fit = calibrate_projection_noise(N_CAL, shots, quadratic=quadratic)
    curve = sum(c * N_CAL ** k for k, c in enumerate(fit.values))
    assert np.allclose(fit.variance_sigmas, curve * np.sqrt(2.0 / 799), rtol=1e-11)
    s2 = np.array([np.var(s, ddof=1) for s in shots])
    assert np.allclose(fit.variances, s2, rtol=1e-15)
    deg = 2 if quadratic else 1
    coef, cov = np.polyfit(N_CAL, s2, deg, w=1.0 / fit.variance_sigmas, cov="unscaled")
    coef, cov = coef[::-1], cov[::-1, ::-1]                # lowest power first
    err = np.sqrt(np.diag(fit.cov))
    assert np.all(np.abs(coef - fit.values) < 1e-9 * err), (coef - fit.values) / err
    assert np.allclose(cov, fit.cov, rtol=1e-9, atol=0.0)
    # chi2 recomputed from the residuals
    r = (s2 - np.polyval(coef[::-1], N_CAL)) / fit.variance_sigmas
    assert fit.chi2 == pytest.approx(float(r @ r), rel=1e-9)


def test_binomial_projection_noise_monte_carlo():
    """200 seeded calibrations of pure projection noise plus detection noise
    of variance 16: no bias beyond 3 standard errors, the scatter of c0 and
    of c1 is 0.8 to 1.2 times the reported error bars (about 4 standard
    errors of a scatter from 200 values), and 1 to 22 of the 200 tests of
    c1 = 1/4 and of the fit give p < 0.05 (10 expected; a binomial count
    outside 1..22 has a probability below 1e-3).  NumPy's binomial draws
    are not the same in every NumPy version, so the bounds are wide enough
    for any stream, not tuned to one."""
    c0, c1, s0, s1, pp, pf = [], [], [], [], [], []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        fit = calibrate_projection_noise(
            N_CAL, [_binomial_shots(rng, n, 1500, 4.0) for n in N_CAL])
        c0.append(fit.detection_variance)
        c1.append(fit.per_spin_variance)
        s0.append(fit.detection_variance_sigma)
        s1.append(fit.per_spin_variance_sigma)
        pp.append(fit.p_projection)
        pf.append(fit.p_value)
    c0, c1 = np.array(c0), np.array(c1)
    assert abs(c0.mean() - 16.0) < 3 * c0.std(ddof=1) / np.sqrt(200)
    assert abs(c1.mean() - 0.25) < 3 * c1.std(ddof=1) / np.sqrt(200)
    assert 0.8 < c0.std(ddof=1) / np.mean(s0) < 1.2
    assert 0.8 < c1.std(ddof=1) / np.mean(s1) < 1.2
    assert 1 <= np.sum(np.array(pp) < 0.05) <= 22
    assert 1 <= np.sum(np.array(pf) < 0.05) <= 22


def _jitter_law(N, s):
    """Var(J_z) when all spins are tipped by one random angle delta ~
    Normal(0, s) from the equator.  Given delta, N_up is binomial with
    p = (1 + sin delta)/2: mean N p, variance N p (1 - p) = N cos^2(delta)/4.
    So Var(J_z) = E[N cos^2 delta / 4] + Var(N sin delta / 2)
    = (N/4) (1 + e)/2 + (N^2/4) (1 - e)/2 with e = E[cos 2 delta] =
    exp(-2 s^2) (characteristic function of the Gaussian) and E[sin delta]
    = 0.  Returns (c1, c2)."""
    e = np.exp(-2.0 * s * s)
    return (1.0 + e) / 8.0, (1.0 - e) / 8.0


def test_pulse_angle_jitter_follows_the_quadratic_law():
    """Spins tipped by a common random angle (s = 0.02 rad): the quadratic
    fit gives back c0, c1 and c2 of the derived law within 4 error bars (one
    seeded dataset; 4 rather than 3 so that the test does not depend on
    the binomial stream of a particular NumPy version), and the straight
    line is rejected by its chi-square test."""
    s, m = 0.02, 4000
    Nv = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0])
    rng = np.random.default_rng(11)
    shots = []
    for n in Nv:
        d = rng.normal(0.0, s, m)
        shots.append(rng.binomial(int(n), 0.5 * (1.0 + np.sin(d))) - n / 2.0
                     + rng.normal(0.0, 2.0, m))
    c1, c2 = _jitter_law(Nv, s)
    q = calibrate_projection_noise(Nv, shots, quadratic=True)
    assert abs(q.per_spin_variance - c1) < 4 * q.per_spin_variance_sigma
    assert abs(q.technical_coefficient - c2) < 4 * q.technical_coefficient_sigma
    assert abs(q.detection_variance - 4.0) < 4 * q.detection_variance_sigma
    assert q.technical_coefficient > 5 * q.technical_coefficient_sigma
    lin = calibrate_projection_noise(Nv, shots)
    assert lin.p_value < 1e-6


def test_empirical_shot_noise_with_heavy_tailed_detection():
    """Detection noise with 5 % of shots at 4 times the width: over 200
    seeds the Gaussian error bar of c0 is more than 1.3 times too small,
    and the empirical one is within 0.8 to 1.2 times the scatter."""
    c0, sg, se = [], [], []
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        shots = []
        for n in N_CAL:
            z = rng.normal(0.0, 1.0, 1500)
            z = np.where(rng.random(1500) < 0.05, 4.0 * z, z)
            shots.append(rng.binomial(int(n), 0.5, 1500) - n / 2.0 + 6.0 * z)
        g = calibrate_projection_noise(N_CAL, shots)
        e = calibrate_projection_noise(N_CAL, shots, shot_noise="empirical")
        assert np.array_equal(g.values, e.values)          # same point estimate
        c0.append(g.detection_variance)
        sg.append(g.detection_variance_sigma)
        se.append(e.detection_variance_sigma)
    scatter = np.std(c0, ddof=1)
    assert scatter / np.mean(sg) > 1.3
    assert 0.8 < scatter / np.mean(se) < 1.2


def test_refusals():
    rng = np.random.default_rng(0)
    good = [_binomial_shots(rng, n, 50, 1.0) for n in N_CAL]
    with pytest.raises(ValueError, match="one shot array"):
        calibrate_projection_noise(N_CAL[:5], good)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_projection_noise(np.r_[N_CAL[:5], np.nan], good)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_projection_noise(np.r_[N_CAL[:5], 0.0], good)
    with pytest.raises(ValueError, match="at least 2 shots"):
        calibrate_projection_noise(N_CAL, good[:5] + [np.array([1.0])])
    with pytest.raises(ValueError, match="non-finite"):
        calibrate_projection_noise(N_CAL, good[:5] + [np.array([1.0, np.inf])])
    with pytest.raises(ValueError, match="zero scatter"):
        calibrate_projection_noise(N_CAL, good[:5] + [np.ones(5)])
    with pytest.raises(ValueError, match="shot_noise"):
        calibrate_projection_noise(N_CAL, good, shot_noise="robust")
    with pytest.raises(ValueError, match="quadratic"):
        calibrate_projection_noise(N_CAL, good, quadratic="yes")
    with pytest.raises(ValueError, match="at least 4 atom numbers"):
        calibrate_projection_noise(N_CAL[:3], good[:3], quadratic=True)
    with pytest.raises(ValueError, match="distinct atom numbers"):
        calibrate_projection_noise([100.0, 100.0, 100.0], good[:3])
    # variances that fall with N: the fitted line goes negative
    with pytest.raises(ValueError, match="<= 0"):
        calibrate_projection_noise([1.0, 2.0, 3.0, 4.0],
                                   [_exact_shots(v) for v in (60.0, 40.0, 2.0, 1.0)])
    with pytest.raises(ValueError, match="did not settle"):
        calibrate_projection_noise([1.0, 2.0, 3.0, 4.0],
                                   [_exact_shots(v) for v in (1e3, 1.0, 1.0, 1e-3)])


# ---------------------------------------------------------------------------
# detection_variance_sigma in estimate_squeezing and the planner
# ---------------------------------------------------------------------------

VMIN, VMAX, TMIN, VDET, NSPIN = 20.0, 300.0, 0.3, 16.0, 400
TH = np.linspace(0.0, np.pi, 12, endpoint=False)
PROFILE = 0.5 * (VMIN + VMAX) - 0.5 * (VMAX - VMIN) * np.cos(2.0 * (TH - TMIN))


def test_subtraction_moves_var_min_by_the_constant_only():
    """The premise of the quadrature rule, checked on data: with the
    detection variance d subtracted, var_min moves by d (to rounding) and
    its error bar and the weights do not change (both weightings and both
    shot_noise options); with detection_variance_sigma the error bar of
    var_min is sqrt(old^2 + sigma^2) and the point estimate is unchanged."""
    rng = np.random.default_rng(5)
    shots = rng.normal(0.0, 1.0, (12, 600)) * np.sqrt(PROFILE + VDET)[:, None]
    for w in ("model", "sample"):
        for sn in ("gaussian", "empirical"):
            e0 = estimate_squeezing(TH, shots, NSPIN, 1.0, weighting=w, shot_noise=sn)
            e1 = estimate_squeezing(TH, shots, NSPIN, 1.0, detection_variance=VDET,
                                    weighting=w, shot_noise=sn)
            assert abs((e0.var_min - e1.var_min) - VDET) < 1e-12 * e0.var_max
            assert e1.var_min_sigma == pytest.approx(e0.var_min_sigma, rel=1e-12)
            assert np.allclose(e1.cov_cab, e0.cov_cab, rtol=1e-12, atol=0.0)
            e2 = estimate_squeezing(TH, shots, NSPIN, 0.9, contrast_sigma=0.01,
                                    detection_variance=VDET, detection_variance_sigma=3.0,
                                    weighting=w, shot_noise=sn)
            e3 = estimate_squeezing(TH, shots, NSPIN, 0.9, contrast_sigma=0.01,
                                    detection_variance=VDET, weighting=w, shot_noise=sn)
            assert e2.xi2_R == e3.xi2_R and e2.var_min == e3.var_min
            assert e2.var_min_sigma == pytest.approx(np.hypot(e3.var_min_sigma, 3.0),
                                                     rel=1e-14)
            assert e2.xi2_S_sigma == pytest.approx(4 * e2.var_min_sigma / NSPIN, rel=1e-14)
            assert e2.xi2_R_sigma == pytest.approx(e2.xi2_R * np.hypot(
                e2.var_min_sigma / e2.var_min, 2 * 0.01 / 0.9), rel=1e-14)
            assert e2.detection_variance_sigma == 3.0 and e3.detection_variance_sigma == 0.0


def test_calibration_into_estimate_monte_carlo():
    """The whole chain, 200 seeds: calibrate the detection variance from
    coherent states (binomial shots), subtract it from the tomography of a
    squeezed state measured with the same detector, and pass its error bar.
    The scatter of xi2_S is 0.8 to 1.2 times the reported error bar; without
    detection_variance_sigma the error bar is more than 1.5 times too small.
    The mean is within 3 standard errors of the truth 4 VMIN / N."""
    xs, s_with, s_without = [], [], []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        cal = calibrate_projection_noise(
            N_CAL, [_binomial_shots(rng, n, 1500, 4.0) for n in N_CAL])
        assert cal.detection_variance > 0.0
        shots = rng.normal(0.0, 1.0, (12, 1500)) * np.sqrt(PROFILE + VDET)[:, None]
        e = estimate_squeezing(TH, shots, NSPIN, 1.0,
                               detection_variance=cal.detection_variance,
                               detection_variance_sigma=cal.detection_variance_sigma)
        e0 = estimate_squeezing(TH, shots, NSPIN, 1.0,
                                detection_variance=cal.detection_variance)
        xs.append(e.xi2_S)
        s_with.append(e.xi2_S_sigma)
        s_without.append(e0.xi2_S_sigma)
    xs = np.array(xs)
    scatter = xs.std(ddof=1)
    assert 0.8 < scatter / np.mean(s_with) < 1.2
    assert scatter / np.mean(s_without) > 1.5
    assert abs(xs.mean() - 4 * VMIN / NSPIN) < 3 * scatter / np.sqrt(200)


def test_plan_with_detection_sigma_equals_the_estimate():
    """At convergence the estimate's var_min_sigma equals the plan at the
    fitted ellipse with the same detection_variance_sigma (to 1e-9), and
    shots_for_squeezing returns the smallest shot count that meets the
    target, with the floor xi2_R sqrt((2 sC/C)^2 + (sd/var_min)^2)
    refused."""
    rng = np.random.default_rng(3)
    m = 400
    shots = rng.normal(0.0, 1.0, (12, m)) * np.sqrt(PROFILE + VDET)[:, None]
    e = estimate_squeezing(TH, shots, NSPIN, 0.95, contrast_sigma=0.002,
                           detection_variance=VDET, detection_variance_sigma=2.0)
    plan = plan_tomography(TH, m, e.var_min, e.var_max, e.theta_min, NSPIN, 0.95, 0.002,
                           detection_variance=VDET, detection_variance_sigma=2.0)
    assert abs(plan["var_min_sigma"] - e.var_min_sigma) < 1e-9 * e.var_min_sigma
    assert plan["xi2_R_sigma"] == pytest.approx(e.xi2_R_sigma, rel=1e-9)

    kw = dict(N=NSPIN, contrast=0.95, contrast_sigma=0.002, theta_min=TMIN,
              detection_variance=VDET, detection_variance_sigma=2.0)
    x = plan_tomography(TH, 2, VMIN, VMAX, TMIN, NSPIN, 0.95, 0.002, VDET)["xi2_R"]
    floor = x * np.hypot(2 * 0.002 / 0.95, 2.0 / VMIN)
    target = 1.2 * floor
    mm, p = shots_for_squeezing(target, TH, VMIN, VMAX, **kw)
    assert p["xi2_R_sigma"] <= target
    prev = plan_tomography(TH, mm - 1, VMIN, VMAX, TMIN, NSPIN, 0.95, 0.002, VDET, 2.0)
    assert prev["xi2_R_sigma"] > target
    with pytest.raises(ValueError, match="detection-variance error"):
        shots_for_squeezing(0.999 * floor, TH, VMIN, VMAX, **kw)
    # without the sigma the old floor and message are unchanged
    with pytest.raises(ValueError, match="contrast floor"):
        shots_for_squeezing(0.5 * x * 2 * 0.002 / 0.95, TH, VMIN, VMAX, N=NSPIN,
                            contrast=0.95, contrast_sigma=0.002)


def test_detection_sigma_refusals():
    rng = np.random.default_rng(0)
    shots = rng.normal(0.0, 1.0, (12, 50)) * np.sqrt(PROFILE)[:, None]
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="detection_variance_sigma"):
            estimate_squeezing(TH, shots, NSPIN, 1.0, detection_variance_sigma=bad)
        with pytest.raises(ValueError, match="detection_variance_sigma"):
            plan_tomography(TH, 50, VMIN, VMAX, detection_variance_sigma=bad)
        with pytest.raises(ValueError, match="detection_variance_sigma"):
            shots_for_squeezing(0.01, TH, VMIN, VMAX, N=NSPIN, contrast=1.0,
                                detection_variance_sigma=bad)
