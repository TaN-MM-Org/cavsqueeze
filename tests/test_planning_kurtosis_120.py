"""1.20: planning a tomography for shots that are not Gaussian
(`plan_tomography(..., kurtosis=...)`, `shots_for_squeezing(..., kurtosis=...)`).

With M shots of kurtosis k the sample variance has the exact variance
sigma^4 f(k, M), f(k, M) = k / M - (M - 3) / (M (M - 1)) (checked by
enumeration in tests/test_tomography_noise.py); f(3, M) = 2 / (M - 1)."""
import numpy as np
import pytest

from cavsqueeze import estimate_squeezing, plan_tomography, shots_for_squeezing
from cavsqueeze.measured import _sample_kurtosis

TH = np.linspace(0.0, np.pi, 9, endpoint=False)
VMIN, VMAX, TMIN, NSPIN = 5.0, 40.0, 0.3, 100
PROFILE = 0.5 * (VMIN + VMAX) - 0.5 * (VMAX - VMIN) * np.cos(2.0 * (TH - TMIN))


def f(k, M):
    return k / M - (M - 3) / (M * (M - 1))


def test_kurtosis_three_is_the_gaussian_plan():
    """kurtosis=3 takes the new code path and gives the Gaussian plan of 1.19
    (kurtosis=None) to 1e-12, for scalar and per-angle shot counts."""
    for m in (400, np.arange(20, 29)):
        g = plan_tomography(TH, m, VMIN, VMAX, TMIN, NSPIN, 0.95, 0.01, 4.0, 0.5)
        k = plan_tomography(TH, m, VMIN, VMAX, TMIN, NSPIN, 0.95, 0.01, 4.0, 0.5,
                            kurtosis=3.0)
        assert np.max(np.abs(k["cov_cab"] - g["cov_cab"])) < 1e-12 * np.max(np.abs(g["cov_cab"]))
        for key in ("var_min_sigma", "xi2_S_sigma", "xi2_R_sigma"):
            assert k[key] == pytest.approx(g[key], rel=1e-12)


@pytest.mark.parametrize("k, M, K", [(1.0, 7, 6), (1.8, 50, 9), (7.65, 400, 12)])
def test_isotropic_state_closed_form(k, M, K):
    """For equal variances V at K equally spaced angles over half a turn the
    normal matrix is diag(K, K/2, K/2) / s^2 with equal weights, so the
    fitted mean variance has the error V sqrt(f(k, M) / K), which is the
    var_min error of an isotropic state."""
    a = np.linspace(0.0, np.pi, K, endpoint=False)
    V = 20.0
    r = plan_tomography(a, M, V, V, kurtosis=k)
    assert r["var_min_sigma"] == pytest.approx(V * np.sqrt(f(k, M) / K), rel=1e-12)


def test_uniform_kurtosis_scales_the_gaussian_covariance():
    """With the same kurtosis and shot count at every angle, each variance's
    error is the Gaussian one times sqrt(f(k, M) (M - 1) / 2), while the
    weights are unchanged, so the covariance is the Gaussian one times that
    factor squared (to 1e-12), also with detection noise subtracted."""
    M = 300
    g = plan_tomography(TH, M, VMIN, VMAX, TMIN, detection_variance=4.0)
    for k in (1.0, 1.8, 3.0, 7.65, 20.0):
        r = plan_tomography(TH, M, VMIN, VMAX, TMIN, detection_variance=4.0, kurtosis=k)
        ratio = f(k, M) * (M - 1) / 2.0
        assert np.allclose(r["cov_cab"], ratio * g["cov_cab"], rtol=1e-12, atol=0.0)


def test_plan_equals_the_empirical_estimate_at_the_fitted_values():
    """estimate_squeezing(shot_noise="empirical") reports the covariance of
    the same fixed-weight fit with each angle's measured kurtosis; the plan
    at the fitted ellipse with those kurtoses gives the same var_min_sigma
    and xi2_R_sigma to 1e-9."""
    rng = np.random.default_rng(4)
    M = 500
    z = rng.standard_t(6, (9, M)) / np.sqrt(6.0 / 4.0)       # unit variance, kurtosis 6
    shots = z * np.sqrt(PROFILE + 4.0)[:, None]
    e = estimate_squeezing(TH, shots, NSPIN, 0.95, contrast_sigma=0.01,
                           detection_variance=4.0, detection_variance_sigma=0.5,
                           shot_noise="empirical")
    kurt = np.array([_sample_kurtosis(r) for r in shots])
    p = plan_tomography(TH, M, e.var_min, e.var_max, e.theta_min, NSPIN, 0.95, 0.01,
                        4.0, 0.5, kurtosis=kurt)
    assert p["var_min_sigma"] == pytest.approx(e.var_min_sigma, rel=1e-9)
    assert p["xi2_R_sigma"] == pytest.approx(e.xi2_R_sigma, rel=1e-9)
    assert np.allclose(p["cov_cab"], e.cov_cab, rtol=1e-9, atol=0.0)


def _mixture(rng, shape, p=0.05, a=3.0):
    """Unit-variance shots: with probability p a Gaussian a times wider."""
    z = rng.normal(0.0, 1.0, shape)
    z = np.where(rng.random(shape) < p, a * z, z)
    return z / np.sqrt(1 - p + p * a * a)


@pytest.mark.parametrize("label", ["outliers", "uniform"])
def test_planned_error_bar_matches_the_scatter(label):
    """300 seeded tomographies (9 angles, 400 shots each).  Outliers: 5 % of
    shots 3 times wider, kurtosis 3 (0.95 + 0.05 * 81) / (0.95 + 0.05 * 9)^2
    = 7.65 (a Gaussian scale mixture has kurtosis 3 E[a^4] / E[a^2]^2).
    Uniform shots: kurtosis 9/5.  The scatter of the fitted var_min is 0.85
    to 1.15 times the planned error bar (0.96 and 0.99 in the test run),
    while the Gaussian plan is too small (outliers, by 1.75) or too large
    (uniform, by 1.6); the mean error bar reported by shot_noise="empirical"
    agrees with the plan to 15 %, and the mean has no bias beyond 3 standard
    errors."""
    if label == "outliers":
        k, draw = 3 * (0.95 + 0.05 * 81) / (0.95 + 0.05 * 9) ** 2, _mixture
    else:
        k = 1.8

        def draw(rng, shape):
            return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), shape)
    M, runs = 400, 300
    vm, sig = [], []
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        e = estimate_squeezing(TH, draw(rng, (9, M)) * np.sqrt(PROFILE)[:, None], NSPIN, 1.0,
                               shot_noise="empirical")
        vm.append(e.var_min)
        sig.append(e.var_min_sigma)
    scatter = np.std(vm, ddof=1)
    planned = plan_tomography(TH, M, VMIN, VMAX, TMIN, kurtosis=k)["var_min_sigma"]
    gauss = plan_tomography(TH, M, VMIN, VMAX, TMIN)["var_min_sigma"]
    assert 0.85 < scatter / planned < 1.15
    if label == "outliers":
        assert scatter / gauss > 1.5
    else:
        assert scatter / gauss < 0.75
    assert 0.85 < np.mean(sig) / planned < 1.15
    assert abs(np.mean(vm) - VMIN) < 3 * scatter / np.sqrt(runs)


@pytest.mark.parametrize("k", [1.0, 1.8, 3.0, 7.65])
@pytest.mark.parametrize("target", [0.05, 0.02, 0.011])
def test_shot_count_against_the_closed_form(k, target):
    """For one kurtosis at all angles the tomography part of the error bar is
    its Gaussian value at M = 2 times sqrt(f(k, M) / 2), so the smallest M
    solves q M^2 - (q + k - 1) M - (3 - k) >= 0 with
    q = 2 (target^2 - floor^2) / tom2^2: a separate calculation from the
    integer search.  The returned M meets the target and M - 1 does not;
    k = 3 gives the shot count of the closed-form Gaussian path."""
    kw = dict(N=NSPIN, contrast=0.95, contrast_sigma=0.005)
    m, plan = shots_for_squeezing(target, TH, 22.0, 30.0, kurtosis=k, **kw)
    base = plan_tomography(TH, 2, 22.0, 30.0, 0.0, NSPIN, 0.95, 0.005)
    tom2 = base["xi2_R"] * base["var_min_sigma"] / 22.0
    floor = base["xi2_R"] * 2 * 0.005 / 0.95
    q = 2 * (target ** 2 - floor ** 2) / tom2 ** 2
    b = q + k - 1
    root = (b + np.sqrt(b * b + 4 * q * (3 - k))) / (2 * q)
    assert m == max(2, int(np.ceil(root)))
    assert plan["xi2_R_sigma"] <= target
    if m > 2:
        prev = plan_tomography(TH, m - 1, 22.0, 30.0, 0.0, NSPIN, 0.95, 0.005, kurtosis=k)
        assert prev["xi2_R_sigma"] > target
    if k == 3.0:
        assert m == shots_for_squeezing(target, TH, 22.0, 30.0, **kw)[0]


def test_per_angle_kurtosis_and_refusals():
    """A kurtosis per angle (here a Gaussian state read by a detector with
    heavy-tailed noise, so the kurtosis falls where the state's own variance
    is large) gives the smallest M meeting the target; the floors and their
    refusals do not depend on the kurtosis; invalid kurtoses are refused."""
    kd, vdet = 9.0, 8.0                      # detector noise: kurtosis 9, variance 8
    # sum of independent zero-mean parts: mu4 = mu4_a + 6 var_a var_b + mu4_b
    kurt = (3 * PROFILE ** 2 + 6 * PROFILE * vdet + kd * vdet ** 2) / (PROFILE + vdet) ** 2
    kw = dict(N=NSPIN, contrast=0.95, contrast_sigma=0.005, theta_min=TMIN,
              detection_variance=vdet, detection_variance_sigma=0.3)
    m, plan = shots_for_squeezing(0.03, TH, VMIN, VMAX, kurtosis=kurt, **kw)
    prev = plan_tomography(TH, m - 1, VMIN, VMAX, TMIN, NSPIN, 0.95, 0.005, vdet, 0.3,
                           kurtosis=kurt)
    assert plan["xi2_R_sigma"] <= 0.03 < prev["xi2_R_sigma"]
    m_gauss = shots_for_squeezing(0.03, TH, VMIN, VMAX, **kw)[0]
    assert m > m_gauss                       # heavier tails cost shots
    x = plan["xi2_R"]
    floor = x * np.hypot(2 * 0.005 / 0.95, 0.3 / VMIN)
    with pytest.raises(ValueError, match="detection-variance error"):
        shots_for_squeezing(0.999 * floor, TH, VMIN, VMAX, kurtosis=kurt, **kw)
    for bad in (0.5, np.nan, np.inf, [3.0, 3.0], np.full((9, 2), 3.0)):
        with pytest.raises(ValueError, match="kurtosis"):
            plan_tomography(TH, 50, VMIN, VMAX, kurtosis=bad)
        with pytest.raises(ValueError, match="kurtosis"):
            shots_for_squeezing(0.05, TH, VMIN, VMAX, N=NSPIN, contrast=1.0, kurtosis=bad)
