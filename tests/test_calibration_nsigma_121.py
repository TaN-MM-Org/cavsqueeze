"""1.21: errors of the atom numbers in `calibrate_projection_noise`.

References, none of them the function under test:

* NumPy's `polyfit` at the weights the fit converged to, and those weights
  rebuilt here from the fitted curve, sqrt(2/(M-1)) and N_sigma;
* binomial shots of N_true independent spins (Var = N_true/4 exactly) with
  the reported atom numbers drawn around N_true with their error bar, and
  the scatter of the fitted coefficients over 200 seeds.
"""
import numpy as np
import pytest

from cavsqueeze import calibrate_projection_noise

N_TRUE = np.arange(500.0, 4001.0, 500.0)
M = 2000
REL = 0.03                      # 3 % error of each reported atom number
DET_SD = 3.0                    # detection noise: c0 = 9


def _record(seed):
    rng = np.random.default_rng(seed)
    shots = [rng.binomial(int(n), 0.5, M) - n / 2 + rng.normal(0.0, DET_SD, M)
             for n in N_TRUE]
    N_meas = N_TRUE + REL * N_TRUE * rng.normal(size=N_TRUE.size)
    return N_meas, shots


@pytest.mark.parametrize("quadratic", [False, True])
def test_weights_and_coefficients_against_polyfit(quadratic):
    """At convergence each point's error is
    sqrt((curve sqrt(2/(M-1)))^2 + ((c1 + 2 c2 N) N_sigma)^2), rebuilt here
    from the returned coefficients (1e-9 relative), and the coefficients and
    covariance equal NumPy's polyfit at those weights (1e-9 of each error
    bar)."""
    N_meas, shots = _record(11)
    sN = REL * N_TRUE
    fit = calibrate_projection_noise(N_meas, shots, quadratic=quadratic, N_sigma=sN)
    c = fit.values
    curve = sum(ck * N_meas ** k for k, ck in enumerate(c))
    slope = c[1] + (2 * c[2] * N_meas if quadratic else 0.0)
    expected = np.hypot(curve * np.sqrt(2.0 / (M - 1)), slope * sN)
    assert np.max(np.abs(fit.variance_sigmas / expected - 1)) < 1e-9   # measured 8e-13
    s2 = np.array([np.var(s, ddof=1) for s in shots])
    deg = 2 if quadratic else 1
    coef, cov = np.polyfit(N_meas, s2, deg, w=1.0 / fit.variance_sigmas, cov="unscaled")
    err = np.sqrt(np.diag(fit.cov))
    assert np.max(np.abs(coef[::-1] - c) / err) < 1e-9                 # measured 1.3e-13
    assert np.max(np.abs(cov[::-1, ::-1] - fit.cov) / np.outer(err, err)) < 1e-9   # measured 1.5e-14
    assert np.array_equal(fit.N_sigma, sN)


def test_no_N_sigma_gives_the_earlier_results():
    """N_sigma=None and N_sigma=0 give exactly the fit of 1.20."""
    N_meas, shots = _record(3)
    for quad in (False, True):
        for noise in ("gaussian", "empirical"):
            a = calibrate_projection_noise(N_meas, shots, quadratic=quad, shot_noise=noise)
            b = calibrate_projection_noise(N_meas, shots, quadratic=quad, shot_noise=noise,
                                           N_sigma=0.0)
            assert np.array_equal(a.values, b.values) and np.array_equal(a.cov, b.cov)
            assert a.chi2 == b.chi2 and a.N_sigma is None and b.N_sigma is None


@pytest.mark.parametrize("quadratic", [False, True])
def test_error_bars_match_the_scatter_with_atom_number_errors(quadratic):
    """200 seeded calibrations, reported atom numbers off by 3 % (rms).
    With N_sigma the scatter of c0 and c1 is 0.8 to 1.2 times the reported
    error bar (line: 0.95 and 1.00; parabola: 0.89 and 0.91), with no bias
    beyond 3 standard errors, and 1 to 22 of the 200 fit p-values are below
    0.05 (7 and 12; 10 expected).  Without it the error bar of c1 is too
    small (scatter 1.37 and 1.24 times it) and the fit test rejects 66 and
    59 of the 200 correct calibrations."""
    with_, without = [], []
    for seed in range(200):
        N_meas, shots = _record(seed)
        with_.append(calibrate_projection_noise(N_meas, shots, quadratic=quadratic,
                                                N_sigma=REL * N_TRUE))
        without.append(calibrate_projection_noise(N_meas, shots, quadratic=quadratic))

    def stats(fits, k):
        v = np.array([f.values[k] for f in fits])
        s = np.array([np.sqrt(f.cov[k, k]) for f in fits])
        return v, v.std(ddof=1) / np.sqrt(np.mean(s ** 2))

    for k, truth in ((0, DET_SD ** 2), (1, 0.25)):
        v, ratio = stats(with_, k)
        assert 0.8 < ratio < 1.2, (k, ratio)        # measured 0.89 to 1.00
        assert abs(v.mean() - truth) < 3 * v.std(ddof=1) / np.sqrt(v.size)
    assert stats(without, 1)[1] > 1.15              # measured 1.37 and 1.24
    p_with = np.array([f.p_value for f in with_])
    p_without = np.array([f.p_value for f in without])
    assert 1 <= np.sum(p_with < 0.05) <= 22         # measured 7 and 12
    assert np.sum(p_without < 0.05) > 40            # measured 66 and 59


def test_N_sigma_refusals():
    N_meas, shots = _record(0)
    for bad in (-1.0, np.nan, np.inf, np.ones(3), np.ones((2, 8))):
        with pytest.raises(ValueError, match="N_sigma"):
            calibrate_projection_noise(N_meas, shots, N_sigma=bad)
