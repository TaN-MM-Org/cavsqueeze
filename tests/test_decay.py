"""1.17: T2 and the line width from measured echo and Ramsey decays.

* Noise-free decays built from the closed forms are recovered exactly
  (to 1e-9), with chi2 ~ 0.
* Independent calculations: the echo decay computed by the cumulant
  solver (with a Gaussian line, refocused by the echo) and the Ramsey
  decay computed by the mean-field solver on a Gaussian line cut into
  20000 classes are fitted, and the fit returns the T2 and FWHM that
  went into the solvers.  This ties the fitted numbers to the inputs of
  `from_hz` and `lineshape` in the package's own conventions.
* Seeded Monte Carlo: the scatter of the fitted values matches the
  reported error bars, the p-values are uniform, and a known T2 with an
  error bar is propagated correctly (without it the width error bar is
  1.5 times too small).
* The linearized propagation of T2_sigma equals a finite-difference
  refit.
* Refusals.
"""
import numpy as np
import pytest

import cavsqueeze as cs
from cavsqueeze import fit_coherence_decay

K = 2.0 * np.sqrt(2.0 * np.log(2.0))
T2 = 5e-3
F = 2 * np.pi * 150.0           # Gaussian FWHM, rad/s
FL = 2 * np.pi * 100.0          # Lorentzian FWHM, rad/s
A = 0.9
T = np.linspace(0.1e-3, 4e-3, 15)


def _gauss(t, A=A, T2=T2, F=F):
    s = F / K
    return A * np.exp(-t / T2 - 0.5 * s * s * t * t)


def test_noise_free_closed_forms_are_recovered():
    sig = np.full(T.size, 0.01)
    f = fit_coherence_decay(T, A * np.exp(-T / T2), sig)
    assert abs(f.T2 / T2 - 1) < 1e-9 and abs(f.amplitude / A - 1) < 1e-9
    assert f.fwhm is None and f.dof == T.size - 2 and f.chi2 < 1e-15
    f = fit_coherence_decay(T, _gauss(T), sig, line="gaussian")
    assert abs(f.T2 / T2 - 1) < 1e-9 and abs(f.fwhm / F - 1) < 1e-9
    assert abs(f.gamma_phi * f.T2 - 1) < 1e-15 and f.dof == T.size - 3
    f = fit_coherence_decay(T, _gauss(T), sig, line="gaussian", T2=T2)
    assert abs(f.fwhm / F - 1) < 1e-9 and f.T2 == T2 and f.T2_given
    y = A * np.exp(-T / T2 - FL * T / 2)
    f = fit_coherence_decay(T, y, sig, line="lorentzian", T2=T2)
    assert abs(f.fwhm / FL - 1) < 1e-9 and f.chi2 < 1e-15


def test_echo_from_the_cumulant_solver_gives_T2():
    """The cumulant solver, no interaction (g = 0), a 2 kHz Gaussian line
    in 16 classes and T2 = 5 ms: the echo refocuses the line, and the fit
    returns the T2 given to `from_hz`."""
    p = cs.from_hz(0.0, 1e4, 3e7, T2=T2)
    ens = cs.equal_probability_classes(cs.lineshape("gaussian", 2 * np.pi * 2e3), 16, 1e6)
    t = np.linspace(0.5e-3, 1e-2, 10)
    c = np.array([cs.squeezing_after(p, ens, x)["contrast"] for x in t])
    f = fit_coherence_decay(t, c, np.full(t.size, 1e-3))
    assert abs(f.T2 / T2 - 1) < 1e-7
    assert abs(f.amplitude - 1) < 1e-7


def test_ramsey_from_the_meanfield_solver_gives_the_line():
    """The mean-field solver on a Gaussian line of FWHM 2 pi x 150 Hz cut
    into 20000 equal-probability classes, with T2 = 5 ms and no
    interaction: fitting its Ramsey contrast returns the FWHM given to
    `lineshape` and the T2 given to `from_hz` (to 5e-5; the 20000-class
    discretization differs from the continuous line by ~1e-5)."""
    p = cs.from_hz(0.0, 1e4, 3e7, T2=T2)
    ens = cs.equal_probability_classes(cs.lineshape("gaussian", F), 20000, 1e6)
    t = np.linspace(0.0, 4e-3, 15)
    c = cs.ramsey_meanfield(p, ens, t)["contrast"]
    f = fit_coherence_decay(t, c, np.full(t.size, 1e-3), line="gaussian")
    assert abs(f.fwhm / F - 1) < 5e-5
    assert abs(f.T2 / T2 - 1) < 5e-5
    g = fit_coherence_decay(t, c, np.full(t.size, 1e-3), line="gaussian", T2=T2)
    assert abs(g.fwhm / F - 1) < 5e-5


def _monte_carlo(line, n=300, seed=3, **kw):
    rng = np.random.default_rng(seed)
    sig = np.full(T.size, 0.01)
    truth = _gauss(T) if line == "gaussian" else A * np.exp(-T / T2 - FL * T / 2)
    out = []
    for _ in range(n):
        extra = dict(kw)
        if "T2_sigma" in kw:
            extra["T2"] = T2 + rng.normal(0.0, kw["T2_sigma"])
        y = truth + rng.normal(0.0, 0.01, T.size)
        out.append(fit_coherence_decay(T, y, sig, line=line, **extra))
    return out


def test_monte_carlo_error_bars_and_p_values():
    fits = _monte_carlo("gaussian")
    for name, truth in (("fwhm", F), ("gamma_phi", 1 / T2)):
        vals = np.array([getattr(f, name) for f in fits])
        sigs = np.array([getattr(f, name + "_sigma") for f in fits])
        ratio = np.std(vals, ddof=1) / np.mean(sigs)
        assert 0.8 < ratio < 1.2, (name, ratio)
        # no bias beyond 3 standard errors of the mean
        assert abs(np.mean(vals) - truth) < 3 * np.std(vals, ddof=1) / np.sqrt(len(vals))
    chi2 = np.array([f.chi2 for f in fits])
    dof = fits[0].dof
    assert abs(np.mean(chi2) - dof) < 3 * np.sqrt(2 * dof / len(fits))
    frac = np.mean(np.array([f.p_value for f in fits]) < 0.05)
    assert 0.02 < frac < 0.09


def test_lorentzian_monte_carlo():
    fits = _monte_carlo("lorentzian", T2_sigma=2e-4)
    vals = np.array([f.fwhm for f in fits])
    sigs = np.array([f.fwhm_sigma for f in fits])
    assert 0.8 < np.std(vals, ddof=1) / np.mean(sigs) < 1.2
    assert abs(np.mean(vals) - FL) < 3 * np.std(vals, ddof=1) / np.sqrt(len(vals))


def test_known_T2_uncertainty_is_propagated():
    """T2 drawn from N(5 ms, 0.2 ms) in each experiment: with its error
    bar given, the width's error bar matches the scatter; ignoring it,
    the error bar is far too small."""
    with_sig = _monte_carlo("gaussian", T2_sigma=2e-4)
    ratio = np.std([f.fwhm for f in with_sig], ddof=1) / np.mean([f.fwhm_sigma for f in with_sig])
    assert 0.8 < ratio < 1.2
    rng = np.random.default_rng(3)
    vals, sigs = [], []
    for _ in range(300):
        T2m = T2 + rng.normal(0.0, 2e-4)
        y = _gauss(T) + rng.normal(0.0, 0.01, T.size)
        f = fit_coherence_decay(T, y, np.full(T.size, 0.01), line="gaussian", T2=T2m)
        vals.append(f.fwhm)
        sigs.append(f.fwhm_sigma)
    assert np.std(vals, ddof=1) / np.mean(sigs) > 1.3


def test_T2_propagation_equals_a_finite_difference_refit():
    """The linearized response of the fitted line variance to T2 equals
    the change of a full refit when T2 is moved by a small step."""
    rng = np.random.default_rng(7)
    y = _gauss(T) + rng.normal(0.0, 0.01, T.size)
    sig = np.full(T.size, 0.01)
    T2s = 3e-4
    a = fit_coherence_decay(T, y, sig, line="gaussian", T2=T2)
    b = fit_coherence_decay(T, y, sig, line="gaussian", T2=T2, T2_sigma=T2s)
    extra = b.cov[1, 1] - a.cov[1, 1]          # (d s2 / d T2)^2 T2_sigma^2
    h = 1e-7
    up = fit_coherence_decay(T, y, sig, line="gaussian", T2=T2 + h).values[1]
    dn = fit_coherence_decay(T, y, sig, line="gaussian", T2=T2 - h).values[1]
    fd = ((up - dn) / (2 * h) * T2s) ** 2
    assert abs(extra / fd - 1) < 2e-2


def test_misfit_is_flagged():
    """A Gaussian-line Ramsey decay fitted as a pure exponential: the fit
    returns numbers, and the p-value says they do not describe the data."""
    rng = np.random.default_rng(1)
    y = _gauss(T) + rng.normal(0.0, 0.01, T.size)
    f = fit_coherence_decay(T, y, np.full(T.size, 0.01), line="none")
    assert f.p_value < 1e-6


@pytest.mark.parametrize("kw, match", [
    (dict(line="lorentzian"), "cannot be separated"),
    (dict(line="voigt"), "line must be"),
    (dict(line="none", T2=T2), "do not pass T2"),
    (dict(T2_sigma=1e-4), "without T2"),
    (dict(line="gaussian", T2=-1.0), "T2 must be"),
    (dict(line="gaussian", T2=T2, T2_sigma=np.nan), "T2_sigma"),
    (dict(min_significance=-1.0), "min_significance"),
])
def test_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        fit_coherence_decay(T, _gauss(T), np.full(T.size, 0.01), **kw)


def test_data_refusals():
    sig = np.full(T.size, 0.01)
    y = _gauss(T)
    with pytest.raises(ValueError, match="same"):
        fit_coherence_decay(T[:-1], y, sig)
    with pytest.raises(ValueError, match="times"):
        fit_coherence_decay(-T, y, sig)
    with pytest.raises(ValueError, match="contrast_sigma"):
        fit_coherence_decay(T, y, 0 * sig)
    with pytest.raises(ValueError, match="finite"):
        fit_coherence_decay(T, np.where(T > 1e-3, np.nan, y), sig)
    with pytest.raises(ValueError, match="at least 4 points"):
        fit_coherence_decay(T[:3], y[:3], sig[:3], line="gaussian")
    with pytest.raises(ValueError, match="distinct times"):
        fit_coherence_decay(np.full(5, 1e-3), np.full(5, 0.5), np.full(5, 0.01), line="gaussian")
    # a decay that the data do not resolve: 1 % of decay over the window
    t = np.linspace(0.0, 1e-5, 10)
    with pytest.raises(ValueError, match="do not\\s+resolve"):
        fit_coherence_decay(t, A * np.exp(-t / T2), np.full(10, 0.01))
    # a pure exponential fitted with a Gaussian line: no curvature to fit
    with pytest.raises(ValueError, match="Gaussian variance"):
        fit_coherence_decay(T, A * np.exp(-T / T2), sig, line="gaussian")
