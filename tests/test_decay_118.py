"""1.18: the Voigt-line Ramsey fit and the stretched-exponential echo fit.

Each result is checked against a closed form, an independent numerical
calculation inside the test, or the scatter of seeded synthetic datasets.
"""
import numpy as np
import pytest
from scipy import integrate, optimize, stats

import cavsqueeze as cs
from cavsqueeze import fit_coherence_decay, fit_stretched_decay

F = 2 * np.pi * 150.0          # total FWHM of the line, rad/s
T2 = 5e-3                      # s
T_RAMSEY = np.linspace(0.2e-3, 6e-3, 14)


def _voigt_ramsey(t, line, amplitude=0.9):
    """Closed form: the Ramsey decay of a Voigt line is the product of the
    characteristic functions of its Gaussian and Lorentzian parts (the
    Fourier transform of a convolution), times the dephasing."""
    return amplitude * np.exp(-t / T2 - line.gamma * t - 0.5 * (line.sigma * t) ** 2)


# ---------------------------------------------------------------------------
# Voigt line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.1, 0.3, 0.6])
def test_voigt_fit_recovers_the_line_from_the_closed_form(frac):
    line = cs.lineshape("voigt", F, frac)
    y = _voigt_ramsey(T_RAMSEY, line)
    r = fit_coherence_decay(T_RAMSEY, y, np.full(T_RAMSEY.size, 0.01), line="voigt", T2=T2)
    assert r.fwhm == pytest.approx(F, rel=1e-9)
    assert r.lorentz_fraction == pytest.approx(frac, abs=1e-9)
    assert r.fwhm_lorentzian == pytest.approx(2 * line.gamma, rel=1e-9)
    assert r.fwhm_gaussian == pytest.approx(2 * np.sqrt(2 * np.log(2)) * line.sigma, rel=1e-9)
    assert r.chi2 < 1e-12 and r.dof == T_RAMSEY.size - 3


def test_voigt_fit_on_a_numerical_fourier_transform_of_the_line():
    """The decay computed by numerical Fourier integration of the package's
    own Voigt profile (no product formula involved) gives back the FWHM and
    lorentz_fraction that lineshape() was given."""
    frac = 0.4
    line = cs.lineshape("voigt", F, frac)
    cf = np.array([2 * integrate.quad(lambda d: float(line.pdf(d)), 0.0, np.inf,
                                      weight="cos", wvar=t, epsabs=1e-13)[0]
                   for t in T_RAMSEY])
    y = 0.9 * np.exp(-T_RAMSEY / T2) * cf
    r = fit_coherence_decay(T_RAMSEY, y, np.full(T_RAMSEY.size, 0.01), line="voigt", T2=T2)
    assert r.fwhm == pytest.approx(F, rel=1e-7)
    assert r.lorentz_fraction == pytest.approx(frac, abs=1e-7)


def test_voigt_error_bars_match_the_scatter():
    """300 seeded datasets; the given T2 is itself drawn with its error bar
    (0.2 ms), which the fit must propagate.  The scatter of the fitted FWHM
    and lorentz_fraction is 0.8 to 1.2 times the mean reported error bar,
    and the mean chi2 is its degrees of freedom within 3 standard errors."""
    frac, T2_sigma, err = 0.3, 0.2e-3, 0.005
    line = cs.lineshape("voigt", F, frac)
    rng = np.random.default_rng(11)
    out = []
    for _ in range(300):
        y = _voigt_ramsey(T_RAMSEY, line) + rng.normal(0.0, err, T_RAMSEY.size)
        r = fit_coherence_decay(T_RAMSEY, y, np.full(T_RAMSEY.size, err), line="voigt",
                                T2=T2 + rng.normal(0.0, T2_sigma), T2_sigma=T2_sigma)
        out.append((r.fwhm, r.fwhm_sigma, r.lorentz_fraction, r.lorentz_fraction_sigma, r.chi2))
    fw, sfw, fr, sfr, chi2 = np.array(out).T
    assert 0.8 < np.std(fw) / np.mean(sfw) < 1.2
    assert 0.8 < np.std(fr) / np.mean(sfr) < 1.2
    assert abs(np.mean(fw) - F) < 3 * np.std(fw) / np.sqrt(fw.size) + 0.01 * np.std(fw)
    dof = T_RAMSEY.size - 3
    assert abs(np.mean(chi2) - dof) < 3 * np.sqrt(2 * dof / chi2.size)


def test_voigt_refusals():
    line = cs.lineshape("voigt", F, 0.3)
    y = _voigt_ramsey(T_RAMSEY, line)
    s = np.full(T_RAMSEY.size, 0.01)
    with pytest.raises(ValueError, match="cannot be separated"):
        fit_coherence_decay(T_RAMSEY, y, s, line="voigt")
    gauss = cs.lineshape("gaussian", F)
    yg = 0.9 * np.exp(-T_RAMSEY / T2 - 0.5 * (gauss.std() * T_RAMSEY) ** 2)
    with pytest.raises(ValueError, match="line='gaussian'"):
        fit_coherence_decay(T_RAMSEY, yg, s, line="voigt", T2=T2)
    yl = 0.9 * np.exp(-T_RAMSEY / T2 - 0.5 * F * T_RAMSEY)
    with pytest.raises(ValueError, match="line='lorentzian'"):
        fit_coherence_decay(T_RAMSEY, yl, s, line="voigt", T2=T2)


# ---------------------------------------------------------------------------
# Stretched exponential echo
# ---------------------------------------------------------------------------

T_ECHO = np.linspace(0.5e-3, 20e-3, 15)
T_M = 8e-3


@pytest.mark.parametrize("x", [0.7, 1.0, 1.5, 2.4])
def test_stretched_fit_recovers_the_closed_form(x):
    y = 0.85 * np.exp(-(T_ECHO / T_M) ** x)
    r = fit_stretched_decay(T_ECHO, y, np.full(T_ECHO.size, 0.01))
    assert r.T_M == pytest.approx(T_M, rel=1e-9)
    assert r.exponent == pytest.approx(x, abs=1e-9)
    assert r.amplitude == pytest.approx(0.85, abs=1e-9)
    assert r.chi2 < 1e-12
    if x == 1.0:
        assert r.p_exponential == pytest.approx(1.0, abs=1e-6)
    else:
        assert r.p_exponential < 1e-10


def test_exponential_reference_is_an_independent_fit():
    """chi2_exponential equals the minimum of the weighted residual sum of
    A exp(-t/T2), found here by SciPy's curve_fit (a separate fitter)."""
    rng = np.random.default_rng(3)
    y = 0.85 * np.exp(-(T_ECHO / T_M) ** 1.3) + rng.normal(0.0, 0.01, T_ECHO.size)
    s = np.full(T_ECHO.size, 0.01)
    r = fit_stretched_decay(T_ECHO, y, s)
    popt, _ = optimize.curve_fit(lambda t, a, k: a * np.exp(-k * t), T_ECHO, y,
                                 p0=[0.8, 100.0], sigma=s, absolute_sigma=True,
                                 xtol=1e-14, ftol=1e-14, gtol=1e-14)
    chi2 = np.sum(((popt[0] * np.exp(-popt[1] * T_ECHO) - y) / s) ** 2)
    assert r.chi2_exponential == pytest.approx(chi2, rel=1e-7)
    assert r.delta_chi2 == pytest.approx(chi2 - r.chi2, rel=1e-6)


def test_stretched_error_bars_and_the_test_of_x_equal_one():
    """Exponential data (x = 1), 200 seeds: the scatter of T_M and x is 0.8
    to 1.2 times the reported error bars, and p_exponential < 0.05 for 1 %
    to 10 % of the datasets (5 % expected).  Data with x = 1.8, 100 seeds:
    the error bars hold too, and every dataset rejects x = 1."""
    s = np.full(T_ECHO.size, 0.01)
    for x, n, check_p in ((1.0, 200, True), (1.8, 100, False)):
        rng = np.random.default_rng(int(10 * x))
        rows = []
        for _ in range(n):
            y = 0.85 * np.exp(-(T_ECHO / T_M) ** x) + rng.normal(0.0, 0.01, T_ECHO.size)
            r = fit_stretched_decay(T_ECHO, y, s)
            rows.append((r.T_M, r.T_M_sigma, r.exponent, r.exponent_sigma, r.p_exponential))
        tm, stm, ex, sex, pe = np.array(rows).T
        assert 0.8 < np.std(tm) / np.mean(stm) < 1.2
        assert 0.8 < np.std(ex) / np.mean(sex) < 1.2
        assert abs(np.mean(ex) - x) < 3 * np.std(ex) / np.sqrt(n)
        frac = np.mean(pe < 0.05)
        if check_p:
            assert 0.01 <= frac <= 0.10
            # the drop in chi2 follows the chi-square law with 1 dof
            assert stats.kstest(pe, "uniform").pvalue > 1e-3
        else:
            assert frac == 1.0


def test_stretched_refusals():
    s = np.full(T_ECHO.size, 0.01)
    y = 0.85 * np.exp(-(T_ECHO / T_M) ** 1.5)
    with pytest.raises(ValueError):
        fit_stretched_decay(T_ECHO[:3], y[:3], s[:3])
    with pytest.raises(ValueError):
        fit_stretched_decay(T_ECHO, y, -s)
    with pytest.raises(ValueError):
        fit_stretched_decay(T_ECHO, y, s[:-1])
    # times far shorter than T_M: the decay is barely sampled and the
    # exponent is not resolved
    short = np.linspace(1e-6, 2e-5, 10)
    with pytest.raises(ValueError):
        fit_stretched_decay(short, 0.85 * np.exp(-(short / T_M) ** 1.5) +
                            np.random.default_rng(0).normal(0, 0.01, 10), np.full(10, 0.01))
