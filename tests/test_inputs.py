"""1.16: the Voigt line has exactly the requested width, and inputs that
used to give silent nan, negative or unphysical results are refused."""
import numpy as np
import pytest
from scipy import integrate, optimize, stats

import cavsqueeze as cs
from cavsqueeze import dtwa


def _convolved_pdf(x, sigma, gamma):
    """Voigt profile by direct numerical convolution of a Gaussian and a
    Lorentzian -- an independent route to scipy.special.voigt_profile."""
    f = lambda u: stats.norm.pdf(u, scale=sigma) * stats.cauchy.pdf(x - u, scale=gamma)
    return integrate.quad(f, -np.inf, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]


@pytest.mark.parametrize("eta", [0.1, 0.3, 0.6, 0.9])
def test_voigt_total_width_is_exact(eta):
    fwhm = 2.0
    v = cs.lineshape("voigt", fwhm, eta)
    assert v.gamma == pytest.approx(eta * fwhm / 2, rel=1e-15)
    ratio = _convolved_pdf(fwhm / 2, v.sigma, v.gamma) / _convolved_pdf(0.0, v.sigma, v.gamma)
    assert abs(ratio - 0.5) < 1e-9
    # the pre-1.16 Olivero-Longbothum split misses the width by up to ~2.3e-4
    old = cs.lineshape("voigt", fwhm, eta, split="olivero")
    half = 0.5 * float(old.pdf(0.0))
    x = optimize.brentq(lambda x: float(old.pdf(x)) - half, 0.0, fwhm, xtol=1e-14)
    assert 1e-6 < abs(2 * x / fwhm - 1) < 2.5e-4


def test_voigt_limits_are_the_pure_lines():
    fwhm = 3.0
    g = cs.lineshape("voigt", fwhm, 0.0)
    assert g.sigma == pytest.approx(fwhm / (2 * np.sqrt(2 * np.log(2))), rel=1e-15)
    assert g.gamma == 0.0
    lo = cs.lineshape("voigt", fwhm, 1.0)
    assert lo.sigma == 0.0 and lo.gamma == fwhm / 2
    # the tabulated quantile function of the pure Lorentzian limit
    q = np.array([0.1, 0.3, 0.45])
    assert np.allclose(lo.ppf(q), stats.cauchy.ppf(q, scale=fwhm / 2), rtol=1e-6)


def test_voigt_cdf_matches_quadrature_of_the_pdf():
    v = cs.lineshape("voigt", 1.0, 0.4)
    for q in (0.05, 0.25, 0.4):
        x = v.ppf(q)
        mass = 0.5 - integrate.quad(lambda t: float(v.pdf(t)), x, 0.0, epsabs=1e-13)[0]
        assert abs(mass - q) < 1e-7


@pytest.mark.parametrize("call", [
    lambda: cs.lineshape("gaussian", -1.0),
    lambda: cs.lineshape("lorentzian", np.nan),
    lambda: cs.lineshape("voigt", 1.0, 1.5),
    lambda: cs.lineshape("voigt", 1.0, -0.2),
    lambda: cs.lineshape("voigt", 1.0, 0.3, split="guess"),
    lambda: cs.lineshape("cauchy", 1.0),
    lambda: cs.Ensemble(delta=[0.0, 1.0], weight=[1.0], n=[1.0, 1.0]),
    lambda: cs.Ensemble(delta=[0.0, 1.0], weight=[1.0, 1.0], n=[1.0, -1.0]),
    lambda: cs.Ensemble(delta=[0.0, np.nan], weight=[1.0, 1.0], n=[1.0, 1.0]),
    lambda: cs.homogeneous(0.0),
    lambda: cs.equal_probability_classes(cs.lineshape("gaussian", 1.0), 0, 10.0),
    lambda: cs.equal_probability_classes(cs.lineshape("gaussian", 1.0), 4, 10.0, weights=[1.0, 1.0]),
    lambda: cs.product_classes(cs.lineshape("gaussian", 1.0), 2, [1.0, 2.0], [1.0, -0.5], 10.0),
    lambda: cs.from_hz(1.0, 1e4, 3e7, T2=-1.0),
    lambda: cs.from_hz(1.0, 1e4, 3e7, T2=0.0),
    lambda: cs.from_hz(1.0, -1e4, 3e7),
    lambda: cs.from_hz(1.0, 0.0, 0.0),
    lambda: cs.from_hz(1.0, 1e4, 3e7, T=-0.1),
    lambda: cs.CavityParams(g=np.nan, kappa=1.0, Delta=1.0),
    lambda: cs.CavityParams(g=1.0, kappa=1.0, Delta=1.0, gamma_phi=-1.0),
    lambda: cs.CavityParams(g=1.0, kappa=1.0, Delta=1.0, omega_s=0.0),
    lambda: cs.thermal_occupation(-1.0, 1.0),
    lambda: cs.clock_allan_deviation(1e-3, 4.29e14, 0.1, tau=-1.0),
    lambda: cs.clock_allan_deviation(1e-3, 4.29e14, 0.1, tau=0.0),
    lambda: cs.clock_allan_deviation(1e-3, 4.29e14, -0.1, tau=1.0),
    lambda: cs.clock_allan_deviation(np.inf, 4.29e14, 0.1, tau=1.0),
    lambda: cs.clock_allan_deviation(1e-3, 0.0, 0.1, tau=1.0),
    lambda: cs.magnetometer_sensitivity(1e-3, 2e9, 0.1, tau=-1.0),
    lambda: cs.magnetometer_sensitivity(1e-3, 0.0, 0.1),
    lambda: cs.metrological_gain_db(-0.5),
    lambda: cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), -1.0, 0.4, 1.0),
    lambda: cs.evolve(cs.css_x(1), cs.Rates.from_params(cs.from_hz(1e2, 1e4, 3e7), cs.homogeneous(1e8)), -1e-4),
    lambda: cs.optimal_squeezing(cs.from_hz(1e2, 1e4, 3e7), cs.homogeneous(1e8), 1e-3, 1e-6),
    lambda: dtwa.evolve(np.zeros(2), np.ones(2), 0.1, [np.nan]),
])
def test_refusals(call):
    with pytest.raises(ValueError):
        call()


def test_valid_edge_inputs_still_work():
    """The limits that were accepted before stay accepted and unchanged."""
    p = cs.from_hz(1.0, 1e4, 3e7, T=0.0, T2=np.inf)
    assert p.gamma_phi == 0.0 and p.n_th == 0.0
    assert cs.from_hz(1.0, 1e4, 3e7, T2=None).gamma_phi == 0.0
    assert cs.from_hz(1.0, 0.0, 3e7).Gamma_SR == 0.0          # lossless cavity
    assert cs.thermal_occupation(-1.0, 0.0) == 0.0            # T = 0: no photons
    # a negative gyromagnetic ratio gives the same field noise as |gamma|
    assert cs.magnetometer_sensitivity(1e-3, -2e9, 0.1) \
        == cs.magnetometer_sensitivity(1e-3, 2e9, 0.1)
    assert cs.metrological_gain_db(np.inf) == -np.inf
    st = cs.css_x(1)
    assert cs.evolve(st, cs.Rates.from_params(p, cs.homogeneous(10.0)), 0.0).s[0] == st.s[0]
