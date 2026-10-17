"""1.18: each test here fails on 1.17.0 and passes on 1.18.0 (the bugs and
the numbers before and after are in CHANGELOG.md)."""
import os
import warnings
from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy.special import gammaln

import cavsqueeze as cs
from cavsqueeze.cumulant import Rates, evolve, evolve_meanfield
from cavsqueeze.protocols import css_x


# ---------------------------------------------------------------------------
# oat_closed_form at large N
# ---------------------------------------------------------------------------

def _decimal_cos_sin(x):
    """cos and sin of a Decimal by their Taylor series (|x| < 1)."""
    c = s = Decimal(0)
    term_c, term_s = Decimal(1), x
    k = 0
    while True:
        c += term_c
        s += term_s
        term_c = -term_c * x * x / ((2 * k + 1) * (2 * k + 2))
        term_s = -term_s * x * x / ((2 * k + 2) * (2 * k + 3))
        k += 1
        if abs(term_c) < Decimal(10) ** -70 and abs(term_s) < Decimal(10) ** -70:
            return c, s


def _kitagawa_ueda_decimal(N, mu):
    """The same closed form, written out in 60-digit decimal arithmetic:
    an independent evaluation that has no rounding problem at large N."""
    getcontext().prec = 60
    m = Decimal(mu)                       # the float, exactly
    c, s = _decimal_cos_sin(m / 2)
    cos_mu = c * c - s * s
    A = 1 - cos_mu ** (N - 2)
    B = 4 * s * c ** (N - 2)
    rad = (A * A + B * B).sqrt()
    Vmin = Decimal(N) / 4 + Decimal(N) * (N - 1) / 16 * (A - rad)
    Jx = Decimal(N) / 2 * c ** (N - 1)
    return float(Jx), float(Vmin)


def _dicke_brute_force(N, mu):
    """Exact one-axis twisting in the (N + 1)-dimensional Dicke basis:
    |CSS_x> = sum_k sqrt(binom(N, k) / 2^N) |m = k - N/2>, then
    exp(-i mu Jz^2 / 2), and the moments from J+ matrix elements."""
    j = N / 2.0
    m = np.arange(N + 1) - j
    amp = np.exp(0.5 * (gammaln(N + 1) - gammaln(np.arange(N + 1) + 1)
                        - gammaln(N - np.arange(N + 1) + 1)) - 0.5 * N * np.log(2.0))
    psi = amp * np.exp(-0.5j * mu * m * m)
    up = np.sqrt(np.maximum(j * (j + 1) - m[:-1] * (m[:-1] + 1), 0.0))
    jp = np.zeros_like(psi)
    jm = np.zeros_like(psi)
    jp[1:] = up * psi[:-1]
    jm[:-1] = up * psi[1:]
    jy = (jp - jm) / 2j
    jz = m * psi
    Jx = np.vdot(psi, 0.5 * (jp + jm)).real
    a = np.vdot(jy, jy).real - np.vdot(psi, jy).real ** 2
    b = np.vdot(jz, jz).real - np.vdot(psi, jz).real ** 2
    c = np.vdot(jy, jz).real - np.vdot(psi, jy).real * np.vdot(psi, jz).real
    return Jx, 0.5 * (a + b) - np.hypot(0.5 * (a - b), c)


@pytest.mark.parametrize("N", [10 ** 7, 10 ** 8, 10 ** 9])
def test_oat_closed_form_is_accurate_for_many_spins(N):
    """1.17.0 lost the smallest variance at large N: at the optimal angle
    (mu = 24^(1/6) (N/2)^(-2/3)) it was off by -4.0e-4 (N = 1e7), +3.9e-2
    (N = 1e8), and for N = 1e9 it was negative (-2251.75; true 275.16)."""
    mu = 24 ** (1 / 6) * (N / 2) ** (-2 / 3)
    r = cs.oat_closed_form(N, mu)
    Jx, Vmin = _kitagawa_ueda_decimal(N, mu)
    assert r["Vmin"] == pytest.approx(Vmin, rel=1e-9)
    assert r["Jx"] == pytest.approx(Jx, rel=1e-12)
    assert r["xi2_S"] == pytest.approx(4 * Vmin / N, rel=1e-9)


def test_oat_closed_form_matches_dicke_brute_force_at_a_million_spins():
    """Exact evolution in the Dicke basis (no closed form involved): 1.17.0
    was off by -6.8e-6 at N = 1e6, mu = 1.348e-4; now within 1e-8."""
    N = 10 ** 6
    for mu in (1.3480061545972783e-4, 2.696e-4):
        Jx, Vmin = _dicke_brute_force(N, mu)
        r = cs.oat_closed_form(N, mu)
        assert r["Vmin"] == pytest.approx(Vmin, rel=1e-8)
        assert r["Jx"] == pytest.approx(Jx, rel=1e-8)


def test_oat_closed_form_small_N_unchanged():
    """For the N of the README and the exact-evolution tests the values are
    the 1.17.0 ones to about 1e-11 of N/4 (the old form is accurate there)."""
    rng = np.random.default_rng(5)
    for N in (2, 3, 5, 8, 40, 100, 1000):
        for mu in rng.uniform(0.0, 7.0, 6):
            r = cs.oat_closed_form(N, mu)
            c, s = np.cos(mu / 2), np.sin(mu / 2)
            A = 1 - np.cos(mu) ** (N - 2)
            B = 4 * s * c ** (N - 2)
            old_vmin = N / 4 + N * (N - 1) / 16 * (A - np.hypot(A, B))
            assert abs(r["Vmin"] - old_vmin) < 1e-10 * N / 4
            assert abs(r["Jx"] - N / 2 * c ** (N - 1)) < 1e-12 * N / 2
    r0 = cs.oat_closed_form(6, 0.0)
    assert r0["Vmin"] == r0["Vmax"] == 1.5 and r0["Jx"] == 3.0


# ---------------------------------------------------------------------------
# Output times of the solvers and the Ramsey functions
# ---------------------------------------------------------------------------

def _free_decay_setup():
    """Negligible interaction and T2 = 10 ms: the Ramsey contrast is
    exactly exp(-t/T2) (pure dephasing of the coherent state)."""
    p = cs.from_hz(g_hz=1e-3, kappa_hz=1e4, Delta_hz=30e6, T=0.0, T2=0.01)
    return p, cs.homogeneous(1e6)


def test_ramsey_times_in_any_order():
    """1.17.0: ramsey_cumulant(p, ens, [1e-3, 0.0]) integrated to the last
    listed time (0) and returned contrast 1.0 at t = 1 ms (true
    exp(-0.1) = 0.904837); ramsey_meanfield raised a SciPy ValueError."""
    p, ens = _free_decay_setup()
    t = [1e-3, 0.0, 5e-4]
    want = np.exp(-np.array(t) / 0.01)
    for f in (cs.ramsey_cumulant, cs.ramsey_meanfield):
        r = f(p, ens, t)
        assert np.allclose(r["contrast"], want, rtol=1e-6, atol=0)
        assert np.array_equal(r["t"], np.array(t))
        # the sorted request gives the same numbers, in sorted order
        r2 = f(p, ens, sorted(t))
        assert np.array_equal(r2["contrast"], np.sort(r["contrast"])[::-1])


def test_ramsey_at_time_zero_only():
    """1.17.0: ramsey_meanfield(p, ens, [0.0]) and evolve_meanfield with
    t = 0 raised TypeError; now they return the initial state."""
    p, ens = _free_decay_setup()
    assert cs.ramsey_meanfield(p, ens, [0.0])["contrast"][0] == 1.0
    assert cs.ramsey_cumulant(p, ens, [0.0])["contrast"][0] == 1.0
    rt = Rates.from_params(p, ens)
    s, z = evolve_meanfield(np.array([0.5 + 0j]), np.array([0j]), rt, 0.0)
    assert s[0] == 0.5 and z[0] == 0.0


def test_output_times_outside_the_span_are_refused():
    """1.17.0: evolve(st, rt, 0.0, t_eval=[0.5]) returned the t = 0 state
    (contrast 1.0) as the state at t = 0.5 s, where the contrast is
    exp(-50) = 2e-22.  Now ValueError, for both solvers."""
    p, ens = _free_decay_setup()
    rt = Rates.from_params(p, ens)
    with pytest.raises(ValueError, match="t_eval"):
        evolve(css_x(1), rt, 0.0, t_eval=[0.5])
    with pytest.raises(ValueError, match="t_eval"):
        evolve(css_x(1), rt, 1e-3, t_eval=[0.0, 2e-3])
    with pytest.raises(ValueError, match="t_eval"):
        evolve_meanfield(np.array([0.5 + 0j]), np.array([0j]), rt, 0.0, t_eval=[0.5])
    with pytest.raises(ValueError, match="t_list"):
        cs.ramsey_meanfield(p, ens, [0.0, np.nan])
    with pytest.raises(ValueError, match="t_list"):
        cs.ramsey_cumulant(p, ens, [-1e-3, 0.0])


# ---------------------------------------------------------------------------
# Shot records, error bars and the Dick floor
# ---------------------------------------------------------------------------

def test_saved_records_load_back_exactly(tmp_path):
    """1.17.0: save_shots_csv dropped an angle with no shots without a word
    (3 angles saved, 2 loaded back), and wrote one-shot angles and nan
    angles that load_shots_csv then refused.  Now each is refused when
    saving; a valid record still round-trips exactly."""
    path = os.path.join(tmp_path, "shots.csv")
    for angles, shots in [([0.0, 1.0, 2.0], [[1.0, 2.0], [], [3.0, 4.0]]),
                          ([0.0, 1.0], [[1.0, 2.0], [5.0]]),
                          ([0.0, np.nan], [[1.0, 2.0], [5.0, 6.0]]),
                          ([0.0, 1.0], [[1.0, np.inf], [5.0, 6.0]])]:
        with pytest.raises(ValueError):
            cs.save_shots_csv(path, angles, shots)
    rng = np.random.default_rng(0)
    angles = [0.1, 0.7, 2.9]
    shots = [rng.normal(size=k) for k in (2, 5, 3)]
    cs.save_shots_csv(path, angles, shots)
    a, s = cs.load_shots_csv(path)
    assert np.array_equal(a, angles)
    assert all(np.array_equal(x, y) for x, y in zip(s, shots))


def test_sample_variance_sigma_refuses_an_impossible_kurtosis():
    """1.17.0: kurtosis = 0.5 or nan returned nan (RuntimeWarning only).
    Every distribution has kurtosis >= 1, with equality for a symmetric
    two-point distribution, whose sample variance of M shots then has
    Var(s^2) = sigma^4 [1/M - (M-3)/(M(M-1))] = 2 sigma^4 / (M (M - 1))."""
    x = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    for bad in (0.5, 0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="kurtosis"):
            cs.sample_variance_sigma(x, kurtosis=bad)
    m = x.size
    s2 = np.var(x, ddof=1)
    assert cs.sample_variance_sigma(x, kurtosis=1.0) == pytest.approx(
        s2 * np.sqrt(2.0 / (m * (m - 1))), rel=1e-14)
    # the measured kurtosis of +-1 shots is exactly 1, the same answer
    assert cs.sample_variance_sigma(x) == pytest.approx(
        cs.sample_variance_sigma(x, kurtosis=1.0), rel=1e-14)


def test_dick_floor_refuses_bad_spectra_and_settings():
    """1.17.0 returned nan for a spectrum with nan or negative values,
    ignored the cutoff check for tail_tol = nan, and truncated a fractional
    m_max (2.7 harmonics -> 2)."""
    args = (1.0, 0.4, 1.0)
    with pytest.raises(ValueError, match="S_y"):
        cs.dick_allan_deviation(lambda f: np.full_like(f, np.nan), *args)
    with pytest.raises(ValueError, match="S_y"):
        cs.dick_allan_deviation(lambda f: -1e-30 + 0.0 * f, *args)
    with pytest.raises(ValueError, match="tail_tol"):
        cs.dick_allan_deviation(cs.power_law_psd(h0=1e-30), *args, tail_tol=np.nan)
    with pytest.raises(ValueError, match="m_max"):
        cs.dick_fourier_coefficients(2.7, 0.4, 1.0)
    assert cs.dick_fourier_coefficients(3.0, 0.4, 1.0)[1].size == 3


def test_dick_cutoff_check_cannot_land_on_zero_coefficients():
    """For T_ramsey / T_cycle = 0.4 and instantaneous pulses the
    coefficients vanish at m = 5, 10, 15, ...  With m_max = 10, 1.17.0
    checked only m = 10 (exactly zero) and returned 8.476e-16, 2.1 % below
    the converged 8.660e-16 for h0 = 1e-30 (the closed form for white
    noise, h0 (1 - eta) / (2 eta tau), gives 8.660e-16).  Now refused;
    the default cutoff is checked on the same harmonics as before."""
    laser = cs.power_law_psd(h0=1e-30)
    with pytest.raises(ValueError, match="m_max = 10"):
        cs.dick_allan_deviation(laser, 1.0, 0.4, 1.0, m_max=10)
    eta = 0.4
    closed = np.sqrt(1e-30 * (1 - eta) / (2 * eta))
    assert cs.dick_allan_deviation(laser, 1.0, 0.4, 1.0) == pytest.approx(closed, rel=6e-5)


def test_clear_refusals_instead_of_numpy_errors():
    """1.17.0 raised numpy's LinAlgError for a nan theta_min in
    plan_tomography and ZeroDivisionError for M_core = 0 in
    tail_resolved_classes; both are now ValueError naming the input."""
    from cavsqueeze.ensemble import tail_resolved_classes
    angles = np.linspace(0.0, np.pi, 9, endpoint=False)
    with pytest.raises(ValueError, match="theta_min"):
        cs.plan_tomography(angles, 50, 90.0, 160.0, theta_min=np.nan)
    with pytest.raises(ValueError, match="theta_min"):
        cs.shots_for_squeezing(0.02, angles, 90.0, 160.0, N=480, contrast=0.92,
                               theta_min=np.inf)
    line = cs.lineshape("lorentzian", 1.0)
    with pytest.raises(ValueError, match="M_core"):
        tail_resolved_classes(line, 100.0, fwhm=1.0, M_core=0)
    with pytest.raises(ValueError, match="core_edge"):
        tail_resolved_classes(line, 100.0, fwhm=1.0, core_edge=2000.0)
    ens = tail_resolved_classes(line, 100.0, fwhm=1.0)
    assert ens.N == pytest.approx(100.0, rel=1e-12)
