"""1.21: the eliminated-cavity model against the full spin-cavity model.

`cavsqueeze.exact.spin_cavity` keeps the cavity mode (photon-number states
up to n_fock) and solves the exact master equation for two spins.  The
model of the rest of the package drops the cavity and keeps its effect
through chi and Gamma_SR (`cavsqueeze.resonator`).  Holding chi g^2 and
Gamma_SR g^2 fixed while the cavity is detuned further (g^2 grows like
Delta), the dropped terms are of order g^2 / Delta^2, so the difference
between the two models must shrink like 1 / Delta.  It does, which checks
the formulas for chi and Gamma_SR, the sign of Delta and the thermal-photon
terms.  The differences quoted are the largest over all first and pair
moments of the spins at 7 times (0 to 3 in units of 1/(chi g^2)).
"""
import numpy as np
import pytest

pytest.importorskip("qutip", reason="the exact references need QuTiP")

import cavsqueeze as cs
from cavsqueeze.cumulant_raw import Rates
from cavsqueeze.ensemble import Ensemble
from cavsqueeze.exact import full_hilbert, spin_cavity
from cavsqueeze.resonator import HBAR, KB, CavityParams

KEYS = ("s", "z", "P", "Q", "R", "Z")
DELTA = np.array([0.3, -0.2])          # spin detunings
W = np.array([0.9, 1.1])               # coupling weights
TIMES = np.linspace(0.0, 3.0, 7)
CSS = [1.0, 0.0, 0.0]
OMEGA_S = 1e3                          # sets n_th through the temperature


def _params(D, kappa_over_D, n_th, sign=1.0):
    """Cavity detuned by D with chi g^2 = 1, so Gamma_SR g^2 = kappa / D."""
    kap = kappa_over_D * D
    g = np.sqrt((4 * D ** 2 + kap ** 2) / (4 * D))
    T = 0.0 if n_th == 0 else HBAR * OMEGA_S / (KB * np.log1p(1.0 / n_th))
    return CavityParams(g=g, kappa=kap, Delta=sign * D, omega_s=OMEGA_S, T=T, gamma_phi=0.05)


def _diff(a, b):
    return max(float(np.max(np.abs(a[k] - b[k]))) for k in KEYS)


def _eliminated(p, **change):
    rt = Rates.from_params(p, Ensemble(delta=DELTA, weight=W, n=np.ones(2)))
    for name, value in change.items():
        setattr(rt, name, value)
    return rt, full_hilbert(rt, CSS, TIMES)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_chi_has_the_sign_of_omega_s_minus_omega_c(sign):
    """With Delta = omega_s - omega_c (spins above the cavity for Delta > 0)
    the eliminated model follows the full one: difference 0.028 (+Delta) and
    0.031 (-Delta) at Delta = 80 chi g^2, kappa = 0.2 Delta, no thermal
    photons.  With the sign of chi flipped (Delta read as omega_c - omega_s,
    as the CavityParams docstring said before 1.21) it misses by 0.54."""
    p = _params(80.0, 0.2, 0.0, sign)
    full = spin_cavity(p, DELTA, W, CSS, TIMES, n_fock=6)
    assert full["p_top"] < 1e-9                     # the photon cutoff does not matter
    rt, model = _eliminated(p)
    assert _diff(full, model) < 0.05                 # measured 0.028 and 0.031
    _, flipped = _eliminated(p, chi1=-rt.chi1)
    assert _diff(full, flipped) > 0.4                # measured 0.54


def test_elimination_error_shrinks_like_one_over_delta_with_thermal_photons():
    """kappa = Delta and n_th = 0.2 (so Gamma_SR g^2 = 0.8, a strong decay
    and absorption): the difference falls 0.0216, 0.0118, 0.0062 for
    Delta = 40, 80, 160 (ratios 1.83 and 1.91).  Before 1.21 the thermal
    shift 2 n_th chi g_j^2 of each spin was missing, and the difference
    stayed at 0.060, 0.063, 0.065: the eliminated model did not approach the
    full one.  Leaving the shift out here reproduces that."""
    errs, errs_without = [], []
    for D in (40.0, 80.0, 160.0):
        p = _params(D, 1.0, 0.2)
        full = spin_cavity(p, DELTA, W, CSS, TIMES, n_fock=10)
        assert full["p_top"] < 1e-5                  # measured 1.2e-7 or less
        rt, model = _eliminated(p)
        errs.append(_diff(full, model))
        # the 1.20 rates: the line detunings only
        _, old = _eliminated(p, delta=DELTA.copy())
        errs_without.append(_diff(full, old))
    assert errs[-1] < 0.01                           # measured 0.0062
    for a, b in zip(errs, errs[1:]):
        assert 1.6 < a / b < 2.4                     # measured 1.83 and 1.91
    assert min(errs_without) > 0.05                  # measured 0.060 to 0.065


def test_thermal_shift_enters_the_detunings():
    """Rates.from_params adds 2 n_th chi1 G^2 to each class detuning, and
    nothing at zero temperature (the 1.20 detunings exactly)."""
    p = _params(80.0, 0.2, 0.2)
    ens = Ensemble(delta=DELTA, weight=W, n=np.ones(2))
    rt = Rates.from_params(p, ens)
    expected = DELTA + 2 * p.n_th * rt.chi1 * (p.g * W) ** 2
    assert np.max(np.abs(rt.delta - expected)) < 1e-12
    assert abs(p.n_th - 0.2) < 1e-12
    cold = Rates.from_params(_params(80.0, 0.2, 0.0), ens)
    assert np.array_equal(cold.delta, DELTA)


def test_sign_of_delta_mirrors_the_state_of_a_symmetric_line():
    """Flipping Delta flips chi and Gamma_SR stays; for a line symmetric
    about the spin frequency the state is the mirror image (J_y -> -J_y),
    so xi2 and the contrast do not change (with thermal photons, n_th = 0.29,
    and the thermal shift on: xi2 equal to 6e-14 relative, the contrast
    exactly, the mirrored mean spin to 1e-17 of N)."""
    N = 1e8
    line = cs.lineshape("gaussian", 2 * np.pi * 2e3)
    ens = cs.equal_probability_classes(line, 8, N)
    out = []
    for D in (30e6, -30e6):
        p = cs.from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=D, T=0.1, T2=0.15)
        rt = cs.Rates.from_params(p, ens)
        st = cs.twist(cs.css_x(rt.M), rt, 2e-5, echo=False)
        out.append((cs.collective_moments(st, rt.n)[0], cs.wineland_xi2(st, rt.n)[0],
                    cs.coherence(st, rt.n)))
    (J1, x1, c1), (J2, x2, c2) = out
    assert abs(x1 / x2 - 1) < 1e-9 and abs(c1 / c2 - 1) < 1e-12
    assert np.max(np.abs(J1 * [1, -1, 1] - J2)) < 1e-12 * N


def test_spin_cavity_refuses_bad_inputs():
    p = _params(20.0, 0.2, 0.0)
    with pytest.raises(ValueError, match="one value per spin"):
        spin_cavity(p, [0.0, 0.1], [1.0], CSS, TIMES)
    with pytest.raises(ValueError, match="n_fock"):
        spin_cavity(p, [0.0], [1.0], CSS, TIMES, n_fock=1)
    with pytest.raises(ValueError, match="n_fock"):
        spin_cavity(p, [0.0], [1.0], CSS, TIMES, n_fock=4.5)
