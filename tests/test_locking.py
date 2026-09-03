"""The mean-field right-hand side is the pendulum of Eq. (5) of the paper.

The package writes the per-class coherence as s = <sigma^-> = (rho/2) exp(-i phi),
so its phi and z carry the opposite sign to the convention of the paper; the
magnitudes are what must agree.
"""
import numpy as np
import pytest

from cavsqueeze import from_hz
from cavsqueeze.cumulant import Rates, _rhs_meanfield
from cavsqueeze.ensemble import Ensemble

TWO_PI = 2.0 * np.pi


def _setup(chiN_hz=2.0e4, gamma_hz=5e3):
    N, kappa = 1e10, 1e4
    Delta = 1e5 * kappa / 2
    gN = np.sqrt(chiN_hz * Delta)
    p = from_hz(gN / np.sqrt(N), kappa, Delta, T=0.0, T2=np.inf)
    delta = np.array([0.0, 0.37 * TWO_PI * gamma_hz])
    ens = Ensemble(delta=delta, weight=np.ones(2), n=np.array([0.85 * N, 0.15 * N]))
    return p, ens, Rates.from_params(p, ens)


def test_meanfield_is_a_pendulum():
    p, ens, rt = _setup()
    phi = np.array([0.13, -0.42])
    z = np.array([0.21, -0.09])
    rho = np.sqrt(1.0 - z**2)
    s = 0.5 * rho * np.exp(-1j * phi)
    d = _rhs_meanfield(0.0, np.concatenate([s, z.astype(complex)]), rt)
    ds, dz = d[:2], d[2:]
    phidot = -np.imag(ds / s)
    zdot = np.real(dz)

    Ws = rt.n @ s
    R = 2.0 * abs(Ws) / rt.n.sum()
    Psi = -np.angle(Ws)
    Omega = p.chi * rt.n.sum() * R
    psi = phi - Psi
    # Eq. (5) of the paper, in the package's sign convention
    phidot_pend = -(ens.delta - Omega * (z / rho) * np.cos(psi))
    zdot_pend = -Omega * rho * np.sin(psi)

    assert np.allclose(phidot, phidot_pend, rtol=2e-3)
    assert np.allclose(zdot, zdot_pend, rtol=2e-3)


def test_locking_field_is_chi_N_times_contrast():
    """Omega must scale with the contrast, not with chi N alone."""
    p, ens, rt = _setup()
    for R_target in (0.4, 0.8):
        rho = R_target
        z = np.sqrt(1.0 - rho**2)
        s = 0.5 * rho * np.ones(2, complex)
        Ws = rt.n @ s
        assert np.isclose(2.0 * abs(Ws) / rt.n.sum(), R_target, rtol=1e-12)


@pytest.mark.parametrize("dratio,locked", [(0.5, True), (0.9, True), (1.5, False), (3.0, False)])
def test_locking_criterion(dratio, locked):
    """Trajectories through (psi, z) = (0, 0) swing only for |delta| < Omega."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        psi, zz = y
        r = np.sqrt(max(1e-16, 1.0 - zz * zz))
        return [dratio - zz * np.cos(psi) / r, r * np.sin(psi)]

    sol = solve_ivp(rhs, (0, 200.0), [0.0, 0.0], rtol=1e-10, atol=1e-12,
                    method="DOP853", dense_output=True)
    psi = sol.sol(np.linspace(0, 200.0, 200001))[0]
    assert (np.abs(psi).max() < np.pi) == locked


def test_energy_integral():
    """rho cos(psi) = 1 - (delta/Omega) z holds along the trajectory."""
    from scipy.integrate import solve_ivp

    dratio = 0.8

    def rhs(t, y):
        psi, zz = y
        r = np.sqrt(max(1e-16, 1.0 - zz * zz))
        return [dratio - zz * np.cos(psi) / r, r * np.sin(psi)]

    t = np.linspace(0, 100.0, 100001)
    sol = solve_ivp(rhs, (0, 100.0), [0.0, 0.0], t_eval=t, rtol=1e-11, atol=1e-13,
                    method="DOP853")
    psi, z = sol.y
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    assert np.max(np.abs(rho * np.cos(psi) - (1.0 - dratio * z))) < 1e-7


def test_cross_file_consistency():
    """The title, version, DOI, citations and cross-references agree across files."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts" / "check_consistency.py").exists():
        pytest.skip("paper-companion scripts not present in this repository")
    r = subprocess.run([sys.executable, str(root / "scripts" / "check_consistency.py")],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
