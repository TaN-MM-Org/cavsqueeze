"""Bridge from the collective spin state to the bosonic quantum stack.

In the Holstein-Primakoff picture, the transverse fluctuations of a
polarized collective spin are one effective bosonic mode: with the mean
spin along e3 and quadratures normalized so that a coherent spin state
maps to the vacuum (Var x = Var p = 1/2, hbar = 1),

    x ~ J_e1 / sqrt(S2/2),   p ~ J_e2 / sqrt(S2/2),

where S2 is the coherent-state normalization the solver already uses for
its squeezing parameters (S2 = N for uniform coupling weights).  This
module extracts that mode from any solver state as a 2x2 Gaussian
covariance matrix, decomposes it into (squeezing r, angle theta, thermal
occupation n_th) through its symplectic eigenvalue, and can hand the
result to QuTiP as an explicit density matrix

    rho = S(r e^{2 i theta}) rho_th(n_th) S(r e^{2 i theta})^dagger,

so that anything downstream of QuTiP (channels, measurements, tomography
tooling, other packages) can consume cavsqueeze output directly.  The
export is validated by reconstructing the covariance from the exported
state; the QuTiP phase convention used here (squeeze(N, r e^{i phi})
squeezes the quadrature at angle phi/2) is asserted in the test suite
rather than assumed.

The mapping keeps second moments only (the Gaussian/Holstein-Primakoff
approximation, accurate for xi-level squeezing at large N); curvature
corrections of the Bloch sphere are outside its scope.
"""
from __future__ import annotations

import numpy as np

from .cumulant import State, collective_moments


def _transverse_basis(J):
    """Orthonormal (e1, e2, e3) with e3 along the mean spin (the same
    construction as cumulant.transverse_variances)."""
    Jn = np.linalg.norm(J)
    if Jn == 0.0:
        raise ValueError("mean spin vanishes; no Holstein-Primakoff frame")
    e3 = J / Jn
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def bosonic_mode(st: State, n, weights=None, spec_n=None, tol: float = 1e-6):
    """The effective Holstein-Primakoff mode of the collective spin.

    Returns a dict with

    * Sigma: 2x2 quadrature covariance (vacuum = I/2),
    * nu: symplectic eigenvalue (1 for any pure Gaussian state; the
      uncertainty relation requires nu >= 1),
    * n_th: thermal occupation (nu - 1)/2,
    * r, theta: squeezing parameter and the angle (rad, in the e1-e2
      plane) of the minimum-variance quadrature,
    * purity: 1/nu,
    * var_min, var_max: extremal quadrature variances,
    * squeezing_db: 10 log10(2 var_min), negative when squeezed below
      vacuum,
    * basis: (e1, e2, e3) frame vectors.

    States violating nu >= 1 by more than ``tol`` (which cannot happen
    physically) raise ValueError; smaller numerical undershoots are
    clamped to nu = 1.
    """
    J, Cov, S1, S2 = collective_moments(st, n, weights, spec_n=spec_n)
    e1, e2, e3 = _transverse_basis(J)
    scale = S2 / 2.0
    Sigma = np.array([
        [e1 @ Cov @ e1, e1 @ Cov @ e2],
        [e2 @ Cov @ e1, e2 @ Cov @ e2],
    ]) / scale
    Sigma = 0.5 * (Sigma + Sigma.T)
    det = float(np.linalg.det(Sigma))
    if det <= 0:
        raise ValueError("covariance is not positive definite")
    nu = 2.0 * np.sqrt(det)
    if nu < 1.0 - tol:
        raise ValueError(
            f"symplectic eigenvalue nu = {nu:.6f} < 1 violates the "
            "uncertainty relation; the state is outside the Gaussian regime")
    nu = max(nu, 1.0)
    evals, evecs = np.linalg.eigh(Sigma)
    lmin, lmax = float(evals[0]), float(evals[1])
    r = 0.25 * np.log(lmax / max(lmin, 1e-300))
    vmin_vec = evecs[:, 0]
    theta = float(np.arctan2(vmin_vec[1], vmin_vec[0])) % np.pi
    return dict(Sigma=Sigma, nu=float(nu), n_th=float((nu - 1.0) / 2.0),
                r=float(r), theta=theta, purity=float(1.0 / nu),
                var_min=lmin, var_max=lmax,
                squeezing_db=float(10.0 * np.log10(2.0 * lmin)),
                basis=(e1, e2, e3))


def _auto_dim(mode) -> int:
    """Fock cutoff for the export.  Squeezed states have heavy-tailed
    photon-number distributions, and it is the squeezed quadrature whose
    delicate cancellation is destroyed first by truncation; empirically a
    cutoff near 40x the mean photon number holds the extremal variances
    to better than a percent even at r = 2.3 (see the test suite)."""
    n_mean = (2 * mode["n_th"] + 1) * np.cosh(2 * mode["r"]) / 2 - 0.5
    return int(max(24, np.ceil(40 * (n_mean + 1))))


def to_qutip(st: State, n, weights=None, spec_n=None, dim: int | None = None,
             check: bool = True, rtol: float = 0.01):
    """Export the Holstein-Primakoff mode as a QuTiP density matrix.

    Builds rho = S(z) rho_th(n_th) S(z)^dagger with z = r e^{2 i theta},
    which reproduces the mode's 2x2 covariance up to Fock truncation;
    ``dim`` defaults to a cutoff well above the mean photon number.  With
    ``check=True`` (the default) the two extremal quadrature variances of
    the exported state are computed and compared against the target
    covariance, and a ValueError names the required accuracy if truncation
    spoiled them, so the export can never silently return a corrupted
    state.  Returns (rho, mode) with ``mode`` from :func:`bosonic_mode`.

    Requires qutip (install the ``exact`` extra).
    """
    import qutip  # optional dependency, imported here on purpose

    mode = bosonic_mode(st, n, weights, spec_n=spec_n)
    N = _auto_dim(mode) if dim is None else int(dim)
    z = mode["r"] * np.exp(2j * mode["theta"])
    S = qutip.squeeze(N, z)
    rho = S * qutip.thermal_dm(N, mode["n_th"]) * S.dag()
    if check:
        a = qutip.destroy(N)
        for phi, target in ((mode["theta"], mode["var_min"]),
                            (mode["theta"] + np.pi / 2, mode["var_max"])):
            x = (a * np.exp(-1j * phi) + a.dag() * np.exp(1j * phi)) / np.sqrt(2)
            got = float(qutip.expect(x * x, rho).real
                        - qutip.expect(x, rho).real ** 2)
            if abs(got - target) > rtol * target:
                raise ValueError(
                    f"Fock truncation at dim={N} distorts the exported state "
                    f"(Var(x_phi) = {got:.4g}, target {target:.4g}); "
                    "pass a larger dim")
    return rho, mode


def covariance_of_qutip_state(rho, phis=None):
    """Measured quadrature variances Var(x_phi) of a single-mode QuTiP
    state, for validation: x_phi = (a e^{-i phi} + a^dag e^{i phi})/sqrt(2).

    Returns (phis, variances).  Used by the test suite to verify that the
    exported state reproduces the solver covariance.
    """
    import qutip

    N = rho.shape[0]
    a = qutip.destroy(N)
    if phis is None:
        phis = np.linspace(0.0, np.pi, 61)
    out = []
    for phi in phis:
        x = (a * np.exp(-1j * phi) + a.dag() * np.exp(1j * phi)) / np.sqrt(2)
        out.append(float(qutip.expect(x * x, rho).real
                         - qutip.expect(x, rho).real ** 2))
    return np.asarray(phis), np.asarray(out)
