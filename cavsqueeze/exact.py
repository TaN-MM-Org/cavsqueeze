"""Exact master-equation references (QuTiP) for validating the cumulant solver.

Three references are provided:
  * `full_hilbert`: N distinguishable spins with arbitrary detunings and
    couplings (N <= ~10), under the model with the cavity eliminated.
  * `dicke_piqs`: N identical spins (homogeneous) using the permutation
    invariant (Dicke) solver PIQS (N up to a few hundred), same model.
  * `spin_cavity` (new in 1.21): a few spins together with the cavity mode
    itself, not eliminated.  It checks the elimination: the rates chi and
    Gamma_SR of `cavsqueeze.resonator`, the sign of the detuning, and how
    large the error of the elimination is for given parameters.

All use the package's dephasing convention L_j = sqrt(gamma_phi/2) sigma z_j
(each spin's coherence decays as exp(-gamma_phi t)); since 1.20 the PIQS
reference does too (before, it had half that dephasing).
"""
from __future__ import annotations

import numpy as np
import qutip as qt

from .cumulant_raw import Rates


def _ops(N):
    sp, sm, sz = [], [], []
    for j in range(N):
        ops = [qt.qeye(2)] * N
        ops[j] = qt.sigmap()
        sp.append(qt.tensor(ops))
        ops[j] = qt.sigmam()
        sm.append(qt.tensor(ops))
        ops[j] = qt.sigmaz()
        sz.append(qt.tensor(ops))
    return sp, sm, sz


def build_liouvillian(delta, G, chi1, Gd, Gu, gamma_phi):
    """Return (H, c_ops, sp, sm, sz) for N distinguishable spins."""
    N = len(delta)
    sp, sm, sz = _ops(N)
    H = 0
    for j in range(N):
        H += 0.5 * delta[j] * sz[j]
        for k in range(N):
            H += chi1 * G[j] * G[k] * sp[j] * sm[k]
    Jm = sum(G[j] * sm[j] for j in range(N))
    Jp = Jm.dag()
    c_ops = []
    if Gd > 0:
        c_ops.append(np.sqrt(Gd) * Jm)
    if Gu > 0:
        c_ops.append(np.sqrt(Gu) * Jp)
    if gamma_phi > 0:
        c_ops += [np.sqrt(gamma_phi / 2.0) * sz[j] for j in range(N)]
    return H, c_ops, sp, sm, sz


def full_hilbert(rt_spins: Rates, bloch, times, extra_ops=None):
    """Exact evolution of distinguishable spins.  `rt_spins` must have one
    class per spin (n = 1 each).  `bloch` is the common initial Bloch vector.
    Returns dict with first and second moments at each time, in the same
    conventions as the cumulant State (per-spin arrays)."""
    N = rt_spins.M
    assert np.allclose(rt_spins.n, 1.0)
    H, c_ops, sp, sm, sz = build_liouvillian(
        rt_spins.delta, rt_spins.G, rt_spins.chi1, rt_spins.Gd, rt_spins.Gu, rt_spins.gamma_phi
    )
    v = np.asarray(bloch, float)
    theta = np.arccos(np.clip(v[2], -1, 1))
    phi = np.arctan2(v[1], v[0])
    single = (np.cos(theta / 2) * qt.basis(2, 0) + np.exp(1j * phi) * np.sin(theta / 2) * qt.basis(2, 1)).unit()
    psi0 = qt.tensor([single] * N)
    e_ops = []
    for j in range(N):
        e_ops += [sp[j], sz[j]]
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            e_ops += [sp[j] * sm[k], sp[j] * sp[k], sz[j] * sp[k], sz[j] * sz[k]]
    res = qt.mesolve(H, psi0, times, c_ops=c_ops, e_ops=e_ops, options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 100000})
    out = {"s": np.zeros((len(times), N), complex), "z": np.zeros((len(times), N)),
           "P": np.zeros((len(times), N, N), complex), "Q": np.zeros((len(times), N, N), complex),
           "R": np.zeros((len(times), N, N), complex), "Z": np.zeros((len(times), N, N))}
    idx = 0
    for j in range(N):
        out["s"][:, j] = res.expect[idx]
        out["z"][:, j] = np.real(res.expect[idx + 1])
        idx += 2
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            out["P"][:, j, k] = res.expect[idx]
            out["Q"][:, j, k] = res.expect[idx + 1]
            out["R"][:, j, k] = res.expect[idx + 2]
            out["Z"][:, j, k] = np.real(res.expect[idx + 3])
            idx += 4
    return out


def spin_cavity(params, delta, weights, bloch, times, n_fock: int = 8):
    """Exact evolution of a few spins coupled to the cavity mode itself
    (new in 1.21), with no adiabatic elimination.

    In the frame that turns at the spin frequency omega_s,

        H = -Delta a^dag a + sum_j (delta_j / 2) sigma z_j
            + g sum_j w_j (a^dag sigma-_j + a sigma+_j),

    with the cavity damped at the rate kappa (its full linewidth) towards a
    bath with n_th thermal photons: jump operators sqrt(kappa (n_th + 1)) a
    and sqrt(kappa n_th) a^dag, plus the dephasing L_j = sqrt(gamma_phi/2)
    sigma z_j.  Here Delta = params.Delta, which in this package's formulas
    is omega_s - omega_c (the cavity sits at -Delta in this frame; see
    `cavsqueeze.resonator`).  The cavity starts in the thermal state with
    n_th photons and the spins in the product state with Bloch vector
    ``bloch``.

    params : CavityParams (g, kappa, Delta, n_th from omega_s and T,
        gamma_phi).
    delta, weights : per-spin detunings (rad/s) and coupling weights, one
        spin each (N <= 4 or so: the space has 2^N * n_fock states).
    times : output times (s).
    n_fock : number of cavity photon-number states kept.

    Returns the dict of `full_hilbert` (per-spin first and pair moments in
    the conventions of the cumulant State) plus "n_photon", the mean photon
    number, and "p_top", the largest population of the highest kept photon
    number over the run (small when n_fock is large enough).  Comparing with
    `full_hilbert` for `Rates.from_params(params, ensemble)` measures the
    error of the elimination (tests/test_spin_cavity_121.py).
    """
    delta = np.asarray(delta, dtype=float)
    w = np.asarray(weights, dtype=float)
    N = delta.size
    if w.shape != delta.shape or N < 1:
        raise ValueError("delta and weights must have one value per spin")
    nf = int(n_fock)
    if nf != n_fock or nf < 2:
        raise ValueError(f"n_fock must be a whole number >= 2; got {n_fock}")
    sp, sm, sz = [], [], []
    for j in range(N):
        ops = [qt.qeye(2)] * N + [qt.qeye(nf)]
        ops[j] = qt.sigmap()
        sp.append(qt.tensor(ops))
        ops[j] = qt.sigmam()
        sm.append(qt.tensor(ops))
        ops[j] = qt.sigmaz()
        sz.append(qt.tensor(ops))
    a = qt.tensor([qt.qeye(2)] * N + [qt.destroy(nf)])
    g = float(params.g)
    H = -float(params.Delta) * a.dag() * a
    for j in range(N):
        H += 0.5 * delta[j] * sz[j] + g * w[j] * (a.dag() * sm[j] + a * sp[j])
    nth = float(params.n_th)
    kap = float(params.kappa)
    c_ops = []
    if kap > 0:
        c_ops.append(np.sqrt(kap * (nth + 1.0)) * a)
        if nth > 0:
            c_ops.append(np.sqrt(kap * nth) * a.dag())
    if params.gamma_phi > 0:
        c_ops += [np.sqrt(params.gamma_phi / 2.0) * sz[j] for j in range(N)]
    v = np.asarray(bloch, float)
    theta = np.arccos(np.clip(v[2], -1, 1))
    phi = np.arctan2(v[1], v[0])
    single = (np.cos(theta / 2) * qt.basis(2, 0)
              + np.exp(1j * phi) * np.sin(theta / 2) * qt.basis(2, 1)).unit()
    spins = qt.tensor([single] * N)
    rho0 = qt.tensor(spins * spins.dag(), qt.thermal_dm(nf, nth))
    top = qt.tensor([qt.qeye(2)] * N + [qt.fock_dm(nf, nf - 1)])
    e_ops = []
    for j in range(N):
        e_ops += [sp[j], sz[j]]
    for j in range(N):
        for k in range(N):
            if j != k:
                e_ops += [sp[j] * sm[k], sp[j] * sp[k], sz[j] * sp[k], sz[j] * sz[k]]
    e_ops += [a.dag() * a, top]
    res = qt.mesolve(H, rho0, times, c_ops=c_ops, e_ops=e_ops,
                     options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 1000000})
    T = len(times)
    out = {"s": np.zeros((T, N), complex), "z": np.zeros((T, N)),
           "P": np.zeros((T, N, N), complex), "Q": np.zeros((T, N, N), complex),
           "R": np.zeros((T, N, N), complex), "Z": np.zeros((T, N, N))}
    idx = 0
    for j in range(N):
        out["s"][:, j] = res.expect[idx]
        out["z"][:, j] = np.real(res.expect[idx + 1])
        idx += 2
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            out["P"][:, j, k] = res.expect[idx]
            out["Q"][:, j, k] = res.expect[idx + 1]
            out["R"][:, j, k] = res.expect[idx + 2]
            out["Z"][:, j, k] = np.real(res.expect[idx + 3])
            idx += 4
    out["n_photon"] = np.real(np.asarray(res.expect[idx]))
    out["p_top"] = float(np.max(np.real(res.expect[idx + 1])))
    return out


def dicke_piqs(N, chi, Gd, Gu, gamma_phi, theta, times):
    """Homogeneous ensemble (all G = 1, delta = 0) solved in the Dicke basis with PIQS.
    H = chi J+ J- (J+ = sum sigma+), collective emission Gd, collective pumping Gu,
    local dephasing gamma_phi in the convention of the rest of the package,
    L_j = sqrt(gamma_phi/2) sigma z_j, so that each spin's coherence decays as
    exp(-gamma_phi t).  Initial coherent spin state at polar angle theta from +z
    (the excited state |up>; <Jz> = (N/2) cos theta at t = 0), azimuth 0 (points
    along +x for theta = pi/2).
    Returns dict with <Jx>,<Jy>,<Jz> and the symmetrized covariance of (Jx,Jy,Jz).

    Before 1.20 gamma_phi was passed to PIQS unchanged.  With PIQS's local
    dephasing rate gamma_D each spin's coherence decays at gamma_D / 2, so the
    reference had half the intended dephasing (for 4 spins with
    gamma_phi = 0.7 and no other terms, <Jx> at t = 1 was 2 exp(-0.35) = 1.409
    instead of 2 exp(-0.7) = 0.993).  It now passes 2 gamma_phi;
    tests/test_fixes_120.py checks the result against the closed form and
    against `full_hilbert` (which builds the Lindblad operators explicitly).
    The docstring also said theta was measured from the south pole; the state
    has always been the one described above."""
    from qutip import piqs

    ens = piqs.Dicke(N=N)
    jx, jy, jz = piqs.jspin(N)
    jp = piqs.jspin(N, "+")
    jm = piqs.jspin(N, "-")
    ens.hamiltonian = chi * jp * jm
    ens.collective_emission = Gd
    ens.collective_pumping = Gu
    ens.dephasing = 2.0 * gamma_phi
    L = ens.liouvillian()
    # coherent spin state: rotate |j, -j> by theta about y
    rho0 = piqs.css(N, x=theta, y=0.0, basis="dicke", coordinates="polar")
    ops = [jx, jy, jz, jx * jx, jy * jy, jz * jz, jx * jy + jy * jx, jx * jz + jz * jx, jy * jz + jz * jy]
    res = qt.mesolve(L, rho0, times, e_ops=ops, options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 200000})
    E = [np.real(e) for e in res.expect]
    J = np.stack(E[:3], axis=1)
    Cov = np.zeros((len(times), 3, 3))
    Cov[:, 0, 0] = E[3] - J[:, 0] ** 2
    Cov[:, 1, 1] = E[4] - J[:, 1] ** 2
    Cov[:, 2, 2] = E[5] - J[:, 2] ** 2
    Cov[:, 0, 1] = Cov[:, 1, 0] = 0.5 * E[6] - J[:, 0] * J[:, 1]
    Cov[:, 0, 2] = Cov[:, 2, 0] = 0.5 * E[7] - J[:, 0] * J[:, 2]
    Cov[:, 1, 2] = Cov[:, 2, 1] = 0.5 * E[8] - J[:, 1] * J[:, 2]
    return {"J": J, "Cov": Cov}


def xi2_from_moments(J, Cov, N):
    """Wineland parameter from collective mean and covariance."""
    Jn = np.linalg.norm(J)
    e3 = J / Jn
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    B = np.stack([e1, e2], axis=1)
    vals = np.linalg.eigvalsh(B.T @ Cov @ B)
    return N * vals[0] / Jn**2
