"""Discrete truncated Wigner approximation, as an independent check on the
cumulant closure for an inhomogeneously broadened ensemble.

The cumulant solver truncates at second order.  For a *uniform* ensemble that
truncation is checked against exact Dicke-basis solutions (`exact.dicke_piqs`),
and for eight distinguishable spins against the exact master equation
(`exact.full_hilbert`).  Neither reference reaches an inhomogeneous line at the
spin numbers the article uses, which is the one place the closure is applied
without an independent check.

This module supplies that check.  Every spin is represented by a classical
vector whose components are sampled from the discrete Wigner function of the
coherent spin state,

    s^x = 1,   s^y = +-1,   s^z = +-1   (each sign with probability 1/2),

each sample is propagated along the classical trajectory of the same
Hamiltonian, and symmetrically ordered moments are read off as averages over
samples.  The approximation is different in kind from a cumulant expansion: it
keeps the full non-Gaussian statistics of the initial state and truncates the
*quantum* corrections to the equations of motion, where the cumulant expansion
keeps the equations and truncates the statistics.  Agreement between the two is
therefore evidence that neither truncation is doing damage.

The check covers the unitary part of the model: twisting plus a spread of
detunings.  That is the part that has no other reference, and it is the part
the wing floor of the article comes from, since the floor contains no cavity
parameter.  Collective emission, which sets the other limit, is checked against
the exact Dicke-basis solution instead.

Reference for the method: J. Schachenmayer, A. Pikovski and A. M. Rey,
Phys. Rev. X 5, 011022 (2015).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def sample_css_x(n_spins: int, n_traj: int, rng) -> np.ndarray:
    """Discrete Wigner samples of a coherent spin state along +x.

    Returns an array of shape (n_traj, n_spins, 3).  The mean over samples is
    (1, 0, 0) per spin and the variance of the transverse and longitudinal
    components is 1, which is the projection noise of a spin one-half.
    """
    s = np.empty((n_traj, n_spins, 3))
    s[:, :, 0] = 1.0
    s[:, :, 1] = rng.choice([-1.0, 1.0], size=(n_traj, n_spins))
    s[:, :, 2] = rng.choice([-1.0, 1.0], size=(n_traj, n_spins))
    return s


def _rhs(t, y, delta, G, chi1, n_traj, n_spins):
    """Classical equations of motion for chi sum_jk G_j G_k sigma^+_j sigma^-_k
    plus the detunings, written for every sample at once.

    Same convention as `cumulant._rhs_meanfield`: sigma^- -> s = (x - i y)/2,
    sigma^z -> z, and the self-term j = k is removed from the collective sum.
    """
    s = y[: 2 * n_traj * n_spins].view(np.complex128).reshape(n_traj, n_spins)
    z = y[2 * n_traj * n_spins:].reshape(n_traj, n_spins)
    coll = (s * G) @ np.ones(n_spins)          # sum_k G_k s_k, per sample
    other = coll[:, None] - G * s              # sum_{k != j} G_k s_k
    ds = 1j * (delta + chi1 * G**2) * s - 1j * chi1 * G * z * other
    dz = 4.0 * chi1 * G * np.imag(s * np.conj(other))
    return np.concatenate([ds.ravel().view(np.float64), dz.ravel()])


def evolve(delta, G, chi1, t_eval, n_traj=2000, seed=0, steps_per_rad=12.0, min_steps=400):
    """Propagate `n_traj` samples of the coherent spin state along +x.

    `delta` and `G` are per-spin arrays in rad/s and dimensionless weights;
    `chi1` is the single-pair interaction, so that chi N is chi1 times the sum
    of G^2.  Returns the collective moments at each time in `t_eval`.

    Fixed-step fourth-order Runge-Kutta.  The step is set by the fastest phase
    in the problem, max(|delta|, chi N), which is what an adaptive integrator
    has to resolve anyway; at these sizes the state is large enough that
    adaptive stepping spends its time moving memory rather than integrating.
    `last_steps` reports the step count so that convergence can be checked by
    doubling `steps_per_rad`.
    """
    global last_steps
    delta = np.asarray(delta, float)
    G = np.asarray(G, float)
    n_spins = len(delta)
    chiN = chi1 * float(np.sum(G ** 2))
    t_eval = np.asarray(t_eval, float)
    T = float(np.max(t_eval))
    fastest = max(float(np.max(np.abs(delta))), abs(chiN), 1.0)
    n_step = int(max(min_steps, np.ceil(steps_per_rad * fastest * T / (2 * np.pi))))
    last_steps = n_step
    dt = T / n_step

    rng = np.random.default_rng(seed)
    v = sample_css_x(n_spins, n_traj, rng)
    s = 0.5 * (v[:, :, 0] - 1j * v[:, :, 1])
    z = v[:, :, 2].astype(float)
    a = delta + chi1 * G ** 2
    idx = np.clip(np.round(t_eval / dt).astype(int), 0, n_step)

    def deriv(s, z):
        coll = (s * G).sum(axis=1)
        other = coll[:, None] - G * s
        ds = 1j * (a * s - chi1 * G * z * other)
        dz = 4.0 * chi1 * G * np.imag(s * np.conj(other))
        return ds, dz

    want = {}
    for k, i in enumerate(idx):
        want.setdefault(int(i), []).append(k)
    out = {}
    if 0 in want:
        m = _moments(2 * np.real(s), -2 * np.imag(s), z)
        for k in want[0]:
            out[k] = m
    for n in range(n_step):
        k1s, k1z = deriv(s, z)
        k2s, k2z = deriv(s + 0.5 * dt * k1s, z + 0.5 * dt * k1z)
        k3s, k3z = deriv(s + 0.5 * dt * k2s, z + 0.5 * dt * k2z)
        k4s, k4z = deriv(s + dt * k3s, z + dt * k3z)
        s = s + (dt / 6.0) * (k1s + 2 * k2s + 2 * k3s + k4s)
        z = z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        if (n + 1) in want:
            m = _moments(2 * np.real(s), -2 * np.imag(s), z)
            for k in want[n + 1]:
                out[k] = m
    return [out[k] for k in range(len(t_eval))]


last_steps = 0


def _moments(x_, y_, z_):
    """Collective mean and symmetric covariance from the sample cloud.

    J_alpha = (1/2) sum_j s^alpha_j for each sample; the mean and covariance
    over samples are the symmetrically ordered quantum moments.
    """
    J = 0.5 * np.stack([x_.sum(1), y_.sum(1), z_.sum(1)], axis=1)   # (n_traj, 3)
    mean = J.mean(0)
    cov = np.cov(J.T, bias=False)
    return dict(mean=mean, cov=cov, n_traj=J.shape[0])


def wineland(mean, cov, N):
    """Wineland parameter from a collective mean and covariance.

    Same definition as the cumulant solver: the smallest variance in the plane
    perpendicular to the mean spin, times N, over the squared spin length.
    """
    Jn = np.linalg.norm(mean)
    if Jn <= 0:
        return np.inf
    e3 = mean / Jn
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    a = e1 @ cov @ e1
    b = e2 @ cov @ e2
    c = e1 @ cov @ e2
    vmin = 0.5 * (a + b) - np.hypot(0.5 * (a - b), c)
    return float(N * vmin / Jn**2)


def sample_line(dist, n_spins: int, rng) -> np.ndarray:
    """Draw `n_spins` detunings from a line shape by inverse transform.

    Stratified: the k-th spin takes the quantile (k + u_k) / n_spins, which
    removes the sampling noise of the line itself so that the comparison sees
    only the difference between the two solvers.
    """
    u = (np.arange(n_spins) + rng.random(n_spins)) / n_spins
    return dist.ppf(u)
