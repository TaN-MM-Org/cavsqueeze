"""Pulse sequences built on the cumulant solver.

All sequences start from the coherent spin state along +x (all spins in the
ground state |down>, followed by a pi/2 pulse about y), which is the state
prepared in the experiments of Fukumori et al. (arXiv:2604.26909).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .cumulant import (
    Rates,
    State,
    product_state,
    rotate,
    rotate_classes,
    evolve,
    evolve_meanfield,
    wineland_xi2,
    collective_moments,
    coherence,
)
from .ensemble import Ensemble
from .resonator import CavityParams

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def css_x(M: int) -> State:
    return product_state(M, X)


def twist(st: State, rt: Rates, t: float, echo: bool = True, **kw) -> State:
    """Free evolution for time t under the cavity-mediated interaction; with
    echo=True a pi pulse about x is applied at t/2 (Hahn echo), which refocuses
    static detunings while leaving the one-axis-twisting term unchanged."""
    if echo:
        st = evolve(st, rt, t / 2, **kw)
        st = rotate(st, X, np.pi)
        st = evolve(st, rt, t / 2, **kw)
        return st
    return evolve(st, rt, t, **kw)


def pulse(st: State, rt: Rates, axis, angle: float, duration: float = 0.0, angles=None, n_steps: int = 20, **kw) -> State:
    """A rotation pulse.  With duration = 0 it is an instantaneous rotation
    (per-class angles if `angles` is given).  With a finite duration the
    cavity-mediated evolution continues during the pulse: the pulse is split
    into n_steps equal rotation increments interleaved with free evolution
    (Strang splitting, error O(duration^2 / n_steps^2))."""
    if duration <= 0:
        return rotate(st, axis, angle) if angles is None else rotate_classes(st, axis, angles)
    dt = duration / n_steps
    inc = angle / n_steps if angles is None else np.asarray(angles, float) / n_steps
    st = evolve(st, rt, dt / 2, **kw)
    for k in range(n_steps):
        st = rotate(st, axis, inc) if angles is None else rotate_classes(st, axis, inc)
        if k < n_steps - 1:
            st = evolve(st, rt, dt, **kw)
    return evolve(st, rt, dt / 2, **kw)


def twist_imperfect(st: State, rt: Rates, t: float, echo: bool = True, pi_duration: float = 0.0,
                    pi_angles=None, **kw) -> State:
    """Echo twist with an imperfect pi pulse: finite duration pi_duration (the
    total sequence still lasts t + pi_duration) and per-class rotation angles
    pi_angles (default pi for every class)."""
    if not echo:
        return evolve(st, rt, t, **kw)
    st = evolve(st, rt, t / 2, **kw)
    st = pulse(st, rt, X, np.pi, duration=pi_duration, angles=pi_angles, **kw)
    return evolve(st, rt, t / 2, **kw)


def squeezing_after(params: CavityParams, ens: Ensemble, t: float, echo: bool = True, weights=None, **kw):
    """Wineland parameter (linear units) and contrast after twisting for time t."""
    rt = Rates.from_params(params, ens)
    st = twist(css_x(rt.M), rt, t, echo=echo, **kw)
    xi2, ang, vmin, vmax, Jn = wineland_xi2(st, rt.n, weights, spec_n=rt.spec_n)
    return dict(xi2=xi2, angle=ang, var_min=vmin, var_max=vmax, contrast=coherence(st, rt.n, spec_n=rt.spec_n), Q=rt.chiN * t, t=t)


def squeezing_trace(params: CavityParams, ens: Ensemble, t_list, echo: bool = True, weights=None, **kw):
    out = [squeezing_after(params, ens, t, echo=echo, weights=weights, **kw) for t in t_list]
    return {k: np.array([o[k] for o in out]) for k in out[0]}


def optimal_squeezing(params: CavityParams, ens: Ensemble, t_lo: float, t_hi: float, echo: bool = True,
                      weights=None, n_coarse: int = 12, points_per_period: int = 8, max_fine: int = 40,
                      **kw) -> dict:
    """Minimize xi^2 over the interaction time in [t_lo, t_hi].

    Stage 1: a coarse geometric grid of n_coarse times locates the envelope
    minimum t*.
    Stage 2: a linear scan between the two coarse-grid neighbors of t* (the
    end point itself when t* is the first or last grid point), with
    points_per_period samples per gap period 2 pi / (chi N), at least 8 and
    at most max_fine points, resolves the oscillations of the squeezing
    parameter at the collective gap frequency.  (Docstrings before 1.17 said
    the scan covered [t*/2, 2 t*]; the code has always used the neighbors.)
    Returns the `squeezing_after` dict at the best time scanned, plus the
    whole scan as scan_t and scan_xi2."""
    if not (np.isfinite(t_lo) and np.isfinite(t_hi) and 0 < t_lo < t_hi):
        raise ValueError(f"need 0 < t_lo < t_hi; got t_lo={t_lo}, t_hi={t_hi}")
    rt = Rates.from_params(params, ens)
    ts = np.geomspace(t_lo, t_hi, n_coarse)
    coarse = [squeezing_after(params, ens, t, echo=echo, weights=weights, **kw) for t in ts]
    vals = np.array([r["xi2"] for r in coarse])
    k = int(np.argmin(vals))
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
    period = 2 * np.pi / max(abs(rt.chiN), 1e-30)
    n_fine = int(np.clip(np.ceil((hi - lo) / period * points_per_period), 8, max_fine))
    tf = np.linspace(lo, hi, n_fine)
    fine = [squeezing_after(params, ens, t, echo=echo, weights=weights, **kw) for t in tf]
    vf = np.array([r["xi2"] for r in fine])
    all_t = np.concatenate([ts, tf])
    all_v = np.concatenate([vals, vf])
    j = int(np.nanargmin(all_v))
    # the scan already holds the result at the best time (before 1.18 it
    # was computed a second time; the solver is deterministic, so the
    # numbers are the same)
    best = dict((coarse + fine)[j])
    best["t"] = float(all_t[j])
    best["scan_t"] = all_t
    best["scan_xi2"] = all_v
    return best


def _sorted_times(t_list):
    """(sorted times, order) for a list of output times in any order.
    Before 1.18 the Ramsey functions integrated up to the LAST listed time,
    so with t_list = [1e-3, 0.0] ramsey_cumulant returned the t = 0
    contrast 1.0 for t = 1 ms (and ramsey_meanfield raised a SciPy error)."""
    t = np.asarray(t_list, dtype=float).ravel()
    if t.size == 0 or not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("t_list must be a non-empty list of finite times >= 0 (s)")
    order = np.argsort(t, kind="stable")
    return t[order], order


def ramsey_cumulant(params: CavityParams, ens: Ensemble, t_list, **kw) -> dict:
    """Ramsey (no echo) evolution from the coherent spin state along +x:
    contrast, Wineland parameter xi2 and transverse variances var_min,
    var_max at each time of ``t_list`` (s, any order, returned in the
    order given)."""
    ts, order = _sorted_times(t_list)
    rt = Rates.from_params(params, ens)
    sts = evolve(css_x(rt.M), rt, float(ts[-1]), t_eval=ts, **kw)
    coh = np.empty(ts.size)
    xi = np.empty((ts.size, 5))
    coh[order] = [coherence(s, rt.n, spec_n=rt.spec_n) for s in sts]
    xi[order] = [wineland_xi2(s, rt.n, spec_n=rt.spec_n) for s in sts]
    return dict(t=np.asarray(t_list, dtype=float).ravel(), contrast=coh, xi2=xi[:, 0],
                var_min=xi[:, 2], var_max=xi[:, 3])


def ramsey_meanfield(params: CavityParams, ens: Ensemble, t_list, **kw) -> dict:
    """Ramsey (no echo) contrast from the mean-field equations at each time
    of ``t_list`` (s, any order, returned in the order given)."""
    ts, order = _sorted_times(t_list)
    rt = Rates.from_params(params, ens)
    s0 = np.full(rt.M, 0.5, complex)
    z0 = np.zeros(rt.M, complex)
    s, z = evolve_meanfield(s0, z0, rt, float(ts[-1]), t_eval=ts, **kw)
    Ntot = rt.N + rt.spec_n.sum()
    ph = np.outer(rt.spec_delta, ts)
    spec = 0.5 * np.exp(1j * ph) * np.exp(-rt.gamma_phi * ts)
    coh = np.empty(ts.size)
    coh[order] = 2.0 * np.abs(rt.n @ s + rt.spec_n @ spec) / Ntot
    return dict(t=np.asarray(t_list, dtype=float).ravel(), contrast=coh)


def twist_untwist(params: CavityParams, ens: Ensemble, t: float, phi: float = 1e-6, echo: bool = True,
                  sigma_det=0.0, weights=None, **kw):
    """Interaction-based readout (Davis, Bentsen, Schleier-Smith, PRL 116, 053601 (2016)):
    twist for time t with +chi, apply a small signal rotation phi about y, untwist for
    time t with -chi (cavity detuning reversed), then measure J_y.

    Returns the metrological gain relative to the standard quantum limit,
        G = (d<J_y>/d phi)^2 / [ N ( Var(J_y) + sigma_det^2 ) ],
    for each detection noise sigma_det (in units of spins), together with the
    amplification factor d<J_y>/dphi / (N/2)."""
    rt = Rates.from_params(params, ens)
    rt_neg = Rates.from_params(params.with_flipped_detuning(), ens)
    n = rt.n
    c = np.ones(rt.M) if weights is None else np.asarray(weights, float)
    S1 = float((n * c).sum())

    def final_state(ph):
        st = twist(css_x(rt.M), rt, t, echo=echo, **kw)
        st = rotate(st, Y, ph)
        st = twist(st, rt_neg, t, echo=echo, **kw)
        return st

    st_p = final_state(+phi)
    st_m = final_state(-phi)
    st_0 = final_state(0.0)
    Jp = collective_moments(st_p, n, weights, spec_n=rt.spec_n)[0]
    Jm = collective_moments(st_m, n, weights, spec_n=rt.spec_n)[0]
    J0, Cov0, S1, S2 = collective_moments(st_0, n, weights, spec_n=rt.spec_n)
    dJ = (Jp - Jm) / (2 * phi)
    # readout quadrature: the direction of the signal, projected perpendicular to the mean spin
    e3 = J0 / np.linalg.norm(J0)
    e = dJ - (dJ @ e3) * e3
    e = e / np.linalg.norm(e)
    slope = float(dJ @ e)
    var = float(e @ Cov0 @ e)
    sig = np.atleast_1d(np.asarray(sigma_det, float))
    # standard quantum limit for the (weighted) collective spin: slope S1/2, variance S2/4
    sql = (S1 / 2) ** 2 / (S2 / 4)
    gain = slope**2 / (var + sig**2) / sql
    return dict(gain=gain, amplification=slope / (S1 / 2), var_ratio=var / (S2 / 4), sigma_det=sig, t=t, J0=J0,
                contrast=np.linalg.norm(J0) / (S1 / 2))


def plain_squeezed_readout(params: CavityParams, ens: Ensemble, t: float, sigma_det=0.0, echo: bool = True,
                           weights=None, **kw) -> dict:
    """Gain of a conventional squeezed-state Ramsey readout: after twisting, the
    squeezed quadrature is rotated to the measurement axis (J_y) and a phase
    signal displaces the mean spin.  The signal slope is |<J>| and the noise is
    the squeezed variance plus detection noise, so
        Gain = 1/(xi^2 + 4 sigma_det^2 / (S2 C^2)),   C = 2|<J>|/S1
    (xi^2 is the Wineland parameter; before 1.17 this docstring left out the
    contrast C, which the code has always included)."""
    rt = Rates.from_params(params, ens)
    st = twist(css_x(rt.M), rt, t, echo=echo, **kw)
    xi2, ang, vmin, vmax, Jn = wineland_xi2(st, rt.n, weights, spec_n=rt.spec_n)
    _, _, S1, S2 = collective_moments(st, rt.n, weights, spec_n=rt.spec_n)
    sig = np.atleast_1d(np.asarray(sigma_det, float))
    sql = (S1 / 2) ** 2 / (S2 / 4)
    gain = Jn**2 / (vmin + sig**2) / sql
    return dict(gain=gain, xi2=xi2, sigma_det=sig, t=t)
