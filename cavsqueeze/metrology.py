"""Experiment-facing metrology projections from the solver output.

This module turns the collective moments computed by the cumulant solver
into the quantities an experimenter designs against, for any platform the
solver can describe (trapped ions, optical-lattice and tweezer clocks,
rare-earth spins, vapor-cell magnetometers): squeezing parameters, single
shot phase sensitivity, the projection-noise-limited Allan deviation of a
Ramsey clock, and the sensitivity of a Ramsey magnetometer.

Conventions and formulas (all standard):

* Kitagawa-Ueda parameter  xi_S^2 = Var_min(J_perp) / (S2/4), the minimal
  transverse variance normalized to its coherent-spin-state value
  (Kitagawa and Ueda, Phys. Rev. A 47, 5138 (1993)).  S2 reduces to N for
  uniform coupling weights.
* Wineland parameter  xi_R^2 = Var_min(J_perp) S1^2 / (|J|^2 S2), the
  phase-estimation figure of merit (Wineland et al., Phys. Rev. A 46,
  R6797 (1992)); xi_R^2 < 1 also witnesses entanglement (Sorensen et al.,
  Nature 409, 63 (2001)).  The two are related by
  xi_R^2 = xi_S^2 / C^2 with the contrast C = 2|J|/S1.
* Single-shot phase uncertainty of a Ramsey fringe, by linear error
  propagation:  dphi = sqrt(Var_min) / |J|; for a coherent spin state
  this is the standard quantum limit 1/sqrt(N).
* Projection-noise-limited Allan deviation of a Ramsey clock
  (Itano et al., Phys. Rev. A 47, 3554 (1993); Ludlow et al., Rev. Mod.
  Phys. 87, 637 (2015)):
      sigma_y(tau) = dphi / (2 pi nu0 T_R) * sqrt(T_c / tau)
  with clock frequency nu0, Ramsey interrogation time T_R, cycle time
  T_c >= T_R and averaging time tau.
* Ramsey magnetometer: a field B accumulates phase phi = gamma B T_R, so
      dB(tau) = dphi / (gamma T_R) * sqrt(T_c / tau)
  with gyromagnetic ratio gamma in rad/(s T).

Everything here is exact post-processing of the solver's first and second
moments; the only approximation is the linearized (small-angle) error
propagation behind dphi, the same approximation that defines xi_R.
"""
from __future__ import annotations

import numpy as np

from .cumulant import State, collective_moments, transverse_variances


def squeezing_parameters(st: State, n, weights=None, spec_n=None):
    """Kitagawa-Ueda and Wineland parameters and their ingredients.

    Returns a dict with xi2_S, xi2_KU (alias), xi2_R, contrast (2|J|/S1),
    var_min, var_max, angle, J (mean collective spin), S1, S2, and the
    single-shot phase uncertainty dphi = sqrt(var_min)/|J|.
    """
    J, Cov, S1, S2 = collective_moments(st, n, weights, spec_n=spec_n)
    Jn = float(np.linalg.norm(J))
    if Jn == 0.0:
        return dict(xi2_S=np.nan, xi2_KU=np.nan, xi2_R=np.inf, contrast=0.0,
                    var_min=np.nan, var_max=np.nan, angle=np.nan,
                    J=J, S1=S1, S2=S2, dphi=np.inf)
    vmin, vmax, ang, _ = transverse_variances(J, Cov)
    xi2_S = float(4.0 * vmin / S2)
    contrast = float(2.0 * Jn / S1)
    xi2_R = float(vmin * S1**2 / (Jn**2 * S2))
    dphi = float(np.sqrt(vmin) / Jn)
    return dict(xi2_S=xi2_S, xi2_KU=xi2_S, xi2_R=xi2_R, contrast=contrast,
                var_min=float(vmin), var_max=float(vmax), angle=float(ang),
                J=J, S1=S1, S2=S2, dphi=dphi)


def metrological_gain_db(xi2_R: float) -> float:
    """Gain over the standard quantum limit in dB: -10 log10(xi_R^2)."""
    return float(-10.0 * np.log10(xi2_R))


def phase_sensitivity(st: State, n, weights=None, spec_n=None) -> float:
    """Single-shot Ramsey phase uncertainty dphi = sqrt(Var_min)/|J| (rad).

    Equals 1/sqrt(N) for a coherent spin state with uniform coupling.
    """
    return squeezing_parameters(st, n, weights, spec_n=spec_n)["dphi"]


def clock_allan_deviation(dphi: float, nu0: float, T_ramsey: float,
                          tau: float, T_cycle: float | None = None) -> float:
    """Projection-noise-limited fractional-frequency Allan deviation.

    sigma_y(tau) = dphi / (2 pi nu0 T_ramsey) * sqrt(T_cycle / tau).

    Parameters: dphi single-shot phase uncertainty (rad, from
    phase_sensitivity), nu0 clock transition frequency (Hz), T_ramsey
    Ramsey interrogation time (s), tau averaging time (s), T_cycle full
    cycle time (s, defaults to T_ramsey; dead time makes it larger).
    """
    Tc = T_ramsey if T_cycle is None else T_cycle
    if Tc < T_ramsey:
        raise ValueError("T_cycle must be at least T_ramsey")
    return float(dphi / (2.0 * np.pi * nu0 * T_ramsey) * np.sqrt(Tc / tau))


def magnetometer_sensitivity(dphi: float, gamma: float, T_ramsey: float,
                             tau: float = 1.0, T_cycle: float | None = None) -> float:
    """Field uncertainty of a Ramsey magnetometer after averaging time tau.

    dB(tau) = dphi / (gamma T_ramsey) * sqrt(T_cycle / tau), with gamma the
    gyromagnetic ratio in rad/(s T).  With tau = 1 s this is the sensitivity
    in T/sqrt(Hz).
    """
    Tc = T_ramsey if T_cycle is None else T_cycle
    if Tc < T_ramsey:
        raise ValueError("T_cycle must be at least T_ramsey")
    return float(dphi / (gamma * T_ramsey) * np.sqrt(Tc / tau))


def oat_closed_form(N: int, mu: float):
    """Exact unitary one-axis-twisting moments (Kitagawa and Ueda,
    Phys. Rev. A 47, 5138 (1993)) for N spin-1/2 particles prepared in
    the coherent spin state along +x and evolved by U = exp(-i mu/2
    Jz^2) -- i.e. mu = 2 chi t for H = chi Jz^2.

    With A = 1 - cos^{N-2}(mu) and B = 4 sin(mu/2) cos^{N-2}(mu/2),

        <Jx>     = (N/2) cos^{N-1}(mu/2)
        <Jz^2>   = N/4                    (Jz is conserved)
        V(+/-)   = N/4 + N(N-1)/16 [A +/- sqrt(A^2 + B^2)]

    are exact for all N and mu; every one of these lines is asserted
    against brute-force exact evolution in the tests rather than
    trusted. Decoherence-free and homogeneous by construction: this is
    the benchmark the dissipative cumulant solver is compared against,
    not a substitute for it.

    Returns dict(Jx, Vmin, Vmax, alpha_min, xi2_S, xi2_R) where
    alpha_min is the transverse angle of the minimal variance
    (J(alpha) = cos(alpha) Jy + sin(alpha) Jz), xi2_S = Vmin/(N/4) the
    Kitagawa-Ueda parameter and xi2_R the Wineland parameter
    xi2_S (N/2)^2 / <Jx>^2.
    """
    N = int(N)
    if N < 2:
        raise ValueError("one-axis twisting needs N >= 2 spins")
    mu = float(mu)
    c, s = np.cos(mu / 2.0), np.sin(mu / 2.0)
    A = 1.0 - np.cos(mu) ** (N - 2)
    B = 4.0 * s * c ** (N - 2)
    rad = np.sqrt(A * A + B * B)
    pref = N * (N - 1) / 16.0
    Vmin = N / 4.0 + pref * (A - rad)
    Vmax = N / 4.0 + pref * (A + rad)
    Jx = (N / 2.0) * c ** (N - 1)
    alpha_min = 0.5 * np.arctan2(-B, -A)
    xi2_S = Vmin / (N / 4.0)
    xi2_R = xi2_S * (N / 2.0) ** 2 / Jx ** 2 if Jx != 0.0 else np.inf
    return dict(Jx=Jx, Vmin=float(Vmin), Vmax=float(Vmax),
                alpha_min=float(alpha_min), xi2_S=float(xi2_S),
                xi2_R=float(xi2_R))
