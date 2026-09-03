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
