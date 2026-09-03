"""Resonator and material parameters for the adiabatically eliminated model.

All rates are angular frequencies (rad/s).  Helper constructors accept
ordinary frequencies in Hz and multiply by 2*pi.

The dispersive cavity-mediated interaction between spins j and k is
    chi_jk = chi * g_j * g_k,   chi = 4 * Delta / (4 Delta^2 + kappa^2),
and the collective (superradiant) decay rate is
    Gamma_jk = Gamma_SR * g_j * g_k,   Gamma_SR = 4 kappa / (4 Delta^2 + kappa^2),
so that for a uniform single-spin coupling g the usual expressions
chi g^2 = 4 g^2 Delta / (4 Delta^2 + kappa^2) and
Gamma_SR g^2 = 4 g^2 kappa / (4 Delta^2 + kappa^2) are recovered
[Fukumori et al., arXiv:2604.26909, Eqs. (S8)-(S9)].
"""
from __future__ import annotations

import dataclasses
import numpy as np

HBAR = 1.054571817e-34
KB = 1.380649e-23
TWO_PI = 2.0 * np.pi


def thermal_occupation(omega: float, temperature: float) -> float:
    """Bose-Einstein occupation at angular frequency omega (rad/s) and temperature (K)."""
    if temperature <= 0:
        return 0.0
    x = HBAR * omega / (KB * temperature)
    if x > 700:
        return 0.0
    return 1.0 / np.expm1(x)


@dataclasses.dataclass
class CavityParams:
    """Cavity and ensemble parameters (angular units).

    g      : single-spin coupling (rad/s) at unit weight; per-class weights
             multiply it (see Ensemble).
    kappa  : total cavity linewidth (rad/s).
    Delta  : cavity-spin detuning omega_c - omega_s (rad/s).
    omega_s: spin transition frequency (rad/s), used for thermal occupation.
    T      : effective temperature of the cavity bath (K).
    gamma_phi: single-spin pure dephasing rate (rad/s), 1/T2.
    """

    g: float
    kappa: float
    Delta: float
    omega_s: float = TWO_PI * 3.08385e9
    T: float = 0.0
    gamma_phi: float = 0.0

    @property
    def chi(self) -> float:
        """Dispersive exchange rate per unit g_j g_k (rad/s)."""
        return 4.0 * self.g**2 * self.Delta / (4.0 * self.Delta**2 + self.kappa**2)

    @property
    def Gamma_SR(self) -> float:
        """Collective decay rate per unit g_j g_k (rad/s)."""
        return 4.0 * self.g**2 * self.kappa / (4.0 * self.Delta**2 + self.kappa**2)

    @property
    def n_th(self) -> float:
        return thermal_occupation(self.omega_s, self.T)

    @property
    def Gamma_down(self) -> float:
        return self.Gamma_SR * (self.n_th + 1.0)

    @property
    def Gamma_up(self) -> float:
        return self.Gamma_SR * self.n_th

    def with_flipped_detuning(self) -> "CavityParams":
        """Same cavity with Delta -> -Delta (chi -> -chi, Gamma_SR unchanged)."""
        return dataclasses.replace(self, Delta=-self.Delta)


def from_hz(g_hz, kappa_hz, Delta_hz, omega_s_hz=3.08385e9, T=0.0, T2=None) -> CavityParams:
    """Build CavityParams from ordinary frequencies in Hz."""
    gamma_phi = 0.0 if T2 is None else 1.0 / T2
    return CavityParams(
        g=TWO_PI * g_hz,
        kappa=TWO_PI * kappa_hz,
        Delta=TWO_PI * Delta_hz,
        omega_s=TWO_PI * omega_s_hz,
        T=T,
        gamma_phi=gamma_phi,
    )


def loop_gap_dispersive(N: float, T: float = 0.08, T2: float = 0.15) -> tuple[CavityParams, float]:
    """The dispersive loop-gap resonator of Fukumori et al. (arXiv:2604.26909):
    g/2pi = 15 mHz, kappa/2pi = 660 kHz, Delta/2pi = 22 MHz, T2 > 150 ms.
    Returns (params, N)."""
    return from_hz(15e-3, 660e3, 22e6, T=T, T2=T2), float(N)
