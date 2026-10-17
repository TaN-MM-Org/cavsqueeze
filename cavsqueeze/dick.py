"""The Dick effect: when squeezing helps a clock, and when the local
oscillator has already decided (new in v1.13).

A pulsed clock samples its local oscillator (LO) only during part of
each cycle, so LO frequency noise at harmonics of the cycle rate
aliases into the measurement. That aliasing -- the Dick effect (G. J.
Dick, Proc. PTTI 1987; G. Santarelli et al., IEEE Trans. Ultrason.
Ferroelectr. Freq. Control 45, 887 (1998)) -- limits the fractional
Allan variance to

    sigma_y^2(tau) = (1 / (tau g0^2)) sum_{m>=1} |g_m|^2 S_y(m / Tc),

with g_m = (1/Tc) integral_0^Tc g(t) exp(-2 pi i m t / Tc) dt the
Fourier coefficients of the sensitivity function over the cycle, g0
its mean, and S_y the one-sided fractional-frequency noise PSD of the
free-running LO (A. Quessada et al., J. Opt. B 5, S150 (2003); the
formula above is quoted in that convention). The Dick floor is
independent of the atom number and of entanglement, which is exactly
why it matters here: squeezing buys nothing once a clock is
Dick-limited, the central caveat of the squeezing-enhanced-clock
analysis of Schulte et al., Nat. Commun. 11, 5955 (2020), and the
regime today's spin-squeezed lattice-clock comparisons (Robinson et
al., Nat. Phys. 2024) engineer around. This module computes the floor
so `cavsqueeze`'s projection-noise numbers can be compared against it
honestly.

The Ramsey sensitivity function implemented is the standard
on-resonance form: rising as sin(pi t / (2 tau_p)) during the first
pi/2 pulse, 1 during free evolution, falling symmetrically during the
second pulse, 0 during dead time. Its Fourier coefficients are
computed in closed form per segment (every segment integral reduces
to exact exponential integrals, with the resonant k = 0 case handled
exactly, not by a small denominator).

Anchors asserted in the tests rather than stated: the closed-form
coefficients equal adaptive numerical quadrature of the sensitivity
function to 1e-9 (two independent code paths); zero dead time with
negligible pulses gives zero Dick variance (every coefficient below
1e-12; the aliasing vanishes when the LO is watched continuously); in
the short-pulse limit the coefficient ratio equals the rectangular-
window form sin(pi m eta)/(pi m eta) to 1e-12; the floor scales as
1/sqrt(tau) to 1e-12; and doubling the harmonic cutoff moves the
answer by less than 1e-3.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["ramsey_sensitivity", "dick_fourier_coefficients",
           "dick_allan_deviation", "total_clock_allan_deviation",
           "power_law_psd", "synchronized_comparison"]


def _check_timing(T, Tc, tp):
    """Refuse an impossible Ramsey cycle.  Since 1.17 non-finite times are
    refused too (before, a nan or infinite T_cycle, or a nan T_ramsey or
    tau_pulse, passed the comparisons below and gave a nan Dick floor)."""
    if not (np.isfinite(T) and np.isfinite(Tc) and np.isfinite(tp)):
        raise ValueError(f"T_ramsey, T_cycle and tau_pulse must be finite (s); got "
                         f"T_ramsey={T}, T_cycle={Tc}, tau_pulse={tp}")
    if T <= 0 or tp < 0 or Tc < T + 2 * tp:
        raise ValueError("need T_ramsey > 0, tau_pulse >= 0 and "
                         "T_cycle >= T_ramsey + 2 tau_pulse "
                         f"(got T_ramsey={T}, T_cycle={Tc}, tau_pulse={tp})")


def _seg_E(k, t1, t2):
    """integral_t1^t2 exp(i k t) dt, exact (k = 0 included)."""
    if abs(k) < 1e-300:
        return complex(t2 - t1)
    return (np.exp(1j * k * t2) - np.exp(1j * k * t1)) / (1j * k)


def _seg_sin(a, phi, w, t1, t2):
    """integral_t1^t2 sin(a t + phi) exp(-i w t) dt, exact."""
    return (np.exp(1j * phi) * _seg_E(a - w, t1, t2)
            - np.exp(-1j * phi) * _seg_E(-a - w, t1, t2)) / 2j


def _seg_E_vec(k, t1, t2):
    """`_seg_E` for an array of k, element by element the same arithmetic."""
    k = np.asarray(k, dtype=float)
    zero = np.abs(k) < 1e-300
    kk = np.where(zero, 1.0, k)
    out = (np.exp(1j * kk * t2) - np.exp(1j * kk * t1)) / (1j * kk)
    out[zero] = complex(t2 - t1)
    return out


def _seg_sin_vec(a, phi, w, t1, t2):
    """`_seg_sin` for an array of w."""
    return (np.exp(1j * phi) * _seg_E_vec(a - w, t1, t2)
            - np.exp(-1j * phi) * _seg_E_vec(-a - w, t1, t2)) / 2j


def ramsey_sensitivity(t, T_ramsey: float, T_cycle: float,
                       tau_pulse: float = 0.0) -> np.ndarray:
    """The Ramsey sensitivity function g(t) over one cycle.

    Layout: [0, tau_p] first pi/2 pulse (sin rise), [tau_p,
    tau_p + T] free evolution (g = 1), [tau_p + T, 2 tau_p + T]
    second pulse (sin fall), then dead time to T_cycle (g = 0).
    """
    t = np.asarray(t, dtype=float)
    T = float(T_ramsey)
    tp = float(tau_pulse)
    Tc = float(T_cycle)
    _check_timing(T, Tc, tp)
    tt = np.mod(t, Tc)
    g = np.zeros_like(tt)
    if tp > 0:
        m1 = tt < tp
        g[m1] = np.sin(np.pi * tt[m1] / (2 * tp))
        m3 = (tt >= tp + T) & (tt < 2 * tp + T)
        g[m3] = np.sin(np.pi * (2 * tp + T - tt[m3]) / (2 * tp))
    m2 = (tt >= tp) & (tt < tp + T)
    g[m2] = 1.0
    return g


def dick_fourier_coefficients(m_max: int, T_ramsey: float, T_cycle: float,
                              tau_pulse: float = 0.0
                              ) -> tuple[float, np.ndarray]:
    """(g0, g_m) of the Ramsey sensitivity function, exactly.

    g_m = (1/Tc) integral_0^Tc g(t) exp(-2 pi i m t / Tc) dt for
    m = 1..m_max, each segment integrated in closed form.
    Returns (g0 float, gm complex array of length m_max).
    """
    T = float(T_ramsey)
    tp = float(tau_pulse)
    Tc = float(T_cycle)
    _check_timing(T, Tc, tp)
    if not (np.isfinite(m_max) and float(m_max) == np.floor(m_max) and m_max >= 1):
        # before 1.18 a fractional cutoff was truncated (2.7 -> 2 harmonics)
        raise ValueError(f"m_max must be a whole number >= 1; got {m_max}")
    m_max = int(m_max)
    a = np.pi / (2 * tp) if tp > 0 else 0.0
    # g0: mean of g -- pulses each contribute integral of the sin arc
    area = T + (2 * tp) * (2 / np.pi) if tp > 0 else T
    g0 = area / Tc
    # all harmonics at once (since 1.18; the same arithmetic as the
    # former loop over m, element by element -- the tests check that the
    # result is bit-identical to that loop)
    w = 2 * np.pi * np.arange(1, m_max + 1) / Tc
    val = _seg_E_vec(-w, tp, tp + T)                # free evolution
    if tp > 0:
        val += _seg_sin_vec(a, 0.0, w, 0.0, tp)     # rise
        # fall: sin(a (2 tp + T - t)) = sin(-a t + a (2 tp + T))
        val += _seg_sin_vec(-a, a * (2 * tp + T), w, tp + T, 2 * tp + T)
    gm = val / Tc
    return g0, gm


def power_law_psd(h0: float = 0.0, h_m1: float = 0.0,
                  h_m2: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    """One-sided fractional-frequency PSD S_y(f) = h0 + h_m1/f +
    h_m2/f^2 (white frequency, flicker frequency, random-walk
    frequency), the standard laboratory characterization of a clock
    laser. Returns a callable S_y(f_hz). Coefficients are YOUR
    laser's measured numbers; none are shipped."""
    h0, h_m1, h_m2 = float(h0), float(h_m1), float(h_m2)
    if not all(np.isfinite(h) and h >= 0 for h in (h0, h_m1, h_m2)):
        raise ValueError("PSD coefficients must be finite and >= 0 "
                         f"(got h0={h0}, h_m1={h_m1}, h_m2={h_m2})")

    def S_y(f):
        f = np.asarray(f, dtype=float)
        return h0 + h_m1 / f + h_m2 / f ** 2

    return S_y


def _tail_start(m_max: int) -> int:
    """Index of the first harmonic in the cutoff check: the top 10 %, but
    at least the top 100 harmonics (at most the top half)."""
    m_max = int(m_max)
    return min(int(0.9 * m_max), m_max - min(100, m_max // 2))


def dick_allan_deviation(S_y: Callable, tau: float, T_ramsey: float,
                         T_cycle: float, tau_pulse: float = 0.0,
                         m_max: int = 20000, tail_tol: float = 1e-3) -> float:
    """Dick-effect-limited fractional-frequency Allan deviation.

    S_y : callable, one-sided LO fractional-frequency PSD (1/Hz) --
        e.g. from `power_law_psd` with your laser's measured
        coefficients.
    tau : averaging time (s). T_ramsey, T_cycle, tau_pulse : the
        interrogation cycle (s).
    m_max : harmonic cutoff; the truncation is CHECKED, not hoped
        for: if the top 10 % of the harmonics (m > 0.9 m_max) still
        contribute more than `tail_tol` of the sum, the cutoff is
        refused as too low.  Since 1.18 the checked window holds at least
        the top 100 harmonics (the top half when m_max < 200): a window
        of a few harmonics can fall on zeros of the coefficients (with
        m_max = 10 and T_ramsey / T_cycle = 0.4 the only checked harmonic,
        m = 10, is exactly zero), and 1.17 then accepted a result 2.1 %
        short of the converged one.  The check bounds the omitted remainder
        only roughly: for terms falling as 1/m^2 (sharp pulse edges
        with white frequency noise, the slowest decay a
        `power_law_psd` spectrum gives) the harmonics beyond m_max
        carry about 9 times what the top 10 % carry, so the relative
        error of the Allan variance can reach
        about 9 tail_tol.  At the default m_max and with instantaneous
        pulses, the Allan variance for white frequency noise agrees with
        its closed form h0 (1 - eta) / (2 eta tau),
        eta = T_ramsey / T_cycle, to 6e-5 (tested for eta = 0.1, 0.4
        and 0.9); the shortfall is the omitted remainder, about
        1 / (pi^2 eta (1 - eta) m_max) of the total.
    """
    if not (np.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be finite and positive (s); got {tau}")
    if not (np.isfinite(tail_tol) and tail_tol > 0):
        # before 1.18 tail_tol = nan switched the cutoff check off silently
        raise ValueError(f"tail_tol must be finite and positive; got {tail_tol}")
    g0, gm = dick_fourier_coefficients(m_max, T_ramsey, T_cycle,
                                       tau_pulse)
    Tc = float(T_cycle)
    m = np.arange(1, int(m_max) + 1)
    Sy_vals = np.asarray(S_y(m / Tc), dtype=float)
    if not (np.all(np.isfinite(Sy_vals)) and np.all(Sy_vals >= 0.0)):
        # before 1.18 a spectrum with nan or negative values (for example a
        # measured PSD interpolated outside its range) gave a nan floor
        raise ValueError("S_y must be finite and >= 0 at every harmonic "
                         f"m / T_cycle, m = 1..{int(m_max)} (1/Hz)")
    terms = (np.abs(gm) ** 2 / g0 ** 2) * Sy_vals
    total = float(terms.sum())
    # the truncation check applies only when the coefficients exceed
    # floating-point noise (a continuously interrogated clock has
    # exactly zero coefficients, stored as ~1e-17 roundoff)
    noise_floor = (1e-12) ** 2 * float(Sy_vals.sum())
    if total > noise_floor:
        tail = float(terms[_tail_start(m_max):].sum())
        if tail > tail_tol * total:
            raise ValueError(
                f"harmonic cutoff m_max = {m_max} is too low: the "
                f"top harmonics (m > {_tail_start(m_max)}) still carry "
                f"{tail / total:.2%} of the sum. Raise m_max")
    return float(np.sqrt(total / float(tau)))


def total_clock_allan_deviation(dphi: float, nu0: float, T_ramsey: float,
                                tau: float, T_cycle: float,
                                S_y: Callable | None = None,
                                tau_pulse: float = 0.0,
                                m_max: int = 20000) -> dict:
    """Quantum projection noise and the Dick floor, combined.

    Returns dict(qpn, dick, total, dick_limited): the projection-
    noise-limited deviation (`clock_allan_deviation`, which is where
    squeezing enters through dphi), the Dick floor (independent of
    atom number and entanglement), their quadrature sum, and whether
    the Dick floor exceeds the projection noise -- the regime where
    squeezing buys nothing until the LO or the duty cycle improves
    (Schulte et al., Nat. Commun. 11, 5955 (2020)).
    """
    from .metrology import clock_allan_deviation
    qpn = clock_allan_deviation(dphi, nu0, T_ramsey, tau,
                                T_cycle=T_cycle)
    if S_y is None:
        dick = 0.0
    else:
        dick = dick_allan_deviation(S_y, tau, T_ramsey, T_cycle,
                                    tau_pulse, m_max)
    return dict(qpn=float(qpn), dick=float(dick),
                total=float(np.hypot(qpn, dick)),
                dick_limited=bool(dick > qpn))


def synchronized_comparison(dphi_a: float, dphi_b: float, nu0: float,
                            T_ramsey: float, tau: float,
                            T_cycle: float | None = None) -> dict:
    """Two ensembles interrogated synchronously by ONE local
    oscillator: the Dick-free comparison (new in v1.15).

    The Dick effect aliases the local oscillator's noise into any
    single clock's record, but two ensembles read out with the SAME
    oscillator in the SAME cycles see that noise as common mode, so
    their frequency DIFFERENCE is free of it and averages down at the
    projection-noise floor alone -- the protocol behind the
    entanglement-enhanced comparisons of Robinson et al., Nat. Phys.
    20, 208 (2024) (1.9(2) dB of stability enhancement at the 10^-17
    level) and Yang et al., PRL 135, 193202 (2025) (2.0(2) dB beyond
    the standard quantum limit, 1.1 x 10^-18 single-clock precision).
    The modeling statement here is exactly that common-mode
    rejection: this function combines the two ensembles' quantum
    projection noise and reports NO Dick term for the difference.
    What it does not model, stated plainly: any noise that is NOT
    common mode (differential path lengths, gradients) adds on top.

    dphi_a, dphi_b : single-shot phase uncertainties of the two
        ensembles (rad, e.g. from `phase_sensitivity`).
    nu0, T_ramsey, tau, T_cycle : as in `clock_allan_deviation`.

    Returns dict(qpn_a, qpn_b, differential, per_clock): the two
    single-ensemble projection-noise deviations, their quadrature sum
    (the instability of the difference), and the conventional
    per-clock figure differential/sqrt(2), exact for two equal
    ensembles -- an identity the tests assert.
    """
    from .metrology import clock_allan_deviation
    for name, v in (("dphi_a", dphi_a), ("dphi_b", dphi_b)):
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be finite and positive "
                             "(rad)")
    qa = clock_allan_deviation(dphi_a, nu0, T_ramsey, tau, T_cycle)
    qb = clock_allan_deviation(dphi_b, nu0, T_ramsey, tau, T_cycle)
    diff = float(np.hypot(qa, qb))
    return {"qpn_a": qa, "qpn_b": qb, "differential": diff,
            "per_clock": diff / np.sqrt(2.0)}
