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

import numpy as np

__all__ = ["ramsey_sensitivity", "dick_fourier_coefficients",
           "dick_allan_deviation", "total_clock_allan_deviation",
           "power_law_psd", "synchronized_comparison"]


def _seg_E(k, t1, t2):
    """integral_t1^t2 exp(i k t) dt, exact (k = 0 included)."""
    if abs(k) < 1e-300:
        return complex(t2 - t1)
    return (np.exp(1j * k * t2) - np.exp(1j * k * t1)) / (1j * k)


def _seg_sin(a, phi, w, t1, t2):
    """integral_t1^t2 sin(a t + phi) exp(-i w t) dt, exact."""
    return (np.exp(1j * phi) * _seg_E(a - w, t1, t2)
            - np.exp(-1j * phi) * _seg_E(-a - w, t1, t2)) / 2j


def ramsey_sensitivity(t, T_ramsey, T_cycle, tau_pulse=0.0):
    """The Ramsey sensitivity function g(t) over one cycle.

    Layout: [0, tau_p] first pi/2 pulse (sin rise), [tau_p,
    tau_p + T] free evolution (g = 1), [tau_p + T, 2 tau_p + T]
    second pulse (sin fall), then dead time to T_cycle (g = 0).
    """
    t = np.asarray(t, dtype=float)
    T = float(T_ramsey)
    tp = float(tau_pulse)
    Tc = float(T_cycle)
    if T <= 0 or tp < 0 or Tc < T + 2 * tp:
        raise ValueError("need T_ramsey > 0, tau_pulse >= 0 and "
                         "T_cycle >= T_ramsey + 2 tau_pulse")
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


def dick_fourier_coefficients(m_max, T_ramsey, T_cycle, tau_pulse=0.0):
    """(g0, g_m) of the Ramsey sensitivity function, exactly.

    g_m = (1/Tc) integral_0^Tc g(t) exp(-2 pi i m t / Tc) dt for
    m = 1..m_max, each segment integrated in closed form.
    Returns (g0 float, gm complex array of length m_max).
    """
    T = float(T_ramsey)
    tp = float(tau_pulse)
    Tc = float(T_cycle)
    if T <= 0 or tp < 0 or Tc < T + 2 * tp:
        raise ValueError("need T_ramsey > 0, tau_pulse >= 0 and "
                         "T_cycle >= T_ramsey + 2 tau_pulse")
    m_max = int(m_max)
    if m_max < 1:
        raise ValueError("m_max must be >= 1")
    a = np.pi / (2 * tp) if tp > 0 else 0.0
    # g0: mean of g -- pulses each contribute integral of the sin arc
    area = T + (2 * tp) * (2 / np.pi) if tp > 0 else T
    g0 = area / Tc
    gm = np.empty(m_max, dtype=complex)
    for m in range(1, m_max + 1):
        w = 2 * np.pi * m / Tc
        val = _seg_E(-w, tp, tp + T)                # free evolution
        if tp > 0:
            val += _seg_sin(a, 0.0, w, 0.0, tp)     # rise
            # fall: sin(a (2 tp + T - t)) = sin(-a t + a (2 tp + T))
            val += _seg_sin(-a, a * (2 * tp + T), w, tp + T, 2 * tp + T)
        gm[m - 1] = val / Tc
    return g0, gm


def power_law_psd(h0=0.0, h_m1=0.0, h_m2=0.0):
    """One-sided fractional-frequency PSD S_y(f) = h0 + h_m1/f +
    h_m2/f^2 (white frequency, flicker frequency, random-walk
    frequency), the standard laboratory characterization of a clock
    laser. Returns a callable S_y(f_hz). Coefficients are YOUR
    laser's measured numbers; none are shipped."""
    h0, h_m1, h_m2 = float(h0), float(h_m1), float(h_m2)
    if min(h0, h_m1, h_m2) < 0:
        raise ValueError("PSD coefficients must be >= 0")

    def S_y(f):
        f = np.asarray(f, dtype=float)
        return h0 + h_m1 / f + h_m2 / f ** 2

    return S_y


def dick_allan_deviation(S_y, tau, T_ramsey, T_cycle, tau_pulse=0.0,
                         m_max=20000, tail_tol=1e-3):
    """Dick-effect-limited fractional-frequency Allan deviation.

    S_y : callable, one-sided LO fractional-frequency PSD (1/Hz) --
        e.g. from `power_law_psd` with your laser's measured
        coefficients.
    tau : averaging time (s). T_ramsey, T_cycle, tau_pulse : the
        interrogation cycle (s).
    m_max : harmonic cutoff; the truncation is CHECKED, not hoped
        for -- if the last decade of harmonics still contributes more
        than `tail_tol` of the sum, the cutoff is refused as too low.
    """
    g0, gm = dick_fourier_coefficients(m_max, T_ramsey, T_cycle,
                                       tau_pulse)
    Tc = float(T_cycle)
    m = np.arange(1, int(m_max) + 1)
    Sy_vals = np.asarray(S_y(m / Tc), dtype=float)
    terms = (np.abs(gm) ** 2 / g0 ** 2) * Sy_vals
    total = float(terms.sum())
    # the truncation check applies only when the coefficients exceed
    # floating-point noise (a continuously interrogated clock has
    # exactly zero coefficients, stored as ~1e-17 roundoff)
    noise_floor = (1e-12) ** 2 * float(Sy_vals.sum())
    if total > noise_floor:
        tail = float(terms[int(0.9 * m_max):].sum())
        if tail > tail_tol * total:
            raise ValueError(
                f"harmonic cutoff m_max = {m_max} is too low: the "
                f"last decade of harmonics still carries "
                f"{tail / total:.2%} of the sum. Raise m_max")
    return float(np.sqrt(total / float(tau)))


def total_clock_allan_deviation(dphi, nu0, T_ramsey, tau, T_cycle,
                                S_y=None, tau_pulse=0.0, m_max=20000):
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


def synchronized_comparison(dphi_a, dphi_b, nu0, T_ramsey, tau,
                            T_cycle=None):
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
