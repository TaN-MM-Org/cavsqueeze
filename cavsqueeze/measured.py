"""Squeezing estimation from measured tomography data.

Every other module of this package predicts squeezing from a model;
this one estimates it from an experiment.  The standard measurement
(Ramsey tomography of the transverse spin) is: prepare the state,
rotate the transverse plane by a known angle theta about the mean-spin
axis, measure the population difference J_z (in spin units,
(N_up - N_down)/2), and repeat for M shots at each of K angles.  The
per-angle shot variances trace out the transverse covariance, and the
minimal variance together with the Ramsey contrast gives the Wineland
parameter -- the same estimand, in the same convention, as the
solver's `wineland_xi2`.

The estimator leans only on exact structure:

* The rotation identity.  Second moments of a rotated quadrature obey

      V(theta) = c + a cos(2 theta) + b sin(2 theta),

  with c = (V1 + V2)/2, a = (V1 - V2)/2, b = C12 -- exactly, for ANY
  state, because it is the transformation law of a 2x2 covariance
  matrix, not a Gaussian assumption.  Fitting (c, a, b) is therefore
  LINEAR weighted least squares with a closed-form solution, and

      V_min/max = c -/+ sqrt(a^2 + b^2),   2 theta_min = atan2(-b, -a)

  are the eigenvalues and eigenvector angle of the covariance
  (asserted against `numpy.linalg.eigvalsh` in the tests).
* Sample-variance uncertainty.  The variance of the unbiased sample
  variance of M Gaussian shots is exactly 2 s^4 / (M - 1); this is
  the one place a Gaussian shot distribution IS assumed, and the
  docstring says so instead of hiding it (for the states this package
  treats, second-order cumulant physics, the assumption is the same
  one the solver itself makes; heavy-tailed detection noise inflates
  these error bars).
* Error propagation.  The (c, a, b) covariance is the exact WLS
  covariance; V_min and xi^2 uncertainties follow by the delta method
  with the analytic gradient (1, -a/r, -b/r), r = sqrt(a^2 + b^2).

Detection noise: a known detection variance (measured on a reference
state, in spin units squared) may be subtracted per angle before the
fit; this is the standard correction, it is OPT-IN and recorded in the
result, and an over-subtraction that drives the fitted minimal
variance at or below zero raises instead of reporting unphysical
infinite squeezing.

Normalization: for N uniformly coupled spins the returned parameters
are xi2_S = 4 V_min / N (Kitagawa-Ueda) and xi2_R = xi2_S / C^2
(Wineland) with C the Ramsey contrast in (0, 1] -- algebraically
identical to `wineland_xi2`'s vmin S1^2/(|J|^2 S2) at S1 = S2 = N,
|J| = C N / 2.  The contrast and its uncertainty come from the user's
fringe fit; this module does not guess them.

Anchors, asserted in the tests rather than stated: exact recovery of a
known covariance from noise-free variances, with V_min/V_max equal to
the eigenvalues and theta_min to the eigenvector angle; the exact
Kitagawa-Ueda one-axis-twisting closed form recovered from
synthetically sampled tomography shots within statistical tolerance;
a coherent-spin-state sample estimating xi2_R compatible with 1 (the
standard quantum limit); Monte-Carlo scatter of the estimate matching
its reported sigma; exact round trip of the detection-noise
subtraction and refusal on over-subtraction; and refusals on
degenerate angle sets.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["SqueezingEstimate", "estimate_squeezing",
           "variance_tomography"]


@dataclasses.dataclass
class SqueezingEstimate:
    """Result of a tomography-based squeezing estimate.

    xi2_S, xi2_R : Kitagawa-Ueda and Wineland parameters (linear
        units; metrological gain in dB is -10 log10(xi2_R), see
        `metrological_gain_db`).
    xi2_S_sigma, xi2_R_sigma : one-standard-deviation uncertainties
        (delta method through the exact WLS covariance, plus the
        contrast error for xi2_R).
    var_min, var_max, var_min_sigma : transverse variances in spin
        units squared, after any detection-noise subtraction.
    theta_min : tomography angle of minimal variance (radians).
    V1, V2, C12 : the fitted transverse covariance matrix elements.
    cov_cab : (3, 3) WLS covariance of (c, a, b).
    contrast, contrast_sigma, N, detection_variance : the inputs that
        entered the normalization, recorded for provenance.
    n_angles, shots_per_angle : data bookkeeping.
    """

    xi2_S: float
    xi2_R: float
    xi2_S_sigma: float
    xi2_R_sigma: float
    var_min: float
    var_max: float
    var_min_sigma: float
    theta_min: float
    V1: float
    V2: float
    C12: float
    cov_cab: np.ndarray
    contrast: float
    contrast_sigma: float
    N: int
    detection_variance: float
    n_angles: int
    shots_per_angle: np.ndarray


def variance_tomography(angles, variances, variance_sigmas):
    """Fit V(theta) = c + a cos 2theta + b sin 2theta by exact WLS.

    Returns dict(c, a, b, cov (3, 3), var_min, var_max, var_min_sigma,
    theta_min, V1, V2, C12).  Refuses fewer than 3 angles or an
    angle set that is degenerate modulo pi (the three basis functions
    must be independent on the design).
    """
    th = np.asarray(angles, dtype=float).ravel()
    v = np.asarray(variances, dtype=float).ravel()
    sg = np.asarray(variance_sigmas, dtype=float).ravel()
    if not (th.shape == v.shape == sg.shape):
        raise ValueError("angles, variances and sigmas must share a "
                         "shape")
    if th.size < 3:
        raise ValueError("need at least 3 tomography angles for the 3 "
                         "covariance parameters")
    if not np.all(np.isfinite(th)) or not np.all(np.isfinite(v)) \
            or not np.all(np.isfinite(sg)) or np.any(sg <= 0.0):
        raise ValueError("inputs must be finite with positive sigmas")
    X = np.column_stack([np.ones_like(th), np.cos(2 * th),
                         np.sin(2 * th)])
    w = 1.0 / sg
    Xw = X * w[:, None]
    A = Xw.T @ Xw
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[-1] <= 1e-10 * sv[0]:
        raise ValueError(
            "tomography angles are degenerate modulo pi (fewer than 3 "
            "distinct quadratures); spread the angles")
    cab = np.linalg.solve(A, Xw.T @ (v * w))
    cov = np.linalg.inv(A)
    c, a, b = (float(x) for x in cab)
    r = float(np.hypot(a, b))
    var_min = c - r
    var_max = c + r
    theta_min = 0.5 * float(np.arctan2(-b, -a))
    if r > 0.0:
        g = np.array([1.0, -a / r, -b / r])
    else:                       # isotropic: variance independent of theta
        g = np.array([1.0, 0.0, 0.0])
    var_min_sigma = float(np.sqrt(g @ cov @ g))
    return dict(c=c, a=a, b=b, cov=cov, var_min=var_min,
                var_max=var_max, var_min_sigma=var_min_sigma,
                theta_min=theta_min, V1=c + a, V2=c - a, C12=b)


def estimate_squeezing(angles, shots, N, contrast, contrast_sigma=0.0,
                       detection_variance=0.0) -> SqueezingEstimate:
    """Estimate xi2_S and xi2_R from tomography shot records.

    Parameters
    ----------
    angles : (K,) tomography rotation angles (radians) about the
        mean-spin axis.
    shots : length-K sequence of per-angle shot arrays (each >= 2
        shots), or a (K, M) array; values are measured J_z in spin
        units, (N_up - N_down)/2.
    N : number of (uniformly coupled) spins.
    contrast, contrast_sigma : Ramsey contrast C in (0, 1] and its
        one-standard-deviation uncertainty, from the user's fringe
        fit.
    detection_variance : known detection-noise variance (spin units
        squared) subtracted from every per-angle sample variance
        before the fit; 0 means no correction.  An over-subtraction
        that drives the fitted minimal variance to <= 0 raises.

    Returns a :class:`SqueezingEstimate`.
    """
    th = np.asarray(angles, dtype=float).ravel()
    rows = [np.asarray(s, dtype=float).ravel() for s in shots]
    if len(rows) != th.size:
        raise ValueError("need one shot array per angle")
    if any(r.size < 2 for r in rows):
        raise ValueError("need at least 2 shots per angle")
    if any(not np.all(np.isfinite(r)) for r in rows):
        raise ValueError("shots contain non-finite values")
    if not (isinstance(N, (int, np.integer)) and N >= 2):
        raise ValueError("N must be an integer >= 2")
    if not (0.0 < contrast <= 1.0) or contrast_sigma < 0.0:
        raise ValueError("contrast must lie in (0, 1] with a "
                         "nonnegative sigma")
    vdet = float(detection_variance)
    if vdet < 0.0 or not np.isfinite(vdet):
        raise ValueError("detection_variance must be finite and >= 0")
    M = np.array([r.size for r in rows])
    s2 = np.array([float(np.var(r, ddof=1)) for r in rows])
    if np.any(s2 <= 0.0):
        raise ValueError("a per-angle sample variance is zero; "
                         "identical shots carry no noise information")
    sig_s2 = s2 * np.sqrt(2.0 / (M - 1))          # Gaussian shots
    fit = variance_tomography(th, s2 - vdet, sig_s2)
    if fit["var_min"] <= 0.0:
        raise ValueError(
            f"fitted minimal variance {fit['var_min']:.3g} <= 0 after "
            f"subtracting detection variance {vdet:.3g}: the "
            "correction over-subtracts (or the data are degenerate); "
            "re-measure the detection noise")
    vmin, svmin = fit["var_min"], fit["var_min_sigma"]
    xi2_S = 4.0 * vmin / N
    xi2_S_sigma = 4.0 * svmin / N
    xi2_R = xi2_S / contrast**2
    xi2_R_sigma = xi2_R * float(np.hypot(
        svmin / vmin, 2.0 * contrast_sigma / contrast))
    return SqueezingEstimate(
        xi2_S=float(xi2_S), xi2_R=float(xi2_R),
        xi2_S_sigma=float(xi2_S_sigma), xi2_R_sigma=float(xi2_R_sigma),
        var_min=float(vmin), var_max=float(fit["var_max"]),
        var_min_sigma=float(svmin), theta_min=float(fit["theta_min"]),
        V1=float(fit["V1"]), V2=float(fit["V2"]), C12=float(fit["C12"]),
        cov_cab=fit["cov"], contrast=float(contrast),
        contrast_sigma=float(contrast_sigma), N=int(N),
        detection_variance=vdet, n_angles=int(th.size),
        shots_per_angle=M)
