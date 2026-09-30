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
* Sample-variance uncertainty.  For M shots from any distribution the
  unbiased sample variance s^2 has the exact variance
  sigma^4 [kurtosis / M - (M - 3) / (M (M - 1))], which is
  2 sigma^4 / (M - 1) for Gaussian shots.  The default error bars use
  the Gaussian value (the same assumption the cumulant solver makes);
  shot_noise="empirical" (new in 1.16) uses each angle's measured
  kurtosis instead, so heavy-tailed detection noise no longer makes the
  error bars too small.
* Weights.  Each angle is weighted by 1 / Var(s^2).  Since 1.16 the
  sigma^4 in that weight is taken from the fitted curve and the fit is
  repeated until it is self-consistent (weighting="model"); taking it
  from the measured s^2 (weighting="sample", the only choice before)
  gives low-variance angles too much weight and biases the fit low.
* Error propagation.  The (c, a, b) covariance is the exact WLS
  covariance (or, with empirical shot noise, the exact covariance of
  the fixed-weight fit); V_min and xi^2 uncertainties follow by the
  delta method with the analytic gradient (1, -a/r, -b/r),
  r = sqrt(a^2 + b^2).
* A remaining bias, not corrected: V_min is the smaller of two noisy
  eigenvalues, so it is biased low.  For a coherent state (V1 = V2,
  C12 = 0) measured at equally spaced angles with many Gaussian shots
  the bias tends to sqrt(pi/3) = 1.02 reported error bars (the mean of
  the Rayleigh-distributed fitted anisotropy r).

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
subtraction and refusal on over-subtraction; refusals on degenerate
angle sets; and, since 1.16, the Var(s^2) identity by exact
enumeration, the fixed-weight covariance against Monte Carlo, the
model-weighted fit's lack of the sample-weight bias, empirical error
bars matching the scatter of heavy-tailed shots, and the coherent-state
V_min bias above.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

__all__ = ["SqueezingEstimate", "estimate_squeezing",
           "variance_tomography", "ContrastEstimate",
           "estimate_contrast", "sample_variance_sigma"]

_MAX_REWEIGHT = 1000


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
    weighting, shot_noise, n_reweight : how the fit was weighted
        ("model" or "sample") and how its error bars were computed (see
        `estimate_squeezing`), and the number of refits made while
        reweighting (0 for weighting="sample").  New in 1.16.
    weighting_fallback : None, or why weighting="model" was requested
        but the fit fell back to sample weights (see
        `estimate_squeezing`).  New in 1.16.
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
    weighting: str = "model"
    shot_noise: str = "gaussian"
    n_reweight: int = 0
    weighting_fallback: str | None = None


def variance_tomography(angles, variances, variance_sigmas,
                        error_sigmas=None):
    """Fit V(theta) = c + a cos 2theta + b sin 2theta by exact WLS.

    variance_sigmas set the fit weights 1/sigma^2.  By default they are
    also taken as the true standard errors of the variances, and the
    returned covariance of (c, a, b) is the usual WLS matrix
    (X^T W X)^-1.  If ``error_sigmas`` is given (new in 1.16), those are
    the true standard errors instead and the covariance is the exact
    "sandwich" of a linear estimator with fixed weights,

        (X^T W X)^-1 X^T W diag(error_sigmas^2) W X (X^T W X)^-1,

    which equals the usual matrix when error_sigmas == variance_sigmas.

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
    if error_sigmas is not None:
        es = np.asarray(error_sigmas, dtype=float).ravel()
        if es.shape != th.shape or not np.all(np.isfinite(es)) \
                or np.any(es <= 0.0):
            raise ValueError("error_sigmas must be finite, positive and "
                             "one per angle")
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
    if error_sigmas is not None:
        L = cov @ (X * (w * w)[:, None]).T       # (3, K): cab = L @ v
        cov = (L * es ** 2) @ L.T
        cov = 0.5 * (cov + cov.T)
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


@dataclasses.dataclass
class ContrastEstimate:
    """Ramsey fringe contrast fitted from a phase scan.

    contrast, contrast_sigma : the fringe contrast C in (0, 1] and
        its one-standard-deviation uncertainty -- exactly the two
        numbers `estimate_squeezing` asks for.
    phi0 : fitted fringe phase (radians).
    offset, offset_sigma : fitted constant offset of the mean J_z
        (spin units; nonzero offsets usually mean an imbalanced
        detection calibration, worth knowing about).
    amplitude : fitted fringe amplitude N C / 2 (spin units).
    cov_abd : (3, 3) WLS covariance of the (A, B, d) linear fit.
    n_phases, shots_per_phase : data bookkeeping.
    """

    contrast: float
    contrast_sigma: float
    phi0: float
    offset: float
    offset_sigma: float
    amplitude: float
    cov_abd: np.ndarray
    n_phases: int
    shots_per_phase: np.ndarray


def estimate_contrast(phases, shots, N) -> ContrastEstimate:
    """Fit the Ramsey fringe contrast from a phase-scan record --
    closing the loop `estimate_squeezing`'s docstring leaves open
    (new in v1.12).

    The record is the same shape as the tomography record: per-phase
    shot arrays of measured J_z, but scanned over the FRINGE phase
    (the final pi/2 pulse phase over a full period) rather than the
    tomography angle. The mean obeys

        <J_z>(phi) = A cos phi + B sin phi + d

    exactly (the rotation law of the mean spin, no lineshape
    assumption), so the fit is closed-form weighted linear least
    squares with per-phase standard errors of the mean, and the
    contrast is C = 2 sqrt(A^2 + B^2) / N with its uncertainty by the
    delta method through the exact WLS covariance -- the same
    structure, and the same honesty, as `variance_tomography`.

    Refusals instead of guesses: fewer than 3 distinct phases modulo
    2 pi (three basis functions need three quadratures), and a fitted
    contrast significantly above 1 (C - 2 sigma_C > 1), which is
    unphysical and usually means N is wrong or the J_z calibration is
    off -- reported as the likely cause rather than clipped.

    phases : (K,) fringe phases (radians).
    shots : length-K sequence of per-phase shot arrays (each >= 2), or
        a (K, M) array; measured J_z in spin units.
    N : number of spins (sets the normalization C = 2 |amplitude| / N).
    """
    ph = np.asarray(phases, dtype=float).ravel()
    rows = [np.asarray(s, dtype=float).ravel() for s in shots]
    if len(rows) != ph.size:
        raise ValueError("need one shot array per phase")
    if any(r.size < 2 for r in rows):
        raise ValueError("need at least 2 shots per phase")
    if any(not np.all(np.isfinite(r)) for r in rows):
        raise ValueError("shots contain non-finite values")
    if not np.all(np.isfinite(ph)):
        raise ValueError("phases must be finite")
    if not (isinstance(N, (int, np.integer)) and N >= 2):
        raise ValueError("N must be an integer >= 2")
    M = np.array([r.size for r in rows])
    mean = np.array([float(r.mean()) for r in rows])
    sem = np.array([float(r.std(ddof=1)) / np.sqrt(r.size)
                    for r in rows])
    if np.any(sem <= 0.0):
        raise ValueError("a per-phase shot array has zero scatter; "
                         "identical shots carry no error information")
    X = np.column_stack([np.cos(ph), np.sin(ph), np.ones_like(ph)])
    w = 1.0 / sem
    Xw = X * w[:, None]
    Amat = Xw.T @ Xw
    sv = np.linalg.svd(Amat, compute_uv=False)
    if sv[-1] <= 1e-10 * sv[0]:
        raise ValueError(
            "fringe phases are degenerate modulo 2 pi (fewer than 3 "
            "distinct quadratures); spread the phases over the fringe")
    abd = np.linalg.solve(Amat, Xw.T @ (mean * w))
    cov = np.linalg.inv(Amat)
    A, B, d = (float(x) for x in abd)
    r = float(np.hypot(A, B))
    C = 2.0 * r / N
    if r > 0.0:
        g = np.array([A / r, B / r, 0.0])
    else:
        g = np.array([1.0, 0.0, 0.0])
    C_sigma = 2.0 * float(np.sqrt(g @ cov @ g)) / N
    if C - 2.0 * C_sigma > 1.0:
        raise ValueError(
            f"fitted contrast {C:.4f} +/- {C_sigma:.4f} exceeds 1 "
            "beyond its uncertainty, which is unphysical: N is "
            "probably wrong, or the J_z calibration is off. Check "
            "both rather than clipping the contrast")
    # the fitted value is returned unclipped: a small statistical
    # overshoot above 1 is data, and rounding it down is the USER's
    # deliberate step before feeding estimate_squeezing (which
    # requires C <= 1), not this function's silent one
    return ContrastEstimate(
        contrast=float(C), contrast_sigma=float(C_sigma),
        phi0=float(np.arctan2(B, A)), offset=float(d),
        offset_sigma=float(np.sqrt(cov[2, 2])), amplitude=float(r),
        cov_abd=cov, n_phases=int(ph.size), shots_per_phase=M)


def sample_variance_sigma(shots, kurtosis=None):
    """Standard error of the unbiased sample variance of one shot array.

    For M independent shots from ANY distribution with variance
    sigma^2 and fourth central moment mu4, the sample variance s^2
    (ddof = 1) has the exact variance

        Var(s^2) = mu4 / M - sigma^4 (M - 3) / (M (M - 1))
                 = sigma^4 [kurtosis / M - (M - 3) / (M (M - 1))],

    with kurtosis = mu4 / sigma^4 (3 for Gaussian shots, where the
    bracket reduces exactly to 2 / (M - 1)).  This function returns the
    square root with sigma^2 replaced by s^2 and, unless ``kurtosis`` is
    given, the kurtosis replaced by the sample value m4 / m2^2 (central
    moments with 1/M normalization).  Those replacements are plug-in
    estimates, accurate to O(1/M); the formula itself is exact (checked
    in the tests by enumerating every outcome of a small discrete
    distribution).  New in 1.16.
    """
    x = np.asarray(shots, dtype=float).ravel()
    m = x.size
    if m < 2 or not np.all(np.isfinite(x)):
        raise ValueError("need at least 2 finite shots")
    s2 = float(np.var(x, ddof=1))
    return s2 * float(np.sqrt(_var_s2_factor(
        _sample_kurtosis(x) if kurtosis is None else float(kurtosis), m)))


def _sample_kurtosis(x):
    d = x - x.mean()
    m2 = float(np.mean(d * d))
    if m2 <= 0.0:
        raise ValueError("zero scatter: kurtosis undefined")
    return float(np.mean(d ** 4)) / m2 ** 2


def _var_s2_factor(kurtosis, m):
    """Var(s^2) / sigma^4 for m shots (exact identity, see
    `sample_variance_sigma`); always > 0 because kurtosis >= 1."""
    m = np.asarray(m, dtype=float)
    return np.asarray(kurtosis, dtype=float) / m - (m - 3.0) / (m * (m - 1.0))


def _reweight(th, s2, vdet, gauss, fit, sig_s2):
    """Iterate the model weights to self-consistency.  Returns (fit,
    sigmas, number of refits), or a string saying why it could not."""
    X = np.column_stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th)])
    n_refit = 0
    while True:
        raw = X @ np.array([fit["c"], fit["a"], fit["b"]]) + vdet
        if np.any(raw <= 0.0):
            return (f"the fitted variance curve (before subtracting "
                    f"detection variance {vdet:.3g}) reaches "
                    f"{raw.min():.3g} <= 0 at a measured angle, so it "
                    "cannot set the fit weights")
        new = raw * gauss
        if np.max(np.abs(new / sig_s2 - 1.0)) <= 1e-12:
            return fit, sig_s2, n_refit      # the weights are self-consistent
        if n_refit == _MAX_REWEIGHT:
            return (f"the model-weighted fit did not settle in "
                    f"{_MAX_REWEIGHT} reweighting steps")
        sig_s2 = new
        fit = variance_tomography(th, s2 - vdet, sig_s2)
        n_refit += 1


def estimate_squeezing(angles, shots, N, contrast, contrast_sigma=0.0,
                       detection_variance=0.0, weighting="model",
                       shot_noise="gaussian") -> SqueezingEstimate:
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
    weighting : "model" (default, new in 1.16) or "sample".  The fit
        weight of each angle is 1 / sigma_k^2 with sigma_k the
        standard error of that angle's sample variance, which is
        proportional to the variance itself.  "sample" takes it from
        the measured sample variance (the only choice before 1.16);
        that makes the weights depend on the data they weight, so
        angles whose variance happened to come out low count more, and
        the fitted curve is biased low (in the tests, 400 experiments
        with 9 angles x 20 shots: the mean of (V1 + V2)/2 is 14 % low).
        "model" takes it from the fitted curve V(theta_k) itself and
        repeats the fit until the weights stop changing (iteratively
        reweighted least squares); in the same test the mean is off
        by -0.2 % +/- 0.6 %, consistent with no bias.  V_min keeps a
        smaller low bias of a different origin (see the module
        docstring).  The model weights never refuse data the sample
        weights can fit: if the reweighting cannot proceed (the fitted
        curve dips to zero or below at a measured angle), does not
        settle, or ends with a minimal variance <= 0 while the sample
        fit's is positive, the result is the weighting="sample" fit, a
        RuntimeWarning says so, and ``weighting_fallback`` records why.
        This happens with few shots per angle (seeded runs, 6 angles,
        300 datasets each: 27-29 % of the datasets with 3 shots,
        11-13 % with 5, 1-2 % with 10; see the 1.16.0 changelog).  So
        the default refuses only data that weighting="sample" refuses
        too.
    shot_noise : "gaussian" (default) or "empirical" (new in 1.16).
        How the error bars treat the scatter of each sample variance.
        "gaussian" uses Var(s^2) = 2 sigma^4 / (M - 1), exact for
        Gaussian shots and too small for heavy-tailed ones.
        "empirical" uses the exact distribution-free identity
        Var(s^2) = sigma^4 [kurtosis / M - (M - 3) / (M (M - 1))] with
        each angle's measured kurtosis, and propagates it through the
        fixed-weight fit exactly (see `variance_tomography`'s
        ``error_sigmas``).  The point estimate is the same either way.

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
    if weighting not in ("model", "sample"):
        raise ValueError("weighting must be 'model' or 'sample'")
    if shot_noise not in ("gaussian", "empirical"):
        raise ValueError("shot_noise must be 'gaussian' or 'empirical'")
    M = np.array([r.size for r in rows])
    s2 = np.array([float(np.var(r, ddof=1)) for r in rows])
    if np.any(s2 <= 0.0):
        raise ValueError("a per-angle sample variance is zero; "
                         "identical shots carry no noise information")
    gauss = np.sqrt(2.0 / (M - 1))                # Gaussian shots
    over = ("the correction over-subtracts (or the data are "
            "degenerate); re-measure the detection noise")
    sig_s2 = s2 * gauss
    fit = variance_tomography(th, s2 - vdet, sig_s2)
    n_refit = 0
    used = weighting
    fallback = None
    if weighting == "model":
        sample_fit, sample_sig = fit, sig_s2
        fallback = _reweight(th, s2, vdet, gauss, fit, sig_s2)
        if isinstance(fallback, tuple):
            fit, sig_s2, n_refit = fallback
            fallback = None
            if fit["var_min"] <= 0.0 < sample_fit["var_min"]:
                fallback = (f"the model-weighted fit gives a minimal "
                            f"variance {fit['var_min']:.3g} <= 0")
        if fallback is not None and sample_fit["var_min"] <= 0.0:
            fit = sample_fit          # neither weighting can fit: refused below
        elif fallback is not None:
            # never refuse data that the 1.15 weights can fit: fall back
            # to them, say so, and record it in the result
            warnings.warn(
                f"estimate_squeezing: {fallback} (too few shots per angle "
                "for model weights); falling back to weighting='sample', "
                "whose fit is biased low with few shots. The result "
                "records this in weighting_fallback.",
                RuntimeWarning, stacklevel=2)
            fit, sig_s2, n_refit, used = sample_fit, sample_sig, 0, "sample"
    if shot_noise == "empirical":
        kurt = np.array([_sample_kurtosis(r) for r in rows])
        err = (sig_s2 / gauss) * np.sqrt(_var_s2_factor(kurt, M))
        fit = variance_tomography(th, s2 - vdet, sig_s2,
                                  error_sigmas=err)
    if fit["var_min"] <= 0.0:
        raise ValueError(
            f"fitted minimal variance {fit['var_min']:.3g} <= 0 after "
            f"subtracting detection variance {vdet:.3g}: {over}")
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
        shots_per_angle=M, weighting=used, shot_noise=shot_noise,
        n_reweight=int(n_refit), weighting_fallback=fallback)
