"""Plan a tomography measurement before taking it.

`cavsqueeze.measured` estimates squeezing from tomography shot records
after the experiment; this module answers the two questions that come
BEFORE it: how well would this set of angles and shot counts pin the
squeezing down, and how many shots does a target error bar cost?

Both answers use exactly the machinery the estimator itself uses --
the closed-form weighted-least-squares covariance of the
V(theta) = c + a cos(2 theta) + b sin(2 theta) fit, with the same
per-angle sample-variance uncertainty s^2 sqrt(2/(M-1)) (the one
place a Gaussian shot distribution is assumed, stated in
`cavsqueeze.measured` and inherited here). Because the fit is linear,
the planned covariance is not an approximation around a guess: it is
the same matrix `variance_tomography` will report, computed from the
design alone. The tests assert that equality to machine precision.

Two facts worth knowing when choosing angles, both exact:

* K equally spaced angles over half a turn (theta_k = k pi / K,
  K >= 3) make the three basis functions exactly orthogonal on the
  design, so no angle wastes shots on redundancy. With equal weights
  the normal matrix is exactly diag(K, K/2, K/2).
* The contrast uncertainty puts a floor under the Wineland-parameter
  error bar that no number of tomography shots removes:
  xi2_R_sigma >= xi2_R * 2 sigma_C / C. `shots_for_squeezing` refuses
  a target below that floor and names it, instead of returning an
  absurd shot count.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plan_tomography", "shots_for_squeezing"]


def _expected_profile(angles, var_min, var_max, theta_min):
    """Exact rotation law: the variance profile of a state with
    principal variances (var_min, var_max) along theta_min."""
    th = np.asarray(angles, dtype=float).ravel()
    c = 0.5 * (var_min + var_max)
    r = 0.5 * (var_max - var_min)
    return c - r * np.cos(2.0 * (th - float(theta_min))), th


def plan_tomography(angles, shots_per_angle, var_min, var_max,
                    theta_min=0.0, N=None, contrast=1.0,
                    contrast_sigma=0.0, detection_variance=0.0):
    """Predicted error bars of a planned tomography measurement.

    angles : (K,) planned tomography angles (radians).
    shots_per_angle : shots at each angle (scalar or (K,), each >= 2).
    var_min, var_max : the transverse variances you expect to see
        (spin units squared) -- e.g. N/4 and N/4 for an unsqueezed
        state, or the solver's prediction for a squeezed one.
    theta_min : expected angle of minimal variance (radians).
    N, contrast, contrast_sigma : supply them to also get the
        predicted xi2_S and xi2_R error bars, with the same
        normalization `estimate_squeezing` uses.
    detection_variance : known detection noise (spin units squared)
        you intend to subtract, which enters the weights exactly as it
        will in the estimate.

    Returns dict(identifiable, condition_number, var_min_sigma,
    cov_cab, and -- when N is given -- xi2_S, xi2_S_sigma, xi2_R,
    xi2_R_sigma). A degenerate angle set (fewer than 3 distinct
    quadratures modulo pi) is reported as identifiable=False, the same
    arithmetic on which `variance_tomography` refuses after the fact.
    """
    if not (np.isfinite(var_min) and np.isfinite(var_max)
            and 0.0 < var_min <= var_max):
        raise ValueError("need 0 < var_min <= var_max (spin units "
                         "squared)")
    vdet = float(detection_variance)
    if vdet < 0.0 or not np.isfinite(vdet):
        raise ValueError("detection_variance must be finite and >= 0")
    v_true, th = _expected_profile(angles, var_min, var_max, theta_min)
    if th.size < 1 or not np.all(np.isfinite(th)):
        raise ValueError("angles must be finite")
    M = np.broadcast_to(np.asarray(shots_per_angle, dtype=int),
                        th.shape).copy()
    if np.any(M < 2):
        raise ValueError("need at least 2 shots per angle")
    s2_raw = v_true + vdet                 # what the shots will show
    sig = s2_raw * np.sqrt(2.0 / (M - 1))  # Gaussian sample variance
    X = np.column_stack([np.ones_like(th), np.cos(2 * th),
                         np.sin(2 * th)])
    w = 1.0 / sig
    Xw = X * w[:, None]
    A = Xw.T @ Xw
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    identifiable = bool(th.size >= 3 and sv[-1] > 1e-10 * sv[0])
    out = {"identifiable": identifiable, "condition_number": cond,
           "var_min_sigma": None, "cov_cab": None}
    if not identifiable:
        return out
    cov = np.linalg.inv(A)
    c = 0.5 * (var_min + var_max)
    r = 0.5 * (var_max - var_min)
    a = -r * np.cos(2.0 * float(theta_min))
    b = -r * np.sin(2.0 * float(theta_min))
    rr = float(np.hypot(a, b))
    # same delta-method gradient as `variance_tomography`
    g = np.array([1.0, -a / rr, -b / rr]) if rr > 0.0 \
        else np.array([1.0, 0.0, 0.0])
    var_min_sigma = float(np.sqrt(g @ cov @ g))
    out.update(var_min_sigma=var_min_sigma, cov_cab=cov)
    if N is not None:
        if not (isinstance(N, (int, np.integer)) and N >= 2):
            raise ValueError("N must be an integer >= 2")
        if not (0.0 < contrast <= 1.0) or contrast_sigma < 0.0:
            raise ValueError("contrast must lie in (0, 1] with a "
                             "nonnegative sigma")
        xi2_S = 4.0 * var_min / N
        xi2_R = xi2_S / contrast ** 2
        out.update(
            xi2_S=float(xi2_S),
            xi2_S_sigma=float(4.0 * var_min_sigma / N),
            xi2_R=float(xi2_R),
            xi2_R_sigma=float(xi2_R * np.hypot(
                var_min_sigma / var_min,
                2.0 * contrast_sigma / contrast)))
    return out


def shots_for_squeezing(target_xi2_R_sigma, angles, var_min, var_max,
                        N, contrast, contrast_sigma=0.0,
                        theta_min=0.0, detection_variance=0.0):
    """How many shots per angle does a target error bar cost?

    Exact inversion of the planned covariance: every per-angle weight
    scales as sqrt(M - 1), so the tomography part of the Wineland
    error bar scales as 1/sqrt(M - 1) exactly, and the smallest
    integer M meeting the target follows in closed form -- no search.

    The contrast term does not shrink with tomography shots: a target
    below the floor xi2_R * 2 sigma_C / C is refused by that exact
    argument, with the floor named, because more tomography cannot buy
    it -- only a better fringe measurement can.

    Returns (shots_per_angle, plan) where `plan` is the
    `plan_tomography` result at the returned shot number.
    """
    t = float(target_xi2_R_sigma)
    if not (np.isfinite(t) and t > 0.0):
        raise ValueError("target_xi2_R_sigma must be positive")
    base = plan_tomography(angles, 2, var_min, var_max, theta_min, N,
                           contrast, contrast_sigma,
                           detection_variance)
    if not base["identifiable"]:
        raise ValueError("the angle set is degenerate modulo pi "
                         "(fewer than 3 distinct quadratures); spread "
                         "the angles")
    xi2_R = base["xi2_R"]
    floor = xi2_R * 2.0 * contrast_sigma / contrast
    if t <= floor:
        raise ValueError(
            f"target {t:.3g} is below the contrast floor {floor:.3g} "
            "= xi2_R * 2 sigma_C / C: no number of tomography shots "
            "reaches it -- improve the fringe (contrast) measurement "
            "instead")
    # tomography term at M shots: tom(M) = tom(M=2) / sqrt(M - 1)
    tom_at_2 = xi2_R * (base["var_min_sigma"] / var_min)
    need = np.sqrt(max(t ** 2 - floor ** 2, 0.0))
    m = int(np.ceil(1.0 + (tom_at_2 / need) ** 2))
    m = max(m, 2)
    plan = plan_tomography(angles, m, var_min, var_max, theta_min, N,
                           contrast, contrast_sigma,
                           detection_variance)
    return m, plan
