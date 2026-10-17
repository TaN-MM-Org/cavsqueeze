"""Coherence time and line width from a measured Ramsey or echo decay
(new in 1.17).

The solver needs two numbers that a laboratory measures on its own
ensemble: the single-spin dephasing rate gamma_phi = 1/T2 (`from_hz`
takes T2) and the shape and full width at half maximum (FWHM) of the
inhomogeneous line (`lineshape`).  This module fits them to measured
contrast decays, with error bars and a check that the data agree with
the model.

What the decays are, in this package's model (no interaction, which
holds when the cavity-mediated rates are negligible on the time scale
of the measurement; otherwise the fitted rates include them):

* Echo (a pi pulse half way, t = total free-evolution time): the static
  spread of frequencies is refocused, and pure dephasing
  L_j = sqrt(gamma_phi / 2) sigma z_j leaves

      C(t) = A exp(-gamma_phi t),         gamma_phi = 1 / T2.

* Ramsey free-induction decay (FID): every spin also turns at its own
  detuning delta, so the mean spin is multiplied by the characteristic
  function of the line, |integral p(delta) exp(i delta t) d delta|.
  For a Gaussian line of FWHM F (rad/s), standard deviation
  s = F / (2 sqrt(2 ln 2)), and for a Lorentzian line of FWHM F_L
  (rad/s), these factors are exp(-s^2 t^2 / 2) and exp(-F_L t / 2):

      Gaussian line:    C(t) = A exp(-gamma_phi t - s^2 t^2 / 2)
      Lorentzian line:  C(t) = A exp(-(gamma_phi + F_L / 2) t)

  A Voigt line (new in 1.18) is the convolution of a Gaussian and a
  Lorentzian, so its factor is the product of theirs (the Fourier
  transform of a convolution is the product of the transforms):

      Voigt line:       C(t) = A exp(-(gamma_phi + F_L / 2) t - s^2 t^2 / 2)

  The fit returns the total FWHM of the Voigt profile (from the exact
  profile, scipy.special.voigt_profile) and lorentz_fraction = F_L / FWHM,
  the two numbers `lineshape("voigt", fwhm, lorentz_fraction)` takes.
  The tests check it on decays computed by numerical Fourier integration
  of that profile.

  A is the contrast at t = 0 (pulse errors and readout make it less
  than 1).  The tests check these forms against the package's own
  solvers (the cumulant solver with an echo, and the mean-field solver
  on a Gaussian line cut into 20000 classes).

The fit is weighted nonlinear least squares on C(t) itself (the
logarithm would make the errors non-Gaussian where the contrast is
small), started from a weighted linear fit of log C.  Error bars come from the inverse
of the curvature matrix J^T W J at the optimum, the standard linearized
(delta-method) result; they assume the given contrast errors are right,
and are not rescaled by the fit quality.  The fit quality is reported
separately: chi2 = sum ((C_k - model_k) / sigma_k)^2, its degrees of
freedom (points minus fitted parameters) and the p-value of a chi-square
distribution with that many degrees of freedom.  A small p-value means
the data do not follow the model within their error bars (for example a
stretched-exponential echo decay from spectral diffusion, which this
model does not contain; since 1.18 `fit_stretched_decay` fits that form
and tests it against the exponential).  In the tests, over 300 seeded noisy datasets,
the scatter of each fitted width and rate is 0.8 to 1.2 times its
reported error bar, and 2 % to 9 % of the p-values fall below 0.05.

Refusals, rather than numbers that look fine: a Lorentzian or Voigt width
without a known T2 (in a Ramsey decay the Lorentzian part and gamma_phi
both decay as exp(-rate t), so only their sum can be measured; fit an
echo first and pass its T2), too few points, non-finite or non-positive
inputs, and a fitted width or rate that is not at least
``min_significance`` (default 2) error bars above zero (the data then
do not resolve it, and the error bar of T2 = 1/gamma_phi or of the
width would be meaningless).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import optimize, special, stats

__all__ = ["DecayFit", "fit_coherence_decay", "StretchedDecayFit", "fit_stretched_decay"]

_K_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # Gaussian FWHM per standard deviation


@dataclasses.dataclass
class DecayFit:
    """Result of `fit_coherence_decay`.  All rates and widths are angular
    frequencies (rad/s), as everywhere in the package.

    line : "none", "gaussian", "lorentzian" or "voigt" (the model fitted).
    amplitude, amplitude_sigma : the contrast A at t = 0.
    gamma_phi, gamma_phi_sigma : the dephasing rate 1/T2 (rad/s);
        `CavityParams(gamma_phi=...)` takes it directly.
    T2, T2_sigma : 1/gamma_phi and its error bar (s); `from_hz(T2=...)`
        takes it.  When T2 was given to the fit, these are the given
        values.
    fwhm, fwhm_sigma : FWHM of the line (rad/s), as `lineshape(line,
        fwhm)` takes it; None for line="none".  For line="voigt" (new in
        1.18) the total FWHM of the Voigt line.
    lorentz_fraction, lorentz_fraction_sigma : line="voigt" only (new in
        1.18): the Lorentzian part's FWHM as a fraction of the total, as
        `lineshape("voigt", fwhm, lorentz_fraction)` takes it.  None
        otherwise.
    fwhm_gaussian, fwhm_lorentzian : line="voigt" only (new in 1.18): the
        FWHM of the Gaussian and of the Lorentzian part (rad/s).
    T2_given : whether T2 was given (and only the line was fitted).
    names, values, cov : the fitted parameters, in the model
        C(t) = A exp(-rate t - s2 t^2 / 2) (names among "A", "rate",
        "s2"; rate in rad/s, s2 in rad^2/s^2), and their covariance.
        With line="gaussian" and a given T2, the rate is fixed at 1/T2
        and the covariance includes the contribution of T2_sigma; with
        line="lorentzian" or "voigt" the fitted rate is the total
        1/T2 + (Lorentzian FWHM)/2, and T2_sigma enters the line's error
        bars only.
    chi2, dof, p_value : the fit quality (see the module docstring).
    n_points : number of data points.
    """

    line: str
    amplitude: float
    amplitude_sigma: float
    gamma_phi: float
    gamma_phi_sigma: float
    T2: float
    T2_sigma: float
    fwhm: float | None
    fwhm_sigma: float | None
    T2_given: bool
    names: tuple
    values: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    p_value: float
    n_points: int
    lorentz_fraction: float | None = None
    lorentz_fraction_sigma: float | None = None
    fwhm_gaussian: float | None = None
    fwhm_lorentzian: float | None = None


def _voigt_fwhm(sigma: float, gamma: float) -> float:
    """Full width at half maximum (rad/s) of the Voigt profile with Gaussian
    standard deviation sigma and Lorentzian half width gamma, from the exact
    profile (scipy.special.voigt_profile) by root finding."""
    if gamma == 0.0:
        return float(_K_FWHM * sigma)
    if sigma == 0.0:
        return float(2.0 * gamma)
    half = 0.5 * float(special.voigt_profile(0.0, sigma, gamma))
    hi = gamma + sigma
    while float(special.voigt_profile(hi, sigma, gamma)) >= half:
        hi *= 2.0
    x = optimize.brentq(lambda u: float(special.voigt_profile(u, sigma, gamma)) - half,
                        0.0, hi, xtol=1e-15 * hi, rtol=1e-14)
    return float(2.0 * x)


def _voigt_width_and_fraction(gamma: float, s2: float) -> np.ndarray:
    """(total FWHM, lorentz_fraction) of the Voigt line with Lorentzian half
    width gamma and Gaussian variance s2, in the convention of
    `lineshape("voigt", fwhm, lorentz_fraction)`."""
    fwhm = _voigt_fwhm(float(np.sqrt(s2)), float(gamma))
    return np.array([fwhm, 2.0 * gamma / fwhm])


def _model(theta, names, fixed_rate, tau):
    """C(tau) and its Jacobian with respect to the free parameters, in
    time units scaled so that the longest time is 1."""
    p = dict(zip(names, theta))
    A = p["A"]
    rate = p.get("rate", fixed_rate)
    s2 = p.get("s2", 0.0)
    e = np.exp(-rate * tau - 0.5 * s2 * tau * tau)
    f = A * e
    cols = {"A": e, "rate": -tau * f, "s2": -0.5 * tau * tau * f}
    return f, np.column_stack([cols[n] for n in names]), -tau * f


def fit_coherence_decay(times, contrast, contrast_sigma, line: str = "none",
                        T2: float | None = None, T2_sigma: float = 0.0,
                        min_significance: float = 2.0) -> DecayFit:
    """Fit a measured echo or Ramsey contrast decay (new in 1.17).

    times : (K,) free-evolution times (s), >= 0.  For an echo, the total
        time of both halves.
    contrast, contrast_sigma : (K,) measured contrasts and their one-
        standard-deviation errors (for example `estimate_contrast` at
        each time).
    line : "none" (an echo, or the decay of a line much narrower than
        1/T2), "gaussian", "lorentzian" or "voigt" (new in 1.18; a
        Ramsey decay of a line of that shape).  See the module docstring
        for the formulas.  For "voigt" both the Lorentzian and the
        Gaussian part must be resolved (at least ``min_significance``
        error bars above zero), or the fit refuses and names the simpler
        line to use.
    T2, T2_sigma : a known coherence time and its error (s), from an
        echo fit with line="none".  Required for "lorentzian" and "voigt"
        (the only way to separate the line from gamma_phi); optional for
        "gaussian", where it makes the fit more precise when the Ramsey
        decay itself barely resolves gamma_phi.  Its error is propagated
        into the fitted width (linearized: the change of the fit with
        T2, times T2_sigma, added in quadrature).
    min_significance : a fitted rate or width must lie at least this many
        error bars above zero, or the fit refuses (default 2).

    Returns a :class:`DecayFit`.  Raises ValueError when the data cannot
    answer the question (see the module docstring).
    """
    t = np.asarray(times, dtype=float).ravel()
    y = np.asarray(contrast, dtype=float).ravel()
    sg = np.asarray(contrast_sigma, dtype=float).ravel()
    if not (t.shape == y.shape == sg.shape):
        raise ValueError("times, contrast and contrast_sigma must have the same "
                         f"length (got {t.size}, {y.size}, {sg.size})")
    if not (np.all(np.isfinite(t)) and np.all(t >= 0.0)):
        raise ValueError("times must be finite and >= 0 (s)")
    if not np.all(np.isfinite(y)):
        raise ValueError("contrast values must be finite")
    if not (np.all(np.isfinite(sg)) and np.all(sg > 0.0)):
        raise ValueError("contrast_sigma values must be finite and positive")
    if line not in ("none", "gaussian", "lorentzian", "voigt"):
        raise ValueError("line must be 'none', 'gaussian', 'lorentzian' or 'voigt'; "
                         f"got {line!r}")
    if not (np.isfinite(min_significance) and min_significance >= 0.0):
        raise ValueError(f"min_significance must be finite and >= 0; got {min_significance}")
    if T2 is not None:
        if not (np.isfinite(T2) and T2 > 0.0):
            raise ValueError(f"T2 must be finite and positive (s); got {T2}")
        if not (np.isfinite(T2_sigma) and T2_sigma >= 0.0):
            raise ValueError(f"T2_sigma must be finite and >= 0 (s); got {T2_sigma}")
    elif T2_sigma != 0.0:
        raise ValueError("T2_sigma is given without T2")
    if line == "none" and T2 is not None:
        raise ValueError("line='none' fits T2 itself; do not pass T2")
    if line in ("lorentzian", "voigt") and T2 is None:
        raise ValueError(
            "a Lorentzian width cannot be separated from gamma_phi = 1/T2 in a "
            "Ramsey decay (both decay as exp(-rate t), so only their sum is "
            "measured): fit an echo decay with line='none' first and pass its "
            "T2 and T2_sigma")

    if line == "gaussian":
        names = ("A", "s2") if T2 is not None else ("A", "rate", "s2")
    elif line == "voigt":
        names = ("A", "rate", "s2")
    else:
        names = ("A", "rate")
    npar = len(names)
    if t.size < npar + 1:
        raise ValueError(f"need at least {npar + 1} points to fit {npar} parameters "
                         f"and test the fit; got {t.size}")
    if np.unique(t).size < npar:
        raise ValueError(f"need at least {npar} distinct times; got {np.unique(t).size}")

    ts = float(t.max())
    if ts <= 0.0:
        raise ValueError("all times are zero; the decay is not sampled")
    tau = t / ts
    fixed_rate = 0.0 if T2 is None else ts / T2     # scaled 1/T2

    # starting point: weighted linear fit of log C on the points with C > 0
    pos = y > 0.0
    cols = {"A": np.ones_like(tau), "rate": -tau, "s2": -0.5 * tau * tau}
    x0 = None
    if np.count_nonzero(pos) >= npar:
        X = np.column_stack([cols[n] for n in names])[pos]
        w = y[pos] / sg[pos]                     # 1 / sigma(log C)
        rhs = np.log(y[pos]) + (fixed_rate * tau[pos] if "rate" not in names else 0.0)
        sol, *_ = np.linalg.lstsq(X * w[:, None], rhs * w, rcond=None)
        if np.all(np.isfinite(sol)):
            x0 = sol.copy()
            x0[0] = np.exp(sol[0])
    if x0 is None:
        x0 = np.array([max(float(np.max(np.abs(y))), 1e-3)] + [1.0] * (npar - 1))

    def resid(theta):
        return (_model(theta, names, fixed_rate, tau)[0] - y) / sg

    def jac(theta):
        return _model(theta, names, fixed_rate, tau)[1] / sg[:, None]

    res = optimize.least_squares(resid, x0, jac=jac, method="lm",
                                 xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                 max_nfev=2000 * (npar + 1))
    if res.status <= 0:
        raise ValueError(f"the fit did not converge: {res.message}")
    theta = res.x
    Jw = jac(theta)
    F = Jw.T @ Jw
    sv = np.linalg.svd(F, compute_uv=False)
    if not (sv[-1] > 1e-12 * sv[0]):
        raise ValueError("the parameters are not separately determined by these "
                         "times (the curvature matrix is singular); spread the "
                         "times over the decay")
    cov = np.linalg.inv(F)
    if T2 is not None and T2_sigma > 0.0 and "rate" not in names:
        # linearized response of the fit to the fixed rate: d theta / d rate
        dfd_rate = _model(theta, names, fixed_rate, tau)[2] / sg
        dtheta = -cov @ (Jw.T @ dfd_rate)
        rate_sigma = ts * T2_sigma / T2 ** 2     # scaled sigma of 1/T2
        cov = cov + np.outer(dtheta, dtheta) * rate_sigma ** 2
    cov = 0.5 * (cov + cov.T)

    # back to physical units: rate in 1/s, s2 in 1/s^2
    scale = np.array([{"A": 1.0, "rate": 1.0 / ts, "s2": 1.0 / ts ** 2}[n] for n in names])
    values = theta * scale
    cov = cov * np.outer(scale, scale)
    sig = np.sqrt(np.diag(cov))
    p = dict(zip(names, values))
    ps = dict(zip(names, sig))

    def resolved(name, value, sigma, label, hint=""):
        if not value > min_significance * sigma:
            raise ValueError(
                f"the fitted {label} {value:.4g} +/- {sigma:.4g} is not "
                f"{min_significance:g} error bars above zero: these data do not "
                "resolve it. Measure further into the decay, reduce the "
                "contrast errors, or use a model without it" +
                (" (for a Ramsey decay dominated by the line, fit an echo with "
                 "line='none' and pass its T2)" if name == "rate" and line == "gaussian"
                 else "") + hint)

    fwhm = fwhm_sigma = None
    if T2 is None:
        resolved("rate", p["rate"], ps["rate"],
                 "dephasing rate 1/T2 (rad/s)")
        g_phi, g_phi_sigma = p["rate"], ps["rate"]
        T2_out, T2_out_sigma = 1.0 / g_phi, g_phi_sigma / g_phi ** 2
    else:
        g_phi, g_phi_sigma = 1.0 / T2, T2_sigma / T2 ** 2
        T2_out, T2_out_sigma = float(T2), float(T2_sigma)
    if line == "gaussian":
        resolved("s2", p["s2"], ps["s2"], "Gaussian variance of the line (rad^2/s^2)")
        sd = np.sqrt(p["s2"])
        fwhm = _K_FWHM * sd
        fwhm_sigma = _K_FWHM * ps["s2"] / (2.0 * sd)
    elif line == "lorentzian":
        half = p["rate"] - g_phi                   # F_L / 2
        half_sigma = float(np.hypot(ps["rate"], g_phi_sigma))
        resolved("width", half, half_sigma,
                 "Lorentzian half width (total rate minus 1/T2, rad/s)")
        fwhm, fwhm_sigma = 2.0 * half, 2.0 * half_sigma
    voigt = {}
    if line == "voigt":
        gam = p["rate"] - g_phi                    # Lorentzian half width
        gam_sigma = float(np.sqrt(cov[1, 1] + g_phi_sigma ** 2))
        resolved("width", gam, gam_sigma,
                 "Lorentzian half width (total rate minus 1/T2, rad/s)",
                 " (a Gaussian line: line='gaussian')")
        resolved("s2", p["s2"], ps["s2"],
                 "Gaussian variance of the line (rad^2/s^2)",
                 " (a Lorentzian line: line='lorentzian')")
        # covariance of (gamma, s2): the fit's, plus the independent error
        # of the given 1/T2 in gamma
        cgs = np.array([[cov[1, 1] + g_phi_sigma ** 2, cov[1, 2]],
                        [cov[2, 1], cov[2, 2]]])
        x0 = np.array([gam, p["s2"]])
        out = _voigt_width_and_fraction(*x0)
        # gradient by central differences of the exact width (a smooth
        # function; steps of 1e-6 of each value)
        jac = np.empty((2, 2))
        for i in range(2):
            h = 1e-6 * x0[i]
            up, dn = x0.copy(), x0.copy()
            up[i] += h
            dn[i] -= h
            jac[:, i] = (_voigt_width_and_fraction(*up) - _voigt_width_and_fraction(*dn)) / (2 * h)
        cov_out = jac @ cgs @ jac.T
        fwhm, fwhm_sigma = float(out[0]), float(np.sqrt(cov_out[0, 0]))
        voigt = dict(lorentz_fraction=float(out[1]),
                     lorentz_fraction_sigma=float(np.sqrt(cov_out[1, 1])),
                     fwhm_gaussian=float(_K_FWHM * np.sqrt(p["s2"])),
                     fwhm_lorentzian=float(2.0 * gam))

    r = resid(theta)
    chi2 = float(r @ r)
    dof = int(t.size - npar)
    return DecayFit(
        line=line, amplitude=float(p["A"]), amplitude_sigma=float(ps["A"]),
        gamma_phi=float(g_phi), gamma_phi_sigma=float(g_phi_sigma),
        T2=float(T2_out), T2_sigma=float(T2_out_sigma),
        fwhm=None if fwhm is None else float(fwhm),
        fwhm_sigma=None if fwhm_sigma is None else float(fwhm_sigma),
        T2_given=T2 is not None, names=names, values=values, cov=cov,
        chi2=chi2, dof=dof, p_value=float(stats.chi2.sf(chi2, dof)),
        n_points=int(t.size), **voigt)


@dataclasses.dataclass
class StretchedDecayFit:
    """Result of `fit_stretched_decay` (new in 1.18).

    amplitude, amplitude_sigma : the contrast A at t = 0.
    T_M, T_M_sigma : the time at which the decay has fallen to A/e (s).
    exponent, exponent_sigma : the stretch exponent x of
        A exp(-(t / T_M)^x); x = 1 is the exponential decay of the model.
    cov : (3, 3) covariance of (A, T_M, x).
    chi2, dof, p_value : fit quality of the stretched form (dof = points
        minus 3).
    chi2_exponential, delta_chi2, p_exponential : the same data fitted with
        x fixed at 1 (the exponential decay `fit_coherence_decay` fits with
        line="none"), the drop in chi2 when x is freed, and the p-value of
        that drop for one degree of freedom.  A small p_exponential says the
        decay is not exponential.
    n_points : number of data points.
    """

    amplitude: float
    amplitude_sigma: float
    T_M: float
    T_M_sigma: float
    exponent: float
    exponent_sigma: float
    cov: np.ndarray
    chi2: float
    dof: int
    p_value: float
    chi2_exponential: float
    delta_chi2: float
    p_exponential: float
    n_points: int


def _stretched(theta, tau):
    """A exp(-(tau/u)^x) and its Jacobian in (A, u, x), tau = t / t_max."""
    A, u, x = theta
    r = tau / u
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(r > 0.0, np.log(np.where(r > 0.0, r, 1.0)), 0.0)
    q = np.where(r > 0.0, r ** x, 0.0)
    e = np.exp(-q)
    f = A * e
    jac = np.column_stack([e, f * q * x / u, -f * q * lr])
    return f, jac


def fit_stretched_decay(times, contrast, contrast_sigma,
                        min_significance: float = 2.0) -> StretchedDecayFit:
    """Fit a measured echo decay with the stretched exponential
    C(t) = A exp(-(t / T_M)^x) (new in 1.18).

    In solids the echo decay is often not exponential: spectral diffusion
    gives decays of this stretched form, with x > 1.  The form
    exp(-(2 tau / T_M)^x) (2 tau the total echo time, here t) is the one
    used for rare-earth spin and optical echoes, for example by M. Le Dantec
    et al., arXiv:2106.14974 (2021) (who fit its square, the echo
    intensity), citing T. Bottger, C. W. Thiel, Y. Sun and R. L. Cone,
    Phys. Rev. B 73, 075101 (2006).

    This package's model has only Markovian dephasing, which gives x = 1
    (`fit_coherence_decay` with line="none").  This fit measures x and
    tests x = 1: if p_exponential is small, the echo is not exponential,
    and T2 = 1/gamma_phi from an exponential fit is not a parameter of the
    model but an average over a decay it does not describe.  T_M is then
    the 1/e time of the measured decay.

    times, contrast, contrast_sigma : as in `fit_coherence_decay` (times in
        s, >= 0; contrast errors finite and positive).
    min_significance : T_M and x must each lie at least this many error
        bars above zero, or the fit refuses (default 2).

    The fit is weighted nonlinear least squares on C(t); error bars come
    from the inverse curvature matrix (the standard linearized result) and
    assume the given contrast errors are right.  delta_chi2 follows a
    chi-square law with one degree of freedom when x = 1 holds and the
    errors are Gaussian (checked by Monte Carlo in the tests).  Refusals:
    fewer than 4 points or 3 distinct times, invalid inputs, a decay that
    the times do not sample (the fit does not converge, or the curvature
    matrix is singular), and an exponent or T_M not resolved.
    """
    t = np.asarray(times, dtype=float).ravel()
    y = np.asarray(contrast, dtype=float).ravel()
    sg = np.asarray(contrast_sigma, dtype=float).ravel()
    if not (t.shape == y.shape == sg.shape):
        raise ValueError("times, contrast and contrast_sigma must have the same "
                         f"length (got {t.size}, {y.size}, {sg.size})")
    if not (np.all(np.isfinite(t)) and np.all(t >= 0.0)):
        raise ValueError("times must be finite and >= 0 (s)")
    if not np.all(np.isfinite(y)):
        raise ValueError("contrast values must be finite")
    if not (np.all(np.isfinite(sg)) and np.all(sg > 0.0)):
        raise ValueError("contrast_sigma values must be finite and positive")
    if not (np.isfinite(min_significance) and min_significance >= 0.0):
        raise ValueError(f"min_significance must be finite and >= 0; got {min_significance}")
    if t.size < 4:
        raise ValueError(f"need at least 4 points to fit 3 parameters and test the fit; got {t.size}")
    if np.unique(t[t > 0.0]).size < 3:
        raise ValueError("need at least 3 distinct times > 0")

    # the exponential fit (x = 1) by the 1.17 routine: the starting point,
    # and the reference for the test of x = 1
    try:
        expo = fit_coherence_decay(t, y, sg, line="none", min_significance=0.0)
    except ValueError as err:
        raise ValueError(f"the decay is not sampled well enough to fit: {err}") from None
    ts = float(t.max())
    tau = t / ts

    def resid(theta):
        return (_stretched(theta, tau)[0] - y) / sg

    def jac(theta):
        return _stretched(theta, tau)[1] / sg[:, None]

    u0 = 1.0 / max(expo.gamma_phi * ts, 1e-6)
    res = optimize.least_squares(resid, np.array([expo.amplitude, u0, 1.0]), jac=jac,
                                 method="trf", bounds=([-np.inf, 1e-9, 1e-3], [np.inf, np.inf, 50.0]),
                                 xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    if res.status <= 0:
        raise ValueError(f"the stretched fit did not converge: {res.message}")
    theta = res.x
    if theta[2] <= 1e-3 * 1.001 or theta[2] >= 50.0 * 0.999 or theta[1] <= 1e-9 * 1.001:
        raise ValueError("the fit ran into the limits of the exponent (0.001 to 50) or of "
                         "T_M: these times do not sample the decay; measure through it")
    Jw = jac(theta)
    F = Jw.T @ Jw
    sv = np.linalg.svd(F, compute_uv=False)
    if not (sv[-1] > 1e-12 * sv[0]):
        raise ValueError("A, T_M and the exponent are not separately determined by these "
                         "times (the curvature matrix is singular); spread the times "
                         "over the decay")
    scale = np.array([1.0, ts, 1.0])
    cov = np.linalg.inv(F) * np.outer(scale, scale)
    cov = 0.5 * (cov + cov.T)
    values = theta * scale
    sig = np.sqrt(np.diag(cov))
    for k, label in ((1, "decay time T_M (s)"), (2, "stretch exponent")):
        if not values[k] > min_significance * sig[k]:
            raise ValueError(f"the fitted {label} {values[k]:.4g} +/- {sig[k]:.4g} is not "
                             f"{min_significance:g} error bars above zero: these data do "
                             "not resolve it")
    r = resid(theta)
    chi2 = float(r @ r)
    dof = int(t.size - 3)
    dchi2 = max(expo.chi2 - chi2, 0.0)
    return StretchedDecayFit(
        amplitude=float(values[0]), amplitude_sigma=float(sig[0]),
        T_M=float(values[1]), T_M_sigma=float(sig[1]),
        exponent=float(values[2]), exponent_sigma=float(sig[2]), cov=cov,
        chi2=chi2, dof=dof, p_value=float(stats.chi2.sf(chi2, dof)),
        chi2_exponential=float(expo.chi2), delta_chi2=float(dchi2),
        p_exponential=float(stats.chi2.sf(dchi2, 1)), n_points=int(t.size))
