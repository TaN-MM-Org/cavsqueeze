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
model does not contain).  In the tests, over 300 seeded noisy datasets,
the scatter of each fitted width and rate is 0.8 to 1.2 times its
reported error bar, and 2 % to 9 % of the p-values fall below 0.05.

Refusals, rather than numbers that look fine: a Lorentzian width
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
from scipy import optimize, stats

__all__ = ["DecayFit", "fit_coherence_decay"]

_K_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))   # Gaussian FWHM per standard deviation


@dataclasses.dataclass
class DecayFit:
    """Result of `fit_coherence_decay`.  All rates and widths are angular
    frequencies (rad/s), as everywhere in the package.

    line : "none", "gaussian" or "lorentzian" (the model fitted).
    amplitude, amplitude_sigma : the contrast A at t = 0.
    gamma_phi, gamma_phi_sigma : the dephasing rate 1/T2 (rad/s);
        `CavityParams(gamma_phi=...)` takes it directly.
    T2, T2_sigma : 1/gamma_phi and its error bar (s); `from_hz(T2=...)`
        takes it.  When T2 was given to the fit, these are the given
        values.
    fwhm, fwhm_sigma : FWHM of the line (rad/s), as `lineshape(line,
        fwhm)` takes it; None for line="none".
    T2_given : whether T2 was given (and only the line was fitted).
    names, values, cov : the fitted parameters, in the model
        C(t) = A exp(-rate t - s2 t^2 / 2) (names among "A", "rate",
        "s2"; rate in rad/s, s2 in rad^2/s^2), and their covariance.
        With line="gaussian" and a given T2, the rate is fixed at 1/T2
        and the covariance includes the contribution of T2_sigma; with
        line="lorentzian" the fitted rate is the total
        1/T2 + fwhm/2, and T2_sigma enters fwhm_sigma only.
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
        1/T2), "gaussian" or "lorentzian" (a Ramsey decay of a line of
        that shape).  See the module docstring for the formulas.
    T2, T2_sigma : a known coherence time and its error (s), from an
        echo fit with line="none".  Required for "lorentzian" (the only
        way to separate the line from gamma_phi); optional for
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
    if line not in ("none", "gaussian", "lorentzian"):
        raise ValueError(f"line must be 'none', 'gaussian' or 'lorentzian'; got {line!r}")
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
    if line == "lorentzian" and T2 is None:
        raise ValueError(
            "a Lorentzian width cannot be separated from gamma_phi = 1/T2 in a "
            "Ramsey decay (both decay as exp(-rate t), so only their sum is "
            "measured): fit an echo decay with line='none' first and pass its "
            "T2 and T2_sigma")

    if line == "gaussian":
        names = ("A", "s2") if T2 is not None else ("A", "rate", "s2")
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

    def resolved(name, value, sigma, label):
        if not value > min_significance * sigma:
            raise ValueError(
                f"the fitted {label} {value:.4g} +/- {sigma:.4g} is not "
                f"{min_significance:g} error bars above zero: these data do not "
                "resolve it. Measure further into the decay, reduce the "
                "contrast errors, or use a model without it" +
                (" (for a Ramsey decay dominated by the line, fit an echo with "
                 "line='none' and pass its T2)" if name == "rate" and line == "gaussian"
                 else ""))

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
        n_points=int(t.size))
