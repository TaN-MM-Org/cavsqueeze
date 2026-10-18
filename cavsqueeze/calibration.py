"""Projection-noise calibration of the readout (new in 1.19).

Before a squeezing measurement, an experiment checks that its detector
sees the projection noise of the spins: it prepares the coherent spin
state on the equator (a pi/2 pulse from the ground state) at several atom
numbers N, measures J_z = (N_up - N_down)/2 many times at each, and fits
the shot-to-shot variance against N.  For N independent spins, each up or
down with probability 1/2, J_z is a binomial variable minus N/2, so

    Var(J_z) = N / 4                      (projection noise),

the population fluctuation studied by W. M. Itano et al., Phys. Rev. A 47,
3554 (1993).  Detection noise that does not depend on the spins adds a
constant, and noise that moves all spins together (for example a jitter of
the pulse angle) adds a term in N^2:

    Var(J_z) = c0 + c1 N + c2 N^2,      c1 = 1/4 for projection noise.

The N^2 form is exact for a jitter of the pulse angle: if every spin is
tipped by the same random angle delta away from the equator, the
probability of "up" is (1 + sin delta)/2, so Var(J_z) = (N/4) E[cos^2
delta] + (N^2/4) Var(sin delta) (the variance of a binomial count plus
the variance of its mean); the tests check this law by simulation.

`calibrate_projection_noise` fits that line (c0 + c1 N) or parabola
(``quadratic=True``) by weighted least squares and returns:

* ``detection_variance`` = c0 with its error bar: the number
  `estimate_squeezing(detection_variance=..., detection_variance_sigma=...)`
  subtracts (both in spin units squared);
* ``projection_ratio`` = 4 c1 with its error bar (1 when the per-spin
  noise is projection noise; a ratio away from 1 means the J_z or N
  calibration is off, or there is noise that grows like N), and
  ``p_projection``, the two-sided p-value of the test c1 = 1/4;
* ``chi2``, ``dof`` and ``p_value``: whether the variances follow the
  fitted law within their errors (a small p-value from the straight line
  usually means an N^2 term: fit again with ``quadratic=True``).

Statistics: the fit is linear in (c0, c1, c2), so it is solved in closed
form.  Each point is weighted by the standard error of its sample
variance, which is proportional to the variance itself; as in
`estimate_squeezing` (weighting="model"), that variance is read from the
fitted curve and the fit is repeated until the weights stop changing,
which avoids the low bias of weights taken from the noisy sample
variances.  ``shot_noise`` has the meaning it has in `estimate_squeezing`:
"gaussian" uses Var(s^2) = 2 sigma^4 / (M - 1), "empirical" the measured
kurtosis of each point's shots, propagated exactly through the
fixed-weight fit.  The error bars are those of a linear estimator, and the
p-values assume nearly Gaussian sample variances (many shots per point).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats

from .measured import _goodness_of_fit, _sample_kurtosis, _var_s2_factor

__all__ = ["ProjectionNoiseFit", "calibrate_projection_noise"]

_MAX_REWEIGHT = 1000


@dataclasses.dataclass
class ProjectionNoiseFit:
    """Result of `calibrate_projection_noise` (new in 1.19).

    detection_variance, detection_variance_sigma : the constant c0 of
        Var(J_z) = c0 + c1 N (+ c2 N^2) and its error bar (spin units
        squared); the input `estimate_squeezing` takes as
        ``detection_variance`` and ``detection_variance_sigma``.  It is
        returned as fitted, also when it comes out negative within its
        error bar (`estimate_squeezing` refuses a negative value; setting
        it to 0 is then the user's decision).
    per_spin_variance, per_spin_variance_sigma : c1 and its error bar
        (1/4 for projection noise).
    projection_ratio, projection_ratio_sigma : 4 c1 and its error bar.
    p_projection : two-sided p-value of the test c1 = 1/4.
    technical_coefficient, technical_coefficient_sigma : c2 and its error
        bar with ``quadratic=True`` (None otherwise).
    names, values, cov : the fitted coefficients ("c0", "c1"[, "c2"]),
        their values and covariance.
    chi2, dof, p_value : goodness of fit of the variance law (dof = points
        minus fitted coefficients).
    N_values, variances, variance_sigmas : the atom numbers, the measured
        sample variances and their standard errors used for the error bars
        (from the fitted curve; with shot_noise="empirical", from the
        fitted curve and each point's measured kurtosis).
    shots_per_point, shot_noise, n_reweight : bookkeeping.
    """

    detection_variance: float
    detection_variance_sigma: float
    per_spin_variance: float
    per_spin_variance_sigma: float
    projection_ratio: float
    projection_ratio_sigma: float
    p_projection: float
    technical_coefficient: float | None
    technical_coefficient_sigma: float | None
    names: tuple
    values: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    p_value: float
    N_values: np.ndarray
    variances: np.ndarray
    variance_sigmas: np.ndarray
    shots_per_point: np.ndarray
    shot_noise: str
    n_reweight: int


def _fit(X, y, sg, es=None):
    """Weighted linear least squares y ~ X beta with weights 1/sg^2.
    Returns (beta, covariance); with ``es`` (the true errors) the
    covariance is the exact sandwich of the fixed-weight estimator."""
    w = 1.0 / sg
    Xw = X * w[:, None]
    A = Xw.T @ Xw
    beta = np.linalg.solve(A, Xw.T @ (y * w))
    cov = np.linalg.inv(A)
    if es is not None:
        L = cov @ (X * (w * w)[:, None]).T
        cov = (L * es ** 2) @ L.T
    return beta, 0.5 * (cov + cov.T)


def calibrate_projection_noise(N_values, shots, quadratic: bool = False,
                               shot_noise: str = "gaussian"
                               ) -> ProjectionNoiseFit:
    """Fit the readout noise of coherent spin states against the atom
    number (new in 1.19; see the module docstring).

    N_values : (K,) atom numbers of the K settings (for example the mean
        measured total population at each), finite and positive.
    shots : length-K sequence of shot arrays (or a (K, M) array): J_z in
        spin units, (N_up - N_down)/2, of a coherent spin state on the
        equator, measured directly (no tomography rotation), at least 2
        shots per setting.
    quadratic : also fit a term c2 N^2 (noise common to all spins).
    shot_noise : "gaussian" (default) or "empirical", as in
        `estimate_squeezing`.

    Returns a :class:`ProjectionNoiseFit`.  Refuses (ValueError) inputs of
    different lengths, non-finite or non-positive atom numbers, fewer than
    2 shots or non-finite shots at a setting, identical shots, fewer
    settings than coefficients plus one (no test of the fit), fewer
    distinct atom numbers than coefficients, and a fitted variance curve
    that reaches zero or below at a measured atom number (it cannot then
    set the weights).
    """
    Nv = np.asarray(N_values, dtype=float).ravel()
    rows = [np.asarray(s, dtype=float).ravel() for s in shots]
    if len(rows) != Nv.size:
        raise ValueError(f"need one shot array per atom number (got {Nv.size} atom "
                         f"numbers and {len(rows)} shot arrays)")
    if not (np.all(np.isfinite(Nv)) and np.all(Nv > 0.0)):
        raise ValueError("N_values must be finite and positive (atom numbers)")
    if any(r.size < 2 for r in rows):
        raise ValueError("need at least 2 shots per atom number")
    if any(not np.all(np.isfinite(r)) for r in rows):
        raise ValueError("shots contain non-finite values")
    if shot_noise not in ("gaussian", "empirical"):
        raise ValueError("shot_noise must be 'gaussian' or 'empirical'")
    if not isinstance(quadratic, (bool, np.bool_)):
        raise ValueError(f"quadratic must be True or False; got {quadratic!r}")
    names = ("c0", "c1", "c2") if quadratic else ("c0", "c1")
    npar = len(names)
    if Nv.size < npar + 1:
        raise ValueError(f"need at least {npar + 1} atom numbers to fit {npar} "
                         f"coefficients and test the fit; got {Nv.size}")
    if np.unique(Nv).size < npar:
        raise ValueError(f"need at least {npar} distinct atom numbers; got "
                         f"{np.unique(Nv).size}")
    M = np.array([r.size for r in rows])
    s2 = np.array([float(np.var(r, ddof=1)) for r in rows])
    if np.any(s2 <= 0.0):
        raise ValueError("a shot array has zero scatter; identical shots carry no "
                         "noise information")

    # columns scaled by the largest N, so the normal matrix is well
    # conditioned for any atom number; undone at the end
    scale = float(Nv.max())
    u = Nv / scale
    X = np.column_stack([u ** k for k in range(npar)])
    sv = np.linalg.svd(X, compute_uv=False)
    if not sv[-1] > 1e-10 * sv[0]:
        raise ValueError("the atom numbers do not separate the coefficients; "
                         "spread them over a wider range")
    gauss = np.sqrt(2.0 / (M - 1))
    sig = s2 * gauss                       # start: weights from the samples
    beta, cov = _fit(X, s2, sig)
    n_refit = 0
    while True:
        curve = X @ beta
        if np.any(curve <= 0.0):
            raise ValueError(
                f"the fitted variance curve reaches {curve.min():.4g} <= 0 at a "
                "measured atom number, so it cannot set the fit weights: the "
                "variances do not follow the law (check the data, or take more "
                "shots per point)")
        new = curve * gauss
        if np.max(np.abs(new / sig - 1.0)) <= 1e-12:
            break
        if n_refit == _MAX_REWEIGHT:
            raise ValueError(f"the weighted fit did not settle in {_MAX_REWEIGHT} "
                             "reweighting steps")
        sig = new
        beta, cov = _fit(X, s2, sig)
        n_refit += 1
    es = None
    if shot_noise == "empirical":
        kurt = np.array([_sample_kurtosis(r) for r in rows])
        es = (sig / gauss) * np.sqrt(_var_s2_factor(kurt, M))
        beta, cov = _fit(X, s2, sig, es)
    chi2, dof, p_value = _goodness_of_fit(X, sig, sig if es is None else es,
                                          s2 - X @ beta)
    back = scale ** -np.arange(npar, dtype=float)       # to physical units
    values = beta * back
    cov = cov * np.outer(back, back)
    err = np.sqrt(np.diag(cov))
    c1, c1_sigma = float(values[1]), float(err[1])
    z = (c1 - 0.25) / c1_sigma
    return ProjectionNoiseFit(
        detection_variance=float(values[0]), detection_variance_sigma=float(err[0]),
        per_spin_variance=c1, per_spin_variance_sigma=c1_sigma,
        projection_ratio=4.0 * c1, projection_ratio_sigma=4.0 * c1_sigma,
        p_projection=float(2.0 * stats.norm.sf(abs(z))),
        technical_coefficient=float(values[2]) if quadratic else None,
        technical_coefficient_sigma=float(err[2]) if quadratic else None,
        names=names, values=values, cov=cov, chi2=chi2, dof=dof, p_value=p_value,
        N_values=Nv, variances=s2, variance_sigmas=sig if es is None else es,
        shots_per_point=M,
        shot_noise=shot_noise, n_reweight=int(n_refit))
