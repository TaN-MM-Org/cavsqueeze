"""Discretization of the inhomogeneous ensemble into classes of identical spins.

A class m carries a detuning delta_m (rad/s), a coupling weight w_m
(dimensionless multiplier of the single-spin coupling g) and an
occupation n_m (number of spins, may be non-integer for very large N).
"""
from __future__ import annotations

import dataclasses
import numpy as np
from scipy import optimize, special, stats

TWO_PI = 2.0 * np.pi


@dataclasses.dataclass
class Ensemble:
    delta: np.ndarray  # (M,) detunings, rad/s
    weight: np.ndarray  # (M,) coupling weights
    n: np.ndarray  # (M,) occupations
    spec_delta: np.ndarray = None  # (K,) detunings of spectator (free) spins, rad/s
    spec_n: np.ndarray = None  # (K,) occupations of spectator spins

    @property
    def M(self) -> int:
        return len(self.delta)

    @property
    def N(self) -> float:
        return float(np.sum(self.n))

    def __post_init__(self):
        self.delta = np.asarray(self.delta, dtype=float)
        self.weight = np.asarray(self.weight, dtype=float)
        self.n = np.asarray(self.n, dtype=float)
        if not (self.delta.ndim == 1 and self.delta.shape == self.weight.shape == self.n.shape):
            raise ValueError("delta, weight and n must be 1-D arrays of the same length "
                             f"(got shapes {self.delta.shape}, {self.weight.shape}, {self.n.shape})")
        if self.spec_delta is None:
            self.spec_delta = np.zeros(0)
            self.spec_n = np.zeros(0)
        self.spec_delta = np.asarray(self.spec_delta, float)
        self.spec_n = np.asarray(self.spec_n, float)
        if not (self.spec_delta.ndim == 1 and self.spec_delta.shape == self.spec_n.shape):
            raise ValueError("spec_delta and spec_n must be 1-D arrays of the same length")
        for name in ("delta", "weight", "n", "spec_delta", "spec_n"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"Ensemble.{name} contains non-finite values")
        if np.any(self.n < 0) or np.any(self.spec_n < 0):
            raise ValueError("class occupations n (and spec_n) must be >= 0")
        if self.M == 0 or not np.sum(self.n) > 0:
            raise ValueError("an ensemble needs at least one class and a positive number of spins")

    @property
    def N_total(self) -> float:
        return float(np.sum(self.n) + np.sum(self.spec_n))


# ---------------------------------------------------------------------------
# Line shapes.  All take the FWHM (rad/s) and return a frozen scipy distribution
# for the detuning (rad/s).
# ---------------------------------------------------------------------------

def _check_fwhm(fwhm):
    if not (np.isfinite(fwhm) and fwhm > 0):
        raise ValueError(f"line FWHM must be finite and positive (rad/s); got {fwhm}")


def gaussian(fwhm: float):
    _check_fwhm(fwhm)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return stats.norm(loc=0.0, scale=sigma)


def lorentzian(fwhm: float):
    _check_fwhm(fwhm)
    return stats.cauchy(loc=0.0, scale=fwhm / 2.0)


class _Voigt:
    """Voigt profile with tabulated cdf/ppf (dense center, geometric tails).

    The Lorentzian part has FWHM f_L = lorentz_fraction * fwhm.  The
    Gaussian part is chosen so that the FWHM of the Voigt profile itself
    equals ``fwhm``: with split="exact" (the default since 1.16) its
    width is solved for by root-finding on the exact profile
    (scipy.special.voigt_profile); the tests check, with an independent
    numerical convolution, that the profile at fwhm/2 is half its peak
    to 1e-9.  With split="olivero" it comes from the Olivero-Longbothum
    approximation f_V ~ 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2), the
    only choice before 1.16, whose total FWHM is off by up to about
    2.3e-4 of fwhm.
    """

    def __init__(self, fwhm: float, lorentz_fraction: float, split: str = "exact"):
        self.fwhm = fwhm
        self.eta = lorentz_fraction
        fL = lorentz_fraction * fwhm
        self.gamma = fL / 2.0
        s_of = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # sigma per unit Gaussian FWHM
        if split == "olivero":
            fG2 = (fwhm - 0.5346 * fL) ** 2 - 0.2166 * fL**2
            fG = np.sqrt(max(fG2, 1e-30))
        elif split == "exact":
            if lorentz_fraction == 0.0:
                fG = fwhm
            elif lorentz_fraction == 1.0:
                fG = 0.0
            else:
                def excess(fg):
                    self.sigma = fg * s_of
                    return self.half_width() - 0.5 * fwhm
                fG = optimize.brentq(excess, 0.0, fwhm, xtol=1e-15 * fwhm, rtol=1e-14)
        else:
            raise ValueError("split must be 'exact' or 'olivero'")
        self.sigma = fG * s_of
        pos = np.geomspace(1e-5 * fwhm, 1e7 * fwhm, 60001)
        grid = np.concatenate([-pos[::-1], [0.0], pos])
        pdf = self.pdf(grid)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
        # add the analytic Lorentzian-like mass beyond the grid (negligible) and normalize
        cdf = cdf / cdf[-1]
        self._grid, self._cdf = grid, cdf

    def half_width(self) -> float:
        """Half width at half maximum of the profile, from the exact pdf."""
        if self.gamma == 0.0:
            return float(self.sigma * np.sqrt(2.0 * np.log(2.0)))
        if self.sigma == 0.0:
            return float(self.gamma)
        half = 0.5 * float(self.pdf(0.0))
        hi = self.gamma + self.sigma
        while float(self.pdf(hi)) >= half:      # widen until the half maximum is bracketed
            hi *= 2.0
        return float(optimize.brentq(lambda x: float(self.pdf(x)) - half, 0.0, hi,
                                     xtol=1e-15 * hi, rtol=1e-14))

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        if self.gamma == 0:
            return stats.norm.pdf(x, scale=self.sigma)
        if self.sigma <= 1e-12 * self.gamma:
            return stats.cauchy.pdf(x, scale=self.gamma)
        return special.voigt_profile(x, self.sigma, self.gamma)

    def cdf(self, x):
        scalar = np.isscalar(x)
        out = np.interp(np.atleast_1d(np.asarray(x, float)), self._grid, self._cdf)
        return float(out[0]) if scalar else out

    def ppf(self, q):
        scalar = np.isscalar(q)
        out = np.interp(np.atleast_1d(np.asarray(q, float)), self._cdf, self._grid)
        return float(out[0]) if scalar else out


def voigt(fwhm: float, lorentz_fraction: float = 0.3, split: str = "exact"):
    """Voigt line of total FWHM ``fwhm`` whose Lorentzian part has FWHM
    lorentz_fraction * fwhm (0 = Gaussian, 1 = Lorentzian).  See _Voigt
    for ``split``."""
    _check_fwhm(fwhm)
    if not (0.0 <= lorentz_fraction <= 1.0):
        raise ValueError(f"lorentz_fraction must lie in [0, 1]; got {lorentz_fraction}")
    return _Voigt(fwhm, lorentz_fraction, split=split)


def lineshape(name: str, fwhm: float, lorentz_fraction: float = 0.3, split: str = "exact"):
    """Gaussian, Lorentzian or Voigt line of full width at half maximum
    ``fwhm`` (rad/s), as a frozen distribution of the detuning with pdf,
    cdf and ppf.  ``lorentz_fraction`` and ``split`` apply to the Voigt
    line only (see `voigt`)."""
    name = name.lower()
    if name == "gaussian":
        return gaussian(fwhm)
    if name == "lorentzian":
        return lorentzian(fwhm)
    if name == "voigt":
        return voigt(fwhm, lorentz_fraction, split=split)
    raise ValueError(f"unknown line shape {name!r}: use gaussian, lorentzian or voigt")


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

def equal_probability_classes(dist, M: int, N: float, weights=None) -> Ensemble:
    """Split the line into M equal-probability bins; each class sits at the
    median detuning of its bin and holds N/M spins.  Works for heavy tails
    because the bin median (not the mean) is used.

    weights: optional (M,) coupling weights (default all ones)."""
    if not (isinstance(M, (int, np.integer)) and M >= 1):
        raise ValueError(f"M must be an integer >= 1; got {M!r}")
    q = (np.arange(M) + 0.5) / M
    delta = np.asarray(dist.ppf(q), dtype=float)
    n = np.full(M, N / M)
    w = np.ones(M) if weights is None else np.asarray(weights, dtype=float)
    return Ensemble(delta=delta, weight=w, n=n)


def homogeneous(N: float, delta: float = 0.0, weight: float = 1.0) -> Ensemble:
    return Ensemble(delta=np.array([delta]), weight=np.array([weight]), n=np.array([N]))


def product_classes(delta_dist, M_delta: int, weight_values, weight_probs, N: float) -> Ensemble:
    """Cartesian product of a detuning discretization and a discrete coupling
    weight distribution (weight_values with probabilities weight_probs)."""
    base = equal_probability_classes(delta_dist, M_delta, 1.0)
    weight_values = np.asarray(weight_values, float).ravel()
    weight_probs = np.asarray(weight_probs, float).ravel()
    if weight_values.shape != weight_probs.shape:
        raise ValueError("weight_values and weight_probs must have the same length")
    if not (np.all(np.isfinite(weight_probs)) and np.all(weight_probs >= 0) and weight_probs.sum() > 0):
        raise ValueError("weight_probs must be finite, >= 0 and not all zero")
    weight_probs = weight_probs / weight_probs.sum()
    d = np.repeat(base.delta, len(weight_values))
    w = np.tile(weight_values, M_delta)
    n = np.repeat(base.n, len(weight_values)) * np.tile(weight_probs, M_delta) * N
    return Ensemble(delta=d, weight=w, n=n)


def log_uniform_weights(dynamic_range: float, K: int):
    """K coupling weights spread log-uniformly over [1/sqrt(D), sqrt(D)] with
    equal probability, normalized so that the mean weight is one."""
    if dynamic_range <= 1.0 or K == 1:
        return np.ones(1), np.ones(1)
    logs = np.linspace(-0.5 * np.log(dynamic_range), 0.5 * np.log(dynamic_range), K)
    w = np.exp(logs)
    w = w / w.mean()
    return w, np.ones(K) / K


def tail_resolved_classes(dist, N: float, M_core: int = 32, M_tail: int = 10, core_edge: float = None,
                          delta_max: float = None, fwhm: float = None, weights=None,
                          spectator_beyond: float = None, M_spec: int = 8) -> Ensemble:
    """Discretization that resolves heavy (Lorentzian) tails.

    * core: |delta| < core_edge (default 3 FWHM), M_core equal-probability bins,
      node = bin median;
    * tail: core_edge < |delta| < spectator_beyond (or delta_max if no spectators),
      M_tail log-spaced bins, node = bin median, occupation = bin mass;
    * spectators (if spectator_beyond is given): |delta| > spectator_beyond, split
      into M_spec log-spaced bins out to delta_max (default 1000 FWHM) with the
      remaining mass lumped into the last bin.  Spectators are far outside the
      interaction gap, do not take part in the exchange dynamics (relative
      corrections (chi N/delta)^2) and are propagated exactly as free spins, which
      removes the stiffness of the far tail from the ODE.
    """
    if fwhm is None:
        raise ValueError("fwhm (rad/s) is required")
    if core_edge is None:
        core_edge = 3.0 * fwhm
    if delta_max is None:
        delta_max = 1000.0 * fwhm
    tail_end = delta_max if spectator_beyond is None else max(spectator_beyond, core_edge * 1.001)
    c_lo = float(dist.cdf(-core_edge))
    c_hi = float(dist.cdf(core_edge))
    q = c_lo + (c_hi - c_lo) * (np.arange(M_core) + 0.5) / M_core
    d_core = np.asarray(dist.ppf(q), float)
    n_core = np.full(M_core, (c_hi - c_lo) / M_core)

    def log_bins(lo_edge, hi_edge, M, lump_last):
        edges = np.geomspace(lo_edge, hi_edge, M + 1)
        dd, nn = [], []
        for k in range(M):
            lo, hi = edges[k], edges[k + 1]
            if lump_last and k == M - 1:
                mass = float(1.0 - dist.cdf(lo))
            else:
                mass = float(dist.cdf(hi) - dist.cdf(lo))
            if not np.isfinite(mass) or mass <= 1e-14:
                continue
            med = float(dist.ppf(dist.cdf(lo) + 0.5 * mass))
            if not np.isfinite(med):
                med = np.sqrt(lo * hi)
            dd.append(med)
            nn.append(mass)
        return np.array(dd), np.array(nn)

    d_tail, n_tail = log_bins(core_edge, tail_end, M_tail, lump_last=spectator_beyond is None)
    delta = np.concatenate([-d_tail[::-1], d_core, d_tail])
    n = np.concatenate([n_tail[::-1], n_core, n_tail])
    if spectator_beyond is not None:
        d_sp, n_sp = log_bins(tail_end, delta_max, M_spec, lump_last=True)
        spec_delta = np.concatenate([-d_sp[::-1], d_sp])
        spec_n = np.concatenate([n_sp[::-1], n_sp])
    else:
        spec_delta, spec_n = np.zeros(0), np.zeros(0)
    total = n.sum() + spec_n.sum()
    n = n / total * N
    spec_n = spec_n / total * N
    w = np.ones(len(delta)) if weights is None else np.asarray(weights, float)
    return Ensemble(delta=delta, weight=w, n=n, spec_delta=spec_delta, spec_n=spec_n)
