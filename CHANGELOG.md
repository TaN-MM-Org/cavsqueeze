# Changelog

## 1.11.0 (2026-09-10)

### Added

- `estimate_squeezing` / `SqueezingEstimate` / `variance_tomography`:
  squeezing estimation from measured Ramsey-tomography count data.
  Exact-WLS fit of the rotation law V(theta) = c + a cos 2theta
  + b sin 2theta (a covariance transformation identity, not a
  Gaussian assumption), closed-form extremal variances and dip
  angle, delta-method uncertainties through the exact WLS
  covariance, the stated-not-hidden Gaussian sample-variance error
  bar 2 s^4/(M-1), opt-in detection-noise subtraction with an
  over-subtraction refusal, and Kitagawa-Ueda / Wineland parameters
  in exactly the solver's `wineland_xi2` convention.
- Anchors: covariance recovery against `numpy.linalg.eigvalsh`; the
  exact one-axis-twisting closed form (`oat_closed_form`) recovered
  from sampled tomography shots; the standard quantum limit from a
  coherent-spin-state sample; Monte-Carlo scatter matching the
  reported sigma; detection-noise round trip; degenerate-design
  refusals.

## 1.10.0 (2026-09-05)

### Added

- `oat_closed_form`: exact unitary one-axis-twisting moments
  (Kitagawa and Ueda, Phys. Rev. A 47, 5138 (1993)) for N spin-1/2
  particles -- mean spin, extremal transverse variances
  V(+/-) = N/4 + N(N-1)/16 [A +/- sqrt(A^2+B^2)], the optimal
  squeezing angle, and both squeezing parameters. Every returned
  quantity is asserted against brute-force exact unitary evolution
  (QuTiP) for even and odd N at random twisting angles, to 1e-12;
  the mu = 0 coherent-state limit is exact. This is the
  decoherence-free benchmark the dissipative cumulant solver is
  measured against, not a substitute for it.

### Changed

- The pulse-sequence layer is now exported at the package root:
  `css_x`, `twist`, `twist_imperfect`, `pulse`, `squeezing_after`,
  `squeezing_trace`, `optimal_squeezing`, `ramsey_cumulant`,
  `ramsey_meanfield`, `twist_untwist`, `plain_squeezed_readout`
  (previously importable only from `cavsqueeze.protocols`).
- CI matrix: Python 3.10, 3.11, 3.12, 3.13.
- CHANGELOG.md added.

Earlier versions: see the release notes on
https://github.com/TaN-MM-Org/cavsqueeze/releases
