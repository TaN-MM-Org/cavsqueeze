# Changelog

## 1.12.0 (2026-09-12)

Experimental-workflow release: the two loose ends of the measured-data
path are closed, so a squeezing estimate needs no hand-entered inputs
beyond the atom number.

### Added

- `estimate_contrast` / `ContrastEstimate`: the Ramsey fringe
  contrast fitted from a phase-scan shot record -- the number
  `estimate_squeezing` previously asked the user to bring from their
  own fringe fit. The mean fringe obeys A cos phi + B sin phi + d
  exactly (the rotation law of the mean spin, no lineshape
  assumption), so the fit is closed-form weighted linear least
  squares with per-phase standard errors, and the contrast
  uncertainty follows by the delta method through the exact WLS
  covariance -- the same structure and honesty as
  `variance_tomography`. Refusals: fewer than 3 distinct phases
  modulo 2 pi, and a fitted contrast above 1 beyond its uncertainty
  (reported as a probable N or J_z-calibration problem, never
  clipped silently; a small statistical overshoot is returned as
  data, with the deliberate rounding left to the user).
- `save_shots_csv` / `load_shots_csv`: a documented plain-text
  contract (`angle_rad,jz`, one row per shot; ragged per-angle shot
  counts allowed) serving both tomography and fringe records, with
  exact round trips and refusals for malformed files and
  single-shot angles.
- Anchors: exact fringe recovery on clean records; Monte-Carlo
  scatter matching the reported contrast sigma; both refusals; exact
  file round trips; and the complete file-to-estimate pipeline
  recovering the standard quantum limit on a coherent-spin-state
  record with no hand-entered contrast.

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
