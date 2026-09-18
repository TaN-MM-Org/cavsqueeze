# Changelog

## 1.15.0 (2026-09-18)

The Dick-free comparison, and a future-proofing pass.

- `dick.synchronized_comparison`: two ensembles interrogated
  synchronously by one local oscillator -- the protocol of the
  2024-2025 record entanglement-enhanced clock comparisons (Robinson
  et al., Nat. Phys. 20, 208 (2024); Yang et al., PRL 135, 193202
  (2025)) -- modeled as exactly what it is: common-mode rejection of
  the oscillator noise, combined projection noise for the
  difference, the conventional per-clock figure, and differential
  non-common-mode noise stated as out of scope.
- CI gains a QuTiP-free Python 3.14 job.
- Anchors: the sqrt(2) and per-clock identities exact for equal
  ensembles; single-ensemble deviations equal
  `clock_allan_deviation` bitwise (two public code paths); squeezing
  gains carry through exactly; the 1/sqrt(tau) law exact; refusals
  pinned.

## 1.14.0 (2026-09-17)

Lab adaptability: the error bars of a tomography measurement,
computed exactly before the measurement exists.

- `lab.plan_tomography`: the planned design's exact weighted-least-
  squares covariance -- the same matrix `variance_tomography` reports
  -- from the angles, shot counts and expected variance scale alone,
  with predicted error bars for var_min, xi2_S and xi2_R, and a
  degenerate-angle verdict on the same arithmetic the estimator
  refuses with.
- `lab.shots_for_squeezing`: the smallest shot count meeting a target
  Wineland error bar, by exact closed-form inversion (the tomography
  term scales as 1/sqrt(M-1) exactly); a target below the exact
  contrast floor xi2_R * 2 sigma_C / C is refused with the floor
  named, because tomography shots cannot buy it.
- Anchors: planned covariance equals the estimator's to machine
  precision; K equally spaced angles give the exactly orthogonal
  normal matrix diag(K, K/2, K/2); the shot inversion is verified on
  both sides of the target; 300 seeded Monte-Carlo experiments match
  the planned error bar; degenerate designs reported and refused
  alike.

## 1.13.0 (2026-09-13)

Physics upgrade from the clock literature: the Dick effect -- the
local-oscillator aliasing floor that decides whether squeezing helps
a clock at all, the central caveat of Schulte et al., Nat. Commun.
11, 5955 (2020), and the regime today's spin-squeezed lattice-clock
comparisons engineer around (Robinson et al., Nat. Phys. 2024).

### Added

- `dick_allan_deviation` / `dick_fourier_coefficients` /
  `ramsey_sensitivity`: the Dick-limited fractional-frequency Allan
  deviation, sigma_y^2 = (1/(tau g0^2)) sum |g_m|^2 S_y(m/Tc)
  (Dick 1987; Santarelli et al., IEEE UFFC 45, 887 (1998); the
  convention of Quessada et al., J. Opt. B 5, S150 (2003)), with the
  standard on-resonance Ramsey sensitivity function (sin-shaped
  pi/2-pulse edges, unity free evolution, dead time) and its Fourier
  coefficients in exact closed form per segment. The harmonic-cutoff
  truncation is checked, not hoped for: a cutoff whose last decade
  still carries weight is refused.
- `power_law_psd`: the standard h0 + h_-1/f + h_-2/f^2 clock-laser
  PSD as a callable -- YOUR laser's measured coefficients; none are
  shipped.
- `total_clock_allan_deviation`: projection noise (where squeezing
  enters) and the Dick floor (which no atom number or entanglement
  moves) combined in quadrature, with an explicit `dick_limited`
  verdict.

### Anchors (asserted in `tests/test_dick.py`, not stated)

- The closed-form Fourier coefficients equal independent adaptive
  quadrature of the sampled sensitivity function to 1e-9 (two code
  paths), for rectangular and finite-pulse cycles.
- Continuous interrogation (zero dead time, negligible pulses) gives
  exactly zero Dick noise -- every harmonic coefficient vanishes.
- The short-pulse limit reproduces the exact rectangular-window ratio
  sin(pi m eta)/(pi m eta) to 1e-12.
- The floor scales exactly as 1/sqrt(tau); the cutoff-convergence
  refusal fires; and the verdict test shows ~10 dB of squeezing
  moving a quiet-LO clock by >3x while moving a Dick-limited clock
  by <10%, with the Dick term itself identical in both cases.

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
