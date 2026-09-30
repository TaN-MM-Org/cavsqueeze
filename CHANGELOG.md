# Changelog

## 1.16.0 (2026-09-30)

Measured-data release: less biased tomography fits, error bars that
hold for non-Gaussian detection noise, an exact Voigt width, moments
of the trajectory solver at the requested times, and input checks
where bad inputs used to give silent `nan` or unphysical numbers.

### Added

- `estimate_squeezing(..., shot_noise="empirical")`: error bars that do
  not assume Gaussian shots. Each angle's sample variance scatters by
  the exact distribution-free amount
  `Var(s^2) = sigma^4 [kurtosis/M - (M-3)/(M(M-1))]`; the kurtosis is
  measured from that angle's shots (a plug-in estimate), and the
  scatter is carried through the fit exactly. The point estimate is
  unchanged; the default stays `"gaussian"`. This lifts the README
  limit "heavy-tailed detection noise makes the error bars too small".
- `variance_tomography(..., error_sigmas=...)`: the exact covariance of
  the fit when the weights are not the true errors
  ((X^T W X)^-1 X^T W Sigma W X (X^T W X)^-1).
- `sample_variance_sigma(shots, kurtosis=None)`: the error bar of one
  sample variance by the same formula.
- `estimate_squeezing(..., weighting=...)` and the result fields
  `weighting`, `shot_noise`, `n_reweight`, `weighting_fallback`.
- `lineshape(..., split=...)` / `voigt(..., split=...)`: `"exact"`
  (default) or `"olivero"` (the pre-1.16 split).
- README example 9 (outlier shots, both error bars).

### Changed

- `estimate_squeezing` weights its fit by the fitted variance curve and
  repeats the fit until the weights stop changing
  (`weighting="model"`, the new default). The 1.15 weights, from the
  measured sample variances, are `weighting="sample"`. Reason: see
  Fixed.
- The Voigt line's Gaussian part is solved for so that the total FWHM
  is the requested one (root-finding on SciPy's exact Voigt profile;
  the test checks the half-maximum point to 1e-9); before, the
  Olivero-Longbothum approximation left it off by up to about 2.3e-4
  (largest value found for `lorentz_fraction` between 0.1 and 0.9).
- `magnetometer_sensitivity` uses `|gamma|`: a negative gyromagnetic
  ratio now gives the same positive field noise as its magnitude
  (before, a negative number).
- `Ensemble` reports mismatched array lengths with `ValueError`
  (before, a bare `assert`, which `python -O` removes).

### Fixed

- Tomography fit bias. With the weights taken from the measured sample
  variances, angles whose variance came out low by chance were
  weighted more, and the fit was biased low. Over 400 seeded
  experiments with 9 angles x 20 shots (test
  `test_model_weights_remove_the_weighting_bias`), the mean of
  `(V1 + V2)/2` was 14.4 % low with the old weights and -0.2 % +/- 0.6 %
  off with the new ones.
- `dtwa.evolve` returned the moments at the step-grid point nearest to
  each requested time, up to half a step away (`t = 0.37` was read at
  `0.370075`, a rotation error of 1.3e-4 in spin units in the new
  test). The interval between requested times is now split into whole
  steps, so every requested time is hit exactly. Times must be finite
  and >= 0.
- New refusals (`ValueError`) for inputs that gave silent `nan`,
  negative or unphysical results: a line width that is not finite and
  positive (was `nan` detunings), `lorentz_fraction` outside [0, 1],
  negative or non-finite ensemble occupations, an ensemble without
  spins, fewer than one class, negative class probabilities,
  non-finite cavity parameters, negative `kappa`, temperature or
  dephasing, `kappa = Delta = 0`, `T2 <= 0` (a negative `T2` gave a
  negative dephasing rate), a negative temperature or a non-positive
  frequency in `thermal_occupation` (was a negative occupation), a
  negative evolution time, `optimal_squeezing` limits that are not
  `0 < t_lo < t_hi`, a clock or magnetometer projection with a
  non-positive or non-finite phase noise, frequency, Ramsey time,
  averaging time or gyromagnetic ratio (a negative `tau` gave `nan`),
  `metrological_gain_db(xi2_R <= 0)`, a non-positive averaging time
  for the Dick floor, and negative or non-finite trajectory-solver
  times.

### Behaviour changes

- `estimate_squeezing` results change because of the new default
  weights. README example 5 (4000 shots per angle): `xi2_R` 0.1421 ->
  0.1420 (`xi2_R_sigma` 0.0043 in both), gain 8.47 dB -> 8.48 dB. With
  few shots per angle the change is larger (see Fixed). With few shots
  per angle the model weights cannot always be computed (the fitted
  curve dips to zero or below at a measured angle, the reweighting does
  not settle, or it ends with a minimal variance <= 0 where the sample
  fit's is positive). The default then returns the `weighting="sample"`
  (1.15) result, issues a `RuntimeWarning` and records the reason in
  the new result field `weighting_fallback`; it refuses only datasets
  that `weighting="sample"` refuses too. How often this happens, from a
  seeded script (6 equally spaced angles, 300 Gaussian datasets per
  row, variances 5/25 and 25/25 spin units squared, N = 100):

  | shots per angle | refused, sample weights (= 1.15) | refused, new default | fell back to sample weights |
  |---|---|---|---|
  | 3  | 22.7 % / 22.7 % | 22.0 % / 20.7 % | 29.3 % / 26.7 % |
  | 5  | 6.3 % / 4.0 %   | 5.3 % / 4.0 %   | 12.7 % / 11.0 % |
  | 10 | 0.3 % / 0.0 %   | 0.3 % / 0.0 %   | 1.7 % / 1.3 %   |

  (The new default can refuse slightly less often than the sample
  weights, when the model-weighted fit is positive and the sample one is
  not.) An intermediate version of this branch refused instead of
  falling back, at 51.3 % / 47.3 % (3 shots), 18.0 % / 15.0 % (5) and
  2.0 % / 1.3 % (10); it was not released.
- Voigt lines: class detunings of a 16-class `equal_probability_classes`
  discretization move by at most 1.5e-4 (`lorentz_fraction` 0.1),
  2.0e-4 (0.3), 7.8e-5 (0.5) and 1.3e-4 (0.9) of their value. Gaussian
  and Lorentzian lines are unchanged.
- `dtwa.evolve`: moments are now at the exact times, and the step count
  can rise slightly (408 instead of 400 steps in the one-axis-twisting
  test, whose best squeezing moves from -14.1731 dB to -14.1736 dB;
  exact -14.2020 dB).
- Inputs listed under Fixed now raise instead of returning a number.
  `clock_allan_deviation` and `total_clock_allan_deviation` now refuse
  an infinite `dphi` (before, they returned `inf`).

### Tests

- 123 tests pass and 3 skip (1.15.1: 69 and 3; the 3 skipped need the
  paper's companion scripts). New: `test_tomography_noise.py` (the
  `Var(s^2)` identity by exact enumeration of a three-valued
  distribution to 1e-12; the fixed-weight covariance against 3000
  simulated fits to 10 % and against the usual matrix to 1e-12; the
  weighting bias; the model-weighted covariance equal to
  `plan_tomography` at the fitted values to 1e-9; empirical error bars
  within 0.85-1.2 of the scatter for 5 % outlier shots, where the
  Gaussian ones are too small by more than 1.5x; the coherent-state
  bias of the smallest variance, 0.85-1.3 reported error bars against
  the asymptotic sqrt(pi/3) = 1.02; the fallback to sample weights
  returning exactly the sample result with one warning and refusing
  only what sample weights refuse, over 200 seeded 3-shot datasets;
  refusals), `test_inputs.py` (the
  Voigt FWHM against a direct numerical convolution to 1e-9, its
  limits and quantiles; every new refusal; accepted edge cases), and
  `test_dtwa.py::test_moments_are_read_at_the_requested_times` (exact
  rigid rotation to 1e-9; 1.15.1 is off by 1.3e-4).

## 1.15.1 (2026-09-22)

A bug fix in the trajectory solver, wider CI coverage, and a rewritten
README.

### Fixed

- `cavsqueeze.dtwa.evolve` read the sign of `y` wrongly. It integrates
  the same equations as the cumulant solver, whose variable is
  `<sigma^+> = (x + i y)/2`, but converted back as if the variable were
  `<sigma^->`. The mean `J_y` and the `xy` and `yz` covariances it
  returned therefore had the opposite sign to the cumulant solver (in the
  4-spin case of the new test at t = 1, `<J_y> = -0.34` against
  `+0.33`). Squeezing parameters
  do not depend on that sign and are unchanged; seeded runs give the
  same squeezing values as before.

### Tests

- `test_dtwa.py::test_mean_spin_follows_the_cumulant_convention`:
  without interaction the trajectory mean spin follows exact free
  precession, and with interaction it agrees with the cumulant solver,
  both to 0.08. It fails on 1.15.0.

### Changed

- CI: the full test matrix now includes Python 3.14 (QuTiP installs
  there), the QuTiP-free 3.14 job is kept, and a new
  `oldest-dependencies` job runs the suite on Python 3.10 with
  NumPy 1.24.0, SciPy 1.10.0, Matplotlib 3.7.0 and QuTiP 5.0.0 (plus
  setuptools, which QuTiP 5.0.0 imports). The classifiers now list
  Python 3.13 and 3.14.
- README rewritten: plain-language guide to the terms, eight runnable
  examples with their printed output, every refusal and every test
  tolerance listed as the tests assert it.
- Docstrings of `dick`, `dtwa` and `interop` corrected where they
  described tests that do not exist (see below).

### Corrections to earlier notes

- 1.13.0 and the old README called the continuous-interrogation zero
  and the rectangular-window ratio "exact"; the tests check them to
  1e-12 (coefficients) and 1e-18 (deviation). The `dick` module
  docstring said the coefficients were checked against a dense-grid
  FFT; the test uses adaptive quadrature (as the 1.13.0 notes say).
- 1.10.0 said every quantity returned by `oat_closed_form` is checked
  against exact evolution to 1e-12. The mean spin and the variance at
  the predicted angle are; the largest and smallest variance over the
  angle grid are checked to 1e-4, and `xi2_S`, `xi2_R` are not
  compared separately.
- 1.14.0 "machine precision" and "exactly orthogonal" mean agreement
  to 1e-12 in the tests.
- The `dtwa` docstring said the cumulant solver was checked against
  eight distinguishable spins; the tests use two and four.
- The `interop` docstring pointed to the test suite for the automatic
  Fock cutoff at r = 2.3; no test covers it (the export's own check
  does, at run time).

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
