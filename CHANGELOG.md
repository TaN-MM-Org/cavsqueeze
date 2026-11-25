# Changelog

## 1.21.0 (2026-11-25)

Cavity check release: an exact reference that keeps the cavity mode, which
corrects the documented sign of the detuning and adds the frequency shift
from thermal cavity photons; error bars of the atom numbers in the
projection-noise calibration; a shot planner that no longer returns one
shot too many; refusals for scan and trajectory settings that gave wrong
or `nan` results; and a faster trajectory solver.

### Added

- `cavsqueeze.exact.spin_cavity(params, delta, weights, bloch, times,
  n_fock=8)` (needs QuTiP): the exact master equation of a few spins
  together with the cavity mode, not eliminated (Hamiltonian
  `-Delta a^dag a + sum (delta_j/2) sigma z_j + g sum w_j (a^dag sigma-_j +
  a sigma+_j)` in the frame of the spins, cavity damped at `kappa` towards
  `n_th` thermal photons, single-spin dephasing). It returns the same
  moments as `full_hilbert`, plus the photon number and the population of
  the highest kept photon state. Comparing the two measures the error of
  dropping the cavity, which the README listed as a limit without a
  number. README example 14: for two spins the largest difference falls
  0.144, 0.057, 0.014 as `Delta` goes 10, 40, 160 (in units of
  `chi g^2`, with `chi g^2` and `Gamma_SR g^2` fixed).
- `calibrate_projection_noise(..., N_sigma=...)`: error bars of the atom
  numbers (one number or one per point), for errors that differ from
  point to point. Each point's error becomes
  `sqrt(sigma_s2^2 + ((c1 + 2 c2 N) N_sigma)^2)`, iterated with the fit
  weights; for the straight line this is exact, for the parabola first
  order. `ProjectionNoiseFit.N_sigma` records it (a new field with a
  default, at the end). The default `None` gives the 1.20 results
  exactly. This narrows the Limits item "point-to-point errors in N are
  not included in the error bars". README example 15.

### Changed

- The sign convention of `Delta` is documented as `omega_s - omega_c`
  (spin minus cavity) in `CavityParams`, `cavsqueeze.resonator`,
  `cavsqueeze.cumulant_raw` and the README (see Fixed). No number changes.
- `Rates.from_params` adds the thermal-photon shift `2 n_th chi1 G_j^2` to
  each class detuning (see Fixed). `Rates.delta` therefore differs from
  `Ensemble.delta` when the temperature is above zero.
- `cavsqueeze.dtwa`: the derivatives are computed by a new module-level
  `_deriv`, with the sum over the other spins written as the full sum
  minus the own term (the same equations, fewer array operations). Equal
  to the 1.20 code to 2.4e-16 of the largest derivative; a seeded run
  gives the 1.20 moments to 8e-16. The run of the 200-spin,
  4000-trajectory test of `tests/test_dtwa.py` took 107 s instead of
  137 s (medians of 4 interleaved runs, about 1.3 times faster), and the
  derivatives alone 10 ms instead of 13 ms per call. These timings come
  from the shared 2-core build machine under a load of about 5, so they
  are rough.
- `shots_for_squeezing` (without a kurtosis) checks its closed-form count
  on the plan itself (see Fixed).
- README: examples 14 and 15, the checks and refusals of 1.21, the
  corrections, and the Limits items on the cavity and the calibration.
  README examples 1 to 13 and the three tutorials print the same output
  as with 1.20.0.

### Fixed

Each fix has a test in `tests/test_fixes_121.py` or, for the cavity terms,
`tests/test_spin_cavity_121.py` that fails on 1.20.0.

- Sign of the detuning. `CavityParams` documented `Delta` as
  `omega_c - omega_s`, but with the package's exchange Hamiltonian `+chi1 G_j G_k sigma+_j
  sigma-_k` and `chi1 = 4 Delta/(4 Delta^2 + kappa^2)` the full
  spin-cavity model is matched only for `Delta = omega_s - omega_c`: for
  two spins at `Delta = 80 chi g^2` the moments agree to 0.028 (0.031
  for `-80`), and with the sign of `chi` flipped they miss by 0.54. The
  documentation is corrected; the code is not changed. For a line
  symmetric about the spin frequency the sign changes neither `xi2` nor
  the contrast (the state is mirrored; `xi2` equal to 6e-14 relative in
  the test).
- Missing thermal shift. Thermal cavity photons shift each spin's
  frequency by `2 n_th chi g_j^2` (the photon-number part of the
  dispersive shift; the vacuum part `chi g_j^2` was already the self term
  of the model). Without it the eliminated model did not approach the
  full one as the cavity was detuned further: with `kappa = Delta` and
  0.2 thermal photons the difference stayed at 0.060, 0.063, 0.065 for
  `Delta` = 40, 80, 160, and it is now 0.0216, 0.0118, 0.0062 (falling
  like `1/Delta`). At zero temperature nothing changes, and for a uniform
  coupling it is a common shift that changes no squeezing parameter or
  contrast. With uneven couplings the effect has relative size `1/N`
  (1000 spins, couplings spread by a factor 4, 0.19 thermal photons,
  20 us echo twist: 5e-5 dB in `xi2`; 10^8 spins: 3e-11 dB). Spectator
  spins get neither this shift nor the vacuum one, as before.
- `shots_for_squeezing` without a kurtosis could return one shot more
  than needed. When the target was exactly the error bar that `m` shots
  give, rounding in the closed form returned `m + 1`: for 1000 and 22116
  shots in the design of README example 6, and for 23 of 40 random counts
  with a detection-variance error. The count is now checked on the plan
  (it meets the target and one shot fewer does not). Targets that are not
  exactly reachable, such as those of examples 6 and 13, give the same
  counts as before.
- `optimal_squeezing` accepted `n_coarse = 1`, which scanned `t_lo` alone
  and returned it as the best time (README example 1: `t = 1e-6` and
  `xi2 = 0.811` instead of 124 us), a `max_fine` below 8, which overrode
  the documented minimum of 8 fine points (0 left out the fine scan), and
  a `nan` `points_per_period`. Each is now refused, as is an empty time
  list in `squeezing_trace` (before, an `IndexError`).
- `cavsqueeze.dtwa.evolve` accepted `n_traj = 1`, which returned a
  covariance of `nan`, and non-finite detunings, couplings or `chi1`,
  which gave `nan` moments; detunings and couplings of different lengths
  ended in a NumPy broadcasting error. Each is now a `ValueError` naming
  the input.
- `load_shots_csv` loaded an infinite `J_z` without a word, split the
  rows of an angle written `nan` into angles of one shot each (refused
  with the misleading "angle nan carries 1 shot(s)"), and refused text
  with Python's message, without the line. A value that is not a finite
  number is now refused with its line number. Valid files load as before.

### Not included

- The solver still has no cavity mode. `exact.spin_cavity` measures the
  error of dropping it, but only for a few spins (the space has
  `2^N n_fock` states); for a large ensemble the error has to be judged
  from scaled-down cases, which is an extrapolation.
- The trajectory solver still has no collective decay and no thermal
  photons. Adding them needs a stochastic treatment of collective jumps
  that has to be derived and checked against PIQS, which was not done in
  this release.
- An error of the atom-number scale common to all calibration points is
  not covered by `N_sigma`: it scales `c1` and cannot be seen in the
  calibration data.
- The continuous measurement with spectator spins stays refused: the
  spectators are kept uncorrelated, and the measurement correlates them
  with the rest.
- The low bias of the fitted smallest variance is still not corrected; a
  correction needs the unknown true anisotropy.
- The solver's dephasing stays Markovian. A stretched echo decay
  (`fit_stretched_decay`) does not fix the dephasing at the other times
  of a twisting sequence without a model of the noise spectrum.

### Checks (tests)

- 297 tests pass and 3 skip (1.20.0: 279 and 3; the 3 skipped need the
  paper's companion scripts), on Python 3.13 and with the oldest allowed
  dependencies (Python 3.10, NumPy 1.24.0, SciPy 1.10.0, Matplotlib
  3.7.0, QuTiP 5.0.0). New:
  - `test_spin_cavity_121.py`: the eliminated model against the full
    spin-cavity model for two spins with detunings and uneven couplings.
    With `chi g^2` and `Gamma_SR g^2` fixed, the difference falls like
    `1/Delta` (0.0216, 0.0118, 0.0062; ratios 1.83 and 1.91, tolerance
    1.6 to 2.4 and below 0.01), while the 1.20 rates stay at 0.060 to
    0.065; both signs of `Delta` agree to 0.028 and 0.031 (tolerance
    0.05) and the flipped sign misses by 0.54; the thermal shift equals
    `2 n_th chi1 G^2` to 1e-12; flipping `Delta` mirrors the state of a
    symmetric line (`xi2` to 6e-14 relative, tolerance 1e-9); the photon
    cutoff holds at most 1.2e-7 of the population.
  - `test_calibration_nsigma_121.py`: weights against the formula rebuilt
    in the test (8e-13) and the coefficients and covariance against
    NumPy's `polyfit` (1.3e-13 of the error bars); `N_sigma=None` and 0
    give the 1.20 fit exactly; 200 seeded calibrations with 3 % errors in
    N, line and parabola: scatter 0.89 to 1.00 times the error bars
    (tolerance 0.8 to 1.2), no bias beyond 3 standard errors, 7 and 12 of
    200 fits with `p < 0.05`, and without `N_sigma` the scatter of `c1` is
    1.37 and 1.24 times its error bar and 66 and 59 fits give `p < 0.05`;
    refusals.
  - `test_dtwa_speed_121.py`: the 1.20 trajectory equations and stepping
    loop kept as the reference (2.4e-16 and 8e-16, tolerances 1e-12 and
    1e-9).
  - `test_fixes_121.py`: each fix above.
  The 18 new tests take about 10 s together.

## 1.20.0 (2026-11-10)

Independent-check and planning release: single-spin dephasing in the
trajectory solver, tomography planning for shots that are not Gaussian,
a corrected dephasing rate in the exact PIQS reference, and refusals for
continuous-measurement settings and a readout step that gave wrong
numbers without an error.

### Added

- `cavsqueeze.dtwa.evolve(..., gamma_phi=...)`: single-spin pure
  dephasing `L_j = sqrt(gamma_phi/2) sigma z_j` (the rate `1/T2` of
  `CavityParams.gamma_phi`) in the trajectory solver, which before
  covered only the twisting and the detunings. The dissipator is exactly
  the average over random turns of each spin about z with phase variance
  `2 gamma_phi t` (a stochastic Hamiltonian `(xi_j/2) sigma z_j` with
  white noise of strength `2 gamma_phi`), so each trajectory turns each
  spin by a Gaussian angle of variance `gamma_phi dt` before and after
  each interaction step (Strang splitting). Checked against the closed
  form without interaction, the exact master equation for 4 spins and
  PIQS for 40 spins (see Tests). The default 0 draws no extra random
  numbers, so seeded runs are unchanged. Refuses a negative or
  non-finite rate.
- `kurtosis` in `plan_tomography` and `shots_for_squeezing` (default
  `None`, the Gaussian results of 1.19): one number or one per angle,
  finite and at least 1. Each sample variance's error becomes the exact
  `sigma^2 sqrt(kurtosis/M - (M-3)/(M(M-1)))`, carried through the
  fixed-weight fit exactly (the covariance `variance_tomography` gives
  with `error_sigmas`), which is the error bar
  `estimate_squeezing(..., shot_noise="empirical")` reports.
  `shots_for_squeezing` then finds the smallest shot count by an exact
  integer search on the plan (the error of a sample variance falls
  strictly with `M` for any kurtosis of at least 1). This lifts the
  Limits item "the planning functions still assume Gaussian shots".
  README example 13: with a measured kurtosis of 7.47, a +/- 0.02 error
  bar on `xi2_R` costs 71506 shots per angle instead of 22116.

### Fixed

Each fix has a test in `tests/test_fixes_120.py` that fails on 1.19.0.

- `cavsqueeze.exact.dicke_piqs` (the exact PIQS reference used in the
  tests) had half the dephasing it was given: PIQS's dephasing rate
  makes each coherence decay at half that rate, and `gamma_phi` was
  passed unchanged, while the package's convention (and `full_hilbert`)
  is a decay at `gamma_phi`. For 4 spins with only dephasing
  (`gamma_phi = 0.7`), `<J_x>` at `t = 1` was 1.409 (`2 exp(-0.35)`) and
  is now 0.993 (`2 exp(-0.7)`), equal to `full_hilbert` to 1e-6 with
  every term on. The package's own solvers are unchanged. In
  `tests/test_exact.py` (40 spins, weak dephasing, `gamma_phi t` below
  0.002) the largest difference to the cumulant solver moves from 0.157
  to 0.164 dB (tolerance 0.2 dB); with strong dephasing
  (`gamma_phi t = 0.3`) the cumulant solver differed from the old
  reference by 1.2 dB and 2.0 in `<J>`, and from the corrected one by
  0.11 dB and 0.006.
- `evolve` did not check the continuous-measurement settings. With
  `Rates.meas_eta = 1.5` the final `Var(J_y) Var(J_z)` was 0.833 of its
  smallest allowed value `<J_x>^2 / 4` (it breaks the uncertainty
  relation); a negative `Rates.meas` made `Var(J_z)` of 10^6 spins grow
  to 2.5e11 (from 2.5e5); `nan` kept the integrator from finishing; and
  with half the spins as spectators, which are not conditioned although
  the measured `J_z` includes them, `Var(J_z)` came out 2.08e5 where
  measuring all of `J_z` gives 1.25e5 (`Gamma_m N t / 4 = 1`). A rate
  that is negative or not finite, an efficiency outside (0, 1], and the
  measurement together with spectators are now refused. Valid settings
  give the same results as before.
- `twist_untwist(..., phi=0)` returned a gain of `nan` (the slope is a
  central difference with steps `+/- phi`); a zero or non-finite `phi`
  is now refused.

### Changed

- Docstrings of `cavsqueeze.exact`: `dicke_piqs`'s `theta` is measured
  from `+z` (`<J_z> = (N/2) cos theta` at `t = 0`; the docstring said
  "from the south pole"), and the module states its dephasing
  convention. The state was always the one now described.
- CI: GitHub's `ubuntu-latest` image has moved to Ubuntu 26.04, which
  lacks some of the Python versions in the test matrix, so
  `.github/workflows/tests.yml` now runs every job on `ubuntu-24.04`.
  Nothing else in the workflows changed (`publish.yml` is unchanged).
- README: example 13 (planning with a kurtosis), the checks of the new
  features, the 1.20 refusals and corrections, and the Limits items on
  the trajectory solver, the measurement with spectators and the
  planners. README examples 1 to 12 print the same output as with
  1.19.0.

### Tests

- 279 tests pass and 3 skip (1.19.0: 251 and 3; the 3 skipped need the
  paper's companion scripts), on Python 3.13 and with the oldest allowed
  dependencies (Python 3.10, NumPy 1.24.0, SciPy 1.10.0, Matplotlib
  3.7.0, QuTiP 5.0.0). New: `test_fixes_120.py` (each fix above;
  `dicke_piqs` against the closed form `(N/2) exp(-gamma_phi t)` and
  against `full_hilbert` to 1e-6, and against the cumulant solver with
  strong dephasing), `test_dtwa_dephasing_120.py` (without interaction
  the mean spin within 4 standard errors of `exp(-gamma_phi t)` times
  free precession and the covariance within 0.05 of the exact spin-1/2
  values, 20000 trajectories; 4 spins with detunings, twisting and
  dephasing against the exact master equation to 0.08, 0.018 to 0.042
  for seeds 1 to 8, while leaving the dephasing out misses by 0.29 to
  0.32; 40 spins with strong dephasing against PIQS to 0.35 dB and 4
  standard errors in `<J_x>`, 0.06 to 0.27 dB for seeds 1 to 8) and
  `test_planning_kurtosis_120.py` (kurtosis 3 equals the Gaussian plan
  to 1e-12; the isotropic closed form `V sqrt(f/K)` and the scaling by
  `f (M-1)/2` to 1e-12; the plan equals the empirical error bars of
  `estimate_squeezing` to 1e-9; 300 seeded tomographies with outliers
  (kurtosis 7.65) and with uniform shots (1.8), scatter 0.85 to 1.15
  times the plan (0.96 and 0.99 measured) where the Gaussian plan is off
  by 1.75 and 0.63 times; the shot count against the separately solved
  quadratic condition for 4 kurtoses and 3 targets; refusals). The 28
  new tests take about 20 s together on the (loaded) build machine,
  most of it the 40-spin trajectory test (12 s).

## 1.19.0 (2026-10-18)

Readout-calibration release: a projection-noise calibration of the
detector, the error of the subtracted detector variance carried into the
squeezing estimate and the shot planner, a faster cumulant solver, and
fixes for repeated output times and invalid pulse inputs.

### Added

- `calibrate_projection_noise` / `ProjectionNoiseFit` (new module
  `cavsqueeze.calibration`): fits the shot-to-shot variance of `J_z` for
  coherent spin states at several atom numbers with `c0 + c1 N` (or
  `c0 + c1 N + c2 N^2` with `quadratic=True`). It reports the detector
  variance `c0`, `projection_ratio = 4 c1` (1 for the projection noise
  `N/4` of independent spins; W. M. Itano et al., Phys. Rev. A 47, 3554
  (1993)) and `p_projection`, the p-value of `c1 = 1/4`, all with error
  bars, and `chi2`, `dof`, `p_value` for the fit. The weights come from
  the fitted curve and are iterated, as in `estimate_squeezing`;
  `shot_noise="empirical"` uses each point's measured kurtosis. Refuses
  data it cannot fit reliably (too few or degenerate atom numbers, bad
  shots, a fitted curve that reaches zero). README example 12.
- `detection_variance_sigma` in `estimate_squeezing`, `plan_tomography`
  and `shots_for_squeezing` (default 0, the 1.18 results): the error bar
  of the subtracted detector variance. Subtracting a constant moves the
  fitted `var_min` by that constant and leaves the weights unchanged, so
  the error adds to that of `var_min` in quadrature without
  approximation. `SqueezingEstimate.detection_variance_sigma` records it
  (a new field with a default, at the end). `shots_for_squeezing` refuses
  a target below the floor `xi2_R sqrt((2 sigma_C / C)^2 +
  (sigma_det / var_min)^2)` and names it.
- Type hints on the public functions touched in this release.

### Fixed

Each fix has a test in `tests/test_fixes_119.py` that fails on 1.18.0.

- `ramsey_cumulant` and `ramsey_meanfield` were documented (1.18) to take
  the times in any order, but a repeated time, even in a sorted list such
  as `[1e-3, 1e-3]`, raised SciPy's `ValueError: Values in t_eval are not
  properly sorted`; so did `evolve` and `evolve_meanfield` with repeated
  or unsorted `t_eval`. They now compute the distinct times and return
  each requested time in the order given: for `T2` = 10 ms without
  coupling, `[2e-3, 1e-3, 2e-3, 0.0, 1e-3]` gives 0.818731, 0.904837,
  0.818731, 1.000000, 0.904837 (`exp(-t/T2)` to 1e-7). Results for
  strictly increasing times are unchanged (the same solver call).
- `pulse(..., duration=-1e-3)` (and `twist_imperfect` with a negative
  `pi_duration`) returned the instantaneous pulse without an error,
  exactly the `duration=0` state; `n_steps=0` raised `ZeroDivisionError`
  and `n_steps=2.5` a `TypeError`. Each is now a `ValueError` naming the
  input, as is a non-finite duration.

### Changed

- Docstrings of `CavityParams.chi` and `CavityParams.Gamma_SR` said "per
  unit g_j g_k", but the values include `g^2`: they are per unit product
  of the coupling weights (the module docstring's rates times `g^2`).
  Corrected; the values are unchanged.
- `tests/test_speed_118.py` compared the faster 1.18 code paths with the
  1.17 code bit for bit. The vectorized Dick coefficients call NumPy's
  array functions where 1.17 called scalar ones, whose last digits can
  differ between processors, and the 1.19 right-hand side forms its sums
  in a different order. These comparisons now use stated tolerances:
  1e-12 of the largest value for single evaluations and 1e-9 of the
  largest entry for a whole solver run. The README no longer calls these
  results bit-identical.
- Packaging: a new `MANIFEST.in` puts `examples/`, `CHANGELOG.md` and
  `CITATION.cff` into the source distribution, so that the tutorial tests
  and the citation-version test run from an unpacked sdist (before, they
  skipped there). The wheel is unchanged. No new build dependency.

### Performance

Measured on the shared 2-CPU build machine (heavily loaded by other jobs),
medians of repeated runs, the 1.18 and 1.19 right-hand sides in the same
session, interleaved.

- The cumulant solver's right-hand side forms each sum over the other
  classes once (the full sum minus the two excluded terms) and adds the
  two complex-conjugate halves of the `Zc` sources as `2 Re(c X)`, about
  half as many array operations. A 1 ms echo twist of one class (README
  example 1's setting) takes 0.554 s instead of 0.853 s (1.54x, medians
  of 7); a 20 us twist of 16 classes (example 2) 1.045 s instead of
  1.572 s (1.50x, medians of 7); README example 1 as a whole
  (`optimal_squeezing`) 21.7 s instead of 29.4 s (1.36x, medians of 3).
  The 1.18 code is kept in `tests/test_speed_119.py`: the right-hand
  sides agree to 1e-12 of the largest entry for random states (5e-16
  measured), and whole runs to 1e-9 of the largest state entry (8e-14
  measured) and 1e-6 in `xi2` (1e-9 measured). README examples 1 to 3
  and 8 print the same numbers as before.

### Tutorials

- `examples/tutorial_detection_noise.py`: coherent-state shots at six
  atom numbers calibrate the detector (projection noise checked, detector
  variance with its error bar); the tomography of a squeezed state read by
  the same detector is fitted without and with the correction (0.343 and
  0.242 +/- 0.015, true 0.2508); the last line plans the next run against
  the floor set by the contrast and calibration errors (about 2 s). The
  output is in the README ("Tutorials") and in the script's docstring;
  `tests/test_tutorials.py` runs it and checks every printed number.

### Tests

- 251 tests pass and 3 skip (1.18.0: 221 and 3; the 3 skipped need the
  paper's companion scripts), on Python 3.13 and with the oldest allowed
  dependencies (Python 3.10, NumPy 1.24.0, SciPy 1.10.0, Matplotlib
  3.7.0, QuTiP 5.0.0). New: `test_calibration_119.py` (exact variances
  recovered to 1e-9; coefficients and covariance against NumPy's
  `polyfit` to 1e-9 of the error bars; binomial projection noise over 200
  seeds, error bars within 0.8 to 1.2 of the scatter; the pulse-jitter law
  derived in the test; empirical error bars with heavy-tailed detection
  noise; the whole calibration-to-estimate chain over 200 seeds; the
  planner against the estimate to 1e-9; refusals), `test_fixes_119.py`
  (each fix above; the Ramsey contrast against `exp(-t/T2)` to 1e-7),
  `test_speed_119.py` (the 1.18 right-hand side as the reference) and the
  third tutorial in `test_tutorials.py`. The 30 new tests take about
  22 s together on the (loaded) build machine.

## 1.18.0 (2026-10-17)

Second audit and lab-tools release: a Voigt line and a stretched echo
from measured decays, two worked tutorials, fixes for wrong numbers that
came without an error (the largest: `oat_closed_form` for many spins),
and faster Dick-floor and single-class solver runs with bit-identical
results.

### Added

- `fit_coherence_decay(..., line="voigt", T2=..., T2_sigma=...)`: the
  Ramsey decay of a Voigt line,
  `A exp(-(1/T2 + F_L/2) t - s^2 t^2/2)` (the Fourier transform of a
  convolution is the product of the transforms). Returns the total FWHM
  of the exact Voigt profile and `lorentz_fraction = F_L / FWHM`, the
  inputs of `lineshape("voigt", fwhm, lorentz_fraction)`, plus
  `fwhm_gaussian` and `fwhm_lorentzian`, with error bars by the delta
  method (the error of the given `T2` included). New `DecayFit` fields
  at the end with default `None`, so existing code is unaffected. It
  refuses a fit without `T2` and a part of the line that is not
  resolved, naming the simpler line to use. This lifts the README limit
  "A Voigt line is not fitted".
- `fit_stretched_decay` / `StretchedDecayFit`: an echo decay fitted with
  `A exp(-(t/T_M)^x)`, the form used for rare-earth echoes (for example
  M. Le Dantec et al., arXiv:2106.14974 (2021), citing T. Bottger et
  al., Phys. Rev. B 73, 075101 (2006)), with error bars, the fit quality,
  and a test of `x = 1` (the exponential decay of the model):
  `chi2_exponential`, `delta_chi2` and `p_exponential` (chi-square with
  one degree of freedom).
- Type hints on the public functions touched in this release.

### Fixed

Each fix has a test in `tests/test_fixes_118.py` that fails on 1.17.0.

- `oat_closed_form` lost its accuracy for many spins: `cos(mu)**(N-2)`
  and `cos(mu/2)**(N-2)` carry about N times the rounding error of the
  cosine, and `1 - cos(mu)**(N-2)` and `A - sqrt(A^2 + B^2)` then lose
  the remaining digits. At the optimal angle
  `mu = 24^(1/6) (N/2)^(-2/3)` the smallest variance was off by
  -4.7e-7 (N = 10^6), -4.0e-4 (10^7) and +3.9e-2 (10^8) of its value
  (at half that angle: -6.8e-6, -1.9e-4 and +1.8e-2), and for N = 10^9
  at `mu = 2.696e-6` it was -2251.75 (true 275.16, `xi2_S` = -9.0e-6). The powers are now formed
  from `log1p(-sin^2(mu/2))` and the difference as
  `-B^2/(A + sqrt(A^2 + B^2))`: the smallest variance agrees with a
  60-digit evaluation to 1e-9 up to N = 10^9 and with exact Dicke-basis
  evolution at N = 10^6 to 1e-8. For N <= 1000 the values move by less
  than 1.1e-11 of `N/4` (README example 4 is unchanged).
- `evolve` with `t = 0` returned the starting state for every requested
  output time, so `evolve(st, rt, 0.0, t_eval=[0.5])` reported it as the
  state at 0.5 s; and `ramsey_cumulant` integrated to the last listed
  time, so `ramsey_cumulant(p, ens, [1e-3, 0.0])` gave contrast 1.0 at
  1 ms (true 0.904837 for `T2` = 10 ms). Output times outside `[0, t]`
  are now refused, and `ramsey_cumulant` and `ramsey_meanfield` accept
  times in any order and return them in the order given (results for
  sorted times are unchanged, bit for bit).
- `evolve_meanfield` with `t = 0`, and so `ramsey_meanfield(p, ens,
  [0.0])`, raised `TypeError`; it now returns the starting state.
- `save_shots_csv` dropped an angle without shots without a word (3
  angles saved, 2 loaded back), and wrote one-shot angles, `nan` angles
  and non-finite shots into files that `load_shots_csv` refused or that
  the estimators refuse. Each is now refused when saving.
- `sample_variance_sigma(shots, kurtosis=k)` returned `nan` for
  `k < (M-3)/(M-1)` or `k = nan` (and the error bar of an impossible
  distribution for `(M-3)/(M-1) <= k < 1`); every distribution has
  kurtosis >= 1, so these are now refused.
- `dick_allan_deviation` returned `nan` for a spectrum with `nan` or
  negative values at the harmonics, ignored the cutoff check for
  `tail_tol = nan`, and `dick_fourier_coefficients` truncated a
  fractional `m_max` (2.7 -> 2 harmonics). Now `ValueError`.
- The cutoff check of `dick_allan_deviation` looked at the top 10 % of
  the harmonics, which for a small `m_max` can be a single harmonic on a
  zero of the coefficients: with `m_max = 10`, `T_ramsey = 0.4`,
  `T_cycle = 1` and `h0 = 1e-30` it checked only `m = 10` (exactly zero)
  and returned 8.476e-16, 2.1 % below the converged 8.660e-16. The
  window now holds at least the top 100 harmonics (the top half when
  `m_max < 200`); for `m_max >= 1000`, the default 20000 included, it is
  the same window as before.
- `plan_tomography` and `shots_for_squeezing` raised NumPy's
  `LinAlgError` for a non-finite `theta_min`, and
  `tail_resolved_classes` raised `ZeroDivisionError` for `M_core = 0`;
  both now raise `ValueError` naming the input (also for class counts
  below 1 and edges that do not satisfy `0 < core_edge < delta_max`).

### Changed

- The `oat_closed_form` docstring said every line was asserted against
  exact evolution; it now says which quantities are checked, and to
  what tolerance.
- `tests/test_decay.py::test_refusals` pinned `line="voigt"` as an
  unknown line; it now expects the refusal without `T2`, and an unknown
  name (`"cauchy"`) for the unknown-line case.
- Packaging (was "Unreleased"): the license is given as the SPDX
  expression `license = "Apache-2.0"` with `license-files =
  ["LICENSE"]`, and the old license table and license classifier are
  gone. Building no longer prints setuptools' deprecation warnings
  (their support ends on 2027-02-18). Building from source now needs
  setuptools 77 or newer; installing the package is unchanged.

### Performance

Measured on the shared 2-CPU build machine, medians of repeated runs,
1.17.0 and 1.18.0 in the same session. Each new path is checked to be
bit-identical to the 1.17.0 code, which `tests/test_speed_118.py` keeps
as the reference.

- `dick_fourier_coefficients` computes all harmonics at once instead of
  looping over them: `dick_allan_deviation` with the default 20000
  harmonics takes 2.5 ms instead of 63.5 ms (25x) without pulses and
  25 ms instead of 368 ms (14.5x) with 10 ms pulses (medians of 9).
- The solver's right-hand side computes the rate constants once per
  `evolve` call and each repeated product once per step: a 1 ms run of
  one class (README example 1's setting) takes 0.197 s instead of
  0.337 s (1.7x, medians of 5); README example 1 as a whole took 19.4 s
  instead of 26.9 s (1.4x, medians of 6 runs on a loaded machine). For
  16 classes (example 2) the time is unchanged within the noise (0.58 s
  and 0.64 s).
- `optimal_squeezing` reuses the scan at the best time instead of
  computing it again.

### Tutorials

- `examples/tutorial_decays_to_model.py`: echo and Ramsey decays to
  `T2`, the stretch exponent and a Voigt line, then a squeezing
  prediction with an error bar (about 5 s).
- `examples/tutorial_tomography_to_clock.py`: fringe and tomography shot
  files to `xi2_R` with its error bar and chi-square checks, then the
  projected clock stability next to a Dick floor (about 1 s).
- Both outputs are in the README ("Tutorials") and in each script's
  docstring; `tests/test_tutorials.py` runs them and checks every
  printed number.

### Tests

- 221 tests pass and 3 skip (1.17.0: 170 and 3; the 3 skipped need the
  paper's companion scripts). New: `test_fixes_118.py` (each fix above;
  the large-N variance against a 60-digit evaluation to 1e-9 and Dicke
  evolution to 1e-8), `test_decay_118.py` (Voigt fit against the
  closed form to 1e-9 and against numerical Fourier integration of the
  profile to 1e-7; error bars within 0.8-1.2 of the scatter over 300
  datasets; stretched fit against the closed form to 1e-9, its
  exponential reference against SciPy's `curve_fit` to 1e-7, error bars
  and the `x = 1` test by Monte Carlo), `test_protocols_118.py`
  (`twist_untwist` and `plain_squeezed_readout`, untested before,
  against the linearized one-axis-twisting shear to 2e-5),
  `test_speed_118.py` (bit-identity with the 1.17.0 code) and
  `test_tutorials.py`. The new tests take about 40 s together.

## 1.17.0 (2026-10-10)

Lab-data release: `T2` and the line width from a laboratory's own echo
and Ramsey decays, a fit-quality test for the tomography and fringe
fits, and refusals for inputs that returned wrong numbers or `nan`
without an error. No result of a valid input changes.

### Added

- `fit_coherence_decay` / `DecayFit` (new module `cavsqueeze.decay`):
  fits a measured echo decay `A exp(-t/T2)` (`line="none"`), or a
  Ramsey decay with a Gaussian line, `A exp(-t/T2 - s^2 t^2/2)` with
  `s = FWHM / (2 sqrt(2 ln 2))`, or a Lorentzian line,
  `A exp(-(1/T2 + FWHM/2) t)`. Weighted nonlinear least squares on the
  contrast, linearized error bars, a known `T2` with its error bar
  propagated into the width, and `chi2`, `dof`, `p_value`. It refuses a
  Lorentzian width without a known `T2` (only the sum of the two rates
  can be measured), too few points, invalid inputs, and any fitted
  rate or width less than `min_significance` (default 2) error bars
  above zero. Results are in the units of `from_hz(T2=...)` and
  `lineshape(..., fwhm)`. README example 10.
- Fit quality for the measured-data fits: `SqueezingEstimate` and
  `ContrastEstimate` gain `chi2`, `dof` and `p_value` (new fields with
  defaults at the end, so existing code is unaffected), and
  `variance_tomography` returns them too. With
  `shot_noise="empirical"` the statistic is the quadratic form of the
  residuals in the pseudo-inverse of their exact covariance, which
  reduces to the usual weighted sum when the weights are the errors.
  README example 11.

### Fixed

Each fix has a test in `tests/test_fixes_117.py` that fails on 1.16.0.

- `oat_closed_form` truncated a number of spins that was not a whole
  number: `oat_closed_form(2.9, 0.3)` returned the 2-spin result
  (`xi2_S = 0.85056`), and `oat_closed_form(100.7, 0.05)` the 100-spin
  one (`xi2_S = 0.12511`). Now `ValueError`; `100.0` is still accepted.
  A non-finite `mu` (which gave `nan`) is refused too.
- `plan_tomography` truncated fractional shot counts: 9.9 shots per
  angle were planned as 9 (`var_min_sigma = 28.7227` for the 9-angle
  test design, where 10 shots give 27.0800). Now `ValueError`.
- `evolve_meanfield` accepted a negative time and integrated
  backwards: with `T2 = 1 ms`, `t = -1 ms` returned a contrast of
  2.7183 (`e`, a spin longer than 1). Now `ValueError`, as in `evolve`.
- Silent `nan` or `inf` results, now `ValueError`: a `nan` or infinite
  `contrast_sigma` in `estimate_squeezing` and `plan_tomography`
  (`xi2_R_sigma` was `nan` or `inf`); a `nan` or infinite `T_cycle` in
  `clock_allan_deviation`, `magnetometer_sensitivity`,
  `total_clock_allan_deviation` and `synchronized_comparison` (`nan`,
  or `inf` for an infinite cycle); a non-finite `T_ramsey`, `T_cycle`
  or `tau_pulse` in the Dick functions (the floor was `nan`, and
  `ramsey_sensitivity` returned zeros); non-finite `power_law_psd`
  coefficients; `log_uniform_weights` with 0 weights (empty arrays) or
  an infinite range (`nan` weights).

### Changed

- Docstrings corrected where they disagreed with the code:
  `optimal_squeezing` (its fine scan runs between the neighbors of the
  best coarse point, not over `[t*/2, 2 t*]`), `plain_squeezed_readout`
  (the gain is `1/(xi2 + 4 sigma_det^2 / (S2 C^2))`; the contrast `C`
  was missing), and `dick_allan_deviation` (the cutoff check looks at
  the top 10 % of the harmonics, not the "last decade", and bounds the
  omitted part only roughly: for terms falling as `1/m^2` the omitted
  part is about 9 times the checked one). The cutoff error message now
  says "top 10 %".
- Error messages name the offending values (cycle and Ramsey times,
  contrast and its error, laser-noise coefficients).
- `oat_closed_form` returns `Jx` as a Python `float` (it was a NumPy
  scalar; the value is identical).
- Type hints on the public functions touched in this release.

### Tests

- 170 tests pass and 3 skip (1.16.0: 123 and 3; the 3 skipped need the
  paper's companion scripts). New: `test_decay.py` (closed forms to
  1e-9; the cumulant solver's echo decay gives back its `T2` to 1e-7
  and the mean-field solver's Ramsey decay on a 20000-class Gaussian
  line gives back its FWHM and `T2` to 5e-5; over 300 seeded datasets
  the scatter is 0.8 to 1.2 times the reported error bars, with and
  without a known `T2`, and the propagated `T2` term equals a
  finite-difference refit to 2 %; a wrong model gives `p < 1e-6`;
  refusals), `test_goodness_of_fit.py` (the reported `chi2` against the
  residual sum recomputed from the shots to 1e-9; the general form
  against the usual sum to 1e-10 and against the chi-square law by
  Monte Carlo; uniform p-values with 200 shots per angle; 5 % angle or
  phase errors flagged), `test_fixes_117.py` (above), and
  `test_dick.py::test_white_frequency_noise_floor_matches_closed_form`
  (the Dick floor for white frequency noise against
  `h0 (1 - eta) / (2 eta tau)` to 6e-5, the shortfall being the omitted
  harmonics, to 1e-6 of itself).

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

### Behavior changes

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
