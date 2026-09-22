# cavsqueeze

[![Tests](https://github.com/TaN-MM-Org/cavsqueeze/actions/workflows/tests.yml/badge.svg)](https://github.com/TaN-MM-Org/cavsqueeze/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/cavsqueeze?label=PyPI&color=blue)](https://pypi.org/project/cavsqueeze/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22278034-blue)](https://doi.org/10.5281/zenodo.22278034)

`cavsqueeze` is a Python package for simulating and measuring **spin
squeezing** in a large group of spins (atoms, ions or defects in a
crystal) that all talk to one microwave or optical **resonator**
(a cavity). Squeezing is a way of preparing the spins so that they
measure a phase or a frequency more precisely than the same number of
independent spins could. The package was developed for
171Yb3+:CaWO4 (ytterbium ions in a calcium tungstate crystal) and
applies to any spin ensemble coupled to a resonator.

It answers these questions:

- How much squeezing does a given cavity and spin ensemble produce,
  and after how long, once the real imperfections of a solid are
  included: a spread of spin frequencies, decay into the cavity,
  thermal photons, loss of coherence, and uneven coupling?
- What does that squeezing buy a clock or a magnetometer?
- Given shot-by-shot data from an experiment, how much squeezing was
  actually there, and with what error bar?
- How many shots does a measurement need to reach a target error bar,
  before it is taken?
- When does the noise of the clock laser (the Dick effect) make
  squeezing useless?

The solver groups spins with the same frequency and coupling into
**classes**. Its memory and running time grow with the number of
classes, not with the number of spins: the state it stores has the
same size for 10^3 spins as for 10^15 (example 2 uses 10^15). The main
results are checked by automated tests against exact solutions, closed
formulas and an independent second solver (see
[How the results are checked](#how-the-results-are-checked)). When an
input cannot give a meaningful answer, the package stops with an error
that says why, rather than returning a number that looks fine but is
not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, requirements and units](#install-requirements-and-units)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Spin, collective spin.** Each spin is a two-level system (up or
  down). Adding up all N spins gives one large **collective spin** `J`
  with components `J_x`, `J_y`, `J_z`. `J_z = (N_up - N_down)/2`
  is what a population measurement reads out.
- **Coherent spin state.** All spins point the same way (here along
  `+x`). Its measurement noise is the **standard quantum limit**
  (SQL): a phase uncertainty of `1/sqrt(N)` radians per shot. This
  noise, which comes from each spin landing randomly up or down when
  it is measured, is called **projection noise**.
- **Squeezing.** Correlations between the spins shrink the noise in
  one direction perpendicular to the mean spin, at the cost of more
  noise in the other. Two numbers measure it, both equal to 1 for a
  coherent spin state and smaller than 1 when squeezed:
  - `xi2_S` (**Kitagawa-Ueda parameter**): the smallest variance
    across the mean spin, divided by its coherent-state value `N/4`.
  - `xi2_R` (**Wineland parameter**): `xi2_S` divided by the contrast
    squared. It is the one that sets phase precision. The gain over
    the SQL in decibels is `-10 log10(xi2_R)`.
- **Contrast.** The length of the mean spin as a fraction of its
  largest possible value, `2|J|/N`. Decay and dephasing shrink it.
- **One-axis twisting (OAT).** The interaction that creates the
  squeezing here: each spin's precession speed depends on `J_z`, which
  shears the noise circle into an ellipse.
- **Cavity-mediated interaction.** The spins do not touch each other
  directly; they exchange energy through the cavity. With the cavity
  tuned away from the spins by `Delta` (the **detuning**), this gives
  a twisting strength `chi` and a collective decay rate `Gamma_SR`
  (formulas in `help(cavsqueeze.resonator)`). `g` is the coupling
  of one spin to the cavity and `kappa` the cavity linewidth.
- **Inhomogeneous line.** In a solid the spins do not all have the
  same frequency. The spread is the **line**, described by its shape
  (Gaussian, Lorentzian or Voigt) and its full width at half maximum
  (FWHM).
- **Cumulant expansion.** The solver follows averages and pairwise
  correlations of the spins and drops higher-order correlations
  (a "second-order", Gaussian approximation). Tests compare it with
  exact solutions where those exist.
- **Echo.** A pi pulse (a pulse that turns every spin by half a
  turn) half way through the twisting. It undoes the dephasing
  (spreading out of the spins' directions) caused by static frequency
  differences.
- **Ramsey measurement, tomography.** A Ramsey sequence turns an
  accumulated phase into a population difference. **Fringe**: the
  mean `J_z` as the phase of the last pulse is scanned. **Tomography**:
  the shot-to-shot variance of `J_z` as the noise ellipse is rotated
  by an angle `theta`; its smallest value is the squeezed variance.
- **Allan deviation.** The standard measure of a clock's fractional
  frequency instability after averaging for a time `tau`.
- **Dick effect.** A pulsed clock looks at its laser (the local
  oscillator) only part of the time, so laser noise leaks into the
  result. It sets a floor that no squeezing can lower.
- **Exact references.** QuTiP is a Python library for quantum
  simulations. A **master-equation** solution computes the full
  quantum state, including decay; it is exact but only feasible for a
  few spins. **PIQS** (part of QuTiP) solves the same problem exactly
  for many *identical* spins by using their symmetry (the **Dicke**
  basis).

## Install, requirements and units

```
pip install cavsqueeze          # core package
pip install cavsqueeze[exact]   # adds QuTiP, needed for cavsqueeze.exact and to_qutip
```

To run the tests from a copy of the source:

```
git clone https://github.com/TaN-MM-Org/cavsqueeze
cd cavsqueeze
pip install -e .[test]
pytest tests
```

It needs Python 3.10 or newer, NumPy 1.24 or newer, SciPy 1.10 or
newer and Matplotlib 3.7 or newer (Matplotlib is a declared
dependency, although no module of the package imports it). QuTiP 5.0
or newer is optional. QuTiP 5.0.0 imports `setuptools`, so with that
QuTiP version `setuptools` must also be installed.

Units and conventions:

- Inside the package every rate and frequency is an **angular**
  frequency in rad/s. `from_hz` takes ordinary frequencies in Hz and
  multiplies them by 2 pi. Line widths passed to `lineshape` are also
  in rad/s.
- Temperature `T` in kelvin, times in seconds, `T2` in seconds
  (`from_hz` sets the dephasing rate to `1/T2`).
- `J_z` and all variances are in spin units: `J_z = (N_up - N_down)/2`,
  variances in spin units squared. Angles and phases are in radians.
- The spin state variable is `<sigma^+> = (x + i y)/2`. A spin whose
  frequency is above the reference (detuning `delta > 0`) turns from
  `+x` towards `+y`.
- The protocols start from the coherent spin state along `+x`.

## Examples

Each example runs as written, and the output shown is what it printed
with cavsqueeze 1.15.1. All device values are illustrative
(chosen to show the behavior), not the values of a specific
experiment. Examples that draw random numbers use a fixed seed.

### 1. How much squeezing does a cavity give, and when?

```python
import numpy as np
from cavsqueeze import from_hz, homogeneous, optimal_squeezing

# Illustrative values, not a specific device.
N = 1e10                                   # number of spins
p = from_hz(g_hz=1e6 / np.sqrt(N),         # coupling of one spin (Hz)
            kappa_hz=1e4,                  # cavity linewidth (Hz)
            Delta_hz=30e6,                 # cavity-spin detuning (Hz)
            T=0.02,                        # temperature (K)
            T2=0.15)                       # spin coherence time (s)
best = optimal_squeezing(p, homogeneous(N), 1e-6, 1e-2)
print(f"best squeezing {10 * np.log10(best['xi2']):.2f} dB "
      f"after {best['t'] * 1e6:.1f} us, contrast {best['contrast']:.4f}")
```

```
best squeezing -23.59 dB after 124.3 us, contrast 0.9992
```

`optimal_squeezing` scans the twisting time between the two limits
given (here 1 us to 10 ms), with an echo pulse in the middle, and
returns the time with the smallest `xi2_R`. It first scans a coarse
grid of 12 times spaced evenly on a log scale, then a finer linear
grid between the neighbors of the best coarse point. `best` also
holds the whole scan (`scan_t`, `scan_xi2`).

### 2. A disordered line, with 10^15 spins

```python
import numpy as np
from cavsqueeze import (from_hz, homogeneous, lineshape,
                        equal_probability_classes, squeezing_after)

N = 1e15
p = from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6,
            T=0.02, T2=0.15)
# A Gaussian line 2 kHz wide (FWHM), cut into 16 classes of equal weight.
line = lineshape("gaussian", 2 * np.pi * 2e3)
for name, ens in [("one class, no disorder", homogeneous(N)),
                  ("16 classes, 2 kHz line", equal_probability_classes(line, 16, N))]:
    r = squeezing_after(p, ens, 2e-5)
    print(f"{name:24s} xi2 = {10 * np.log10(r['xi2']):6.2f} dB, "
          f"contrast {r['contrast']:.4f}")
```

```
one class, no disorder   xi2 = -12.86 dB, contrast 0.9999
16 classes, 2 kHz line   xi2 = -12.84 dB, contrast 0.9972
```

Both runs take seconds: the size of the problem is the number of
classes (1 or 16), not the number of spins. `squeezing_after` uses an
echo by default, which removes most of the effect of the static
frequency spread; the contrast still drops a little.

### 3. What the squeezed state buys a clock

```python
import numpy as np
from cavsqueeze import (Rates, from_hz, homogeneous, css_x, twist,
                        squeezing_parameters, metrological_gain_db,
                        clock_allan_deviation)

N = 1e8
p = from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6,
            T=0.02, T2=0.15)
ens = homogeneous(N)
rt = Rates.from_params(p, ens)            # the model's rates, per class
state = twist(css_x(rt.M), rt, 5e-5)      # 50 us of twisting, with an echo

m = squeezing_parameters(state, ens.n)
print(f"xi2_S = {m['xi2_S']:.5f}, xi2_R = {m['xi2_R']:.5f}, "
      f"contrast = {m['contrast']:.5f}")
print(f"gain over the standard quantum limit: "
      f"{metrological_gain_db(m['xi2_R']):.2f} dB")
print(f"phase noise per shot: {m['dphi']:.3e} rad "
      f"(standard quantum limit {1 / np.sqrt(N):.3e} rad)")
# Illustrative clock: 429 THz transition, 0.1 s Ramsey time, 1 s averaging.
print(f"projected Allan deviation: "
      f"{clock_allan_deviation(m['dphi'], nu0=4.29e14, T_ramsey=0.1, tau=1.0):.3e}")
```

```
xi2_S = 0.01013, xi2_R = 0.01013, contrast = 0.99967
gain over the standard quantum limit: 19.94 dB
phase noise per shot: 1.007e-05 rad (standard quantum limit 1.000e-04 rad)
projected Allan deviation: 1.181e-20
```

`twist` evolves the state; `squeezing_parameters` reads both squeezing
parameters, the contrast and the per-shot phase noise `dphi` from it.
`clock_allan_deviation` uses the standard projection-noise formula
`sigma_y = dphi / (2 pi nu0 T_ramsey) * sqrt(T_cycle / tau)`, with the
cycle time equal to the Ramsey time unless you give `T_cycle`.

### 4. The exact one-axis-twisting benchmark

```python
from cavsqueeze import oat_closed_form

for mu in (0.0, 0.05, 0.2):
    r = oat_closed_form(100, mu)
    print(f"mu = {mu:4.2f}: <Jx> = {r['Jx']:7.3f}, Vmin = {r['Vmin']:6.3f}, "
          f"xi2_S = {r['xi2_S']:.4f}, xi2_R = {r['xi2_R']:.4f}")
```

```
mu = 0.00: <Jx> =  50.000, Vmin = 25.000, xi2_S = 1.0000, xi2_R = 1.0000
mu = 0.05: <Jx> =  48.477, Vmin =  3.128, xi2_S = 0.1251, xi2_R = 0.1331
mu = 0.20: <Jx> =  30.453, Vmin =  3.946, xi2_S = 0.1578, xi2_R = 0.4255
```

`oat_closed_form(N, mu)` gives the Kitagawa-Ueda closed-form results for
N spins twisted by the angle `mu` with no decay and no disorder
(`mu = 2 chi t`). The dissipative solver can be compared against it.
At `mu = 0.2` the smallest variance has grown again and the contrast
has fallen, so `xi2_R` is much worse than `xi2_S`: twisting too long
costs precision.

### 5. Squeezing from measured shot data

```python
import os, tempfile
import numpy as np
from cavsqueeze import (oat_closed_form, save_shots_csv, load_shots_csv,
                        estimate_contrast, estimate_squeezing,
                        metrological_gain_db)

# Make a synthetic, seeded experiment: N = 100 spins, one-axis-twisted
# by mu = 0.05 (the exact answer is known from oat_closed_form).
N, rng = 100, np.random.default_rng(0)
oat = oat_closed_form(N, 0.05)
C_true = 2 * oat["Jx"] / N                         # true fringe contrast

# Fringe scan: mean J_z = (N C / 2) cos(phase), plus projection noise.
phases = np.linspace(0, 2 * np.pi, 12, endpoint=False)
fringe = [0.5 * N * C_true * np.cos(ph) + rng.normal(0, 5, 400)
          for ph in phases]
# Tomography: J_z variance at angle theta follows the squeezed ellipse.
a = oat["alpha_min"]
cov = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ \
    np.diag([oat["Vmin"], oat["Vmax"]]) @ \
    np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
angles = np.linspace(0, np.pi, 12, endpoint=False)
tomo = [rng.normal(0, np.sqrt(np.array([np.cos(t), np.sin(t)]) @ cov
                              @ np.array([np.cos(t), np.sin(t)])), 4000)
        for t in angles]

folder = tempfile.mkdtemp()
save_shots_csv(os.path.join(folder, "fringe.csv"), phases, fringe)
save_shots_csv(os.path.join(folder, "tomography.csv"), angles, tomo)

ph, fr = load_shots_csv(os.path.join(folder, "fringe.csv"))
cest = estimate_contrast(ph, fr, N=N)
th, sh = load_shots_csv(os.path.join(folder, "tomography.csv"))
est = estimate_squeezing(th, sh, N=N, contrast=min(cest.contrast, 1.0),
                         contrast_sigma=cest.contrast_sigma)
print(f"contrast: {cest.contrast:.4f} +/- {cest.contrast_sigma:.4f} "
      f"(true {C_true:.4f})")
print(f"xi2_R:    {est.xi2_R:.4f} +/- {est.xi2_R_sigma:.4f} "
      f"(true {oat['xi2_R']:.4f})")
print(f"gain:     {metrological_gain_db(est.xi2_R):.2f} dB")
```

```
contrast: 0.9713 +/- 0.0020 (true 0.9695)
xi2_R:    0.1421 +/- 0.0043 (true 0.1331)
gain:     8.47 dB
```

The data here are synthetic, drawn from the exact twisted state of
example 4 so the true answer is known. With real data you would only
run the last block. The steps:

1. `save_shots_csv` / `load_shots_csv` read and write shot records in a
   plain-text format: a header `angle_rad,jz`, then one row per shot.
   Rows with the same angle form that angle's shots.
2. `estimate_contrast` fits the fringe. The mean follows
   `A cos(phi) + B sin(phi) + d` for any state, so the fit is a
   weighted linear least-squares fit: a standard fit whose answer is
   given by a formula, with no iterative search.
3. `estimate_squeezing` fits the tomography. The variance at angle
   `theta` follows `c + a cos(2 theta) + b sin(2 theta)` for any state
   (it is how a 2x2 covariance matrix rotates), so this fit is also
   linear. Its only assumption is in the error bars: the uncertainty
   of each sample variance is taken as `s^2 sqrt(2/(M-1))`, which holds
   for Gaussian shot noise. A known detection-noise variance can be
   subtracted with `detection_variance=...`.

In this run the estimate is about two error bars from the true value,
which is within normal scatter for one experiment.

### 6. Planning the number of shots

```python
import numpy as np
from cavsqueeze import plan_tomography, shots_for_squeezing

# Illustrative plan: 480 spins, expected variances 90 and 160 (spin units^2),
# contrast 0.92 +/- 0.01, nine equally spaced angles.
angles = np.linspace(0.0, np.pi, 9, endpoint=False)
plan = plan_tomography(angles, shots_per_angle=50, var_min=90.0,
                       var_max=160.0, N=480, contrast=0.92,
                       contrast_sigma=0.01)
print(f"50 shots per angle: xi2_R = {plan['xi2_R']:.4f} "
      f"+/- {plan['xi2_R_sigma']:.4f}")

m, plan = shots_for_squeezing(0.02, angles, 90.0, 160.0, N=480,
                              contrast=0.92, contrast_sigma=0.01)
print(f"for +/- 0.02: {m} shots per angle (gives {plan['xi2_R_sigma']:.4f})")

try:
    shots_for_squeezing(0.015, angles, 90.0, 160.0, N=480,
                        contrast=0.92, contrast_sigma=0.01)
except ValueError as err:
    print("refused:", err)
```

```
50 shots per angle: xi2_R = 0.8861 +/- 0.1159
for +/- 0.02: 22116 shots per angle (gives 0.0200)
refused: target 0.015 is below the contrast floor 0.0193 = xi2_R * 2 sigma_C / C: no number of tomography shots reaches it -- improve the fringe (contrast) measurement instead
```

Because the tomography fit is linear, its error bars depend only on
the design (angles, shots per angle, the variances you expect), so
they can be computed before measuring. `plan_tomography` returns them;
`shots_for_squeezing` finds the smallest number of shots per angle
that meets a target error bar on `xi2_R`. The contrast error sets a
floor, `xi2_R * 2 sigma_C / C`, that more tomography shots cannot
lower, so a target below it is refused. Equally spaced angles over
half a turn are the efficient choice: they make the three fitted
numbers independent.

### 7. When the laser, not the atoms, limits a clock

```python
from cavsqueeze import (power_law_psd, total_clock_allan_deviation,
                        synchronized_comparison)

# Illustrative clock: 429 THz, 0.4 s Ramsey time in a 1 s cycle.
# The laser noise level h0 is a placeholder -- use your laser's measured value.
laser = power_law_psd(h0=1e-30)
for label, dphi in [("no squeezing", 1e-2), ("10 dB squeezing", 3.16e-3)]:
    r = total_clock_allan_deviation(dphi, nu0=4.29e14, T_ramsey=0.4,
                                    tau=1.0, T_cycle=1.0, S_y=laser)
    print(f"{label:16s} projection {r['qpn']:.2e}  Dick {r['dick']:.2e}  "
          f"total {r['total']:.2e}  Dick-limited: {r['dick_limited']}")

# Two ensembles read by the same laser: the Dick term cancels in the difference.
out = synchronized_comparison(3.16e-3, 3.16e-3, nu0=4.29e14, T_ramsey=0.4,
                              tau=1.0, T_cycle=1.0)
print(f"synchronized comparison: difference {out['differential']:.2e}, "
      f"per clock {out['per_clock']:.2e}")
```

```
no squeezing     projection 9.27e-18  Dick 8.66e-16  total 8.66e-16  Dick-limited: True
10 dB squeezing  projection 2.93e-18  Dick 8.66e-16  total 8.66e-16  Dick-limited: True
synchronized comparison: difference 4.14e-18, per clock 2.93e-18
```

With this laser noise the Dick floor is about 100 times the projection
noise, so 10 dB of squeezing leaves the total unchanged
(`dick_limited` is `True`). The Dick floor is computed from the
laser's noise spectrum through the Fourier coefficients of the Ramsey
timing, in closed form; `power_law_psd` builds the usual
`h0 + h_-1/f + h_-2/f^2` spectrum from your measured coefficients (the
package ships none).

Record clock comparisons avoid the Dick effect by reading two
ensembles with the same laser in the same cycles, so the laser noise
is common to both and cancels in their difference (Robinson et al.,
Nat. Phys. 20, 208 (2024): 1.9(2) dB of stability enhancement at the
10^-17 level; Yang et al., PRL 135, 193202 (2025): 2.0(2) dB beyond
the standard quantum limit at 1.1x10^-18). `synchronized_comparison`
models exactly that: the difference has the two projection noises
added in quadrature and no Dick term, and `per_clock` is the
difference divided by sqrt(2). Noise that is not common to both
ensembles (for example path-length differences or gradients) is not
modeled and adds on top.

### 8. Handing the state to QuTiP

```python
import numpy as np
from cavsqueeze import (Rates, from_hz, homogeneous, css_x, twist, bosonic_mode,
                        to_qutip)

N = 1e8
p = from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6,
            T=0.02, T2=0.15)
rt = Rates.from_params(p, homogeneous(N))
state = twist(css_x(rt.M), rt, 2e-5)

mode = bosonic_mode(state, rt.n)
print(f"r = {mode['r']:.4f}, thermal occupation = {mode['n_th']:.5f}, "
      f"purity = {mode['purity']:.5f}, squeezing = {mode['squeezing_db']:.2f} dB")
rho, mode = to_qutip(state, rt.n)            # needs QuTiP
print("QuTiP density matrix:", rho.shape, f"trace {rho.tr().real:.6f}")
```

```
r = 1.4828, thermal occupation = 0.00239, purity = 0.99525, squeezing = -12.86 dB
QuTiP density matrix: (216, 216) trace 1.000000
```

For a strongly polarized spin, the small fluctuations across the mean
spin behave like one bosonic mode (the Holstein-Primakoff picture).
`bosonic_mode` gives that mode's 2x2 covariance, scaled so that a
coherent spin state is the vacuum (variance 1/2), and splits it into
a squeezing strength `r`, an angle, a thermal occupation and a purity.
`to_qutip` builds the matching QuTiP density matrix, then checks that
the exported state reproduces the two extreme variances to 1 %, and
raises if the Fock-space cutoff (the number of photon-number levels
QuTiP keeps) was too small. It keeps second moments (variances and
covariances) only.

## What is in the package

Every name below is importable from `cavsqueeze` directly. Each
docstring (`help(cavsqueeze.estimate_squeezing)`, for example) gives the
inputs, units and conventions.

**Cavity and spins**

- `CavityParams`, `from_hz` -- coupling `g`, cavity linewidth `kappa`,
  detuning `Delta`, spin frequency, temperature and dephasing; the
  twisting and collective-decay rates follow from them.
- `thermal_occupation` -- the thermal photon number at a frequency and
  temperature.
- `loop_gap_dispersive(N, T, T2)` -- the parameters of the dispersive
  loop-gap resonator of Fukumori et al. (arXiv:2604.26909):
  g/2pi = 15 mHz, kappa/2pi = 660 kHz, Delta/2pi = 22 MHz.
- `Ensemble`, `homogeneous` -- a set of classes (detuning, coupling
  weight, number of spins); one class with no disorder.
- `lineshape` -- Gaussian, Lorentzian or Voigt line from its FWHM.
- `equal_probability_classes` -- cuts a line into M classes of equal
  weight. `product_classes` combines this with a spread of coupling
  strengths, and `log_uniform_weights` makes such a spread.
  (`cavsqueeze.ensemble.tail_resolved_classes` resolves the far tails
  of a Lorentzian line and treats spins far outside it as free
  "spectator" spins.)

**Solver** (second-order cumulant expansion)

- `Rates` -- the model rates per class, from `Rates.from_params(params,
  ensemble)`. Setting `Rates.meas` (and `meas_eta`, the detection
  efficiency) adds a continuous measurement of `J_z` through the
  resonator and conditions the state on it.
- `State`, `product_state` -- the solver state: per-class averages and
  pair correlations.
- `evolve`, `evolve_meanfield` -- time evolution with and without
  correlations.
- `rotate` -- an instantaneous rotation (pulse) of all spins.
- `collective_moments`, `transverse_variances`, `wineland_xi2`,
  `coherence` -- mean collective spin and covariance; the variances
  across the mean spin; the Wineland parameter; the Ramsey contrast.

**Pulse sequences**

- `css_x` -- the starting state (coherent spin state along `+x`).
- `twist`, `twist_imperfect`, `pulse` -- twisting with or without an
  echo; an echo with a pulse of finite length or uneven angle; a
  pulse during which the interaction continues.
- `squeezing_after`, `squeezing_trace`, `optimal_squeezing` --
  squeezing after one time, over a list of times, and at the best
  time.
- `ramsey_cumulant`, `ramsey_meanfield` -- Ramsey decay of the
  contrast.
- `twist_untwist` -- the twist-untwist readout of Davis, Bentsen,
  Schleier-Smith, PRL 116, 053601 (2016), with detection noise.
- `plain_squeezed_readout` -- the gain of an ordinary squeezed Ramsey
  readout with detection noise.

**Metrology**

- `squeezing_parameters`, `phase_sensitivity`, `metrological_gain_db`
  -- `xi2_S`, `xi2_R`, contrast and per-shot phase noise; gain in dB.
- `clock_allan_deviation`, `magnetometer_sensitivity` -- the
  projection-noise-limited clock instability and Ramsey magnetometer
  field noise.
- `oat_closed_form` -- exact one-axis twisting without decay.

**Dick effect**

- `ramsey_sensitivity`, `dick_fourier_coefficients` -- the Ramsey
  sensitivity function over one cycle and its Fourier coefficients in
  closed form.
- `power_law_psd` -- the laser noise spectrum from its coefficients.
- `dick_allan_deviation`, `total_clock_allan_deviation` -- the Dick
  floor; the floor and the projection noise together, with a
  `dick_limited` verdict.
- `synchronized_comparison` -- two ensembles read by one laser.

**Measured data**

- `save_shots_csv`, `load_shots_csv` -- the shot-record file format.
- `estimate_contrast` (returns `ContrastEstimate`) -- fringe contrast
  and its error bar.
- `estimate_squeezing` (returns `SqueezingEstimate`) and the fit behind
  it, `variance_tomography` -- squeezing parameters and error bars
  from tomography shots.
- `plan_tomography`, `shots_for_squeezing` -- predicted error bars and
  the shot count for a target.

**Bridge to QuTiP**

- `bosonic_mode`, `to_qutip` -- see example 8.

**Reference solvers** (not imported at the package root)

- `cavsqueeze.exact` (needs QuTiP) -- exact master-equation solutions
  for a few distinguishable spins (`full_hilbert`) and the
  permutation-invariant Dicke solver PIQS for identical spins
  (`dicke_piqs`).
- `cavsqueeze.cumulant_raw` -- the same cumulant equations written in
  raw moments, kept as a test reference.
- `cavsqueeze.dtwa` -- an independent solver based on the discrete
  truncated Wigner approximation (J. Schachenmayer, A. Pikovski and
  A. M. Rey, Phys. Rev. X 5, 011022 (2015)): many classical
  trajectories started from random samples of the initial state. It
  covers twisting and detunings (no decay).

## When it refuses, and why

`cavsqueeze` raises `ValueError` instead of guessing when:

- `lineshape` is given a name other than gaussian, lorentzian or voigt;
- `oat_closed_form` is asked for fewer than 2 spins;
- a clock or magnetometer cycle time `T_cycle` is shorter than the
  Ramsey time;
- the Ramsey timing is impossible (`T_ramsey <= 0`, a negative pulse
  length, or a cycle shorter than the Ramsey time plus two pulses), or
  fewer than 1 harmonic is asked for;
- laser noise coefficients are negative;
- the harmonic cutoff of `dick_allan_deviation` is too low: if the top
  10 % of the harmonics still carry more than `tail_tol` (default
  1e-3) of the sum, it asks for a larger `m_max` rather than returning
  a truncated answer;
- `synchronized_comparison` gets a per-shot phase noise that is not
  finite and positive;
- the mean spin is zero (there is then no direction to squeeze across),
  the covariance is not positive definite, or it breaks the
  uncertainty relation (`bosonic_mode`);
- the Fock-space cutoff distorts the state exported by `to_qutip`;
- a fit has fewer than 3 angles, or angles that give fewer than 3
  different directions (angles that differ by a half turn are the same
  direction for the tomography; by a full turn for the fringe);
- a shot array has fewer than 2 shots, non-finite values or zero
  scatter, or the number of shot arrays does not match the angles;
- `N` is not a whole number of at least 2;
- the contrast given to `estimate_squeezing` or `plan_tomography` is
  not in (0, 1], or its error is negative;
- the fitted fringe contrast exceeds 1 by more than two error bars
  (usually a wrong `N` or a `J_z` calibration problem; the message
  says so). A smaller overshoot is returned unclipped;
- the detection variance is negative or not finite, or subtracting it
  leaves a smallest variance of zero or less;
- a shot file is empty, has the wrong header, a row without exactly two
  fields, no data rows, or an angle with fewer than 2 shots; saving
  refuses repeated angles;
- the planned variances do not satisfy `0 < var_min <= var_max`, or a
  target error bar is not positive, uses a degenerate angle set, or is
  below the contrast floor (example 6).

Also: `evolve` raises `RuntimeError` if SciPy's integrator reports a
failure, and a planned design with fewer than 3 distinct directions is
reported by `plan_tomography` as `identifiable = False` (and refused
by `shots_for_squeezing`).

## How the results are checked

69 tests run on every push to `main` and every pull request, on
Python 3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.10
with the oldest versions the package allows (NumPy 1.24.0, SciPy 1.10.0,
Matplotlib 3.7.0, QuTiP 5.0.0). A further job runs on Python 3.14
without QuTiP; there the tests that need QuTiP are skipped. 3 more
tests always skip here: they check scripts of the paper's companion
repository, which are not in this repository. The whole suite takes a
few minutes; most of it is one test of the trajectory solver.

**The cumulant solver**

- For 2 spins the second-order equations are complete, so the solver
  must equal the exact QuTiP master-equation solution: every pair
  correlation agrees to 2e-6 and every average to 2e-6 plus 1e-5 of
  its size, for three random sets of detunings and couplings, with
  decay, thermal absorption and dephasing switched on.
- For 4 spins it agrees with the exact solution to 2e-3 at short times
  and to 0.15 at the end of the run (the error of the approximation).
- For 40 identical spins with decay, thermal absorption and dephasing,
  its Wineland parameter agrees with the exact PIQS solution to 0.2 dB
  and its mean spin to 5 % of `N/2`, up to a twisting strength
  `chi N t = 1.5`.
- The production solver (correlations stored with the averages
  subtracted) agrees with the raw-moment form: averages to 1e-8 plus
  1e-5 of their size, correlations to 1e-7, the Wineland parameter to
  1e-7, for classes of one and of several spins.
- Rotations: a coherent spin state keeps `xi2 = 1` to 1e-12, and
  rotating back and forth returns the state to 1e-12 plus 1e-5 of
  each value's size. A pulse of vanishing length equals an
  instantaneous rotation to 1e-5.
- With the measurement switched on and nothing else, the variance of
  `J_z` follows the closed-form result `(N/4)/(1 + Gamma_m N t/4)` to
  1e-5 relative, and the uncertainty product `Var(J_y) Var(J_z)` stays
  at its smallest allowed value, `(N/4)^2`, to 1e-6 relative.
- The mean-field equations (averages only, no correlations) match the
  pendulum equations of the paper to 0.2 %.
- A leading-order formula for the growth rate of the `J_z` variance
  under collective emission (Sec. S7 of the paper) matches an exact
  6-spin master-equation solution to 0.2 %. This test uses QuTiP only;
  it checks the formula, not the solver.

**The independent trajectory solver**

- For 200 identical spins, its best Wineland parameter matches the
  exact one-axis-twisting result to 0.05 dB (4000 trajectories, seed
  1). This tolerance is smaller than the random scatter between
  seeds: with seeds 1 to 4 the difference is +0.03, -0.13, +0.03 and
  -0.22 dB, so the test passes for seed 1 but would not for every
  seed.
- Making its time step four times smaller (200 to 800 steps) changes
  the answer by less than 0.01 dB.
- Without interaction, a Gaussian line dephases as `exp(-t^2/2)` to
  0.02.
- For 4 spins with different detunings, its mean spin follows exact
  free precession when there is no interaction, and agrees with the
  cumulant solver when there is, to 0.08 in spin units (4000
  trajectories). This pins the sign of `J_y` (new in 1.15.1).

**Metrology and one-axis twisting**

- A coherent spin state gives `xi2_S = xi2_R = 1`, contrast 1 and
  `dphi = 1/sqrt(N)`.
- On a state squeezed by the solver, `xi2_R = xi2_S / contrast^2` to
  1e-10 and equals the solver's own Wineland parameter to 1e-12.
- The clock and magnetometer formulas follow the expected scalings
  with `tau`, atom number and dead time.
- `oat_closed_form` against brute-force exact evolution for N = 2, 5
  and 8 at random twisting angles: the mean spin, and the variance at
  the predicted best angle, to 1e-12; the smallest and largest
  variance over a 2001-point angle grid to 1e-4. `mu = 0` gives exactly
  `xi2_S = 1`.

**Dick effect**

- The closed-form Fourier coefficients equal independent adaptive
  numerical quadrature to 1e-9 (three cycle layouts, with and without
  finite pulses).
- Continuous interrogation (no dead time, no pulses) gives every
  coefficient below 1e-12 and a Dick deviation below 1e-18.
- Without pulses, the size of each coefficient divided by the zeroth
  one equals `|sin(pi m eta)/(pi m eta)|` to 1e-12, where `eta` is the
  Ramsey time divided by the cycle time.
- The floor scales as `1/sqrt(tau)` to 1e-12; doubling the harmonic
  cutoff changes it by less than 1e-3; too low a cutoff is refused.
- With a noisy laser, about 10 dB of squeezing improves the total by
  less than 10 %; with a quiet laser, by more than 3 times.
- `synchronized_comparison`: for two equal ensembles the difference is
  sqrt(2) times one clock (to 1e-18 relative) and `per_clock` equals
  one clock (to 1e-15 relative); the single-ensemble values equal
  `clock_allan_deviation` exactly.

**Measured data and planning**

- A noise-free tomography record returns the covariance matrix to
  1e-9, and the smallest and largest variances equal NumPy's
  eigenvalues to 1e-9.
- Shots drawn from the exact one-axis-twisting state return its
  `xi2_R` within 4 reported error bars.
- Coherent-spin-state shots return `xi2_R` compatible with 1 (within 4
  error bars). Through the full file-to-estimate path, with a fitted
  contrast, the test allows 4 error bars plus 0.1 (and the contrast
  4 error bars plus 0.02 from 1).
- Over 150 (squeezing) and 60 (contrast) seeded repeats, the scatter of
  the estimates is between 0.6 and 1.6 times (squeezing) and 0.5 and
  2 times (contrast) the reported error bar.
- A clean fringe returns contrast and phase to 1e-7.
- Shot files round-trip exactly.
- The planned covariance equals the fitted one to 1e-12, equally
  spaced angles give the diagonal design to 1e-12, and over 300 seeded
  experiments the scatter matches the planned error bar to 20 %.
- `bosonic_mode` maps a coherent spin state to the vacuum, and its
  squeezing agrees with `squeezing_parameters` to 1e-10; the QuTiP
  export reproduces the covariance at 25 angles to 1e-4; a deliberately
  small Fock cutoff is caught.

The remaining tests check refusals, further identities of the same
functions, properties of the paper's pendulum model (the locking
condition and a conserved quantity), and that the version number is
the same in the package, its metadata and CITATION.cff.

## Corrections in earlier versions

**1.15.1 fixed a sign in the trajectory solver.** `cavsqueeze.dtwa`
integrates the same equations as the cumulant solver, whose variable
is `<sigma^+> = (x + i y)/2`, but read the results out as if the
variable were `<sigma^->`. The `J_y` component of the mean spin, and
the `xy` and `yz` covariances, therefore came out with the wrong sign
(in the 4-spin case of the new test at `t = 1`, `<J_y> = -0.34`
where the cumulant solver gives `+0.33`).
Squeezing parameters did not change, because they do not depend on
that sign. The read-out is fixed, seeded runs give the same squeezing
parameters as before, and a new test checks the sign.

**Corrected claims (documentation only, 1.15.1).**

- Earlier documents said the Dick coefficients gave "exactly zero"
  noise for continuous interrogation and the "exact"
  rectangular-window ratio; the tests check these to 1e-12. The
  module docstring of `cavsqueeze.dick` also said the coefficients
  were checked against an FFT; the test uses adaptive quadrature.
- The 1.10.0 notes said every quantity returned by `oat_closed_form`
  is checked against exact evolution to 1e-12. The mean spin and the
  smallest variance are; the largest variance is checked to 1e-4 on
  an angle grid, and `xi2_S` and `xi2_R` are computed from these,
  not compared separately.
- "Machine precision" and "exact identity" for the planner mean
  agreement to 1e-12.
- The `dtwa` module said the cumulant solver was checked against 8
  distinguishable spins; the tests use 2 and 4.

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- The cavity is eliminated: its field follows the spins instantly,
  which holds when the cavity is far detuned or much faster than the
  spin dynamics.
- The solver keeps averages and pair correlations only. Tests bound
  the resulting error for small systems (above); for strongly
  non-Gaussian states it grows.
- The trajectory solver covers the twisting and the detunings, not
  decay or thermal photons.
- The continuous-measurement option does not condition the spectator
  spins of `tail_resolved_classes`; use a discretization without
  spectators with it.
- The error bars of the estimators assume Gaussian shot noise for the
  sample-variance uncertainty. Heavy-tailed detection noise makes them
  too small.
- The clock and magnetometer formulas use linear error propagation
  around the operating point (the same approximation that defines
  `xi2_R`).
- The Voigt line splits its width between the Gaussian and Lorentzian
  parts with the Olivero-Longbothum approximation.
- The bosonic export keeps second moments only.

## Where it comes from

The package was developed for 171Yb3+:CaWO4 and applies to any spin
ensemble coupled to a resonator. The physics, the conventions and the
validation are described in: T. M. Mahim, M. M. Rahman, A. S. M.
Mohsin, *Synchronization sets the coherence and the squeezing limit of
a spin ensemble in a cavity*. The paper's companion repository,
[yb-cawo4-cavity-squeezing](https://github.com/Tanvir-Mahmud-Mahim/yb-cawo4-cavity-squeezing),
contains the scripts, datasets and figures that reproduce the paper and
is archived on Zenodo; this repository is the software's home for
development, releases and support.

Other sources named in the code: Kitagawa and Ueda, Phys. Rev. A 47,
5138 (1993); Wineland et al., Phys. Rev. A 46, R6797 (1992); Sorensen
et al., Nature 409, 63 (2001); Itano et al., Phys. Rev. A 47, 3554
(1993); Ludlow et al., Rev. Mod. Phys. 87, 637 (2015); G. J. Dick,
Proc. PTTI 1987; G. Santarelli et al., IEEE Trans. Ultrason.
Ferroelectr. Freq. Control 45, 887 (1998); A. Quessada et al., J. Opt.
B 5, S150 (2003); Schulte et al., Nat. Commun. 11, 5955 (2020).

## Citing, support and license

Please cite the paper if you use this code. To cite the software, use
the concept DOI
[10.5281/zenodo.22278034](https://doi.org/10.5281/zenodo.22278034),
which always resolves to the latest release. Citation metadata is in
[CITATION.cff](CITATION.cff).

The package is maintained by Tanvir Mahmud Mahim (BRAC University),
who reviews issues and pull requests; the scientific content is
developed with the co-authors of the associated paper. Bug reports,
questions and pull requests are welcome through
[GitHub issues](https://github.com/TaN-MM-Org/cavsqueeze/issues); see
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the
testing requirements. Tagged releases are published to PyPI by CI.

Licensed under Apache-2.0 (see [LICENSE](LICENSE)).
