"""Tutorial: from measured echo and Ramsey decays to a squeezing prediction.

A laboratory measures two decays of its own spin ensemble, with the cavity
tuned far away so that the spins do not interact:

1. a Hahn echo, which gives the coherence time T2 (and tests whether the
   echo decay is exponential, as the model assumes);
2. a Ramsey decay, which also contains the inhomogeneous line, here a Voigt
   line (Gaussian and Lorentzian parts together).

The fitted T2 and line go into the model, which then predicts the
squeezing after a chosen twisting time, with an error bar propagated from
the fits.  The "measured" contrasts below are synthetic (seeded), drawn
from known values so that the answer can be compared with the truth:
T2 = 5 ms, a Voigt line of total FWHM 150 Hz with a Lorentzian fraction of
0.3, contrast 0.9 at t = 0, and an error of 0.005 on each contrast.

Run:  python examples/tutorial_decays_to_model.py   (about 5 s)

Printed output (cavsqueeze 1.18.0):

    echo:     T2 = 5.028 +/- 0.038 ms, exponent x = 1.002 +/- 0.016, p(x = 1) = 0.89
    Ramsey:   FWHM = 148.8 +/- 1.9 Hz, Lorentzian fraction = 0.315 +/- 0.026, p = 0.61
    model:    xi2_R after 1.0 ms = -4.55 +/- 0.06 dB (true line and T2: -4.56 dB)
"""
import numpy as np

from cavsqueeze import (equal_probability_classes, fit_coherence_decay,
                        fit_stretched_decay, from_hz, lineshape, squeezing_after)

rng = np.random.default_rng(2026)
err = 0.005                                        # error of each contrast

# --- synthetic "measurements" -------------------------------------------
T2_true, F_true, frac_true = 5e-3, 2 * np.pi * 150.0, 0.3
true_line = lineshape("voigt", F_true, frac_true)  # sigma and gamma of its parts
t_echo = np.linspace(0.5e-3, 15e-3, 15)            # total echo time (s)
echo = 0.9 * np.exp(-t_echo / T2_true) + rng.normal(0, err, t_echo.size)
t_ramsey = np.linspace(0.2e-3, 6e-3, 15)           # s
ramsey = (0.9 * np.exp(-t_ramsey / T2_true - true_line.gamma * t_ramsey
                       - 0.5 * (true_line.sigma * t_ramsey) ** 2)
          + rng.normal(0, err, t_ramsey.size))
sig = np.full(15, err)

# --- 1. echo: T2, and is the decay exponential? --------------------------
e = fit_coherence_decay(t_echo, echo, sig)                      # exp(-t/T2)
s = fit_stretched_decay(t_echo, echo, sig)                      # exp(-(t/T_M)^x)
print(f"echo:     T2 = {e.T2 * 1e3:.3f} +/- {e.T2_sigma * 1e3:.3f} ms, "
      f"exponent x = {s.exponent:.3f} +/- {s.exponent_sigma:.3f}, "
      f"p(x = 1) = {s.p_exponential:.2f}")

# --- 2. Ramsey: the Voigt line, with the echo's T2 and its error bar ------
r = fit_coherence_decay(t_ramsey, ramsey, sig, line="voigt", T2=e.T2, T2_sigma=e.T2_sigma)
print(f"Ramsey:   FWHM = {r.fwhm / (2 * np.pi):.1f} +/- {r.fwhm_sigma / (2 * np.pi):.1f} Hz, "
      f"Lorentzian fraction = {r.lorentz_fraction:.3f} +/- {r.lorentz_fraction_sigma:.3f}, "
      f"p = {r.p_value:.2f}")

# --- 3. the model with the measured inputs ------------------------------
N, t_twist = 1e12, 1e-3                             # illustrative device


def xi2_db(T2, fwhm, frac):
    p = from_hz(g_hz=1e5 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=T2)
    ens = equal_probability_classes(lineshape("voigt", fwhm, frac), 16, N)
    return 10 * np.log10(squeezing_after(p, ens, t_twist)["xi2"])


best = xi2_db(e.T2, r.fwhm, r.lorentz_fraction)
# error bar: change each input by its error bar, add the changes in quadrature
# (the fitted FWHM and fraction are treated as independent here)
shifts = [xi2_db(e.T2 + e.T2_sigma, r.fwhm, r.lorentz_fraction) - best,
          xi2_db(e.T2, r.fwhm + r.fwhm_sigma, r.lorentz_fraction) - best,
          xi2_db(e.T2, r.fwhm, r.lorentz_fraction + r.lorentz_fraction_sigma) - best]
truth = xi2_db(T2_true, F_true, frac_true)
print(f"model:    xi2_R after {t_twist * 1e3:.1f} ms = {best:.2f} +/- "
      f"{np.sqrt(np.sum(np.square(shifts))):.2f} dB (true line and T2: {truth:.2f} dB)")
