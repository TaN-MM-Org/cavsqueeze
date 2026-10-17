"""Tutorial: from shot records of a squeezed state to a clock's stability.

An experiment with N = 1000 spins records two shot files (one row per shot,
the format of save_shots_csv): a fringe scan, which gives the contrast,
and a tomography scan of the squeezed state, which gives the squeezed
variance.  This script goes from those files to the squeezing parameter
with its error bar, checks that both scans follow their laws (chi2 and
p-values), and turns the result into the projected stability of a clock,
next to the Dick floor of an illustrative laser.

The shots are synthetic (seeded) and drawn from the exact one-axis-twisted
state of N = 1000 spins at mu = 0.003 (oat_closed_form), with Gaussian
noise, so the true values are known: contrast 0.9989, xi2_R = 0.2508
(6.01 dB).  With real data, start at "load_shots_csv".

Run:  python examples/tutorial_tomography_to_clock.py   (about 1 s)

Printed output (cavsqueeze 1.18.0):

    contrast: 0.9988 +/- 0.0007 (fringe fit chi2 = 8.8 for 13 dof, p = 0.79)
    xi2_R:    0.2473 +/- 0.0096 = 6.07 +/- 0.17 dB (tomography chi2 = 3.5 for 9 dof, p = 0.94)
    clock:    projection 1.459e-17 +/- 2.8e-19 (no squeezing 2.933e-17), Dick 2.739e-18, total 1.484e-17 at 1 s
"""
import os
import shutil
import tempfile

import numpy as np

from cavsqueeze import (estimate_contrast, estimate_squeezing, load_shots_csv,
                        metrological_gain_db, oat_closed_form, power_law_psd,
                        save_shots_csv, total_clock_allan_deviation)

N = 1000
rng = np.random.default_rng(18)

# --- synthetic records, written as the experiment would write them --------
oat = oat_closed_form(N, 0.003)
C = 2 * oat["Jx"] / N                                   # true contrast
phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
fringe = [0.5 * N * C * np.cos(ph) + rng.normal(0, np.sqrt(N) / 2, 300) for ph in phases]
a = oat["alpha_min"]
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
cov = R @ np.diag([oat["Vmin"], oat["Vmax"]]) @ R.T
angles = np.linspace(0, np.pi, 12, endpoint=False)
tomo = [rng.normal(0, np.sqrt(np.array([np.cos(t), np.sin(t)]) @ cov
                              @ np.array([np.cos(t), np.sin(t)])), 1000) for t in angles]
folder = tempfile.mkdtemp()
save_shots_csv(os.path.join(folder, "fringe.csv"), phases, fringe)
save_shots_csv(os.path.join(folder, "tomography.csv"), angles, tomo)

# --- analysis -------------------------------------------------------------
ph, fr = load_shots_csv(os.path.join(folder, "fringe.csv"))
c = estimate_contrast(ph, fr, N=N)
print(f"contrast: {c.contrast:.4f} +/- {c.contrast_sigma:.4f} "
      f"(fringe fit chi2 = {c.chi2:.1f} for {c.dof} dof, p = {c.p_value:.2f})")

th, sh = load_shots_csv(os.path.join(folder, "tomography.csv"))
shutil.rmtree(folder)                                   # the synthetic files
est = estimate_squeezing(th, sh, N=N, contrast=min(c.contrast, 1.0),
                         contrast_sigma=c.contrast_sigma)
gain = metrological_gain_db(est.xi2_R)
gain_sigma = 10 / np.log(10) * est.xi2_R_sigma / est.xi2_R   # dB, delta method
print(f"xi2_R:    {est.xi2_R:.4f} +/- {est.xi2_R_sigma:.4f} = {gain:.2f} +/- "
      f"{gain_sigma:.2f} dB (tomography chi2 = {est.chi2:.1f} for {est.dof} dof, "
      f"p = {est.p_value:.2f})")

# --- what it buys a clock: projection noise and the Dick floor ------------
# phase noise per shot dphi = sqrt(xi2_R / N); illustrative clock at 429 THz,
# 0.4 s Ramsey time in a 1 s cycle, averaged for 1 s, and a laser with a
# white frequency noise level h0 (a placeholder for a measured value)
dphi = np.sqrt(est.xi2_R / N)
dphi_sigma = 0.5 * dphi * est.xi2_R_sigma / est.xi2_R
laser = power_law_psd(h0=1e-35)
out = total_clock_allan_deviation(dphi, nu0=4.29e14, T_ramsey=0.4, tau=1.0,
                                  T_cycle=1.0, S_y=laser)
sql = total_clock_allan_deviation(1 / np.sqrt(N), nu0=4.29e14, T_ramsey=0.4,
                                  tau=1.0, T_cycle=1.0, S_y=laser)
print(f"clock:    projection {out['qpn']:.3e} +/- {out['qpn'] * dphi_sigma / dphi:.1e} "
      f"(no squeezing {sql['qpn']:.3e}), Dick {out['dick']:.3e}, "
      f"total {out['total']:.3e} at 1 s")
