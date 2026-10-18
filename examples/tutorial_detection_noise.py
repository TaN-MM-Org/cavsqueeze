"""Tutorial: calibrate the detector on coherent states, then remove its noise
from a squeezing measurement, with the calibration's error bar carried along.

A real detector adds noise of its own to every J_z shot.  Left in the data,
it makes a squeezed state look less squeezed.  The standard remedy has two
steps:

1. Projection-noise calibration.  Prepare the coherent spin state on the
   equator at several atom numbers N and measure J_z many times at each.
   The shot-to-shot variance follows Var(J_z) = c0 + c1 N: c1 = 1/4 is
   the projection noise of independent spins, and c0 is the detector's own
   variance.  `calibrate_projection_noise` fits this line, tests c1 = 1/4,
   and returns c0 with its error bar.
2. Subtraction.  `estimate_squeezing(..., detection_variance=c0,
   detection_variance_sigma=...)` subtracts c0 from the tomography
   variances and adds its error bar to that of the squeezing.

Finally the error of c0 sets a floor under the error bar of xi2_R that no
number of tomography shots can lower; `shots_for_squeezing` names it and
plans the shots for a target above it.

All shots are synthetic (seeded).  The coherent-state shots are binomial
(each of N spins up with probability 1/2); the squeezed state is the exact
one-axis-twisted state of N = 1000 spins at mu = 0.003 (oat_closed_form:
contrast 0.9989, xi2_R = 0.2508).  The detector adds Gaussian noise of
variance 25 (spin units squared) to every shot.  With real data, start at
the two fits.

Run:  python examples/tutorial_detection_noise.py   (about 2 s)

Printed output (cavsqueeze 1.19.0):

    calibration: detection variance 25.3 +/- 3.2, 4 c1 = 1.014 +/- 0.027 (p = 0.60), fit p = 0.89
    squeezing:   xi2_R = 0.3430 +/- 0.0088 raw, 0.2417 +/- 0.0154 = 6.17 +/- 0.28 dB corrected (true 0.2508)
    planning:    error-bar floor 0.0127 from contrast and calibration; +/- 0.015 needs 2396 shots per angle
"""
import numpy as np

from cavsqueeze import (calibrate_projection_noise, estimate_contrast,
                        estimate_squeezing, metrological_gain_db, oat_closed_form,
                        shots_for_squeezing)

rng = np.random.default_rng(19)
DET_SD = 5.0                                      # detector noise (spin units)

# --- 1. calibration shots: coherent states at six atom numbers -------------
N_cal = np.array([200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
# each shot: every one of the n spins is up with probability 1/2 (J_z is
# then a binomial count minus n/2), plus the detector's noise
cal_shots = [(rng.random((2000, int(n))) < 0.5).sum(axis=1) - n / 2
             + rng.normal(0, DET_SD, 2000) for n in N_cal]

# --- 2. the squeezed state: fringe and tomography shots ---------------------
N = 1000
oat = oat_closed_form(N, 0.003)
C = 2 * oat["Jx"] / N                              # true contrast
phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
fringe = [0.5 * N * C * np.cos(ph) + rng.normal(0, np.sqrt(N / 4 + DET_SD ** 2), 300)
          for ph in phases]
a = oat["alpha_min"]
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
cov = R @ np.diag([oat["Vmin"], oat["Vmax"]]) @ R.T
angles = np.linspace(0, np.pi, 12, endpoint=False)
tomo = [rng.normal(0, np.sqrt(np.array([np.cos(t), np.sin(t)]) @ cov
                              @ np.array([np.cos(t), np.sin(t)]) + DET_SD ** 2), 2000)
        for t in angles]

# --- analysis ---------------------------------------------------------------
cal = calibrate_projection_noise(N_cal, cal_shots)
print(f"calibration: detection variance {cal.detection_variance:.1f} +/- "
      f"{cal.detection_variance_sigma:.1f}, 4 c1 = {cal.projection_ratio:.3f} +/- "
      f"{cal.projection_ratio_sigma:.3f} (p = {cal.p_projection:.2f}), "
      f"fit p = {cal.p_value:.2f}")

c = estimate_contrast(phases, fringe, N=N)
contrast = min(c.contrast, 1.0)
raw = estimate_squeezing(angles, tomo, N=N, contrast=contrast,
                         contrast_sigma=c.contrast_sigma)
est = estimate_squeezing(angles, tomo, N=N, contrast=contrast,
                         contrast_sigma=c.contrast_sigma,
                         detection_variance=cal.detection_variance,
                         detection_variance_sigma=cal.detection_variance_sigma)
db_sigma = 10 / np.log(10) * est.xi2_R_sigma / est.xi2_R      # dB, delta method
print(f"squeezing:   xi2_R = {raw.xi2_R:.4f} +/- {raw.xi2_R_sigma:.4f} raw, "
      f"{est.xi2_R:.4f} +/- {est.xi2_R_sigma:.4f} = {metrological_gain_db(est.xi2_R):.2f} "
      f"+/- {db_sigma:.2f} dB corrected (true {oat['xi2_R']:.4f})")

# --- planning the next run with this calibration ----------------------------
floor = est.xi2_R * np.hypot(2 * c.contrast_sigma / contrast,
                             cal.detection_variance_sigma / est.var_min)
m, plan = shots_for_squeezing(0.015, angles, est.var_min, est.var_max, N=N,
                              contrast=contrast, contrast_sigma=c.contrast_sigma,
                              theta_min=est.theta_min,
                              detection_variance=cal.detection_variance,
                              detection_variance_sigma=cal.detection_variance_sigma)
print(f"planning:    error-bar floor {floor:.4f} from contrast and calibration; "
      f"+/- 0.015 needs {m} shots per angle")
