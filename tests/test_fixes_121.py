"""1.21: each test here fails on 1.20.0 and passes on 1.21.0 (the bugs and
the numbers before and after are in CHANGELOG.md).  The missing thermal
shift and the sign of Delta are tested in tests/test_spin_cavity_121.py."""
import os
import tempfile

import numpy as np
import pytest

import cavsqueeze as cs
from cavsqueeze import dtwa, load_shots_csv, plan_tomography, shots_for_squeezing

ANGLES = np.linspace(0.0, np.pi, 9, endpoint=False)
DESIGN = dict(var_min=90.0, var_max=160.0, N=480, contrast=0.92, contrast_sigma=0.01)


def test_shots_for_squeezing_returns_the_smallest_count_for_a_reachable_target():
    """With the target set to the error bar that m shots give,
    shots_for_squeezing must return m.  1.20 returned m + 1 for 1000 and
    22116 shots in the design of README example 6, and for 23 of 40 random
    counts with a detection-variance error (the closed form rounded up)."""
    for m0 in (50, 1000, 22116):
        t = plan_tomography(ANGLES, m0, **DESIGN)["xi2_R_sigma"]
        assert shots_for_squeezing(t, ANGLES, **DESIGN)[0] == m0
    rng = np.random.default_rng(1)
    extra = dict(detection_variance=5.0, detection_variance_sigma=1.0)
    for m0 in rng.integers(2, 3000, 40):
        m0 = int(m0)
        t = plan_tomography(ANGLES, m0, **DESIGN, **extra)["xi2_R_sigma"]
        m, plan = shots_for_squeezing(t, ANGLES, **DESIGN, **extra)
        assert m == m0 and plan["xi2_R_sigma"] <= t
        # the result promises: m meets the target, m - 1 does not
        assert plan_tomography(ANGLES, m - 1, **DESIGN, **extra)["xi2_R_sigma"] > t
    # an ordinary target is unchanged (README example 6)
    assert shots_for_squeezing(0.02, ANGLES, **DESIGN)[0] == 22116


def _write(rows):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as fh:
        fh.write("angle_rad,jz\n" + "".join(r + "\n" for r in rows))
    return path


def test_load_shots_csv_names_the_line_of_a_bad_value():
    """1.20 loaded an infinite J_z without a word, split the rows of the
    angle "nan" into angles of one shot each (refused as "angle nan carries
    1 shot(s)"), and gave Python's message without the line for text."""
    for rows, line in ((["0.0,1.0", "0.0,inf", "0.5,2.0", "0.5,3.0"], 3),
                       (["nan,1.0", "nan,2.0"], 2),
                       (["0.0,1.0", "0.0,abc"], 3)):
        path = _write(rows)
        try:
            with pytest.raises(ValueError, match=f"line {line}"):
                load_shots_csv(path)
        finally:
            os.remove(path)
    path = _write(["0.0,1.0", "0.0,-2.5", "1e-3,3.0", "1e-3,4.0"])
    try:
        th, sh = load_shots_csv(path)                # valid files load as before
        assert np.array_equal(th, [0.0, 1e-3]) and np.array_equal(sh[0], [1.0, -2.5])
    finally:
        os.remove(path)


def test_optimal_squeezing_refuses_scan_settings_that_do_not_scan():
    """n_coarse = 1 scanned t_lo alone and returned it as the best time
    (1.20: t = 1e-6, xi2 = 0.811 for README example 1, whose best is
    124 us); max_fine = 0 left out the fine scan; a nan points_per_period
    gave 8 fine points without a word."""
    N = 1e10
    p = cs.from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=0.15)
    ens = cs.homogeneous(N)
    for kw in (dict(n_coarse=1), dict(n_coarse=0), dict(n_coarse=2.0),
               dict(max_fine=0), dict(max_fine=7), dict(points_per_period=np.nan),
               dict(points_per_period=0)):
        with pytest.raises(ValueError):
            cs.optimal_squeezing(p, ens, 1e-6, 1e-2, **kw)
    with pytest.raises(ValueError, match="t_list"):
        cs.squeezing_trace(p, ens, [])                # 1.20: IndexError


def test_trajectory_solver_checks_its_inputs():
    """n_traj = 1 returned a covariance of nan (one sample has no spread);
    delta and G of different lengths ended in a NumPy broadcasting error;
    a nan chi1 gave nan moments."""
    t = [0.0, 0.1]
    with pytest.raises(ValueError, match="n_traj"):
        dtwa.evolve(np.zeros(4), np.ones(4), 1.0, t, n_traj=1)
    with pytest.raises(ValueError, match="one value per spin"):
        dtwa.evolve(np.zeros(4), np.ones(3), 1.0, t, n_traj=10)
    with pytest.raises(ValueError, match="finite"):
        dtwa.evolve(np.zeros(4), np.ones(4), np.nan, t, n_traj=10)
    with pytest.raises(ValueError, match="finite"):
        dtwa.evolve(np.array([0.0, np.nan]), np.ones(2), 1.0, t, n_traj=10)
    out = dtwa.evolve(np.zeros(4), np.ones(4), 1.0, t, n_traj=2)
    assert np.all(np.isfinite(out[-1]["cov"]))
