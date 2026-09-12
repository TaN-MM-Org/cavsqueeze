"""v1.12 experimental anchors: the fringe contrast fit held to exact
recovery, sampled-shot statistics and its refusals; the shot-record
file contract's exact round trip; and the complete file-to-estimate
pipeline feeding estimate_squeezing with a contrast the package
itself fitted."""
import numpy as np
import pytest

import cavsqueeze as cs


def _fringe_shots(N, C, phi0, offset, noise, K, M, rng):
    ph = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    shots = [0.5 * N * C * np.cos(p - phi0) + offset
             + rng.normal(0.0, noise, M) for p in ph]
    return ph, shots


def test_contrast_exact_recovery_on_clean_fringe():
    """Deterministic shots (tiny scatter for the SEM) recover the
    generating contrast, phase and offset to high precision."""
    rng = np.random.default_rng(0)
    N, C, phi0, off = 400, 0.87, 0.6, 3.0
    ph, shots = _fringe_shots(N, C, phi0, off, 1e-6, 12, 20, rng)
    est = cs.estimate_contrast(ph, shots, N)
    assert abs(est.contrast - C) < 1e-7
    assert abs(est.phi0 - phi0) < 1e-7
    assert abs(est.offset - off) < 1e-6
    assert abs(est.amplitude - 0.5 * N * C) < 1e-4


def test_contrast_statistics_and_reported_sigma():
    """Monte-Carlo scatter of the estimate matches its reported
    uncertainty, and the truth lies within it."""
    N, C, phi0 = 400, 0.9, 0.3
    noise = 0.5 * np.sqrt(N) / 2.0
    ests, sigs = [], []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        ph, shots = _fringe_shots(N, C, phi0, 0.0, noise, 10, 50, rng)
        e = cs.estimate_contrast(ph, shots, N)
        ests.append(e.contrast)
        sigs.append(e.contrast_sigma)
    ests = np.asarray(ests)
    assert abs(ests.mean() - C) < 3.0 * ests.std() / np.sqrt(ests.size)
    assert 0.5 < ests.std() / np.mean(sigs) < 2.0


def test_contrast_refusals():
    rng = np.random.default_rng(1)
    N = 100
    # degenerate phases: 0 and 2 pi are the same quadrature
    ph = np.array([0.0, 2.0 * np.pi, 4.0 * np.pi])
    shots = [rng.normal(0.0, 1.0, 10) for _ in ph]
    with pytest.raises(ValueError, match="degenerate"):
        cs.estimate_contrast(ph, shots, N)
    # an impossible contrast is reported as an N/calibration problem
    ph2, shots2 = _fringe_shots(400, 0.9, 0.0, 0.0, 1e-6, 10, 20, rng)
    with pytest.raises(ValueError, match="exceeds 1"):
        cs.estimate_contrast(ph2, shots2, 100)     # N four times too small
    with pytest.raises(ValueError):
        cs.estimate_contrast(ph2[:2], shots2[:2], 400)


def test_shot_record_round_trip_and_refusals(tmp_path):
    rng = np.random.default_rng(2)
    angles = np.array([0.0, 0.7, 1.9, 2.4])
    shots = [rng.normal(0.0, 3.0, m) for m in (5, 8, 6, 7)]   # ragged
    p = tmp_path / "shots.csv"
    cs.save_shots_csv(p, angles, shots)
    a2, s2 = cs.load_shots_csv(p)
    assert np.array_equal(angles, a2)
    for x, y in zip(shots, s2):
        assert np.array_equal(x, y)
    q = tmp_path / "bad.csv"
    q.write_text("theta,jz\n0.0,1.0\n")
    with pytest.raises(ValueError, match="header"):
        cs.load_shots_csv(q)
    q.write_text("angle_rad,jz\n0.0,1.0\n0.0,2.0\n0.5,1.0\n")
    with pytest.raises(ValueError, match="at least 2"):
        cs.load_shots_csv(q)
    with pytest.raises(ValueError, match="distinct"):
        cs.save_shots_csv(q, np.array([0.1, 0.1]), [shots[0], shots[1]])


def test_file_to_squeezing_pipeline(tmp_path):
    """The complete experimental path with no hand-entered contrast:
    fringe record -> estimate_contrast, tomography record ->
    estimate_squeezing, both through the file contract. A
    coherent-spin-state record must be compatible with the standard
    quantum limit xi2_R = 1."""
    rng = np.random.default_rng(3)
    N = 400
    # fringe scan of the CSS (full contrast), projection-noise scatter
    ph, fringe = _fringe_shots(N, 1.0, 0.0, 0.0, np.sqrt(N) / 2.0,
                               12, 80, rng)
    pf = tmp_path / "fringe.csv"
    cs.save_shots_csv(pf, ph, fringe)
    ph2, fringe2 = cs.load_shots_csv(pf)
    cest = cs.estimate_contrast(ph2, fringe2, N)
    assert abs(cest.contrast - 1.0) < 4.0 * cest.contrast_sigma + 0.02
    # tomography record of the CSS: isotropic variance N/4
    th = np.linspace(0.0, np.pi, 7, endpoint=False)
    tomo = [rng.normal(0.0, np.sqrt(N) / 2.0, 60) for _ in th]
    pt = tmp_path / "tomo.csv"
    cs.save_shots_csv(pt, th, tomo)
    th2, tomo2 = cs.load_shots_csv(pt)
    est = cs.estimate_squeezing(th2, tomo2, N=N,
                                contrast=min(cest.contrast, 1.0),
                                contrast_sigma=cest.contrast_sigma)
    assert abs(est.xi2_R - 1.0) < 4.0 * est.xi2_R_sigma + 0.1
