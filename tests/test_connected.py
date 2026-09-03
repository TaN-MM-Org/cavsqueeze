"""The connected-variable solver must reproduce the raw-moment solver exactly
(same hierarchy, different variables), including classes with several spins."""
import numpy as np
from cavsqueeze import cumulant as C
from cavsqueeze import cumulant_raw as Raw


def _run(seed, M, nvec):
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=M) * 1.0
    G = rng.uniform(0.6, 1.4, size=M)
    rt = C.Rates(delta=delta, G=G, n=np.array(nvec, float), chi1=0.4, Gd=0.3, Gu=0.08, gamma_phi=0.05)
    v = np.array([np.sin(1.2) * np.cos(0.4), np.sin(1.2) * np.sin(0.4), np.cos(1.2)])
    times = np.linspace(0, 1.5, 4)
    raw = Raw.evolve(Raw.product_state(M, v), rt, times[-1], t_eval=times, rtol=1e-11, atol=1e-13)
    con = C.evolve(C.product_state(M, v), rt, times[-1], t_eval=times, rtol=1e-11, atol=1e-14)
    for k in range(len(times)):
        P, Q, R, Z = con[k].raw()
        assert np.allclose(con[k].s, raw[k].s, atol=1e-8)
        assert np.allclose(con[k].z, raw[k].z, atol=1e-8)
        for a, b in ((P, raw[k].P), (Q, raw[k].Q), (R, raw[k].R), (Z, raw[k].Z)):
            assert np.max(np.abs(a - b)) < 1e-7


def test_connected_equals_raw_single_spins():
    _run(0, 4, [1, 1, 1, 1])


def test_connected_equals_raw_classes():
    _run(1, 3, [2, 5, 3])


def test_pulses_and_squeezing_agree():
    M = 3
    rt = C.Rates(delta=np.array([-0.3, 0.0, 0.4]), G=np.array([0.9, 1.0, 1.1]), n=np.array([4.0, 6.0, 5.0]),
                 chi1=0.2, Gd=0.1, Gu=0.02, gamma_phi=0.01)
    st_c = C.rotate(C.evolve(C.rotate(C.product_state(M, [1, 0, 0]), [1, 0, 0], np.pi), rt, 0.8), [0, 1, 0], 0.3)
    st_r = Raw.rotate(Raw.evolve(Raw.rotate(Raw.product_state(M, [1, 0, 0]), [1, 0, 0], np.pi), rt, 0.8), [0, 1, 0], 0.3)
    xc = C.wineland_xi2(st_c, rt.n)[0]
    xr = Raw.wineland_xi2(st_r, rt.n)[0]
    assert abs(xc - xr) < 1e-7
    xcw = C.wineland_xi2(st_c, rt.n, weights=rt.G)[0]
    xrw = Raw.wineland_xi2(st_r, rt.n, weights=rt.G)[0]
    assert abs(xcw - xrw) < 1e-7


def test_class_rotation_and_finite_pulse_limits():
    """rotate_classes with equal angles equals the global rotation, and a pulse of
    vanishing duration equals an instantaneous rotation."""
    from cavsqueeze.protocols import pulse
    M = 3
    rt = C.Rates(delta=np.array([-0.3, 0.0, 0.4]), G=np.array([0.9, 1.0, 1.1]), n=np.array([4.0, 6.0, 5.0]),
                 chi1=0.2, Gd=0.1, Gu=0.02, gamma_phi=0.01)
    rt.spec_delta = np.zeros(0)
    rt.spec_n = np.zeros(0)
    st = C.evolve(C.product_state(M, [1, 0, 0]), rt, 0.5)
    a = C.rotate(st, [1, 0, 0], 0.7)
    b = C.rotate_classes(st, [1, 0, 0], np.full(M, 0.7))
    assert np.max(np.abs(a.pack() - b.pack())) < 1e-12
    c = pulse(st, rt, np.array([1.0, 0.0, 0.0]), 0.7, duration=1e-6, n_steps=4, rtol=1e-10)
    assert np.max(np.abs(a.pack() - c.pack())) < 1e-5
    # per-class angles: each Bloch vector is rotated by its own angle
    ang = np.array([0.2, 0.5, 0.9])
    d = C.rotate_classes(st, [0, 0, 1], ang)
    v0, _ = C.to_cartesian(st)
    v1, _ = C.to_cartesian(d)
    for m in range(M):
        R = C.rotation_matrix([0, 0, 1], ang[m])
        assert np.allclose(v1[m], R @ v0[m], atol=1e-12)
