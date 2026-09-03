"""Validation of the cumulant equations against exact master-equation solutions.

Test 1: N = 2 distinguishable spins.  Every operator on two spins is at most a
        pair moment, so the second-order hierarchy is closed and must agree with
        the exact solution to integration accuracy.  This checks the algebra of
        every term (Hamiltonian, both collective jumps, dephasing, ordering).
Test 2: N = 4 spins with random detunings and couplings; the truncation error
        must be small at short times and grow smoothly.
Test 3: Homogeneous N = 40 versus PIQS: squeezing parameter from the cumulant
        solver agrees with the exact Dicke-basis solution in the Gaussian regime.
Test 4: Rotations are consistent with exact rotations of a product state and
        preserve the coherent-state squeezing parameter xi^2 = 1.
"""
import numpy as np
import pytest

from cavsqueeze.cumulant_raw import Rates, State, product_state, evolve, rotate, wineland_xi2, collective_moments
from cavsqueeze.exact import full_hilbert, dicke_piqs, xi2_from_moments


def _compare(ex, st, k, atol):
    N = st.M
    assert np.allclose(ex["s"][k], st.s, atol=atol)
    assert np.allclose(ex["z"][k], st.z.real, atol=atol)
    mask = ~np.eye(N, dtype=bool)
    for key, arr in (("P", st.P), ("Q", st.Q), ("R", st.R), ("Z", st.Z)):
        err = np.max(np.abs(ex[key][k][mask] - arr[mask]))
        assert err < atol, (key, err)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_spins_closed_hierarchy(seed):
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=2) * 1.0
    G = rng.uniform(0.6, 1.4, size=2)
    rt = Rates(delta=delta, G=G, n=np.ones(2), chi1=0.7, Gd=0.35 * 1.4, Gu=0.35 * 0.4, gamma_phi=0.1)
    times = np.linspace(0, 3.0, 7)
    v = np.array([np.sin(1.1) * np.cos(0.3), np.sin(1.1) * np.sin(0.3), np.cos(1.1)])
    ex = full_hilbert(rt, v, times)
    st0 = product_state(2, v)
    sts = evolve(st0, rt, times[-1], t_eval=times, rtol=1e-10, atol=1e-12)
    for k in range(len(times)):
        _compare(ex, sts[k], k, atol=2e-6)
    # Z must stay real, P Hermitian, Q symmetric
    st = sts[-1]
    assert np.max(np.abs(st.Z.imag)) < 1e-8
    assert np.allclose(st.P, st.P.conj().T, atol=1e-8)
    assert np.allclose(st.Q, st.Q.T, atol=1e-8)


def test_four_spins_truncation_error_small():
    rng = np.random.default_rng(3)
    N = 4
    delta = rng.normal(size=N) * 0.5
    G = rng.uniform(0.7, 1.3, size=N)
    rt = Rates(delta=delta, G=G, n=np.ones(N), chi1=0.3, Gd=0.05, Gu=0.01, gamma_phi=0.02)
    times = np.linspace(0, 1.0, 6)
    v = np.array([1.0, 0.0, 0.0])
    ex = full_hilbert(rt, v, times)
    sts = evolve(product_state(N, v), rt, times[-1], t_eval=times)
    # short time: truncation error is third order in t
    _compare(ex, sts[1], 1, atol=2e-3)
    # errors at t = 1 stay moderate (Gaussian truncation for 4 spins)
    _compare(ex, sts[-1], -1, atol=0.15)


def test_homogeneous_vs_piqs_squeezing():
    N = 40
    chi = 1.0
    Gd, Gu, gphi = 0.6, 0.15, 0.05
    times = np.linspace(0, 1.5 / N, 7)  # Q = chi N t up to 1.5
    ref = dicke_piqs(N, chi, Gd, Gu, gphi, theta=np.pi / 2, times=times)
    rt = Rates(delta=np.zeros(1), G=np.ones(1), n=np.array([N], float), chi1=chi, Gd=Gd, Gu=Gu, gamma_phi=gphi)
    sts = evolve(product_state(1, [1.0, 0.0, 0.0]), rt, times[-1], t_eval=times)
    for k, st in enumerate(sts):
        J, Cov, _, _ = collective_moments(st, rt.n)
        assert np.allclose(J, ref["J"][k], atol=0.05 * N / 2)
        xi_c = wineland_xi2(st, rt.n)[0]
        xi_e = xi2_from_moments(ref["J"][k], ref["Cov"][k], N)
        assert abs(10 * np.log10(xi_c) - 10 * np.log10(xi_e)) < 0.2, (k, xi_c, xi_e)


def test_rotation_consistency():
    st = product_state(3, [0.0, 0.0, -1.0])
    n = np.array([5.0, 7.0, 9.0])
    assert abs(wineland_xi2(rotate(st, [0, 1, 0], np.pi / 2), n)[0] - 1.0) < 1e-12
    st2 = rotate(st, [0, 1, 0], np.pi / 2)
    v, C = None, None
    from cavsqueeze.cumulant_raw import to_cartesian

    v, C = to_cartesian(st2)
    assert np.allclose(np.abs(v[:, 0]), 1.0)
    assert np.allclose(C[:, :, 0, 0], 1.0)
    # rotating back and forth is the identity
    st3 = rotate(rotate(st2, [1, 0, 0], 0.7), [1, 0, 0], -0.7)
    for a, b in ((st2.s, st3.s), (st2.P, st3.P), (st2.Q, st3.Q), (st2.R, st3.R), (st2.Z, st3.Z)):
        assert np.allclose(a, b, atol=1e-12)
