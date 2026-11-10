"""1.20: single-spin dephasing in the trajectory solver (`dtwa.evolve(...,
gamma_phi=...)`), checked against a closed form, the exact master equation
for 4 spins and the exact PIQS solution for 40 spins.

Single-spin dephasing L_j = sqrt(gamma_phi/2) sigma z_j is the average over
random turns of each spin about z with phase variance 2 gamma_phi t, which
each trajectory applies (see the module docstring of `cavsqueeze.dtwa`)."""
import numpy as np
import pytest

from cavsqueeze import dtwa


def test_free_spins_with_dephasing_follow_the_closed_form():
    """Without interaction the random turns commute with the detunings, and
    the trajectories are exact in distribution: <sigma x_j> + i <sigma y_j> =
    exp(-gamma_phi t) exp(i delta_j t), and each spin-1/2 has <sigma^2> = 1,
    so Var(J_x) = (1/4) sum_j (1 - <sigma x_j>^2), Var(J_y) likewise,
    Cov(J_x, J_y) = -(1/4) sum_j <sigma x_j> <sigma y_j> and Var(J_z) = N/4.
    The means agree within 4 standard errors of the sample mean, the
    variances within 0.05 (about 4 standard errors at 20000 trajectories)."""
    delta = np.linspace(-1.0, 1.0, 6)
    g, n_traj = 0.5, 20000
    t = np.array([0.0, 0.5, 1.0, 2.0])
    # without interaction the step only has to resolve the precession (RK4)
    out = dtwa.evolve(delta, np.ones(6), 0.0, t, n_traj=n_traj, seed=1, gamma_phi=g,
                      min_steps=50)
    for tt, o in zip(t, out):
        sx = np.exp(-g * tt) * np.cos(delta * tt)
        sy = np.exp(-g * tt) * np.sin(delta * tt)
        mean = 0.5 * np.array([sx.sum(), sy.sum(), 0.0])
        se = np.sqrt(np.diag(o["cov"]) / n_traj)
        assert np.all(np.abs(o["mean"] - mean) < 4 * se + 1e-12), (tt, o["mean"], mean)
        cov = np.array([[np.sum(1 - sx ** 2), -np.sum(sx * sy), 0.0],
                        [-np.sum(sx * sy), np.sum(1 - sy ** 2), 0.0],
                        [0.0, 0.0, 6.0]]) / 4
        assert np.max(np.abs(o["cov"] - cov)) < 0.05, (tt, o["cov"], cov)


def test_four_spins_with_dephasing_match_the_master_equation():
    """Four spins with random detunings, twisting and dephasing: the mean
    spin agrees with the exact master-equation solution (`full_hilbert`) to
    0.08 in spin units, the tolerance of the existing 4-spin test (seeds 1
    to 8 give 0.018 to 0.042).  Leaving the dephasing out of the trajectories
    misses by 0.29 to 0.32, so the test is sensitive to it."""
    pytest.importorskip("qutip")
    from cavsqueeze.cumulant_raw import Rates
    from cavsqueeze.exact import full_hilbert

    rng = np.random.default_rng(3)
    N, chi1, g = 4, 0.3, 0.3
    delta = rng.normal(size=N) * 0.5
    t = np.linspace(0.0, 2.0, 5)
    rt = Rates(delta=delta, G=np.ones(N), n=np.ones(N), chi1=chi1, Gd=0.0, Gu=0.0,
               gamma_phi=g)
    ex = full_hilbert(rt, [1.0, 0.0, 0.0], t)
    J_ex = 0.5 * np.stack([(2 * ex["s"].real).sum(1), (2 * ex["s"].imag).sum(1),
                           ex["z"].sum(1)], axis=1)
    got = np.array([o["mean"] for o in dtwa.evolve(delta, np.ones(N), chi1, t, n_traj=4000,
                                                   seed=1, gamma_phi=g)])
    assert np.max(np.abs(got - J_ex)) < 0.08
    none = np.array([o["mean"] for o in dtwa.evolve(delta, np.ones(N), chi1, t, n_traj=4000,
                                                    seed=1)])
    assert np.max(np.abs(none - J_ex)) > 0.2


def test_forty_spins_with_dephasing_match_piqs():
    """40 identical spins, twisting with strong dephasing (gamma_phi t = 0.3
    at the end, chi N t up to 1.5): the Wineland parameter agrees with the
    exact PIQS solution to 0.35 dB at every time and <J_x> to 4 standard
    errors of the sample mean.  4000 trajectories; seeds 1 to 8 give 0.06 to
    0.27 dB and at most 1.9 standard errors.  (With the PIQS reference of
    1.19, which had half this dephasing, the difference is about 2 dB.)"""
    pytest.importorskip("qutip")
    from cavsqueeze.exact import dicke_piqs, xi2_from_moments

    N, chi, g, n_traj = 40, 1.0, 8.0, 4000
    t = np.linspace(0.0, 1.5 / N, 7)
    ref = dicke_piqs(N, chi, 0.0, 0.0, g, theta=np.pi / 2, times=t)
    out = dtwa.evolve(np.zeros(N), np.ones(N), chi, t, n_traj=n_traj, seed=1, gamma_phi=g)
    for k in range(1, len(t)):
        o = out[k]
        xi_d = dtwa.wineland(o["mean"], o["cov"], N)
        xi_e = xi2_from_moments(ref["J"][k], ref["Cov"][k], N)
        assert abs(10 * np.log10(xi_d / xi_e)) < 0.35, (k, xi_d, xi_e)
        se = np.sqrt(o["cov"][0, 0] / n_traj)
        assert abs(o["mean"][0] - ref["J"][k][0]) < 4 * se, (k, o["mean"][0], ref["J"][k][0])


def test_dephasing_rate_is_checked_and_zero_is_the_old_solver():
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="gamma_phi"):
            dtwa.evolve(np.zeros(2), np.ones(2), 0.1, [0.5], gamma_phi=bad)
    # gamma_phi = 0 draws no extra random numbers: the same samples and steps
    a = dtwa.evolve(np.zeros(5), np.ones(5), 0.3, [0.2, 0.4], n_traj=50, seed=7)
    b = dtwa.evolve(np.zeros(5), np.ones(5), 0.3, [0.2, 0.4], n_traj=50, seed=7,
                    gamma_phi=0.0)
    for x, y in zip(a, b):
        assert np.array_equal(x["mean"], y["mean"]) and np.array_equal(x["cov"], y["cov"])
