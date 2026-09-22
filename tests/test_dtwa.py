"""The trajectory solver reproduces the exact one-axis-twisting optimum.

This is what licenses using it as an independent check on the cumulant closure
for an inhomogeneous line: before it is trusted where the answer is unknown, it
is checked where the answer is known in closed form.
"""
import numpy as np

from cavsqueeze import dtwa


def kitagawa_ueda(N, mu):
    """Exact Wineland parameter for one-axis twisting of a coherent spin state.

    Kitagawa and Ueda, Phys. Rev. A 47, 5138 (1993), with mu = 2 chi t.
    """
    A = 1.0 - np.cos(mu) ** (N - 2)
    B = 4.0 * np.sin(mu / 2.0) * np.cos(mu / 2.0) ** (N - 2)
    var = 1.0 + 0.25 * (N - 1) * (A - np.hypot(A, B))
    contrast = np.cos(mu / 2.0) ** (N - 1)
    return var / contrast**2


def test_matches_exact_one_axis_twisting():
    """The sampled trajectories reproduce the exact optimum to 0.05 dB."""
    N, chi1 = 200, 1.0
    t = np.linspace(0.0, 0.06, 25)
    out = dtwa.evolve(np.zeros(N), np.ones(N), chi1, t, n_traj=4000, seed=1)
    xi = np.array([dtwa.wineland(o["mean"], o["cov"], N) for o in out])
    exact = np.array([kitagawa_ueda(N, 2 * chi1 * tt) for tt in t])
    got = 10 * np.log10(xi.min())
    want = 10 * np.log10(exact.min())
    assert abs(got - want) < 0.05, (got, want)


def test_step_size_is_converged():
    """A four times smaller step (200 -> 800 steps) changes the answer by
    less than a hundredth of a dB."""
    N, chi1 = 100, 1.0
    t = np.linspace(0.0, 0.08, 12)
    a = dtwa.evolve(np.zeros(N), np.ones(N), chi1, t, n_traj=600, seed=3,
                    steps_per_rad=12.0, min_steps=200)
    b = dtwa.evolve(np.zeros(N), np.ones(N), chi1, t, n_traj=600, seed=3,
                    steps_per_rad=12.0, min_steps=800)
    xa = min(dtwa.wineland(o["mean"], o["cov"], N) for o in a)
    xb = min(dtwa.wineland(o["mean"], o["cov"], N) for o in b)
    assert abs(10 * np.log10(xa / xb)) < 0.01


def test_free_spins_only_dephase():
    """With no interaction the collective spin decays by the line shape alone."""
    N = 400
    rng = np.random.default_rng(0)
    delta = rng.normal(0.0, 1.0, N)
    t = np.array([0.0, 0.5, 1.0, 2.0])
    out = dtwa.evolve(delta, np.ones(N), 0.0, t, n_traj=400, seed=2)
    got = np.array([np.hypot(o["mean"][0], o["mean"][1]) / (N / 2) for o in out])
    want = np.exp(-0.5 * t**2)            # Gaussian line of unit width
    assert np.allclose(got, want, atol=2e-2), (got, want)


def test_mean_spin_follows_the_cumulant_convention():
    """The trajectory variable is <sigma^+> = (x + i y)/2, as in the
    cumulant solver, so a spin detuned by delta > 0 precesses towards +y
    (<J_y> = (1/2) sum_j sin(delta_j t) without interaction). Before
    1.15.1 the read-out flipped the sign of y (and of the xy and yz
    covariances) relative to the cumulant solver."""
    delta = np.array([-1.0, -0.3, 0.4, 1.1])
    t = np.array([0.0, 0.5, 1.0])
    out = dtwa.evolve(delta, np.ones(4), 0.0, t, n_traj=4000, seed=0)
    for tt, o in zip(t, out):
        want = 0.5 * np.array([np.cos(delta * tt).sum(),
                               np.sin(delta * tt).sum(), 0.0])
        assert np.allclose(o["mean"], want, atol=0.08), (tt, o["mean"], want)
    # with the interaction on, the mean spin agrees with the cumulant solver
    from cavsqueeze import cumulant as C
    from cavsqueeze.protocols import css_x
    rt = C.Rates(delta=delta, G=np.ones(4), n=np.ones(4), chi1=0.2,
                 Gd=0.0, Gu=0.0, gamma_phi=0.0)
    rt.spec_delta, rt.spec_n = np.zeros(0), np.zeros(0)
    ref = C.evolve(css_x(4), rt, 1.0, rtol=1e-10)
    J = C.collective_moments(ref, rt.n)[0]
    got = dtwa.evolve(delta, np.ones(4), 0.2, [1.0], n_traj=4000, seed=0)[0]
    assert np.allclose(got["mean"], J, atol=0.08), (got["mean"], J)
