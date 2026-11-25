"""1.21: the trajectory solver's equations with fewer array operations.

The 1.20 derivatives are kept here as the reference.  The new ones are the
same equations rearranged (the sum over k != j written as the full sum minus
the j term), so they must agree to rounding, and a whole seeded run must
give the 1.20 moments to rounding as well.
"""
import numpy as np

from cavsqueeze import dtwa


def _deriv_120(s, z, G, a, chi1):
    """The 1.20 code (the nested `deriv` of `dtwa.evolve`)."""
    coll = (s * G).sum(axis=1)
    other = coll[:, None] - G * s
    ds = 1j * (a * s - chi1 * G * z * other)
    dz = 4.0 * chi1 * G * np.imag(s * np.conj(other))
    return ds, dz


def _evolve_120(delta, G, chi1, t_eval, n_traj, seed, gamma_phi, steps_per_rad=12.0,
                min_steps=400):
    """The 1.20 stepping loop of `dtwa.evolve` with the 1.20 derivatives."""
    delta, G, t_eval = (np.asarray(x, float) for x in (delta, G, t_eval))
    chiN = chi1 * float(np.sum(G ** 2))
    T = float(np.max(t_eval))
    fastest = max(float(np.max(np.abs(delta))), abs(chiN), 1.0)
    n_nominal = int(max(min_steps, np.ceil(steps_per_rad * fastest * T / (2 * np.pi))))
    h_max = T / n_nominal
    rng = np.random.default_rng(seed)
    v = dtwa.sample_css_x(len(delta), n_traj, rng)
    s = 0.5 * (v[:, :, 0] - 1j * v[:, :, 1])
    z = v[:, :, 2].astype(float)
    a = delta + chi1 * G ** 2
    out, t_now = {}, 0.0
    for t_next in np.unique(t_eval):
        span = float(t_next) - t_now
        if span > 0:
            n_seg = max(1, int(np.ceil(span / h_max * (1.0 - 1e-12))))
            dt = span / n_seg
            half_turn = np.sqrt(gamma_phi * dt)
            for _ in range(n_seg):
                if gamma_phi > 0.0:
                    s = s * np.exp(1j * half_turn * rng.standard_normal(s.shape))
                k1s, k1z = _deriv_120(s, z, G, a, chi1)
                k2s, k2z = _deriv_120(s + 0.5 * dt * k1s, z + 0.5 * dt * k1z, G, a, chi1)
                k3s, k3z = _deriv_120(s + 0.5 * dt * k2s, z + 0.5 * dt * k2z, G, a, chi1)
                k4s, k4z = _deriv_120(s + dt * k3s, z + dt * k3z, G, a, chi1)
                s = s + (dt / 6.0) * (k1s + 2 * k2s + 2 * k3s + k4s)
                z = z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
                if gamma_phi > 0.0:
                    s = s * np.exp(1j * half_turn * rng.standard_normal(s.shape))
            t_now = float(t_next)
        out[float(t_next)] = dtwa._moments(2 * np.real(s), 2 * np.imag(s), z)
    return [out[float(tt)] for tt in t_eval]


def test_derivatives_equal_the_120_form():
    """Random sample clouds (not just the starting state), 1, 7 and 50
    spins: agreement to 1e-12 of the largest derivative (measured 2.4e-16)."""
    rng = np.random.default_rng(0)
    for n_spins in (1, 7, 50):
        s = rng.normal(size=(30, n_spins)) + 1j * rng.normal(size=(30, n_spins))
        z = rng.normal(size=(30, n_spins))
        G = rng.uniform(0.5, 1.5, n_spins)
        delta = rng.normal(size=n_spins)
        chi1 = 0.37
        a = delta + chi1 * G ** 2
        new = dtwa._deriv(s, z, G, a, chi1)
        old = _deriv_120(s, z, G, a, chi1)
        scale = max(np.max(np.abs(y)) for y in old)
        for x, y in zip(new, old):
            assert np.max(np.abs(x - y)) < 1e-12 * scale


def test_whole_run_equals_the_120_code():
    """A seeded run with detunings, uneven couplings, twisting and
    dephasing: mean spin and covariance at every time equal those of the
    1.20 loop to 1e-9 of the largest entry (measured 8e-16)."""
    rng = np.random.default_rng(4)
    delta = rng.normal(0.0, 1.0, 6)
    G = rng.uniform(0.7, 1.3, 6)
    t = [0.0, 1.0, 2.5]
    new = dtwa.evolve(delta, G, 0.4, t, n_traj=500, seed=2, gamma_phi=0.1)
    old = _evolve_120(delta, G, 0.4, t, n_traj=500, seed=2, gamma_phi=0.1)
    for x, y in zip(new, old):
        assert np.max(np.abs(x["mean"] - y["mean"])) < 1e-9 * np.max(np.abs(y["mean"]))
        assert np.max(np.abs(x["cov"] - y["cov"])) < 1e-9 * np.max(np.abs(y["cov"]))
