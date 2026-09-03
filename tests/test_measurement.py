"""Measurement route (scripts/run_measurement.py): the emission noise of a
coherent ensemble, the product-state <J+J-> and the closed-form steady state.

The unconditional growth of Var(J_z) under the collective emission jump
sqrt(Gamma) J_- is checked against an exact master-equation solution for a
small symmetric ensemble, and the Riccati steady state against
S_ss = 4 g sqrt(eta n_bar) / kappa."""
import os
import sys

import numpy as np
import pytest

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
needs_paper_scripts = pytest.mark.skipif(
    not os.path.exists(os.path.join(_SCRIPTS, "run_measurement.py")),
    reason="paper-companion scripts not present in this repository",
)


def test_emission_noise_matches_exact_master_equation():
    qutip = __import__("qutip")
    N, G = 6, 0.3
    Jm = qutip.jmat(N / 2, "-")
    Jz = qutip.jmat(N / 2, "z")
    Jp = Jm.dag()
    psi0 = qutip.spin_coherent(N / 2, np.pi / 2, 0.0)  # coherent spin state on the equator
    dt = 1e-3
    res = qutip.mesolve(0 * Jz, psi0, [0, dt], c_ops=[np.sqrt(G) * Jm], e_ops=[Jz, Jz * Jz, Jp * Jm, Jp * Jm * Jz])
    var0 = res.expect[1][0] - res.expect[0][0] ** 2
    var1 = res.expect[1][1] - res.expect[0][1] ** 2
    rate_exact = (var1 - var0) / dt
    jpjm, jz = res.expect[2][0], res.expect[0][0]
    cov = res.expect[3][0] - jpjm * jz
    rate_formula = G * (jpjm - 2 * cov)  # exact leading-order expression (Sec. S7)
    assert abs(rate_exact / rate_formula - 1) < 2e-3
    # product-state value <J+J-> = N^2/4 + N/4 for the equatorial coherent state
    assert abs(jpjm - (N**2 / 4 + N / 4)) < 1e-9


@needs_paper_scripts
def test_product_state_jpjm():
    from run_measurement import meanfield_trajectory
    from common import from_hz
    N = 1e6
    p = from_hz(1e6 / np.sqrt(N), 1e4, 3e7, T=0.02, T2=0.15)
    tr = meanfield_trajectory(p, N, np.array([0.0, 1e-9]), M=50)
    assert abs(tr["JpJm"][0] - (N**2 / 4 + N / 4)) < 1e-6 * N**2


@needs_paper_scripts
def test_riccati_steady_state_closed_form():
    from run_measurement import gamma_m
    from common import from_hz
    from scipy.integrate import solve_ivp
    N, eta, nb = 1e10, 0.5, 1e9
    p = from_hz(1e6 / np.sqrt(N), 1e4, 3e7, T=0.02, T2=0.15)
    Gm = gamma_m(p, nb, eta)
    D = p.Gamma_SR * N**2 / 4
    sol = solve_ivp(lambda t, y: [-Gm * y[0] ** 2 + D], (0, 0.05), [N / 4], rtol=1e-10, atol=1e-6, method="LSODA")
    S_num = (N / 4) / sol.y[0, -1]
    S_formula = 4 * p.g * np.sqrt(eta * nb) / p.kappa
    assert abs(S_num / S_formula - 1) < 1e-3


def test_conditional_cumulant_solver_pure_measurement():
    """With the measurement term switched on and no dynamics, the class-resolved
    solver must reproduce Var(J_z) = (N/4)/(1 + Gamma_m N t/4) (Riccati, up to
    (N-1)/N) and keep Var(J_y) Var(J_z) at the minimum-uncertainty value for
    unit detection efficiency."""
    from cavsqueeze import cumulant as C
    from cavsqueeze.protocols import css_x
    M, N = 3, 1e6
    rt = C.Rates(delta=np.zeros(M), G=np.ones(M), n=np.full(M, N / M), chi1=0.0, Gd=0.0, Gu=0.0, gamma_phi=0.0)
    rt.spec_delta, rt.spec_n = np.zeros(0), np.zeros(0)
    rt.meas, rt.meas_eta = 1e-6, 1.0
    for t in [0.1, 1.0, 10.0]:
        s = C.evolve(css_x(M), rt, t, rtol=1e-9)
        J, Cov, S1, S2 = C.collective_moments(s, rt.n)
        assert abs(Cov[2, 2] / ((N / 4) / (1 + rt.meas * N / 4 * t)) - 1) < 1e-5
        assert abs(Cov[1, 1] * Cov[2, 2] / (N / 4) ** 2 - 1) < 1e-6
