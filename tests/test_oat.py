"""v1.10 anchor: the Kitagawa-Ueda one-axis-twisting closed forms
against brute-force exact unitary evolution -- every returned quantity
(mean spin, extremal transverse variances, optimal angle, both
squeezing parameters), for even and odd N and random twisting angles."""
import numpy as np
import pytest

qt = pytest.importorskip("qutip")

from cavsqueeze.metrology import metrological_gain_db, oat_closed_form


def _ops(N, k, op):
    return qt.tensor(*[op if j == k else qt.qeye(2) for j in range(N)])


def _collective(N, op):
    return sum(_ops(N, k, op) / 2 for k in range(N))


@pytest.mark.parametrize("N", [2, 5, 8])
def test_oat_closed_form_matches_exact_unitary_evolution(N):
    Jx = _collective(N, qt.sigmax())
    Jy = _collective(N, qt.sigmay())
    Jz = _collective(N, qt.sigmaz())
    css = qt.tensor(*[(qt.basis(2, 0) + qt.basis(2, 1)).unit()
                      for _ in range(N)])
    rng = np.random.default_rng(N)
    for mu in rng.uniform(0.05, 2.0, 2):
        psi = (-1j * (mu / 2.0) * Jz ** 2).expm() * css
        cf = oat_closed_form(N, mu)
        assert abs(qt.expect(Jx, psi) - cf["Jx"]) < 1e-12
        # variance at the predicted optimal angle equals predicted Vmin
        a = cf["alpha_min"]
        Ja = np.cos(a) * Jy + np.sin(a) * Jz
        V = qt.expect(Ja * Ja, psi) - qt.expect(Ja, psi) ** 2
        assert abs(V - cf["Vmin"]) < 1e-12
        # and it is the global minimum over the transverse circle,
        # with Vmax the global maximum (grid-resolution tolerance)
        grid = np.linspace(0.0, np.pi, 2001)
        vs = np.array([qt.expect((np.cos(g) * Jy + np.sin(g) * Jz) ** 2,
                                 psi)
                       - qt.expect(np.cos(g) * Jy + np.sin(g) * Jz,
                                   psi) ** 2 for g in grid])
        assert abs(vs.min() - cf["Vmin"]) < 1e-4
        assert abs(vs.max() - cf["Vmax"]) < 1e-4
        # Jz is conserved: <Jz^2> stays N/4 exactly
        assert abs(qt.expect(Jz * Jz, psi) - N / 4.0) < 1e-12


def test_oat_squeezing_parameters_and_limits():
    """mu = 0 is the coherent state exactly (xi2 = 1, V = N/4); the
    Wineland parameter carries the contrast penalty."""
    cf0 = oat_closed_form(6, 0.0)
    assert cf0["xi2_S"] == 1.0
    assert cf0["Vmin"] == cf0["Vmax"] == 1.5
    assert cf0["Jx"] == 3.0
    cf = oat_closed_form(40, 0.2)
    assert cf["xi2_S"] < 1.0                    # squeezed
    assert cf["xi2_R"] > cf["xi2_S"]            # contrast penalty
    assert cf["xi2_R"] == pytest.approx(
        cf["xi2_S"] * (20.0) ** 2 / cf["Jx"] ** 2, rel=1e-14)
    assert metrological_gain_db(cf["xi2_R"]) > 0.0


def test_oat_refuses_fewer_than_two_spins():
    with pytest.raises(ValueError):
        oat_closed_form(1, 0.3)
