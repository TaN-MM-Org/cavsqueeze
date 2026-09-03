"""Holstein-Primakoff export: exact identities and a full round trip
through QuTiP, including the phase-convention lock."""
import numpy as np
import pytest

from cavsqueeze import from_hz, homogeneous
from cavsqueeze.cumulant import Rates
from cavsqueeze.interop import bosonic_mode, covariance_of_qutip_state, to_qutip
from cavsqueeze.metrology import squeezing_parameters
from cavsqueeze.protocols import css_x, twist


def _squeezed(N=1e8, t=2e-5):
    p = from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6,
                T=0.02, T2=0.15)
    rt = Rates.from_params(p, homogeneous(N))
    return twist(css_x(rt.M), rt, t), rt.n


def test_css_maps_to_vacuum():
    ens = homogeneous(1e6)
    m = bosonic_mode(css_x(1), ens.n)
    assert np.allclose(m["Sigma"], 0.5 * np.eye(2))
    assert np.isclose(m["nu"], 1.0)
    assert np.isclose(m["r"], 0.0, atol=1e-8)
    assert np.isclose(m["n_th"], 0.0)
    assert np.isclose(m["purity"], 1.0)
    assert np.isclose(m["squeezing_db"], 0.0, atol=1e-8)


def test_decomposition_reconstructs_covariance():
    st, n = _squeezed()
    m = bosonic_mode(st, n)
    assert m["nu"] >= 1.0
    c, s = np.cos(m["theta"]), np.sin(m["theta"])
    R = np.array([[c, -s], [s, c]])
    D = np.diag([m["nu"] / 2 * np.exp(-2 * m["r"]),
                 m["nu"] / 2 * np.exp(2 * m["r"])])
    assert np.allclose(R @ D @ R.T, m["Sigma"], atol=1e-10)
    # extremal variances are the eigenvalues, and det gives nu back
    assert np.isclose(m["var_min"] * m["var_max"], (m["nu"] / 2) ** 2)


def test_mode_squeezing_agrees_with_metrology():
    """The mode's var_min equals the metrology module's transverse minimum
    in the same normalization: 2 var_min = xi_S^2."""
    st, n = _squeezed()
    m = bosonic_mode(st, n)
    sq = squeezing_parameters(st, n)
    assert np.isclose(2.0 * m["var_min"], sq["xi2_S"], rtol=1e-10)
    assert m["squeezing_db"] < 0.0            # genuinely squeezed


def test_qutip_convention_lock():
    """qutip.squeeze(N, r e^{i phi}) squeezes the quadrature at angle
    phi/2; the exporter relies on exactly this."""
    qutip = pytest.importorskip("qutip")
    N, r, phz = 60, 0.7, 0.8
    S = qutip.squeeze(N, r * np.exp(1j * phz))
    v = qutip.ket2dm(S * qutip.basis(N, 0))
    a = qutip.destroy(N)

    def var(phi):
        x = (a * np.exp(-1j * phi) + a.dag() * np.exp(1j * phi)) / np.sqrt(2)
        return float(qutip.expect(x * x, v).real - qutip.expect(x, v).real ** 2)

    assert np.isclose(var(phz / 2), np.exp(-2 * r) / 2, rtol=1e-6)
    assert np.isclose(var(phz / 2 + np.pi / 2), np.exp(2 * r) / 2, rtol=1e-6)


def test_qutip_round_trip():
    """The exported density matrix reproduces the solver covariance at
    every quadrature angle, and its purity equals 1/nu."""
    pytest.importorskip("qutip")
    st, n = _squeezed()
    rho, mode = to_qutip(st, n)
    assert np.isclose(float(rho.tr().real), 1.0, atol=1e-9)
    phis, vs = covariance_of_qutip_state(rho, phis=np.linspace(0, np.pi, 25))
    for ph, v in zip(phis, vs):
        u = np.array([np.cos(ph), np.sin(ph)])
        assert np.isclose(v, u @ mode["Sigma"] @ u, rtol=1e-4)
    assert np.isclose(float((rho * rho).tr().real), mode["purity"], rtol=1e-6)


def test_truncation_check_raises():
    """A deliberately starved Fock cutoff must be caught, not returned."""
    pytest.importorskip("qutip")
    st, n = _squeezed(t=5e-5)                  # strong squeezing, r ~ 2.3
    with pytest.raises(ValueError):
        to_qutip(st, n, dim=60)
