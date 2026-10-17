"""1.18: the readout protocols, which no test covered before, against the
linearized (Holstein-Primakoff) one-axis-twisting closed forms.

For N spins along +x and pure one-axis twisting of total strength
Q = chi N t (no decay), the transverse fluctuations (J_y, J_z) / sqrt(N/4)
are sheared, y -> y + Q z.  Then
  * the squeezed variance is the smallest eigenvalue of [[1 + Q^2, Q], [Q, 1]],
    xi2 = 1 + Q^2/2 - Q sqrt(1 + Q^2/4);
  * twisting, a small rotation phi about y, and untwisting (Q -> -Q) moves
    the mean spin by (N/2) phi (Q, 1) while the noise returns to N/4, so the
    gain over the standard quantum limit is (1 + Q^2) / (1 + 4 sigma_det^2 / N)
    (Davis, Bentsen and Schleier-Smith, PRL 116, 053601 (2016) describe
    the protocol).
Corrections are of relative order Q^2 / N; the test uses N = 1e6.
"""
import numpy as np
import pytest

import cavsqueeze as cs
from cavsqueeze.cumulant import Rates

N = 1e6
P = cs.from_hz(g_hz=1e3 / np.sqrt(N), kappa_hz=0.0, Delta_hz=1e6, T=0.0)   # no decay
ENS = cs.homogeneous(N)
CHI_N = Rates.from_params(P, ENS).chiN


@pytest.mark.parametrize("Q", [0.5, 2.0])
def test_twist_untwist_gain_is_one_plus_Q_squared(Q):
    sig = 100.0
    r = cs.twist_untwist(P, ENS, Q / CHI_N, sigma_det=[0.0, sig])
    assert r["gain"][0] == pytest.approx(1 + Q ** 2, rel=2e-5)
    assert r["gain"][1] == pytest.approx((1 + Q ** 2) / (1 + 4 * sig ** 2 / N), rel=2e-5)
    assert r["amplification"] == pytest.approx(np.sqrt(1 + Q ** 2), rel=2e-5)
    assert r["var_ratio"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("Q", [0.5, 2.0])
def test_plain_squeezed_readout_against_the_shear(Q):
    sig = 100.0
    xi2 = 1 + Q ** 2 / 2 - Q * np.sqrt(1 + Q ** 2 / 4)
    t = Q / CHI_N
    r = cs.plain_squeezed_readout(P, ENS, t, sigma_det=[0.0, sig])
    sa = cs.squeezing_after(P, ENS, t)
    assert r["gain"][0] == pytest.approx(1 / xi2, rel=2e-5)
    # the documented form 1 / (xi2 + 4 sigma_det^2 / (N C^2))
    assert r["gain"][1] == pytest.approx(
        1 / (sa["xi2"] + 4 * sig ** 2 / (N * sa["contrast"] ** 2)), rel=1e-12)


def test_ideal_imperfect_echo_is_the_plain_echo():
    """twist_imperfect with an instantaneous, exact pi pulse is twist."""
    rt = Rates.from_params(P, ENS)
    st0 = cs.css_x(1)
    a = cs.twist(st0, rt, 1.0 / CHI_N)
    b = cs.twist_imperfect(st0, rt, 1.0 / CHI_N, pi_duration=0.0)
    assert np.array_equal(a.pack(), b.pack())
