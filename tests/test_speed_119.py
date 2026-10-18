"""1.19 speed-up of the cumulant solver's right-hand side.

The 1.18.0 `cumulant._rhs` is kept below verbatim as the reference.  The
1.19 code forms the same sums in a different order (the sums over the other
classes once per quantity, and the two complex-conjugate halves of the Zc
sources as 2 Re(c X)), so the results agree to rounding, not bit for bit:

* the right-hand side, for random complex states (not only physical ones),
  1, 3 and 16 classes, with and without the correlation feedback and with
  the continuous measurement switched on, to 1e-12 of its largest entry
  (2e-15 or less on the build machine; each entry is a sum of a few tens
  of terms of at most that size, so rounding stays near 1e-14);
* whole solver runs (README examples 1 and 2), against SciPy's integrator
  driven by the 1.18 right-hand side, to 1e-9 of the largest entry of the
  final state (8e-14 measured) and 1e-6 relative in xi2 (1e-9 measured:
  xi2 comes from the small transverse variance, a difference of much larger
  numbers, so it carries a larger relative rounding error than the state).
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import cavsqueeze as cs
from cavsqueeze import cumulant as C


def _rhs_118(t, y, rt, feedback=True, pre=None):
    """cavsqueeze 1.18.0 cumulant._rhs, verbatim (the reference)."""
    M = rt.M
    st = C.State.unpack(y, M)
    s, z, Pc, Qc, Rc, Zc = st.s, st.z, st.Pc, st.Qc, st.Rc, st.Zc
    G, n = rt.G, rt.n
    Gd, Gu = rt.Gd, rt.Gu
    k = C._rhs_constants(rt) if pre is None else pre
    c, cc, Ac, W, Gm, Gn = k["c"], k["cc"], k["Ac"], k["W"], k["Gm"], k["Gn"]
    sc = np.conj(s)
    Rcc = np.conj(Rc)

    # products used several times (each evaluated once since 1.18)
    Ws = W @ s
    Wsc = np.conj(Ws)
    PcW, WPc = Pc @ W, W @ Pc
    QcW, WQc = Qc @ W, W @ Qc
    RcW, RccW = Rc @ W, Rcc @ W
    dP, dQ, dR, dRcc = np.diag(Pc), np.diag(Qc), np.diag(Rc), np.diag(Rcc)

    # ---------------- first moments ----------------
    # sum_{k != a} G_k <sigma z_a sigma+_k> = sum_p G_p (n_p - d_pm)(Rc_mp + z_m s_p)
    fb = 1.0 if feedback else 0.0
    S_zp = fb * (RcW - G * dR) + z * (Ws - G * s)
    ds = -Ac * s + cc * G * S_zp
    # sum_{k != a} G_k <sigma+_a sigma-_k> = sum_p G_p (n_p - d_pm)(Pc_mp + s_m s*_p)
    S_pm = fb * (PcW - G * dP) + s * np.conj(Ws - G * s)
    dz = -4.0 * G * np.real(c * S_pm) - k["G2"] * ((Gd - Gu) + (Gd + Gu) * z)

    # ---------------- helpers ----------------
    zm, zn = z[:, None], z[None, :]
    sm, sn = s[:, None], s[None, :]
    scm, scn = sc[:, None], sc[None, :]

    def wsum(full, at_m, at_n):
        """sum_p G_p (n_p - d_pm - d_pn) T(p) from the full sum and the p = m, p = n values."""
        return full - Gm * at_m - Gn * at_n

    # ---------------- Pc ----------------
    # (iii) from d sigma+_a:  sum_p W_p [ z_m Pc_pn + s_p Rc*_mn ]
    S1 = wsum(zm * WPc[None, :] + Rcc * Ws,
              zm * Pc + Rcc * sm,           # p = m: z_m Pc_mn + s_m Rc*_mn
              zm * dP[None, :] + Rcc * sn)  # p = n: z_m Pc_nn + s_n Rc*_mn
    # (iii) from d sigma-_b:  sum_p W_p [ z_n Pc_mp + s*_p Rc_nm ]
    S2 = wsum(zn * PcW[:, None] + Rc.T * Wsc,
              zn * dP[:, None] + Rc.T * scm,
              zn * Pc + Rc.T * scn)
    src_a = 0.5 * (zm + zm * zn + Zc) - Rc * scn - zm * (sn * scn)
    src_b = 0.5 * (zn + zm * zn + Zc) - sm * np.conj(Rc.T) - zn * (sm * scm)
    dPc = (
        k["negAc_A"] * Pc
        + k["cc_Gm"] * (Gn * src_a + S1)
        + k["c_Gn"] * (Gm * src_b + S2)
        + k["Gu_GG"] * (Zc + zm * zn)
    )

    # ---------------- Qc ----------------
    S3 = wsum(zm * WQc[None, :] + Rc * Ws,
              zm * Qc + Rc * sm,
              zm * dQ[None, :] + Rc * sn)
    S4 = wsum(zn * QcW[:, None] + Rc.T * Ws,
              zn * dQ[:, None] + Rc.T * sm,
              zn * Qc + Rc.T * sn)
    dQc = (
        k["negAc_Ac"] * Qc
        + k["cc_Gm"] * (k["neg_Gn"] * (Rc + zm * sn) * sn + S3)
        + k["cc_Gn"] * (k["neg_Gm"] * sm * (Rc.T + zn * sm) + S4)
    )

    # ---------------- Rc ----------------
    # (iii) from d sigma z_a: sum_p W_p [ c (s_m Pc_np + s*_p Qc_mn) + c* (s_p Pc_nm + s*_m Qc_pn) ]
    S5 = wsum(c * (sm * PcW[None, :] + Qc * Wsc) + cc * (Pc.T * Ws + scm * WQc[None, :]),
              c * (sm * Pc.T + Qc * scm) + cc * (Pc.T * sm + scm * Qc),
              c * (sm * dP[None, :] + Qc * scn) + cc * (Pc.T * sn + scm * dQ[None, :]))
    # (iii) from d sigma+_b: sum_p W_p [ z_n Rc_mp + s_p Zc_mn ]
    S7 = wsum(zn * RcW[:, None] + Zc * Ws,
              zn * dR[:, None] + Zc * sm,
              zn * Rc + Zc * sn)
    Pmn_raw = Pc + sm * scn
    Pnm_raw = Pc.T + sn * scm
    src_za = c * (0.5 * (sm - Rc.T - sm * zn) - Pmn_raw * sn) + cc * (-Pnm_raw * sn)
    dRc = (
        k["decay_R"] * Rc
        - k["two_GG"] * src_za
        - k["two_Gm"] * S5
        - k["Ac"][None, :] * Rc
        + k["cc_GG"] * (1.0 - zm) * (Rc.T + sm * zn)
        + k["cc_Gn"] * S7
        - k["two_Gd_GG"] * (Rc.T + sm * zn)
    )

    # ---------------- Zc ----------------
    # (iii) from d sigma z_a: sum_p W_p [ c (s_m Rc*_np + s*_p Rc_nm) + c* (s_p Rc*_nm + s*_m Rc_np) ]
    S8 = wsum(c * (sm * RccW[None, :] + Rc.T * Wsc) + cc * (Rcc.T * Ws + scm * RcW[None, :]),
              c * (sm * Rcc.T + Rc.T * scm) + cc * (Rcc.T * sm + scm * Rc.T),
              c * (sm * dRcc[None, :] + Rc.T * scn) + cc * (Rcc.T * sn + scm * dR[None, :]))
    # (iii) from d sigma z_b: sum_p W_p [ c (s_n Rc*_mp + s*_p Rc_mn) + c* (s_p Rc*_mn + s*_n Rc_mp) ]
    S9 = wsum(c * (sn * RccW[:, None] + Rc * Wsc) + cc * (Rcc * Ws + scn * RcW[:, None]),
              c * (sn * dRcc[:, None] + Rc * scm) + cc * (Rcc * sm + scn * dR[:, None]),
              c * (sn * Rcc + Rc * scn) + cc * (Rcc * sn + scn * Rc))
    src_zz_a = c * Pmn_raw * (1.0 - zn) - cc * Pnm_raw * (1.0 + zn)
    src_zz_b = -c * Pnm_raw * (1.0 + zm) + cc * Pmn_raw * (1.0 - zm)
    dZc = (
        k["decay_Z"] * Zc
        - k["two_GG"] * (src_zz_a + src_zz_b)
        - k["two_Gm"] * S8
        - k["two_Gn"] * S9
        + k["four_GG"] * (Gd * Pmn_raw + Gu * Pnm_raw)
    )

    if getattr(rt, "meas", 0.0):
        dPc, dQc, dRc, dZc = C._add_measurement(rt.meas, getattr(rt, "meas_eta", 1.0), n, st, dPc, dQc, dRc, dZc)
    return C.State(ds, dz, dPc, dQc, dRc, dZc).pack()




def _random_rates(M, rng, meas=0.0):
    rt = C.Rates(delta=rng.normal(size=M), G=rng.uniform(0.5, 1.5, M),
                 n=rng.uniform(1.0, 100.0, M), chi1=0.3, Gd=0.2, Gu=0.05,
                 gamma_phi=0.1)
    rt.meas = meas
    return rt


@pytest.mark.parametrize("M", [1, 3, 16])
@pytest.mark.parametrize("feedback", [True, False])
@pytest.mark.parametrize("meas", [0.0, 0.7])
def test_rhs_agrees_with_the_118_code(M, feedback, meas):
    rng = np.random.default_rng(100 + M)
    rt = _random_rates(M, rng, meas)
    pre = C._rhs_constants(rt)
    for _ in range(5):
        y = rng.normal(size=2 * M + 4 * M * M) + 1j * rng.normal(size=2 * M + 4 * M * M)
        ref = _rhs_118(0.0, y, rt, feedback)
        scale = np.max(np.abs(ref))
        for got in (C._rhs(0.0, y, rt, feedback), C._rhs(0.0, y, rt, feedback, pre)):
            assert got.dtype == ref.dtype and got.shape == ref.shape
            assert np.max(np.abs(got - ref)) <= 1e-12 * scale


@pytest.mark.parametrize("N, M, t", [(1e10, 1, 1e-3), (1e15, 16, 2e-5)])
def test_solver_run_agrees_with_the_118_right_hand_side(N, M, t):
    p = cs.from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=0.15)
    ens = cs.homogeneous(N) if M == 1 else cs.equal_probability_classes(
        cs.lineshape("gaussian", 2 * np.pi * 2e3), M, N)
    rt = C.Rates.from_params(p, ens)
    st = C.product_state(rt.M, [1.0, 0.0, 0.0])
    new = C.evolve(st, rt, t)
    ref = solve_ivp(_rhs_118, (0.0, t), st.pack(), args=(rt, True), rtol=1e-8,
                    atol=1e-10 / N, method="DOP853").y[:, -1]
    assert np.max(np.abs(new.pack() - ref)) <= 1e-9 * np.max(np.abs(ref))
    x_new = C.wineland_xi2(new, rt.n)[0]
    x_ref = C.wineland_xi2(C.State.unpack(ref, rt.M), rt.n)[0]
    assert abs(x_new / x_ref - 1.0) < 1e-6
