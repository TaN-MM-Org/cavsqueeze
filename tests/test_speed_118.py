"""1.18 speed-ups: each new code path is checked against the 1.17.0 code,
kept here verbatim as the reference.  The results must be bit-identical."""
import numpy as np
import pytest

import cavsqueeze as cs
from cavsqueeze import cumulant as C


# ---------------------------------------------------------------------------
# Dick coefficients: 1.17.0 looped over the harmonics
# ---------------------------------------------------------------------------

def _seg_E_117(k, t1, t2):
    if abs(k) < 1e-300:
        return complex(t2 - t1)
    return (np.exp(1j * k * t2) - np.exp(1j * k * t1)) / (1j * k)


def _seg_sin_117(a, phi, w, t1, t2):
    return (np.exp(1j * phi) * _seg_E_117(a - w, t1, t2)
            - np.exp(-1j * phi) * _seg_E_117(-a - w, t1, t2)) / 2j


def _dick_coefficients_117(m_max, T, Tc, tp):
    a = np.pi / (2 * tp) if tp > 0 else 0.0
    area = T + (2 * tp) * (2 / np.pi) if tp > 0 else T
    g0 = area / Tc
    gm = np.empty(m_max, dtype=complex)
    for m in range(1, m_max + 1):
        w = 2 * np.pi * m / Tc
        val = _seg_E_117(-w, tp, tp + T)
        if tp > 0:
            val += _seg_sin_117(a, 0.0, w, 0.0, tp)
            val += _seg_sin_117(-a, a * (2 * tp + T), w, tp + T, 2 * tp + T)
        gm[m - 1] = val / Tc
    return g0, gm


@pytest.mark.parametrize("T, Tc, tp", [
    (0.4, 1.0, 0.0), (0.4, 1.0, 0.01), (0.1, 1.0, 0.0), (0.9, 1.0, 0.0),
    (0.3, 1.0, 0.025), (1.0, 1.0, 0.0), (0.5, 1.0, 0.125),   # exact resonances
    (0.98, 1.0, 0.01), (2e-3, 0.7, 1e-4)])
def test_dick_coefficients_are_bit_identical_to_the_loop(T, Tc, tp):
    g0, gm = cs.dick_fourier_coefficients(20000, T, Tc, tp)
    g0_ref, gm_ref = _dick_coefficients_117(20000, T, Tc, tp)
    assert g0 == g0_ref
    assert np.array_equal(gm, gm_ref)


# ---------------------------------------------------------------------------
# Cumulant right-hand side: 1.17.0 recomputed the rate constants and
# several products at every call
# ---------------------------------------------------------------------------

def _rhs_117(t, y, rt, feedback=True):
    """cavsqueeze 1.17.0 cumulant._rhs, verbatim (the reference)."""
    M = rt.M
    st = C.State.unpack(y, M)
    s, z, Pc, Qc, Rc, Zc = st.s, st.z, st.Pc, st.Qc, st.Rc, st.Zc
    G, n, delta = rt.G, rt.n, rt.delta
    chi1, Gd, Gu, gphi = rt.chi1, rt.Gd, rt.Gu, rt.gamma_phi

    c = 1j * chi1 + 0.5 * (Gd - Gu)
    cc = np.conj(c)
    A = 1j * (delta + chi1 * G**2) + gphi + 0.5 * (Gd + Gu) * G**2
    Ac = np.conj(A)
    W = G * n
    sc = np.conj(s)
    Rcc = np.conj(Rc)

    # ---------------- first moments ----------------
    # sum_{k != a} G_k <sigma z_a sigma+_k> = sum_p G_p (n_p - d_pm)(Rc_mp + z_m s_p)
    Ws = W @ s
    fb = 1.0 if feedback else 0.0
    S_zp = fb * (Rc @ W - G * np.diag(Rc)) + z * (Ws - G * s)
    ds = -Ac * s + cc * G * S_zp
    # sum_{k != a} G_k <sigma+_a sigma-_k> = sum_p G_p (n_p - d_pm)(Pc_mp + s_m s*_p)
    S_pm = fb * (Pc @ W - G * np.diag(Pc)) + s * np.conj(Ws - G * s)
    dz = -4.0 * G * np.real(c * S_pm) - G**2 * ((Gd - Gu) + (Gd + Gu) * z)

    # ---------------- helpers ----------------
    Gm, Gn = G[:, None], G[None, :]
    zm, zn = z[:, None], z[None, :]
    sm, sn = s[:, None], s[None, :]
    scm, scn = sc[:, None], sc[None, :]
    GG = Gm * Gn

    def wsum(full, at_m, at_n):
        """sum_p G_p (n_p - d_pm - d_pn) T(p) from the full sum and the p = m, p = n values."""
        return full - Gm * at_m - Gn * at_n

    # ---------------- Pc ----------------
    # (iii) from d sigma+_a:  sum_p W_p [ z_m Pc_pn + s_p Rc*_mn ]
    S1 = wsum(zm * (W @ Pc)[None, :] + Rcc * Ws,
              zm * Pc + Rcc * sm,           # p = m: z_m Pc_mn + s_m Rc*_mn
              zm * np.diag(Pc)[None, :] + Rcc * sn)  # p = n: z_m Pc_nn + s_n Rc*_mn
    # (iii) from d sigma-_b:  sum_p W_p [ z_n Pc_mp + s*_p Rc_nm ]
    S2 = wsum(zn * (Pc @ W)[:, None] + Rc.T * np.conj(Ws),
              zn * np.diag(Pc)[:, None] + Rc.T * scm,
              zn * Pc + Rc.T * scn)
    src_a = 0.5 * (zm + zm * zn + Zc) - Rc * scn - zm * (sn * scn)
    src_b = 0.5 * (zn + zm * zn + Zc) - sm * np.conj(Rc.T) - zn * (sm * scm)
    dPc = (
        -(Ac[:, None] + A[None, :]) * Pc
        + cc * Gm * (Gn * src_a + S1)
        + c * Gn * (Gm * src_b + S2)
        + Gu * GG * (Zc + zm * zn)
    )

    # ---------------- Qc ----------------
    S3 = wsum(zm * (W @ Qc)[None, :] + Rc * Ws,
              zm * Qc + Rc * sm,
              zm * np.diag(Qc)[None, :] + Rc * sn)
    S4 = wsum(zn * (Qc @ W)[:, None] + Rc.T * Ws,
              zn * np.diag(Qc)[:, None] + Rc.T * sm,
              zn * Qc + Rc.T * sn)
    dQc = (
        -(Ac[:, None] + Ac[None, :]) * Qc
        + cc * Gm * (-Gn * (Rc + zm * sn) * sn + S3)
        + cc * Gn * (-Gm * sm * (Rc.T + zn * sm) + S4)
    )

    # ---------------- Rc ----------------
    # (iii) from d sigma z_a: sum_p W_p [ c (s_m Pc_np + s*_p Qc_mn) + c* (s_p Pc_nm + s*_m Qc_pn) ]
    S5 = wsum(c * (sm * (Pc @ W)[None, :] + Qc * np.conj(Ws)) + cc * (Pc.T * Ws + scm * (W @ Qc)[None, :]),
              c * (sm * Pc.T + Qc * scm) + cc * (Pc.T * sm + scm * Qc),
              c * (sm * np.diag(Pc)[None, :] + Qc * scn) + cc * (Pc.T * sn + scm * np.diag(Qc)[None, :]))
    # (iii) from d sigma+_b: sum_p W_p [ z_n Rc_mp + s_p Zc_mn ]
    S7 = wsum(zn * (Rc @ W)[:, None] + Zc * Ws,
              zn * np.diag(Rc)[:, None] + Zc * sm,
              zn * Rc + Zc * sn)
    Pmn_raw = Pc + sm * scn
    Pnm_raw = Pc.T + sn * scm
    src_za = c * (0.5 * (sm - Rc.T - sm * zn) - Pmn_raw * sn) + cc * (-Pnm_raw * sn)
    dRc = (
        -Gm**2 * (Gd + Gu) * Rc
        - 2.0 * GG * src_za
        - 2.0 * Gm * S5
        - Ac[None, :] * Rc
        + cc * GG * (1.0 - zm) * (Rc.T + sm * zn)
        + cc * Gn * S7
        - 2.0 * Gd * GG * (Rc.T + sm * zn)
    )

    # ---------------- Zc ----------------
    # (iii) from d sigma z_a: sum_p W_p [ c (s_m Rc*_np + s*_p Rc_nm) + c* (s_p Rc*_nm + s*_m Rc_np) ]
    S8 = wsum(c * (sm * (Rcc @ W)[None, :] + Rc.T * np.conj(Ws)) + cc * (Rcc.T * Ws + scm * (Rc @ W)[None, :]),
              c * (sm * Rcc.T + Rc.T * scm) + cc * (Rcc.T * sm + scm * Rc.T),
              c * (sm * np.diag(Rcc)[None, :] + Rc.T * scn) + cc * (Rcc.T * sn + scm * np.diag(Rc)[None, :]))
    # (iii) from d sigma z_b: sum_p W_p [ c (s_n Rc*_mp + s*_p Rc_mn) + c* (s_p Rc*_mn + s*_n Rc_mp) ]
    S9 = wsum(c * (sn * (Rcc @ W)[:, None] + Rc * np.conj(Ws)) + cc * (Rcc * Ws + scn * (Rc @ W)[:, None]),
              c * (sn * np.diag(Rcc)[:, None] + Rc * scm) + cc * (Rcc * sm + scn * np.diag(Rc)[:, None]),
              c * (sn * Rcc + Rc * scn) + cc * (Rcc * sn + scn * Rc))
    src_zz_a = c * Pmn_raw * (1.0 - zn) - cc * Pnm_raw * (1.0 + zn)
    src_zz_b = -c * Pnm_raw * (1.0 + zm) + cc * Pmn_raw * (1.0 - zm)
    dZc = (
        -(Gm**2 + Gn**2) * (Gd + Gu) * Zc
        - 2.0 * GG * (src_zz_a + src_zz_b)
        - 2.0 * Gm * S8
        - 2.0 * Gn * S9
        + 4.0 * GG * (Gd * Pmn_raw + Gu * Pnm_raw)
    )

    if getattr(rt, "meas", 0.0):
        dPc, dQc, dRc, dZc = C._add_measurement(rt.meas, getattr(rt, "meas_eta", 1.0), n, st, dPc, dQc, dRc, dZc)
    return C.State(ds, dz, dPc, dQc, dRc, dZc).pack()


@pytest.mark.parametrize("M", [1, 3, 16])
@pytest.mark.parametrize("feedback", [True, False])
def test_rhs_is_bit_identical(M, feedback):
    rng = np.random.default_rng(M)
    rt = C.Rates(delta=rng.normal(size=M), G=rng.uniform(0.5, 1.5, M),
                 n=rng.uniform(1.0, 100.0, M), chi1=0.3, Gd=0.2, Gu=0.05,
                 gamma_phi=0.1)
    pre = C._rhs_constants(rt)
    for _ in range(5):
        y = rng.normal(size=2 * M + 4 * M * M) + 1j * rng.normal(size=2 * M + 4 * M * M)
        ref = _rhs_117(0.0, y, rt, feedback)
        assert np.array_equal(C._rhs(0.0, y, rt, feedback), ref)
        assert np.array_equal(C._rhs(0.0, y, rt, feedback, pre), ref)


def test_evolve_is_bit_identical_to_the_117_right_hand_side():
    """A whole solver run (echo twist of 1e10 spins in 16 classes) with the
    1.17.0 right-hand side passed to the same integrator."""
    from scipy.integrate import solve_ivp
    N = 1e10
    p = cs.from_hz(g_hz=1e6 / np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=0.15)
    ens = cs.equal_probability_classes(cs.lineshape("gaussian", 2 * np.pi * 2e3), 16, N)
    rt = C.Rates.from_params(p, ens)
    st = C.product_state(rt.M, [1.0, 0.0, 0.0])
    new = C.evolve(st, rt, 2e-5)
    ref = solve_ivp(_rhs_117, (0.0, 2e-5), st.pack(), args=(rt, True), rtol=1e-8,
                    atol=1e-10 / N, method="DOP853")
    assert np.array_equal(new.pack(), ref.y[:, -1])


def test_optimal_squeezing_reuses_the_scan():
    """The best entry equals a fresh squeezing_after call at that time,
    bit for bit (1.17.0 made that call)."""
    N = 1e6
    p = cs.from_hz(g_hz=1e6 / np.sqrt(1e10), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=0.15)
    best = cs.optimal_squeezing(p, cs.homogeneous(N), 1e-4, 1e-2, n_coarse=6, max_fine=8)
    again = cs.squeezing_after(p, cs.homogeneous(N), best["t"])
    for key, value in again.items():
        assert best[key] == value, key
