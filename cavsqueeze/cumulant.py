"""Second-order cumulant expansion in *connected* (cumulant) variables.

This is the production solver.  It integrates the same second-order cumulant
hierarchy as `cumulant_raw.py` (which is validated against exact master
equations) but in terms of connected correlations

    Pc[m,n] = <sigma+_a sigma-_b> - <sigma+_a><sigma-_b>
    Qc[m,n] = <sigma+_a sigma+_b> - <sigma+_a><sigma+_b>
    Rc[m,n] = <sigma z_a sigma+_b> - <sigma z_a><sigma+_b>
    Zc[m,n] = <sigma z_a sigma z_b> - <sigma z_a><sigma z_b>

(a in class m, b in class n, a != b).  Connected correlations are O(1/N) while
raw moments are O(1); evolving them directly avoids the catastrophic
cancellation that makes raw moments useless for N > 1e8.  The equations are
linear in the connected variables, with source terms of order (rate)/N, and
the first moments receive O(1/N) feedback from the correlations, so that the
two formulations are mathematically identical (tests/test_connected.py).

Model (cavity adiabatically eliminated, all rates in rad/s):

    H   = sum_{j,k} chi1 G_j G_k sigma+_j sigma-_k + sum_j (delta_j/2) sigma z_j
    L1  = sqrt(Gd) sum_j G_j sigma-_j      collective emission, Gd = Gamma_SR (n_th + 1)
    L2  = sqrt(Gu) sum_j G_j sigma+_j      collective absorption, Gu = Gamma_SR n_th
    L_j = sqrt(gamma_phi/2) sigma z_j      individual pure dephasing

The truncation rule for the connected third-order moments is
<XYB>_c = <X><YB>_c + <Y><XB>_c (Gaussian closure).
"""
from __future__ import annotations

import dataclasses
import numpy as np
from scipy.integrate import solve_ivp

from .ensemble import Ensemble
from .resonator import CavityParams
from .cumulant_raw import Rates as _RatesRaw, rotation_matrix  # noqa: F401


class Rates(_RatesRaw):
    """Rates of the interacting classes plus the spectator (free) spins."""

    spec_delta: np.ndarray = None
    spec_n: np.ndarray = None
    meas: float = 0.0  # continuous QND measurement of J_z: conditional variance 1/(meas t) for pure measurement
    meas_eta: float = 1.0  # detection efficiency: the back-action dephasing rate is meas/(8 meas_eta)

    @classmethod
    def from_params(cls, params: CavityParams, ens: Ensemble) -> "Rates":
        base = _RatesRaw.from_params(params, ens)
        rt = cls(delta=base.delta, G=base.G, n=base.n, chi1=base.chi1, Gd=base.Gd, Gu=base.Gu, gamma_phi=base.gamma_phi)
        rt.spec_delta = np.asarray(ens.spec_delta, float)
        rt.spec_n = np.asarray(ens.spec_n, float)
        return rt

    @property
    def K(self) -> int:
        return len(self.spec_delta) if self.spec_delta is not None else 0

    @property
    def N_total(self) -> float:
        return float(self.n.sum() + (self.spec_n.sum() if self.spec_n is not None else 0.0))


@dataclasses.dataclass
class State:
    """Means and connected correlations of an M-class ensemble."""

    s: np.ndarray
    z: np.ndarray
    Pc: np.ndarray
    Qc: np.ndarray
    Rc: np.ndarray
    Zc: np.ndarray
    vs: np.ndarray = None  # (K,3) Bloch vectors of spectator (free, uncorrelated) spins

    def __post_init__(self):
        if self.vs is None:
            self.vs = np.zeros((0, 3))

    @property
    def M(self):
        return len(self.s)

    def pack(self) -> np.ndarray:
        return np.concatenate(
            [self.s, self.z.astype(complex), self.Pc.ravel(), self.Qc.ravel(), self.Rc.ravel(), self.Zc.ravel()]
        )

    @classmethod
    def unpack(cls, y: np.ndarray, M: int, vs=None) -> "State":
        s = y[:M]
        z = y[M : 2 * M]
        b = y[2 * M :].reshape(4, M, M)
        return cls(s=s, z=z, Pc=b[0], Qc=b[1], Rc=b[2], Zc=b[3], vs=vs)

    def copy(self):
        return State(self.s.copy(), self.z.copy(), self.Pc.copy(), self.Qc.copy(), self.Rc.copy(), self.Zc.copy(), self.vs.copy())

    # raw moments (for comparison with cumulant_raw / exact solvers)
    def raw(self):
        s, z = self.s, self.z
        return (
            self.Pc + np.outer(s, np.conj(s)),
            self.Qc + np.outer(s, s),
            self.Rc + np.outer(z, s),
            self.Zc + np.outer(z, z),
        )


def product_state(M: int, v, K_spec: int = 0) -> State:
    v = np.asarray(v, dtype=float)
    s = np.full(M, 0.5 * (v[0] + 1j * v[1]), dtype=complex)
    z = np.full(M, v[2], dtype=complex)
    zero = np.zeros((M, M), dtype=complex)
    vs = np.tile(v, (K_spec, 1))
    return State(s, z, zero.copy(), zero.copy(), zero.copy(), zero.copy(), vs)


# ---------------------------------------------------------------------------
# Cartesian representation, rotations, collective moments
# ---------------------------------------------------------------------------

def to_cartesian(st: State):
    """Bloch vectors v (M,3) and connected correlation tensor Cc (M,M,3,3)."""
    s, z, P, Q, R, Z = st.s, st.z, st.Pc, st.Qc, st.Rc, st.Zc
    v = np.stack([2 * s.real, 2 * s.imag, z.real], axis=1)
    C = np.empty((st.M, st.M, 3, 3), dtype=complex)
    Qc = np.conj(Q)
    PT = P.T
    C[:, :, 0, 0] = Q + P + PT + Qc
    C[:, :, 0, 1] = -1j * (Q - P + PT - Qc)
    C[:, :, 1, 0] = -1j * (Q + P - PT - Qc)
    C[:, :, 1, 1] = -(Q - P - PT + Qc)
    C[:, :, 0, 2] = 2 * R.T.real
    C[:, :, 1, 2] = 2 * R.T.imag
    C[:, :, 2, 0] = 2 * R.real
    C[:, :, 2, 1] = 2 * R.imag
    C[:, :, 2, 2] = Z
    return v, C


def from_cartesian(v, C) -> State:
    s = 0.5 * (v[:, 0] + 1j * v[:, 1])
    z = v[:, 2].astype(complex)
    Cxx, Cxy, Cyx, Cyy = C[:, :, 0, 0], C[:, :, 0, 1], C[:, :, 1, 0], C[:, :, 1, 1]
    P = 0.25 * (Cxx - 1j * Cxy + 1j * Cyx + Cyy)
    Q = 0.25 * (Cxx + 1j * Cxy + 1j * Cyx - Cyy)
    R = 0.5 * (C[:, :, 2, 0] + 1j * C[:, :, 2, 1])
    Z = C[:, :, 2, 2]
    return State(s, z, P, Q, R, Z)


def rotate(st: State, axis, angle: float) -> State:
    """Global rotation of every Bloch vector by `angle` about `axis` (a pulse)."""
    Rm = rotation_matrix(axis, angle)
    v, C = to_cartesian(st)
    out = from_cartesian(v @ Rm.T, np.einsum("ac,bd,mncd->mnab", Rm, Rm, C))
    out.vs = st.vs @ Rm.T
    return out


def rotate_classes(st: State, axis, angles, spec_angles=None) -> State:
    """Rotation of each class m by its own angle angles[m] about `axis` (a pulse
    whose rotation angle varies between classes, e.g. with the coupling weight).
    Spectator spins rotate by spec_angles (default: the mean of angles)."""
    angles = np.asarray(angles, float)
    Rs = np.stack([rotation_matrix(axis, a) for a in angles])  # (M,3,3)
    v, C = to_cartesian(st)
    vn = np.einsum("mab,mb->ma", Rs, v)
    Cn = np.einsum("mac,nbd,mncd->mnab", Rs, Rs, C)
    out = from_cartesian(vn, Cn)
    K = st.vs.shape[0]
    if K:
        sa = np.full(K, float(angles.mean())) if spec_angles is None else np.asarray(spec_angles, float)
        out.vs = np.stack([st.vs[k] @ rotation_matrix(axis, sa[k]).T for k in range(K)])
    else:
        out.vs = st.vs
    return out


def collective_moments(st: State, n, weights=None, spec_n=None, spec_weights=None):
    """Mean <J> and symmetrised covariance of J_alpha = 1/2 sum_a c_a sigma^alpha_a,
    including spectator spins (uncorrelated, each in a pure state with Bloch vector vs)."""
    n = np.asarray(n, float)
    c = np.ones(st.M) if weights is None else np.asarray(weights, float)
    v, C = to_cartesian(st)
    nc = n * c
    J = 0.5 * (nc @ v)
    pair = np.einsum("m,n,mnab->ab", nc, nc, C) - np.einsum("m,mmab->ab", n * c * c, C)
    same = np.sum(n * c * c) * np.eye(3) - np.einsum("m,ma,mb->ab", n * c * c, v, v)
    Cov = 0.25 * (pair.real + same)
    S1 = float(nc.sum())
    S2 = float(np.sum(n * c * c))
    K = st.vs.shape[0]
    if K and spec_n is not None:
        sn = np.asarray(spec_n, float)
        sc = np.ones(K) if spec_weights is None else np.asarray(spec_weights, float)
        J = J + 0.5 * ((sn * sc) @ st.vs)
        Cov = Cov + 0.25 * (np.sum(sn * sc * sc) * np.eye(3) - np.einsum("k,ka,kb->ab", sn * sc * sc, st.vs, st.vs))
        S1 += float((sn * sc).sum())
        S2 += float(np.sum(sn * sc * sc))
    Cov = 0.5 * (Cov + Cov.T)
    return J, Cov, S1, S2


def transverse_variances(J, Cov):
    """(var_min, var_max, angle) of the covariance in the plane perpendicular to J."""
    Jn = np.linalg.norm(J)
    e3 = J / Jn
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    B = np.stack([e1, e2], axis=1)
    vals, vecs = np.linalg.eigh(B.T @ Cov @ B)
    return vals[0], vals[1], float(np.arctan2(vecs[1, 0], vecs[0, 0])), Jn


def wineland_xi2(st: State, n, weights=None, spec_n=None):
    """Wineland parameter xi_R^2 (=1 for a coherent spin state); for weighted
    collective spins the coherent-state normalization S2/S1^2 is used.
    Returns (xi2, angle, var_min, var_max, |J|)."""
    J, Cov, S1, S2 = collective_moments(st, n, weights, spec_n=spec_n)
    if np.linalg.norm(J) == 0:
        return np.inf, 0.0, np.nan, np.nan, 0.0
    vmin, vmax, ang, Jn = transverse_variances(J, Cov)
    return float(vmin * S1**2 / (Jn**2 * S2)), ang, float(vmin), float(vmax), float(Jn)


def coherence(st: State, n, spec_n=None) -> float:
    """Ramsey contrast 2|<J_perp>|/N."""
    J, _, S1, _ = collective_moments(st, n, spec_n=spec_n)
    return float(2.0 * np.hypot(J[0], J[1]) / S1)


# ---------------------------------------------------------------------------
# Right-hand side (connected variables)
# ---------------------------------------------------------------------------

def _rhs_constants(rt: Rates) -> dict:
    """The parts of `_rhs` that depend only on the rates.  Since 1.18
    `evolve` computes them once per call instead of at every step; each is
    formed by the operations the 1.17 `_rhs` applied (tests/test_speed_118.py
    compares with that code)."""
    G, n, delta = rt.G, rt.n, rt.delta
    chi1, Gd, Gu, gphi = rt.chi1, rt.Gd, rt.Gu, rt.gamma_phi
    c = 1j * chi1 + 0.5 * (Gd - Gu)
    cc = np.conj(c)
    A = 1j * (delta + chi1 * G**2) + gphi + 0.5 * (Gd + Gu) * G**2
    Ac = np.conj(A)
    Gm, Gn = G[:, None], G[None, :]
    GG = Gm * Gn
    return dict(
        c=c, cc=cc, A=A, Ac=Ac, W=G * n, G2=G**2, Gm=Gm, Gn=Gn, GG=GG,
        negAc_A=-(Ac[:, None] + A[None, :]),
        negAc_Ac=-(Ac[:, None] + Ac[None, :]),
        cc_Gm=cc * Gm, c_Gn=c * Gn, cc_Gn=cc * Gn, Gu_GG=Gu * GG, cc_GG=cc * GG,
        neg_Gm=-Gm, neg_Gn=-Gn,
        two_GG=2.0 * GG, two_Gm=2.0 * Gm, two_Gn=2.0 * Gn, four_GG=4.0 * GG,
        two_Gd_GG=2.0 * Gd * GG,
        decay_R=-Gm**2 * (Gd + Gu),
        decay_Z=-(Gm**2 + Gn**2) * (Gd + Gu),
    )


def _rhs(t, y, rt: Rates, feedback: bool = True, pre: dict | None = None):
    """Time derivative of the packed state (connected variables).

    Since 1.19 the sums over the other classes p != m, n are formed once per
    quantity, as the full sum minus the p = m and p = n terms ("brackets"
    B below), and the two complex-conjugate halves of the Zc sources are
    added as 2 Re(c X).  These are rearrangements of the 1.18 expressions
    (kept in tests/test_speed_119.py, which checks agreement to 1e-12 of the
    largest derivative); they take about half as many array operations."""
    M = rt.M
    s = y[:M]
    z = y[M:2 * M]
    b = y[2 * M:].reshape(4, M, M)
    Pc, Qc, Rc, Zc = b[0], b[1], b[2], b[3]
    G = rt.G
    Gd, Gu = rt.Gd, rt.Gu
    k = _rhs_constants(rt) if pre is None else pre
    c, cc, W, Gm, Gn = k["c"], k["cc"], k["W"], k["Gm"], k["Gn"]
    sc = np.conj(s)
    Rcc = np.conj(Rc)
    RcT = Rc.T

    Ws = W @ s
    PcW, WPc = Pc @ W, W @ Pc
    QcW, WQc = Qc @ W, W @ Qc
    RcW = Rc @ W
    dP, dQ, dR = Pc.diagonal(), Qc.diagonal(), Rc.diagonal()
    gs = G * s
    u = Ws - gs                     # sum_{p != m} G_p n_p s_p, minus the self term

    # ---------------- first moments ----------------
    # sum_{k != a} G_k <sigma z_a sigma+_k> = sum_p G_p (n_p - d_pm)(Rc_mp + z_m s_p)
    fb = 1.0 if feedback else 0.0
    S_zp = fb * (RcW - G * dR) + z * u
    ds = -k["Ac"] * s + cc * G * S_zp
    # sum_{k != a} G_k <sigma+_a sigma-_k> = sum_p G_p (n_p - d_pm)(Pc_mp + s_m s*_p)
    S_pm = fb * (PcW - G * dP) + s * np.conj(u)
    dz = -4.0 * G * np.real(c * S_pm) - k["G2"] * ((Gd - Gu) + (Gd + Gu) * z)

    # ---------------- helpers ----------------
    zm, zn = z[:, None], z[None, :]
    sm, sn = s[:, None], s[None, :]
    scm, scn = sc[:, None], sc[None, :]
    # sum_p G_p (n_p - d_pm - d_pn) T(p): the full sum minus the p = m and
    # p = n terms.  E is that sum for T(p) = s_p; the brackets are those for
    # the correlations (the index p in the position named).
    E = u[:, None] - gs[None, :]
    Ec = np.conj(E)
    B_Pp_n = WPc[None, :] - Gm * Pc - Gn * dP[None, :]       # T(p) = Pc_pn
    B_Pm_p = PcW[:, None] - Gm * dP[:, None] - Gn * Pc       # T(p) = Pc_mp
    B_Qp_n = WQc[None, :] - Gm * Qc - Gn * dQ[None, :]       # T(p) = Qc_pn
    B_Qm_p = QcW[:, None] - Gm * dQ[:, None] - Gn * Qc       # T(p) = Qc_mp
    B_Rm_p = RcW[:, None] - Gm * dR[:, None] - Gn * Rc       # T(p) = Rc_mp

    # ---------------- Pc ----------------
    # (iii) from d sigma+_a:  sum_p W_p [ z_m Pc_pn + s_p Rc*_mn ]
    S1 = zm * B_Pp_n + Rcc * E
    # (iii) from d sigma-_b:  sum_p W_p [ z_n Pc_mp + s*_p Rc_nm ]
    S2 = zn * B_Pm_p + RcT * Ec
    src_a = 0.5 * (zm + zm * zn + Zc) - Rc * scn - zm * (sn * scn)
    src_b = 0.5 * (zn + zm * zn + Zc) - sm * Rcc.T - zn * (sm * scm)
    dPc = (
        k["negAc_A"] * Pc
        + k["cc_Gm"] * (Gn * src_a + S1)
        + k["c_Gn"] * (Gm * src_b + S2)
        + k["Gu_GG"] * (Zc + zm * zn)
    )

    # ---------------- Qc ----------------
    S3 = zm * B_Qp_n + Rc * E
    S4 = zn * B_Qm_p + RcT * E
    dQc = (
        k["negAc_Ac"] * Qc
        + k["cc_Gm"] * (k["neg_Gn"] * (Rc + zm * sn) * sn + S3)
        + k["cc_Gn"] * (k["neg_Gm"] * sm * (RcT + zn * sm) + S4)
    )

    # ---------------- Rc ----------------
    # (iii) from d sigma z_a: sum_p W_p [ c (s_m Pc_np + s*_p Qc_mn) + c* (s_p Pc_nm + s*_m Qc_pn) ]
    S5 = c * (sm * B_Pm_p.T + Qc * Ec) + cc * (Pc.T * E + scm * B_Qp_n)
    # (iii) from d sigma+_b: sum_p W_p [ z_n Rc_mp + s_p Zc_mn ]
    S7 = zn * B_Rm_p + Zc * E
    Pmn_raw = Pc + sm * scn
    Pnm_raw = Pc.T + sn * scm
    RcT_szn = RcT + sm * zn
    src_za = c * (0.5 * (sm - RcT - sm * zn) - Pmn_raw * sn) - cc * (Pnm_raw * sn)
    dRc = (
        k["decay_R"] * Rc
        - k["two_GG"] * src_za
        - k["two_Gm"] * S5
        - k["Ac"][None, :] * Rc
        + k["cc_GG"] * (1.0 - zm) * RcT_szn
        + k["cc_Gn"] * S7
        - k["two_Gd_GG"] * RcT_szn
    )

    # ---------------- Zc ----------------
    # (iii) from d sigma z_a: sum_p W_p [ c (s_m Rc*_np + s*_p Rc_nm) + c* (s_p Rc*_nm + s*_m Rc_np) ];
    # the c* half is the complex conjugate of the c half, so the sum is 2 Re(c X)
    S8 = 2.0 * np.real(c * (sm * np.conj(B_Rm_p.T) + RcT * Ec))
    # (iii) from d sigma z_b: sum_p W_p [ c (s_n Rc*_mp + s*_p Rc_mn) + c* (s_p Rc*_mn + s*_n Rc_mp) ]
    S9 = 2.0 * np.real(c * (sn * np.conj(B_Rm_p) + Rc * Ec))
    src_zz = (c * Pmn_raw * (1.0 - zn) - cc * Pnm_raw * (1.0 + zn)
              - c * Pnm_raw * (1.0 + zm) + cc * Pmn_raw * (1.0 - zm))
    dZc = (
        k["decay_Z"] * Zc
        - k["two_GG"] * src_zz
        - k["two_Gm"] * S8
        - k["two_Gn"] * S9
        + k["four_GG"] * (Gd * Pmn_raw + Gu * Pnm_raw)
    )

    if getattr(rt, "meas", 0.0):
        st = State.unpack(y, M)
        dPc, dQc, dRc, dZc = _add_measurement(rt.meas, getattr(rt, "meas_eta", 1.0), rt.n, st, dPc, dQc, dRc, dZc)
    out = np.empty(y.shape, dtype=complex)
    out[:M] = ds
    out[M:2 * M] = dz
    ob = out[2 * M:].reshape(4, M, M)
    ob[0] = dPc
    ob[1] = dQc
    ob[2] = dRc
    ob[3] = dZc
    return out


def _add_measurement(Gm, eta, n, st, dPc, dQc, dRc, dZc):
    """Conditioning on a continuous quantum non-demolition measurement of
    J_z = (1/2) sum_i sigma z_i at rate Gm (Gaussian, Kalman form): the pair
    covariance of every two spins changes as
        d Cov(sigma^a_i, sigma^b_j) = -Gm Cov(sigma^a_i, J_z) Cov(sigma^b_j, J_z) dt,
    with Cov(sigma^a_i, J_z) = (1/2)[delta_az - v^a_i v^z_i + sum_{j != i} C^{az}_{ij}].
    For an ensemble without dynamics this gives dV/dt = -Gm V^2 for V = Var(J_z).
    The back-action of the probe (photon-number fluctuations) rotates all spins
    about z by a common random angle with d Var(angle)/dt = 2 Gamma_phi,
    Gamma_phi = Gm/(8 eta), which adds 2 Gamma_phi (e_z x v_i)^a (e_z x v_j)^b to the
    pair covariance; with eta = 1 the product Var(J_y) Var(J_z) stays minimal.
    Spectator spins are not conditioned (use a discretization without spectators)."""
    v, C = to_cartesian(st)
    dphi = np.stack([-v[:, 1], v[:, 0], np.zeros(st.M)], axis=1)     # d v / d(angle) for a rotation about z
    ez = np.array([0.0, 0.0, 1.0])
    same = ez[None, :] - v * v[:, [2]]                                  # (M,3): delta_az - v^a v^z
    Cz = C[:, :, :, 2]                                                  # (M,M,3): Cov(sigma^a_m, sigma^z_n)
    pair = np.einsum("n,mna->ma", n, Cz) - np.einsum("mma->ma", Cz)
    cvec = 0.5 * (same + pair)                                          # (M,3) complex
    dC = -Gm * np.einsum("ma,nb->mnab", cvec, cvec) + (Gm / (4.0 * eta)) * np.einsum("ma,nb->mnab", dphi, dphi)
    extra = from_cartesian(np.zeros((st.M, 3)), dC)
    return dPc + extra.Pc, dQc + extra.Qc, dRc + extra.Rc, dZc + extra.Zc


def _rhs_meanfield(t, y, rt: Rates):
    M = rt.M
    s, z = y[:M], y[M:]
    G, n, delta = rt.G, rt.n, rt.delta
    chi1, Gd, Gu, gphi = rt.chi1, rt.Gd, rt.Gu, rt.gamma_phi
    c = 1j * chi1 + 0.5 * (Gd - Gu)
    A = 1j * (delta + chi1 * G**2) + gphi + 0.5 * (Gd + Gu) * G**2
    W = G * n
    Ws = W @ s
    ds = -np.conj(A) * s + np.conj(c) * G * z * (Ws - G * s)
    dz = -4.0 * G * np.real(c * s * np.conj(Ws - G * s)) - G**2 * ((Gd - Gu) + (Gd + Gu) * z)
    return np.concatenate([ds, dz])


def _check_t_eval(t_eval, t):
    """Output times must lie in [0, t].  (Before 1.18, evolve with t = 0
    returned the initial state for every requested time, so
    evolve(st, rt, 0.0, t_eval=[0.5]) reported the t = 0 state as the
    state at 0.5 without an error.)"""
    if t_eval is None:
        return
    te = np.asarray(t_eval, dtype=float).ravel()
    if te.size == 0 or not np.all(np.isfinite(te)) or np.any(te < 0) or np.any(te > t):
        raise ValueError(f"t_eval must be a non-empty set of finite times in [0, t] = [0, {t}]")


def _distinct_t_eval(t_eval):
    """(sorted distinct times, index of each requested time among them), or
    None when t_eval is already strictly increasing (then it is passed to
    SciPy unchanged).  Since 1.19 output times may repeat or come in any
    order; before, SciPy refused them ("Values in `t_eval` are not properly
    sorted"), also for a sorted list with a repeated time such as
    ramsey_cumulant(p, ens, [1e-3, 1e-3])."""
    if t_eval is None:
        return None
    te = np.asarray(t_eval, dtype=float).ravel()
    if np.all(np.diff(te) > 0):
        return None
    uniq, inverse = np.unique(te, return_inverse=True)
    return uniq, inverse.ravel()


def evolve(st: State, rt: Rates, t: float, t_eval=None, rtol: float = 1e-8, atol: float | None = None,
           method: str = "DOP853", feedback: bool = True) -> State | list[State]:
    """Evolve for time t >= 0.  Returns the final State, or a list of States
    at the times t_eval (in [0, t]; since 1.19 in any order and possibly
    repeated, each returned in the order given)."""
    if not (np.isfinite(t) and t >= 0):
        raise ValueError(f"evolution time must be finite and >= 0; got {t} (to reverse the "
                         "twisting, flip the cavity detuning instead, as twist_untwist does)")
    _check_t_eval(t_eval, t)
    distinct = _distinct_t_eval(t_eval) if t > 0 else None
    if distinct is not None:
        states = evolve(st, rt, t, t_eval=distinct[0], rtol=rtol, atol=atol, method=method,
                        feedback=feedback)
        return [states[j].copy() for j in distinct[1]]
    if t == 0:
        return st.copy() if t_eval is None else [st.copy() for _ in t_eval]
    K = getattr(rt, "K", 0)
    if st.vs.shape[0] != K:
        # spectators start in the same product state as the classes
        v0 = np.array([2 * st.s[0].real, 2 * st.s[0].imag, st.z[0].real])
        st = State(st.s, st.z, st.Pc, st.Qc, st.Rc, st.Zc, np.tile(v0, (K, 1)))
    y0 = st.pack()
    if atol is None:
        # connected correlations are O(1/N): scale the absolute tolerance accordingly
        atol = 1e-10 / max(rt.N, 1.0)
    sol = solve_ivp(_rhs, (0.0, t), y0, args=(rt, feedback, _rhs_constants(rt)), t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    M = rt.M

    def spec(tau):
        # exact free evolution of the spectators: <sigma+> ~ exp(+i delta t) (same
        # convention as the interacting classes) with single-spin dephasing
        if st.vs.shape[0] == 0:
            return st.vs
        ph = np.asarray(rt.spec_delta) * tau
        damp = np.exp(-rt.gamma_phi * tau)
        x, y, z = st.vs[:, 0], st.vs[:, 1], st.vs[:, 2]
        xn = damp * (x * np.cos(ph) - y * np.sin(ph))
        yn = damp * (y * np.cos(ph) + x * np.sin(ph))
        return np.stack([xn, yn, z], axis=1)

    if t_eval is None:
        return State.unpack(sol.y[:, -1], M, spec(t))
    return [State.unpack(sol.y[:, k], M, spec(sol.t[k])) for k in range(sol.y.shape[1])]


def evolve_meanfield(s, z, rt: Rates, t: float, t_eval=None, rtol=1e-8, atol=1e-11,
                     method="DOP853") -> tuple[np.ndarray, np.ndarray]:
    """Evolve the mean-field equations (averages only, no correlations) for
    time t >= 0, from the class averages s = <sigma+> and z = <sigma z>.
    Returns (s, z) at t, or arrays of shape (M, len(t_eval)) at t_eval (in
    [0, t]; since 1.19 in any order and possibly repeated)."""
    if not (np.isfinite(t) and t >= 0):
        # before 1.17 a negative t silently integrated backwards, which for a
        # dissipative model returns an unphysical state (e.g. contrast e > 1
        # after t = -T2)
        raise ValueError(f"evolution time must be finite and >= 0; got {t}")
    _check_t_eval(t_eval, t)
    distinct = _distinct_t_eval(t_eval) if t > 0 else None
    if distinct is not None:
        s_u, z_u = evolve_meanfield(s, z, rt, t, t_eval=distinct[0], rtol=rtol, atol=atol,
                                    method=method)
        return s_u[:, distinct[1]], z_u[:, distinct[1]]
    y0 = np.concatenate([np.asarray(s, complex), np.asarray(z, complex)])
    M = rt.M
    if t == 0:
        # before 1.18 SciPy was called with an empty time span, and t = 0
        # ended in a TypeError (ramsey_meanfield(..., [0.0]) among others)
        if t_eval is None:
            return y0[:M].copy(), y0[M:].copy()
        k = len(np.atleast_1d(t_eval))
        return np.repeat(y0[:M, None], k, axis=1), np.repeat(y0[M:, None], k, axis=1)
    sol = solve_ivp(_rhs_meanfield, (0.0, t), y0, args=(rt,), t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    M = rt.M
    if t_eval is None:
        return sol.y[:M, -1], sol.y[M:, -1]
    return sol.y[:M, :], sol.y[M:, :]
