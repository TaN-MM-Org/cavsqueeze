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

def _rhs(t, y, rt: Rates, feedback: bool = True):
    M = rt.M
    st = State.unpack(y, M)
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
        dPc, dQc, dRc, dZc = _add_measurement(rt.meas, getattr(rt, "meas_eta", 1.0), n, st, dPc, dQc, dRc, dZc)
    return State(ds, dz, dPc, dQc, dRc, dZc).pack()


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


def evolve(st: State, rt: Rates, t: float, t_eval=None, rtol=1e-8, atol=None, method="DOP853", feedback=True):
    """Evolve for time t.  Returns the final State (or a list at t_eval)."""
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
    sol = solve_ivp(_rhs, (0.0, t), y0, args=(rt, feedback), t_eval=t_eval, rtol=rtol, atol=atol, method=method)
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


def evolve_meanfield(s, z, rt: Rates, t: float, t_eval=None, rtol=1e-8, atol=1e-11, method="DOP853"):
    y0 = np.concatenate([np.asarray(s, complex), np.asarray(z, complex)])
    sol = solve_ivp(_rhs_meanfield, (0.0, t), y0, args=(rt,), t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    M = rt.M
    if t_eval is None:
        return sol.y[:M, -1], sol.y[M:, -1]
    return sol.y[:M, :], sol.y[M:, :]
