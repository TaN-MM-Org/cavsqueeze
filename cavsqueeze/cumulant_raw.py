"""Second-order cumulant expansion for the cavity-mediated spin model.

Effective master equation (cavity adiabatically eliminated):

    H   = sum_{j,k} chi1 G_j G_k sigma+_j sigma-_k + sum_j (delta_j/2) sigma z_j
    L1  = sqrt(Gd) sum_j G_j sigma-_j      (collective emission, Gd = Gamma_SR (n_th+1))
    L2  = sqrt(Gu) sum_j G_j sigma+_j      (collective absorption, Gu = Gamma_SR n_th)
    L_j = sqrt(gamma_phi/2) sigma z_j      (individual pure dephasing)

with G_j = g * w_j the coupling of spin j, chi1 = 4 Delta/(4 Delta^2 + kappa^2)
and Gamma_SR = 4 kappa/(4 Delta^2 + kappa^2).

Spins are grouped into M classes of identical spins.  State variables:
    s[m]    = <sigma+_a>,           a in m
    z[m]    = <sigma z_a>
    P[m,n]  = <sigma+_a sigma-_b>,  a in m, b in n, a != b
    Q[m,n]  = <sigma+_a sigma+_b>
    R[m,n]  = <sigma z_a sigma+_b>
    Z[m,n]  = <sigma z_a sigma z_b>
Third-order moments are truncated with the Gaussian (second-order cumulant)
rule <ABC> = <AB><C> + <AC><B> + <BC><A> - 2<A><B><C>.

The equations were derived by hand (see Supplement) and are verified against
exact master-equation solutions in tests/test_exact.py.
"""
from __future__ import annotations

import dataclasses
import numpy as np
from scipy.integrate import solve_ivp

from .ensemble import Ensemble
from .resonator import CavityParams


@dataclasses.dataclass
class Rates:
    """Model rates in the class basis."""

    delta: np.ndarray  # (M,)
    G: np.ndarray  # (M,) coupling of each class (rad/s)
    n: np.ndarray  # (M,) occupations
    chi1: float  # exchange rate per unit G_j G_k
    Gd: float  # collective emission per unit G_j G_k
    Gu: float  # collective absorption per unit G_j G_k
    gamma_phi: float = 0.0

    @classmethod
    def from_params(cls, params: CavityParams, ens: Ensemble) -> "Rates":
        denom = 4.0 * params.Delta**2 + params.kappa**2
        chi1 = 4.0 * params.Delta / denom
        Gsr = 4.0 * params.kappa / denom
        nth = params.n_th
        return cls(
            delta=ens.delta.copy(),
            G=params.g * ens.weight,
            n=ens.n.copy(),
            chi1=chi1,
            Gd=Gsr * (nth + 1.0),
            Gu=Gsr * nth,
            gamma_phi=params.gamma_phi,
        )

    @property
    def M(self) -> int:
        return len(self.delta)

    @property
    def N(self) -> float:
        return float(self.n.sum())

    @property
    def chiN(self) -> float:
        """Collective exchange rate sum_k chi1 G_k^2 n_k (the OAT gap chi N for uniform G)."""
        return float(self.chi1 * np.sum(self.G**2 * self.n))

    @property
    def GammaN(self) -> float:
        return float((self.Gd - self.Gu) * np.sum(self.G**2 * self.n))


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class State:
    s: np.ndarray
    z: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z: np.ndarray

    @property
    def M(self):
        return len(self.s)

    def pack(self) -> np.ndarray:
        return np.concatenate(
            [self.s, self.z.astype(complex), self.P.ravel(), self.Q.ravel(), self.R.ravel(), self.Z.ravel()]
        )

    @classmethod
    def unpack(cls, y: np.ndarray, M: int) -> "State":
        s = y[:M]
        z = y[M : 2 * M]
        blocks = y[2 * M :].reshape(4, M, M)
        return cls(s=s, z=z, P=blocks[0], Q=blocks[1], R=blocks[2], Z=blocks[3])

    def copy(self):
        return State(self.s.copy(), self.z.copy(), self.P.copy(), self.Q.copy(), self.R.copy(), self.Z.copy())


def product_state(M: int, v) -> State:
    """All spins in the same pure single-spin state with Bloch vector v = (vx, vy, vz)."""
    v = np.asarray(v, dtype=float)
    s = np.full(M, 0.5 * (v[0] + 1j * v[1]), dtype=complex)
    z = np.full(M, v[2], dtype=complex)
    # <sigma^alpha_a sigma^beta_b> = v^alpha v^beta for a product state
    P = np.full((M, M), s[0] * np.conj(s[0]), dtype=complex)
    Q = np.full((M, M), s[0] * s[0], dtype=complex)
    R = np.full((M, M), z[0] * s[0], dtype=complex)
    Z = np.full((M, M), z[0] * z[0], dtype=complex)
    return State(s, z, P, Q, R, Z)


# ---------------------------------------------------------------------------
# Rotations (global pulses)
# ---------------------------------------------------------------------------

def rotation_matrix(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def to_cartesian(st: State):
    """Return first-moment Bloch vectors v (M,3) and correlation tensor C (M,M,3,3)
    with C[m,n,alpha,beta] = <sigma^alpha_a sigma^beta_b>, a in m, b in n, a != b."""
    s, z, P, Q, R, Z = st.s, st.z, st.P, st.Q, st.R, st.Z
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
    """Apply a global rotation of every Bloch vector by `angle` about `axis`."""
    Rm = rotation_matrix(axis, angle)
    v, C = to_cartesian(st)
    v2 = v @ Rm.T
    C2 = np.einsum("ac,bd,mncd->mnab", Rm, Rm, C)
    return from_cartesian(v2, C2)


# ---------------------------------------------------------------------------
# Collective spin moments and squeezing
# ---------------------------------------------------------------------------

def collective_moments(st: State, n: np.ndarray, weights=None):
    """Mean vector <J> and symmetrised covariance matrix of the collective spin
    J_alpha = 1/2 sum_a c_a sigma^alpha_a with c_a = 1 (weights None) or the
    supplied per-class weights.  Returns (J, Cov, S1, S2) where S1 = sum c_a and
    S2 = sum c_a^2."""
    M = st.M
    c = np.ones(M) if weights is None else np.asarray(weights, float)
    v, C = to_cartesian(st)
    nc = n * c
    J = 0.5 * (nc @ v)
    # pair part: sum_{m,n} n_m c_m (n_n c_n - delta_mn c_m) C_mn
    pair = np.einsum("m,n,mnab->ab", nc, nc, C) - np.einsum("m,mmab->ab", n * c * c, C)
    same = np.sum(n * c * c) * np.eye(3)
    Cov = 0.25 * (pair.real + same) - np.outer(J, J)
    Cov = 0.5 * (Cov + Cov.T)
    return J, Cov, float(nc.sum()), float(np.sum(n * c * c))


def wineland_xi2(st: State, n: np.ndarray, weights=None):
    """Wineland squeezing parameter xi_R^2 (1 for a coherent spin state).
    For weighted collective spins the coherent-state normalization
    S2 / S1^2 is used so that xi^2 = 1 for a coherent product state.
    Returns (xi2, angle_of_min_variance, var_min, var_max, |J|)."""
    J, Cov, S1, S2 = collective_moments(st, n, weights)
    Jn = np.linalg.norm(J)
    if Jn == 0:
        return np.inf, 0.0, np.nan, np.nan, 0.0
    e3 = J / Jn
    # orthonormal basis of the transverse plane
    trial = np.array([0.0, 0.0, 1.0]) if abs(e3[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    B = np.stack([e1, e2], axis=1)
    C2 = B.T @ Cov @ B
    vals, vecs = np.linalg.eigh(C2)
    vmin, vmax = vals[0], vals[1]
    ang = np.arctan2(vecs[1, 0], vecs[0, 0])
    xi2 = vmin * S1**2 / (Jn**2 * S2)
    return float(xi2), float(ang), float(vmin), float(vmax), float(Jn)


def coherence(st: State, n: np.ndarray) -> float:
    """Normalized transverse coherence 2|<J_perp>|/N used in Ramsey experiments."""
    J, _, S1, _ = collective_moments(st, n)
    return float(2.0 * np.hypot(J[0], J[1]) / S1)


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------

def _rhs(t, y, rt: Rates):
    M = rt.M
    st = State.unpack(y, M)
    s, z, P, Q, R, Z = st.s, st.z, st.P, st.Q, st.R, st.Z
    G, n, delta = rt.G, rt.n, rt.delta
    chi1, Gd, Gu, gphi = rt.chi1, rt.Gd, rt.Gu, rt.gamma_phi

    c = 1j * chi1 + 0.5 * (Gd - Gu)
    cc = np.conj(c)
    A = 1j * (delta + chi1 * G**2) + gphi + 0.5 * (Gd + Gu) * G**2  # (M,)
    Ac = np.conj(A)
    W = G * n  # (M,)
    sc = np.conj(s)
    Rc = np.conj(R)
    Ws = W @ s  # sum_p W_p s_p
    Wsc = np.conj(Ws)
    RW = R @ W  # (RW)_m = sum_p W_p R_mp
    RcW = Rc @ W
    PW = P @ W  # (PW)_m = sum_p W_p P_mp
    WP = W @ P  # (WP)_n = sum_p W_p P_pn
    QW = Q @ W
    WQ = W @ Q
    diagP = np.diag(P)
    diagR = np.diag(R)
    diagQ = np.diag(Q)

    # ---------------- first order ----------------
    ds = -Ac * s + cc * G * (RW - G * diagR)
    dz = -4.0 * G * np.real(c * (PW - G * diagP)) - G**2 * (Gd * (1 + z) - Gu * (1 - z))

    # helpers for the weighted third-order sums S[T](m,n) = sum_p W_p T(m,n,p) - G_m T(m,n,m) - G_n T(m,n,n)
    Gm = G[:, None]
    Gn = G[None, :]
    zm = z[:, None]
    zn = z[None, :]
    sm = s[:, None]
    sn = s[None, :]
    scm = sc[:, None]
    scn = sc[None, :]

    def S_of(full, at_m, at_n):
        return full - Gm * at_m - Gn * at_n

    # ---- T1(m,n,p) = R_mp s*_n + R*_mn s_p + P_pn z_m - 2 z_m s_p s*_n
    T1_full = RW[:, None] * scn + Rc * Ws + WP[None, :] * zm - 2.0 * zm * Ws * scn
    T1_m = diagR[:, None] * scn + Rc * sm + P * zm - 2.0 * zm * sm * scn  # p = m
    T1_n = R * scn + Rc * sn + diagP[None, :] * zm - 2.0 * zm * sn * scn  # p = n
    S1 = S_of(T1_full, T1_m, T1_n)
    # ---- T2(m,n,p) = R_nm s*_p + P_mp z_n + R*_np s_m - 2 s_m z_n s*_p
    T2_full = R.T * Wsc + PW[:, None] * zn + RcW[None, :] * sm - 2.0 * sm * zn * Wsc
    T2_m = R.T * scm + diagP[:, None] * zn + Rc.T * sm - 2.0 * sm * zn * scm
    T2_n = R.T * scn + P * zn + np.diag(Rc)[None, :] * sm - 2.0 * sm * zn * scn
    S2 = S_of(T2_full, T2_m, T2_n)
    dP = (
        -(Ac[:, None] + A[None, :]) * P
        + cc * Gm * (Gn * 0.5 * (zm + Z) + S1)
        + c * Gn * (Gm * 0.5 * (zn + Z) + S2)
        + Gu * Gm * Gn * Z
    )

    # ---- T3(m,n,p) = R_mp s_n + R_mn s_p + Q_pn z_m - 2 z_m s_p s_n
    T3_full = RW[:, None] * sn + R * Ws + WQ[None, :] * zm - 2.0 * zm * Ws * sn
    T3_m = diagR[:, None] * sn + R * sm + Q * zm - 2.0 * zm * sm * sn
    T3_n = R * sn + R * sn + diagQ[None, :] * zm - 2.0 * zm * sn * sn
    S3 = S_of(T3_full, T3_m, T3_n)
    # ---- T4(m,n,p) = R_nm s_p + Q_mp z_n + R_np s_m - 2 s_m z_n s_p
    T4_full = R.T * Ws + QW[:, None] * zn + RW[None, :] * sm - 2.0 * sm * zn * Ws
    T4_m = R.T * sm + diagQ[:, None] * zn + R.T * sm - 2.0 * sm * zn * sm
    T4_n = R.T * sn + Q * zn + diagR[None, :] * sm - 2.0 * sm * zn * sn
    S4 = S_of(T4_full, T4_m, T4_n)
    dQ = -(Ac[:, None] + Ac[None, :]) * Q + cc * Gm * S3 + cc * Gn * S4

    # ---- T5(m,n,p) = P_mp s_n + Q_mn s*_p + P_np s_m - 2 s_m s*_p s_n
    T5_full = PW[:, None] * sn + Q * Wsc + PW[None, :] * sm - 2.0 * sm * sn * Wsc
    T5_m = diagP[:, None] * sn + Q * scm + P.T * sm - 2.0 * sm * sn * scm
    T5_n = P * sn + Q * scn + diagP[None, :] * sm - 2.0 * sm * sn * scn
    S5 = S_of(T5_full, T5_m, T5_n)
    # ---- T6(m,n,p) = P_pm s_n + Q_pn s*_m + P_nm s_p - 2 s_p s*_m s_n
    T6_full = WP[:, None] * sn + WQ[None, :] * scm + P.T * Ws - 2.0 * Ws * scm * sn
    T6_m = diagP[:, None] * sn + Q * scm + P.T * sm - 2.0 * sm * scm * sn
    T6_n = P.T * sn + diagQ[None, :] * scm + P.T * sn - 2.0 * sn * scm * sn
    S6 = S_of(T6_full, T6_m, T6_n)
    # ---- T7(m,n,p) = Z_mn s_p + R_mp z_n + R_np z_m - 2 z_m z_n s_p
    T7_full = Z * Ws + RW[:, None] * zn + RW[None, :] * zm - 2.0 * zm * zn * Ws
    T7_m = Z * sm + diagR[:, None] * zn + R.T * zm - 2.0 * zm * zn * sm
    T7_n = Z * sn + R * zn + diagR[None, :] * zm - 2.0 * zm * zn * sn
    S7 = S_of(T7_full, T7_m, T7_n)
    dR = (
        -Ac[None, :] * R
        - Gm**2 * ((Gd - Gu) * sn + (Gd + Gu) * R)
        - 2.0 * Gm * c * (Gn * 0.5 * (sm - R.T) + S5)
        - 2.0 * Gm * cc * S6
        + cc * Gn * (Gm * R.T + S7)
        - 2.0 * Gd * Gm * Gn * R.T
    )

    # ---- T8(m,n,p) = P_mp z_n + R_nm s*_p + R*_np s_m - 2 s_m s*_p z_n   (from sigma z_a dot times sigma z_b)
    T8_full = PW[:, None] * zn + R.T * Wsc + RcW[None, :] * sm - 2.0 * sm * zn * Wsc
    T8_m = diagP[:, None] * zn + R.T * scm + Rc.T * sm - 2.0 * sm * zn * scm
    T8_n = P * zn + R.T * scn + np.diag(Rc)[None, :] * sm - 2.0 * sm * zn * scn
    S8 = S_of(T8_full, T8_m, T8_n)
    # ---- T9(m,n,p) = P_pm z_n + R_np s*_m + R*_nm s_p - 2 s_p s*_m z_n
    T9_full = WP[:, None] * zn + RW[None, :] * scm + Rc.T * Ws - 2.0 * Ws * scm * zn
    T9_m = diagP[:, None] * zn + R.T * scm + Rc.T * sm - 2.0 * sm * scm * zn
    T9_n = P.T * zn + diagR[None, :] * scm + Rc.T * sn - 2.0 * sn * scm * zn
    S9 = S_of(T9_full, T9_m, T9_n)
    # ---- T8b(m,n,p) = R_mn s*_p + R*_mp s_n + P_np z_m - 2 z_m s_n s*_p   (from sigma z_a times sigma z_b dot)
    T8b_full = R * Wsc + RcW[:, None] * sn + PW[None, :] * zm - 2.0 * zm * sn * Wsc
    T8b_m = R * scm + np.diag(Rc)[:, None] * sn + P.T * zm - 2.0 * zm * sn * scm
    T8b_n = R * scn + Rc * sn + diagP[None, :] * zm - 2.0 * zm * sn * scn
    S8b = S_of(T8b_full, T8b_m, T8b_n)
    # ---- T9b = T1
    S9b = S1
    dZ = (
        -2.0 * Gm * (c * (Gn * P + S8) + cc * (-Gn * P.T + S9))
        - Gm**2 * ((Gd - Gu) * zn + (Gd + Gu) * Z)
        - 2.0 * Gn * (c * (-Gm * P.T + S8b) + cc * (Gm * P + S9b))
        - Gn**2 * ((Gd - Gu) * zm + (Gd + Gu) * Z)
        + 4.0 * Gd * Gm * Gn * P
        + 4.0 * Gu * Gm * Gn * P.T
    )

    return State(ds, dz, dP, dQ, dR, dZ).pack()


def _rhs_meanfield(t, y, rt: Rates):
    M = rt.M
    s = y[:M]
    z = y[M:]
    G, n, delta = rt.G, rt.n, rt.delta
    chi1, Gd, Gu, gphi = rt.chi1, rt.Gd, rt.Gu, rt.gamma_phi
    c = 1j * chi1 + 0.5 * (Gd - Gu)
    A = 1j * (delta + chi1 * G**2) + gphi + 0.5 * (Gd + Gu) * G**2
    W = G * n
    Ws = W @ s
    ds = -np.conj(A) * s + np.conj(c) * G * z * (Ws - G * s)
    dz = -4.0 * G * np.real(c * s * np.conj(Ws - G * s)) - G**2 * (Gd * (1 + z) - Gu * (1 - z))
    return np.concatenate([ds, dz])


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------

def evolve(st: State, rt: Rates, t: float, t_eval=None, rtol=1e-8, atol=1e-11, method="DOP853"):
    """Evolve the cumulant state for time t.  Returns the final State, or, if
    t_eval is given, the list of States at those times."""
    y0 = st.pack()
    if t == 0:
        return st.copy() if t_eval is None else [st.copy() for _ in t_eval]
    sol = solve_ivp(_rhs, (0.0, t), y0, args=(rt,), t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    M = rt.M
    if t_eval is None:
        return State.unpack(sol.y[:, -1], M)
    return [State.unpack(sol.y[:, k], M) for k in range(sol.y.shape[1])]


def evolve_meanfield(s, z, rt: Rates, t: float, t_eval=None, rtol=1e-8, atol=1e-11, method="DOP853"):
    y0 = np.concatenate([np.asarray(s, complex), np.asarray(z, complex)])
    sol = solve_ivp(_rhs_meanfield, (0.0, t), y0, args=(rt,), t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    M = rt.M
    if t_eval is None:
        return sol.y[:M, -1], sol.y[M:, -1]
    return sol.y[:M, :], sol.y[M:, :]
