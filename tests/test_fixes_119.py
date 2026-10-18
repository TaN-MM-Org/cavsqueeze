"""1.19: each test here fails on 1.18.0 and passes on 1.19.0 (the bugs and
the numbers before and after are in CHANGELOG.md)."""
import numpy as np
import pytest

import cavsqueeze as cs
from cavsqueeze.cumulant import Rates, evolve, evolve_meanfield
from cavsqueeze.protocols import X, css_x, pulse, twist_imperfect

T2 = 0.01


def _free_spins():
    """No coupling (g = 0): the Ramsey contrast is exp(-t/T2) exactly
    (pure dephasing of each spin, d<sigma+>/dt = -<sigma+>/T2)."""
    return cs.from_hz(g_hz=0.0, kappa_hz=1e4, Delta_hz=30e6, T2=T2), cs.homogeneous(1e4)


@pytest.mark.parametrize("ramsey", [cs.ramsey_cumulant, cs.ramsey_meanfield])
def test_ramsey_accepts_repeated_times(ramsey):
    """1.18 said the times may come in any order, but a repeated time
    (sorted or not) ended in SciPy's "Values in `t_eval` are not properly
    sorted" (ValueError).  Now each time is returned in the order given,
    equal to the result for the distinct times, and the contrast is the
    closed form exp(-t/T2) to 1e-7 (the solver's tolerance)."""
    p, ens = _free_spins()
    times = [2e-3, 1e-3, 2e-3, 0.0, 1e-3]
    out = ramsey(p, ens, times)
    ref = ramsey(p, ens, [0.0, 1e-3, 2e-3])
    lookup = dict(zip([0.0, 1e-3, 2e-3], ref["contrast"]))
    assert np.array_equal(out["t"], times)
    assert np.array_equal(out["contrast"], [lookup[t] for t in times])
    assert np.allclose(out["contrast"], np.exp(-np.array(times) / T2), rtol=1e-7, atol=0.0)
    two = ramsey(p, ens, [1e-3, 1e-3])
    assert two["contrast"][0] == two["contrast"][1] == ramsey(p, ens, [1e-3])["contrast"][0]


def test_evolve_with_repeated_or_unsorted_output_times():
    """evolve and evolve_meanfield: output times in any order, with repeats,
    give the states at the distinct times (same solver call), each in the
    order asked; the returned states are separate objects."""
    p, ens = _free_spins()
    rt = Rates.from_params(p, ens)
    st = css_x(rt.M)
    got = evolve(st, rt, 2e-3, t_eval=[2e-3, 5e-4, 2e-3])
    ref = evolve(st, rt, 2e-3, t_eval=[5e-4, 2e-3])
    assert np.array_equal(got[0].pack(), ref[1].pack())
    assert np.array_equal(got[1].pack(), ref[0].pack())
    assert np.array_equal(got[2].pack(), ref[1].pack())
    assert got[0] is not got[2]
    s0, z0 = np.full(1, 0.5 + 0j), np.zeros(1, complex)
    s, z = evolve_meanfield(s0, z0, rt, 2e-3, t_eval=[2e-3, 5e-4, 2e-3])
    s_ref, z_ref = evolve_meanfield(s0, z0, rt, 2e-3, t_eval=[5e-4, 2e-3])
    assert np.array_equal(s, s_ref[:, [1, 0, 1]]) and np.array_equal(z, z_ref[:, [1, 0, 1]])
    # the closed form: <sigma+> = (1/2) exp(-t/T2)
    assert np.allclose(s[0], 0.5 * np.exp(-np.array([2e-3, 5e-4, 2e-3]) / T2), rtol=1e-7)


def test_pulse_refuses_a_negative_duration_and_bad_step_counts():
    """1.18 treated a negative duration as an instantaneous pulse without a
    word (pulse(..., duration=-1e-3) returned exactly the duration = 0
    state), and n_steps = 0 raised ZeroDivisionError."""
    N = 1e6
    p = cs.from_hz(g_hz=1e6 / np.sqrt(1e10), kappa_hz=1e4, Delta_hz=30e6, T2=0.15)
    rt = Rates.from_params(p, cs.homogeneous(N))
    st = css_x(rt.M)
    for bad in (-1e-3, np.nan, np.inf):
        with pytest.raises(ValueError, match="duration"):
            pulse(st, rt, X, np.pi, duration=bad)
        with pytest.raises(ValueError, match="duration"):
            twist_imperfect(st, rt, 1e-4, pi_duration=bad)
    for bad in (0, -2, 2.5, True):
        with pytest.raises(ValueError, match="n_steps"):
            pulse(st, rt, X, np.pi, duration=1e-6, n_steps=bad)
    # valid inputs are unchanged: duration 0 is the instantaneous rotation
    assert np.array_equal(pulse(st, rt, X, 0.3).pack(), cs.rotate(st, X, 0.3).pack())
