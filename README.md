# cavsqueeze

[![Tests](https://github.com/TaN-MM-Org/cavsqueeze/actions/workflows/tests.yml/badge.svg)](https://github.com/TaN-MM-Org/cavsqueeze/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/cavsqueeze?label=PyPI&color=blue)](https://pypi.org/project/cavsqueeze/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22278035-blue)](https://doi.org/10.5281/zenodo.22278035)

Beyond-mean-field simulation of cavity-mediated spin squeezing (one-axis
twisting) for solid-state clock-transition ensembles, and of squeezing by
measurement of the spin population through the resonator. Developed for
171Yb3+:CaWO4 and applicable to any spin ensemble coupled to a resonator.

`cavsqueeze` implements

* the adiabatically eliminated Tavis-Cummings model with collective emission,
  collective thermal absorption, single-spin dephasing and coupling inhomogeneity
  (`resonator.py`);
* discretization of Gaussian, Lorentzian and Voigt lines into frequency classes with
  tail resolution (`ensemble.py`);
* a class-resolved second-order cumulant expansion in *connected* variables, whose
  cost is set by the number of classes rather than by N, so that N = 1e15 is no
  harder than N = 1e3, and which loses no precision to cancellation at large N
  because the means are subtracted analytically (`cumulant.py`), together with
  the raw-moment form used as a test reference (`cumulant_raw.py`); the solver can
  condition the ensemble on a continuous measurement of J_z through the resonator
  (`Rates.meas`, `Rates.meas_eta`);
* exact references: QuTiP master equation for distinguishable spins and the
  permutation-invariant Dicke solver PIQS (`exact.py`);
* pulse sequences: echo twist, Ramsey, twist-untwist readout, plain squeezed readout
  (`protocols.py`);
* far-detuned spectator spins propagated analytically, which removes the stiffness of
  heavy-tailed lines (`ensemble.tail_resolved_classes`, `cumulant.evolve`);
* an independent solver for cross-checking: discrete truncated Wigner trajectories
  (`dtwa.py`), which truncate the equations of motion rather than the statistics.

## Installation

From PyPI:

```
pip install cavsqueeze          # core solver (numpy, scipy, matplotlib)
pip install cavsqueeze[exact]   # adds QuTiP for the exact reference solvers
```

Or from source, to run the tests:

```
git clone https://github.com/TaN-MM-Org/cavsqueeze
cd cavsqueeze
pip install -e .[test]
pytest tests              # validation testbench (about 2 minutes)
```

The test suite validates the cumulant solver against exact QuTiP and PIQS
references, against closed-form limits, and against the independent discrete
truncated Wigner solver; it runs in CI on every push and pull request.

## Minimal example

```python
import numpy as np
from cavsqueeze import from_hz, homogeneous
from cavsqueeze.protocols import optimal_squeezing
N = 1e10
p = from_hz(g_hz=1e6/np.sqrt(N), kappa_hz=1e4, Delta_hz=30e6, T=0.02, T2=0.15)
best = optimal_squeezing(p, homogeneous(N), 1e-6, 1e-2)
print(10*np.log10(best["xi2"]), "dB at", best["t"], "s")
```

## Metrology projections

The `metrology` module turns the solver's collective moments into the
quantities an experiment is designed against, for any platform the solver
can describe: Kitagawa-Ueda and Wineland squeezing parameters, single-shot
phase sensitivity, the projection-noise-limited Allan deviation of a Ramsey
clock, and the field sensitivity of a Ramsey magnetometer.

```python
from cavsqueeze import (squeezing_parameters, clock_allan_deviation,
                        metrological_gain_db)
m = squeezing_parameters(state, ens.n)          # xi2_S, xi2_R, contrast, dphi
print(metrological_gain_db(m["xi2_R"]), "dB over the SQL")
print(clock_allan_deviation(m["dphi"], nu0=4.29e14, T_ramsey=0.1, tau=1.0))
```

The formulas are the standard ones (Kitagawa-Ueda 1993; Wineland 1992;
Itano 1993; Ludlow RMP 2015) and are tested against the solver's own exact
references and closed-form limits.

## Into the bosonic quantum stack

The `interop` module extracts the Holstein-Primakoff mode of the collective
spin from any solver state: the 2x2 Gaussian quadrature covariance (a
coherent spin state maps to the vacuum), its symplectic eigenvalue,
squeezing parameter, angle, thermal occupation and purity, and an export to
a QuTiP density matrix that reproduces that covariance. The export verifies
itself against the target covariance so Fock truncation can never silently
corrupt it, and the QuTiP squeeze-phase convention is locked by a test
rather than assumed.

```python
from cavsqueeze import bosonic_mode, to_qutip
mode = bosonic_mode(state, ens.n)     # Sigma, nu, r, theta, n_th, purity
rho, mode = to_qutip(state, ens.n)    # QuTiP density matrix of the mode
```

## Associated paper

The physics, the conventions and the validation of this package are described
in: T. M. Mahim, M. M. Rahman, A. S. M. Mohsin, *Synchronization sets the
coherence and the squeezing limit of a spin ensemble in a cavity*. The paper's
companion repository,
[yb-cawo4-cavity-squeezing](https://github.com/Tanvir-Mahmud-Mahim/yb-cawo4-cavity-squeezing),
contains the scripts, datasets and figures that reproduce the paper and is
archived on Zenodo; this repository is the software's home for development,
releases and support.

## Contributing and support

Bug reports, questions and pull requests are welcome through
[GitHub issues](https://github.com/TaN-MM-Org/cavsqueeze/issues); see
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the testing
requirements. Tagged releases are published to PyPI by CI.

## License

Apache-2.0 (see LICENSE). Please cite the paper if you use this code; citation
metadata is in [CITATION.cff](CITATION.cff).
