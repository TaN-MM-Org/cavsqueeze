"""cavsqueeze: beyond-mean-field cavity-mediated spin squeezing for solid-state
clock-transition ensembles (second-order cumulant expansion with disorder,
collective decay, thermal photons and coupling inhomogeneity)."""
from .resonator import CavityParams, from_hz, thermal_occupation, loop_gap_dispersive
from .ensemble import Ensemble, equal_probability_classes, homogeneous, lineshape, product_classes, log_uniform_weights
from .interop import bosonic_mode, to_qutip
from .metrology import clock_allan_deviation, magnetometer_sensitivity, metrological_gain_db, phase_sensitivity, squeezing_parameters
from .cumulant import Rates, State, product_state, rotate, evolve, evolve_meanfield, wineland_xi2, collective_moments, coherence, transverse_variances

__version__ = "1.9.0"
