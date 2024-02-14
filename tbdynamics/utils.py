from math import log, exp
from jax import numpy as jnp
import numpy as np
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


def round_sigfig(value: float, sig_figs: int) -> float:
    """
    Round a number to a certain number of significant figures,
    rather than decimal places.

    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    if np.isinf(value):
        return "infinity"
    else:
        return (
            round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1))
            if value != 0.0
            else 0.0
        )

def triangle_wave_func(
    time: float,
    start: float,
    duration: float,
    peak: float,
) -> float:
    """Generate a peaked triangular wave function
    that starts from and returns to zero.

    Args:
        time: Model time
        start: Time at which wave starts
        duration: Duration of wave
        peak: Peak flow rate for wave

    Returns:
        The wave function
    """
    gradient = peak / (duration * 0.5)
    peak_time = start + duration * 0.5
    time_from_peak = jnp.abs(peak_time - time)
    return jnp.where(
        time_from_peak < duration * 0.5, peak - time_from_peak * gradient, 0.0
    )

