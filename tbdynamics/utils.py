from math import log, exp
from jax import numpy as jnp
import numpy as np
from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List


BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


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


def get_average_sigmoid(low_val, upper_val, inflection):
    """
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase with age.
    This is the approach used in Ragonnet et al. (BMC Medicine, 2019)
    """
    return (
        log(1.0 + exp(upper_val - inflection)) - log(1.0 + exp(low_val - inflection))
    ) / (upper_val - low_val)


def tanh_based_scaleup(t, shape, inflection_time, start_asymptote, end_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param start_asymptote: lowest asymptotic value
    :param end_asymptote: highest asymptotic value
    :return: a function
    """
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote


def get_average_age_for_bcg(agegroup, age_breakpoints):
    """
    Calculates the average age for a given age group, particularly for determining
    BCG vaccination relevance.

    The function assumes age groups are represented by their starting age and uses
    the provided age breakpoints to calculate the midpoint between the given age group
    and the next. This midpoint is considered the average age for the group. If the
    specified age group is the last in the list, a warning is printed, and the age group
    itself is returned as the average age, since the last age group is typically not
    targeted for BCG vaccination.

    Parameters:
    - agegroup (str): The starting age of the age group.
    - age_breakpoints (list): A list of integers representing the starting ages of each
      age group, sorted in ascending order.

    Returns:
    - float: The average age of the specified age group. If the age group is the last
      in the list, returns the starting age of that group.

    Notes:
    - This function is designed with the assumption that BCG vaccination policies target
      specific age groups, excluding the oldest group represented in age_breakpoints.
    """
    agegroup_idx = age_breakpoints.index(int(agegroup))
    if agegroup_idx == len(age_breakpoints) - 1:
        # We should normally never be in this situation because the last agegroup is not affected by BCG anyway.
        print(
            "Warning: the agegroup name is being used to represent the average age of the group"
        )
        return float(agegroup)
    else:
        return 0.5 * (age_breakpoints[agegroup_idx] + age_breakpoints[agegroup_idx + 1])


def bcg_multiplier_func(tfunc, fmultiplier):
    """
    Calculates the BCG vaccination effect multiplier based on a transmission function
    and a fractional multiplier.

    This function is used to adjust the impact of BCG vaccination on disease transmission
    by calculating a multiplier based on the reduction in transmission (as a percentage)
    and a fractional multiplier that represents the remaining susceptibility after vaccination.

    Parameters:
    - tfunc (float): The sigmoidal function, reflecting the percentage reduction in transmission due to BCG vaccination.
    - fmultiplier (float): The fractional multiplier representing the remaining susceptibility
      or the efficacy of the BCG vaccine in preventing disease transmission.

    Returns:
    - float: A multiplier representing the adjusted impact of BCG vaccination on disease
      transmission.

    """
    return 1.0 - tfunc / 100.0 * (1.0 - fmultiplier)


def calculate_treatment_outcomes(
    duration, prop_death_among_non_success, natural_death_rate, tsr
):
    """
    Calculates the adjusted proportions of treatment outcomes over a specified duration,
    considering the natural death rate and predefined treatment success rate (TSR).

    This function determines the proportions of individuals who complete treatment successfully,
    die from the treatment, relapse after treatment, or die from natural causes while undergoing
    treatment, adjusted for the specified duration of treatment.

    Parameters:
    - duration (float): The duration of the treatment period, in the same time units as the
      natural_death_rate (e.g., years).
    - prop_death_among_non_success (float): The proportion of non-successful treatment outcomes
      that result in death due to the treatment, excluding natural deaths.
    - natural_death_rate (float): The annual rate of natural death, applied to the population
      under treatment.
    - tsr (float): The target success rate of the treatment, defined as the proportion of
      treatment episodes resulting in success.

    Returns:
    - tuple of float: A tuple containing the adjusted proportions of treatment success,
      deaths from treatment (with a floor of zero), and relapse, each divided by the duration
      of the treatment. These proportions are calculated after accounting for the natural death
      rate during treatment.

    Notes:
    - The function assumes exponential decay for calculating the proportion of natural deaths
      during treatment, which is a common model for time-to-event data.
    - The function ensures that the calculated death from treatment cannot be negative, by
      setting a floor of zero.
    """
    # Calculate the proportion of people dying from natural causes while on treatment
    prop_natural_death_while_on_treatment = 1.0 - jnp.exp(
        -duration * natural_death_rate
    )

    # Calculate the target proportion of treatment outcomes resulting in death based on requests
    requested_prop_death_on_treatment = (1.0 - tsr) * prop_death_among_non_success

    # Calculate the actual rate of deaths on treatment, with floor of zero
    prop_death_from_treatment = jnp.max(
        jnp.array(
            (
                requested_prop_death_on_treatment
                - prop_natural_death_while_on_treatment,
                0.0,
            )
        )
    )

    # Calculate the proportion of treatment episodes resulting in relapse
    relapse_prop = (
        1.0 - tsr - prop_death_from_treatment - prop_natural_death_while_on_treatment
    )

    return tuple(
        [param * duration for param in [tsr, prop_death_from_treatment, relapse_prop]]
    )


def round_sigfig(
    value: float, 
    sig_figs: int
) -> float:
    """
    Round a number to a certain number of significant figures, 
    rather than decimal places.
    
    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    if np.isinf(value):
        return 'infinity'
    else:
        return round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1)) if value != 0.0 else 0.0

def get_target_from_name(
    targets: list, 
    name: str,
) -> pd.Series:
    """Get the data for a specific target from a set of targets from its name.

    Args:
        targets: All the targets
        name: The name of the desired target

    Returns:
        Single target to identify
    """
    return next((t.data for t in targets if t.name == name), None)

def get_row_col_for_subplots(i_panel, n_cols):
    return int(np.floor(i_panel / n_cols)) + 1, i_panel % n_cols + 1


def get_standard_subplot_fig(
    n_rows: int, 
    n_cols: int, 
    titles: List[str],
    share_y: bool=False,
) -> go.Figure:
    """Start a plotly figure with subplots off from standard formatting.

    Args:
        n_rows: Argument to pass through to make_subplots
        n_cols: Pass through
        titles: Pass through

    Returns:
        Figure with nothing plotted
    """
    heights = [320, 600, 680]
    height = 680 if n_rows > 3 else heights[n_rows - 1]
    fig = make_subplots(n_rows, n_cols, subplot_titles=titles, vertical_spacing=0.08, horizontal_spacing=0.05, shared_yaxes=share_y)
    return fig.update_layout(margin={i: 25 for i in ['t', 'b', 'l', 'r']}, height=height)

def calculate_cdr_adjustments(case_detection_rate, infect_death, self_recovery):
    return case_detection_rate * (infect_death + self_recovery) / (1 - case_detection_rate)
