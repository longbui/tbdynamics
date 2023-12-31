from math import log, exp
import jax
from jax import numpy as jnp
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


def get_treatment_outcomes(
    duration, prop_death_among_non_success, natural_death_rate, tsr
):
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


def get_average_sigmoid(low_val, upper_val, inflection):
    """
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase with age.
    This is the approach used in Ragonnet et al. (BMC Medicine, 2019)
    """
    return (
        log(1.0 + exp(upper_val - inflection)) - log(1.0 + exp(low_val - inflection))
    ) / (upper_val - low_val)


def get_average_age_for_bcg(agegroup, age_breakpoints):
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
    return 1.0 - tfunc / 100.0 * (1.0 - fmultiplier)


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


def make_linear_curve(x_0, x_1, y_0, y_1):
    assert x_1 > x_0
    slope = (y_1 - y_0) / (x_1 - x_0)

    @jax.jit
    def curve(x):
        return y_0 + slope * (x - x_0)

    return curve


def get_latency_with_diabetes(
    t,
    prop_diabetes,
    previous_progression_rate,
    rr_progression_diabetes,
):
    diabetes_scale_up = tanh_based_scaleup(
        t, shape=0.05, inflection_time=1980, start_asymptote=0.0, end_asymptote=1.0
    )
    return (
        1.0 - diabetes_scale_up(t) * prop_diabetes * (1.0 - rr_progression_diabetes)
    ) * previous_progression_rate


def detection_func(tfunc, val):
    return tfunc * val


def build_contact_matrix(
    # age_strata,
    # filename
):
    values = [
        [
            1250.691457,
            740.4900331,
            1255.1379411,
            755.99800388,
            351.92836824,
            36.16826398,
        ],
        [
            314.45730134,
            3330.01566881,
            992.71557558,
            924.45607039,
            256.90791233,
            34.21905256,
        ],
        [
            221.15622694,
            710.12087625,
            4321.24105361,
            1576.70431504,
            604.00225891,
            22.48129456,
        ],
        [
            224.33879994,
            751.4812055,
            2148.38482347,
            2289.91571398,
            721.8062501,
            34.84227864,
        ],
        [
            192.9874274,
            481.62767533,
            1300.78238257,
            1044.86906425,
            717.99705055,
            39.63652457,
        ],
        [
            81.9016913,
            334.39864375,
            382.67445579,
            432.07589528,
            297.04273652,
            108.95684725,
        ],
    ]
    matrix = np.array(values)
    # matrix_figsize = 800
    # matrix_fig = go.Figure()
    # matrix_fig.add_trace(go.Heatmap(x=age_strata, y=age_strata, z = matrix, coloraxis="coloraxis"))
    # matrix_fig.update_layout(
    #     xaxis = dict(
    #             tick0 = 0,
    #             tickmode = 'array',
    #             tickvals = age_strata,
    #     ),
    #     yaxis = dict(
    #             tick0 = 0,
    #             tickmode = 'array',
    #             tickvals = age_strata,
    #     )
    # )
    # matrix_fig.update_layout(width=matrix_figsize, height=matrix_figsize * 1.15)
    # matrix_fig.write_image(SUPPLEMENT_PATH / filename)
    # matrix_fig_text = f"Year contact rates by age group (row), contact age group (column) "
    return matrix


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


def replace_underscore_with_space(input_string):
    if "_" in input_string:
        return input_string.replace("_", " ")
    else:
        return input_string


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
