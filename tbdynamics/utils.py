from math import log, exp
import jax
from jax import numpy as jnp
import numpy as np

def get_treatment_outcomes(duration, prop_death_among_non_success, natural_death_rate, tsr):
    
    # Calculate the proportion of people dying from natural causes while on treatment
    prop_natural_death_while_on_treatment = 1.0 - jnp.exp(-duration * natural_death_rate)

    # Calculate the target proportion of treatment outcomes resulting in death based on requests
    requested_prop_death_on_treatment = (1.0 - tsr) * prop_death_among_non_success
    
    # Calculate the actual rate of deaths on treatment, with floor of zero
    prop_death_from_treatment = jnp.max(jnp.array((requested_prop_death_on_treatment - prop_natural_death_while_on_treatment, 0.0)))
    
    # Calculate the proportion of treatment episodes resulting in relapse
    relapse_prop = 1.0 - tsr - prop_death_from_treatment - prop_natural_death_while_on_treatment
    
    return tuple([param * duration for param in [tsr, prop_death_from_treatment, relapse_prop]])


def get_average_sigmoid(low_val, upper_val, inflection):
    """
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase with age.
    This is the approach used in Ragonnet et al. (BMC Medicine, 2019)
    """
    return (log(1.0 + exp(upper_val - inflection)) - log(1.0 + exp(low_val - inflection))) / (upper_val - low_val)


def get_average_age_for_bcg(agegroup, age_breakpoints):
    agegroup_idx = age_breakpoints.index(int(agegroup))
    if agegroup_idx == len(age_breakpoints) - 1:
        # We should normally never be in this situation because the last agegroup is not affected by BCG anyway.
        print("Warning: the agegroup name is being used to represent the average age of the group")
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
    assymp_range = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * assymp_range + start_asymptote

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
    diabetes_scale_up = tanh_based_scaleup(t, shape=0.05, inflection_time=1980, start_asymptote=0.0, end_asymptote=1.0)
    return (1.0 - diabetes_scale_up(t) * prop_diabetes * (1.0 - rr_progression_diabetes)) * previous_progression_rate

def detection_func(tfunc, val):
    return tfunc * val

def build_contact_matrix():
    values = [[ 398.43289672,  261.82020387,  643.68286218,  401.62199159,
          356.13449939],
        [ 165.78966683,  881.63067677,  532.84120554,  550.75979227,
          285.62836724],
        [ 231.75164317,  311.38983781,  915.52884268,  673.30894113,
          664.14577066],
        [ 141.94492435,  310.88835505,  786.13676958, 1134.31076003,
          938.03403291],
        [  67.30073632,  170.46333134,  647.30153978, 1018.81243422,
         1763.57657715]]
    matrix = np.array(values).T
    return matrix


