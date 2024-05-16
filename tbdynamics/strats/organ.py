from typing import List, Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter, Function, Time
from summer2.functions.time import get_piecewise_function, get_linear_interpolation_function
from tbdynamics.utils import tanh_based_scaleup
import numpy as np


def get_organ_strat(
    infectious_compartments: List[str],
    organ_strata: List[str],
    fixed_params: Dict[str, any],
) -> Stratification:
    """
    Creates and configures an organ stratification for the model. This includes defining
    adjustments for infectiousness, infection death rates, and self-recovery rates based
    on organ involvement, as well as adjusting progression rates by organ using requested
    incidence proportions.

    Args:
        infectious_compartments: A list of names of compartments that can transmit infection.
        organ_strata: A list of organ strata names for stratification (e.g., 'lung', 'extrapulmonary').
        fixed_params: A dictionary containing fixed parameters for the model, including
                      multipliers for infectiousness by organ, death rates by organ, and
                      incidence proportions for different organ involvements.

    Returns:
        A Stratification object configured with organ-specific adjustments.
    """
    strat = Stratification("organ", organ_strata, infectious_compartments)
    # Define infectiousness adjustment by organ status
    inf_adj = {
        stratum: Multiply(fixed_params[f"{stratum}_infect_multiplier"])
        for stratum in organ_strata
    }
    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    # Define different natural history (self recovery) by organ status
    self_recovery_adjustments = {
        stratum: Overwrite(
            Parameter(
                f"{'smear_negative' if stratum == 'extrapulmonary' else stratum}_self_recovery"
            )
        )
        for stratum in organ_strata
    }
    strat.set_flow_adjustments("self_recovery", self_recovery_adjustments)

    # Define different detection rates by organ status.
    detection_adjs = {}
    detection_func = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("screening_scaleup_shape"),
            Parameter("screening_inflection_time"),
            0.0,
            Parameter("screening_end_asymp"),
        ],
    )
    detection_covid_reduction = get_linear_interpolation_function([2020, 2021, 2021.9], [1.0, Parameter("detection_reduction"),1.0])
    cdr_covid_adjusted = get_piecewise_function([2020, 2022], [detection_func, detection_func * detection_covid_reduction, detection_func])
    detection_adjs = {}
    infect_death_adjs = {}

# Detection and infect death
    for organ_stratum in organ_strata:
        # Adjust detection by organ status
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = cdr_covid_adjusted * fixed_params[param_name]

        # Calculate infection death adjustment using detection adjustment
        death_rate_param_name = f"{organ_stratum if organ_stratum != 'extrapulmonary' else 'smear_negative'}_death_rate"
        infect_death_adjs[organ_stratum] = (1.0 - detection_adjs[organ_stratum]) * Parameter(death_rate_param_name) # Calculate the infect-death for untreated TB patients


# Apply the Multiply function to the detection adjustments
    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    infect_death_adjs = {k: Overwrite(v) for k, v in infect_death_adjs.items()}
# Set flow adjustments
    strat.set_flow_adjustments("detection", detection_adjs)
    strat.set_flow_adjustments("infect_death", infect_death_adjs)

    splitting_proportions = {
        "smear_positive": fixed_params["incidence_props_pulmonary"]
        * fixed_params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": fixed_params["incidence_props_pulmonary"]
        * (1.0 - fixed_params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1.0 - fixed_params["incidence_props_pulmonary"],
    }
    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)
    return strat
