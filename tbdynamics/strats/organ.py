from typing import List, Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter, Function, Time
from summer2.functions.time import get_piecewise_function
from tbdynamics.utils import tanh_based_scaleup, calculate_cdr_adjustments
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
    strat.set_population_split(
        {"smear_positive": 0.5, "smear_negative": 0.25, "extrapulmonary": 0.25}
    )

    # Define infectiousness adjustment by organ status
    inf_adj = {
        stratum: Multiply(fixed_params[f"{stratum}_infect_multiplier"])
        for stratum in organ_strata
    }
    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    # Define different natural history (infection death) by organ status
    infect_death_adjs = {
        stratum: Overwrite(
            Parameter(
                f"{stratum if stratum != 'extrapulmonary' else 'smear_negative'}_death_rate"
            )
        )
        for stratum in organ_strata
    }
    strat.set_flow_adjustments("infect_death", infect_death_adjs)

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
    detection_covid_reduction = get_piecewise_function(
        np.array((2021, 2022)), [detection_func, detection_func * Parameter("detection_reduction"), detection_func]
    )
    for organ_stratum in organ_strata:
        detection_adjs[organ_stratum] = Function(
            calculate_cdr_adjustments,
            [
                detection_covid_reduction,
                Parameter(
                    f"{organ_stratum if organ_stratum == 'smear_positive' else 'smear_negative'}_death_rate"
                ),
                Parameter(
                    f"{organ_stratum if organ_stratum == 'smear_positive' else 'smear_negative'}_self_recovery"
                ),
            ],
        )

    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    strat.set_flow_adjustments("detection", detection_adjs)

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
