from typing import List, Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter, Function, Time
from tbdynamics.utils import tanh_based_scaleup


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
        stratum: Multiply(fixed_params.get(f"{stratum}_infect_multiplier", 1))
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
    for organ_stratum in organ_strata:
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = (
            Function(
                tanh_based_scaleup,
                [
                    Time,
                    Parameter("screening_scaleup_shape"),
                    Parameter("screening_inflection_time"),
                    Parameter("screening_start_asymp"),
                    Parameter("screening_end_asymp"),
                ],
            )
            * fixed_params[param_name]
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
