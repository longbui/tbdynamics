from pathlib import Path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Function, Time
from summer2 import AgeStratification, Stratification, Overwrite, Multiply
from .utils import (
    triangle_wave_func,
    get_average_sigmoid,
    bcg_multiplier_func,
    get_average_age_for_bcg,
    get_treatment_outcomes,
    tanh_based_scaleup
)
from .inputs import get_birth_rate, get_death_rate, process_death_rate
from .constants import organ_strata, bcg_multiplier_dict

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


def build_model(
    compartments,
    latent_compartments,
    infectious_compartments,
    age_strata,
    time_start,
    time_end,
    time_step,
    fixed_params,
    matrix,
):
    model = CompartmentalModel(
        times=(time_start, time_end),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=time_step,
    )

    birth_rates = get_birth_rate()
    death_rates = get_death_rate()
    death_df = process_death_rate(death_rates, age_strata, birth_rates.index)
    start_pop = Parameter("start_population_size")
    model.set_initial_population({"susceptible": start_pop})
    seed_infectious(model)
    add_entry_flow(model, birth_rates)
    add_natural_death_flow(model)
    add_infection_flow(model)
    add_latency_flow(model)
    add_infect_death_flow(model)
    add_self_recovery_flow(model)
    add_detection(model, fixed_params)
    add_treatment_related_outcomes(model)
    stratify_model_by_age(
        model,
        compartments,
        infectious_compartments,
        age_strata,
        death_df,
        fixed_params,
        matrix,
    )

    stratify_model_by_organ(model, infectious_compartments, organ_strata, fixed_params)

    request_model_outputs(
        model,
        compartments,
        latent_compartments,
        infectious_compartments,
        age_strata,
        organ_strata,
    )
    return model


def add_entry_flow(model, birth_rates):
    process = "birth"
    crude_birth_rate = get_sigmoidal_interpolation_function(
        birth_rates.index, birth_rates.values
    )
    model.add_crude_birth_flow(process, crude_birth_rate, "susceptible")


def add_natural_death_flow(model):
    model.add_universal_death_flows(
        "universal_death", death_rate=1.0
    )  # Adjusted later by age stratification


def add_infection_flow(model):
    infection_flows = [
        ("susceptible", None),
        (
            "late_latent",
            "rr_infection_latent",
        ),
        (
            "recovered",
            "rr_infection_recovered",
        ),
    ]
    for origin, modifier in infection_flows:
        process = f"infection_from_{origin}"
        modifier = Parameter(modifier) if modifier else 1.0
        flow_rate = Parameter("contact_rate") * modifier
        model.add_infection_frequency_flow(process, flow_rate, origin, "early_latent")


def add_latency_flow(model):
    latency_flows = [
        ("stabilisation", 1.0, "early_latent", "late_latent"),
        ("early_activation", 1.0, "early_latent", "infectious"),
        ("late_activation", 1.0, "late_latent", "infectious"),
    ]
    for latency_flow in latency_flows:
        model.add_transition_flow(*latency_flow)


def add_self_recovery_flow(model):
    model.add_transition_flow("self_recovery", 0.2, "infectious", "recovered")


def add_infect_death_flow(model):
    model.add_death_flow("infect_death", 0.2, "infectious")


def add_detection(model, fixed_params):
    detection_rate = get_sigmoidal_interpolation_function(
        list(fixed_params["detection_rate"].keys()),
        list(fixed_params["detection_rate"].values()),
    )

    # Adding a transition flow named 'detection' to the model
    model.add_transition_flow("detection", detection_rate, "infectious", "on_treatment")


def add_treatment_related_outcomes(model):
    """
    Adds treatment-related outcome flows to the compartmental model. This includes flows for treatment recovery,
    treatment-related death, and relapse. Initial rates are set as placeholders, with the expectation that
    they may be adjusted later based on specific factors such as organ involved or patient age.
    """

    treatment__outcomes_flows = [
        ("treatment_recovery", 1.0, "recovered"),
        ("relapse", 1.0, "infectious"),
    ]

    # Add each transition flow defined in treatment_flows
    for flow_name, rate, to_compartment in treatment__outcomes_flows:
        model.add_transition_flow(
            flow_name,
            rate,  # Directly using the rate for now
            "on_treatment",
            to_compartment,
        )

    # Define and add treatment death flow separately since it uses a different method
    treatment_death_flow = ["treatment_death", 1.0, "on_treatment"]
    model.add_death_flow(*treatment_death_flow)


def stratify_model_by_age(
    model,
    compartments,
    infectious_compartments,
    age_strata,
    death_df,
    fixed_params,
    matrix,
):
    age_strat = get_age_strat(
        compartments,
        infectious_compartments,
        age_strata,
        death_df,
        fixed_params,
        matrix,
    )
    model.stratify_with(age_strat)


def get_age_strat(compartments, infectious, age_strata, death_df, fixed_params, matrix):
    strat = AgeStratification("age", age_strata, compartments)
    strat.set_mixing_matrix(matrix)
    universal_death_funcs, death_adjs = {}, {}
    for age in age_strata:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(
            death_df.index, death_df[age]
        )
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)
    # Set age-specific latency rate
    for flow_name, latency_params in fixed_params["age_latency"].items():
        adjs = {
            str(t): latency_params[max([k for k in latency_params if k <= t])]
            * (
                Parameter("progression_multiplier")
                if flow_name == "late_activation"
                else 1
            )
            for t in age_strata
        }
        adjs = {str(k): Overwrite(v) for k, v in adjs.items()}
        strat.set_flow_adjustments(flow_name, adjs)

    inf_switch_age = fixed_params["age_infectiousness_switch"]
    for comp in infectious:
        inf_adjs = {}
        for i, age_low in enumerate(age_strata):
            infectiousness = (
                1.0
                if age_low == age_strata[-1]
                else get_average_sigmoid(age_low, age_strata[i + 1], inf_switch_age)
            )
            inf_adjs[str(age_low)] = Multiply(infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    # Add BCG effect without stratifying for BCG
    bcg_adjs = {}  # Initialize dictionary to hold BCG adjustments
    for age, multiplier in bcg_multiplier_dict.items():
        bcg_adjs[age] = calculate_bcg_adjustment(
            age,
            multiplier,
            age_strata,
            list(fixed_params["time_variant_bcg_perc"].keys()),
            list(fixed_params["time_variant_bcg_perc"].values()),
        )
    strat.set_flow_adjustments("infection_from_susceptible", bcg_adjs)

     # Get the treatment outcomes, using the get_treatment_outcomes function above and apply to model
    # Initialize dictionaries to hold treatment outcome functions by age strata
    time_variant_tsr = get_sigmoidal_interpolation_function(
        list(fixed_params["time_variant_tsr"].keys()),
        list(fixed_params["time_variant_tsr"].values()),
    )
    treatment_recovery_funcs = {}
    treatment_death_funcs = {}
    treatment_relapse_funcs = {}
    for age in age_strata:
        death_rate = universal_death_funcs[age]
        treatment_outcomes = Function(
            get_treatment_outcomes,
                [
                    fixed_params["treatment_duration"],
                    fixed_params["prop_death_among_negative_tx_outcome"],
                    death_rate,
                    time_variant_tsr,
                ],
            )
        treatment_recovery_funcs[str(age)] = Multiply(treatment_outcomes[0])
        treatment_death_funcs[str(age)] = Multiply(treatment_outcomes[1])
        treatment_relapse_funcs[str(age)] = Multiply(treatment_outcomes[2])
    strat.set_flow_adjustments("treatment_recovery", treatment_recovery_funcs)
    strat.set_flow_adjustments("treatment_death", treatment_death_funcs)
    strat.set_flow_adjustments("relapse", treatment_relapse_funcs)
    return strat


def stratify_model_by_organ(model, infectious_compartments, organ_strata, fixed_params):
    organ_strat = get_organ_strat(infectious_compartments, organ_strata, fixed_params)
    model.stratify_with(organ_strat)


def get_organ_strat(
    infectious_compartments: list,
    organ_strata,
    fixed_params: dict,
):
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


    # Adjust the progression rates by organ using the requested incidence proportions
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


def seed_infectious(model: CompartmentalModel):
    """Seed infectious.
    Args:
        model: The summer epidemiological model
        latent_compartments: The names of the latent compartments
    """
    seed_args = [
        Time,
        Parameter("seed_time"),
        Parameter("seed_duration"),
        Parameter("seed_num"),
    ]
    voc_seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        voc_seed_func,
        "infectious",
        split_imports=False,
    )


def calculate_bcg_adjustment(
    age, multiplier, age_strata, bcg_time_keys, bcg_time_values
):
    if multiplier < 1.0:
        # Calculate age-adjusted multiplier using a sigmoidal interpolation function
        age_adjusted_time = Time - get_average_age_for_bcg(age, age_strata)
        interpolation_func = get_sigmoidal_interpolation_function(
            bcg_time_keys,
            bcg_time_values,
            age_adjusted_time,
        )
        return Multiply(Function(bcg_multiplier_func, [interpolation_func, multiplier]))
    else:
        # No adjustment needed for multipliers of 1.0
        return None


def request_model_outputs(
    model,
    compartments,
    latent_compartments,
    infectious_compartments,
    age_strata,
    organ_strata,
):
    # Request total population size
    total_pop = model.request_output_for_compartments("total_population", compartments)

    # Calculate and request percentage of latent population
    latent_pop_size = model.request_output_for_compartments(
        "latent_population_size", latent_compartments
    )
    model.request_function_output(
        "percentage_latent", 100.0 * latent_pop_size / total_pop
    )

    # Calculate and request prevalence of infectious population
    infectious_pop_size = model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments
    )
    model.request_function_output(
        "prevalence_infectious", 1e5 * infectious_pop_size / total_pop
    )

    model.request_output_for_flow("notification", "detection", save_results=True)

    # Request proportion of each compartment in the total population
    for compartment in compartments:
        compartment_size = model.request_output_for_compartments(
            f"number_{compartment}", compartment
        )
        model.request_function_output(
            f"prop_{compartment}", compartment_size / total_pop
        )

    # Request total population by age stratum
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )
    for organ_stratum in organ_strata:
        organ_size = model.request_output_for_compartments(
            f"total_populationXorgan_{organ_stratum}",
            compartments,
            strata={"organ": str(organ_stratum)},
        )
        model.request_function_output(
            f"prop_{organ_stratum}", organ_size / infectious_pop_size
        )
