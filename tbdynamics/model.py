from pathlib import Path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Function, Time, DerivedOutput
from summer2 import AgeStratification, Stratification, Overwrite, Multiply
from .utils import triangle_wave_func, get_average_sigmoid
from .inputs import get_birth_rate, get_death_rate, process_death_rate

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
    add_triangular=True,
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
    initialize_model_conditions(model, add_triangular)
    add_entry_flow(model, birth_rates)
    add_natural_death_flow(model)
    add_infection_flow(model)
    add_latency_flow(model)
    add_infect_death_flow(model)
    add_self_recovery_flow(model)
    stratify_model_by_age(
        model,
        compartments,
        infectious_compartments,
        age_strata,
        death_df,
        fixed_params,
        matrix,
    )

    stratify_model_by_organ(model, infectious_compartments, fixed_params)

    request_model_outputs(
        model, compartments, latent_compartments, infectious_compartments, age_strata
    )
    return model


def initialize_model_conditions(model, add_triangular):
    # Set the initial population with either 0 or 1 infectious individual(s)
    start_pop = Parameter("start_population_size")
    if add_triangular:
        model.set_initial_population(
            {
                "infectious": 0,
                "susceptible": start_pop - 0,
            }
        )
        seed_infectious(model)
    else:
        model.set_initial_population(
            {
                "infectious": 1,
                "susceptible": start_pop - 1,
            }
        )


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
        ("stabilisation", 1.0 ,"early_latent", "late_latent"),
        ("early_activation", 1.0 ,"early_latent", "infectious"),
        (
            "late_activation",
            1.0,
            "late_latent",
            "infectious"
        )
    ]
    for latency_flow in latency_flows:
        model.add_transition_flow(*latency_flow)


def add_self_recovery_flow(model):
    model.add_transition_flow("self_recovery", 0.2, "infectious", "recovered")


def add_infect_death_flow(model):
    model.add_death_flow("infect_death", 0.2, "infectious")


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
            str(t): latency_params[max([k for k in latency_params if k <= t])] * (Parameter('progression_multiplier') if flow_name == "late_activation" else 1)
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
    return strat


def stratify_model_by_organ(model, infectious_compartments, fixed_params):
    organ_strat = get_organ_strat(
        infectious_compartments,
        fixed_params,
    )
    model.stratify_with(organ_strat)


def get_organ_strat(
    infectious_compartments: list,
    fixed_params: dict,
):
    ORGAN_STRATA = [
        "smear_positive",
        "smear_negative",
        "extrapulmonary",
    ]
    strat = Stratification("organ", ORGAN_STRATA, infectious_compartments)

    # Define infectiousness adjustment by organ status
    inf_adj = {
        stratum: Multiply(fixed_params.get(f"{stratum}_infect_multiplier", 1))
        for stratum in ORGAN_STRATA
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
        for stratum in ORGAN_STRATA
    }
    strat.set_flow_adjustments("infect_death", infect_death_adjs)

    # Define different natural history (self recovery) by organ status
    self_recovery_adjustments = {
        stratum: Overwrite(
            Parameter(
                f"{'smear_negative' if stratum == 'extrapulmonary' else stratum}_self_recovery"
            )
        )
        for stratum in ORGAN_STRATA
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


def request_model_outputs(
    model, compartments, latent_compartments, infectious_compartments, age_strata
):
    model.request_output_for_compartments("total_population", compartments)
    model.request_output_for_compartments("latent_population_size", latent_compartments)
    model.request_function_output(
        "percentage_latent",
        100.0
        * DerivedOutput("latent_population_size")
        / DerivedOutput("total_population"),
    )
    model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments
    )
    model.request_function_output(
        "prevalence_infectious",
        1e5
        * DerivedOutput("infectious_population_size")
        / DerivedOutput("total_population"),
    )
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )
