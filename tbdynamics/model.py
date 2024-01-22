from jax import numpy as jnp
from pathlib import Path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput, Function, Time
from summer2 import AgeStratification, Overwrite, Multiply, Stratification
from tbdynamics.utils import triangle_wave_func
from .inputs import get_birth_rate, process_death_rate
from .utils import (
    get_average_sigmoid,
)


BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""


def build_model(
    compartments,
    infectious_compartments,
    latent_compartments,
    age_strata,
    time_start,
    time_end,
    time_step,
    matrix,
    fixed_params,
    organ_strat=True,
    add_triangular=True,
):
    model = CompartmentalModel(
        times=(time_start, time_end),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=time_step,
    )
    if add_triangular:
        set_starting_conditions(model, 0)
        seed_infectious(
            model
        )  # I set infectious seed here by injecting the triangular function
    else:
        set_starting_conditions(model, 1)
    add_entry_flow(model)
    add_natural_death_flow(model)
    add_infection(model)
    add_latency(model)
    add_self_recovery(model)
    add_infect_death(model)
    age_strat = get_age_strat(
        compartments, infectious_compartments, age_strata, fixed_params, matrix
    )
    model.stratify_with(age_strat)
    if organ_strat:
        organ_strat = get_organ_strat(fixed_params, infectious_compartments)
        model.stratify_with(organ_strat)
    # seed_infectious(model, 'infectious', age_strata)
    request_output(
        model, age_strata, compartments, latent_compartments, infectious_compartments
    )
    return model


def set_starting_conditions(model, num_infectious):
    start_pop = Parameter("start_population_size")
    init_pop = {
        "infectious": num_infectious,
        "susceptible": start_pop - num_infectious,
    }

    # Assign to the model
    model.set_initial_population(init_pop)


def add_entry_flow(model: CompartmentalModel):
    process = "birth"
    birth_rates = get_birth_rate()
    destination = "susceptible"
    crude_birth_rate = get_sigmoidal_interpolation_function(
        birth_rates.index, birth_rates
    )
    model.add_crude_birth_flow(
        process,
        crude_birth_rate,
        destination,
    )


def add_natural_death_flow(model: CompartmentalModel):
    process = "universal_death"
    universal_death_rate = 1.0
    model.add_universal_death_flows(process, death_rate=universal_death_rate)


def add_infection(model: CompartmentalModel):
    # seed_args = [Time, Parameter('seed_time'), 0.1, 1.0]
    # voc_seed_func = Function(triangle_wave_func, seed_args)
    # model.add_importation_flow("seeding_infection",voc_seed_func,latent_compartments,split_imports=False) # Set seed at time

    process = "infection"
    origin = "susceptible"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process, Parameter("contact_rate"), origin, destination
    )

    process = "infection_from_latent"
    origin = "late_latent"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_latent"),
        "late_latent",
        "early_latent",
    )

    process = "infection_from_recovered"
    origin = "recovered"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_recovered"),
        origin,
        destination,
    )


def add_latency(model: CompartmentalModel):
    # add stabilization process
    process = "stabilisation"
    origin = "early_latent"
    destination = "late_latent"
    model.add_transition_flow(
        process,
        1,  # later adjusted by age group
        origin,
        destination,
    )

    # Add the early activattion process
    process = "early_activation"
    origin = "early_latent"
    destination = "infectious"
    model.add_transition_flow(
        process,
        1.0,  # later adjusted by age group
        origin,
        destination,
    )

    process = "late_activation"
    origin = "late_latent"
    destination = "infectious"
    model.add_transition_flow(
        process,
        Parameter("progression_multiplier"),  # Set to the adjuster, rather than one
        origin,
        destination,
    )


def add_self_recovery(model: CompartmentalModel):
    process = "self_recovery"
    origin = "infectious"
    destination = "recovered"
    model.add_transition_flow(
        process,
        0.2,
        origin,
        destination,
    )


def add_infect_death(model: CompartmentalModel) -> str:
    process = "infect_death"
    origin = "infectious"
    model.add_death_flow(
        process,
        0.2,
        origin,
    )

def get_age_strat(
    compartments,
    infectious,
    age_strata,
    fixed_params,
    matrix,
) -> str:
    strat = AgeStratification("age", age_strata, compartments)
    if matrix is not None:
        strat.set_mixing_matrix(matrix)
    universal_death_funcs, death_adjs = {}, {}
    death_df = process_death_rate(age_strata)
    for age in age_strata:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(
            death_df.index, death_df[age]
        )
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)
    # Set age-specific late activation rate
    for flow_name, latency_params in fixed_params["age_latency"].items():
        # is_activation_flow = flow_name in ["late_activation"]
        # if flow_name in ["late_activation"]: # adjust stratification for late activation
        adjs = {
            str(t): Multiply(latency_params[max([k for k in latency_params if k <= t])])
            for t in age_strata
        }
        strat.set_flow_adjustments(flow_name, adjs)

    # inflate for diabetes
    # is_activation_flow = flow_name in ['early_activation', 'late_activation']
    # if fixed_params['inflate_reactivation_for_diabetes'] and is_activation_flow:
    #         # Inflate reactivation rate to account for diabetes.
    #         for age in age_strata:
    #             adjs[str(age)] = Function(
    #                 get_latency_with_diabetes,
    #                 [
    #                     Time,
    #                     fixed_params['prop_diabetes'][age],
    #                     adjs[str(age)],
    #                     Parameter('rr_progression_diabetes'),
    #                 ],
    #             )
    inf_switch_age = fixed_params["age_infectiousness_switch"]
    for comp in infectious:
        inf_adjs = {}
        for i, age_low in enumerate(age_strata):
            if comp != "on_treatment":
                infectiousness = (
                    1.0
                    if age_low == age_strata[-1]
                    else get_average_sigmoid(age_low, age_strata[i + 1], inf_switch_age)
                )
            else:
                infectiousness *= 1
            # if comp == "on_treatment":
            #     # Infectiousness multiplier for treatment
            #     infectiousness *= Parameter("on_treatment_infect_multiplier")

            inf_adjs[str(age_low)] = Multiply(infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    return strat


def get_organ_strat(
    fixed_params: dict,
    infectious_compartments: list,
):
    ORGAN_STRATA = [
        "smear_positive",
        "smear_negative",
        "extrapulmonary",
    ]
    strat = Stratification("organ", ORGAN_STRATA, infectious_compartments)
    # Better if do in loop
    # Define infectiousness adjustment by organ status
    inf_adj = {}
    inf_adj["smear_positive"] = Multiply(1)
    inf_adj["smear_negative"] = Multiply(
        fixed_params["smear_negative_infect_multiplier"]
    )
    inf_adj["extrapulmonary"] = Multiply(
        fixed_params["extrapulmonary_infect_multiplier"]
    )
    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    # Define different natural history (infection death) by organ status
    infect_death_adjs = {}
    infect_death_adjs["smear_positive"] = Overwrite(
        Parameter("smear_positive_death_rate")
    )
    infect_death_adjs["smear_negative"] = Overwrite(
        Parameter("smear_negative_death_rate")
    )
    infect_death_adjs["extrapulmonary"] = Overwrite(
        Parameter("smear_negative_death_rate")
    )
    strat.set_flow_adjustments("infect_death", infect_death_adjs)

    # Define different natural history (self recovery) by organ status
    self_recovery_adjs = {}
    self_recovery_adjs["smear_positive"] = Overwrite(
        Parameter("smear_positive_self_recovery")
    )
    self_recovery_adjs["smear_negative"] = Overwrite(
        Parameter("smear_negative_self_recovery")
    )
    self_recovery_adjs["extrapulmonary"] = Overwrite(
        Parameter("smear_negative_self_recovery")
    )
    strat.set_flow_adjustments("self_recovery", self_recovery_adjs)

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


def get_gender_strat(age_strata, compartments, fixed_params):
    requested_strata = fixed_params["gender"]["strata"]
    strat = Stratification("gender", requested_strata, compartments)
    # props = fixed_params['gender']['proportions']
    # Pre-process generic flow adjustments:
    # IF infection is adjusted and other infection flows NOT adjusted
    # THEN use the infection adjustment for the other infection flows

    adjustments = fixed_params["gender"]["adjustments"]
    adjs = {}
    if "infection" in adjustments.keys():
        inf_adjs = fixed_params["gender"]["adjustments"]["infection"]
        item = {"infection": {k: v for k, v in inf_adjs.items()}}
        adjs.update(item)
        for stage in ["latent", "recovered"]:
            flow_name = f"infection_from_{stage}"
            if flow_name not in adjs:
                adjs[flow_name] = adjs["infection"]

    # print(adjs)

    # Set birth flow adjustments
    adjs["birth"] = fixed_params["gender"]["proportions"]
    # # # Set birth flow adjustments. Do not adjust for age under 15
    for flow_name, adjustment in adjs.items():
        for age in age_strata:  # only set adjustment for age > 15
            if flow_name == "birth":
                if age < 15:
                    adj = {k: Multiply(1.0) for k in adjustment.keys()}
                else:
                    adj = {k: Multiply(v) for k, v in adjustment.items()}
                strat.set_flow_adjustments(flow_name, adj)
            else:
                adj = {k: Multiply(v) for k, v in adjustment.items()}

    strat.set_flow_adjustments(flow_name, adj, source_strata={"age": str(age)})
    return strat


def seed_infectious(
    model: CompartmentalModel,
):
    """Seed infectious.

    Args:
        model: The summer epidemiological model
        latent_compartments: The names of the latent compartments

    """

    seed_time = "seed_time"
    seed_duration = "seed_duration"
    seed_args = [Time, Parameter(seed_time), Parameter(seed_duration), 1]
    voc_seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        voc_seed_func,
        "infectious",
        split_imports=False,
    )


def request_output(
    model: CompartmentalModel,
    age_strata,
    compartments,
    latent_compartments,
    infectious_compartments,
    implement_acf=False,
):
    """
    Get the applicable outputs
    """
    request_compartment_output(
        model, "total_population", age_strata, compartments, save_results=True
    )
    model.request_output_for_compartments(
        "latent_population_size", latent_compartments, save_results=True
    )
    # latency
    model.request_function_output(
        "percentage_latent",
        100.0
        * DerivedOutput("latent_population_size")
        / DerivedOutput("total_population"),
    )

    # Prevalence
    model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments, save_results=True
    )
    model.request_function_output(
        "prevalence_infectious",
        1e5
        * DerivedOutput("infectious_population_size")
        / DerivedOutput("total_population"),
    )


def request_compartment_output(
    model, output_name, ages, compartments, save_results=True
):
    model.request_output_for_compartments(
        output_name, compartments, save_results=save_results
    )
    for age_stratum in ages:
        # For age-specific population calculations
        age_output_name = f"{output_name}Xage_{age_stratum}"
        model.request_output_for_compartments(
            age_output_name,
            compartments,
            strata={"age": str(age_stratum)},
            save_results=save_results,
        )
