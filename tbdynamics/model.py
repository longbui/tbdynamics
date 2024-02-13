from jax import numpy as jnp
from pathlib import Path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Function, Time,  DerivedOutput
from summer2 import AgeStratification, Overwrite, Multiply

from tbdynamics.utils import triangle_wave_func
from .inputs import get_birth_rate, process_death_rate
from .utils import (
    get_average_sigmoid,
)


BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"

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
    age_strat = get_age_strat(
        compartments, infectious_compartments, age_strata, fixed_params, matrix
    )
    model.stratify_with(age_strat)
    model.request_output_for_compartments(
        "total_population", compartments, save_results=True
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
        1.0,  # later adjusted by age group
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
):
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
    
    strat.set_flow_adjustments('universal_death', death_adjs)


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
            inf_adjs[str(age_low)] = Multiply(infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    return strat


def seed_infectious(model: CompartmentalModel):
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
    compartments,
):
    """
    Get the applicable outputs
    """
    request_compartment_output(
        model, "total_population", compartments, save_results=True
    )

def request_compartment_output(
    model, output_name, compartments, save_results=True
):
    model.request_output_for_compartments(
        output_name, compartments, save_results=save_results
    )
