from pathlib import Path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Function, Time
from summer2 import AgeStratification, Overwrite, Multiply
from tbdynamics.utils import triangle_wave_func, get_average_sigmoid
from .inputs import get_birth_rate, process_death_rate

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"

def build_model(
    compartments,
    infectious_compartments,
    age_strata,
    time_start,
    time_end,
    time_step,
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
        seed_infectious(model)
    else:
        set_starting_conditions(model, 1)
    add_entry_flow(model)
    add_natural_death_flow(model)
    add_infection(model)
    add_latency(model)
    add_infect_death(model)
    add_self_recovery(model)
    age_strat = get_age_strat(compartments, infectious_compartments,age_strata, fixed_params)
    model.stratify_with(age_strat)
    model.request_output_for_compartments(
        "total_population", compartments, save_results=True
    )
    for age_stratum in age_strata:
        # For age-specific population calculations
        age_output_name = f"total_populationXage_{age_stratum}"
        model.request_output_for_compartments(
            age_output_name,
            compartments,
            strata={
                "age": str(age_stratum)
            },
            save_results=True,
        )
    return model

def set_starting_conditions(model: CompartmentalModel, num_infectious):
    """Set starting condition for the model

    Args:
        model (CompartmentalModel): the model
        num_infectious: number of infectious seed
    """
    start_pop = Parameter("start_population_size")
    init_pop = {
        "infectious": num_infectious,
        "susceptible": start_pop - num_infectious,
    }
    # Assign to the model
    model.set_initial_population(init_pop)


def add_entry_flow(model: CompartmentalModel):
    """Add birth flow to the model

    Args:
        model (CompartmentalModel): the model
    """
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
    """Add natural death flow to the model

    Args:
        model (CompartmentalModel): the model
    """
    process = "universal_death"
    universal_death_rate = 1.0  # later adjusted by age stratification
    model.add_universal_death_flows(process, death_rate=universal_death_rate)

def add_infection(model: CompartmentalModel):
    """Add infectious flow to the model

    Args:
        model (CompartmentalModel): The modeel
    """
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
        0.2, # unstratified
        origin,
        destination,
    )


def add_infect_death(model: CompartmentalModel) -> str:
    process = "infect_death"
    origin = "infectious"
    model.add_death_flow(
        process,
        0.2, # unstratified
        origin,
    )
    

def get_age_strat(
    compartments,
    infectious,
    age_strata,
    fixed_params,
):
    strat = AgeStratification("age", age_strata, compartments)
    universal_death_funcs, death_adjs = {}, {}
    death_df = process_death_rate(age_strata)
    for age in age_strata:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(
            death_df.index, death_df[age]
        )
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)
    # Set age-specific latency rate
    for flow_name, latency_params in fixed_params["age_latency"].items():
        adjs = {
            str(t): Multiply(latency_params[max([k for k in latency_params if k <= t])])
            for t in age_strata
        }
        strat.set_flow_adjustments(flow_name, adjs)

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
