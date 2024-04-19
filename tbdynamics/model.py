from pathlib import Path
import numpy as np
from typing import List, Dict
from summer2 import CompartmentalModel
from summer2.functions.time import get_sigmoidal_interpolation_function, get_linear_interpolation_function
from summer2.parameters import Parameter, Function, Time

from .utils import triangle_wave_func
from .inputs import get_birth_rate, get_death_rate, process_death_rate
from .constants import organ_strata
from .outputs import request_model_outputs, request_cdr
from .strats import get_age_strat, get_organ_strat


BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


def build_model(
    compartments: List[str],
    latent_compartments: List[str],
    infectious_compartments: List[str],
    age_strata: List[int],
    fixed_params: Dict[str, any],
    matrix,
) -> CompartmentalModel:
    """
    Builds and returns a compartmental model for epidemiological studies, incorporating
    various flows and stratifications based on age, organ status, and treatment outcomes.

    Args:
        compartments: List of compartment names in the model.
        latent_compartments: List of latent compartment names.
        infectious_compartments: List of infectious compartment names.
        age_strata: List of age groups for stratification.
        time_start: Start time for the model simulation.
        time_end: End time for the model simulation.
        time_step: Time step for the model simulation.
        fixed_params: Dictionary of parameters with fixed values.
        matrix: Mixing matrix for age stratification.

    Returns:
        A configured CompartmentalModel object.
    """
    model = CompartmentalModel(
        times=(fixed_params['time_start'], fixed_params['time_end']),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=fixed_params['time_step'],
    )

    birth_rates = get_birth_rate()
    death_rates = get_death_rate()
    death_df = process_death_rate(death_rates, age_strata, birth_rates.index)
    model.set_initial_population({"susceptible": Parameter("start_population_size")})
    seed_infectious(model)
    add_entry_flow(model, birth_rates)
    add_natural_death_flow(model)
    add_infection_flow(model)
    add_latency_flow(model)
    add_infect_death_flow(model)
    add_self_recovery_flow(model)
    add_detection(model)
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
    request_cdr(model)
    return model


def add_entry_flow(model: CompartmentalModel, birth_rates: Dict):
    """
    Adds a crude birth flow to the model based on given birth rates.

    Args:
        model: The compartmental model to add the flow to.
        birth_rates: A dictionary containing birth rates data.
    """
    process = "birth"
    crude_birth_rate = get_sigmoidal_interpolation_function(
        birth_rates.index, birth_rates.values
    )
    model.add_crude_birth_flow(process, crude_birth_rate, "susceptible")


def add_natural_death_flow(model: CompartmentalModel):
    """
    Adds a universal death flow to the model, to be adjusted later by age stratification.
    Args:
        model: The compartmental model to add the flow to.
    """
    model.add_universal_death_flows(
        "universal_death", 1.0
    )  # Adjusted later by age stratification


def add_infection_flow(model: CompartmentalModel):
    """
    Adds infection flows to the model, allowing for the transition of individuals from
    specific compartments (e.g., susceptible, late latent, recovered) to the early latent
    state. The transitions are defined by infection modifiers that adjust the base contact
    rate, which represents the frequency of infection transmission.

    Args:
        model: The compartmental model to which the infection flows are to be added.

    Each flow is defined by a pair (origin, modifier):
    - `origin`: The name of the compartment from which individuals will transition.
    - `modifier`: A parameter name that modifies the base contact rate for the specific flow.
      If `None`, the contact rate is used without modification.
    """
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
    """
    Adds latency flows to the compartmental model, representing the progression of the disease
    through different stages of latency. This function defines three main flows: stabilization,
    early activation, and late activation.

    - Stabilization flow represents the transition of individuals from the 'early_latent' compartment
      to the 'late_latent' compartment, indicating a period where the disease does not progress or show symptoms.

    - Early activation flow represents the transition from 'early_latent' to 'infectious', modeling the
      scenario where the disease becomes active and infectious shortly after the initial infection.

    - Late activation flow represents the transition from 'late_latent' to 'infectious', modeling the
      scenario where the disease becomes active and infectious after a longer period of latency.

    Each flow is defined with a name, a rate (set to 1.0 and will be adjusted based on empirical data or model needs), and the source and destination compartments.

    Args:
        model: The compartmental model to which the latency flows are to be added.
    """
    latency_flows = [
        ("stabilisation", 1.0, "early_latent", "late_latent"),
        ("early_activation", 1.0, "early_latent", "infectious"),
        ("late_activation", 1.0, "late_latent", "infectious"),
    ]
    for latency_flow in latency_flows:
        model.add_transition_flow(*latency_flow)


def add_self_recovery_flow(model: CompartmentalModel) -> None:
    """
    Adds a self-recovery flow to the model, enabling individuals in the 'infectious' compartment
    to recover spontaneously without medical intervention. This flow represents the natural
    recovery process of individuals who overcome the infection through their immune response.

    Args:
        model: The compartmental model to which the self-recovery flow is to be added.
    """
    model.add_transition_flow("self_recovery", 1.0, "infectious", "recovered") #later adjusted by organ status


def add_infect_death_flow(model: CompartmentalModel) -> None:
    """
    Adds an infection-induced death flow to the model, accounting for individuals in the
    'infectious' compartment who succumb to the disease. This flow represents the fatal
    progression of the infection in some individuals, leading to death.

    Args:
        model: The compartmental model to which the infect-death flow is to be added.
    """
    model.add_death_flow("infect_death", 1.0, "infectious") #later adjusted by organ status


def add_detection(model) -> None:
    """
    Adds a detection flow to the model, transitioning individuals from the 'infectious'
    compartment to the 'on_treatment' compartment based on a dynamically calculated detection rate.

    Args:
        model: The compartmental model to which the detection flow is to be added.
        fixed_params: A dictionary containing model parameters, including keys and values
                      for calculating the detection rate.
    """

    # Adding a transition flow named 'detection' to the model
    model.add_transition_flow("detection", 1.0, "infectious", "on_treatment") # will be adjusted later


def add_treatment_related_outcomes(model: CompartmentalModel) -> None:
    """
    Adds treatment-related outcome flows to the compartmental model. This includes flows for treatment recovery,
    treatment-related death, and relapse. Initial rates are set as placeholders, with the expectation that
    they may be adjusted later based on specific factors such as organ involved or patient age.
    """

    treatment__outcomes_flows = [
        ("treatment_recovery", 1.0, "recovered"), #later adjusted by organ
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
) -> None:
    """
    Applies organ-based stratification to the model, adjusting for different disease dynamics
    based on organ involvement.

    Args:
        model: The compartmental model to apply stratification to.
        infectious_compartments: List of infectious compartment names.
        organ_strata: List of organ strata for stratification.
        fixed_params: Dictionary of parameters with fixed values.
    """
    age_strat = get_age_strat(
        compartments,
        infectious_compartments,
        age_strata,
        death_df,
        fixed_params,
        matrix,
    )
    model.stratify_with(age_strat)


def stratify_model_by_organ(
    model: CompartmentalModel,
    infectious_compartments: List[str],
    organ_strata: List[str],
    fixed_params: Dict[str, any],
) -> None:
    """
    Applies organ-based stratification to the model. This stratification adjusts the model
    to account for different disease dynamics based on the specific organ involved. The
    function retrieves an organ stratification configuration via `get_organ_strat` and
    applies it to the provided model.

    Args:
        model: The compartmental model to which the stratification will be applied.
        infectious_compartments: A list of compartments within the model that are considered infectious.
        organ_strata: A list of organ strata names used for stratification, indicating different
                      disease dynamics based on the specific organ involved.
        fixed_params: A dictionary of parameters with fixed values that are used to configure
                      the stratification, such as modifiers for infectiousness or death rates
                      specific to each organ stratum.
    """
    organ_strat = get_organ_strat(infectious_compartments, organ_strata, fixed_params)
    model.stratify_with(organ_strat)


def seed_infectious(model: CompartmentalModel):
    """
    Adds an importation flow to the model to simulate the initial seeding of infectious individuals.
    This is used to introduce the disease into the population at any time of the simulation.

    Args:
        model: The compartmental model to which the infectious seed is to be added.
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
        split_imports=True,
    )

