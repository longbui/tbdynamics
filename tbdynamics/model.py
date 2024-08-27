from pathlib import Path
import numpy as np
from typing import List, Dict
from summer2 import CompartmentalModel
from summer2.functions.time import (
    get_sigmoidal_interpolation_function,
    get_linear_interpolation_function,
)
from summer2.parameters import Parameter, Function, Time

from .utils import triangle_wave_func
from .inputs import get_birth_rate, get_death_rate, process_death_rate
from .constants import organ_strata
from .outputs import request_model_outputs
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
    covid_effects: Dict[str, bool],
    improved_detection_multiplier: float = None,
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
        times=(fixed_params["time_start"], fixed_params["time_end"]),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=fixed_params["time_step"],
    )

    birth_rates = get_birth_rate()
    death_rates = get_death_rate()
    death_df = process_death_rate(death_rates, age_strata, birth_rates.index)
    model.set_initial_population({"susceptible": Parameter("start_population_size")})
    seed_infectious(model)
    # add birth flow
    crude_birth_rate = get_sigmoidal_interpolation_function(
        birth_rates.index, birth_rates.values
    )
    model.add_crude_birth_flow("birth", crude_birth_rate, "susceptible")
    # Add natural death flow
    model.add_universal_death_flows(
        "universal_death", 1.0
    )  # Adjusted later by age stratification
    add_infection_flow(model, covid_effects['contact_reduction'])
    add_latency_flow(model)
    # Add self-recovery flow
    model.add_transition_flow(
        "self_recovery", 1.0, "infectious", "recovered"
    )  # later adjusted by organ status
    # Add detection
    model.add_transition_flow(
        "detection", 1.0, "infectious", "on_treatment"
    )  # will be adjusted later
    add_treatment_related_outcomes(model)
    # Add infect death flow
    model.add_death_flow(
        "infect_death", 1.0, "infectious"
    )  # later adjusted by organ status
    age_strat = get_age_strat(
        compartments,
        infectious_compartments,
        age_strata,
        death_df,
        fixed_params,
        matrix,
    )
    model.stratify_with(age_strat)
    organ_strat = get_organ_strat(infectious_compartments, organ_strata, fixed_params, covid_effects['detection_reduction'], improved_detection_multiplier)
    model.stratify_with(organ_strat)
    request_model_outputs(
        model,
        compartments,
        latent_compartments,
        infectious_compartments,
        age_strata,
        organ_strata,
        covid_effects['detection_reduction']
    )
    return model


def add_infection_flow(model: CompartmentalModel, contact_reduction):
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
    contact_rate = Parameter("contact_rate") * (
        get_linear_interpolation_function(
            [2020.0, 2021.0, 2022], [1.0, 1 - Parameter("contact_reduction"), 1.0]
        )
        if contact_reduction
        else 1.0
    )
    for origin, modifier in infection_flows:
        process = f"infection_from_{origin}"
        modifier = Parameter(modifier) if modifier else 1.0
        flow_rate = contact_rate * modifier
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


def add_treatment_related_outcomes(model: CompartmentalModel) -> None:
    """
    Adds treatment-related outcome flows to the compartmental model. This includes flows for treatment recovery,
    treatment-related death, and relapse. Initial rates are set as placeholders, with the expectation that
    they may be adjusted later based on specific factors such as organ involved or patient age.
    """

    treatment_outcomes_flows = [
        ("treatment_recovery", 1.0, "recovered"),  # later adjusted by age
        ("relapse", 1.0, "infectious"),
    ]

    # Add each transition flow defined in treatment_flows
    for flow_name, rate, to_compartment in treatment_outcomes_flows:
        model.add_transition_flow(
            flow_name,
            rate,  # Directly using the rate for now
            "on_treatment",
            to_compartment,
        )

    # Define and add treatment death flow separately since it uses a different method
    treatment_death_flow = ["treatment_death", 1.0, "on_treatment"]
    model.add_death_flow(*treatment_death_flow)


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
