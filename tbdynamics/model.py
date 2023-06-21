from datetime import datetime, timedelta
from summer2 import AgeStratification, Overwrite, Multiply
from jax import numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput, Function, Time

from .inputs import *
from .utils import *

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""

def build_base_model(
    compartments: list,
    infectious_compartments,
    start_date,
    end_date,
    time_step,
) -> tuple:
    """
    Args:
        ref_date: Arbitrary reference date
        compartments: Starting unstratified compartments
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        Simple model starting point for extension through the following functions
        with text description of the process.
    """
    model = CompartmentalModel(
        times=(start_date, end_date),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=time_step,
    )

    description = f"The base model consists of {len(compartments)} states, " \
        f"representing the following states: {', '.join(compartments)}. " \
        f"Only the {infectious_compartments} compartment contributes to the force of infection. " \
        f"The model is run from {str(start_date)} to {str(end_date)}. "
    
    return model, description

def get_pop_data():
    pop_data = load_pop_data()
    description = f"For estimates of the Camau" 

def set_starting_conditions(
    model,
    start_population_size,
    seed,
) -> str:
    init_pop = {
        "infectious": seed,
        "susceptible": start_population_size - seed,
    }

    # Assign to the model
    model.set_initial_population(init_pop)
    return f"The simulation starts with {str(round(start_population_size / 1e6, 3))} million fully susceptible persons, " \
        "with infectious persons introduced later through strain seeding as described below. "

def add_infection(
    model: CompartmentalModel,
) -> tuple:
    """
    Args:
        model: Working compartmental model

    Returns:
        Description of process added
    """
    process = "infection"
    origin = "susceptible"
    destination = "early_latent"
    model.add_infection_frequency_flow(process, Parameter("contact_rate"), origin, destination)
    des1 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "


    process= "infection_from_latent"
    origin = "late_latent"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_latent") ,
        "late_latent",
        "early_latent",
    )
    des2 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    process = "infection_from_recovered"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_recovered"),
        "recovered",
        "early_latent",
    )
    des3 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    return des1, des2, des3

def add_latency(
    model: CompartmentalModel,
) -> tuple:
    stabilisation_rate = 1.0
    process =  "stabilisation"
    origin = "susceptible"
    destination = "early_latent"
    model.add_transition_flow(
        process,
        stabilisation_rate,
        origin,
        destination,
    )
    des1 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    early_activation_rate = 1.0
    process = "early_activation"
    origin = "susceptible"
    destination = "early_latent"
    model.add_transition_flow(
        process,
        early_activation_rate,
        origin,
        destination,
    )
    des2 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    process = "late_activation"
    origin = "late_latent"
    destination = "infectious"
    model.add_transition_flow(
        process,
        Parameter("progression_multiplier"),  # Set to the adjuster, rather than one
        origin,
        destination,
    )
    des3 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    return des1, des2, des3

def add_detection(
        model: CompartmentalModel,
):
    detection_rate = 1.0 # later adjusted by organ
    process = "early_activation"
    origin = "infectious"
    destination = "on_treatment"
    model.add_transition_flow(
        process,
        detection_rate,
        origin,
        destination,
    )
    return f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

def add_treatment(
        model : CompartmentalModel
) -> tuple:
    #Treatment recovery, releapse, death flows.
    treatment_recovery_rate = 1.0 # will be adjusted later
    process = "early_activation"
    origin = "on_treatment"
    destination = "recovered"
    model.add_transition_flow(
        "early_activation",
        treatment_recovery_rate,
        origin,
        destination,
    )
    des1 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    treatment_death_rate = 1.0
    process = "death_flow"
    origin = "on_treatment"
    model.add_death_flow(
        process,
        treatment_death_rate,
        "on_treatment",
    )
    des2 = f"The {process} process moves people from the {origin} " \
        f"compartment to the death, " \
        "under the frequency-dependent transmission assumption. "
    
    relapse_rate = 1.0
    process = "early_activation"
    origin = "on_treatment"
    destination = "infectious"
    model.add_transition_flow(
        "relapse",
        relapse_rate,
        "on_treatment",
        "infectious",
    )
    des3 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "
    
    return des1, des2, des3

def add_entry_flow(
        model: CompartmentalModel
):
    process = "birth"
    birth_rates = load_pop_data()[1]
    birth_rates.iloc[:,1] /= 1000.0  # Birth rates are provided / 1000 population
    crude_birth_rate = get_sigmoidal_interpolation_function(birth_rates.iloc[:,0], birth_rates.iloc[:,1])
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        "susceptible",
    )
    return f"The {process} process add newborns to the model " \
        
    

def add_natural_death_flow(
    model: CompartmentalModel 
):
    universal_death_rate = 1.0
    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)

def add_infect_death(
    model: CompartmentalModel 
):
    model.add_death_flow(
        "infect_death",
        Parameter("infect_death_rate_unstratified"),
        "infectious",
    )

def implement_acf(
    model: CompartmentalModel,
    time_variant_screening_rate,
    acf_screening_sensitivity    
):
            # Universal active case funding is applied
    times = list(Parameter(time_variant_screening_rate))
    vals = [
                v * Parameter(acf_screening_sensitivity)
                for v in Parameter(time_variant_screening_rate)
            ]
    acf_detection_rate = get_sigmoidal_interpolation_function(times, vals)

    model.add_transition_flow(
        "acf_detection",
        acf_detection_rate,
        "infectious",
        "on_treatment",
    )
     
def add_age_strat(
    compartments,
    age_strata,
    matrix
) -> tuple:
    age_strata = Parameter("age_breakpoints")
    strat = AgeStratification("age", age_strata, compartments)
    strat.set_mixing_matrix(matrix)
    universal_death_funcs, death_adjs = {}, {}
    for age in age_strata:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(death_rate_years, death_rates_by_age[age])
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)

def build_polymod_matrix(
        age_strata
):
    values = [[ 398.43289672,  261.82020387,  643.68286218,  401.62199159,
          356.13449939],
        [ 165.78966683,  881.63067677,  532.84120554,  550.75979227,
          285.62836724],
        [ 231.75164317,  311.38983781,  915.52884268,  673.30894113,
          664.14577066],
        [ 141.94492435,  310.88835505,  786.13676958, 1134.31076003,
          938.03403291],
        [  67.30073632,  170.46333134,  647.30153978, 1018.81243422,
         1763.57657715]]
    matrix = np.array(values).T
    matrix_fig = px.imshow(matrix, x=age_strata, y=age_strata)
    return matrix, matrix_fig

