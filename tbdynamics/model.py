from datetime import datetime, timedelta
from summer2 import AgeStratification, Overwrite, Multiply
from jax import numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2.functions.interpolate import build_sigmoidal_multicurve
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput, Function, Time

from .inputs import *
from .utils import *
from .outputs import *

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
) -> str:
    start_pop = Parameter("start_population_size") 
    seed = Parameter("infectious_seed")
    init_pop = {
        "infectious": seed,
        "susceptible": start_pop - seed,
    }

    # Assign to the model
    model.set_initial_population(init_pop)
    return f"The simulation starts with {start_pop} million fully susceptible persons, " \
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
    origin = "recovered"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_recovered"),
        origin,
        destination,
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
    origin = "early_latent"
    destination = "late_latent"
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
    origin = "early_latent"
    destination = "infectious"
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
    process = "detection"
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

def add_treatment_related_outcomes(
    model : CompartmentalModel
) -> tuple:
    #Treatment recovery, releapse, death flows.
    treatment_recovery_rate = 1.0 # will be adjusted later
    process = "treatment_recovery"
    origin = "on_treatment"
    destination = "recovered"
    model.add_transition_flow(
        process,
        treatment_recovery_rate,
        origin,
        destination,
    )
    des1 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    treatment_death_rate = 1.0
    process = "treatment_death"
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
    crude_birth_rate = get_sigmoidal_interpolation_function(birth_rates.iloc[:,0], birth_rates.iloc[:,1])
    print(crude_birth_rate)
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        "susceptible",
    )
    return f"The {process} process add newborns to the model"
        

def add_natural_death_flow(
    model: CompartmentalModel 
):
    universal_death_rate = 0.21
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
    infectious,
    age_strata,
    matrix,
    params
):
    strat = AgeStratification("age", age_strata, compartments)
    strat.set_mixing_matrix(matrix)
    universal_death_funcs, death_adjs = {}, {}
    for age in age_strata:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(death_rate_years, death_rates_by_age[age])
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)
    for flow_name, latency_params in params['age_stratification'].items():
        #is_activation_flow = flow_name in ["late_activation"]
        #if is_activation_flow:
        adjs = {str(t): Multiply(latency_params[max([k for k in latency_params if k <= t])]) for t in age_strata}
        strat.set_flow_adjustments(flow_name, adjs)

    inf_switch_age = params['age_infectiousness_switch']
    for comp in infectious:
        inf_adjs = {}
        for i, age_low in enumerate(age_strata):
            infectiousness = 1.0 if age_low == age_strata[-1] else get_average_sigmoid(age_low, age_strata[i + 1], inf_switch_age)

            # Infectiousness multiplier for treatment (ideally move to model.py, but has to be set in stratification with current summer)
            if comp == 'on_treatment':
                infectiousness *= Parameter("on_treatment_infect_multiplier")

            inf_adjs[str(age_low)] = Multiply(infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    # Get the time-varying treatment success proportions
    time_variant_tsr = get_sigmoidal_interpolation_function(
            list(params['time_variant_tsr'].keys()), list(params['time_variant_tsr'].values())
        )
     

    # Get the treatment outcomes, using the get_treatment_outcomes function above and apply to model
    treatment_recovery_funcs, treatment_death_funcs, treatment_relapse_funcs = {}, {}, {}
    for age in age_strata:
        death_rate = universal_death_funcs[age]
        treatment_outcomes = Function(
            get_treatment_outcomes,
            [
                params['treatment_duration'],
                params['prop_death_among_negative_tx_outcome'],
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

    #Add BCG effect without stratifying for BCG
    bcg_multiplier_dict = {'0': 0.3, '5': 0.3, '15': 0.7375, '35': 1.0, '50': 1.0} # Ragonnet et al. (IJE, 2020)
    bcg_adjs = {}
    for age, multiplier in bcg_multiplier_dict.items():
        if multiplier < 1.0:
            bcg_adjs[str(age)] = Multiply(
                Function(
                bcg_multiplier_func, 
                [
                get_sigmoidal_interpolation_function(
                                                    list(params['time_variant_bcg_perc'].keys()),
                                                    list(params['time_variant_bcg_perc'].values()), 
                                                    Time - get_average_age_for_bcg(age, age_strata)
                                                ),
                multiplier
                ])
            )
        else:
            bcg_adjs[str(age)] = None


    strat.set_flow_adjustments("infection", bcg_adjs)

    return strat

def build_contact_matrix():
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
    return matrix

def request_output(
        model: CompartmentalModel,
        compartments,
        latent_compartments,
        infectious_compartments
):
    """
    Get the applicable outputs
    """
    model.request_output_for_compartments(
        "total_population", compartments, save_results=True
    )
    model.request_output_for_compartments(
        "latent_population_size", latent_compartments, save_results=True
    )
    # latency
    model.request_function_output("percentage_latent", 100.0 * DerivedOutput("latent_population_size") / DerivedOutput("total_population"))

    # Prevalence
    model.request_output_for_compartments("infectious_population_size", infectious_compartments, save_results=True)
    model.request_function_output("prevalence_infectious", 1e5 * DerivedOutput("infectious_population_size") / DerivedOutput("total_population"),)

    # Death
    model.request_output_for_flow("mortality_infectious_raw", "infect_death", save_results=True)

    sources = ["mortality_infectious_raw"]
    request_aggregation_output(model, "mortality_raw", sources, save_results=False)
    model.request_cumulative_output("cumulative_deaths", "mortality_raw", start_time=2000)

    # Disease incidence
    request_flow_output(model,"incidence_early_raw", "early_activation", save_results=False)
    request_flow_output(model,"incidence_late_raw", "late_activation", save_results=False)
    sources = ["incidence_early_raw", "incidence_late_raw"]
    request_aggregation_output(model,"incidence_raw", sources, save_results=False)
    model.request_cumulative_output("cumulative_diseased", "incidence_raw", start_time=2000)

    # Normalise incidence so that it is per unit time (year), not per timestep
    request_normalise_flow_output(model, "incidence_early", "incidence_early_raw")
    request_normalise_flow_output(model, "incidence_late", "incidence_late_raw")
    request_normalise_flow_output(model, "incidence_norm", "incidence_raw", save_results=False)
    model.request_function_output(
        "incidence", 1e5 * DerivedOutput("incidence_norm") / DerivedOutput("total_population")
    )
    request_flow_output(model, "passive_notifications_raw", "detection", save_results=False)
    # request notifications
    request_normalise_flow_output(model, "notifications", "passive_notifications_raw")



