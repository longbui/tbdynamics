from summer2 import CompartmentalModel
from typing import List
from summer2.functions.time import get_piecewise_function, get_linear_interpolation_function
from summer2.parameters import Function, Parameter, Time, DerivedOutput
from tbdynamics.utils import tanh_based_scaleup
import numpy as np


def request_model_outputs(
    model: CompartmentalModel,
    compartments: List[str],
    latent_compartments: List[str],
    infectious_compartments: List[str],
    age_strata: List[int],
    organ_strata: List[str],
):
    """
    Requests various model outputs

    Args:
        model: The compartmental model from which outputs are requested.
        compartments: A list of all compartment names in the model.
        latent_compartments: A list of latent compartment names.
        infectious_compartments: A list of infectious compartment names.
        age_strata: A list of age groups used for stratification.
        organ_strata: A list of organ strata used for stratification.
    """
    # Request total population size
    model.request_output_for_compartments("total_population", compartments)

    # Calculate and request percentage of latent population
    model.request_output_for_compartments(
        "latent_population_size", latent_compartments
    )
    model.request_function_output(
        "percentage_latent", 100.0 * DerivedOutput("latent_population_size") / DerivedOutput("total_population")
    )
    #Death
    model.request_output_for_flow("mortality_infectious_raw", "infect_death", save_results=False)
    model.request_output_for_flow("mortality_on_treatment_raw", "treatment_death", save_results=False)
    model.request_aggregate_output("mortality_raw", ["mortality_infectious_raw", "mortality_on_treatment_raw"])
    model.request_function_output("mortality", 1e5 * DerivedOutput("mortality_raw") / DerivedOutput("total_population"))

    # Calculate and request prevalence of pulmonary
    for organ_stratum in organ_strata:
        model.request_output_for_compartments(
            f"infectious_size_X{organ_stratum}",
            infectious_compartments,
            strata={"organ": organ_stratum},
            save_results=False,
        )
    pulmonary_outputs = [
        f"infectious_size_X{organ_stratum}"
        for organ_stratum in ["smear_positive", "smear_negative"]
    ]
    model.request_aggregate_output(
        "pulmonary_pop_size", pulmonary_outputs
    )
    model.request_function_output(
        "prevalence_pulmonary", 1e5 * DerivedOutput("pulmonary_pop_size") / DerivedOutput("total_population")
    )
    # total prevalence
    model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments
    )
    model.request_function_output(
        "prevalence_infectious", 1e5 * DerivedOutput("infectious_population_size") / DerivedOutput("total_population")
    )

    # incidence
    model.request_output_for_flow(
        "incidence_early_raw", "early_activation", save_results=False
    )
    model.request_output_for_flow(
        "incidence_late_raw", "late_activation", save_results=False
    )

    incidence_raw = model.request_aggregate_output(
        "incidence_raw",
        ["incidence_early_raw", "incidence_late_raw"],
        save_results=False,
    )
    model.request_function_output("incidence", 1e5 * incidence_raw / DerivedOutput("total_population"))

    # notification
    model.request_output_for_flow("notification", "detection", save_results=True)

    # Request proportion of each compartment in the total population
    for compartment in compartments:
        model.request_output_for_compartments(
            f"number_{compartment}", compartment
        )
        model.request_function_output(
            f"prop_{compartment}", DerivedOutput(f"number_{compartment}") / DerivedOutput("total_population")
        )

    # Request total population by age stratum
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )
    for organ_stratum in organ_strata:
        model.request_output_for_compartments(
            f"total_populationXorgan_{organ_stratum}",
            compartments,
            strata={"organ": str(organ_stratum)},
        )
        model.request_function_output(
            f"prop_{organ_stratum}", DerivedOutput(f"total_populationXorgan_{organ_stratum}") / DerivedOutput("infectious_population_size")
        )


def request_cdr(model):
    detection_func = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("screening_scaleup_shape"),
            Parameter("screening_inflection_time"),
            0.0,
            Parameter("screening_end_asymp"),
        ],
    )
    detection_covid_reduction = get_linear_interpolation_function([2020, 2021, 2021.9], [1.0, Parameter("detection_reduction"),1.0])
    cdr_covid_adjusted = get_piecewise_function([2020, 2022], [detection_func, detection_func * detection_covid_reduction, detection_func])

    model.add_computed_value_func("cdr", cdr_covid_adjusted)
    model.request_computed_value_output("cdr")
