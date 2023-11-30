from jax import numpy as jnp
from pathlib import Path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput, Function, Time
from summer2 import AgeStratification, Overwrite, Multiply, Stratification
from tbdynamics.utils import triangle_wave_func
from emutools.tex import StandardTexDoc

from .inputs import get_birth_rate, process_death_rate
from .utils import (
    get_average_age_for_bcg,
    get_average_sigmoid,
    get_latency_with_diabetes,
    get_treatment_outcomes,
    tanh_based_scaleup,
    bcg_multiplier_func,
    replace_underscore_with_space,
)
from .outputs import (
    request_aggregation_output,
    request_flow_output,
    request_normalise_flow_output,
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
    tex_doc: StandardTexDoc,
    organ_strat=True,
    add_triangular=False
):
    model = build_base_model(
        compartments, infectious_compartments, time_start, time_end, time_step, tex_doc
    )
    if add_triangular:
        set_starting_conditions(model,0, tex_doc)
        seed_infectious(model,age_strata) # I set infectious seed here by injecting the triangular function
    else:
        set_starting_conditions(model,1, tex_doc)
    add_entry_flow(model, tex_doc)
    add_natural_death_flow(model, tex_doc)
    add_infection(model, tex_doc)
    add_latency(model, tex_doc)
    #add_detection(model, tex_doc)
    #add_treatment_related_outcomes(model, tex_doc)
    add_self_recovery(model, tex_doc)
    # add_acf(model, fixed_params, tex_doc)
    add_infect_death(model, tex_doc)
    # add_acf(model, fixed_params, tex_doc)
    age_strat = get_age_strat(
        compartments, infectious_compartments, age_strata, fixed_params, matrix, tex_doc
    )
    model.stratify_with(age_strat)
    if organ_strat:
        organ_strat = get_organ_strat(fixed_params, infectious_compartments, tex_doc)
        model.stratify_with(organ_strat)
    #seed_infectious(model, 'infectious', age_strata)
    request_output(
        model, age_strata, compartments, latent_compartments, infectious_compartments
    )
    return model


def build_base_model(
    compartments: list,
    infectious_compartments: list,
    time_start,
    time_end,
    time_step,
    tex_doc: StandardTexDoc,
):
    model = CompartmentalModel(
        times=(time_start, time_end),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=time_step,
    )
    desc = (
        f"The base model consists of {len(compartments)} states, "
        f'representing the following states: {", ".join(compartments).replace("_", "")}. '
        f"Each of the infectious compartments contribute equally to the force of infection. \n"
    )

    time_desc = f"A simulation is run from {time_start} to {time_end}"
    tex_doc.add_line(desc, "Model Structure")
    tex_doc.add_line(time_desc, "Population")

    return model


def set_starting_conditions(model, num_infectious, tex_doc: StandardTexDoc):
    start_pop = Parameter("start_population_size")
    init_pop = {
        'infectious': num_infectious,
        'susceptible': start_pop - num_infectious,
    }

    # Assign to the model
    model.set_initial_population(init_pop)
    desc = (
        f"The simulation starts with {start_pop} million fully susceptible persons, "
        "with infectious persons introduced later through strain seeding as described below. "
    )
    tex_doc.add_line(desc, "Population")


def add_entry_flow(model: CompartmentalModel, tex_doc: StandardTexDoc):
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
    desc = f"The {process} process add newborns to the {destination} compartment of the model"
    tex_doc.add_line(desc, "Model Structure")


def add_natural_death_flow(model: CompartmentalModel, tex_doc: StandardTexDoc):
    process = "universal_death"
    universal_death_rate = 1.0
    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)
    desc = f"The {replace_underscore_with_space(process)} process add universal death to the model."
    tex_doc.add_line(desc, "Model Structure")


def add_infection(
    model: CompartmentalModel, tex_doc: StandardTexDoc
):
    """
    Args:
        model: Working compartmental model

    Returns:

    """
    # seed_args = [Time, Parameter('seed_time'), 0.1, 1.0]
    # voc_seed_func = Function(triangle_wave_func, seed_args)
    # model.add_importation_flow("seeding_infection",voc_seed_func,latent_compartments,split_imports=False) # Set seed at time

    process = "infection"
    origin = "susceptible"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process, Parameter("contact_rate"), origin, destination
    )
    desc1 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the {replace_underscore_with_space(destination)} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc1, "Model Structure")

    process = "infection_from_latent"
    origin = "late_latent"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_latent"),
        "late_latent",
        "early_latent",
    )
    desc2 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the {replace_underscore_with_space(destination)} compartment, "
        "under the frequency-dependent transmission assumption. "
    )

    tex_doc.add_line(desc2, "Model Structure")

    process = "infection_from_recovered"
    origin = "recovered"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_recovered"),
        origin,
        destination,
    )
    desc3 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {origin} "
        f"compartment to the {replace_underscore_with_space(destination)} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc3, "Model Structure")


def add_latency(model: CompartmentalModel, tex_doc: StandardTexDoc):
    # add stabilization process 
    process = "stabilisation"
    origin = "early_latent"
    destination = "late_latent"
    model.add_transition_flow(
        process,
        1, # later adjusted by age group
        origin,
        destination,
    )
    desc1 = (
        f"The {process} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the {replace_underscore_with_space(destination)} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc1, "Model Structure")
    # Add the early activattion process
    process = "early_activation"
    origin = "early_latent"
    destination = "infectious"
    model.add_transition_flow(
        process,
        1.0, # later adjusted by age group
        origin,
        destination,
    )
    desc2 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the {destination} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc2, "Model Structure")

    process = "late_activation"
    origin = "late_latent"
    destination = "infectious"
    model.add_transition_flow(
        process,
        Parameter("progression_multiplier"),  # Set to the adjuster, rather than one
        origin,
        destination,
    )
    desc3 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the {destination} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc3, "Model Structure")

    # return des1, des2, des3


def add_detection(model: CompartmentalModel, tex_doc: StandardTexDoc) -> str:
    detection_rate = 1.0  # later adjusted by organ
    process = "detection"
    origin = "infectious"
    destination = "on_treatment"
    model.add_transition_flow(
        process,
        detection_rate,
        origin,
        destination,
    )
    desc = (
        f"The {process} process moves people from the {origin} "
        f"compartment to the {replace_underscore_with_space(destination)} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc, "Model Structure")


def add_treatment_related_outcomes(
    model: CompartmentalModel,
    tex_doc: StandardTexDoc,
):
    # Treatment recovery, releapse, death flows.
    treatment_recovery_rate = 1.0  #  later adjusted by organ
    process = "treatment_recovery"
    origin = "on_treatment"
    destination = "recovered"
    model.add_transition_flow(
        process,
        treatment_recovery_rate,
        origin,
        destination,
    )
    desc1 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the {destination} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc1, "Model Structure")
    treatment_death_rate = 1.0  #  later adjusted by age
    process = "treatment_death"
    origin = "on_treatment"
    model.add_death_flow(
        process,
        treatment_death_rate,
        "on_treatment",
    )
    desc2 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the death, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc2, "Model Structure")

    relapse_rate = 1.0  #  later adjusted by age
    process = "early_activation"
    origin = "on_treatment"
    destination = "infectious"
    model.add_transition_flow(
        "relapse",
        relapse_rate,
        "on_treatment",
        "infectious",
    )
    desc3 = (
        f"The {replace_underscore_with_space(process)} process moves people from the {replace_underscore_with_space(origin)} "
        f"compartment to the {destination} compartment, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc3, "Model Structure")


def add_self_recovery(model: CompartmentalModel, tex_doc: StandardTexDoc):
    process = "self_recovery"
    origin = "infectious"
    destination = "recovered"
    model.add_transition_flow(
        "self_recovery",
        .2,
        origin,
        destination,
    )
    desc = (
        f"The {replace_underscore_with_space(process)} process moves people from the {origin} "
        f"compartment to the {destination}, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc, "Model Structure")

def add_infect_death(model: CompartmentalModel, tex_doc: StandardTexDoc) -> str:
    process = "infect_death"
    origin = "infectious"
    model.add_death_flow(
        "infect_death",
        0.2,
        "infectious",
    )
    desc = f"The {replace_underscore_with_space(process)} process moves people from the {origin}"
    tex_doc.add_line(desc, "Model Structure")


def add_acf(model: CompartmentalModel, fixed_params, tex_doc: StandardTexDoc):
    # Universal active case funding is applied
    times = list(fixed_params["time_variant_screening_rate"])
    vals = [
        v * fixed_params["acf_screening_sensitivity"]
        for v in fixed_params["time_variant_screening_rate"].values()
    ]

    acf_detection_rate = get_sigmoidal_interpolation_function(times, vals)

    process = "acf_detection"
    origin = "infectious"
    destination = "on_treatment"
    model.add_transition_flow(
        process,
        acf_detection_rate,
        origin,
        destination,
    )

    desc = (
        f"The {replace_underscore_with_space(process)} process moves people from the {origin} "
        f"compartment to the {replace_underscore_with_space(destination)}, "
        "under the frequency-dependent transmission assumption. "
    )
    tex_doc.add_line(desc, "Model Structure")


def get_age_strat(
    compartments,
    infectious,
    age_strata,
    fixed_params,
    matrix,
    tex_doc: StandardTexDoc,
) -> str:
    strat = AgeStratification('age', age_strata, compartments)
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
    # Set age-specific late activation rate
    for flow_name, latency_params in fixed_params['age_latency'].items():
        #is_activation_flow = flow_name in ["late_activation"] 
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
    inf_switch_age = fixed_params['age_infectiousness_switch']
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

    # Get the time-varying treatment success proportions
    time_variant_tsr = get_sigmoidal_interpolation_function(
        list(fixed_params["time_variant_tsr"].keys()),
        list(fixed_params["time_variant_tsr"].values()),
    )
        # Get the treatment outcomes, using the get_treatment_outcomes function above and apply to model
    # treatment_recovery_funcs, treatment_death_funcs, treatment_relapse_funcs = (
    #         {},
    #         {},
    #         {},
    #     )
    # for age in age_strata:
    #     death_rate = universal_death_funcs[age]
    #     treatment_outcomes = Function(
    #         get_treatment_outcomes,
    #             [
    #                 fixed_params["treatment_duration"],
    #                 fixed_params["prop_death_among_negative_tx_outcome"],
    #                 death_rate,
    #                 time_variant_tsr,
    #             ],
    #         )
    #     treatment_recovery_funcs[str(age)] = Multiply(treatment_outcomes[0])
    #     treatment_death_funcs[str(age)] = Multiply(treatment_outcomes[1])
    #     treatment_relapse_funcs[str(age)] = Multiply(treatment_outcomes[2])
    # strat.set_flow_adjustments("treatment_recovery", treatment_recovery_funcs)
    # strat.set_flow_adjustments("treatment_death", treatment_death_funcs)
    # strat.set_flow_adjustments("relapse", treatment_relapse_funcs)

        # Add BCG effect without stratifying for BCG
        # bcg_multiplier_dict = {
        #     "0": 0.3,
        #     "5": 0.3,
        #     "15": 0.7375,
        #     "35": 1.0,
        #     "50": 1.0,
        #     "70": 1.0,
        # }  # Ragonnet et al. (IJE, 2020)
        # bcg_adjs = {}
        # for age, multiplier in bcg_multiplier_dict.items():
        #     if multiplier < 1.0:
        #         bcg_adjs[str(age)] = Multiply(
        #             Function(
        #                 bcg_multiplier_func,
        #                 [
        #                     get_sigmoidal_interpolation_function(
        #                         list(fixed_params["time_variant_bcg_perc"].keys()),
        #                         list(fixed_params["time_variant_bcg_perc"].values()),
        #                         Time - get_average_age_for_bcg(age, age_strata),
        #                     ),
        #                     multiplier,
        #                 ],
        #             )
        #         )
        #     else:
        #         bcg_adjs[str(age)] = None

        # strat.set_flow_adjustments("infection", bcg_adjs)
    desc = (
        "The age stratification adjusts following process: (1) The universal death by age group. The data was taken from autumn's database."
        " (2) The early and late activation"
        "(3) Age infectioness switched at age of 15"
        "(4) Infectiousness multiplier for treatment"
        "(5) Treatment outcomes: relapse, recovery and death"
    )
    tex_doc.add_line(desc, "Stratification", subsection="Age")
    return strat


def get_organ_strat(
    fixed_params: dict,
    infectious_compartments: list,
    tex_doc: StandardTexDoc,
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

    # Define different detection rates by organ status.
    detection_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        # adj_vals = sensitivity[organ_stratum]
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = (
            Function(
                tanh_based_scaleup,
                [
                    Time,
                    Parameter("acf_scaleup_shape"),
                    1990,
                    Parameter("acf_start_asymp"),
                    1.0,
                ],
            )
            * fixed_params[param_name]
        )

        # detection_adjs[organ_stratum] = Function(tanh_based_scaleup,
        #                                          [Time, .1,
        #                                           2000,
        #                                           0.,
        #                                           0.4]) * fixed_params[param_name]

    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    strat.set_flow_adjustments("detection", detection_adjs)

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

    desc = (
        "The age stratification adjusts following:"
        "(1) Infectiousness adjustment by organ status"
        "(2) Different natural history (infection death) by organ status"
        "(3) Different detection rates by organ status"
        "(4) The progression rates by organ using the requested incidence proportions"
    )
    tex_doc.add_line(desc, "Stratification", subsection="Age")

    return strat


def get_gender_strat(age_strata, compartments, fixed_params, tex_doc: StandardTexDoc):
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

    desc = "This is stratification for gender. We adjust the infection flow, do not adjust the birth flow for those who are younger than 15."
    tex_doc.add_line(desc, "Stratification", subsection="Gender")
    return strat


def seed_infectious(
    model: CompartmentalModel,
    age_strata
):
    """Seed infectious.

    Args:
        model: The summer epidemiological model
        latent_compartments: The names of the latent compartments

    """

    seed_time = 'seed_time'
    seed_duration = 'seed_duration'
    seed_rate = 'seed_rate'
    seed_args = [Time, Parameter(seed_time), Parameter(seed_duration), Parameter(seed_rate)]
    voc_seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        'seed_infectious',
        voc_seed_func,
        'infectious',
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
            strata={
                "age": str(age_stratum)
            },  
            save_results=save_results,
        )
