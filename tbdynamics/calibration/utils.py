from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
from estival.sampling import tools as esamp
import arviz as az
import pandas as pd
from typing import List, Dict
from tbdynamics.model import build_model
from tbdynamics.inputs import load_params, load_targets, matrix
from tbdynamics.constants import (
    age_strata,
    compartments,
    latent_compartments,
    infectious_compartments,
    quantiles
)
from numpyro import distributions as dist

def get_bcm(params, covid_effects = None, improved_detection_multiplier = None) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model.
    Parameters:
    - params (dict): A dictionary containing fixed parameters for the model.

    Returns:
    - BayesianCompartmentalModel: An instance of the BayesianCompartmentalModel class, ready for
      simulation and analysis. This model encapsulates the TB compartmental model, the dynamic
      and fixed parameters, prior distributions for Bayesian inference, and target data for model
      validation or calibration.
    """
    params = params or {}
    fixed_params = load_params()
    tb_model = build_model(
        compartments,
        latent_compartments,
        infectious_compartments,
        age_strata,
        fixed_params,
        matrix,
        covid_effects,
        improved_detection_multiplier
    )
    priors = get_all_priors(covid_effects)
    targets = get_targets()
    return BayesianCompartmentalModel(tb_model, params, priors, targets)


def get_all_priors(covid_effects) -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    priors = [
        esp.UniformPrior("contact_rate", (0.001, 0.05)),
        # esp.TruncNormalPrior("contact_rate", 0.0255, 0.00817,  (0.001, 0.05)),
        # esp.UniformPrior("start_population_size", (2000000.0, 4000000.0)),
        esp.BetaPrior("rr_infection_latent", 3.0, 8.0),
        esp.BetaPrior("rr_infection_recovered", 2.0, 2.0),
        # esp.UniformPrior("rr_infection_latent", (0.2, 0.5)),
        # esp.UniformPrior("rr_infection_recovered", (0.1, 1.0)),
        esp.GammaPrior.from_mode("progression_multiplier", 1.0, 2.0),
        # esp.UniformPrior("seed_time", (1800.0, 1840.0)),
        # esp.UniformPrior("seed_num", (1.0, 100.00)),
        # esp.UniformPrior("seed_duration", (1.0, 20.0)),
        esp.TruncNormalPrior(
            "smear_positive_death_rate", 0.389, 0.0276, (0.335, 0.449)
        ),
        esp.TruncNormalPrior(
            "smear_negative_death_rate", 0.025, 0.0041, (0.017, 0.035)
        ),
        esp.TruncNormalPrior(
            "smear_positive_self_recovery", 0.231, 0.0276, (0.177, 0.288)
        ),
        esp.TruncNormalPrior(
            "smear_negative_self_recovery", 0.130, 0.0291, (0.073, 0.209)
        ),
        esp.UniformPrior("screening_scaleup_shape", (0.05, 0.5)),
        esp.TruncNormalPrior("screening_inflection_time", 2000, 3.5, (1990, 2010)),
        esp.GammaPrior.from_mode("time_to_screening_end_asymp", 2.0, 5.0),
    ]
    if covid_effects["contact_reduction"]:
        priors.append(esp.UniformPrior("contact_reduction", (0.01, 0.8)))
    if covid_effects["detection_reduction"]:
        priors.append(esp.UniformPrior("detection_reduction", (0.01, 0.8)))
    for prior in priors:
        prior._pymc_transform_eps_scale = 0.1
    return priors


def get_targets() -> List:
    """
    Loads target data for a model and constructs a list of NormalTarget instances.

    This function is designed to load external target data, presumably for the purpose of
    model calibration or validation. It then constructs and returns a list of NormalTarget
    instances, each representing a specific target metric with associated observed values
    and standard deviations. These targets are essential for fitting the model to observed
    data, allowing for the estimation of model parameters that best align with real-world
    observations.

    Returns:
    - list: A list of Target instances.
    """
    target_data = load_targets()
    notif_dispersion = esp.UniformPrior("notif_dispersion", (1000.0, 15000.0))
    prev_dispersion = esp.UniformPrior("prev_dispersion", (20.0, 70.0))
    # sptb_dispersion = esp.UniformPrior("sptb_dispersion", (5.0,30.0))
    return [
        est.NormalTarget(
            "total_population", target_data["total_population"], stdev=100000.0
        ),
        est.NormalTarget("notification", target_data["notification"], notif_dispersion),
        est.NormalTarget(
            "adults_prevalence_pulmonary",
            target_data["adults_prevalence_pulmonary_target"],
            prev_dispersion,
        ),
        # est.NormalTarget("prevalence_smear_positive", target_data["prevalence_smear_positive_target"], sptb_dispersion),
    ]



def convert_prior_to_numpyro(prior):
    """
    Converts a given custom prior to a corresponding Numpyro distribution and its bounds based on its type.

    Args:
        prior: A custom prior object.

    Returns:
        A tuple of (Numpyro distribution, bounds).
    """
    if isinstance(prior, esp.UniformPrior):
        return dist.Uniform(low=prior.start, high=prior.end), (prior.start, prior.end)
    elif isinstance(prior, esp.TruncNormalPrior):
        return dist.TruncatedNormal(
            loc=prior.mean,
            scale=prior.stdev,
            low=prior.trunc_range[0],
            high=prior.trunc_range[1],
        ), (prior.trunc_range[0], prior.trunc_range[1])
    elif isinstance(prior, esp.GammaPrior):
        rate = 1.0 / prior.scale
        return dist.Gamma(concentration=prior.shape, rate=rate), None
    elif isinstance(prior, esp.BetaPrior):
        return dist.Beta(concentration1=prior.a, concentration0=prior.b), (0, 1)
    else:
        raise TypeError(f"Unsupported prior type: {type(prior).__name__}")


def convert_all_priors_to_numpyro(priors):
    """
    Converts a dictionary of custom priors to a dictionary of corresponding Numpyro distributions.

    Args:
        priors: Dictionary of custom prior objects.

    Returns:
        Dictionary of Numpyro distributions.
    """
    numpyro_priors = {}
    for key, prior in priors.items():
        numpyro_prior, _ = convert_prior_to_numpyro(prior)
        numpyro_priors[key] = numpyro_prior
    return numpyro_priors


def calculate_covid_diff_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str],
    years: List[int],
    covid_analysis: int = 1,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the models for the specified scenarios and calculate the absolute and relative differences.
    Store the quantiles in DataFrames with years as the index and quantiles as columns.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to calculate differences for.
        years: List of years for which to calculate the differences.
        covid_analysis: Integer specifying which analysis to run (1 or 2).
            - 1: Compare scenario 1 and scenario 0.
            - 2: Compare scenario 2 and scenario 0.

    Returns:
        A dictionary containing two dictionaries:
        - "abs": Stores DataFrames for absolute differences (keyed by indicator name).
        - "rel": Stores DataFrames for relative differences (keyed by indicator name).
    """

    # Validate that covid_analysis is either 1 or 2
    if covid_analysis not in [1, 2]:
        raise ValueError("Invalid value for covid_analysis. Must be 1 or 2.")
    covid_scenarios = [
        {"detection_reduction": False, "contact_reduction": False},  # No reduction
        {
            "detection_reduction": True,
            "contact_reduction": True,
        },  # With detection + contact reduction
        {
            "detection_reduction": True,
            "contact_reduction": False,
        },  # No contact reduction
    ]
    # Run models for the specified scenarios
    scenario_results = []
    for covid_effects in covid_scenarios:
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm)
        scenario_results.append(spaghetti_res.results)

    # Calculate the differences based on the covid_analysis value
    abs_diff = (
        scenario_results[covid_analysis][indicators] - scenario_results[0][indicators]
    )
    rel_diff = abs_diff / scenario_results[0][indicators]

    # Calculate the differences for each indicator and store them in DataFrames
    diff_quantiles_abs = {}
    diff_quantiles_rel = {}
    for ind in indicators:
        diff_quantiles_df_abs = pd.DataFrame(
            {
                quantile: [abs_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in quantiles
            },
            index=years,
        )

        diff_quantiles_df_rel = pd.DataFrame(
            {
                quantile: [rel_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in quantiles
            },
            index=years,
        )

        diff_quantiles_abs[ind] = diff_quantiles_df_abs
        diff_quantiles_rel[ind] = diff_quantiles_df_rel

    return {"abs": diff_quantiles_abs, "rel": diff_quantiles_rel}

def calculate_scenario_and_diff_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str],
    years: List[int],
    detection_multipliers: List[float],
    calculate_diff: bool = True, 
    scenario_choice: int =2,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calculate the model results for each scenario with different detection multipliers
    and optionally compute the differences compared to a base scenario.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to calculate differences for.
        years: List of years for which to calculate the differences.
        detection_multipliers: List of multipliers for improved detection to loop through.
        calculate_diff: Boolean to indicate whether to calculate the differences relative to the base scenario.

    Returns:
        A dictionary containing results for each scenario and optionally the differences.
    """

    if scenario_choice == 1:
        scenario_config = {"detection_reduction": True, "contact_reduction": True}
    elif scenario_choice == 2:
        scenario_config = {"detection_reduction": True, "contact_reduction": False}
    else:
        raise ValueError("Invalid scenario_choice. Choose 1 or 2.")
    # Base scenario (without improved detection)
    bcm = get_bcm(params, scenario_config)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, quantiles)

    # Store results for each detection multiplier
    scenario_outputs = {"base_scenario": base_quantiles}
    detection_diff_results = {}

    for multiplier in detection_multipliers:
        # Improved detection scenario
        bcm = get_bcm(params, {"detection_reduction": True, "contact_reduction": False}, multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, quantiles)

        # Store the results for this scenario
        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

        if calculate_diff:
            # Calculate the differences compared to the base scenario
            abs_diff = scenario_quantiles[indicators] - base_quantiles[indicators]
            rel_diff = abs_diff / base_quantiles[indicators]

            # Calculate the differences for each indicator and store them in DataFrames
            diff_quantiles_abs = {}
            diff_quantiles_rel = {}

            for ind in indicators:
                if ind not in abs_diff.columns or ind not in rel_diff.columns:
                    print(f"Warning: Indicator '{ind}' not found in the results. Skipping.")
                    continue

                diff_quantiles_df_abs = pd.DataFrame(
                    {
                        quantile: [abs_diff[ind].loc[year].quantile(quantile) for year in years]
                        for quantile in quantiles
                    },
                    index=years,
                )

                diff_quantiles_df_rel = pd.DataFrame(
                    {
                        quantile: [rel_diff[ind].loc[year].quantile(quantile) for year in years]
                        for quantile in quantiles
                    },
                    index=years,
                )

                diff_quantiles_abs[ind] = diff_quantiles_df_abs
                diff_quantiles_rel[ind] = diff_quantiles_df_rel

            detection_diff_results[scenario_key] = {
                "abs": diff_quantiles_abs,
                "rel": diff_quantiles_rel,
            }

    # Return the scenario outputs and, if calculated, the differences
    return {
        "scenario_outputs": scenario_outputs,
        "detection_diff_results": detection_diff_results if calculate_diff else None
    }
