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
import numpy as np

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

def calculate_covid_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    cumulative_start_time: int = 2020,
    covid_analysis: int = 2,
    years: List[float] = [2021.0, 2022.0, 2025.0, 2030.0, 2035.0],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the models for the specified scenarios, calculate cumulative diseased and death values,
    and return quantiles for absolute and relative differences between scenarios.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        cumulative_start_time: Year to start calculating the cumulative values.
        covid_analysis: Integer specifying which analysis to run (default is 2).
        years: List of years for which to calculate the differences.

    Returns:
        A dictionary containing quantiles for absolute and relative differences between scenarios.
    """

    # Validate that covid_analysis is either 1 or 2
    if covid_analysis not in [1, 2]:
        raise ValueError("Invalid value for covid_analysis. Must be 1 or 2.")

    # Define the scenarios
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

    scenario_results = []
    for covid_effects in covid_scenarios:
        # Get the model results
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm).results

        # Filter the results to include only the rows where the index (year) is an integer
        yearly_data = spaghetti_res.loc[
            (spaghetti_res.index >= cumulative_start_time) & 
            (spaghetti_res.index % 1 == 0)
        ]

        # Calculate cumulative sums for each sample
        cumulative_diseased_yearly = yearly_data['incidence_raw'].cumsum()
        cumulative_deaths_yearly = yearly_data['mortality_raw'].cumsum()

        # Store the cumulative results in the list
        scenario_results.append({
            "cumulative_diseased": cumulative_diseased_yearly,
            "cumulative_deaths": cumulative_deaths_yearly,
        })

    # Calculate the differences based on the covid_analysis value
    abs_diff = {
        "cumulative_diseased": scenario_results[covid_analysis]["cumulative_diseased"] - scenario_results[0]["cumulative_diseased"],
        "cumulative_deaths": scenario_results[covid_analysis]["cumulative_deaths"] - scenario_results[0]["cumulative_deaths"]
    }
    rel_diff = {
        "cumulative_diseased": abs_diff["cumulative_diseased"] / scenario_results[0]["cumulative_diseased"],
        "cumulative_deaths": abs_diff["cumulative_deaths"] / scenario_results[0]["cumulative_deaths"]
    }

    # Calculate quantiles for absolute and relative differences
    diff_quantiles_abs = {}
    diff_quantiles_rel = {}

    for ind in ["cumulative_diseased", "cumulative_deaths"]:
        # Calculate absolute difference quantiles
        diff_quantiles_df_abs = pd.DataFrame(
            {
                quantile: [abs_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in quantiles
            },
            index=years,
        )

        # Calculate relative difference quantiles
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


def calculate_notifications_for_covid(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate model outputs for each scenario defined in covid_configs and store the results
    in a dictionary where the keys correspond to the keys in covid_configs.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to calculate outputs for.

    Returns:
        A dictionary where each key corresponds to a scenario in covid_configs and the value is 
        another dictionary containing DataFrames with outputs for the given indicators.
    """

    # Define the covid_configs inside the function
    covid_configs = {
        'no_covid': {
            "detection_reduction": False,
            "contact_reduction": False
        },  # No reduction
        'detection_and_contact_reduction': {
            "detection_reduction": True,
            "contact_reduction": True
        },  # With detection + contact reduction
        'case_detection_reduction_only': {
            "detection_reduction": True,
            "contact_reduction": False
        },  # No contact reduction
        'contact_reduction_only': {
            "detection_reduction": False,
            "contact_reduction": True
        }  # Only contact reduction
    }

    scenario_outputs = {}

    # Loop through each scenario in covid_configs
    for scenario_name, covid_effects in covid_configs.items():
        # Run the model for the current scenario
        bcm = get_bcm(params, covid_effects)
        model_results = esamp.model_results_for_samples(idata_extract, bcm)
        spaghetti_res = model_results.results
        ll_res = model_results.extras  # Extract additional results (e.g., log-likelihoods)
        scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, quantiles)

        # Initialize a dictionary to store indicator-specific outputs
        indicator_outputs = {}

         # Extract the results only for the "notification" indicator
        notification_indicator = "notification"  # Replace with the exact name of the notification indicator
        if notification_indicator in scenario_quantiles:
            indicator_outputs[notification_indicator] = scenario_quantiles[notification_indicator]

        # Store the outputs and ll_res in the dictionary with the scenario name as the key
        scenario_outputs[scenario_name] = {
            "indicator_outputs": indicator_outputs,
            "ll_res": ll_res
        }

    return scenario_outputs

def calculate_dic(log_likelihoods: np.ndarray) -> float:
    """
    Calculate the Deviance Information Criterion (DIC) given log-likelihood values.

    Args:
        log_likelihoods: Array of log-likelihood values.

    Returns:
        The DIC value.
    """
    mean_log_likelihood = np.mean(log_likelihoods)
    deviance = -2 * log_likelihoods
    mean_deviance = np.mean(deviance)
    dic = 2 * mean_deviance - (-2 * mean_log_likelihood)
    return dic

def calculate_dic_for_scenarios(covid_outputs: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, float]:
    """
    Calculate DIC for each scenario based on the 'loglikelihood' values in 'll_res' for each scenario.

    Args:
        covid_outputs: Dictionary containing outputs for each scenario, including log-likelihoods.

    Returns:
        A dictionary where each key is a scenario name and the value is the DIC calculated from the 'loglikelihood'.
    """
    dic_results = {}

    for scenario, results in covid_outputs.items():
        ll_res = results['ll_res']  # Get the DataFrame containing log-likelihoods

        # Calculate DIC only for the 'loglikelihood' column
        if 'loglikelihood' in ll_res.columns:
            dic_value = calculate_dic(ll_res['loglikelihood'].values)
            dic_results[scenario] = dic_value

    return dic_results

def loo_cross_validation(log_likelihoods: np.ndarray) -> float:
    """
    Calculate the Leave-One-Out Information Criterion (LOO-IC).

    Args:
        log_likelihoods: Array of log-likelihood values. Can be 1D (for a single observation)
                         or 2D with shape (n_samples, n_observations).

    Returns:
        The LOO-IC value.
    """
    # Ensure log_likelihoods is a 2D array
    if log_likelihoods.ndim == 1:
        log_likelihoods = log_likelihoods[:, np.newaxis]
    
    n_samples, n_observations = log_likelihoods.shape

    # Calculate the log of the mean of the exponential of the log-likelihoods excluding each data point
    loo_log_likelihoods = np.zeros(n_observations)
    for i in range(n_observations):
        loo_log_likelihoods[i] = np.log(np.mean(np.exp(log_likelihoods[:, i])))

    # LOO-IC is computed as -2 times the sum of these values
    loo_ic = -2 * np.sum(loo_log_likelihoods)
    return loo_ic

def calculate_loo_for_scenarios(covid_outputs: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, float]:
    """
    Calculate LOO-IC for each scenario based on the 'loglikelihood' values in 'll_res' for each scenario.

    Args:
        covid_outputs: Dictionary containing outputs for each scenario, including log-likelihoods.

    Returns:
        A dictionary where each key is a scenario name and the value is the LOO-IC calculated from the 'loglikelihood'.
    """
    loo_results = {}

    for scenario, results in covid_outputs.items():
        ll_res = results['ll_res']  # Get the DataFrame containing log-likelihoods

        # Calculate LOO-IC only for the 'loglikelihood' column
        if 'loglikelihood' in ll_res.columns:
            loo_ic_value = loo_cross_validation(ll_res['loglikelihood'].values)
            loo_results[scenario] = loo_ic_value

    return loo_results

def calculate_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'mortality_raw'],
    detection_multipliers: List[float] = [2.0, 5.0, 12.0],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate the model results for each scenario with different detection multipliers
    and return the baseline and scenario outputs.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to return for the other scenarios (default: ['incidence', 'mortality_raw']).
        detection_multipliers: List of multipliers for improved detection to loop through (default: [2.0, 5.0, 12.0]).

    Returns:
        A dictionary containing results for the baseline and each scenario.
    """

    # Fixed scenario configuration
    scenario_config = {"detection_reduction": True, "contact_reduction": False}

    # Base scenario (calculate outputs for all indicators)
    bcm = get_bcm(params, scenario_config)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, quantiles)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each detection multiplier scenario
    for multiplier in detection_multipliers:
        bcm = get_bcm(params, scenario_config, multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, quantiles)

        # Store the results for this scenario
        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

    # Extract only the relevant indicators for each scenario
    for scenario_key in scenario_outputs:
        if scenario_key != "base_scenario":
            scenario_outputs[scenario_key] = scenario_outputs[scenario_key][indicators]

    return scenario_outputs

def calculate_scenario_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    detection_multipliers: List[float],
    cumulative_start_time: int = 2020,
    scenario_choice: int = 2,
    years: List[int] = [2021, 2022, 2025, 2030, 2035],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calculate the cumulative incidence and deaths for each scenario with different detection multipliers,
    compute the differences compared to a base scenario, and return quantiles for absolute and relative differences.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        detection_multipliers: List of multipliers for improved detection to loop through.
        cumulative_start_time: Year to start calculating the cumulative values.
        scenario_choice: Integer specifying which scenario to use (1 or 2).
        years: List of years for which to calculate the quantiles.

    Returns:
        A dictionary containing the quantiles for absolute and relative differences between scenarios.
    """

    # Set scenario configuration based on scenario_choice
    if scenario_choice == 1:
        scenario_config = {"detection_reduction": True, "contact_reduction": True}
    elif scenario_choice == 2:
        scenario_config = {"detection_reduction": True, "contact_reduction": False}
    else:
        raise ValueError("Invalid scenario_choice. Choose 1 or 2.")

    # Base scenario (without improved detection)
    bcm = get_bcm(params, scenario_config)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results

    # Calculate cumulative sums for the base scenario
    yearly_data_base = base_results.loc[
        (base_results.index >= cumulative_start_time) & 
        (base_results.index % 1 == 0)
    ]
    cumulative_diseased_base = yearly_data_base['incidence_raw'].cumsum()
    cumulative_deaths_base = yearly_data_base['mortality_raw'].cumsum()

    # Store results for each detection multiplier
    detection_diff_results = {}

    for multiplier in detection_multipliers:
        # Improved detection scenario
        bcm = get_bcm(params, scenario_config, multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results

        # Calculate cumulative sums for each scenario
        yearly_data = scenario_result.loc[
            (scenario_result.index >= cumulative_start_time) & 
            (scenario_result.index % 1 == 0)
        ]
        cumulative_diseased = yearly_data['incidence_raw'].cumsum()
        cumulative_deaths = yearly_data['mortality_raw'].cumsum()

        # Calculate differences compared to the base scenario
        abs_diff = {
            "cumulative_diseased": cumulative_diseased - cumulative_diseased_base,
            "cumulative_deaths": cumulative_deaths - cumulative_deaths_base,
        }
        rel_diff = {
            "cumulative_diseased": abs_diff["cumulative_diseased"] / cumulative_diseased_base,
            "cumulative_deaths": abs_diff["cumulative_deaths"] / cumulative_deaths_base,
        }

        # Calculate quantiles for absolute and relative differences
        diff_quantiles_abs = {}
        diff_quantiles_rel = {}

        for ind in ["cumulative_diseased", "cumulative_deaths"]:
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

        # Store the quantile results
        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        detection_diff_results[scenario_key] = {
            "abs": diff_quantiles_abs,
            "rel": diff_quantiles_rel,
        }

    # Return the quantiles for absolute and relative differences
    return detection_diff_results