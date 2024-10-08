from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
from estival.sampling import tools as esamp
import arviz as az
import pandas as pd
from typing import List, Dict
from tbdynamics.model import build_model
from tbdynamics.inputs import load_params, load_targets, matrix
from tbdynamics.constants import quantiles, covid_configs
from pathlib import Path
from numpyro import distributions as dist



def get_bcm(
    params, covid_effects=None, improved_detection_multiplier=None
) -> BayesianCompartmentalModel:
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
        fixed_params, matrix, covid_effects, improved_detection_multiplier
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
        esp.BetaPrior("rr_infection_recovered", 3.0, 8.0),
        # esp.BetaPrior("rr_infection_recovered", 2.0, 2.0),
        # esp.UniformPrior("rr_infection_latent", (0.2, 0.5)),
        # esp.UniformPrior("rr_infection_recovered", (0.2, 1.0)),
        # esp.TruncNormalPrior("rr_infection_latent", 0.35, 0.1, (0.2, 0.5)),
        # esp.TruncNormalPrior("rr_infection_recovered", 0.6, 0.2, (0.2, 1.0)),
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


def calculate_covid_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    cumulative_start_time: float = 2020.0,
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
    covid_configs = [
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

    covid_results = []
    for covid_effects in covid_configs:
        # Get the model results
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm).results

        # Filter the results to include only the rows where the index (year) is an integer
        yearly_data = spaghetti_res.loc[
            (spaghetti_res.index >= cumulative_start_time)
            & (spaghetti_res.index % 1 == 0)
        ]

        # Calculate cumulative sums for each sample
        cumulative_diseased_yearly = yearly_data["incidence_raw"].cumsum()
        cumulative_deaths_yearly = yearly_data["mortality_raw"].cumsum()

        # Store the cumulative results in the list
        covid_results.append(
            {
                "cumulative_diseased": cumulative_diseased_yearly,
                "cumulative_deaths": cumulative_deaths_yearly,
            }
        )

    # Calculate the differences based on the covid_analysis value
    abs_diff = {
        "cumulative_diseased": covid_results[covid_analysis]["cumulative_diseased"]
        - covid_results[0]["cumulative_diseased"],
        "cumulative_deaths": covid_results[covid_analysis]["cumulative_deaths"]
        - covid_results[0]["cumulative_deaths"],
    }
    rel_diff = {
        "cumulative_diseased": abs_diff["cumulative_diseased"]
        / covid_results[0]["cumulative_diseased"],
        "cumulative_deaths": abs_diff["cumulative_deaths"]
        / covid_results[0]["cumulative_deaths"],
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

    covid_outputs = {}

    # Loop through each scenario in covid_configs
    for covid_name, covid_effects in covid_configs.items():
        # Run the model for the current scenario
        bcm = get_bcm(params, covid_effects)
        model_results = esamp.model_results_for_samples(idata_extract, bcm)
        spaghetti_res = model_results.results
        ll_res = model_results.extras
        scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, quantiles)

        # Initialize a dictionary to store indicator-specific outputs
        indicator_outputs = {}

        # Extract the results only for the "notification" indicator
        notification_indicator = (
            "notification"  # Replace with the exact name of the notification indicator
        )
        if notification_indicator in scenario_quantiles:
            indicator_outputs[notification_indicator] = scenario_quantiles[
                notification_indicator
            ]

        # Store the outputs and ll_res in the dictionary with the scenario name as the key
        covid_outputs[covid_name] = {
            "indicator_outputs": indicator_outputs,
            "ll_res": ll_res,
        }

    return covid_outputs

def calculate_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ["incidence", "mortality_raw"],
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
    covid_choice: int = 2,
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
    if covid_choice == 1:
        covid_config = {"detection_reduction": True, "contact_reduction": True}
    elif covid_choice == 2:
        covid_config = {"detection_reduction": True, "contact_reduction": False}
    else:
        raise ValueError("Invalid scenario_choice. Choose 1 or 2.")

    # Base scenario (without improved detection)
    bcm = get_bcm(params, covid_config)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results

    # Calculate cumulative sums for the base scenario
    yearly_data_base = base_results.loc[
        (base_results.index >= cumulative_start_time) & (base_results.index % 1 == 0)
    ]
    cumulative_diseased_base = yearly_data_base["incidence_raw"].cumsum()
    cumulative_deaths_base = yearly_data_base["mortality_raw"].cumsum()

    # Store results for each detection multiplier
    detection_diff_results = {}

    for multiplier in detection_multipliers:
        # Improved detection scenario
        bcm = get_bcm(params, covid_config, multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results

        # Calculate cumulative sums for each scenario
        yearly_data = scenario_result.loc[
            (scenario_result.index >= cumulative_start_time)
            & (scenario_result.index % 1 == 0)
        ]
        cumulative_diseased = yearly_data["incidence_raw"].cumsum()
        cumulative_deaths = yearly_data["mortality_raw"].cumsum()

        # Calculate differences compared to the base scenario
        abs_diff = {
            "cumulative_diseased": cumulative_diseased - cumulative_diseased_base,
            "cumulative_deaths": cumulative_deaths - cumulative_deaths_base,
        }
        rel_diff = {
            "cumulative_diseased": abs_diff["cumulative_diseased"]
            / cumulative_diseased_base,
            "cumulative_deaths": abs_diff["cumulative_deaths"] / cumulative_deaths_base,
        }

        # Calculate quantiles for absolute and relative differences
        diff_quantiles_abs = {}
        diff_quantiles_rel = {}

        for ind in ["cumulative_diseased", "cumulative_deaths"]:
            diff_quantiles_df_abs = pd.DataFrame(
                {
                    quantile: [
                        abs_diff[ind].loc[year].quantile(quantile) for year in years
                    ]
                    for quantile in quantiles
                },
                index=years,
            )

            diff_quantiles_df_rel = pd.DataFrame(
                {
                    quantile: [
                        rel_diff[ind].loc[year].quantile(quantile) for year in years
                    ]
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

# Load all inference data for different COVID configurations
def load_idata(out_path, covid_configs):
    inference_data_dict = {}
    for config_name in covid_configs.keys():
        calib_file = Path(out_path) / f"calib_full_out_{config_name}.nc"
        if calib_file.exists():
            idata_raw = az.from_netcdf(calib_file)
            inference_data_dict[config_name] = idata_raw
        else:
            raise FileNotFoundError(f"File {calib_file} does not exist.")
    
    return inference_data_dict

# Extract and save inference data for each COVID configuration
def extract_and_save_idata(idata_dict, output_dir, num_samples=500):
    for config_name, burnt_idata in idata_dict.items():
        # Extract samples (you might adjust the number of samples as needed)
        idata_extract = az.extract(burnt_idata, num_samples=num_samples)
        # Convert extracted data into InferenceData object
        inference_data = az.convert_to_inference_data(idata_extract.reset_index('sample'))
        # Save the extracted InferenceData object to a netCDF file
        output_file = Path(output_dir) / f"idata_{config_name}.nc"
        az.to_netcdf(inference_data, output_file)
        print(f"Saved extracted inference data for {config_name} to {output_file}")

def load_extracted_idata(out_path, covid_configs):
    inference_data_dict = {}
    for config_name in covid_configs.keys():
        input_file = Path(out_path) / f"idata_{config_name}.nc"
        if input_file.exists():
            idata = az.from_netcdf(input_file)
            inference_data_dict[config_name] = idata
        else:
            print(f"File {input_file} does not exist.")
    
    return inference_data_dict

def run_model_for_covid(params, covid_configs, output_dir, quantiles):
    covid_outputs = {}
    # Load the extracted InferenceData
    inference_data_dict = load_extracted_idata(output_dir, covid_configs)
    for covid_name, covid_effects in covid_configs.items():
        # Load the inference data for this specific scenario
        if covid_name not in inference_data_dict:
            print(f"Skipping {covid_name} as no inference data was loaded.")
            continue
        idata_extract = inference_data_dict[covid_name]
        
        # Run the model for the current scenario
        bcm = get_bcm(params, covid_effects)  # Adjust this function as needed
        model_results = esamp.model_results_for_samples(idata_extract, bcm)
        
        # Extract results from the model output
        spaghetti_res = model_results.results
        ll_res = model_results.extras  # Extract additional results (e.g., log-likelihoods)
        scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, quantiles)

        # Define the indicators you're interested in
        indicators = ["notification", "total_population", "adults_prevalence_pulmonary"]

        missing_indicators = [indicator for indicator in indicators if indicator not in scenario_quantiles.columns]
        if missing_indicators:
            print(f"Missing indicators {missing_indicators} in scenario {covid_name}. Skipping this scenario.")
            continue

        # Store the DataFrame of quantiles directly for the defined indicators
        indicator_outputs = scenario_quantiles[indicators]

        # Store the outputs and log-likelihoods in the dictionary with the scenario name as the key
        covid_outputs[covid_name] = {
            "indicator_outputs": indicator_outputs,
            "ll_res": ll_res,
        }
    return covid_outputs

def convert_ll_to_idata(ll_res):
    # Convert log-likelihoods into a DataFrame
    ds = pd.DataFrame(ll_res).to_xarray()
    # Convert the DataFrame into an xarray.Dataset
    # Create an InferenceData object
    idata = az.from_dict(
        posterior={"logposterior": ds["logposterior"]},
        prior={"logprior": ds["logprior"]},
        log_likelihood={"total_loglikelihood": ds["loglikelihood"]}
    )
    return idata

def calculate_waic_comparison(covid_outputs):
    waic_dict = {}
    for covid_name, output in covid_outputs.items():
        # Extract the log-likelihoods (ll_res) for the current scenario
        ll_res = output["ll_res"]
        # Convert ll_res to InferenceData
        idata = convert_ll_to_idata(ll_res)
        # Store InferenceData in the dictionary for WAIC comparison
        waic_dict[covid_name] = idata
    
    # Compare the WAIC across all scenarios
    waic_results = {config_name: az.waic(idata) for config_name, idata in waic_dict.items()}
    # Compare using az.compare
    waic_comparison = az.compare(waic_results, ic="waic")  # Using WAIC for information criterion
    return waic_comparison

def get_idata_ll(covid_configs, inference_data_dict, params, get_bcm):
    idata_dict = {}  # Dictionary to store idata for each model for comparison
    for covid_name, covid_effects in covid_configs.items():
        # Load the inference data for this specific scenario
        if covid_name not in inference_data_dict:
            print(f"Skipping {covid_name} as no inference data was loaded.")
            continue

        idata = inference_data_dict[covid_name]

        # Get the model configuration for the current scenario
        bcm = get_bcm(params, covid_effects)

        # Calculate extra likelihood measures and get log_lik DataFrame
        # Assuming esamp.likelihood_extras_for_idata updates idata and returns log_lik DataFrame
        log_lik = esamp.likelihood_extras_for_idata(idata, bcm)
        # Extract the 'loglikelihood' column
        df_log_likelihood = log_lik[['loglikelihood']]
        # Rename the column to 'log_likelihood' if necessary
        df_log_likelihood = df_log_likelihood.rename(columns={'loglikelihood': 'log_likelihood'})

        # Convert to xarray Dataset
        ds_log_likelihood = df_log_likelihood.to_xarray()

        # Add the 'log_likelihood' group to idata
        if not hasattr(idata, 'log_likelihood'):
            # Add the 'log_likelihood' group to idata
            idata.add_groups({'log_likelihood': ds_log_likelihood})
        else:
            print(f"'log_likelihood' group already exists in idata for {covid_name}. Skipping adding.")

        # Add the updated idata to the dictionary for comparison
        idata_dict[covid_name] = idata

    # Compare models using az.compare with ic='waic'
    # compare_df = az.compare(idata_dict, ic='waic')
    return idata_dict