from tbdynamics.model import build_model
from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
from tbdynamics.inputs import load_params, load_targets
from tbdynamics.constants import (
    age_strata,
    compartments,
    latent_compartments,
    infectious_compartments,
)
from tbdynamics.inputs import conmat


def get_bcm(params) -> BayesianCompartmentalModel:
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
    fixed_params = load_params()
    tb_model = build_model(
        compartments,
        latent_compartments,
        infectious_compartments,
        age_strata,
        fixed_params,
        conmat,
    )
    priors = get_all_priors()
    targets = get_targets()
    return BayesianCompartmentalModel(tb_model, params, priors, targets)


def get_all_priors() -> list:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    return [
        esp.UniformPrior("start_population_size", (2000000.0, 8000000.0)),
        esp.UniformPrior("contact_rate", (0.0001, 0.2)),
        esp.UniformPrior("rr_infection_latent", (0.2, 0.5)),
        esp.UniformPrior("rr_infection_recovered", (0.2, 0.5)),
        esp.UniformPrior("progression_multiplier", (1.0, 2.0)),
        esp.UniformPrior("seed_time", (1890.0, 1950.0)),
        esp.UniformPrior("seed_num", (1.0, 100.00)),
        esp.UniformPrior("seed_duration", (1.0, 5.0)),
        esp.UniformPrior("smear_positive_death_rate", (0.335, 0.449)),
        esp.UniformPrior("smear_negative_death_rate", (0.017, 0.035)),
        esp.UniformPrior("smear_positive_self_recovery", (0.177, 0.288)),
        esp.UniformPrior("smear_negative_self_recovery", (0.073, 0.209)),
        esp.UniformPrior("screening_scaleup_shape", (0.07, 0.1)),
        esp.UniformPrior("screening_inflection_time", (1993, 2005)),
        esp.UniformPrior("screening_end_asymp", (0.5, 0.65)),
    ]


def get_targets() -> list:
    """
    Loads target data for a model and constructs a list of NormalTarget instances.

    This function is designed to load external target data, presumably for the purpose of
    model calibration or validation. It then constructs and returns a list of NormalTarget
    instances, each representing a specific target metric with associated observed values
    and standard deviations. These targets are essential for fitting the model to observed
    data, allowing for the estimation of model parameters that best align with real-world
    observations.

    Returns:
    - list: A list of NormalTarget instances. Each NormalTarget specifies a model target
      metric (e.g., total population, notification rates), the observed value for that
      metric, and a standard deviation representing the uncertainty around the observed
      value. The standard deviations are predefined and serve as a measure of variance
      in the observed data.
    """
    target_data = load_targets()
    return [
        est.NormalTarget("total_population", target_data["pop"], stdev=10000.0),
        est.NormalTarget("notification", target_data["notifs"], stdev=100.0),
    ]
