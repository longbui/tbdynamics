
from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
import arviz as az
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
from typing import List, Dict
from tbdynamics.model import build_model
from tbdynamics.inputs import load_params, load_targets, matrix
from tbdynamics.constants import (
    age_strata,
    compartments,
    latent_compartments,
    infectious_compartments,
    indicator_names
)
from tbdynamics.utils import get_row_col_for_subplots, get_standard_subplot_fig
from tbdynamics.constants import indicator_names

pio.templates.default = "simple_white"

def get_bcm(params=None) -> BayesianCompartmentalModel:
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
    )
    priors = get_all_priors()
    targets = get_targets()
    return BayesianCompartmentalModel(tb_model, params, priors, targets)


def get_all_priors() -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    return [
        esp.UniformPrior("contact_rate", (0.005, 0.010)),
        # esp.UniformPrior("start_population_size", (2000000.0, 2500000.0)),
        esp.UniformPrior("rr_infection_latent", (0.2, 0.5)),
        esp.UniformPrior("rr_infection_recovered", (0.2, 0.5)),
        esp.UniformPrior("progression_multiplier", (1.5, 2.0)),
        # esp.UniformPrior("seed_time", (1800.0, 1840.0)),
        # esp.UniformPrior("seed_num", (1.0, 100.00)),
        # esp.UniformPrior("seed_duration", (1.0, 20.0)),
        esp.UniformPrior("smear_positive_death_rate", (0.335, 0.449)),
        esp.UniformPrior("smear_negative_death_rate", (0.017, 0.035)),
        esp.UniformPrior("smear_positive_self_recovery", (0.177, 0.288)),
        esp.UniformPrior("smear_negative_self_recovery", (0.073, 0.209)),
        esp.UniformPrior("screening_scaleup_shape", (0.05, 0.15)),
        esp.UniformPrior("screening_inflection_time", (1990, 2010)),
        esp.UniformPrior("screening_end_asymp", (0.55, 0.7)),
        esp.UniformPrior("detection_reduction", (0.7, 0.9)),
    ]


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
    return [
        est.NormalTarget("total_population", target_data["total_population"], stdev=100000.0),
        # est.NormalTarget("incidence", target_data["incidence"], 1.0),
        est.NormalTarget("notification", target_data["notification"], stdev=5000.0),
        # est.NormalTarget("percentage_latent", target_data["percentage_latent"], 1.0),
        # est.NormalTarget("prevalence_pulmonary", target_data["prevalence_pulmonary"], 1.0),
        #est.NormalTarget("cdr", target_data["cdr"], 0.1)
    ]

def plot_spaghetti(
    spaghetti: pd.DataFrame, 
    indicators: List[str], 
    n_cols: int, 
    target_data: Dict,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023
) -> go.Figure:
    """Generate a spaghetti plot to compare any number of requested outputs to targets.

    Args:
        spaghetti: The values from the sampled runs
        indicators: The names of the indicators to look at
        n_cols: Number of columns for the figure
        targets: The calibration targets

    Returns:
        The spaghetti plot figure object
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(nrows, n_cols, [(indicator_names[ind] if ind in indicator_names else ind.replace('_', ' ')) for ind in indicators])
    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)

        # Model outputs
        ind_spagh = spaghetti[ind]
        ind_spagh.columns = ind_spagh.columns.map(lambda col: f'{col[0]}, {col[1]}')
        fig.add_traces(px.line(ind_spagh).data, rows=row, cols=col)

        # Targets
        if ind in target_data.keys():
            target = target_data[ind]
            target_marker_config = dict(size=5.0, line=dict(width=0.5, color='DarkSlateGrey'))
            lines = go.Scatter(x=target.index, y=target, marker=target_marker_config, name='targets', mode='markers')
            fig.add_trace(lines, row=row, col=col)
        fig.update_layout(yaxis4=dict(range=[0, 2.2]))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=None)
    return fig.update_xaxes(range=[plot_start_date, plot_end_date])

def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame], 
    target_data: Dict, 
    indicators: List[str], 
    quantiles: List[float], 
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023,
    max_alpha: float=0.7
) -> go.Figure:
    """Plot the credible intervals with subplots for each output,
    for a single run of interest.

    Args:
        quantile_outputs: Dataframes containing derived outputs of interest for each analysis type
        targets: Calibration targets
        output: User-requested output of interest
        analysis: The key for the analysis type
        quantiles: User-requested quantiles for the patches to be plotted over
        max_alpha: Maximum alpha value to use in patches

    Returns:
        The interactive figure
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(nrows, n_cols, [(indicator_names[ind] if ind in indicator_names else ind.replace('_', ' ')) for ind in indicators])
    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = quantile_outputs[ind]
        for q, quant in enumerate(quantiles):
            alpha = min((q, len(quantiles) - q)) / np.floor(len(quantiles) / 2) * max_alpha
            fill_colour = f'rgba(0,30,180,{str(alpha)})'
            fig.add_traces(go.Scatter(x=data.index, y=data[quant], fill='tonexty', fillcolor=fill_colour, line={'width': 0}, name=quant), rows=row, cols=col)
        fig.add_traces(go.Scatter(x=data.index, y=data[0.5], line={'color': 'black'}, name='median'), rows=row, cols=col)
        if ind in target_data.keys():
            target = target_data[ind]
            marker_format = {'size': 5.0, 'color': 'rgba(250, 135, 206, 0.2)', 'line': {'width': 1.0}}
            fig.add_traces(go.Scatter(x=target.index, y=target, mode='markers', marker=marker_format), rows=row, cols=col)
    fig.update_xaxes(range=[plot_start_date, plot_end_date])
    return fig.update_layout(yaxis4={'range': [0.0, 2.5]}, showlegend=False)