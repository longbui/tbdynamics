from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
import arviz as az
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from tbdynamics.model import build_model
from tbdynamics.inputs import load_params, load_targets, matrix
from tbdynamics.constants import (
    age_strata,
    compartments,
    latent_compartments,
    infectious_compartments,
    indicator_names,
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
    # prev_dispersion = esp.UniformPrior("prev_dispersion", (10, 50))
    notif_dispersion = esp.UniformPrior("notif_dispersion", (2000, 10000))
    target_data = load_targets()

    targets.extend(
        [
            # est.NormalTarget(
            #     "adults_prevalence_pulmonary",
            #     target_data["adults_prevalence_pulmonary"],
            #     stdev=36.0,
            # ),
            est.NormalTarget("notification", target_data["notification"], stdev=notif_dispersion),
            # est.NormalTarget(
            #     "prevalence_smear_positive",
            #     target_data["prevalence_smear_positive"],
            #     20.0,
            # ),
        ]
    )
    return BayesianCompartmentalModel(tb_model, params, priors, targets)


def get_all_priors() -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    return [
        esp.GammaPrior.from_mode("contact_rate", 0.02, 0.05),
        # esp.UniformPrior("start_population_size", (2000000.0, 3000000.0)),
        esp.BetaPrior("rr_infection_latent", 3.0, 8.0), #The weighted adjusted risk ratio was 0.21 (95% CI: .14–.30)
        esp.BetaPrior("rr_infection_recovered", 2.0, 2.0),
        esp.GammaPrior.from_mean("progression_multiplier", 1.0, 2.0),
        # esp.UniformPrior("seed_time", (1800.0, 1840.0)),
        # esp.UniformPrior("seed_num", (1.0, 100.00)),
        # esp.UniformPrior("seed_duration", (1.0, 20.0)),
        esp.TruncNormalPrior("smear_positive_death_rate", 0.389, 0.0276, (0.335, 0.449)),
        esp.TruncNormalPrior("smear_negative_death_rate", 0.025, 0.0041, (0.017, 0.035)),
        esp.TruncNormalPrior("smear_positive_self_recovery",0.231, 0.0276, (0.177, 0.288)),
        esp.TruncNormalPrior("smear_negative_self_recovery", 0.130, 0.0291, (0.073, 0.209)),
        # esp.UniformPrior("screening_scaleup_shape", (0.05, 0.5)),
        esp.UniformPrior("screening_inflection_time", (1990, 2010)),
        esp.GammaPrior.from_mode("time_to_screening_end_asymp", 1.0, 2.0),
        # esp.TruncNormalPrior("time_to_screening_end_asymp", 1.3, 0.077, (0.0, 12.8)),
        esp.UniformPrior("detection_reduction", (0.01, 0.5)),
        # esp.UniformPrior("contact_reduction", (0.01, 0.8)),
        # esp.UniformPrior("incidence_props_smear_positive_among_pulmonary", (0.1, 0.8)),
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
        est.NormalTarget(
            "total_population", target_data["total_population"], stdev=100000.0
        ),
        # est.NormalTarget("notification", target_data["notification"], stdev=6000.0),
        # est.NormalTarget(
        #     "adults_prevalence_pulmonary",
        #     target_data["adults_prevalence_pulmonary"],
        #     stdev=36.0
        # ),
        # est.NormalTarget("prevalence_smear_positive", target_data["prevalence_smear_positive"], 15.0),
        # est.NormalTarget("case_detection_rate", target_data["case_detection_rate"], 5.0),
    ]


def plot_spaghetti(
    spaghetti: pd.DataFrame,
    indicators: List[str],
    n_cols: int,
    target_data: Dict,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023,
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
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [
            (indicator_names[ind] if ind in indicator_names else ind.replace("_", " "))
            for ind in indicators
        ],
    )
    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)

        # Model outputs
        ind_spagh = spaghetti[ind]
        ind_spagh.columns = ind_spagh.columns.map(lambda col: f"{col[0]}, {col[1]}")
        fig.add_traces(px.line(ind_spagh).data, rows=row, cols=col)

        # Targets
        if ind in target_data.keys():
            target = target_data[ind]
            target_marker_config = dict(
                size=5.0, line=dict(width=0.5, color="DarkSlateGrey")
            )
            lines = go.Scatter(
                x=target.index,
                y=target,
                marker=target_marker_config,
                name="targets",
                mode="markers",
            )
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
    max_alpha: float = 0.7,
) -> go.Figure:
    """Plot the credible intervals with subplots for each output,
    for a single run of interest.

    Args:
        quantile_outputs: Dataframes containing derived outputs of interest for each analysis type.
        target_data: Calibration targets.
        indicators: List of indicators to plot.
        quantiles: List of quantiles for the patches to be plotted over.
        n_cols: Number of columns for the subplots.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value to use in patches.

    Returns:
        The interactive Plotly figure.
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [
            (indicator_names[ind] if ind in indicator_names else ind.replace("_", " "))
            for ind in indicators
        ],
    )
    
    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = quantile_outputs[ind]

        # Filter data by date range
        filtered_data = data[(data.index >= plot_start_date) & (data.index <= plot_end_date)]

        for q, quant in enumerate(quantiles):
            if quant not in filtered_data.columns:
                continue

            alpha = min((quantiles.index(quant), len(quantiles) - quantiles.index(quant))) / (len(quantiles) / 2) * max_alpha
            fill_color = f"rgba(0,30,180,{alpha})"
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[quant],  # Multiply by 100 as specified
                    fill="tonexty",
                    fillcolor=fill_color,
                    line={"width": 0},
                    name=f'{quant}',
                ),
                row=row,
                col=col,
            )

        # Plot the median line
        if 0.5 in filtered_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[0.5],  # Multiply by 100 as specified
                    line={"color": "black"},
                    name='median',
                ),
                row=row,
                col=col,
            )

        # Plot the target data
        if ind in target_data.keys():
            target = target_data[ind]
            filtered_target = target[(target.index >= plot_start_date) & (target.index <= plot_end_date)]
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target,  # Multiply by 100 as specified
                    mode="markers",
                    marker={"size": 5.0, "color": "rgba(250, 135, 206, 0.2)", "line": {"width": 1.0}},
                    name='target',
                ),
                row=row,
                col=col,
            )

        # Update x-axis range to fit the filtered data
        x_min = filtered_data.index.min()
        x_max = filtered_data.index.max()
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

        # Update y-axis range dynamically for each subplot
        y_min = min(filtered_data.min().min(), filtered_target.min() if ind in target_data.keys() else float('inf'))
        y_max = max(filtered_data.max().max(), filtered_target.max() if ind in target_data.keys() else float('-inf'))
        y_range = y_max - y_min
        padding = 0.1 * y_range
        fig.update_yaxes(range=[y_min - padding, y_max + padding], row=row, col=col)
      

    # Update layout for the whole figure
    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='',
        showlegend=False
    )

    return fig

def plot_quantiles_for_case_notifications(
    quantile_df: pd.DataFrame,  # Directly pass the DataFrame
    case_notifications: pd.Series,
    quantiles: List[float],
    plot_start_date: int = 2010,
    plot_end_date: int = 2020,  # Adjust end date based on your data
    max_alpha: float = 0.7
) -> go.Figure:
    """Plot the case notification rates divided by quantiles.

    Args:
        quantile_df: DataFrame containing quantile outputs for case notifications.
        case_notifications: Case notification rates as a Pandas Series.
        quantiles: List of quantiles to plot.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value for the fill color.

    Returns:
        The interactive Plotly figure.
    """
    # Prepare plot
    fig = go.Figure()

    # Ensure the index of quantile_df matches the years in case_notifications
    data_aligned = quantile_df.reindex(case_notifications.index)
    
    # Plot quantiles
    for quant in quantiles:
        if quant in quantile_df.columns:
            alpha = min((quantiles.index(quant), len(quantiles) - quantiles.index(quant))) / (len(quantiles) / 2) * max_alpha
            fill_color = f"rgba(0,30,180,{alpha})"
            
            # Divide notification values by quantile values and multiply by 100
            division_result = (case_notifications / data_aligned[quant] ) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=case_notifications.index,
                    y=division_result,
                    fill="tonexty",
                    fillcolor=fill_color,
                    line={"width": 0},
                    name=f'Quantile {quant}'
                )
            )

    # Add median line
    if 0.500 in quantile_df.columns:
        fig.add_trace(
            go.Scatter(
                x=case_notifications.index,
                y=(case_notifications / quantile_df[0.500].reindex(case_notifications.index)) * 100,
                line={"color": "black"},
                name='Median'
            )
        )

    # Update layout and axis
    fig.update_layout(
        title='Case Notification Rates Divided by Quantiles',
        xaxis_title='Year',
        yaxis_title='Division Result (%)',  # Updated y-axis title
        xaxis=dict(range=[plot_start_date, plot_end_date]),
        showlegend=True
    )

    return fig
