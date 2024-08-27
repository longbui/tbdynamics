from estival.sampling import tools as esamp
import arviz as az
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from typing import List, Dict

from tbdynamics.constants import indicator_names, quantiles
from tbdynamics.utils import (
    get_row_col_for_subplots,
    get_standard_subplot_fig,
)
from tbdynamics.constants import indicator_names
from .utils import convert_prior_to_numpyro
from numpyro import distributions as dist

pio.templates.default = "simple_white"


def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict[str, pd.Series],
    indicators: List[str],
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
        filtered_data = data[
            (data.index >= plot_start_date) & (data.index <= plot_end_date)
        ]

        for q, quant in enumerate(quantiles):
            if quant not in filtered_data.columns:
                continue

            alpha = (
                min((quantiles.index(quant), len(quantiles) - quantiles.index(quant)))
                / (len(quantiles) / 2)
                * max_alpha
            )
            fill_color = f"rgba(0,30,180,{alpha})"

            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[quant],
                    fill="tonexty",
                    fillcolor=fill_color,
                    line={"width": 0},
                    name=f"{quant}",
                ),
                row=row,
                col=col,
            )

        # Plot the median line
        if 0.5 in filtered_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[0.5],
                    line={"color": "black"},
                    name="median",
                ),
                row=row,
                col=col,
            )

        # Plot the point estimates with error bars for indicators with uncertainty bounds
        if ind in [
            "prevalence_smear_positive",
            "adults_prevalence_pulmonary",
            "incidence",
        ]:
            target_series = target_data[f"{ind}_target"]
            lower_bound_series = target_data[f"{ind}_lower_bound"]
            upper_bound_series = target_data[f"{ind}_upper_bound"]

            filtered_target = target_series[
                (target_series.index >= plot_start_date)
                & (target_series.index <= plot_end_date)
            ]
            filtered_lower_bound = lower_bound_series[
                (lower_bound_series.index >= plot_start_date)
                & (lower_bound_series.index <= plot_end_date)
            ]
            filtered_upper_bound = upper_bound_series[
                (upper_bound_series.index >= plot_start_date)
                & (upper_bound_series.index <= plot_end_date)
            ]

            # Plot the point estimates with error bars
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target.values,
                    mode="markers",
                    marker={"size": 4.0, "color": "black"},
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=filtered_upper_bound - filtered_target,
                        arrayminus=filtered_target - filtered_lower_bound,
                        color="black",
                        thickness=0.5,
                        width=5,
                    ),
                    name="",  # No name for legend
                ),
                row=row,
                col=col,
            )

        else:
            # For other indicators, just plot the point estimate if available
            if ind in target_data.keys():
                target = target_data[ind]
                filtered_target = target[
                    (target.index >= plot_start_date) & (target.index <= plot_end_date)
                ]

                # Plot the target point estimates
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target,
                        mode="markers",
                        marker={"size": 4.0, "color": "black"},
                        name="",  # No name for legend
                    ),
                    row=row,
                    col=col,
                )

        # Update x-axis range to fit the filtered data
        x_min = max(filtered_data.index.min(), plot_start_date)
        x_max = filtered_data.index.max()
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

        # Update y-axis range dynamically for each subplot
        y_min = 0
        y_max = max(
            filtered_data.max().max(),
            (
                max(
                    [
                        filtered_target.max()
                        for filtered_target in [
                            filtered_target,
                            filtered_lower_bound,
                            filtered_upper_bound,
                        ]
                    ]
                )
                if ind
                in [
                    "prevalence_smear_positive",
                    "adults_prevalence_pulmonary",
                    "incidence",
                ]
                else (
                    filtered_target.max()
                    if ind in target_data.keys()
                    else float("-inf")
                )
            ),
        )
        y_range = y_max - y_min
        padding = 0.1 * y_range
        fig.update_yaxes(range=[y_min - padding, y_max + padding], row=row, col=col)

    # Update layout for the whole figure
    fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False)

    return fig


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


def plot_post_prior_comparison(idata, priors, params_name):
    """
    Plot comparison of model posterior outputs against priors.

    Args:
        idata: Arviz inference data from calibration.
        priors: Dictionary of custom prior objects.
        params_name: Dictionary mapping parameter names to descriptive titles.

    Returns:
        The figure object.
    """
    # Filter priors to exclude those containing '_dispersion'
    req_vars = [var for var in priors.keys() if "_dispersion" not in var]
    num_vars = len(req_vars)
    num_rows = (num_vars + 1) // 2  # Ensure even distribution across two columns

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    axs = axs.ravel()

    for i_ax, ax in enumerate(axs):
        if i_ax < num_vars:
            var_name = req_vars[i_ax]
            posterior_samples = idata.posterior[var_name].values.flatten()
            low_post = np.min(posterior_samples)
            high_post = np.max(posterior_samples)
            x_vals_posterior = np.linspace(low_post, high_post, 100)

            numpyro_prior, prior_bounds = convert_prior_to_numpyro(priors[var_name])
            if prior_bounds:
                low_prior, high_prior = prior_bounds
                x_vals_prior = np.linspace(low_prior, high_prior, 100)
            else:
                x_vals_prior = (
                    x_vals_posterior  # Fallback if no specific prior bounds are given
                )

            # Compute the original prior density using NumPy's exp function
            prior_density = np.exp(numpyro_prior.log_prob(x_vals_prior))

            # Compute the posterior density using a kernel density estimate
            posterior_density = np.histogram(posterior_samples, bins=100, density=True)[
                0
            ]
            x_vals_posterior = np.linspace(low_post, high_post, len(posterior_density))

            ax.fill_between(
                x_vals_prior,
                prior_density,
                color="k",
                alpha=0.2,
                linewidth=2,
                label="Prior",
            )
            ax.plot(
                x_vals_posterior,
                posterior_density,
                color="b",
                linewidth=1,
                linestyle="solid",
                label="Posterior",
            )

            # Set the title using the descriptive name from params_name
            title = params_name.get(
                var_name, var_name
            )  # Use var_name if not in params_name
            ax.set_title(title)
            ax.legend()
        else:
            ax.axis("off")  # Turn off empty subplots if the number of req_vars is odd

    plt.tight_layout()
    plt.show()


def plot_trace(idata: az.data.inference_data.InferenceData, params_name: dict):
    """
    Plot trace plots for the InferenceData object, excluding parameters containing '_dispersion'.
    Adds descriptive titles from `params_name`.

    Args:
        idata: InferenceData object from ArviZ containing calibration outputs.
        params_name: Dictionary mapping parameter names to descriptive titles.
    """
    # Filter out parameters containing '_dispersion'
    filtered_posterior = idata.posterior.drop_vars(
        [var for var in idata.posterior.data_vars if "_dispersion" in var]
    )

    # Plot trace plots with the filtered parameters
    trace_fig = az.plot_trace(
        filtered_posterior, figsize=(16, 3.1 * len(filtered_posterior.data_vars))
    )

    # Set titles for each row of plots
    var_names = list(
        filtered_posterior.data_vars.keys()
    )  # Get the list of variable names
    for i, var_name in enumerate(var_names):
        row_axes = trace_fig[i, :]  # Get the axes in the current row
        title = params_name.get(
            var_name, var_name
        )  # Get the title from params_name or default to var_name
        row_axes[0].set_title(
            title, fontsize=14, loc="center"
        )  # Set title for the first column
        row_axes[1].set_title("")  # Clear the title for the second column

    plt.tight_layout()
    plt.show()


def plot_output_ranges_for_scenarios(
    params: Dict[str, float],
    idata: az.InferenceData,
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023,
    max_alpha: float = 0.7,
) -> go.Figure:
    """Plot the credible intervals with subplots for each output across two scenarios.

    Args:
        params: Dictionary containing model parameters.
        idata: InferenceData object containing the model data.
        target_data: Calibration targets.
        indicators: List of indicators to plot.
        n_cols: Number of columns for the subplots.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value to use in patches.

    Returns:
        The interactive Plotly figure.
    """

    # Define the two covid_effects scenarios
    covid_scenarios = [
        {"detection_reduction": True, "contact_reduction": True},
        {"detection_reduction": True, "contact_reduction": False},
    ]
    scenario_labels = ["Both Reductions", "Detection Reduction Only"]
    colors = [
        "rgba(0,30,180,{alpha})",
        "rgba(180,30,180,{alpha})",
    ]  # Light purple for the second scenario

    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [
            (indicator_names[ind] if ind in indicator_names else ind.replace("_", " "))
            for ind in indicators
        ],
    )

    for scenario_idx, covid_effects in enumerate(covid_scenarios):
        # Create bcm and run the model for the current covid_effects scenario
        bcm = get_bcm(params, covid_effects)
        idata_extract = az.extract(idata, num_samples=500)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm)
        quantile_outputs = esamp.quantiles_for_results(spaghetti_res.results, quantiles)

        for i, ind in enumerate(indicators):
            row, col = get_row_col_for_subplots(i, n_cols)
            data = quantile_outputs[ind]

            # Filter data by date range
            filtered_data = data[
                (data.index >= plot_start_date) & (data.index <= plot_end_date)
            ]

            for q, quant in enumerate(quantiles):
                if quant not in filtered_data.columns:
                    continue

                alpha = (
                    min(
                        (
                            quantiles.index(quant),
                            len(quantiles) - quantiles.index(quant),
                        )
                    )
                    / (len(quantiles) / 2)
                    * max_alpha
                )
                fill_color = colors[scenario_idx].format(alpha=alpha)

                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[quant],
                        fill="tonexty",
                        fillcolor=fill_color,
                        line={"width": 0},
                        name=f"{scenario_labels[scenario_idx]} - {quant}",
                    ),
                    row=row,
                    col=col,
                )

            # Plot the median line for the current scenario
            if 0.5 in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[0.5],
                        line={"color": colors[scenario_idx].replace("{alpha}", "1.0")},
                        name=f"{scenario_labels[scenario_idx]} - Median",
                    ),
                    row=row,
                    col=col,
                )

            # Handle specific indicators with uncertainty bounds
            if ind in [
                "prevalence_smear_positive",
                "adults_prevalence_pulmonary",
                "incidence",
            ]:
                target_series = target_data[f"{ind}_target"]
                lower_bound_series = target_data[f"{ind}_lower_bound"]
                upper_bound_series = target_data[f"{ind}_upper_bound"]

                filtered_target = target_series[
                    (target_series.index >= plot_start_date)
                    & (target_series.index <= plot_end_date)
                ]
                filtered_lower_bound = lower_bound_series[
                    (lower_bound_series.index >= plot_start_date)
                    & (lower_bound_series.index <= plot_end_date)
                ]
                filtered_upper_bound = upper_bound_series[
                    (upper_bound_series.index >= plot_start_date)
                    & (upper_bound_series.index <= plot_end_date)
                ]

                # Plot the target point estimates with round markers
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target.values,
                        mode="markers",
                        marker={"size": 5.0, "color": "black", "line": {"width": 0.8}},
                        name="",  # No name
                    ),
                    row=row,
                    col=col,
                )

                # Plot the vertical solid lines for uncertainties connecting upper and lower bounds
                for date in filtered_target.index:
                    fig.add_trace(
                        go.Scatter(
                            x=[date, date],
                            y=[
                                filtered_lower_bound.loc[date],
                                filtered_upper_bound.loc[date],
                            ],
                            mode="lines",
                            line={"color": "black", "width": 1.0},
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

                # Plot the lower bounds with dashed markers
                fig.add_trace(
                    go.Scatter(
                        x=filtered_lower_bound.index,
                        y=filtered_lower_bound.values,
                        mode="markers",
                        marker={"size": 5.0, "color": "black", "symbol": "x"},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                # Plot the upper bounds with dashed markers
                fig.add_trace(
                    go.Scatter(
                        x=filtered_upper_bound.index,
                        y=filtered_upper_bound.values,
                        mode="markers",
                        marker={"size": 5.0, "color": "black", "symbol": "x"},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            else:
                # For other indicators, just plot the point estimate
                if ind in target_data.keys():
                    target = target_data[ind]
                    filtered_target = target[
                        (target.index >= plot_start_date)
                        & (target.index <= plot_end_date)
                    ]

                    # Plot the target point only if it's not a list
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_target.index,
                            y=filtered_target,
                            mode="markers",
                            marker={"size": 5.0, "color": "black"},
                            name="",  # No name
                        ),
                        row=row,
                        col=col,
                    )

            # Update x-axis range to fit the filtered data
            x_min = max(filtered_data.index.min(), plot_start_date)
            x_max = filtered_data.index.max()
            fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

            # Update y-axis range dynamically for each subplot
            y_min = 0
            y_max = max(
                filtered_data.max().max(),
                (
                    max(
                        [
                            filtered_target.max()
                            for filtered_target in [
                                filtered_target,
                                filtered_lower_bound,
                                filtered_upper_bound,
                            ]
                        ]
                    )
                    if ind
                    in [
                        "prevalence_smear_positive",
                        "adults_prevalence_pulmonary",
                        "incidence",
                    ]
                    else (
                        filtered_target.max()
                        if ind in target_data.keys()
                        else float("-inf")
                    )
                ),
            )
            y_range = y_max - y_min
            padding = 0.1 * y_range
            fig.update_yaxes(range=[y_min - padding, y_max + padding], row=row, col=col)

    # Update layout for the whole figure
    fig.update_layout(xaxis_title="", yaxis_title="", showlegend=True)

    return fig


def plot_covid_scenarios_comparison(
    diff_quantiles, indicators, years, plot_type="abs", n_cols=1
):
    """
    Plot the median differences with error bars indicating the range from 0.025 to 0.975 quantiles
    for given indicators across multiple years in one plot per indicator.

    Args:
        diff_quantiles: A dictionary containing the calculated quantile differences (output from `calculate_diff_quantiles`).
        indicators: List of indicators to plot.
        years: List of years for which to plot the data.
        plot_type: "abs" for absolute differences, "rel" for relative differences.
        n_cols: Number of columns in the subplot layout.

    Returns:
        A Plotly figure with separate plots for each indicator, each containing horizontal bars for multiple years.
    """
    nrows = len(indicators)
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [indicator_names.get(ind, ind.replace("_", " ")) for ind in indicators],
        share_y=True,  # Use a shared y-axis for all subplots
    )

    # Dictionary to store consistent colors for each indicator
    indicator_colors = {
        "incidence": "rgba(0, 123, 255)",  # Blue for the first indicator
        "mortality_raw": "rgba(40, 167, 69)",  # Green for the second indicator
    }

    # Plot the data
    for ind_index, ind in enumerate(indicators):
        color = indicator_colors.get(
            ind, "rgba(0, 123, 255)"
        )  # Default to blue if not specified

        # Ensure years are in the index
        if not all(year in diff_quantiles[plot_type][ind].index for year in years):
            raise ValueError(
                f"Some years are missing in the index for indicator: {ind}"
            )

        median_diffs = []
        lower_diffs = []
        upper_diffs = []
        for year in years:
            quantile_data = diff_quantiles[plot_type][ind].loc[year]
            median_diffs.append(round(quantile_data[0.5]))
            lower_diffs.append(round(quantile_data[0.025]))
            upper_diffs.append(round(quantile_data[0.975]))

        # Plot a single trace for all years for this indicator
        fig.add_trace(
            go.Bar(
                y=[
                    str(int(year)) for year in years
                ],  # Convert years to integers and then to strings
                x=median_diffs,  # Median differences
                orientation="h",
                marker=dict(color=color),  # Use consistent color for each indicator
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        upper - median
                        for upper, median in zip(upper_diffs, median_diffs)
                    ],  # Upper bound error
                    arrayminus=[
                        median - lower
                        for median, lower in zip(median_diffs, lower_diffs)
                    ],  # Lower bound error
                    color=color,  # Ensure error bars are the same color
                ),
                showlegend=False,  # No legend for the bar
            ),
            row=ind_index + 1,
            col=1,
        )

    # Update layout specifics
    fig.update_layout(
        title="",  # No title for the plot
        yaxis_title="",  # Removing the y-axis title
        xaxis_title="",  # Removing the x-axis title
        barmode="group",  # Group bars by year within each subplot
        showlegend=False,  # Remove the legend
    )

    # Apply consistent y-axis settings with reversed year order
    for i in range(1, nrows + 1):
        fig.update_yaxes(
            tickvals=[
                str(int(year)) for year in reversed(years)
            ],  # Reverse and format years as integers
            tickformat="d",  # Ensure years are displayed as integers
            showline=False,  # Remove the y-axis line
            ticks="",  # Remove tick marks
            row=i,
            col=1,
            categoryorder="array",  # Preserve the reversed order of years
            categoryarray=[
                str(int(year)) for year in reversed(years)
            ],  # Ensure the provided reversed order is followed
        )
        fig.update_xaxes(
            range=[0, None], row=i, col=1
        )  # Ensure x-axes start at zero for clarity

    # Add a vertical annotation labeled "Year"
    fig.add_annotation(
        text="Year",
        xref="paper",
        yref="paper",
        x=-0.05,  # Position it to the left of the y-axis
        y=0.5,  # Center it vertically in the plot
        showarrow=False,
        font=dict(size=14),
        textangle=-90,  # Rotate the text to be vertical
    )

    return fig


def tabulate_calib_results(
    idata: az.data.inference_data.InferenceData, params_name
) -> pd.DataFrame:
    """
    Get tabular outputs from calibration inference object,
    except for the dispersion parameters, and standardize formatting.

    Args:
        idata: InferenceData object from ArviZ containing calibration outputs.
        priors: List of parameter names as strings.

    Returns:
        Calibration results table in standard format.
    """
    # Generate summary table
    table = az.summary(idata)

    # Filter out dispersion parameters
    table = table[~table.index.str.contains("_dispersion")]

    # Round and format the relevant columns
    for col_to_round in [
        "mean",
        "sd",
        "hdi_3%",
        "hdi_97%",
        "ess_bulk",
        "ess_tail",
        "r_hat",
    ]:
        table[col_to_round] = table.apply(
            lambda x: str(round(x[col_to_round], 3)), axis=1
        )

    # Create the HDI column
    table["hdi"] = table.apply(lambda x: f'{x["hdi_3%"]} to {x["hdi_97%"]}', axis=1)

    # Drop unnecessary columns
    table = table.drop(["mcse_mean", "mcse_sd", "hdi_3%", "hdi_97%"], axis=1)

    # Rename columns for standardized format
    table.columns = [
        "Mean",
        "Standard deviation",
        "ESS bulk",
        "ESS tail",
        "\\textit{\^{R}}",
        "High-density interval",
    ]
    table.index = table.index.map(lambda x: params_name.get(x, x))
    table.index.name = "Parameter"
    return table
