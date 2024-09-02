import arviz as az
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict

from tbdynamics.constants import indicator_names, quantiles
from tbdynamics.utils import get_row_col_for_subplots, get_standard_subplot_fig
from tbdynamics.constants import indicator_names, scenario_names
from .utils import convert_prior_to_numpyro

# Define the custom template
extended_layout = pio.templates["simple_white"].layout

# Update the layout with custom settings
extended_layout.update(
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        title_font=dict(
            family="Arial",  # Use Arial Black for bold font
            size=12,
            color="black",
        ),
        tickfont=dict(
            family="Arial", size=12, color="black"  # Set x-axis tick font to Arial
        ),
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        title_font=dict(
            family="Arial",  # Use Arial Black for bold font
            size=12,
            color="black",
        ),
        tickfont=dict(
            family="Arial", size=12, color="black"  # Set y-axis tick font to Arial
        ),
    ),
    title=dict(
        font=dict(
            family="Arial",  # Use Arial Black for bold font
            size=14,
            color="black",
        )
    ),
    font=dict(family="Arial", size=14),  # General font settings for the figure
    legend=dict(
        font=dict(family="Arial", size=12, color="black")  # Set legend font to Arial
    ),
)


# Create a new template using the updated layout
custom_template = go.layout.Template(layout=extended_layout)
# Register the custom template
pio.templates["custom_template"] = custom_template
# Set the custom template as the default
pio.templates.default = "custom_template"


def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
    history: bool = False,  # New argument
    show_title: bool = True,
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
        history: If True, set tick intervals to 50 years.

    Returns:
        The interactive Plotly figure.
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        (
            [
                (
                    f"<b>{indicator_names[ind]}</b>"
                    if ind in indicator_names
                    else f"<b>{ind.replace('_', ' ').capitalize()}</b>"
                )
                for ind in indicators
            ]
            if show_title
            else ["" for _ in indicators]
        ),  # Conditionally set titles with bold tags
    )

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = quantile_outputs[ind]

        # Set plot_start_date to 2005 if the indicator is "prevalence_smear_positive"
        current_plot_start_date = (
            2005 if ind == "prevalence_smear_positive" else plot_start_date
        )

        # Filter data by date range
        filtered_data = data[
            (data.index >= current_plot_start_date) & (data.index <= plot_end_date)
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
                (target_series.index >= current_plot_start_date)
                & (target_series.index <= plot_end_date)
            ]
            filtered_lower_bound = lower_bound_series[
                (lower_bound_series.index >= current_plot_start_date)
                & (lower_bound_series.index <= plot_end_date)
            ]
            filtered_upper_bound = upper_bound_series[
                (upper_bound_series.index >= current_plot_start_date)
                & (upper_bound_series.index <= plot_end_date)
            ]

            # Plot the point estimates with error bars
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target.values,
                    mode="markers",
                    marker={"size": 4.0, "color": "red"},
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=filtered_upper_bound - filtered_target,
                        arrayminus=filtered_target - filtered_lower_bound,
                        color="red",
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
                    (target.index >= current_plot_start_date)
                    & (target.index <= plot_end_date)
                ]

                # Plot the target point estimates
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target,
                        mode="markers",
                        marker={"size": 4.0, "color": "red"},
                        name="",  # No name for legend
                    ),
                    row=row,
                    col=col,
                )

        # Update x-axis range to fit the filtered data
        x_min = max(filtered_data.index.min(), current_plot_start_date)
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
        padding = 0.05 * y_range  # Consistent padding for all scenarios
        fig.update_yaxes(range=[y_min - padding, y_max + padding], row=row, col=col)

    tick_interval = 50 if history else 1  # Set tick interval based on history
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=tick_interval,  # Adjust tick increment
    )

    # Update layout for the whole figure
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=10, r=5, t=30, b=40),
    )

    return fig


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


def plot_trace(idata: az.InferenceData, params_name: dict):
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
        [
            (
                indicator_names[ind]
                if ind in indicator_names
                else ind.replace("_", " ").capitalize()
            )
            for ind in indicators
        ],
        share_y=True,  # Use a shared y-axis for all subplots
    )
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    for ind_index, ind in enumerate(indicators):
        color = indicator_colors.get(
            ind, "rgba(0, 123, 255)"
        )  # Default to blue if not specified

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

        fig.add_trace(
            go.Bar(
                y=[str(int(year)) for year in years],  # Convert years to strings
                x=median_diffs,  # Median differences
                orientation="h",
                marker=dict(color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        upper - median
                        for upper, median in zip(upper_diffs, median_diffs)
                    ],
                    arrayminus=[
                        median - lower
                        for median, lower in zip(median_diffs, lower_diffs)
                    ],
                    color="black",
                    thickness=1.5,
                    width=3,
                ),
                showlegend=False,
            ),
            row=ind_index + 1,
            col=1,
        )

    fig.update_layout(
        title="Rererence: counterfactual no COVID-19",
        yaxis_title="",
        xaxis_title="",
        barmode="group",
        showlegend=False,
    )

    # Ensure the y-axis is visible by adjusting its properties
    for i in range(1, nrows + 1):
        fig.update_yaxes(
            tickvals=[str(int(year)) for year in reversed(years)],
            tickformat="d",
            showline=True,  # Ensure the line is shown
            linecolor="black",  # Set the color of the y-axis line
            linewidth=1,  # Adjust the width of the y-axis line
            mirror=True,  # Ensure the axis line is mirrored
            ticks="outside",  # Show ticks outside the plot
            row=i,
            col=1,
            categoryorder="array",
            categoryarray=[str(int(year)) for year in reversed(years)],
        )
        fig.update_xaxes(
            range=[0, None], row=i, col=1
        )  # Ensure x-axes start at zero for clarity

    fig.add_annotation(
        text="Year",
        xref="paper",
        yref="paper",
        x=-0.05,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
        textangle=-90,
    )

    return fig


def plot_detection_scenario_comparison_box(diff_quantiles, indicators, plot_type="abs"):
    """
    Plot the quantile differences for given indicators across multiple scenarios.

    Args:
        diff_quantiles (dict): The quantile difference data structured as a dictionary.
        indicators (list): List of indicators to plot (e.g., 'cumulative_diseased', 'cumulative_deaths').
        plot_type (str): "abs" for absolute differences, "rel" for relative differences.

    Returns:
        fig: A Plotly figure object.
    """
    nrows = len(indicators)
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    fig = go.Figure()

    for i, indicator in enumerate(list(indicators)):
        color = indicator_colors.get(indicator, "rgba(0, 123, 255)")

        # Extract data for the given indicator and plot_type
        scenarios = list(diff_quantiles.keys())  # Extract scenario names
        medians = []
        lower_errors = []
        upper_errors = []

        for scenario in scenarios:
            median_val = diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.500
            ]
            lower_val = diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.025
            ]
            upper_val = diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.975
            ]

            medians.append(median_val)
            lower_errors.append(median_val - lower_val)
            upper_errors.append(upper_val - median_val)

        # Add trace for this indicator in the specified order
        fig.add_trace(
            go.Bar(
                y=[
                    scenario_names.get(
                        scenario, scenario.replace("_", " ").capitalize()
                    )
                    for scenario in scenarios
                ],  # Descriptive scenario names
                x=medians,  # Median values on x-axis
                orientation="h",
                marker=dict(color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=upper_errors,  # Upper bound error
                    arrayminus=lower_errors,  # Lower bound error
                    color="black",  # Black color for error bars
                    thickness=1.5,  # Thicker error bars
                    width=3,  # Wider error bars
                ),
                name=indicator.replace(
                    "_", " "
                ).capitalize(),  # Use indicator name for legend
            )
        )

    # Ensure traces are ordered according to indicators list
    fig.data = sorted(
        fig.data,
        key=lambda trace: indicators.index(trace.name.lower().replace(" ", "_")),
    )

    # Update layout with tight margins and ordered legend
    fig.update_layout(
        title={"text": "Reference: Status-quo scenario", "x": 0.5},
        xaxis_title="",
        yaxis_title="",
        barmode="group",
        height=320,  # Adjust height based on the number of rows
        margin=dict(
            l=20, r=5, t=40, b=40
        ),  # Tight layout with more bottom margin for legend
        yaxis=dict(
            tickangle=-45,  # Rotate y-axis labels by 45 degrees
            categoryorder="array",  # Ensure the order follows the scenarios
            categoryarray=[
                scenario_names.get(scenario, scenario.replace("_", " ").capitalize())
                for scenario in reversed(scenarios)
            ],
        ),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=-0.3,  # Position the legend below the plot
            xanchor="center",
            x=0.5,
            itemsizing="constant",  # Consistent item sizing
            traceorder="normal",  # Keep the legend order as per the traces added
        ),
    )

    return fig


def plot_covid_scenarios_comparison_combined(
    diff_quantiles, indicators, years, plot_type="abs", n_cols=1
):
    """
    Plot the median differences with error bars indicating the range from 0.025 to 0.975 quantiles
    for given indicators across multiple years in a single plot.

    Args:
        diff_quantiles: A dictionary containing the calculated quantile differences (output from `calculate_diff_quantiles`).
        indicators: List of indicators to plot.
        years: List of years for which to plot the data.
        plot_type: "abs" for absolute differences, "rel" for relative differences.
        n_cols: (Deprecated) No longer relevant since all indicators are plotted in one plot.

    Returns:
        A Plotly figure with all indicators plotted together, each containing horizontal bars for multiple years.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    for ind in indicators:
        color = indicator_colors.get(
            ind, "rgba(0, 123, 255)"
        )  # Default to blue if not specified

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

        fig.add_trace(
            go.Bar(
                y=[str(int(year)) for year in years],  # Convert years to strings
                x=median_diffs,  # Median differences
                orientation="h",
                name=indicator_names.get(ind, ind.replace("_", " ").capitalize()),
                marker=dict(color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        upper - median
                        for upper, median in zip(upper_diffs, median_diffs)
                    ],
                    arrayminus=[
                        median - lower
                        for median, lower in zip(median_diffs, lower_diffs)
                    ],
                    color="black",
                    thickness=1.5,
                    width=3,
                ),
            )
        )

    fig.update_layout(
        title={
            "text": "Reference: Counterfactual no COVID-19",
            "x": 0.5,
            # 'y': 0.95,
            "xanchor": "center",
            "yanchor": "top",
        },
        yaxis_title="",
        xaxis_title="",
        height=320,
        barmode="group",
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal orientation for the legend
            yanchor="bottom",  # Anchor the legend at the bottom
            y=-0.3,  # Move the legend below the x-axis
            xanchor="center",  # Center the legend horizontally
            x=0.5,
        ),
        margin=dict(l=20, r=5, t=40, b=40),
    )

    # Ensure the y-axis is visible by adjusting its properties
    fig.update_yaxes(
        tickvals=[str(int(year)) for year in reversed(years)],
        tickformat="d",
        showline=True,  # Ensure the line is shown
        linecolor="black",  # Set the color of the y-axis line
        linewidth=1,  # Adjust the width of the y-axis line
        mirror=True,  # Ensure the axis line is mirrored
        ticks="outside",  # Show ticks outside the plot
        categoryorder="array",
        categoryarray=[str(int(year)) for year in reversed(years)],
    )
    fig.update_xaxes(range=[0, None])  # Ensure x-axes start at zero for clarity

    return fig


def hex_to_rgb(hex_color):
    """
    Convert hex color (e.g., '#636EFA') to an rgb color tuple (e.g., (99, 110, 250)).
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def plot_scenario_output_ranges(
    scenario_outputs: Dict[str, Dict[str, pd.DataFrame]],
    indicators: List[str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023,
    max_alpha: float = 0.7,
) -> go.Figure:
    """
    Plot the credible intervals for each indicator in a single plot across multiple scenarios.

    Args:
        scenario_outputs: Dictionary containing scenario outputs, with scenario names as keys.
        indicators: List of indicators to plot.
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
            (
                indicator_names[ind]
                if ind in indicator_names
                else ind.replace("_", " ").capitalize()
            )
            for ind in indicators
        ],
    )

    base_color = (0, 30, 180)  # Base scenario RGB color as a tuple
    target_color = "red"  # Use a consistent color for 2035 target points
    scenario_colors = (
        px.colors.qualitative.Plotly
    )  # Use Plotly colors for other scenarios

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        for scenario_idx, (scenario_name, quantile_outputs) in enumerate(
            scenario_outputs.items()
        ):
            display_name = scenario_names.get(
                scenario_name, scenario_name
            )  # Get display name

            # Determine the color to use for this scenario
            if (
                scenario_name.lower() == "base_scenario"
            ):  # Check if it's the base scenario
                rgb_color = base_color
            else:
                hex_color = scenario_colors[scenario_idx % len(scenario_colors)]
                rgb_color = hex_to_rgb(hex_color)  # Convert hex to RGB tuple

            data = quantile_outputs[ind]

            # Filter data by date range
            filtered_data = data[
                (data.index >= plot_start_date) & (data.index <= plot_end_date)
            ]

            # Show the legend only for the first indicator
            show_legend = i == 0

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
                fill_color = f"rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, {alpha})"  # Use rgba with appropriate alpha

                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[quant],
                        fill="tonexty",
                        fillcolor=fill_color,
                        line={"width": 0},
                        name=(
                            display_name if quant == 0.5 and show_legend else None
                        ),  # Show legend only for the first figure
                        showlegend=quant == 0.5
                        and show_legend,  # Show legend only for the first figure
                        legendgroup=display_name,
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
                        line={
                            "color": f"rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})"
                        },
                        name=(
                            display_name if show_legend else None
                        ),  # Show legend only for the first figure
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        # Add specific points for "incidence" and "mortality_raw" at 2035 with consistent color
        if ind == "incidence":
            fig.add_trace(
                go.Scatter(
                    x=[2035],
                    y=[10],
                    mode="markers+text",
                    marker=dict(size=4, color=target_color),
                    name="2035 End TB Target",
                    showlegend=True if i == 0 else False,  # Show legend only once
                    legendgroup="Target",
                ),
                row=row,
                col=col,
            )

        if ind == "mortality_raw":
            fig.add_trace(
                go.Scatter(
                    x=[2035],
                    y=[900],
                    mode="markers+text",
                    marker=dict(size=4, color=target_color),
                    showlegend=False,  # No additional legend entry for repeated points
                    legendgroup="Target",
                ),
                row=row,
                col=col,
            )

    # Update layout for the whole figure
    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        showlegend=True,
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=-0.25,  # Position the legend below the plot
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
    )

    # Update x-axis ticks to increase by 1 year
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=1,  # Set tick increment to 1 year
    )

    return fig


def plot_scenario_output_ranges_by_col(
    scenario_outputs: Dict[str, Dict[str, pd.DataFrame]],
    plot_start_date: float = 2025.0,
    plot_end_date: float = 2036.0,
    max_alpha: float = 0.7,
) -> go.Figure:
    """
    Plot the credible intervals for incidence and mortality_raw with scenarios as rows.

    Args:
        scenario_outputs: Dictionary containing scenario outputs, with scenario names as keys.
        plot_start_date: Start year for the plot as float.
        plot_end_date: End year for the plot as float.
        max_alpha: Maximum alpha value to use in patches.

    Returns:
        The interactive Plotly figure.
    """
    indicators = ["incidence", "mortality_raw"]
    n_scenarios = len(scenario_outputs)
    n_cols = len(indicators)

    # Define the color scheme using Plotly's qualitative palette
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    # Define the scenario titles manually
    y_axis_titles = ["Status-quo scenario", "Scenario 1", "Scenario 2", "Scenario 3"]

    # Create the subplots without shared y-axis
    fig = make_subplots(
        rows=n_scenarios,
        cols=n_cols,
        shared_yaxes=False,
        vertical_spacing=0.05,
        horizontal_spacing=0.07,
        column_titles=[
            "<b>TB incidence (per 100,000 populations)</b>",
            "<b>TB deaths</b>",
        ],  # Titles for columns
    )

    target_color = "red"  # Use a consistent color for 2035 target points
    show_legend_for_target = True  # To ensure the legend is shown only once

    for scenario_idx, (scenario_key, quantile_outputs) in enumerate(
        scenario_outputs.items()
    ):
        row = scenario_idx + 1

        # Get the formatted scenario name from the manual list
        display_name = y_axis_titles[scenario_idx]

        for j, indicator_name in enumerate(indicators):
            col = j + 1
            color = indicator_colors[indicator_name]
            data = quantile_outputs[
                indicator_name
            ]  # Access the correct indicator data for the scenario

            # Ensure the index is of float type and filter data by date range
            filtered_data = data[
                (data.index >= plot_start_date) & (data.index <= plot_end_date)
            ]

            for quant in quantiles:
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
                fill_color = f"rgba({hex_to_rgb(color)[0]}, {hex_to_rgb(color)[1]}, {hex_to_rgb(color)[2]}, {alpha})"  # Ensure correct alpha blending

                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[quant],
                        fill="tonexty",
                        fillcolor=fill_color,
                        line={"width": 0},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Plot the median line (0.5 quantile)
            if 0.5 in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[0.5],
                        line={"color": color},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Add specific points for "incidence" and "mortality_raw" at 2035 with consistent color and size
            if indicator_name == "incidence":
                fig.add_trace(
                    go.Scatter(
                        x=[2035.0],
                        y=[10],
                        mode="markers",
                        marker=dict(size=4, color=target_color),
                        name="2035 End TB Target" if show_legend_for_target else None,
                        showlegend=show_legend_for_target,
                        legendgroup="Target",
                    ),
                    row=row,
                    col=col,
                )
                show_legend_for_target = False  # Only show legend once

            if indicator_name == "mortality_raw":
                fig.add_trace(
                    go.Scatter(
                        x=[2035.0],
                        y=[900],
                        mode="markers",
                        marker=dict(size=4, color=target_color),
                        showlegend=False,
                        legendgroup="Target",
                    ),
                    row=row,
                    col=col,
                )
            fig.update_yaxes(
                title_text=display_name,
                # title_standoff=15,
                title_font=dict(size=12),
                row=row,
                col=1,
            )

            # Only show x-ticks for the last row
            if row < n_scenarios:
                fig.update_xaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(
        height=680,  # Adjust height based on the number of scenarios
        title="",
        xaxis_title="",  # No title on individual x-axes
        # yaxis_title="",
        showlegend=True,
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=-0.15,  # Position the legend below the plot
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(l=20, r=5, t=30, b=40),  # Adjust margins to accommodate titles
    )

    # Update x-axis ticks to increase by 1 year
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=1,  # Set tick increment to 1 year
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
