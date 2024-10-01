import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from typing import List, Dict

from tbdynamics.constants import indicator_names, indicator_legends, quantiles
from tbdynamics.utils import get_row_col_for_subplots, get_standard_subplot_fig
from tbdynamics.constants import indicator_names, scenario_names
from tbdynamics.calibration.utils import calculate_loo_for_covid


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
            family="Arial", size=10, color="black"  # Set x-axis tick font to Arial
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
            size=10,
            color="black",
        ),
        tickfont=dict(
            family="Arial", size=10, color="black"  # Set y-axis tick font to Arial
        ),
    ),
    title=dict(
        font=dict(
            family="Arial",  # Use Arial Black for bold font
            size=12,
            color="black",
        )
    ),
    font=dict(family="Arial", size=12),  # General font settings for the figure
    legend=dict(
        font=dict(family="Arial", size=12, color="black")  # Set legend font to Arial
    ),
)
# Create a new template using the updated layout
custom_template = go.layout.Template(layout=extended_layout)
# Register the custom template
pio.templates["custom_template"] = custom_template
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
        quantile_outputs: DataFrames containing derived outputs of interest for each analysis type.
        target_data: Calibration targets.
        indicators: List of indicators to plot.
        n_cols: Number of columns for the subplots.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value to use in patches.
        history: If True, set tick intervals to 50 years.

    Returns:
        The interactive Plotly figure.
    """
    # Assume 'indicator_names' and 'quantiles' are imported from external modules
    # Assume 'indicator_legends' is also imported

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
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)  # Set font size for titles

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
                    showlegend=False,  # Hide legend for quantile traces
                ),
                row=row,
                col=col,
            )

        # Plot the median line
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[0.5],
                line={"color": "black"},
                name="Median",
                showlegend=False,  # Hide legend for median line
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
                        thickness=1,
                        width=2,
                    ),
                    name="",  # No name for legend
                    showlegend=False,  # Hide legend for point estimates
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
                        showlegend=False,  # Hide legend for point estimates
                    ),
                    row=row,
                    col=col,
                )

        # Add indicator legend as annotation at the bottom right of each subplot
        legend_text = indicator_legends.get(ind, "")
        if legend_text and not history:
            # Compute axis ID for the subplot
            axis_id = (row - 1) * n_cols + col
            # Determine xref and yref for the annotation
            if axis_id == 1:
                xref = "x domain"
                yref = "y domain"
            else:
                xref = f"x{axis_id} domain"
                yref = f"y{axis_id} domain"

            # Add the annotation with a red point before the legend text
            fig.add_annotation(
                text=f'<span style="color:red; font-size:12px">&#9679;</span> <span style="font-size:12px">{legend_text}</span>',
                x=0.99,  # Right end of the x-axis domain
                y=0.05,  # Bottom of the y-axis domain
                xref=xref,
                yref=yref,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                bordercolor="black",  # Color of the border
                borderwidth=1,  # Thickness of the border
                # borderpad=5,  # Padding between text and border
            )

        # Update x-axis range to fit the filtered data
        x_min = max(filtered_data.index.min(), current_plot_start_date)
        x_max = filtered_data.index.max()
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

    tick_interval = 50 if history else 2  # Set tick interval based on history
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


def plot_outputs_for_covid(
    covid_outputs: Dict[str, Dict[str, pd.DataFrame]],
    target_data: Dict[str, pd.Series],
    plot_start_date: int = 2011,
    plot_end_date: int = 2025,
    max_alpha: float = 0.7,
) -> go.Figure:
    """
    Plot the "notification" indicator for each scenario in a 2x2 grid with subplot titles
    based on configuration keys, include target points, and show LOO-IC in the bottom left.

    Args:
        covid_outputs: Dictionary containing outputs for each scenario.
        target_data: Calibration targets.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value to use in patches.

    Returns:
        A Plotly figure with all scenarios plotted in a 2x2 grid, with LOO-IC values annotated.
    """

    # Custom titles for each subplot
    covid_titles = {
        "no_covid": "Assumption 1",
        "case_detection_reduction_only": "Assumption 2",
        "contact_reduction_only": "Assumption 3",
        "detection_and_contact_reduction": "Assumption 4",
    }

    # Calculate the LOO-IC for each scenario
    loo_results = calculate_loo_for_covid(covid_outputs)

    # Define the 2x2 grid
    n_cols = 2
    n_rows = int(np.ceil(len(covid_titles) / n_cols))

    # Create the subplot figure
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        subplot_titles=[
            f"<b>{covid_titles.get(scenario_name, scenario_name.replace('_', ' ').capitalize())}</b>"
            for scenario_name in covid_titles.keys()
        ],
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)  # Set font size for titles

    # Loop through each scenario and plot it on the grid
    for i, (scenario_name, title) in enumerate(covid_titles.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1
        quantile_outputs = covid_outputs[scenario_name]["indicator_outputs"]
        data = quantile_outputs["notification"]

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
                    name=f"{scenario_name} {quant}",
                    showlegend=False,  # Disable legend to avoid clutter
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
                    name=f"{scenario_name} median",
                    showlegend=False,  # Disable legend to avoid clutter
                ),
                row=row,
                col=col,
            )

        # Add target points if available
        if "notification" in target_data:
            targets = target_data["notification"]
            fig.add_trace(
                go.Scatter(
                    x=targets.index,
                    y=targets,
                    mode="markers",
                    marker=dict(size=4, color="red"),
                    name="Target",
                    showlegend=False,  # Hide legend for targets
                ),
                row=row,
                col=col,
            )

        # Add LOO-IC annotation to the bottom left of the subplot
        loo_ic = loo_results.get(scenario_name, "N/A")
        fig.add_annotation(
            text=f"Loo-IC: {loo_ic:.2f}",
            xref=f"x{i+1}",  # Refers to the x-axis of the current subplot
            yref=f"y{i+1}",  # Refers to the y-axis of the current subplot
            x=plot_start_date + 0.5,
            y=5000.0,  # Place it near the bottom left
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="left",
            yanchor="bottom",
            bordercolor="black",  # Color of the border
            borderwidth=1,  # Thickness of the border
        )
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=2,  # Adjust tick increment
    )
    # Update layout for the whole figure
    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        height=600,  # Set the figure height to 600 pixels
        margin=dict(l=10, r=5, t=30, b=40),
        font=dict(size=8),
    )

    return fig


def plot_covid_configs_comparison(
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
        text="<b>Year</b>",
        xref="paper",
        yref="paper",
        x=-0.05,
        y=0.5,
        showarrow=False,
        font=dict(size=12),
        textangle=-90,
    )

    return fig


def plot_covid_configs_comparison_box(
    diff_quantiles: Dict[str, Dict[str, pd.DataFrame]], plot_type: str = "abs"
):
    """
    Plot the median differences with error bars indicating the range from 0.025 to 0.975 quantiles
    for given indicators across multiple years in a single plot.

    Args:
        diff_quantiles: A dictionary containing the calculated quantile differences (output from `calculate_diff_quantiles`).
        plot_type: "abs" for absolute differences, "rel" for relative differences.

    Returns:
        A Plotly figure with all indicators plotted together, each containing horizontal bars for multiple years.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    indicators = diff_quantiles[plot_type].keys()
    years = diff_quantiles[plot_type][list(indicators)[0]].index

    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    for ind in indicators:
        color = indicator_colors.get(
            ind, "rgba(0, 123, 255)"
        )  # Default to blue if not specified

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
                name=ind.replace("_", " ").capitalize(),
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
                    thickness=1,
                    width=2,
                ),
            )
        )

    fig.update_layout(
        title={
            "text": "Reference: COVID-19 has no effect on TB notifications",
            "x": 0.08,
            "xanchor": "left",
            "yanchor": "top",
        },
        yaxis_title="",
        xaxis_title="",
        height=320,
        barmode="group",
        legend=dict(
            orientation="h",  # Horizontal orientation for the legend
            yanchor="bottom",  # Anchor the legend at the bottom
            y=-0.2,  # Move the legend below the x-axis
            xanchor="center",  # Center the legend horizontally
            x=0.5,
            font=dict(size=12),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=20, r=5, t=30, b=40),
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


def plot_scenario_output_ranges_by_col(
    scenario_outputs: Dict[str, Dict[str, pd.DataFrame]],
    plot_start_date: float = 2025.0,
    plot_end_date: float = 2036.0,
    max_alpha: float = 0.7,
) -> go.Figure:
    """
    Plot the credible intervals for incidence and mortality_raw with scenarios as rows.
    Also plot 2030 SDG targets in purple and 2035 End TB targets in red.

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
    n_cols = 2

    # Define the color scheme using Plotly's qualitative palette
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    # Define the scenario titles manually
    y_axis_titles = [
        "<i>'Status-quo'</i> scenario",
        "Scenario 1",
        "Scenario 2",
        "Scenario 3",
    ]

    # Create the subplots without shared y-axis
    fig = make_subplots(
        rows=n_scenarios,
        cols=n_cols,
        shared_yaxes=False,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        column_titles=[
            "<b>TB incidence (/100,000/y)</b>",
            "<b>TB deaths</b>",
        ],  # Titles for columns
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)  # Set font size for titles

    # Colors for the targets
    sdg_target_color = "purple"
    end_tb_target_color = "red"

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

            # Add specific points for "incidence" and "mortality_raw" at 2030 SDG and 2035 End TB targets
            if indicator_name == "incidence":
                # 2030 SDG Target (Purple) - Legend rank 1
                fig.add_trace(
                    go.Scatter(
                        x=[2030.0],
                        y=[31],  # 2030 SDG target for incidence
                        mode="markers",
                        marker=dict(size=4, color=sdg_target_color),
                        name="2030 SDG Target",
                        showlegend=show_legend_for_target,
                        legendgroup="Targets",  # Group both targets together
                        legendrank=2,  # Set legend rank to ensure it appears first
                    ),
                    row=row,
                    col=col,
                )
                # 2035 End TB Target (Red) - Legend rank 2
                fig.add_trace(
                    go.Scatter(
                        x=[2035.0],
                        y=[10],  # 2035 End TB target for incidence
                        mode="markers",
                        marker=dict(size=4, color=end_tb_target_color),
                        name="2035 End TB Target",
                        showlegend=show_legend_for_target,
                        legendgroup="Targets",  # Group both targets together
                        legendrank=1,  # Set legend rank to ensure it appears second
                    ),
                    row=row,
                    col=col,
                )
                show_legend_for_target = False  # Only show legend once

            if indicator_name == "mortality_raw":
                # 2030 SDG Target (Purple) - no legend this time, but keep the same group
                fig.add_trace(
                    go.Scatter(
                        x=[2030.0],
                        y=[1913],  # 2030 SDG target for deaths
                        mode="markers",
                        marker=dict(size=4, color=sdg_target_color),
                        showlegend=False,
                        legendgroup="Targets",
                    ),
                    row=row,
                    col=col,
                )
                # 2035 End TB Target (Red) - no legend this time, but keep the same group
                fig.add_trace(
                    go.Scatter(
                        x=[2035.0],
                        y=[957],  # 2035 End TB target for deaths
                        mode="markers",
                        marker=dict(size=4, color=end_tb_target_color),
                        showlegend=False,
                        legendgroup="Targets",
                    ),
                    row=row,
                    col=col,
                )

            fig.update_yaxes(
                title_text=f"<b>{display_name}</b>",
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
        xaxis_title="",
        showlegend=True,
        legend=dict(
            title="",
            orientation="v",  # Vertical orientation for legend
            yanchor="top",
            y=0.2,  # Position at the top of the last plot
            xanchor="right",
            x=1,  # Position to the right
            # font=dict(size=12),
            tracegroupgap=0,  # Remove any gap between traces
            itemwidth=40,  # Ensure enough space for both target legends to fit
            bordercolor="black",  # Set the border color (e.g., black)
            borderwidth=1,  # Set the border width
        ),
        margin=dict(l=20, r=5, t=30, b=40),  # Adjust margins to accommodate titles
    )

    # Update x-axis ticks to increase by 1 year
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=2,  # Set tick increment to 1 year
    )

    return fig


def plot_detection_scenarios_comparison_box(
    diff_quantiles: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    plot_type: str = "abs",
):
    """
    Plot the quantile differences for the fixed indicators across multiple scenarios.

    Args:
        diff_quantiles (dict): The quantile difference data structured as a dictionary.
        plot_type (str): "abs" for absolute differences, "rel" for relative differences.

    Returns:
        fig: A Plotly figure object.
    """
    # Fixed indicators
    indicators = ["cumulative_diseased", "cumulative_deaths"]
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    fig = go.Figure()

    for i, indicator in enumerate(indicators):
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
                    thickness=1,  # Thicker error bars
                    width=2,  # Wider error bars
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
        title={
            "text": "Reference: <i>'Status-quo'</i> scenario",
            "x": 0.99,
            "xanchor": "right",
            "yanchor": "top",
        },
        xaxis_title="",
        yaxis_title="",
        barmode="group",
        height=320,  # Adjust height based on the number of rows
        margin=dict(
            l=10, r=5, t=30, b=40
        ),  # Tight layout with more bottom margin for legend
        yaxis=dict(
            tickfont=dict(size=12),
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
            y=-0.2,  # Position the legend below the plot
            xanchor="center",
            x=0.5,
            itemsizing="constant",  # Consistent item sizing
            traceorder="normal",  # Keep the legend order as per the traces added
        ),
    )
    return fig