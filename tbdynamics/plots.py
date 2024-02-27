import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_population_chart(df, years, groups):
    """
    Plots faceted chart showing total and infectious population by age group for selected years.
    
    Parameters:
    - df: DataFrame with population data.
    - years: List of years to plot.
    - groups: List of age groups.
    """
    rows = int(np.ceil(len(years) / 2))  # Rows for subplots
    fig = make_subplots(rows=rows, cols=2, subplot_titles=[f'Year {year}' for year in years], shared_xaxes=True)
    total_legend, latent_legend = False, False
    index = [(i, j) for i in range(1, rows + 1) for j in range(1, 3)]

    for count, year in enumerate(years):
        row, col = index[count]
        for age in groups:
            year_df = df.loc[round(year,1)]  # Extract data for the year
            total, latent = year_df[f'total_populationXage_{age}'], year_df[f'latent_population_sizeXage_{age}']
            # Add total population bar
            fig.add_trace(go.Bar(x=[f'Age {age}'], y=[total], marker=dict(color='rgba(100, 150, 240, 0.6)'),
                                 legendgroup='Total', showlegend=not total_legend), row=row, col=col)
            # Add infectious population bar
            fig.add_trace(go.Bar(x=[f'Age {age}'], y=[latent], marker=dict(color='rgba(255, 100, 100, 0.6)'),
                                 legendgroup='Latent', showlegend=not latent_legend), row=row, col=col)
            total_legend, latent_legend = True, True

    fig.update_layout(height=300*rows, width=700, title="Population by Age Group & Year", barmode='group', xaxis_tickangle=-45,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), bargap=0.005)
    fig.show()
