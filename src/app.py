import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from flask_caching import Cache

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Caching setup to avoid recomputation
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

# Load data once at startup and downcast numeric columns for memory efficiency
data_file_ibs = 'IBS-Data.csv'
data_file_obs = 'OBS-Data.csv'

@cache.memoize()  # Cache to avoid re-reading the files on every request
def load_data(sensor_type):
    data_file = data_file_ibs if sensor_type == 'IBS' else data_file_obs
    data = pd.read_csv(data_file, low_memory=False)
    # Optimize memory usage
    data['R_orig'] = pd.to_numeric(data['R_orig'], errors='coerce', downcast='float')
    data['T_orig'] = pd.to_numeric(data['T_orig'], errors='coerce', downcast='float')
    data['N_orig'] = pd.to_numeric(data['N_orig'], errors='coerce', downcast='float')
    return data

# Preload IBS data for dropdown population
data = load_data('IBS')
unique_dates = data['Date'].unique()

# Layout definition
app.layout = html.Div([
    html.H1(id='main-title', style={'font-size': '24px'}),
    dcc.RadioItems(
        id='sensor-radio',
        options=[
            {'label': 'In Board Sensor', 'value': 'IBS'},
            {'label': 'Out Board Sensor', 'value': 'OBS'}
        ],
        value='IBS',
        style={'font-size': '18px'}
    ),
    dcc.Dropdown(
        id='date-dropdown',
        options=[{'label': date, 'value': date} for date in unique_dates],
        value=['2023-10-07', '2023-04-10'],
        multi=True,
        style={'font-size': '16px'}
    ),
    dcc.Graph(id='magnetic-field-graph')
])

# Replace first 100 seconds and apply convergence
def replace_first_100_seconds_with_random_slice(df, time_column, value_column):
    first_100_seconds = df.loc[df[time_column] <= 100]
    rest_of_time = df.loc[df[time_column] > 100]
    if len(rest_of_time) > len(first_100_seconds):
        start_idx = np.random.randint(0, len(rest_of_time) - len(first_100_seconds))
        random_slice = rest_of_time.iloc[start_idx:start_idx + len(first_100_seconds)][value_column].values
        df.loc[df[time_column] <= 100, value_column] = random_slice
    return df

def apply_convergence_to_zero(series, time_series):
    cutoff_time = time_series.max() - 50
    mask = time_series >= cutoff_time
    if mask.sum() > 0:
        scale_factors = np.linspace(1, 0, mask.sum())
        series.loc[mask] = series.loc[mask] * scale_factors
    return series

# Callback to update graph based on selected sensor and dates
@app.callback(
    [Output('main-title', 'children'),
     Output('magnetic-field-graph', 'figure')],
    [Input('date-dropdown', 'value'),
     Input('sensor-radio', 'value')]
)
def update_graph(selected_dates, selected_sensor):
    data = load_data(selected_sensor)
    
    # If no dates are selected, return empty figure
    if not selected_dates:
        return f"{selected_sensor} Magnetic Field Data", go.Figure()

    # Set up figure
    fig = make_subplots(rows=3, cols=len(selected_dates), shared_xaxes=True, shared_yaxes=True)

    components = ['R', 'T', 'N']
    colors = ['red', 'green', 'orange']

    for i, date in enumerate(selected_dates):
        date_specific_data = data.loc[data['Date'] == date]
        for component, color in zip(components, colors):
            date_specific_data[f'{component}_pred_orig'] = apply_convergence_to_zero(
                date_specific_data[f'{component}_orig'], date_specific_data['Time']
            )
            
            fig.add_trace(go.Scattergl(
                x=date_specific_data['Time'],
                y=date_specific_data[f'{component}_orig'],
                mode='lines',
                line=dict(color=color),
                showlegend=(i == 0),
                name=f"{component} Original"
            ), row=1, col=i+1)

            fig.add_trace(go.Scattergl(
                x=date_specific_data['Time'],
                y=date_specific_data[f'{component}_pred_orig'],
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ), row=2, col=i+1)

            ambient_field = replace_first_100_seconds_with_random_slice(
                date_specific_data, 'Time', f'{component}_pred_orig'
            )

            fig.add_trace(go.Scattergl(
                x=ambient_field['Time'],
                y=ambient_field[f'{component}_pred_orig'],
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ), row=3, col=i+1)

    fig.update_layout(height=600, title=f"{selected_sensor} Magnetic Field Data")
    return f"{selected_sensor} Magnetic Field Data", fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=False, port=9020)
