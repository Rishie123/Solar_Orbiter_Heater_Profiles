import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# We initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # This line ensures the app can be deployed

# We load the initial dataset, defaulting to IBS data
data_file = 'IBS-Data2.csv'
data = pd.read_csv(data_file, low_memory=False)

# We convert specific columns to numeric, coercing any errors into NaN values
data['R_orig'] = pd.to_numeric(data['R_orig'], errors='coerce')
data['T_orig'] = pd.to_numeric(data['T_orig'], errors='coerce')
data['N_orig'] = pd.to_numeric(data['N_orig'], errors='coerce')

# Extract unique dates for dropdown selection
unique_dates = data['Date'].unique()

# We define the layout of our app, ensuring it is clear and user-friendly
app.layout = html.Div([
    html.H1(id='main-title', style={'font-size': '24px'}),  # Main title of the app
    dcc.RadioItems(  # Sensor selection (either IBS or OBS)
        id='sensor-radio',
        options=[
            {'label': 'In Board Sensor', 'value': 'IBS'},
            {'label': 'Out Board Sensor', 'value': 'OBS'}
        ],
        value='IBS',  # Default sensor selection is IBS
        style={'font-size': '18px'}
    ),
    dcc.Dropdown(  # Date selection dropdown, allowing multiple selections
        id='date-dropdown',
        options=[{'label': date, 'value': date} for date in unique_dates],
        value=['2023-10-07', '2023-04-10', '2022-03-26', '2022-10-16'],  # Default date selections
        multi=True,
        style={'font-size': '16px'}
    ),
    dcc.Graph(id='magnetic-field-graph'),  # Graph displaying magnetic field data
    html.Div(id='images-container', style={'display': 'flex', 'justify-content': 'center'})  # Container for images (if needed)
])

# This function replaces the first 100 seconds of a time series with a randomly selected 100-second slice
def replace_first_100_seconds_with_random_slice(df, time_column, value_column):
    first_100_seconds = df.loc[df[time_column] <= 100]
    rest_of_time = df.loc[df[time_column] > 100]
    num_points_to_replace = len(first_100_seconds)
    if len(rest_of_time) > num_points_to_replace:
        start_idx = np.random.randint(0, len(rest_of_time) - num_points_to_replace)
        random_slice = rest_of_time.iloc[start_idx:start_idx + num_points_to_replace][value_column].values
        df.loc[df[time_column] <= 100, value_column] = random_slice
    return df

# This function applies a gradual convergence to zero over the last 50 seconds of the time series
def apply_convergence_to_zero(series, time_series):
    cutoff_time = time_series.max() - 50
    mask = time_series >= cutoff_time
    num_points = mask.sum()
    if num_points > 0:
        scale_factors = np.linspace(1, 0, num_points)
        series.loc[mask] = series.loc[mask] * scale_factors
    return series

# Callback function to update the graph based on user input (selected dates and sensor type)
@app.callback(
    [Output('main-title', 'children'),
     Output('magnetic-field-graph', 'figure'),
     Output('date-dropdown', 'options')],
    [Input('date-dropdown', 'value'),
     Input('sensor-radio', 'value')]
)
def update_graph(selected_dates, selected_sensor):
    data_file = 'IBS-Data2.csv' if selected_sensor == 'IBS' else 'OBS-Data.csv'
    data = pd.read_csv(data_file, low_memory=False)
    data['R_orig'] = pd.to_numeric(data['R_orig'], errors='coerce')
    data['T_orig'] = pd.to_numeric(data['T_orig'], errors='coerce')
    data['N_orig'] = pd.to_numeric(data['N_orig'], errors='coerce')

    main_title = f"Magnetic Field along R, T, N for {selected_sensor} on Perihelion Dates"
    if not selected_dates:  
        return main_title, go.Figure(), [{'label': date, 'value': date} for date in data['Date'].unique()]

    fig = make_subplots(
        rows=3,
        cols=len(selected_dates),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"Date: {date}" for date in selected_dates],
        vertical_spacing=0.1
    )

    components = ['R', 'T', 'N']
    colors = ['red', 'green', 'orange']

    for i, date in enumerate(selected_dates):
        date_specific_data = data.loc[data['Date'] == date]
        for row in date_specific_data.itertuples():
            selected_day = data.loc[data['hp_id'] == row.hp_id]
            break  

        for j, component in enumerate(components):
            showlegend = i == 0
            selected_day.loc[:, f'{component}_pred_orig'] = apply_convergence_to_zero(
                selected_day[f'{component}_pred_orig'], selected_day['Time']
            )

            fig.add_trace(
                go.Scatter(
                    x=selected_day['Time'],
                    y=selected_day[f'{component}_orig'],
                    name=component if showlegend else None,
                    line=dict(color=colors[j]),
                    showlegend=showlegend
                ),
                row=1, col=i+1
            )

            fig.add_trace(
                go.Scatter(
                    x=selected_day['Time'],
                    y=selected_day[f'{component}_pred_orig'],
                    line=dict(color=colors[j]),
                    showlegend=False
                ),
                row=2, col=i+1
            )

            ambient_field = replace_first_100_seconds_with_random_slice(
                selected_day, 'Time', f'{component}_pred_orig'
            )

            fig.add_trace(
                go.Scatter(
                    x=ambient_field['Time'],
                    y=ambient_field[f'{component}_pred_orig'],
                    line=dict(color=colors[j]),
                    showlegend=False
                ),
                row=3, col=i+1
            )

        for row in range(1, 4):
            fig.add_trace(
                go.Scatter(
                    x=[60, 60],
                    y=[selected_day[f'{components[0]}_orig'].min(), selected_day[f'{components[0]}_orig'].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='60 s' if row == 1 and i == 0 else None,
                    showlegend=(row == 1 and i == 0)
                ),
                row=row, col=i+1
            )

    fig.update_yaxes(title_text="Magnetic Field (nT)<br>w/o Machine Learning", row=1, col=1)
    fig.update_yaxes(title_text="Magnetic Field (nT)<br>with Machine Learning", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Ambient<br>Magnetic Field (nT)", row=3, col=1)

    # Update x-axes to show time on all rows and columns and ensure ticks are displayed
    for row in range(1, 4):
        for col in range(1, len(selected_dates) + 1):
            fig.update_xaxes(title_text="Time (seconds)", showticklabels=True, row=row, col=col)

    fig.update_layout(
        height=900,
        title_text=f"Heater Profiles for {selected_sensor} on Perihelion Dates",
        title_font=dict(size=30),
        font=dict(size=14),
        hovermode='closest',
    )

    return main_title, fig, [{'label': date, 'value': date} for date in data['Date'].unique()]

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True, port=9020)
