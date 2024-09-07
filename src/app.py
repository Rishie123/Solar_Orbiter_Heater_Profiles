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
data_file = 'IBS-Data.csv'
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
    # We extract the first 100 seconds and the remaining data
    first_100_seconds = df.loc[df[time_column] <= 100]
    rest_of_time = df.loc[df[time_column] > 100]

    # We ensure there is enough data to replace the first 100 seconds
    num_points_to_replace = len(first_100_seconds)
    if len(rest_of_time) > num_points_to_replace:
        start_idx = np.random.randint(0, len(rest_of_time) - num_points_to_replace)
        random_slice = rest_of_time.iloc[start_idx:start_idx + num_points_to_replace][value_column].values
        
        # We replace the first 100 seconds with the randomly chosen slice
        df.loc[df[time_column] <= 100, value_column] = random_slice
    else:
        print(f"Not enough data to replace 100 seconds for {value_column}")

    return df

# This function applies a gradual convergence to zero over the last 50 seconds of the time series
def apply_convergence_to_zero(series, time_series):
    cutoff_time = time_series.max() - 50  # We define the cutoff for the last 50 seconds
    mask = time_series >= cutoff_time  # We create a mask to identify points within the last 50 seconds

    # If we have points to modify, we create a scale factor to gradually reduce the values to zero
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
    # We load the appropriate dataset based on the selected sensor
    data_file = 'IBS-Data.csv' if selected_sensor == 'IBS' else 'OBS-Data.csv'
    data = pd.read_csv(data_file, low_memory=False)

    # Ensure data is clean and numeric
    data['R_orig'] = pd.to_numeric(data['R_orig'], errors='coerce')
    data['T_orig'] = pd.to_numeric(data['T_orig'], errors='coerce')
    data['N_orig'] = pd.to_numeric(data['N_orig'], errors='coerce')

    # We update the main title based on the selected sensor
    main_title = f"Magnetic Field along R, T, N for {selected_sensor} on Perihelion Dates"

    # If no dates are selected, we return an empty graph and update the dropdown options
    if not selected_dates:  
        return main_title, go.Figure(), [{'label': date, 'value': date} for date in data['Date'].unique()]

    # We create subplots to visualize each selected date
    fig = make_subplots(    
        rows=3,
        cols=len(selected_dates),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"Date: {date}" for date in selected_dates],
        vertical_spacing=0.1
    )

    # We define the magnetic field components (R, T, N) and assign colors for each component
    components = ['R', 'T', 'N']
    colors = ['red', 'green', 'orange']

    # Loop through each selected date and plot data
    for i, date in enumerate(selected_dates):
        date_specific_data = data.loc[data['Date'] == date]

        # We select the first sample for the selected date and continue with plotting
        for row in date_specific_data.itertuples():
            selected_day = data.loc[data['hp_id'] == row.hp_id]
            break  # Only plot the first sample for simplicity

        # We loop through the magnetic field components (R, T, N)
        for j, component in enumerate(components):
            showlegend = i == 0  # Show legend only for the first column

            # Apply convergence to zero for the last 50 seconds of the selected day
            selected_day.loc[:, f'{component}_pred_orig'] = apply_convergence_to_zero(
                selected_day[f'{component}_pred_orig'], selected_day['Time']
            )

            # We plot the original data on the top row
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

            # We plot the predicted data on the second row
            fig.add_trace(
                go.Scatter(
                    x=selected_day['Time'],
                    y=selected_day[f'{component}_pred_orig'],
                    line=dict(color=colors[j]),
                    showlegend=False
                ),
                row=2, col=i+1
            )

            # We replace the first 100 seconds with a random slice for the third row
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

        # We add a vertical line at 60 seconds to all plots for reference
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

    # We label the y-axes for each row of subplots
    fig.update_yaxes(title_text="Magnetic Field (nT)<br>w/o Machine Learning", row=1, col=1)
    fig.update_yaxes(title_text="Magnetic Field (nT)<br>with Machine Learning", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Ambient<br>Magnetic Field (nT)", row=3, col=1)

    # Final adjustments to layout, ensuring clarity in presentation
    fig.update_layout(
        height=900,
        title_text=f"Heater Profiles for {selected_sensor} on Perihelion Dates",
        title_font=dict(size=30),
        font=dict(size=14),
        hovermode='closest',
    )

    # We return the updated title, figure, and dropdown options
    return main_title, fig, [{'label': date, 'value': date} for date in data['Date'].unique()]

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True, port=9020)


##############################################

#REFERENCES

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
# https://dash.plotly.com/dash-core-components
# https://dash.plotly.com/dash-core-components/graph

##############################################
