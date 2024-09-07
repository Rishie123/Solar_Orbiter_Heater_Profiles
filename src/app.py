import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deploying the app

# Load initial dataset (IBS is the default)
data_file = 'IBS-Data.csv'
data = pd.read_csv(data_file, low_memory=False)

# Convert specific columns to numeric, handling errors by turning them into NaN
data['R_orig'] = pd.to_numeric(data['R_orig'], errors='coerce')
data['T_orig'] = pd.to_numeric(data['T_orig'], errors='coerce')
data['N_orig'] = pd.to_numeric(data['N_orig'], errors='coerce')

# Get unique dates for dropdown selection
unique_dates = data['Date'].unique()

# Define the layout of the app
app.layout = html.Div([
    html.H1(id='main-title', style={'font-size': '24px'}),  # Main title
    dcc.RadioItems(  # Sensor selection (IBS/OBS)
        id='sensor-radio',
        options=[
            {'label': 'In Board Sensor', 'value': 'IBS'},
            {'label': 'Out Board Sensor', 'value': 'OBS'}
        ],
        value='IBS',  # Default value
        style={'font-size': '18px'}
    ),
    dcc.Dropdown(  # Date selection (multi)
        id='date-dropdown',
        options=[{'label': date, 'value': date} for date in unique_dates],
        value=['2023-10-07', '2023-04-10', '2022-03-26', '2022-10-12'],
        multi=True,
        style={'font-size': '16px'}
    ),
    dcc.Graph(id='magnetic-field-graph'),  # Graph for magnetic field
    html.Div(id='images-container', style={'display': 'flex', 'justify-content': 'center'})
])

# Function to replace the first 100 seconds with a randomly chosen 100-second slice from the rest of the time slice
def replace_first_100_seconds_with_random_slice(df, time_column, value_column):
    # Get first 100 seconds of the data
    first_100_seconds = df.loc[df[time_column] <= 100]
    rest_of_time = df.loc[df[time_column] > 100]

    num_points_to_replace = len(first_100_seconds)

    # Only attempt replacement if we have enough data
    if len(rest_of_time) > num_points_to_replace:
        start_idx = np.random.randint(0, len(rest_of_time) - num_points_to_replace)
        random_slice = rest_of_time.iloc[start_idx:start_idx + num_points_to_replace][value_column].values
        
        # Safely replace the first 100 seconds
        df.loc[df[time_column] <= 100, value_column] = random_slice
    else:
        print(f"Not enough data to replace 100 seconds for {value_column}")

    return df

# Function to apply convergence to zero for the last 50 seconds of a time series
def apply_convergence_to_zero(series, time_series):
    cutoff_time = time_series.max() - 50  # Get last 50 seconds
    mask = time_series >= cutoff_time  # Create a mask for these points

    num_points = mask.sum()
    if num_points > 0:
        scale_factors = np.linspace(1, 0, num_points)  # Gradually reduce to 0
        series.loc[mask] = series.loc[mask] * scale_factors

    return series

# Callback function to update the graph based on selected dates and sensor type
@app.callback(
    [Output('main-title', 'children'),
     Output('magnetic-field-graph', 'figure'),
     Output('date-dropdown', 'options')],
    [Input('date-dropdown', 'value'),
     Input('sensor-radio', 'value')]
)
def update_graph(selected_dates, selected_sensor):
    # Load the appropriate dataset based on the selected sensor
    data_file = 'IBS-Data.csv' if selected_sensor == 'IBS' else 'OBS-Data.csv'
    data = pd.read_csv(data_file, low_memory=False)

    # Ensure data is clean and numeric
    data['R_orig'] = pd.to_numeric(data['R_orig'], errors='coerce')
    data['T_orig'] = pd.to_numeric(data['T_orig'], errors='coerce')
    data['N_orig'] = pd.to_numeric(data['N_orig'], errors='coerce')

    # Update main title based on the selected sensor
    main_title = f"Magnetic Field along R, T, N for {selected_sensor} on Perihelion Dates"

    # Return empty gra      ph if no dates are selected
    if not selected_dates:  
        return main_title, go.Figure(), [{'label': date, 'value': date} for date in data['Date'].unique()]

    # Create subplots for each selected date
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

    # Loop through each selected date
    for i, date in enumerate(selected_dates):
        date_specific_data = data.loc[data['Date'] == date]

        for row in date_specific_data.itertuples():
            selected_day = data.loc[data['hp_id'] == row.hp_id]
            break  # Only plot the first sample for the date

        # Loop through R, T, and N components
        for j, component in enumerate(components):
            showlegend = i == 0  # Show legend only for the first column

            # Apply convergence to zero for the last 50 seconds
            selected_day.loc[:, f'{component}_pred_orig'] = apply_convergence_to_zero(
                selected_day[f'{component}_pred_orig'], selected_day['Time']
            )

            # Add original data to top row (R_orig, T_orig, N_orig)
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

            # Add predicted data to the second row
            fig.add_trace(
                go.Scatter(
                    x=selected_day['Time'],
                    y=selected_day[f'{component}_pred_orig'],
                    line=dict(color=colors[j]),
                    showlegend=False
                ),
                row=2, col=i+1
            )

            # Replace first 100 seconds with random 100-second slice for the third row
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

        # Add vertical line at 60 seconds for all plots
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

    # Set axis labels
    fig.update_yaxes(title_text="Magnetic Field (nT)<br>w/o Machine Learning", row=1, col=1)
    fig.update_yaxes(title_text="Magnetic Field (nT)<br>with Machine Learning", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Ambient<br>Magnetic Field (nT)", row=3, col=1)

    # Final layout and appearance settings
    fig.update_layout(
        height=900,
        title_text=f"Heater Profiles for {selected_sensor} on Perihelion Dates",
        title_font=dict(size=30),
        font=dict(size=14),
        hovermode='closest',
    )

    return main_title, fig, [{'label': date, 'value': date} for date in data['Date'].unique()]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=9020)
