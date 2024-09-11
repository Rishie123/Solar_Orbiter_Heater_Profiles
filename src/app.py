import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Initialize the app
app = dash.Dash(__name__)

server = app.server
# Load initial data (IBS by default)
data_file = 'IBS-Data2.csv'
data = pd.read_csv(data_file)

# Fetch unique dates from the updated CSV file
unique_dates = data['Date'].unique()

app.layout = html.Div([
    html.H1(id='main-title', style={'font-size': '24px'}),  # Increase font size for main title
    dcc.RadioItems(
        id='sensor-radio',
        options=[
            {'label': 'In Board Sensor', 'value': 'IBS'},
            {'label': 'Out Board Sensor', 'value': 'OBS'}
        ],
        value='IBS',
        style={'font-size': '18px'}  # Increase font size for radio buttons
    ),
    dcc.Dropdown(
        id='date-dropdown',
        options=[{'label': date, 'value': date} for date in unique_dates],
        value=['2023-10-07', '2023-04-10', '2022-03-26', '2022-10-12'],  # default dates
        multi=True,
        style={'font-size': '16px'}  # Increase font size of dropdown
    ),
    dcc.Graph(id='magnetic-field-graph'),
    html.Div(id='images-container', style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'flex-direction': 'column'})
])

def apply_convergence_to_zero(series, time_series):
    # Find the indices corresponding to the last 50 seconds
    cutoff_time = time_series.max() - 50
    mask = time_series >= cutoff_time
    
    # Apply a linear scale to these points to bring them to zero
    num_points = mask.sum()  # Number of points in the last 50 seconds
    if num_points > 0:
        scale_factors = np.linspace(1, 0, num_points)
        series[mask] = series[mask] * scale_factors
    
    return series

@app.callback(
    [Output('main-title', 'children'),
     Output('magnetic-field-graph', 'figure'),
     Output('date-dropdown', 'options')],
    [Input('date-dropdown', 'value'),
     Input('sensor-radio', 'value')]
)
def update_graph(selected_dates, selected_sensor):
    # Update data source based on selected sensor
    data_file = 'IBS-Data2.csv' if selected_sensor == 'IBS' else 'OBS-Data.csv'
    data = pd.read_csv(data_file)

    # Update main title based on selected sensor
    main_title = f"Magnetic Field along R, T, N for {selected_sensor} on Perihelion Dates with & without Machine Learning."+"<br>Time starts at -1 seconds, therefore peak at 59 seconds. "

    if not selected_dates:
        return main_title, go.Figure(), [], [{'label': date, 'value': date} for date in data['Date'].unique()]

    fig = make_subplots(
        rows=2,
        cols=len(selected_dates),
        shared_xaxes=True,
        shared_yaxes=True,  
        subplot_titles=[f"Date: {date}" for date in selected_dates],
        vertical_spacing=0.1
    )

    components = ['R', 'T', 'N']
    colors = ['red', 'green', 'orange']

    for i, date in enumerate(selected_dates):
        date_specific_data = data[data['Date'] == date]

        for row in date_specific_data.itertuples():
            # Select data for the current hp_id
            selected_day = data[data['hp_id'] == row.hp_id]
            break  # Assuming we plot only the first sample per date

        for j, component in enumerate(components):
            showlegend = i == 0  # Show legend only for the first column

            # Apply convergence to zero for the predicted data in the last 50 seconds
            selected_day[f'{component}_pred_orig'] = apply_convergence_to_zero(
                selected_day[f'{component}_pred_orig'],
                selected_day['Time']
            )

            # Upper plot (Original Data)
            fig.add_trace(
                go.Scatter(
                    x=selected_day['Time'],
                    y=selected_day[f'{component}_orig'],
                    name=component,
                    line=dict(color=colors[j]),
                    showlegend=showlegend
                ),
                row=1, col=i+1
            )

            # Lower plot (Predicted Data)
            fig.add_trace(
                go.Scatter(
                    x=selected_day['Time'],
                    y=selected_day[f'{component}_pred_orig'],
                    name=component,
                    line=dict(color=colors[j]),
                    showlegend=False
                ),
                row=2, col=i+1
            )

        # Add a black dashed vertical line at Time == 60 seconds for both plots
        if i == 0:  # Only add to the legend for the first date
            fig.add_trace(
                go.Scatter(
                    x=[60, 60],
                    y=[selected_day[f'{components[0]}_orig'].min(), selected_day[f'{components[0]}_orig'].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='59 s',
                    showlegend=True
                ),
                row=1, col=i+1
            )

            fig.add_trace(
                go.Scatter(
                    x=[60, 60],
                    y=[selected_day[f'{components[0]}_pred_orig'].min(), selected_day[f'{components[0]}_pred_orig'].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='60 s',
                    showlegend=False
                ),
                row=2, col=i+1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[60, 60],
                    y=[selected_day[f'{components[0]}_orig'].min(), selected_day[f'{components[0]}_orig'].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    showlegend=False
                ),
                row=1, col=i+1
            )

            fig.add_trace(
                go.Scatter(
                    x=[60, 60],
                    y=[selected_day[f'{components[0]}_pred_orig'].min(), selected_day[f'{components[0]}_pred_orig'].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    showlegend=False
                ),
                row=2, col=i+1
            )

        # Update x-axes
        fig.update_xaxes(
            title_text="Time after heaters turn on (sec)",
            row=2, col=i+1,
            title_font=dict(size=20),
            tickfont=dict(size=18)
        )
        fig.update_xaxes(
            showticklabels=True,
            row=1, col=i+1,
            tickfont=dict(size=18)
        )
        fig.update_xaxes(
            showticklabels=True,
            row=2, col=i+1,
            tickfont=dict(size=18)
        )

    # Update y-axes titles and layout
    fig.update_yaxes(
        title_text="3-Day avg.<br> Magnetic Field (nT)",
        row=1, col=1,
        title_font=dict(size=20),
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_text="Magnetic Field (nT) with<br> Machine Learning",   
        row=2, col=1,
        title_font=dict(size=20),
        tickfont=dict(size=18)
    )
    fig.update_layout(
        height=600,
        title_text=f"Heater Profiles on perihelion dates, for {selected_sensor} along R, T and N directions with & without Machine Learning",
        title_font=dict(size=30),
        font=dict(size=14),
        legend=dict(
            font=dict(size=30)
        ),
        hovermode=False  # Disable hover interactions
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=20)

    return main_title, fig, [{'label': date, 'value': date} for date in data['Date'].unique()]

if __name__ == '__main__':
    app.run_server(debug=True, port=9020)
