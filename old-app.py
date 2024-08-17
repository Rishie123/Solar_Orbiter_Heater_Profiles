import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

# Read the updated CSV file
data_file = 'IBS-Data.csv'
data = pd.read_csv(data_file)

app = dash.Dash(__name__)

# Fetch unique dates from the updated CSV file
unique_dates = data['Date'].unique()

app.layout = html.Div([
    html.H1("Magnetic Field along R,T,N for In Board Sensor on Perihilon Dates with & without Machine Learning"),
    dcc.Dropdown(
        id='date-dropdown',
        options=[{'label': date, 'value': date} for date in unique_dates],
        value=['2023-10-07', '2023-04-10', '2022-03-26', '2022-10-12'],  # default dates
        multi=True,
        style={'font-size': '16px'}  # Increase font size of dropdown
    ),
    dcc.Graph(id='magnetic-field-graph')
])

@app.callback(
    Output('magnetic-field-graph', 'figure'),
    Input('date-dropdown', 'value')
)
def update_graph(selected_dates):
    if not selected_dates:
        return go.Figure()

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
  
        fig.update_xaxes(
            title_text="Time (seconds)",
            row=2, col=i+1,
            title_font=dict(size=18),
            tickfont=dict(size=18)
        )
        fig.update_xaxes(
            showticklabels=True,
            row=1, col=i+1,
            tickfont=dict(size=18)
        )  # Ensure ticks are shown on the top plots
        fig.update_xaxes(
            showticklabels=True,
            row=2, col=i+1,
            tickfont=dict(size=18)
        )  # Ensure ticks are shown on the top plots

    fig.update_yaxes(
        title_text="WITHOUT MACHINE LEARNING",
        row=1, col=1,
        title_font=dict(size=16),
        tickfont=dict(size=20)
    )
    fig.update_yaxes(
        title_text="WITH MACHINE LEARNING",
        row=2, col=1,
        title_font=dict(size=16),
        tickfont=dict(size=20)
    )
    fig.update_layout(
        height=600,
        title_text="Magnetic Field measurements at In Board Sensor                                                  on Perihelion Dates",
        title_font=dict(size=30),  # Increase font size for the title
        font=dict(size=14)  # Increase font size for the entire plot
    )

    # Increase font size for subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=20)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
