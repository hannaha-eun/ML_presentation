import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
# import seaborn as sns
import io
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("ML presentation"),
        html.Label("Upload Machine Learning Model to Test:"),
        dcc.Upload(
            id='upload-model',
            children=html.Button('Upload Model'),
            multiple=False
        ),
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': 'Dataset 1', 'value': 'dataset1'},
                {'label': 'Dataset 2', 'value': 'dataset2'},
            ],
            placeholder="Select a preloaded dataset"
        ),
        html.Label("Upload Annotation File:"),
        dcc.Upload(
            id='upload-annotation',
            children=html.Button('Upload Annotation File'),
            multiple=False
        ),
        html.Label("Number of Samples for Randomization:"),
        dcc.Input(
            id='num-samples',
            type='number',
            value=100
        ),
        html.Label("Number of randomisation:"),
        dcc.Input(
            id='num_random',
            type='number',
            value=10
        ),
        html.Label("Balanced/Non-Balanced Data:"),
        dcc.RadioItems(
            id='balanced-radio',
            options=[
                {'label': 'Balanced', 'value': 'balanced'},
                {'label': 'Non-Balanced', 'value': 'non-balanced'}
            ],
            value='balanced'
        ),
        html.Div(id='graph1'),
        html.Div(id='graph2')
    ])
])

# Call back function to run calculate accuracy and output plots
@app.callback(
    [Output('graph1', 'children'),
     Output('graph2', 'children')],
    [Input('dataset-dropdown', 'value'),
     Input('num-samples', 'value'),
     Input('num_random', 'value'),
     Input('balanced-radio', 'value')]
)
def update_output(dataset, num_samples, num_random, balance):

    # Calculate confusion matrix
    # Calculate accuracy
    # Create confusion matrix heatmap
    # Create accuracy histogram
    return dcc.Graph(), dcc.Graph()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=7777)

