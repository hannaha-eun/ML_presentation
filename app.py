import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
# import seaborn as sns
import io
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/superhero/bootstrap.min.css'])

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("ML presentation"),
        html.Div([
            html.Div([
                html.Label("Upload Machine Learning Model to Test:"),
                dcc.Upload(
                    id='upload-model',
                    children=html.Button('Upload Model', className="btn btn-primary mb-3"),
                    multiple=False
                ),
            ], className="col-md-2"),
            html.Div([
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'xgboost 12', 'value': './model_fake/xgboost_12'},
                        {'label': 'rforest 4', 'value': './model_fake/rforest_4'},
                        {'label': 'rforest 19', 'value': './model_fake/rforest_19'},
                    ],
                    placeholder="Select a preloaded model", className=" mt-5 mb-3" , style={"color": "black"}
                ),
            ], className="col-md-4"),
        ], className="row"),
        
        html.Label("Upload Dataset :"),
        dcc.Upload(
            id='upload-dataset',
            children=html.Button('Upload File', className="btn btn-primary mb-3"),
            multiple=False
        ),

        html.Label("Upload Annotation File:"),
        dcc.Upload(
            id='upload-annotation',
            children=html.Button('Upload Annotation File', className="btn btn-primary mb-3"),
            multiple=False
        ),
        html.Label("Number of Samples for Randomization:"),
        dcc.Input(
            id='num-samples',
            type='number',
            value=100, className="form-control mb-3 col-md-4"
        ),
        html.Label("Number of randomisation:"),
        dcc.Input(
            id='num_random',
            type='number',
            value=10, className="form-control mb-3 col-md-4"
        ),
        html.Label("Balanced/Non-Balanced Data:"),
        dcc.RadioItems(
            id='balanced-radio',
            options=[
                {'label': 'Balanced', 'value': 'balanced'},
                {'label': 'Non-Balanced', 'value': 'non-balanced'}
            ],
            value='balanced', className="mb-3"
        ),
        html.Div(id='graph1', className="mb-4"),
        html.Div(id='graph2', className="mb-4")
    ], className="container")
])
# app.layout = html.Div([
#     html.Div([
#         html.H1("ML presentation"),
#         html.Label("Upload Machine Learning Model to Test:"),
#         dcc.Upload(
#             id='upload-model',
#             children=html.Button('Upload Model' , className="btn btn-primary mb-3"),
#             multiple=False
#         ),
#         dcc.Dropdown(
#             id='model-dropdown',
#             options=[
#                 {'label': 'Dataset 1', 'value': 'dataset1'},
#                 {'label': 'Dataset 2', 'value': 'dataset2'},
#             ],
#             placeholder="Select a preloaded model" , className="form-control mb-3"
#         ),
        
#         html.Label("Select Dataset:"),
      
#         html.Div(id='upload-dataset-div', style={'display': 'none'}, children=[
#             html.Label("Upload Dataset:"),
#             dcc.Upload(
#                 id='upload-dataset',
#                 children=html.Button('Upload Dataset', className="btn btn-primary mb-3"),
#                 multiple=False
#             ) 
#          ]),
#         html.Label("Upload Annotation File:"),
#         dcc.Upload(
#             id='upload-annotation',
#             children=html.Button('Upload Annotation File' , className="btn btn-primary mb-3"),
#             multiple=False
#         ),
#         html.Label("Number of Samples for Randomization:"),
#         dcc.Input(
#             id='num-samples',
#             type='number',
#             value=100 ,  className="form-control mb-3"
#         ),
#         html.Label("Number of randomisation:"),
#         dcc.Input(
#             id='num_random',
#             type='number',
#             value=10  , className="form-control mb-3"
#         ),
#         html.Label("Balanced/Non-Balanced Data:"),
#         dcc.RadioItems(
#             id='balanced-radio',
#             options=[
#                 {'label': 'Balanced', 'value': 'balanced'},
#                 {'label': 'Non-Balanced', 'value': 'non-balanced'}
#             ],
#             value='balanced' , className="mb-3"
#         ),
#         html.Div(id='graph1' , className="mb-4"),
#         html.Div(id='graph2' , className="mb-4")
#     ], className="container")
# ])

# Call back function to run calculate accuracy and output plots
@app.callback(
    [Output('graph1', 'children'),
     Output('graph2', 'children')],
    [Input('model-dropdown', 'value'),
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

