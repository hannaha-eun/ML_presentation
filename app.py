import base64
import dash
from dash import dcc, html, Input, Output, State ,dash_table
import pandas as pd
import plotly.express as px
# import seaborn as sns
import io
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/superhero/bootstrap.min.css'])

# # Define the layout of the app
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
        
        html.Div(id='annotation-summary', className="mb-3"),  # Summary of annotation file
        
        html.Div([
            dcc.ConfirmDialogProvider(
                children=html.Button('Start Running', className="btn btn-primary"),
                id='confirm-start',
                message='Are you sure you want to start running?'
            ),
        ]),
        
        html.Div(id='graph1', className="mb-4"),
        html.Div(id='graph2', className="mb-4"),
   
        html.Div(id='datatable_test')
    ], className="container")
])



def decode_table(contents) :
    if contents is not None:
        #read uploaded csv file as df 
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'openxmlformats' in content_type :
            df = pd.read_excel(io.BytesIO(decoded))
        else : 
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        return df



@app.callback(
    [Output('annotation-summary', 'children'),
     Output('confirm-start', 'submit_n_clicks')],
    [Input('upload-annotation', 'contents')],
    [State('upload-annotation', 'filename')]
)
def update_annotation_summary(annotation, filename):
    if annotation is None:
        return '', None
    
    else :
        summary_text = f"Annotation file '{filename}' uploaded.\n"
        annotation_df=decode_table(annotation)
        annotation_list=list(set(annotation_df['Annotation'].to_list()))
        num_samples=str(annotation_df['Sample'].value_counts().shape[0])
        summary_text += f"Total number of samples: {num_samples}\n"
        for i in annotation_list : 
            count=annotation_df['Annotation'].value_counts()[i]
            summary_text += f"Number of {i}: {count}\n"
    
    return dcc.Markdown(summary_text), None

# Call back function to run calculate accuracy and output plots
@app.callback(
    [Output('graph1', 'children'),
     Output('graph2', 'children'),
    Output('datatable_test', 'children')# this was for testing purpose 
    ],
    
    [Input('upload-model', 'contents'),
    Input('model-dropdown', 'value'),
    Input('upload-dataset', 'contents'),
    Input('upload-annotation', 'contents'),
     Input('num-samples', 'value'),
     Input('num_random', 'value'),
     Input('balanced-radio', 'value')]
)
def update_output(model, pre_model,count_table,annotation, num_samples, num_random, balance):
    count_df=decode_table(count_table)
    annotation_df=decode_table(annotation)
    
    # Calculate confusion matrix
    # Calculate accuracy
    # Create confusion matrix heatmap
    # Create accuracy histogram
    return dcc.Graph(), dcc.Graph() ,  dash_table.DataTable(data=count_df.to_dict('records'))

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=7777)

# upload-model # model-dropdown -> inputs model 
# upload-dataset -> count table 
# upload-annotation -> annotation file 
# num-samples -> # samples to choose each randomization
# num_random -> # randomization test 
# balanced-radio -> whether to balance data / not 