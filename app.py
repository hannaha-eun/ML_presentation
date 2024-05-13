import base64
import io

import pandas as pd
import numpy as np
import random

import dash
from dash import dcc, html, Input, Output, State ,dash_table

import plotly.express as px

# import seaborn as sns
## pip install scikit-learn==1.2.2
import pickle
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score


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
        html.Label("Real Annotation/ Random Annotation:"),
        dcc.RadioItems(
            id='balanced-radio',
            options=[
                {'label': 'Real', 'value': 'real'},
                {'label': 'Random', 'value': 'random'}
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
        dcc.Graph(id="graph1" , className="mb-4"),
        dcc.Graph(id="graph2" , className="mb-4"),
        # html.Div(id='graph1', className="mb-4"),
        # html.Div(id='graph2', className="mb-4"),
   
        html.Div(id='datatable_test' , className="mb-4" , style={"color": "black"})
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
        summary_text = f"Annotation file '{filename}' uploaded.  \n"
        annotation_df=decode_table(annotation)
        annotation_list=list(set(annotation_df.iloc[:,1].to_list()))
        # annotation_list=list(set(annotation_df['Annotation'].to_list()))
        num_samples=str(annotation_df.iloc[:,0].value_counts().shape[0])
        summary_text += f"Total number of samples: {num_samples}  \n"
        for i in annotation_list : 
            count=annotation_df.iloc[:,1].value_counts()[i]
            summary_text += f"Number of {i}: {count}  \n"
    
    return dcc.Markdown(summary_text), None


@app.callback(
    [Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('datatable_test', 'children')],
    [Input('confirm-start', 'submit_n_clicks')],
    [State('upload-model', 'contents'),
     State('model-dropdown', 'value'),
     State('upload-dataset', 'contents'),
     State('upload-annotation', 'contents'),
     State('num-samples', 'value'),
     State('num_random', 'value'),
     State('balanced-radio', 'value')]
)
def update_output(submit_n_clicks, model_contents, pre_model, count_table, annotation, num_samples, num_random, option):
    if submit_n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Load machine learning model
    if model_contents is not None:
        content_type, content_string = model_contents.split(',')
        decoded = base64.b64decode(content_string)
        model = pickle.loads(decoded)
    elif pre_model is not None:
        # Load preselected machine learning model
        model = pickle.load(open(pre_model, "rb"))
    else:
        # Handle case when no model is provided
        return "Error: No model provided.", dash.no_update, dash.no_update



    n_random=int(num_random)
    n_sample = int(num_samples)

    count_df = decode_table(count_table)
    count_df=count_df.set_index(count_df.columns[0])
    annotation_df = decode_table(annotation)
    annotation_df=annotation_df.set_index(annotation_df.columns[0])
    # merge count table and annotation table on sample name 
    count_df=count_df.T
    merged_df=pd.merge(count_df, annotation_df, right_index=True, left_index=True)    

    for index,row in merged_df.iterrows() : #annot_df
        annot=row['Class']
        # this needs to be more sophisicagted for the actual annotation/dataset 
        ## this is for test purpose -> web app demo purpose 
        if annot == 'Positive' :
            merged_df.loc[index, 'Sample_Class'] = 1
        else : 
            merged_df.loc[index, 'Sample_Class'] = 0

    case_df=merged_df[merged_df['Sample_Class'] == 1]
    control_df=merged_df[merged_df['Sample_Class'] == 0]

    accuracy_list=[]
    accuracy_dict={}
    accuracy_text = "" 
    combined_confusion_matrix = np.zeros((2, 2), dtype=int)
    # shuffle the annotation for 'fake' data 
    shuffled_df=merged_df.copy()
    shuffled_df['Sample_Class'] = np.random.permutation(shuffled_df['Sample_Class'])

    for randomization in range(n_random):
        # list of selected sample
        if option =='random' :
            selected_samples = random.sample(list(shuffled_df.T.columns), n_sample)
            randomized_df = shuffled_df.loc[selected_samples]

        elif option =='real' : 
            selected_samples = random.sample(list(merged_df.T.columns), n_sample)
            # extract selected sample from count table 
            randomized_df = merged_df.loc[selected_samples]
            
            
        randomized_df_sample=randomized_df.drop(columns=['Sample_Class' ,'Class'])
        # extract selected sample from annotation table 
        annot_df2 = randomized_df['Sample_Class']
        # predictuib
        testing_y_pred = model.predict(randomized_df_sample)

        # calculate accuracy score 
        acc = accuracy_score(annot_df2, testing_y_pred)
        accuracy_list.append(acc)
        accuracy_dict[randomization+1]=acc
        
        # print(f"Randomization {randomization + 1}: Accuracy = {acc}  n")
        accuracy_text += f"Randomization {randomization + 1}: Accuracy = {acc}  \n"

        # plot graphs
        confusion_matrix = metrics.confusion_matrix(annot_df2, testing_y_pred)
        combined_confusion_matrix += confusion_matrix
        
    # print(f"confusion matrix{ combined_confusion_matrix}")
    acc_df = pd.DataFrame(list(accuracy_dict.items()),columns = ['n_random','accuracy']) 
    fig = px.histogram(accuracy_list, range_x=[0, 1], nbins=50)

    

    fig.update_layout(title='Accuracy Distribution',  xaxis=dict(title='Accuracy'), yaxis=dict(title='# of times'))
    # fig.update_traces(xbins=dict( # bins used for histogram
    #     start=0.0,
    #     end=60.0,
    #     size=2
    # ))


    heat_fig = px.imshow(combined_confusion_matrix ,x=['Positive','Negative'] ,y=['Positive','Negative'] , text_auto=True, aspect="auto"  )

    # Customize the layout
    heat_fig.update_layout(
        title='Combined Confusion Matrix',
        xaxis=dict(title='Predicted value'),
        yaxis=dict(title='Real value'),
        xaxis_showticklabels=False,
        yaxis_showticklabels=False, 
        coloraxis_colorbar=dict(title='Count')  
    )


    return fig, heat_fig  , dash_table.DataTable(data=acc_df.to_dict('records')) #, dcc.Markdown(accuracy_text)


# def update_output(submit_n_clicks, model_contents, pre_model, count_table, annotation, num_samples, num_random, balance):
#     if submit_n_clicks is None:
#         return dash.no_update, dash.no_update, dash.no_update
    
#     # Load machine learning model
#     if model_contents is not None:
#         content_type, content_string = model_contents.split(',')
#         decoded = base64.b64decode(content_string)
#         model = pickle.loads(decoded)
#     elif pre_model is not None:
#         # Load preselected machine learning model
#         model = pickle.load(open(pre_model, "rb"))
#     else:
#         # Handle case when no model is provided
#         return "Error: No model provided.", dash.no_update, dash.no_update



#     n_random=int(num_random)
#     n_sample = int(num_samples)

#     count_df = decode_table(count_table)
#     count_df=count_df.set_index(count_df.columns[0])
#     annotation_df = decode_table(annotation)
#     annotation_df=annotation_df.set_index(annotation_df.columns[0])
#     # merge count table and annotation table on sample name 
#     count_df=count_df.T
#     merge_pd=pd.merge(count_df, annotation_df, right_index=True, left_index=True)    

#     for index,row in merge_pd.iterrows() : #annot_df
#         annot=row['Class']
#         # this needs to be more sophisicagted for the actual annotation/dataset 
#         ## this is for test purpose -> web app demo purpose 
#         if annot == 'Positive' :
#             merge_pd.loc[index, 'Sample_Class'] = 1
#         else : 
#             merge_pd.loc[index, 'Sample_Class'] = 0

#     case_df=merge_pd[merge_pd['Sample_Class'] == 1]
#     control_df=merge_pd[merge_pd['Sample_Class'] == 0]

#     accuracy_list=[]
#     accuracy_dict={}
#     accuracy_text = "" 
#     combined_confusion_matrix = np.zeros((2, 2), dtype=int)
#     for randomization in range(n_random):
#         # list of selected sample
#         if balance =='random' :
#             selected_samples = random.sample(list(merge_pd.T.columns), n_sample)

#         elif balance =='balanced' : 

#             selected_control_samples=random.sample(control_df.index.tolist(), n_sample)
#             selected_case_samples=random.sample(case_df.index.tolist(), n_sample)
#             selected_samples=selected_case_samples + selected_control_samples

#         # extract selected sample from count table 
#         randomized_df = merge_pd.loc[selected_samples]
#         randomized_df_sample=randomized_df.drop(columns=['Sample_Class' ,'Class'])
#         # extract selected sample from annotation table 
#         annot_df2 = randomized_df['Sample_Class']
        
#         # predictuib
#         testing_y_pred = model.predict(randomized_df_sample)
#         # calculate accuracy score 
#         acc = accuracy_score(annot_df2, testing_y_pred)
#         accuracy_list.append(acc)
#         accuracy_dict[randomization+1]=acc
        
#         print(f"Randomization {randomization + 1}: Accuracy = {acc}  n")
#         accuracy_text += f"Randomization {randomization + 1}: Accuracy = {acc}  \n"

#         # plot graphs
#         confusion_matrix = metrics.confusion_matrix(annot_df2, testing_y_pred)
#         combined_confusion_matrix += confusion_matrix
#     print(f"confusion matrix{ combined_confusion_matrix}")
#     acc_df = pd.DataFrame(list(accuracy_dict.items()),columns = ['n_random','accuracy']) 
#     fig = px.histogram(accuracy_list, range_x=[0, 1], nbins=50)

    

#     fig.update_layout(title='Accuracy Distribution',  xaxis=dict(title='Accuracy'), yaxis=dict(title='# of times'))
#     # fig.update_traces(xbins=dict( # bins used for histogram
#     #     start=0.0,
#     #     end=60.0,
#     #     size=2
#     # ))


#     heat_fig = px.imshow(combined_confusion_matrix ,x=['Positive','Negative'] ,y=['Positive','Negative'] , text_auto=True, aspect="auto"  )

#     # Customize the layout
#     heat_fig.update_layout(
#         title='Combined Confusion Matrix',
#         xaxis=dict(title='Predicted value'),
#         yaxis=dict(title='Real value'),
#         xaxis_showticklabels=False,
#         yaxis_showticklabels=False, 
#         coloraxis_colorbar=dict(title='Count')  
#     )


#     return fig, heat_fig  , dash_table.DataTable(data=acc_df.to_dict('records')) #, dcc.Markdown(accuracy_text)




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=7777)


# upload-model # model-dropdown -> inputs model 
# upload-dataset -> count table 
# upload-annotation -> annotation file 
# num-samples -> # samples to choose each randomization
# num_random -> # randomization test 
# balanced-radio -> whether to balance data / not 