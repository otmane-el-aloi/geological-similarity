""" Dashbord using dash plotly"""
# Standard
import os
import joblib
import base64

# Dash related
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Ploting 
import plotly.express as px

# Data wrangling
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Model
from models.model import FeatureExtractor

# Internal 
from dataLoader.dataLoader import dataLoader
from configs.config import CFG





# Reduced Data directory
REDUCED_DATA_PATH = "dimensionality_reduction/reduced_data.csv"

# Upload directory
UPLOAD_DIRECTORY = "static/images"

# Static data directory
STATIC_DATA_Directory = "static/images"

# KNeighbors model directory 
KNEIGHBORS_DIRECTORY = "kneighbors_finding/kneighbors_model.joblib"

# PCA model 
PCA_MODEL_DIRECTORY = "dimensionality_reduction/pca_model.joblib"



if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


def save_file(name, content):
    """Decode and store an image uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, "image_of_interest.jpg"), "wb") as fp:
        fp.write(base64.decodebytes(data))

def uploaded_images():
    """List the images in the upload directory."""
    images = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            images.append(filename)
    return images


# Reduced data
df = pd.read_csv(REDUCED_DATA_PATH)
df["class"] = df["class"].astype("str")
df = df.drop(["Unnamed: 0"], axis=1)

# Test figure
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color="class")

fig.update_layout(margin=dict(l=100, r=0, b=0, t=0),
                  paper_bgcolor="rgb(0,0,0,0)")
fig.update_scenes(xaxis_visible=False,
                  yaxis_visible=False, zaxis_visible=False)


app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.MATERIA],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

# Layout: Bootstrap ^^

app.layout = dbc.Container([
    dbc.Row([
            dbc.Col(html.H1([dbc.Badge("Geological Similarity",color="dark", className="ml-1")],
                            className='text-left'), style={"padding": "6px", "font-family": "Georgia",
                                                            "font-weight": "bold", "font-size": "24px", "margin-left": "10px"}),
            dbc.Col(html.Div(
                    dbc.Button("View on Github",href = "https://github.com/", outline=True,
                               color="info", className="mr-1")
                    ), width=2, style={"padding": "4px"})
            ],
            style={"margin-bottom": "5px", "background-color": "lightblue", "border-radius": "0% 0% 0% 0%"}),

    dbc.Row([dbc.Col(html.P("This app allows geologist to easily find the 6 most similar images to an input image"), width=12, style={"text-align": "center"}),
             ],
            style={"margin-top": "15px", "margin-bottom": "15px", "margin-left": "2px"}),

    dbc.Row([dbc.Col([html.Div(html.H3("Controls", className="text-left"),style={"border-bottom": "solid lightblue"}),
            dbc.FormGroup(
                [
                    dbc.Label("Choose a class"),
                    dbc.Checklist(
                        options=[
                            {"label": "andesite", "value": "andesite"},
                            {"label": "gneiss", "value": "gneiss"},
                            {"label": "marble", "value": "marble"},
                            {"label": "quartzite", "value": "quartzite"},
                            {"label": "rhyolite", "value": "rhyolite"},
                            {"label": "schist", "value": "schist"},
                            {"label": "uploaded_image", "value": "uploaded_image"}
                        ],
                        value=[1],
                        id="checklist-classes",
                    ),
                ])
    ], width={"size": 2}),
        dbc.Col([html.Div(html.H3("Image of Interest", className="text-left"), style={"border-bottom": "solid lightblue"}),
                 # First image to upload
                 dcc.Upload(id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')]),
                            style={
                                'width': '100%',
                                'height': '40%',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                            ),
                 html.Div(id='output-image-upload'),
                 # Selected point on graph
                 dbc.CardBody(
            html.P("Uploaded image", className="card-text")),
            dbc.CardImg(id="uploaded-image",
                src="", className="img-thumbnail", bottom=True),
            html.Div(dbc.Button("Lunch feature extraction", id ="feature-extraction-button", color="info", outline = True, className="mr-1"), style = {"margin": "10px", "position": "center"})
        ],
        width={"size": 2}),

        dbc.Col([html.Div(html.H3("Images in 3D space", className="text-left"), style={"border-bottom": "solid lightblue"}),
                 dcc.Graph(id='my-scatter', figure={})
                 ], width={"size": 8})
    ], no_gutters=False, justify="center"),

    dbc.Row([
        html.Div(
            html.H3("The most similar images to the input image", className="text-center", style={"border-bottom": "solid lightblue"}
                    ), style={"padding": "4px"})
    ],  justify="center"),

    dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(id = "neighbor-1",
                            src="", top=True),
                        dbc.CardBody(
                            html.P("",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(id = "neighbor-2",
                            src="", top=True, bottom=False),
                        dbc.CardBody(
                            html.P("",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(id = "neighbor-3",
                            src="", top=True),
                        dbc.CardBody(
                            html.P("",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(id = "neighbor-4",
                            src="", top=True),
                        dbc.CardBody(
                            html.P("",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(id = "neighbor-5",
                            src="", top=True),
                        dbc.CardBody(
                            html.P("",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(id = "neighbor-6",
                            src="", top=True),
                        dbc.CardBody(
                            html.P("",
                                   className="card-text")
                        )
                    ], color="lightblue"
                ))], no_gutters=False, justify="center")

], fluid=True)


# Util functions 
def extract_features(img):
    """ This function takes an image of interest, extracts feautres and reduces its dimension """
    # Load models
    model = FeatureExtractor(CFG)
    model.load_model()
    feature_extractor = model.feature_extractor()

    # extract features 
    extracted_features = feature_extractor.predict(img)

    # reduce dimension
    pca_model = joblib.load(PCA_MODEL_DIRECTORY)
    reduced_img = pca_model.transform(extracted_features)

    return reduced_img


# Callback for uploading and showing the image of interest
@app.callback(
    Output("uploaded-image", "src"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def updateOutput(uploaded_image_name, uploaded_image_content):
    """Save uploaded files and regenerate the file list."""

    if uploaded_image_name is not None and uploaded_image_content is not None:
            save_file(uploaded_image_name[0], uploaded_image_content[0])
    files = uploaded_images()

    if "image_of_interest.jpg" not in files:
        print("image don't exist")
        return dash.no_update
    else:
        
        print("image loaded with sucess")
        return "static/images/image_of_interest.jpg"



# Callback for modifying the graph based on the chosed classes
# @app.callback(
#     Output("my-scatter", "figure"), 
#     [Input("checklist-classes", "value")])

# def generateChart(classes):
#     classes = [str(c) for c in classes]

#     dff = df.loc[df["class"].isin(classes)]

#     fig = px.scatter_3d(dff, x='PC1', y='PC2', z='PC3', color="class")

#     fig.update_layout(margin=dict(l=100, r=0, b=0, t=0),
#                     paper_bgcolor="rgb(0,0,0,0)")
#     fig.update_scenes(xaxis_visible=False,
#                     yaxis_visible=False, zaxis_visible=False)
#     return fig

# Callback for feature extraction
@app.callback(
        [Output("neighbor-1", "src"),
        Output("neighbor-2", "src"),
        Output("neighbor-3", "src"),
        Output("neighbor-4", "src"),
        Output("neighbor-5", "src"),
        Output("neighbor-6", "src"),
        Output("my-scatter", "figure")],
        [Input("feature-extraction-button", "n_clicks"),
        Input("checklist-classes", "value")]
)

def findNeighrest(n_clicks, classes):
    if n_clicks!=None:
        df = pd.read_csv(REDUCED_DATA_PATH)
        df["class"] = df["class"].astype("str")
        df = df.drop(["Unnamed: 0"], axis=1)
        if "image_of_interest.jpg" in  os.listdir(UPLOAD_DIRECTORY):
            data_loader_object = dataLoader()
            neighrest_images = []
            neighrest_classes = []
            neighrest_directories = []

            # Loading the uploaded image and performing feature extraction
            image_of_interest = data_loader_object.load_uploaded_img(UPLOAD_DIRECTORY, "image_of_interest.jpg")
            preprocessed_img = data_loader_object.preprocess_image(image_of_interest)
            reduced_image = extract_features(preprocessed_img) 

            # Finding the 6 neighrest images 
            neigh = joblib.load(KNEIGHBORS_DIRECTORY)
            reduced_image = reduced_image.reshape(3)
            neighbors = neigh.kneighbors([reduced_image], return_distance=False)
            print(neighbors)
            for index_neighbor in neighbors[0]:
                neighbor_name = df.loc[index_neighbor, "name"]
                neighbor_class = df.loc[index_neighbor, "class"]
                neighrest_images.append(neighbor_name)
                neighrest_classes.append(neighbor_class)
            
            i = 0
            for class_name, name in zip(neighrest_classes, neighrest_images):
                neighrest_directories.append("static/images/"+str(name))
                i += 1     
            print(neighrest_directories)
            # Plotting the uploaded image to the 3D scatter
            df2 = {'PC2': reduced_image[0], 'PC2': reduced_image[1], 'PC3': reduced_image[2], 'class': 'uploaded_image', 'name': " "}

            df = df.append(df2, ignore_index = True)

            classes = [str(c) for c in classes]

            dff = df.loc[df["class"].isin(classes)]

            fig = px.scatter_3d(dff, x='PC1', y='PC2', z='PC3', color="class")

            fig.update_layout(margin=dict(l=100, r=0, b=0, t=0),
                            paper_bgcolor="rgb(0,0,0,0)")
            fig.update_scenes(xaxis_visible=False,
                            yaxis_visible=False, zaxis_visible=False)


            app = dash.Dash(__name__,
                            external_stylesheets=[dbc.themes.LUX],
                            meta_tags=[{'name': 'viewport',
                                        'content': 'width=device-width, initial-scale=1.0'}]
                            )


            return neighrest_directories[0], neighrest_directories[1], neighrest_directories[2], neighrest_directories[3], neighrest_directories[4], neighrest_directories[5], fig





if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
