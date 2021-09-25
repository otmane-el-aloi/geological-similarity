""" Dashbord using dash plotly"""
import os
from pandas.core.indexes import multi
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Image directory
image_directory = os.getcwd() + '/src/'

df = pd.DataFrame(np.random.normal(0, 1, size=(100, 3)),
                  columns=["X", "Y", "Z"])

fig = px.scatter_3d(df, x='X', y='Y', z='Z')

fig.update_layout(margin=dict(l=100, r=0, b=0, t=0),
                  paper_bgcolor="rgb(0,0,0,0)")
fig.update_scenes(xaxis_visible=False,
                  yaxis_visible=False, zaxis_visible=False)


app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LUX],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

# Layout: Bootstrap ^^

app.layout = dbc.Container([
    dbc.Row([
            dbc.Col(html.H1("Geological Similarity app",
                            className='text-left'), style={"padding": "6px", "margin-left": "10px"}),
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
                      html.H4("Labels", className="text-left"),
            dbc.FormGroup(
                [
                    dbc.Label("Choose a class"),
                    dbc.Checklist(
                        options=[
                            {"label": "andesite", "value": 1},
                            {"label": "gneiss", "value": 2},
                            {"label": "marble", "value": 3},
                            {"label": "quartzite", "value": 4},
                            {"label": "rhyolite", "value": 5},
                            {"label": "schist", "value": 6}
                        ],
                        value=[1],
                        id="checklist-input",
                    ),
                ])
    ], width={"size": 2}),
        dbc.Col([html.Div(html.H3("Image of Interest", className="text-left"), style={"border-bottom": "solid lightblue"}),
                 # First image to upload
                 dcc.Upload(id='upload-image',
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
            dbc.CardImg(
            src="/static/images/0TI1D.jpg", className="img-thumbnail", bottom=True)
        ],
        width={"size": 2}),

        dbc.Col([html.Div(html.H3("Images in 3D space", className="text-left"), style={"border-bottom": "solid lightblue"}),
                 dcc.Graph(id='my-hist', figure=fig)
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
                        dbc.CardImg(
                            src="/static/images/0A0VE.jpg", top=True),
                        dbc.CardBody(
                            html.P("This card has an image at the top",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/static/images/0A3V2.jpg", top=True, bottom=False, alt="image error"),
                        dbc.CardBody(
                            html.P("This is a test image",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/static/images/0A3ZH.jpg", top=True),
                        dbc.CardBody(
                            html.P("This card has an image at the top",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/static/images/0A5NL.jpg", top=True),
                        dbc.CardBody(
                            html.P("This card has an image at the top",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/static/images/0ALDG.jpg", top=True),
                        dbc.CardBody(
                            html.P("This card has an image at the top",
                                   className="card-text")
                        )
                    ], color="lightblue"
                )),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/static/images/0TI1D.jpg", top=True),
                        dbc.CardBody(
                            html.P("This card has an image at the top",
                                   className="card-text")
                        )
                    ], color="lightblue"
                ))], no_gutters=False, justify="center")

], fluid=True)


if __name__ == "__main__":
    app.run_server(debug=True)
