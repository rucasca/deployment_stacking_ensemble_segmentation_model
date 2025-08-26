import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import html, Output, Input, dcc, ctx
import plotly.graph_objects as go

import os
import datetime
import numpy as np
from PIL import Image
import io
import base64
import plotly.express as px
import json

dash.register_page(__name__,path="/results",name="Ejemplos")

path_input = r"inputs\samples"
DIR_CONSTANTS = r"inputs\constants.json"


with open(DIR_CONSTANTS,'r') as file:
    CONSTANTS=json.load(file)
    CONSTANTS["cons_threshold"]=0.5


OBJECTIVES=CONSTANTS["objetives"]
CATEGORIES=CONSTANTS["categories"]

ID_OBJECTIVES=CONSTANTS["id_objetives"]
CATEGORY_INFO_OBJECTIVE=CONSTANTS["category_info_objetive"]

DICT_CLASS_INDEX=CONSTANTS["dict_class_index"]
CONS_TRHESHOLD=CONSTANTS["cons_threshold"]



def numpy_to_base64(img_array):
    image=Image.fromarray(img_array)
    buffered=io.BytesIO()
    image.save(buffered,format="PNG")
    img_procesada=base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_procesada}"

def create_card(image_data, title ,id_value ):

    image=Image.fromarray(image_data.astype('uint8'))
    buffer=io.BytesIO()
    image.save(buffer,format="PNG")
    buffer.seek(0)
    imagen_encoded=base64.b64encode(buffer.read()).decode('utf-8')
    data_url=f"data:image/png;base64,{imagen_encoded}"


    # card =  dbc.Card(
    #     [
    #         dbc.CardImg(src=data_url, top=True),
    #         dbc.CardBody(
    #             html.H5(title, className="card-title")
    #         ),
    #     ],
    #     style={"width": "100%", "margin-bottom": "20px"},
    #     id={"type": "selected-card-result", "index": id},
    #     n_clicks = 0
    # )

    result =  html.Div(
        dbc.Card(
            [
            dbc.CardImg(src=data_url,top=True,style={"width":"100%",
                    "aspect-ratio":"4 / 3",
                    "object-fit":"cover",
                    "border-top-left-radius":"0.5rem",
                    "border-top-right-radius":"0.5rem",
                },),
                dbc.CardBody(html.H5(title,className="card-title",style={"margin":"0","text-align":"center"}), style={"padding":"1rem"},),
            ],
            style={
            "width":"100%",
            "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
            "border-radius": "0.5rem",
            "overflow":"hidden",
            "background-color":"#ffffff",
            "margin-bottom":"20px",
            "transition":"transform 0.2s",
        },
        ),
        id=id_value,
        n_clicks=0,
        style={"cursor":"pointer",
            "width":"100%",
            "padding":"10px",
            "transition":"transform 0.2s",}
    )


    return result




def load_saved_inferencias():
    images = []
    titles = []
    for i, archivo in enumerate(os.listdir(path_input)):
        if archivo.endswith(".npz"):
            ruta=os.path.join(path_input, archivo)
            dict_np_array=np.load(ruta)

            imagen=dict_np_array['image']
            #imagen_encoded = numpy_to_base64(imagen)
            images.append(imagen)


            model_name=archivo.split("_")[0]

            date_str=archivo.split("_")[1]
            date_obj=datetime.datetime.strptime(date_str,"%Y%m%d")
            formatted_date=date_obj.strftime("%d/%m/%Y") 


            titles.append(f"Inferencia de {model_name} ({formatted_date})")

    rows = []

    for i in range(0,len(images),3):
        row_cards = [
            html.Div(
                create_card(
                    img,
                    title=titles[i + j],
                    id_value={"type":"selected-card-result","index": i+j}
                ),
                style={
                    "flex":"1 1 30%",
                    "margin":"10px",
                    "boxSizing":"border-box",
                    "maxWidth":"30%",
                },
            )
            for j, img in enumerate(images[i:i+3])
        ]

        row=html.Div(
            row_cards,
            style={
                "display":"flex",
                "flexWrap":"wrap",
                "justifyContent":"space-between",
                "marginBottom":"20px",
            }
        )
        rows.append(row)

    return html.Div(rows,style={"width":"100%"})


def generate_plots_modal(image, inferencia):
    fig_input=px.imshow(image)
    fig_input.update_layout(
        title="Imagen original",
        coloraxis_showscale=False,
        margin=dict(l=40,r=10,t=60,b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    # print(f"·my image shape is {image.shape} and my inferencia shape is inferencia {inferencia.shape}" )

    texto_hover=np.vectorize(lambda x: f"{CATEGORY_INFO_OBJECTIVE.get(str(x),'None')}")(inferencia)
    inferencia_preprocessed=np.flipud(inferencia)
    flipped_texto_hover=np.flipud(texto_hover)
    fugura_generada = go.Figure(data=go.Heatmap(
        z=inferencia_preprocessed,
        text=flipped_texto_hover,
        hoverinfo='text',
        colorscale='Viridis',
        showscale=False,
        colorbar=dict(title='Clase:')
    ))
    fugura_generada.update_layout(
        title="Predicción del modelo ",
        margin=dict(l=40, r=10, t=60, b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    # fugura_generada.update_yaxes(scaleanchor="x", scaleratio=1)fig_input.update_xaxes(scaleanchor="y", constrain='domain')
    fig_input.update_xaxes(scaleanchor="y",constrain='domain')
    fig_input.update_yaxes(constrain='domain')

    fugura_generada.update_xaxes(scaleanchor="y", constrain='domain')
    fugura_generada.update_yaxes(constrain='domain')

    return fig_input,fugura_generada



layout = html.Div([
    html.H1('Ejemplos'),
    html.P("Algunas de las predicciones de ejemplo son tales que:"),

    load_saved_inferencias(),

    dbc.Modal(id = "modal-result-img", is_open=False,backdrop="static", className="modal-content custom-centered-modal",keyboard=True),

])




@dash.callback(
    Output("modal-result-img","is_open"),
    Output("modal-result-img","children"),
    Input({"type": "selected-card-result","index": dash.ALL},"n_clicks"),
    prevent_initial_call=True
)
def on_card_click(n_clicks_list):

    if(not(any(n_clicks_list))):
        return False,dash.no_update
        
    id_activacion=ctx.id_activacion
    if id_activacion is None:
        return False,dash.no_update
    
    
    # print(f"{id_activacion}")
    index=id_activacion["index"]
    files=sorted(os.listdir(path_input))
    # print("index", index)
    # print("files", files)
    target_file=files[index]

    ruta=os.path.join(path_input,target_file)
    # print(f"{np.load(ruta).files=}")
    dict_np_array=np.load(ruta)
    imagen=dict_np_array['image']
    inferencia=dict_np_array['result']

    model_name=target_file.split('_')[0]
    plot1,plot2 = generate_plots_modal(imagen, inferencia)

    results = [
        dbc.ModalHeader(
            dbc.ModalTitle("Resultados de la inferencia", className="modal-title"),
            close_button=True,

            className="modal-header"
        ),
        dbc.ModalBody(
            [
                    html.H4(
                        f"Resultados del modelo {model_name}:",
                        className="text-center mb-4"
                    ),
        html.Div(
            [
                    html.Div(dcc.Graph(figure=plot1),style={"flex": "0 0 30%"}),
                    html.Div(dcc.Graph(figure=plot2),style={"flex": "0 0 30%"}),
            ],
            style={
                "display":"flex",
                "justifyContent": "space-around",
                "gap": "20px"
            }
            )
        ],
            className="modal-body"
        ),
        dbc.ModalFooter(
            [],
            className="modal-footer"
        )
    ]
    return True,results
