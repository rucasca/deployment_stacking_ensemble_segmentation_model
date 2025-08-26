import dash
from dash import html
import os
import dash_bootstrap_components as dbc
from dash import html, Output, Input, dcc, ctx
import plotly.graph_objects as go
import datetime

import numpy as np
from PIL import Image
import io
import base64
import plotly.express as px
import json

dash.register_page(__name__,path ="/history",name="Historial")


path_input=r"outputs"
DIR_CONSTANTS=r"inputs\constants.json"

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
    img_str=base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def create_card(image_data,title,id_value):

    image=Image.fromarray(image_data.astype('uint8'))
    buffer=io.BytesIO()
    image.save(buffer,format="PNG")
    buffer.seek(0)
    encoded_image=base64.b64encode(buffer.read()).decode('utf-8')
    data_url=f"data:image/png;base64,{encoded_image}"


    # card =  dbc.Card(
    #     [
    #         dbc.CardImg(src=data_url, top=True),
    #         dbc.CardBody(
    #             html.H5(title, className="card-title")
    #         ),
    #     ],
    #     style={"width": "100%", "margin-bottom": "20px"},
    #     id={"type": "selected-card", "index": id},
    #     n_clicks = 0
    # )

    result =    html.Div(
        dbc.Card(
            [
                dbc.CardImg(src=data_url, top=True, style={"width":"100%",
                    "aspect-ratio": "4 / 3",
                    "object-fit": "cover",
                    "border-top-left-radius":"0.5rem",
                    "border-top-right-radius":"0.5rem",
                },),
                dbc.CardBody(html.H5(title,className="card-title",style={"margin":"0","text-align":"center"}),style={"padding": "1rem"},),
            ],
            style={
            "width":"100%",
            "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
            "border-radius":"0.5rem",
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




def load_saved_inferences():
    images = []
    titles = []

    if(len(os.listdir(path_input))==0):
        return html.P("Ninguna segmentación realizada hasta la fecha")
    
    for i,archivo in enumerate(os.listdir(path_input)):
        if archivo.endswith(".npz"):
            ruta=os.path.join(path_input,archivo)
            dict_np_array=np.load(ruta)

            imagen=dict_np_array['image']
            #encoded_image = numpy_to_base64(imagen)
            images.append(imagen)


            model_name=archivo.split("_")[0]

            date_str=archivo.split("_")[1]
            date_obj=datetime.datetime.strptime(date_str,"%Y%m%d")
            formatted_date=date_obj.strftime("%d/%m/%Y") 


            titles.append(f"Inferencia de {model_name} ({formatted_date})")

    rows = []

    for i in range(0, len(images), 3):
        row_cards = [
            html.Div(
                create_card(
                        img,
                        title=titles[i+j],
                        id_value={"type": "selected-card","index":i+j}
                ),
                style={
                    "flex":"1 1 30%",
                    "margin":"10px",
                    "boxSizing":"border-box",
                    "maxWidth":"30%",
                },
            )
            for j,img in enumerate(images[i:i+3])
        ]

        row = html.Div(
            row_cards,
            style={
                "display":"flex",
                "flexWrap":"wrap",
                "justifyContent":"space-between",
                "marginBottom":"20px",
            }
        )
        rows.append(row)

    return html.Div(rows, style={"width":"100%"})


def generate_plots_modal(image, inference):
    fig_entrada=px.imshow(image)
    fig_entrada.update_layout(
        title="Imagen original",
        coloraxis_showscale=False,
        margin=dict(l=40,r=10,t=60,b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    # print(f"·my image shape is {image.shape} and my inference shape is inference {inference.shape}" )

    info_texto=np.vectorize(lambda x: f"{CATEGORY_INFO_OBJECTIVE.get(str(x),'None')}")(inference)
    inferencia_preprocesada=np.flipud(inference)
    clases_preprocesadas=np.flipud(info_texto)
    fig_class=go.Figure(data=go.Heatmap(
        z=inferencia_preprocesada,
        text=clases_preprocesadas,
        hoverinfo='text',
        colorscale='Viridis',
        showscale=False,
        colorbar=dict(title='Clase:')
    ))
    fig_class.update_layout(
        title="Predicción del modelo ",
        margin=dict(l=40,r=10,t=60,b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    fig_entrada.update_xaxes(scaleanchor="y",constrain='domain')
    fig_entrada.update_yaxes(constrain='domain')

    fig_class.update_xaxes(scaleanchor="y",constrain='domain')
    fig_class.update_yaxes(constrain='domain')

    return fig_entrada,fig_class



layout = html.Div([
    html.H1('Resultados guardados'),
    html.P("El histórico de resultados ha sido:"),
    load_saved_inferences(),

    dbc.Modal(id="modal-img",is_open=False,backdrop="static",className="modal-content custom-centered-modal",keyboard=True),

])




@dash.callback(
    Output("modal-img","is_open"),
    Output("modal-img","children"),
    Input({"type":"selected-card","index": dash.ALL},"n_clicks"),
    prevent_initial_call=True
)
def on_card_click(n_clicks_list):

    if(not(any(n_clicks_list))):
        return False,dash.no_update
        
    triggered_id=ctx.triggered_id
    if triggered_id is None:
        return False,dash.no_update
    
    
    # print(f"{triggered_id}")
    index=triggered_id["index"]
    files=sorted(os.listdir(path_input))
    # print("index", index)
    # print("files", files)
    target_file=files[index]

    ruta=os.path.join(path_input, target_file)
    # print(f"{np.load(ruta).files=}")
    dict_np_array=np.load(ruta)
    imagen=dict_np_array['image']
    inference=dict_np_array['inference']

    model_name=target_file.split('_')[0]

    plot1,plot2= generate_plots_modal(imagen,inference)

    results = [
        dbc.ModalHeader(
                dbc.ModalTitle("Resultados de la inferencia", className="modal-title"),
                close_button=True,
                className="modal-header"
        ),
        dbc.ModalBody(
            [

                html.H4(
                    f"Resultados del modelo {model_name}:",className="text-center mb-4"


                ),
        html.Div(
            [
            html.Div(dcc.Graph(figure=plot1), style={"flex":"0 0 30%"}),
            html.Div(dcc.Graph(figure=plot2), style={"flex":"0 0 30%"}),
            ],
            style={
                "display": "flex",
                "justifyContent":"space-around",
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
    return True, results
