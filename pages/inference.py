import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
from PIL import Image
import numpy as np
import os
import dash_daq as daq
from utils_dash.model_inference import inference_model_pipeline
import json
from datetime import datetime


dash.register_page(__name__,path ="/",name="Inferencia")

MODELS = {"Base U-Net":"unet_base",
          "RetinaNet + SAM": "retinanet_sam",
          "Yolov8 + SAM": "yolo_sam",
          "CLIP + SAM": "sam_clip",
          "RetinaNet + SAM + U-Net": "final_model"}



STORE_IMG=None
RESULT_INFERENCE=None
FILENAME =None

DIR_CONSTANTS=r"inputs\constants.json"

with open(DIR_CONSTANTS,'r') as file:
    CONSTANTS =json.load(file)
    CONSTANTS["cons_threshold"] = 0.5


OBJECTIVES=CONSTANTS["objetives"]
CATEGORIES=CONSTANTS["categories"]

ID_OBJECTIVES=CONSTANTS["id_objetives"]
CATEGORY_INFO_OBJECTIVE=CONSTANTS["category_info_objetive"]

DICT_CLASS_INDEX=CONSTANTS["dict_class_index"]
CONS_TRHESHOLD=CONSTANTS["cons_threshold"]


#####   DEFAULT VALUES   #######

CONS_CAT_INDEX_BY_NAME= {'background':0,
 'person':1,
 'car':2,
 'motorcycle':3,
 'bus':4,
 'traffic light':5,
 'backpack':6,
 'handbag':7,
 'chair':8,
 'dining table':9,
 'cell phone':10}

CONS_INFO_OBJ = {1:'person',
 3:'car',
 4:'motorcycle',
 6:'bus',
 10:'traffic light',
 27:'backpack',
 31:'handbag',
 77:'cell phone',
 62:'chair',
 67:'dining table',
 0:'background'}


CONS_DIV_ERROR_CASE1=html.Div("❌ Formato no soportado (admite .png y .jpg)", style={
                'color':'rgb(211, 47, 47)',
                'fontWeight':'bold',
                'padding':'10px',
                'border':'1px solid rgb(211, 47, 47)',
                'borderRadius':'5px',
                'backgroundColor':'rgb(255, 235, 238)',
                "margin-top":"10px",
            })

CONS_DIV_ERROR_CASE2 = html.Div("❌ Error en el procesamiento del fichero", style={
                'color':'rgb(211, 47, 47)',
                'fontWeight':'bold',
                'padding':'10px',
                'border':'1px solid rgb(211, 47, 47)',
                'borderRadius':'5px',
                'backgroundColor':'rgb(255, 235, 238)',
                "margin-top":"10px",
            })


layout = html.Div([
    html.H1('Generador de segmentaciones'),
    dcc.Store(id='image-container'),
    dcc.Loading(
        type="default",

        children = html.Div([

        html.P('Generador automático de segmentaciones semánticas mediante modelos tipo ensemble ',className="p-info" ),
        
        dbc.Container([
            

            dbc.Card([

                html.P("Configuración a aplicar:",className="p-info", style = {"margin":" 0px 0px 20px 0px","font-size": "14px"}),
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Modelo seleccionado", className="fw-bold mb-2"),

                                    dcc.Dropdown(id="dropdown-model-selected", value=list(MODELS.keys())[0], options=list(MODELS.keys()), style={"minWidth": "200px", "maxWidth": "300px"})
                                ],
                                style = {"maxWidth": "300px"},
                                className="d-flex flex-column justify-content-center me-1"
                            ),
                            dbc.Col([
                                html.Label("Segmentación ligera", className="fw-bold mb-2"),
                                daq.BooleanSwitch(
                                    id="switch-is-default-class", 
                                    on=False,className="my-auto",color ="#1E1E1E",
                                )],
                                style = {"maxWidth": "300px"},
                                className="d-flex align-items-center me-1"
                            ),
                            dbc.Col(
                                dbc.Button(
                                    html.Div([html.I(className= "fas fa-microchip"), html.Span("Calcular Inferencia")]),
                                    id="buttoon-inference",style={"display": "none"},className="button-inference"
                                ),
                                style = {"maxWidth": "300px"},
                                className="d-flex align-items-center"
                            ),
                        ],
                        className="d-flex flex-row align-items-center mb-3",
                        style={"display":"flex","justify-content":"space-between","align-items":"center"}
                    )
                )],
                className="shadow-sm rounded",
                
            )],
            fluid=True,
            className="d-flex justify-content-center align-items-center vh-100 container-settings",
            
        ),
        html.Div([
            html.Div(id='output-data-upload'),
            dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Arrastre o ',
                        html.A('seleccione una imagen'),
                        " (formatos soportados: png y jpg)"
                    ]),
                    style={
                        'width':'100%',
                        'height':'60px',
                        'lineHeight':'60px',
                        'borderWidth':'1px',
                        'borderStyle':'dashed',
                        'borderRadius':'5px',
                        'textAlign':'center',
                        "margin":"20px 0px"
                    },
                    multiple=False
                ),

                html.Div(id='result-data-upload'),



            
        ])
    ], id = "layout")

    )

])



@callback(Output('output-data-upload','children'),
          Output('buttoon-inference','style'),
              Input('upload-data','contents'),
              State('upload-data','filename')
        )
def allow_inference(contents,filename):

    global STORE_IMG
    global FILENAME

    if contents is None:
        return dash.no_update, {"display":"none"}

    if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
        return CONS_DIV_ERROR_CASE1,{"display": "none"}
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        image = Image.open(io.BytesIO(decoded)).convert('RGB')
        np_array = np.array(image)

        STORE_IMG = np_array
        FILENAME = filename

        log_success = html.Div(f"✅ Imagen cargada: {filename}", style={
        'color': '#155724',
        'backgroundColor':'#d4edda',
        'border':'1px solid #c3e6cb',
        'padding':'10px',
        'borderRadius':'5px',
        'fontWeight':'bold',
        'marginTop':'10px'
    })

        return log_success, {"display":"inline-block"}

    except Exception as e:
        return CONS_DIV_ERROR_CASE2,{"display":"none"}

@callback(
        Output("result-data-upload","children"),
        Input('buttoon-inference','n_clicks'),
        State('dropdown-model-selected','value'),
        State('switch-is-default-class','on')
        )
def generate_inference(n_clicks,model,has_all_classes):

    if(n_clicks==0  or n_clicks==None):
        return dash.no_update
    
    global STORE_IMG
    

    modal_result=get_plots_inference(STORE_IMG,model,has_all_classes,class_names=None)
 
    return modal_result



def get_plots_inference(image,selected_model,has_all_classes,class_names):
    
    print("processing inference")
    class_map, probs = inference_model_pipeline(image = image, type_model = MODELS[selected_model])

    print("generating plot output")
    fig1, fig2 = generate_plots_modal(image,class_map )

    print("generating new layout")
    layout_card_save = generate_card_plot(fig1, fig2, model_name=selected_model)

    

    global RESULT_INFERENCE
    RESULT_INFERENCE = class_map


    return layout_card_save



@dash.callback(
    Output("url","pathname", allow_duplicate= True),
    Input('save-modal','n_clicks'),
    prevent_initial_call=True

)
def save_results(n_clicks):

    if(n_clicks==0 or n_clicks==None):
        return dash.no_update
    
    global STORE_IMG
    global RESULT_INFERENCE
    global FILENAME

    output_path=r"C:\Users\ruben\Desktop\code_tfm\src\deployment\src\outputs"
    filename_date=datetime.now().strftime("%Y%m%d_%H%M%S")
    filename=FILENAME.split(".")[0]+"_"+filename_date+".npz"
    full_path=os.path.join(output_path,filename)
    np.savez(full_path,image=STORE_IMG,inference=RESULT_INFERENCE)

    return "/history"



def generate_plots_modal(image,inference):
    fig_entrada=px.imshow(image)
    fig_entrada.update_layout(
        title="Imagen original",
        coloraxis_showscale=False,
        margin=dict(l=40,r=10,t=60,b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    # print(f"·my image shape is {image.shape} and my inference shape is inference {inference.shape}" )

    hover_text=np.vectorize(lambda x: f"{CATEGORY_INFO_OBJECTIVE.get(str(x), 'None')}")(inference)
    inferencia_preprocessed=np.flipud(inference)
    flipped_hover_text=np.flipud(hover_text)
    fig_class=go.Figure(data=go.Heatmap(
        z=inferencia_preprocessed,
        text=flipped_hover_text,
        hoverinfo='text',
        colorscale='Viridis',
        showscale=False,
        colorbar=dict(title='Clase:')
    ))
    fig_class.update_layout(
        title = "Predicción del modelo ",
        margin=dict(l=40, r=10, t=60, b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    # fig_class.update_yaxes(scaleanchor="x", scaleratio=1)fig_entrada.update_xaxes(scaleanchor="y", constrain='domain')
    fig_entrada.update_xaxes(scaleanchor="y",constrain='domain')
    fig_entrada.update_yaxes(constrain='domain',)

    fig_class.update_xaxes(scaleanchor="y",constrain='domain')
    fig_class.update_yaxes(constrain='domain')

    return fig_entrada,fig_class



def generate_card_plot(plot1,plot2,model_name):
    content_modal = [
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
                html.Div(dcc.Graph(figure=plot1), style={"flex": "0 0 30%"}),
                    html.Div(dcc.Graph(figure=plot2), style={"flex": "0 0 30%"}),
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
            [
                dbc.Button(
                    "Cerrar",
                    id="close-modal",color="primary",
                    style={
                        "background-color": "#6f42c2",
                        "border": "none",
                        "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                        "transition": "0.3s",
                    },className="me-2 custom-purple-btn modal-footer-button"
                ),
                dbc.Button(
                    "Guardar",
                    id="save-modal",
                    color="primary",
                    style={
                        "background-color":"#6f42c2", 
                        "border":"none",
                        "box-shadow": "0 4px 8px rgba(0,0,0,0.2)",
                        "transition": "0.3s",
                    },
                    className="custom-purple-btn modal-footer-button"
                ),
            ],
            className="modal-footer"
        )
    ]

    result=dbc.Modal(id="result-inference", is_open=True,backdrop="static", className="modal-content custom-centered-modal",keyboard=True, children=content_modal)


    return result


@dash.callback(
    Output("url","pathname",allow_duplicate= True),
    Input('close-modal','n_clicks'),
    prevent_initial_call=True

)
def save_results(n_clicks):

    if(n_clicks==0 or n_clicks==None):
        return dash.no_update
    
    return "/results"
