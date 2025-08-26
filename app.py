import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


app = Dash(__name__, use_pages=True, external_stylesheets=[
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
])
# print("loaded pages are",[page['name'] for page in  dash.page_registry.values()])


icons=["fas fa-brain","fas fa-clock-rotate-left", "fas fa-chart-line"]
class_colors_icons=["pink-icon", "None", "None2"]

navbar=  html.Div([
        
        html.Img(src="assets/web-icon.png", style={
            'width': '80%',  
            'display': 'block',  
            'margin': '0 auto' ,
            "margin-bottom": "100px"
        })] + [


        html.Div(
                dcc.Link(
                     html.Div([
                         html.I(className=icons[i],id=class_colors_icons[i]),
                         html.Span(page['name'])
                     ],className="navbar-entry"),
                    href=page["relative_path"],className="navbar-link",style={"margin":"10px 0", })
                    
        ) for i, page in enumerate(dash.page_registry.values())
    ], className="navbar")



app.layout = html.Div([
    dcc.Location(id="url"),
    navbar, 
    html.Div(dash.page_container,className="content")
    
], className="web-layout")

if __name__ == '__main__':
    app.run(debug=True)