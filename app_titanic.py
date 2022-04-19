# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.express as px
import pandas as pd
from sklearn.metrics import roc_auc_score
import dash_bootstrap_components as dbc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification


app = Dash(__name__)

# Data de interes
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv('data/titanic.csv')
df.Age = round(df.Age, 0)
df_count = df.groupby(['Age', 'Sex', 'Survived']).count()
df_count.reset_index(inplace=True)
df_roc = pd.read_csv('data/data_roc.csv')


# ---------------------------------------------------------------------------------------------------
# GRAFICA 1
fig1 = px.area(
    x=df_roc.fpr, y=df_roc.tpr,
    title=f'ROC CURVE (AUC={auc(df_roc.fpr, df_roc.tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate')
)

fig1.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

# ---------------------------------------------------------------------------------------------------
# GRAFICA 2
fig2 = px.bar(df_count, x="Age", y="PassengerId", color='Sex', barmode='relative', title= 'BAR PLOT')


# ---------------------------------------------------------------------------------------------------
# ESTRUCTURA DE SALIDA
app.layout = html.Div(children=[
    html.H1(children='Titanic Supervenience'),

    html.Div([
        html.H2('General Result of model'),
    ],
        style={'width': '49%', 'display': 'inline-block'}
    ),

    html.Div([
        html.H2('Multi-Select respect of category, sex and years by kind of person'),
    ],
        style={'width': '50%', 'display': 'inline-block'}
    ),

    html.Div([

        dcc.Graph(
            id='example-graph1',
            figure=fig1
                ),
        ],
        style={'width': '50%', 'display': 'inline-block'}
        ),

    html.Div([

        html.H3(children='''____________________________________________________________________'''),
        html.H3(children='''Clase'''),

        dcc.RadioItems(['Primera', 'Segunda','Tercera'], ' ', inline=True),

        html.H3(children='''____________________________________________________________________'''),
        html.H3(children='''Grupo de edad aparentada'''),

        dcc.RadioItems(['Menor de edad', 'Adulto'], ' ', inline=True),
        html.H3(children='''____________________________________________________________________'''),

        html.H3(children='''Sexo'''),
        dcc.RadioItems(['Mujer', 'Hombre'], ' ', inline=True),
        html.H3(children='''____________________________________________________________________'''),

        ],
        style={'width': '50%', 'display': 'inline-block'}
        ),

    html.H2(children='Filter Result'),

    dcc.Graph(
        id='example-graph2',
        figure=fig2
    )

])

if __name__ == '__main__':
    app.run_server(debug=True)