from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.express as px
import pandas as pd
import numpy as np
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
df['age_group'] = np.where(df.Age.values<17, 'younger', 'adult')
df['Pclass'] = df['Pclass'].astype(str)

# agrupado por adulto y joven
df_count = df[['Sex', 'age_group','Survived', 'Pclass']].groupby(['Sex', 'age_group', 'Pclass']).count()
df_count.reset_index(inplace=True)
total_people = sum(df_count.Survived)

df_sum = df[['Sex', 'age_group','Survived', 'Pclass']].groupby(['Sex', 'age_group', 'Pclass']).sum()
df_sum.reset_index(inplace=True)
total_survived = sum(df_sum.Survived)

# solo total de los grupos
df_sum['total'] = df_count.Survived
df_sum['relative_survived'] = df_sum.Survived/df_count.Survived

# agrupado por edad puntual
# agrupado por adulto y joven
df_count_age = df[['Sex', 'Age','Survived', 'Pclass']].groupby(['Sex', 'Age', 'Pclass']).count()
df_count_age.reset_index(inplace=True)
total_people_age = sum(df_count.Survived)

df_sum_age = df[['Sex', 'Age','Survived', 'Pclass']].groupby(['Sex', 'Age', 'Pclass']).sum()
df_sum_age.reset_index(inplace=True)
total_survived_age = sum(df_sum.Survived)

# solo total de los grupos
df_sum_age['total'] = df_count_age.Survived
df_sum_age['relative_survived'] = df_sum_age.Survived/df_count_age.Survived


# estimaciones
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
fig2 = px.bar(df_sum_age, x="Age", y="PassengerId", color='Sex', barmode='relative', title= 'BAR PLOT Passegers')


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