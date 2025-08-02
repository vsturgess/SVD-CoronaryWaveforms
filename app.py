import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
import math

# Load in Matrix A data and make each waveform
t = np.arange(0, 2, 0.01)
data = {}
pigs =[]
pigsraw =[]
pigsnorm =[]
n=15
with open("PigMatrixAraw.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) != 32:
            continue  # skip malformed lines
        pig_id = parts[0]
        pig_zero_freq = float(parts[1])
        real_arr = np.array(list(map(float, parts[2:n+2])))
        imag_arr = np.array(list(map(float, parts[n+2:2*n+2])))
        y = pig_zero_freq
        complex_arr=real_arr + 1j * imag_arr
        for idx, complex_val in enumerate(complex_arr):
            mag_val=np.abs(complex_val)
            theta = np.angle(complex_val)
            y=y+2.0*mag_val*np.cos(2*math.pi*(idx+1)*t+theta)
        data[pig_id] = y
        pigs.append(pig_id)
        pigsraw.append(pig_id)

data_df_raw = pd.DataFrame(data)

data = {}
with open("PigMatrixAnorm.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) != 31:
            continue  # skip malformed lines
        pig_id = parts[0]
        real_arr = np.array(list(map(float, parts[1:n+1])))
        imag_arr = np.array(list(map(float, parts[n+1:2*n+1])))
        y = 0.0
        complex_arr=real_arr + 1j * imag_arr
        for idx, complex_val in enumerate(complex_arr):
            mag_val=np.abs(complex_val)
            theta = np.angle(complex_val)
            y=y+2.0*mag_val*np.cos(2*math.pi*(idx+1)*t+theta)
        data[pig_id] = y
        pigsnorm.append(pig_id)

data_df_norm = pd.DataFrame(data)
#print(f"Length of average features (raw): {len(data_df_norm['OSS1179'])}")

#Load in the Left Singular Vectors
modes_raw = {}
with open("LeftSVs_raw.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        mode_id = parts[0]
        mode_vals=np.array(list(map(float,parts[1:2*n+2])))
        modes_raw[mode_id] = mode_vals
modes_df_raw = pd.DataFrame(modes_raw)

#print(f"Length of average features (raw): {len(modes_df_raw['Avg'])}")
#print(f"Length of Mode1 features (raw): {len(modes_df_raw['Mode1'])}")

modes_norm = {}
with open("LeftSVs_norm.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        mode_id = parts[0]
        mode_vals=np.array(list(map(float,parts[1:2*n+1])))
        modes_norm[mode_id] = mode_vals
modes_df_norm = pd.DataFrame(modes_norm)

#print(f"Length of average features (norm): {len(modes_df_norm['Avg'])}")
#print(f"Length of Mode1 features (noprm): {len(modes_df_norm['Mode1'])}")

#Load in the Singular Values
with open("singularValues_raw.txt", "r") as file:
    singular_values_raw = [float(line.strip()) for line in file.readlines()]

sigma1_raw, sigma2_raw, sigma3_raw, sigma4_raw, sigma5_raw = singular_values_raw[:5]

with open("singularValues_norm.txt", "r") as file:
    singular_values_norm = [float(line.strip()) for line in file.readlines()]

sigma1_norm, sigma2_norm, sigma3_norm, sigma4_norm, sigma5_norm = singular_values_norm[:5]

#Load in the standard deviations of Raw Right Singular Vectors
with open("RSV_stdevs_raw.txt", "r") as file:
    RSV_stdevs_raw = [float(line.strip()) for line in file.readlines()]

RSV_s1_raw, RSV_s2_raw, RSV_s3_raw, RSV_s4_raw, RSV_s5_raw = RSV_stdevs_raw[:5]

with open("RSV_stdevs_norm.txt", "r") as file:
    RSV_stdevs_norm = [float(line.strip()) for line in file.readlines()]

RSV_s1_norm, RSV_s2_norm, RSV_s3_norm, RSV_s4_norm, RSV_s5_norm = RSV_stdevs_norm[:5]

#Calculate Avergae Signals
avg_signal_raw = np.zeros_like(t)
avg_raw=modes_df_raw['Avg']   
zero_freq=avg_raw.iloc[0]
real_arr = avg_raw.iloc[1:16].to_numpy()
imag_arr = avg_raw.iloc[16:31].to_numpy()
complex_arr = real_arr + 1j * imag_arr
avg_signal_raw=avg_signal_raw+zero_freq
for idx, complex_val in enumerate(complex_arr):
    #print("Complex value:", complex_val)
    mag_val=np.abs(complex_val)
    theta = np.angle(complex_val)
    avg_signal_raw=avg_signal_raw+2.0*mag_val*np.cos(2*math.pi*(idx+1)*t+theta)

avg_signal_norm = np.zeros_like(t)
avg_norm=modes_df_norm['Avg']   
real_arr = avg_norm.iloc[0:15].to_numpy()
imag_arr = avg_norm.iloc[15:30].to_numpy()
complex_arr = real_arr + 1j * imag_arr
for idx, complex_val in enumerate(complex_arr):
    #print("Complex value:", complex_val)
    mag_val=np.abs(complex_val)
    theta = np.angle(complex_val)
    avg_signal_norm=avg_signal_norm+2.0*mag_val*np.cos(2*math.pi*(idx+1)*t+theta)


#Load in data for correlation matrix
Corr_df_raw = pd.read_csv('CorrDataRaw.txt')
Corr_df_norm = pd.read_csv('CorrDataNorm.txt')
print(Corr_df_raw.columns)

fact1_raw=sigma1_raw*RSV_s1_raw
fact2_raw=sigma2_raw*RSV_s2_raw
fact3_raw=sigma3_raw*RSV_s3_raw
fact4_raw=sigma4_raw*RSV_s4_raw
fact5_raw=sigma5_raw*RSV_s5_raw

fact1_norm=sigma1_norm*RSV_s1_norm
fact2_norm=sigma2_norm*RSV_s2_norm
fact3_norm=sigma3_norm*RSV_s3_norm
fact4_norm=sigma4_norm*RSV_s4_norm
fact5_norm=sigma5_norm*RSV_s5_norm

# Default slider values (for initialization)
default_a = -1.0
default_b = 0.0
default_c = 0.0
default_d = 0.0
default_e = 0.0

xaxis_label_lookup = {
    'Heart Weight': "Heart Weight (g)",
    'Body Weight': "Body Weight (kg)",
    'EndoEpi': "Endo/Epi Flow Ratio",
    'HR': "Heart Rate (beats per minutes)",
    'MBP': "Mean BP (mmHg)",
    'dPLVdtmax': "max dPLV/dt (mmHg/s)",
    'dPLVdtmin': "min dPLV/dt (mmHg/s)",
    'Average Flow': "Avg Flow (mL/min)",
    'max LV pressure': "Max LV pressure (mmHg)",
    'min LV pressure': "Min LV pressure (mmHg)",
    'Flow Pulsatility': "Flow Pulsatility (mL/min)",
    'LVEDP': "Left Ventricular End-Diastolic Pressure (mmHg)",
    'Venous O2 Content': "Coronary Venous O2 Content (UNITS)",
    'O2 Extraction': "O2 Extraction (UNITS)",
    'Hyperemic Flow': "Hyperemic Flow (mL/min)",
    'CFR': 'Coronary Flow Reserve',
    'Avgerage Flow- HW Normalized': "Avg Flow Normalized to HW (mL/min/g)",
    'CCV value': "CCV (s)",
    'CCV percent': "CCV (%)",
    'MSE_raw': "MSE",
    'MSE_norm': "MSE"
}

yaxis_label_lookup = {
    'Mode1': "Mode 1 (Z-Score)",
    'Mode2': "Mode 2 (Z-Score)",
    'Mode3': "Mode 3 (Z-Score)",
    'Mode4': "Mode 4 (Z-Score)",
    'Mode5': "Mode 5 (Z-Score)",
}

# Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Pig Signal Viewer"

app.layout = html.Div([ 
    html.Div(style={'display': 'flex', 'height': '90vh'}, children=[

        # Left: Vertical Tabs and dynamic content
        html.Div(style={'width': '25%', 'borderRight': '1px solid #ccc', 'padding': '20px'}, children=[
            dcc.Tabs(
                id='left-tabs',
                value='tab-data',  # Default value set to 'tab-data' to trigger the plot on startup
                vertical=False,
                children=[
                    dcc.Tab(label='General Info', value='tab-info'),
                    dcc.Tab(label='Interact with Data', value='tab-data'),
                ],
                style={'height': '100%'}
            ),
            html.Div(id='left-tab-content', style={'marginTop': '20px'})
        ]),

        
        # Right: Graph always shown
        html.Div(style={'width': '70%', 'padding': '20px'}, children=[
            html.H1("SVD Shape Analysis of Coronary Waveform",style={'textAlign': 'center'}),
            html.Div([
                dcc.Graph(id='waveform-plot', style={'marginTop': '0px','marginBottom': '0px','flex': '1 1 350px','maxWidth': '1200px',})
            ]),
            
            html.Div(style={'height': '15px'}),
            html.Div("Correlation Plots",
                        style={
                            'fontSize': '24px',
                            'fontFamily': 'Segoe UI',
                            'textAlign': 'center',
                            'margin': '10', 
                            "color": '#383838 ',
                        }
                    ),
            html.Div(
                        id='legend-container',
                        style={
                            'flex': '0 1 200px',  # Doesn't grow, but can shrink
                            'minWidth': '50px',
                            'display': 'flex',
                            'margin': '10',
                            'flexDirection': 'column',
                            'justifyContent': 'center',
                            'alignItems': 'center',
                            'backgroundColor': "#ffffff",
                            "color": '#808080 ',
                            'fontSize': '14px',
                            
                        },
                        children=[
                            html.H4("Legend")
                        ]),
            html.Div(
                style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'space-evenly',
                },
                children=[
                    dcc.Graph(
                        id='scatter-plot-1',
                        style={'flex': '1 1 200px', 'minWidth': '50px','maxWidth': '300px',},
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                        }
                    ),
                    dcc.Graph(
                        id='scatter-plot-2',
                        style={'flex': '1 1 200px', 'minWidth': '50px','maxWidth': '300px'},
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                        }
                    ),
                    
                ]
            )

        ])

    ])
])

# Callback to render content in the left panel based on selected tab
@app.callback(
    Output('left-tab-content', 'children'),
    Input('left-tabs', 'value')
)
def render_left_tab(tab):
    if tab == 'tab-info':
        return html.Div([
            html.H4("App Overview"),
            html.P("This app allows users to visualize temporal features of the shapes of coronary flow waveforms and interact with results from a corresponding singular value decomposition."),
            html.P("To adjust plots on the rihgt, switch to the 'Interact with Data' tab above."),
            html.Div(style={'height': '20px'}),
            html.P("This work has been presented in XXX."),
        ])
    elif tab == 'tab-data':
        return html.Div([
            dcc.Dropdown(
                id='Analysis Type',
                options=['Raw Waveform', 'Normalized Waveform'],
                value='Raw Waveform',  # You can adjust the default selected pig
                placeholder="Select which analysis to use"
            ),

            html.H3("Waveform Settings",style={'textAlign': 'center'}),
            html.Label("Select Pigs to Display"),
            dcc.Dropdown(
                id='pig-checklist',
                options=[{'label': pig, 'value': pig} for pig in pigs if pig != 'Avg'],
                value=[],  # You can adjust the default selected pig
                multi=True,
                placeholder="Select pigs to display"
            ),
            html.Div(style={'height': '20px'}),
            html.Label("Mode 1"),
            dcc.Slider(id='slider-a', min=-2, max=2, value=default_a,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style={'height': '5px'}),
            html.Label("Mode 2"),
            dcc.Slider(id='slider-b', min=-2, max=2, value=default_b,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style={'height': '5px'}),
            html.Label("Mode 3"),
            dcc.Slider(id='slider-c', min=-2, max=2, value=default_c,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style={'height': '5px'}),
            html.Label("Mode 4"),
            dcc.Slider(id='slider-d', min=-2, max=2, value=default_d,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style={'height': '5px'}),
            html.Label("Mode 5"),
            dcc.Slider(id='slider-e', min=-2, max=2, value=default_e,
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Div(style={'height': '25px'}),
            html.Hr(),
            html.Div(style={'height': '10px'}),
            html.H3("Correlation Plots",style={'textAlign': 'center'}),

            html.Label("Select mode and data for left plot:"),
            html.Div(style={'height': '10px'}),
            dcc.Dropdown(
                id='mode1_choice',
                options=[{'label': col, 'value': col} for col in Corr_df_norm.columns if 'Mode' in col],
                value='Mode1',  # You can adjust the default selected pig
                placeholder="Select mode for left scatter plot"
            ),
            html.Div(style={'height': '10px'}),
            dcc.Dropdown(
                id='mode1_compare',
                options=[{'label': col, 'value': col} for col in Corr_df_norm.columns if 'Mode' not in col and 'Row' not in col],
                value='Average Flow',  # You can adjust the default selected pig
                placeholder="Select data for left scatter plot"
            ),

            html.Div(style={'height': '30px'}),

            html.Label("Select mode and data for right plot:"),
            html.Div(style={'height': '10px'}),
            dcc.Dropdown(
                id='mode2_choice',
                options=[{'label': col, 'value': col} for col in Corr_df_norm.columns if 'Mode' in col],
                value='Mode2',  # You can adjust the default selected pig
                placeholder="Select mode for right scatter plot"
            ),
            html.Div(style={'height': '10px'}),
            dcc.Dropdown(
                id='mode2_compare',
                options=[{'label': col, 'value': col} for col in Corr_df_norm.columns if 'Mode' not in col and col != 'Row'],
                value='HR',  # You can adjust the default selected pig
                placeholder="Select data for right scatter plot"
            ),

            html.Div(style={'height': '30px'}),

            html.Label("Select color option:"),
            html.Div(style={'height': '10px'}),
            dcc.Dropdown(
                id='ColorType',
                options=['None', 'By Experimental Group'],
                value='None',  # You can adjust the default selected pig
                placeholder="Select color option"
            ),

        ])

# Callback to update the plot based on pigs and sliders
@app.callback(
    Output('waveform-plot', 'figure'),
    Output('scatter-plot-1', 'figure'),
    Output('scatter-plot-2', 'figure'),
    Output('legend-container', 'children'),  # New output for legend content
    Input('Analysis Type', 'value'),
    Input('pig-checklist', 'value'),
    Input('slider-a', 'value'),
    Input('slider-b', 'value'),
    Input('slider-c', 'value'),
    Input('slider-d', 'value'),
    Input('slider-e', 'value'),
    Input('mode1_choice', 'value'),
    Input('mode1_compare', 'value'),
    Input('mode2_choice', 'value'),
    Input('mode2_compare', 'value'),
    Input('ColorType', 'value'),
)
def update_plot(T,selected_pigs, a, b, c, d, e, m1, d1, m2, d2,CT):
    fig = go.Figure()

    #force line to appear at y=0
    fig.add_trace(go.Scatter(
        x=[0, 2],
        y=[0, 0],
        mode='lines',
        line=dict(color='lightgrey', width=1),
        name='Zero Line',
        hoverinfo='skip',
        showlegend=False
    ))

    # Always show Avg
    if T=='Raw Waveform':
        fig.add_trace(go.Scatter(
            x=t,
            y=avg_signal_raw,
            mode='lines',
            name='Avg',
            line=dict(color='black', width=2)
        ))
    elif T=='Normalized Waveform':
        fig.add_trace(go.Scatter(
            x=t,
            y=avg_signal_norm,
            mode='lines',
            name='Avg',
            line=dict(color='black', width=2)
        ))

    # Show selected pigs
    if T=='Raw Waveform':
        color_palette = px.colors.qualitative.Pastel2
        for i, pig in enumerate(selected_pigs):
            color = color_palette[i % len(color_palette)]  # Ensure we don't go out of bounds and avoiding red
            fig.add_trace(go.Scatter(
                x=t,
                y=data_df_raw[pig],
                mode='lines',
                name=pig,
                line=dict(color=color)
            ))
    elif T=='Normalized Waveform':
        color_palette = px.colors.qualitative.Pastel2
        for i, pig in enumerate(selected_pigs):
            color = color_palette[i % len(color_palette)]  # Ensure we don't go out of bounds and avoiding red
            fig.add_trace(go.Scatter(
                x=t,
                y=data_df_norm[pig],
                mode='lines',
                name=pig,
                line=dict(color=color)
            ))

    # Show test signal
    y_test = np.zeros_like(t)

    if T=='Raw Waveform':
        pig_test=modes_df_raw['Avg']+fact1_raw*a*modes_df_raw['Mode1']+fact2_raw*b*modes_df_raw['Mode2']+ \
            fact3_raw*c*modes_df_raw['Mode3']+fact4_raw*d*modes_df_raw['Mode4']+fact5_raw*e*modes_df_raw['Mode5']
        
        zero_freq=pig_test.iloc[0]
        real_arr = pig_test.iloc[1:16].to_numpy()
        imag_arr = pig_test.iloc[16:31].to_numpy()
        complex_arr = real_arr + 1j * imag_arr
        y_test=y_test+zero_freq
        for idx, complex_val in enumerate(complex_arr):
            #print("Complex value:", complex_val)
            mag_val=np.abs(complex_val)
            theta = np.angle(complex_val)
            y_test=y_test+2.0*mag_val*np.cos(2*math.pi*(idx+1)*t+theta)

    elif T=='Normalized Waveform':
        pig_test=modes_df_norm['Avg']+fact1_norm*a*modes_df_norm['Mode1']+fact2_norm*b*modes_df_norm['Mode2']+ \
            fact3_norm*c*modes_df_norm['Mode3']+fact4_norm*d*modes_df_norm['Mode4']+fact5_norm*e*modes_df_norm['Mode5']
        print(f"Length of Pig Test array: {len(pig_test)}")
        real_arr = pig_test.iloc[0:15].to_numpy()
        imag_arr = pig_test.iloc[15:30].to_numpy()
        complex_arr = real_arr + 1j * imag_arr
        for idx, complex_val in enumerate(complex_arr):
            #print("Complex value:", complex_val)
            mag_val=np.abs(complex_val)
            theta = np.angle(complex_val)
            y_test=y_test+2.0*mag_val*np.cos(2*math.pi*(idx+1)*t+theta)
    #print("y_test[:10]:", y_test[:10]) # outputs in terminal, can use to check what values are

    fig.add_trace(go.Scatter(
        x=t,
        y=y_test,
        mode='lines',
        name='Test Signal',
        line=dict(color='red', dash='dash', width=2)
    ))
    if T=='Raw Waveform':
        fig.update_layout(
            plot_bgcolor='white',
            title={'text':'Coronary Flow Waveforms','x':0.5, 'xanchor': 'center','font': {'size': 24,'family': 'Segoe UI','color': '#383838'}},
            xaxis={'title': 'Cardiac Cycles', 'range': [0, 2], 'fixedrange': True,'gridcolor':'lightgrey'},  # Set x-axis range to [0, 5]
            yaxis={'title': 'Flow (mL/min)', 'range': [-30, 150], 'fixedrange': True,'gridcolor':'lightgrey','tick0': 0.0,'dtick': 20},
            legend=dict(orientation='h', yanchor='bottom', y=0.9, xanchor='right', x=1),
            #shapes=[dict(type='line',x0=0, x1=2, y0=0, y1=0, line=dict(color='lightgrey', width=1), xref='x',yref='y', layer='above')]
        )
    elif T=='Normalized Waveform':
        fig.update_layout(
            plot_bgcolor='white',
            #title={'text':'Coronary Flow Waveforms','x':0.5, 'xanchor': 'center'},
            xaxis={'title': 'Cardiac Cycles', 'range': [0, 2], 'fixedrange': True,'gridcolor':'lightgrey'},  # Set x-axis range to [0, 5]
            yaxis={'title': 'Normalized Flow Shape', 'range': [-1.55, 1.55], 'fixedrange': True,'gridcolor':'lightgrey','tick0': 0.0,'dtick': 0.5},
            legend=dict(orientation='h', yanchor='bottom', y=0.9, xanchor='right', x=1),
            #shapes=[dict(type='line',x0=0, x1=2, y0=0, y1=0, line=dict(color='lightgrey', width=1), xref='x',yref='y', layer='above')]
        )

    print(m1)
    print(d1)
    print(d2)

    # If d1 or d2 are None, return empty scatter plots
    if d1 is None or m1 is None:
        scatter_fig1 = go.Figure()
        scatter_fig1.update_layout(
            plot_bgcolor='white',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': "No data selected",
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
    else:
        if T=='Raw Waveform':
            y_scatter1=Corr_df_raw[m1]
            x_scatter1=Corr_df_raw[d1]
            text=Corr_df_raw['Row']
        else:
            y_scatter1=Corr_df_norm[m1]
            x_scatter1=Corr_df_norm[d1]
            text=Corr_df_norm['Row']

        x_min1 = x_scatter1.min()
        x_max1 = x_scatter1.max()
        x_pad1 = (x_max1 - x_min1) * 0.1

        y_lim1 = max(abs(y_scatter1))
        y_pad1 = y_lim1 * 0.3

        xaxis_title1 = xaxis_label_lookup.get(d1, None)
        yaxis_title1 = yaxis_label_lookup.get(m1, None)

        scatter_fig1 = go.Figure()

        # Grey horizontal line at y=0
        scatter_fig1.add_trace(go.Scatter(
            x=[x_min1-x_pad1, x_max1+x_pad1],
            y=[0, 0],
            mode='lines',
            line=dict(color='lightgrey', width=1),
            name='Zero Line',
            hoverinfo='skip',
            showlegend=False
        ))

        # Main data
        if CT is None or CT =='None':
            scatter_fig1.add_trace(go.Scatter(
                x=x_scatter1, 
                y=y_scatter1, 
                mode='markers',
                marker=dict(color='black', size=10),
                name='Data',
                showlegend=False,
                text=Corr_df_norm['Row'], #The names of the norm and raw Corr matrix must be in the same order
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "X: %{x}<br>" +
                    "Y: %{y}<br>" +
                    "<extra></extra>"
                )
            ))
        elif CT == 'By Experimental Group':
            colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'pink', 5: 'black'}  # assign your desired colors
            groupNames = {1: 'Oss Lean', 2: 'Oss Lean Paced', 3: 'Oss Obese Paced', 4:'York Lean', 5:'York Lean Paced'}  # assign your desired colors
            for group in [1, 2, 3, 4, 5]:
                mask = Corr_df_raw['Row Type'] == group
                custom_data = list(zip(
                        Corr_df_raw['Row'][mask],
                        [groupNames[group]] * mask.sum()
                    ))
                scatter_fig1.add_trace(go.Scatter(
                    x=x_scatter1[mask],
                    y=y_scatter1[mask],
                    mode='markers',
                    marker=dict(color=colors[group], size=10),
                    text=Corr_df_raw['Row'][mask],
                    hovertemplate=(
                        "<b>%{text}</b><br>" +  # Pig name
                        "X: %{x}<br>" +
                        "Y: %{y}<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))
                '''
            scatter_fig1.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0.9,
                    xanchor="center",
                    x=0.5,
                    traceorder="normal",
                    itemwidth=40,  # Adjust so that ~3 items fit per row
                    font=dict(size=12)
                )
            )'''

        scatter_fig1.update_layout(
            plot_bgcolor='white',
            yaxis={'title': yaxis_title1, 'range': [-y_lim1-y_pad1, y_lim1+y_pad1], 'fixedrange': True,'gridcolor':'lightgrey'},
            xaxis={'title': xaxis_title1, 'gridcolor':'lightgrey','range': [x_min1 - x_pad1, x_max1 + x_pad1],'fixedrange': True,},
            title={'text':'Scatter Plot 1','x':0.5, 'xanchor': 'center','font': {'size': 18,'family': 'Segoe UI','color': '#383838'}}, 
            margin=dict(l=20, r=20, t=30, b=20))

    # If m1 or d2 are None, return empty scatter plots
    if m2 is None or d2 is None:
        scatter_fig2 = go.Figure()
        scatter_fig2.update_layout(
            plot_bgcolor='white',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': "No data selected",
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
    else:
        if T=='Raw Waveform':
            y_scatter2=Corr_df_raw[m2]
            x_scatter2=Corr_df_raw[d2]
            text=Corr_df_raw['Row']
        else:
            y_scatter2=Corr_df_norm[m2]
            x_scatter2=Corr_df_norm[d2]
            text=Corr_df_norm['Row']

        x_min2 = x_scatter2.min()
        x_max2 = x_scatter2.max()
        x_pad2 = (x_max2 - x_min2) * 0.1

        y_lim2 = max(abs(y_scatter2))
        y_pad2 = y_lim2 * 0.3
        xaxis_title2 = xaxis_label_lookup.get(d2, None)
        yaxis_title2 = yaxis_label_lookup.get(m2, None)


        scatter_fig2 = go.Figure()

        # Grey horizontal line at y=0
        scatter_fig2.add_trace(go.Scatter(
            x=[x_min2-x_pad2, x_max2+x_pad2],
            y=[0, 0],
            mode='lines',
            line=dict(color='lightgrey', width=1),
            name='Zero Line',
            hoverinfo='skip',
            showlegend=False
        ))

        # Main data
        if CT is None or CT =='None':
            scatter_fig2.add_trace(go.Scatter(
                x=x_scatter2, 
                y=y_scatter2, 
                mode='markers',
                marker=dict(color='black', size=10),
                text=Corr_df_raw['Row'],
                hovertemplate=(
                    "<b>%{text}</b><br>" +  # Pig name
                    "X: %{x}<br>" +
                    "Y: %{y}<br>" +
                    "<extra></extra>"
                ),
                name='Data',
                showlegend=False
            ))
        elif CT == 'By Experimental Group':
            colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'pink', 5: 'black'}  # assign your desired colors
            groupNames = {1: 'OssLean', 2: 'OssLeanPaced', 3: 'OssObesePaced', 4:'YorkLean', 5:'YorkLeanPaced'}  # assign your desired colors
            for group in [1, 2, 3, 4, 5]:
                mask = Corr_df_raw['Row Type'] == group
                custom_data = list(zip(
                        Corr_df_raw['Row'][mask],
                        [groupNames[group]] * mask.sum()
                    ))
                scatter_fig2.add_trace(go.Scatter(
                    x=x_scatter2[mask],
                    y=y_scatter2[mask],
                    mode='markers',
                    marker=dict(color=colors[group], size=10),
                    name=groupNames[group],
                    text=Corr_df_raw['Row'][mask],
                    hovertemplate=(
                        "<b>%{text}</b><br>" +  # Pig name
                        "X: %{x}<br>" +
                        "Y: %{y}<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))
                '''
            scatter_fig2.update_layout(legend=dict(
                    orientation='v',         # Horizontal layout
                    yanchor='bottom',
                    y=0.5,
                    xanchor="center",
                    x=1.2,
                    traceorder="normal",
                    itemwidth=40,  # Adjust so that ~3 items fit per row,
                    font=dict(size=10)
            ))'''

        scatter_fig2.update_layout(
            plot_bgcolor='white',
            title={'text':'Scatter Plot 2','x':0.5, 'xanchor': 'center','font': {'size': 18,'family': 'Segoe UI','color': '#383838'}}, 
            yaxis={'title': yaxis_title2, 'range': [-y_lim2-y_pad2, y_lim2+y_pad2], 'fixedrange': True,'gridcolor':'lightgrey'},
            xaxis={'title': xaxis_title2, 'gridcolor':'lightgrey','range': [x_min2 - x_pad2, x_max2 + x_pad2],'fixedrange': True,},
            margin=dict(l=20, r=20, t=30, b=20))
        
        if CT == 'By Experimental Group':
            legend_children = [
                #html.P("Select modes and experimental data in the tab to the left", style={'margin': '0'}),
                html.P("Hover over a marker to see the pig ID", style={'margin': '0'}),
                html.P("Experimental group types:", style={'margin': '0'}),
                html.P("Oss Lean - red", style={'margin': '0', 'color': 'red'}),
                html.P("Oss Lean Paced - green", style={'margin': '0', 'color': 'green'}),
                html.P("Oss Obese Paced - blue", style={'margin': '0', 'color': 'blue'}),
                html.P("York Lean - pink", style={'margin': '0', 'color': 'pink'}),
                html.P("York Lean Paced - black", style={'margin': '0', 'color': 'black'}),
            ]
        else:
            legend_children = [html.P("Hover over a marker to see the pig ID", style={'margin': '0'}),
                               html.P("To see experimental groups, change the color option", style={'margin': '0'})]

    return fig, scatter_fig1, scatter_fig2,legend_children

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

