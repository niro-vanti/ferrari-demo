import streamlit as st
import os
import pandas as pd
import time
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

page_title = "Real-Time Anomaly Detection App"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"


# --------------------------------------
def get_reason(type):
    ind = np.random.randint(0, 2, 1)[0]
    sensor_reasons = [
        'Sensor value is ok, but trend is unlikely',
        'Sensor value is unlikely',
        'Sensor is showing a slow change over time that is indicative of an issue'
    ]

    situation_reasons = [
        'All sensors are ok but there''s an unlikely combination of values',
        '2 or more sensors are registering an unlikely value'
        ''
    ]

    if type == 'sensor':
        return sensor_reasons[ind]
    if type == 'situation':
        return situation_reasons[ind]
    return 'no clear cut root cause'

def color_survived(val):
    color = '#52de97' if val else 'red'
    return f'background-color: {color}'


def highlight_survived(s):
    return ['color: #52de97'] * len(s) if s['alert type'] == 'Situation' else ['color: #000000'] * len(s)


st.set_page_config(page_title=page_title, page_icon=page_icon)  # , layout=layout)
# color_scale = alt.Scale(range=['#FAFA37', '#52de97', '#c9c9c9'])

df = pd.read_csv('assets/data/anomaly.csv', index_col=0)
df['sen_alert'] = 0
df['sit_alert'] = 0

with st.sidebar:
    token = st.text_input('plug in your Vanti application Token')
    connect = st.button('connect')
    if connect:
        for i in range(10000000):
            a = 1
        st.success('connected to to model')
    with st.expander('Product Tree'):
        for col in df.columns:
            qwer = str(col)
            s1 = st.button(qwer)
    st.text(' ')
    batch = st.file_uploader("upload batch file")

st.image('assets/Images/Vanti - Main Logo@4x copy.png', width=200)
st.title(page_title)
st.text(' ')

# st.image('assets/Images/car-pano-1.jpg')


clist = ['All Sensors']

for i in df.columns.to_list():
    clist.append(i)
feats = st.multiselect("Select Sensors", clist)

st.text("You selected: {}".format(", ".join(feats)))
if 'All Sensors' in feats:
    feats = df.columns.to_list()

c1, c2 = st.columns(2)
mode = c1.radio('select alert mode',
                    ['alert me only when there''s a situation anomaly',
                     'alert me only when there''s a sensor anomaly',
                     'I want all alerts'])
with c2.expander('what are these alerts?'):
    st.write('a **sensor** anomaly is when a single sensor is tracked by Vanti''s model and the model decides to '
             'alert the user')
    st.write('a **situation** anomaly is when all sensors are tracked together by Vanti''s model and the the model '
             'decides to alert the user')

a = 1

if mode == 'alert me only when there''s a situation anomaly':
    MODE = 0
elif mode == 'alert me only when there''s a sensor anomaly':
    MODE = 1
elif mode == 'I want all alerts':
    MODE = 2
print(MODE)



sensitivity = c1.slider('alert sensitivity', 0.0, 100.0, 50.0)
with c2.expander("what is model sensitivity?"):
    st.write("_sensitivity 100 --> alert me on **everything**_")
    st.write("_sensitivity 0 --> alert me on **critical things only**_")

ms = {i: df[i].mean() for i in feats}
ss = {i: df[i].std() for i in feats}

alerts = pd.DataFrame()
c1, c2, c3 = st.columns(3)
stream = c1.button('Start Injection mode')
dont = c3.button('Start Real Time Monitor')
sensitivity = (100 - sensitivity) / 10

temp = df[feats].iloc[:2].copy()

window = 300
pl = st.empty()
pl2 = st.empty()
alerts = pd.DataFrame()

# tab1, tab2 = st.tabs(['alert table','alert graph'])

if stream:
    stop_stream = c2.button('Stop Injection')
    if stop_stream:
        stream = False

    for i in range(df.shape[0]):
        s = max(0, i - window)
        e = min(i, df.shape[0])
        temp = df[feats].iloc[s:e]
        temp['sen_alert'] = df['sen_alert'].iloc[s:e]
        temp['sit_alert'] = df['sit_alert'].iloc[s:e]
        count = 0
        for f in feats:
            if np.abs(df[f].iloc[i] - ms[f]) > (ss[f] * sensitivity):
                count = count + 1
                rr = get_reason('sensor')
                q = pd.DataFrame({
                    'time stamp': [df.index[i]],
                    'sensor': [f],
                    'reason': [rr],
                    'alert type': ['sensor']
                })
                if MODE == 1 or MODE == 2:
                    alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
                    df['sen_alert'].iloc[i] = 1
                    # print('writing sen alert to ', i)

                    with pl2.container():
                        sss = max(0, i - 10)
                        eee = min(i, df.shape[0])
                        temp2 = df[f].iloc[sss:eee]

                        fig3 = px.line(temp2)
                    with st.expander('sensor-alert zoom-in @' + str(df.index[i])):
                        st.write(fig3, title=str(df.index[i]))

        if MODE == 0 or MODE == 2:
            if count > 3:
                rr = get_reason('situation')
                q = pd.DataFrame({
                    'time stamp': [df.index[i]],
                    'sensor': ['combination'],
                    'reason': [rr],
                    'alert type': ['Situation']
                })
                alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
                df['sit_alert'].iloc[i] = 1

                with pl2.container():
                    sss = max(0, i - 10)
                    eee = min(i, df.shape[0])
                    temp2 = df[feats].iloc[sss:eee]
                    fig3 = px.line(temp2)
                with st.expander('situation-alert zoom-in @' + str(df.index[i])):
                    st.write(fig3, title=str(df.index[i]))


        with pl.container():
            # st.text(str(np.round(i / df.shape[0] * 100, 2)) + ' %')
            fig = px.line(data_frame=temp)
            st.write(fig)
            # time.sleep(0.1)
            st.dataframe(alerts.style.apply(highlight_survived, axis=1))
            fig2 = px.line(temp[['sen_alert', 'sit_alert']].cumsum())
            st.write(fig2)


            # st.line_chart(temp2.iloc[sss:eee])

# with st.expander('alert table'):
#     st.dataframe(alerts.style.apply(highlight_survived, axis=1))

with st.expander('see training data'):
    st.dataframe(df)



