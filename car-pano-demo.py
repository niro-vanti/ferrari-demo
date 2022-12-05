import altair as alt
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
import os
import pandas as pd

page_title = "Paint shop Visual Defect Detection App"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"

st.set_page_config(page_title=page_title, page_icon=page_icon)  # , layout=layout)
color_scale = alt.Scale(range=['#FAFA37', '#52de97', '#c9c9c9'])

N = 146
zoom_names = []
defect_list = {}
for file in os.listdir('assets/Images'):
    if '_zoom' in file:
        it = file.split('_')[0]
        dif = file.split('_')[-1].split('.')[0]
        defect_list[it] = dif
# print(defect_list)



def get_class():
    classes = ['scratch','dirt','discoloration']
    ind = np.random.randint(0,2,1)[0]
    return classes[ind]

alerts = pd.DataFrame()


with st.sidebar:

    token = st.text_input('plug in your Vanti application Token')
    connect = st.button('connect')
    if connect:
        for i in range(10000000):
            a = 1
        st.success('connected to model')
    stream = st.button('Start Injection mode')
    stop_stream = st.button('Stop Injection')



st.image('assets/Images/Vanti - Main Logo@4x copy.png', width=200)
st.title(page_title)
st.text(' ')
st.image('assets/Images/ferrari-cropped.png')

c1, c2 = st.columns(2)
# stream = c1.button('Start Injection mode')

pl = st.empty()
p2 = st.empty()
N = 146
cls = ''
is_error = False


if stream:
    # stop_stream = c2.button('Stop Injection')
    if stop_stream:
        stream = False

    for i in range(1000):

        with pl.container():

            image = Image.open('assets/Images/' + str(i % N) + '_rect.png')
            data = np.asarray(image)
            # covMat = np.array(covMat, dtype=float)
            fig = px.imshow(data).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            st.write(fig)

            defect = defect_list[str(i % N)]
            print(i, defect)

            if defect == 'no-defect':
                st.success('no defect!')
                is_error = False
            else:
                st.error('found a defect - '+defect)
                image = Image.open('assets/Images/' + str(i % N) + '_zoom_' + defect + '.png')
                data = np.asarray(image)
                err_fig = px.imshow(data).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                # cls = get_class()
                is_error = True
                q = pd.DataFrame({
                    'section': [i % N],
                    'defect': [defect],
                })
                alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
            # with st.expander('defect list'):
            #     sbc = st.empty()
            #     with sbc.container():
            #         st.write(alerts)
        if is_error:
            with st.expander(defect+'  ::  defect-alert zoom-in @ section' + str(i % N)):
                st.write(err_fig)
