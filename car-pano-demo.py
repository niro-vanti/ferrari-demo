import altair as alt
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
import os
import pandas as pd


# functions
def files():
    with st.expander('files'):
        # st.header("files")
        uploaded_file_int = st.file_uploader("upload car image", accept_multiple_files=False)
        # dontcare_int = st.file_uploader("upload 'drift' file", accept_multiple_files=False)
        return uploaded_file_int




# page
page_title = "Paint shop Visual Defect Detection App"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"

st.set_page_config(page_title=page_title, page_icon=page_icon)  # , layout=layout)
color_scale = alt.Scale(range=['#FAFA37', '#52de97', '#c9c9c9'])



N = 146
zoom_names = []
defect_list = {}

alerts = pd.DataFrame()


for file in os.listdir('assets/Images'):
    if '_zoom' in file:
        it = file.split('_')[0]
        dif = file.split('_')[-1].split('.')[0]
        defect_list[it] = dif




with st.sidebar:
    st.image('assets/Images/Vanti - Main Logo@4x copy.png')
    # sbc1, sbc2 = st.columns((2,1))
    token = st.text_input('Vanti Model id', "####-production")
    connect = st.button('connect')
    if connect:
        for i in range(10000000):
            a = 1
        st.success('connected to model')
    sbc1, sbc2 = st.columns(2)
    stream = sbc1.button('Start Injection')
    stop_stream = sbc2.button('Stop Injection')
    sensitivity = st.slider('model sensitivity', 0, 100, 50)
    speed = st.slider('select path size',16, 64, 32)
    files()



st.title(page_title)
st.text(' ')
st.image('assets/Images/ferrari-cropped.png')

c1, c2 = st.columns(2)

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
            st.image('assets/Images/' + str(i % N) + '_rect.png')

            defect = defect_list[str(i % N)]
            print(i, defect)

            if defect == 'no-defect':
                st.success('no defect!')
                is_error = False
            else:
                is_error = True
                q = pd.DataFrame({
                    'section': [i % N],
                    'defect': [defect],
                })
                alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
        if is_error:
            with st.expander(defect+'  ::  defect-alert zoom-in @ section' + str(i % N)):
                st.image('assets/Images/' + str(i % N) + '_zoom_' + defect + '.png')

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            root {
                primary="#52de97";
                backgroundColor = "#FFFFFF";
                textColor = "#000000";
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

