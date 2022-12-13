import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

page_title = "Medical Device Application"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "wide"

raw = pd.read_csv('assets/Data/medical-data.csv')
raw = raw.sample(frac=1).reset_index(drop=True)
kpi = 'S_Scrap'
KPI = raw[kpi].copy()
df = raw.copy()
df.drop(columns=[kpi], inplace=True)

# --------------------------------------
window = 30


# --------------------------------------
def files():
    with st.expander('files'):
        uploaded_file = st.file_uploader("upload data files", accept_multiple_files=True)
        return uploaded_file


def get_reason():
    F = np.random.randint(1, 4, 1)[0]
    n = df.shape[1]
    cols = df.columns.to_list()
    phrases = []
    for j in range(F):
        feat = cols[np.random.randint(0, n-1, 1)[0]]
        s = " > " if np.random.rand() < 0.5 else " < "
        m = np.round(ms[feat] + np.random.randn(), 2)
        d = np.round(ss[feat], 2)
        if s == " > ":
            v = m+d
        else:
            v = m-d
        v = str(np.round(v, 2))
        phrases.append(feat+s+v)
    phrase = ' and '.join(phrases)

    return phrase


# --------------------------------------
st.set_page_config(page_title=page_title, page_icon=page_icon, layout='wide')  # , layout=layout)
st.title(page_title)
st.subheader('Early Fault Prediction')

# --------------------------------------
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
    with st.expander("what is model sensitivity?"):
        st.write("_sensitivity 100 --> find **everything**_ I'm ok with some false alarms")
        st.write("_sensitivity 0 --> find  **critical things only**_ with high certainty")
    # speed = st.slider('select path size', 16, 64, 32)
    files = files()

# --------------------------------------
clist = ['All Measurements']

for i in df.columns.to_list():
    clist.append(i)
feats = st.multiselect("Select Data", clist)

st.text("You selected: {}".format(", ".join(feats)))
if 'All Measurements' in feats:
    feats = df.columns.to_list()

ms = {i: df[i].mean() for i in feats}
ss = {i: df[i].std() for i in feats}

data_graph = st.empty()
error_inv = st.empty()
metrics = st.empty()
if stream:
    if stop_stream:
        stream = False

    feed1, feed2 = st.columns([1, 4])
    fail_counter = 0
    i = 0

    for i in range(df.shape[0]):
        if KPI[i] == 1:
            fail_counter = fail_counter+1
        s = 0
        e = min(i, df.shape[0])
        temp = df[feats].iloc[s:e]

        with data_graph.container():
            sss = max(0, i - 10)
            eee = min(i, df.shape[0])
            temp2 = df.iloc[sss:eee]

            fig3 = px.line(temp, markers=True)
            fig3.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
            # hide and lock down axes
            fig3.update_xaxes(visible=True, fixedrange=True)
            fig3.update_yaxes(visible=True, fixedrange=True)
            # remove facet/subplot labels
            fig3.update_layout(annotations=[], overwrite=True)
            st.write(fig3)
            with metrics.container():
                st.metric(label="Predictions", value=i)
                st.metric(label="Fails", value=fail_counter)
                st.metric(label="Ratio", value=str(np.round(fail_counter / (i + 1) * 100, 1)) + "%")
        with error_inv.container():
            if KPI[i] == 1:
                feed1.error('FAIL @SN = ' + str(df.index[i]))
                # feed2.error('@index :: ' + str(i))
                feed2.info(get_reason())

# --------------------------------------
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
