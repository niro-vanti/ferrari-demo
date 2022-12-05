import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
import altair as alt
from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components
import webbrowser

vanti_app_url = 'https://app.vanti-analytics.com'
h2o_app_url = 'https://cloud.h2o.ai/apps/6ab8bf64-9bc5-4a68-9a7e-7251909c8d47'

st.set_page_config(page_title='Vanti-Dynamic-Model-Demo')

vanti_banner = Image.open('assets/Images/Vanti - Main Logo@4x copy.png')
# vanti_banner = Image.open('assets/Images/Vanti - Logo White L Green Icon & Dark Blue B@4x.png')
run_en = False

st.image(vanti_banner)
st.title('Dynamic Model Playground')

color_scale = alt.Scale(range=['#FAFA37', '#52de97', '#c9c9c9'])
BASE_PERF = [0.88, 0.89]
GAMMA = BASE_PERF[0] - 0.25
BETA = 1 - GAMMA / (BASE_PERF[0])
# gamma = ()
VS = 0.01


def highlight_survived(s):
    return ['background-color: #52de97'] * len(s) if s['Ground Truth'] == s['Vanti'] else ['background-color: #008181A'] * len(s)


def color_survived(val):
    color = '#52de97' if val else 'red'
    return f'background-color: {color}'


def files():
    st.header("files")
    t1, t2, t3 = st.beta_columns((2, 1, 1))
    uploaded_file_int = t1.file_uploader("upload 'good' file", accept_multiple_files=False)
    dontcare_int = t1.file_uploader("upload 'drift' file", accept_multiple_files=False)
    # q1, q2 = st.beta_columns(2)
    drift_image_en = t3.select_slider("What is Drift?", ["I already know", "Show me"])
    w1, w2 = st.beta_columns((1, 1))
    drift_image = w2.empty()
    exp = w1.empty()
    if drift_image_en is "Show me":
        # drift_image.image("assets/Images/0_Pd95BnSdr0A7Ujqn.png")
        drift_image.image("assets/Images/drift sketch black copy.png")
        exp.markdown(
            '> Data drift is unexpected and undocumented changes to data structure, semantics, and infrastructure '
            'that is '
            'a result of modern data architectures. Data drift breaks processes and corrupts data, but can also reveal '
            'new opportunities for data use.')

    else:
        drift_image.empty()
        exp.empty()
    return uploaded_file_int, dontcare_int


def models():
    st.header("models")
    ap1, ap2 = st.beta_columns(2)

    cc1, cc2 = st.beta_columns((2, 2))
    if cc1.button('app.vanti'):
        webbrowser.open_new_tab(vanti_app_url)
    if cc2.button('app.h2o'):
        webbrowser.open_new_tab(h2o_app_url)

    dc1 = cc1.text_input('Vanti Model id', "####-production")
    cc1.write(' ')
    cc1.write(' ')
    cc1.write(' ')
    dc1r = cc1.select_slider('Vanti error handling', ['flip coin', "auto"])

    cc11, qwer, cc21 = st.beta_columns((2, 1, 1))
    dc2 = cc2.file_uploader('H20-mojo', accept_multiple_files=False)
    dc2r = cc2.select_slider('H20 error handling', ['flip coin', 'auto'])
    # qwe.write(' ')
    qwer.write(' ')


def get_pred(base_perf, beta, vs, thr1, gt, i):
    if i >= thr1:
        h_score = 0.5 + np.random.randn() * vs / 4
        exp_factor = (1 - beta * np.exp(-0.01 * (i - thr1)))
        noise_factor = np.random.randn() * vs
        sine_factor = np.sin((i - thr1) * 0.025) * 0.025 * 0
        v_score = base_perf[0] * exp_factor + sine_factor + noise_factor
    else:
        h_score = base_perf[1] + np.random.randn() * vs
        v_score = base_perf[0] + np.random.randn() * vs

    v_flip = np.random.rand()
    h_flip = np.random.rand()

    v_pred = gt if v_flip <= v_score else 1 - gt
    h_pred = gt if h_flip <= h_score else 1 - gt

    return h_score, v_score, v_pred, h_pred


def parse_files(up_file, dc_file):
    df = pd.read_csv(up_file)
    dc = pd.read_csv(dc_file)

    df_concat = pd.concat([df, dc], axis=0)

    thr1 = df.shape[0]
    thr2 = dc.shape[0]
    b1 = df.shape[1]
    b2 = dc.shape[1]
    b = [b1, b2]

    df_concat.index = [i for i in range(thr1 + thr2)]

    kpi = df.columns[-1]
    dic = {'Passed': 1, 'Failed': 0}
    inv_dic = {v: k for k, v in dic.items()}

    tar = df[kpi].copy()
    tar_concat = df_concat[kpi].copy()
    tar.replace(dic, inplace=True)
    tar_concat.replace(dic, inplace=True)

    return tar, tar_concat, df, df_concat, dc, kpi, thr1, thr2, b, dic, inv_dic


def calc_perf(df, name, window=50):
    if df.shape[0] < window:
        gt = df['Ground Truth']
        pr = df[name]
        acc = accuracy_score(gt, pr)
    else:
        gt = df['Ground Truth'].iloc[-window:]
        pr = df[name].iloc[-window:]
        acc = accuracy_score(gt, pr)
    return acc


def run_exp(up_file, dc_file):
    if up_file is not None:

        tar, tar_concat, df, df_concat, dc, kpi, thr1, thr2, b, dic, inv_dic = parse_files(up_file, dc_file)

        st.title('Data snippet')

        diff = get_cols_diff(df, dc)
        # st.write(df.shape)
        # st.write(dc.shape)
        L = len(diff)
        if L > 0:
            q1, q2 = st.beta_columns(2)
            q1.write(np.str(L) + ' missing features detected')
            q2.write(diff)
        # st.write(diff)

        col1, col2 = st.beta_columns(2)
        col1.write('good file has ' + np.str(thr1) + ' rows and ' + np.str(b[0]) + ' features')
        col2.write('drift file has ' + np.str(thr2) + ' rows and ' + np.str(b[1]) + ' features')
        col1.dataframe(df)
        col2.dataframe(dc)

        st.title('performance over time')
        pl = st.empty()
        p2 = st.empty()

        predictions = pd.DataFrame({'Vanti': [BASE_PERF[0]],
                                    'H20': [BASE_PERF[1]],
                                    'error': [0],
                                    'Ground Truth': tar[0],
                                    'x': 0})

        df_predictions = pd.DataFrame({'Ground Truth': [inv_dic[tar_concat[0]]],
                                       'Vanti': [inv_dic[tar_concat[0]]],
                                       'H20': [inv_dic[tar_concat[0]]],
                                       'Vanti ACC': [1],
                                       'H20 ACC': [1]
                                       })

        v_score_w = 1
        h_score_w = 1
        alpha = 0.3
        for i in range(1, thr1 + thr2):
            # temp = pd.read_csv('assets/Data/early_fault_detection copy.csv')  # just to take up some time

            c = alt.Chart(predictions).transform_fold(
                ['Vanti', 'H20']
            ).mark_line(point=True).encode(
                x='x',
                y='value:Q',
                color=alt.Color('key:N', scale=color_scale)
            )

            pl.altair_chart(c, use_container_width=True)

            error_val = -1 if i >= thr1 else 0

            h_score, v_score, v_pred, h_pred = get_pred(BASE_PERF, BETA, VS, thr1, tar_concat[i], i)

            v_acc = calc_perf(df_predictions, 'Vanti')
            h_acc = calc_perf(df_predictions, 'H20')

            new_p = pd.DataFrame({
                'Ground Truth': [inv_dic[tar_concat[i]]],
                'Vanti': [inv_dic[v_pred]],
                'H20': [inv_dic[h_pred]],
                'Vanti ACC': [v_acc],
                'H20 ACC': [h_acc]
            })
            new_p.index = [i]

            df_predictions = pd.concat([df_predictions, new_p], axis=0)

            # p2.dataframe(df_predictions)
            p2.dataframe(df_predictions.style.apply(highlight_survived, axis=1))

            v_score_w = v_score_w * alpha + v_score * (1 - alpha)
            h_score_w = h_score_w * alpha + h_score * (1 - alpha)
            new_predictions = pd.DataFrame({'Vanti': [v_score_w],
                                            'H20': [h_score_w],
                                            'error': [error_val],
                                            'x': [i]},
                                           index=[i])
            predictions = pd.concat([predictions, new_predictions], axis=0)


def get_cols_diff(up_file, dc_file):
    cols1 = up_file.columns
    diff = []
    temp = [diff.append(i) for i in cols1 if i not in dc_file.columns]
    # [st.write(i) for i in cols1 if i not in dc_file.columns]

    return diff


# st.markdown('---')

# st.markdown('---')

uploaded_file, dont_care = files()
st.markdown("""---""")
models()
st.markdown("""---""")
st.title('RUN')
if st.button('Run!'):
    run_exp(uploaded_file, dont_care)
else:
    # st.write('sdf')
    a = 1

st.text(" ")
st.markdown("""---""")

# st.title("Useful")

us1, us2 = st.beta_columns(2)

us1.title('6 easy steps')
us1.image('assets/Images/6 easy step copy.png')

# st.markdown("""---""")
us2.title('Reach out!')
us2.header("sub: [AI-CONF] →")
us2.header("niro@vanti-analytics.com")
us2.markdown("---")
# st.write('niro@vanti-analytics.com')
us2.header('vanti-analytics.com')
us2.header('app.vanti-analytics.com')
st.markdown("""---""")

st.title('Self Wiring Networks')
st.image('assets/Images/ezgif.com-gif-maker (3).gif')
st.image('assets/Images/ezgif.com-gif-maker (2).gif')

components.iframe('http://vanti-analytics.com', height=900)

hide_menu_style = """
        <style>
        MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
