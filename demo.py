import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components
import webbrowser
import time
import plotly.express as px
from sklearn.decomposition import PCA

vanti_app_url = 'https://app.vanti.ai'
h2o_app_url = 'https://cloud.h2o.ai/apps/6ab8bf64-9bc5-4a68-9a7e-7251909c8d47'

# st.set_page_config(page_title='Vanti-Dynamic-Model-Demo')
page_title = "Adaptive AI DEMO App"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"

st.set_page_config(page_title=page_title, page_icon=page_icon)  # , layout=layout)
# st.set_page_config(page_title='sdfsdf')
run_en = False

# st.image('assets/Images/Vanti - Main Logo@4x copy.png', width=200)
st.title('Dynamic Model Playground')

color_scale = alt.Scale(range=['#FAFA37', '#52de97', '#c9c9c9'])
BASE_PERF = [0.88, 0.89]
GAMMA = BASE_PERF[0] - 0.25
BETA = 1 - GAMMA / (BASE_PERF[0])
# gamma = ()
VS = 0.01


def highlight_survived(s):
    return ['color: #52de97'] * len(s) if s['Ground Truth'] == s['Vanti'] else ['background-color: #008181A'] * len(s)


def color_survived(val):
    color = '#52de97' if val else 'red'
    return f'background-color: {color}'


def my_pcs(df, n_comp):
    pca = PCA(n_components=n_comp, random_state=22)
    pca.fit(df)
    x = pca.transform(df)
    x = pd.DataFrame(x)
    x.index = df.index
    col_names = ['component_' + str(i) for i in range(x.shape[1])]
    x.columns = col_names
    return x


def files():
    with st.expander('files'):
        # st.header("files")
        uploaded_file_int = st.file_uploader("upload 'good' file", accept_multiple_files=False)
        dontcare_int = st.file_uploader("upload 'drift' file", accept_multiple_files=False)
        return uploaded_file_int, dontcare_int


def models():
    st.header("models")
    # ap1, ap2 = st.beta_columns(2)

    with st.expander('build models'):
        cc1, cc2 = st.columns((2, 2))
        if cc1.button('app.vanti'):
            webbrowser.open_new_tab(vanti_app_url)
        if cc2.button('app.h2o'):
            webbrowser.open_new_tab(h2o_app_url)

    dc1 = st.text_input('Vanti Model id', "####-production")
    b1 = st.button('connect to Vanti')
    if b1:
        for i in range(90000):
            a = 1
        st.success('connected')
    dc2 = st.file_uploader('H20-mojo', accept_multiple_files=False)
    z1, z2 = st.columns(2)
    dc1r = z1.radio('Vanti error handling', ['flip coin', "auto"])
    dc2r = z2.radio('H20 error handling', ['flip coin', 'auto'])
    with st.expander('what does error handling mean?'):
        st.write('traditional models throw an error when structrual drifts occure. ')
        st.write('flip coin - instead of an error the model will return a random pass/fail')
        st.write('auto - the model returns is default answer of error / respons')


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
    pl = st.empty()
    pl2 = st.empty()
    pca_plot = st.empty()

    recovery = False
    drop = False
    if up_file is not None:

        tar, tar_concat, df, df_concat, dc, kpi, thr1, thr2, b, dic, inv_dic = parse_files(up_file, dc_file)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        new_df = df.select_dtypes(include=numerics)
        df_pca = my_pcs(new_df, 2)
        df_pca = (df_pca - df_pca.mean()) / df_pca.std()

        with st.expander('Data snippet'):

            diff = get_cols_diff(df, dc)
            diff_length = len(diff)
            if diff_length > 0:
                q1, q2 = st.columns(2)
                q1.write(np.str(diff_length) + ' missing features detected')
                q2.write(diff)

            col1, col2 = st.columns(2)
            col1.write('good file has ' + np.str(thr1) + ' rows and ' + np.str(b[0]) + ' features')
            col2.write('drift file has ' + np.str(thr2) + ' rows and ' + np.str(b[1]) + ' features')
            col1.dataframe(df)
            col2.dataframe(dc)

        st.title('performance over time')
        pl = st.empty()
        pl2 = st.empty()

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
        feed1, feed2 = st.columns([1, 4])
        vanti_prev = 0
        h20_prev = 0
        for i in range(1, thr1 + thr2):
            # time.sleep(0.05)



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

            v_score_w = v_score_w * alpha + v_score * (1 - alpha)
            h_score_w = h_score_w * alpha + h_score * (1 - alpha)
            new_predictions = pd.DataFrame({'Vanti': [v_score_w],
                                            'H20': [h_score_w],
                                            'error': [error_val],
                                            'x': [i]},
                                           index=[i])
            predictions = pd.concat([predictions, new_predictions], axis=0)

            h20_val = np.round(h_score_w, 2)
            vanti_val = np.round(predictions['Vanti'].iloc[i], 2)
            with pl.container():
                m1, m2 = st.columns([2, 1])
                fig = px.line(predictions[['Vanti', 'H20']], markers=True)
                fig.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
                fig['data'][0]['line']['color'] = "#52de97"
                # fig['data'][1]['line']['color'] = "#FAFA37"
                fig['data'][1]['line']['color'] = "#ff3c78"
                fig.update_xaxes(range=[1, 300])
                fig.update_yaxes(range=[0.45, 1])

                m1.write(fig)

            vanti_delta = np.round(vanti_val - vanti_prev, 2) * 100
            h20_delta = np.round(h20_val - h20_prev, 2) * 100
            m2.metric(label="Vanti Accuracy", value=vanti_val * 100, delta=vanti_delta)
            m2.metric(label="H20 Accuracy", value=h20_val * 100, delta=h20_delta)
            h20_prev = h20_val
            vanti_prev = vanti_val
            with pl2.container():
                n1 = df_pca.columns[0]
                n2 = df_pca.columns[1]

                data = df_pca.iloc[:i].copy()

                data['pred'] = 'old'
                data['pred'].iloc[-1] = 'new'
                sct = px.scatter(data, x=n1, y=n2, color='pred')
                sct.update_layout(plot_bgcolor='#ffffff')
                # sct.update_xaxes(range=[0, 13])
                # sct.update_yaxes(range=[-13, 0])

                # st.write(sct)

                if i == 1:
                    feed1.success('@index :: ' + str(i))
                    feed2.success('all is good!')
                if not drop and h20_val < 0.7 and i > 100:
                    feed1.success('@index :: ' + str(i))
                    feed1.error('alert')
                    feed1.info('notice')
                    feed2.error('drift detected! - 3 missing columns')
                    feed2.error('H20 accuracy -->  50%')
                    feed2.info('Vanti: analyzing affected nodes')
                    # st.info('This is a purely informational message', icon="ℹ️")
                    drop = True
                    recovery = True

                if vanti_val > 0.7 and recovery and np.random.rand() < 0.1:
                    node = np.random.randint(0, 10, 1)
                    layer = np.random.randint(0, 10, 1)
                    feed1.success('@index :: ' + str(i))
                    feed1.info('notice')
                    feed1.info('notice')
                    feed2.success('updating Vanti')
                    feed2.info('replacing node ' + str(node) + ' layer ' + str(layer) + ' with new node')
                    feed2.info('Vanti: accuracy restored to ' + str(np.round(v_acc * 100)) + '%')


def get_cols_diff(up_file, dc_file):
    cols1 = up_file.columns
    diff = []
    temp = [diff.append(i) for i in cols1 if i not in dc_file.columns]
    # [st.write(i) for i in cols1 if i not in dc_file.columns]

    return diff


with st.sidebar:
    st.image('assets/Images/Vanti - Main Logo@4x copy.png')

    models()
    uploaded_file, dont_care = files()

st.subheader('Run Experiment')
if st.button('Run!'):
    run_exp(uploaded_file, dont_care)
else:
    a = 1

# st.text(" ")
st.markdown("""---""")
with st.expander('what is drift?'):
    st.image("assets/Images/drift sketch black copy.png")
    st.markdown(
        '> Data drift is unexpected and undocumented changes to data structure, semantics, and infrastructure '
        'that is '
        'a result of modern data architectures. Data drift breaks processes and corrupts data, but can also reveal '
        'new opportunities for data use.')

# st.title("Useful")
with st.expander('6 easy steps'):
    st.title('6 easy steps')
    st.image('6 easy step copy.png')

    # st.markdown("""---""")

with st.expander('reach out to our CTO'):
    st.title('Reach out!')
    st.text("sub: [AI-CONF] →")
    st.text("niro@vanti.ai")
    st.markdown("---")
    # st.write('niro@vanti-analytics.com')
    st.text('vanti.ai')
    st.text('app.vanti.ai')
    st.markdown("""---""")

with st.expander('How does adaptive AI work?'):
    st.title('Self Wiring Networks')
    st.image('assets/Images/ezgif.com-gif-maker (3).gif')
    st.image('assets/Images/ezgif.com-gif-maker (2).gif')

with st.expander('Visit Vanti.AI'):
    components.iframe('http://vanti.ai', height=900)

# hide_menu_style = """
#         <style>
#         MainMenu {visibility: hidden; }
#         footer {visibility: hidden;}
#         primary: #52DE97;
#
#         </style>
#
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style', unsafe_allow_html=True)

# def get_color_styles(color: str) -> str:
#     """Compile some hacky CSS to override the theme color."""
#     # fmt: off
#     color_selectors = ["a", "a:hover", "*:not(textarea).st-ex:hover", ".st-en:hover"]
#     bg_selectors = [".st-da", "*:not(button).st-en:hover"]
#     border_selectors = [".st-ft", ".st-fs", ".st-fr", ".st-fq", ".st-ex:hover", ".st-en:hover"]
#     # fmt: on
#     css_root = "#root { --primary: %s }" % color
#     css_color = ", ".join(color_selectors) + "{ color: %s !important }" % color
#     css_bg = ", ".join(bg_selectors) + "{ background-color: %s !important }" % color
#     css_border = ", ".join(border_selectors) + "{ border-color: %s !important }" % color
#     other = ".decoration { background: %s !important }" % color
#     return f"<style>{css_root}{css_color}{css_bg}{css_border}{other}</style>"
#
#
# st.write(get_color_styles("#00818A"), unsafe_allow_html=True)