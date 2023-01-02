# import numpy as np
# import streamlit as st
# import pandas as pd
# from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components
# import webbrowser
import time
import plotly.express as px
# from sklearn.decomposition import PCA
import os
import toml
from auxFunctions import *

# constants
vanti_app_url = 'https://app.vanti.ai'
h2o_app_url = 'https://cloud.h2o.ai/apps/6ab8bf64-9bc5-4a68-9a7e-7251909c8d47'

page_title = "Vanti Apps"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/

window = 30
BASE_PERF = [0.88, 0.89]
GAMMA = BASE_PERF[0] - 0.25
BETA = 1 - GAMMA / (BASE_PERF[0])
# gamma = ()
VS = 0.01
stream = False

nodes = ['anomaly remover', 'formatter', 'mini decision tree', 'regressor', 'classifier', 'SVM', 'perceptron',
         'nan filler', 'normalizer', 'encoder', 'balancer']

st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

primaryColor = toml.load(".streamlit/config.toml")['theme']['primaryColor']
s = f"""
<style>
div.stButton > button:first-child {{ border: 2px solid {primaryColor}; border-radius:10px 10px 10px 10px; }}
div.stButton > button:hover {{ background-color: {primaryColor}; color:#000000;}}
footer {{ visibility: hidden;}}
# header {{ visibility: hidden;}}
<style>
"""
st.markdown(s, unsafe_allow_html=True)


# files

# functions
def run_exp(up_file, dc_file, base_perf, beta, vs):
    # pl = st.empty()
    # pl2 = st.empty()
    # pca_plot = st.empty()

    recovery = False
    drop = False
    if up_file is not None:

        tar, tar_concat, df, df_concat, dc, kpi, thr1, thr2, b, dic, inv_dic = parse_files(up_file, dc_file)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        new_df = df.select_dtypes(include=numerics)
        df_pca = my_pcs(new_df, 2)
        # df_pca = (df_pca - df_pca.mean()) / df_pca.std()

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

        predictions = pd.DataFrame({'Vanti': [base_perf[0]],
                                    'Standard Model': [base_perf[1]],
                                    'error': [0],
                                    'Ground Truth': tar[0],
                                    'x': 0})

        df_predictions = pd.DataFrame({'Ground Truth': [inv_dic[tar_concat[0]]],
                                       'Vanti': [inv_dic[tar_concat[0]]],
                                       'Standard Model': [inv_dic[tar_concat[0]]],
                                       'Vanti ACC': [1],
                                       'Standard Model ACC': [1]
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

            h_score, v_score, v_pred, h_pred = get_pred(base_perf, beta, vs, thr1, tar_concat[i], i)

            v_acc = calc_perf(df_predictions, 'Vanti')
            h_acc = calc_perf(df_predictions, 'Standard Model')

            new_p = pd.DataFrame({
                'Ground Truth': [inv_dic[tar_concat[i]]],
                'Vanti': [inv_dic[v_pred]],
                'Standard Model': [inv_dic[h_pred]],
                'Vanti ACC': [v_acc],
                'Standard Model ACC': [h_acc]
            })
            new_p.index = [i]

            df_predictions = pd.concat([df_predictions, new_p], axis=0)

            v_score_w = v_score_w * alpha + v_score * (1 - alpha)
            h_score_w = h_score_w * alpha + h_score * (1 - alpha)
            new_predictions = pd.DataFrame({'Vanti': [v_score_w],
                                            'Standard Model': [h_score_w],
                                            'error': [error_val],
                                            'x': [i]},
                                           index=[i])
            predictions = pd.concat([predictions, new_predictions], axis=0)

            h20_val = np.round(h_score_w, 2)
            vanti_val = np.round(predictions['Vanti'].iloc[i], 2)
            with pl.container():
                m1, m2 = st.columns([2, 1])
                fig = px.line(predictions[['Vanti', 'Standard Model']], markers=True)
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
            m2.metric(label="Standard Model Accuracy", value=h20_val * 100, delta=h20_delta)
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
                    feed2.error('standard model accuracy -->  50%')
                    feed2.info('Vanti: analyzing affected nodes')
                    # st.info('This is a purely informational message', icon="ℹ️")
                    drop = True
                    recovery = True

                if vanti_val > 0.7 and recovery and np.random.rand() < 0.1:
                    node = np.random.randint(0, 10, 1)[0]
                    node = nodes[node]
                    new_node = np.random.randint(0, 10, 1)[0]
                    new_node = nodes[new_node]
                    layer = np.random.randint(0, 10, 1)
                    feed1.success('@index :: ' + str(i))
                    feed1.info('notice')
                    feed1.info('notice')
                    feed2.success('updating Vanti')
                    feed2.info('replacing node ' + str(node) + ' in layer ' + str(layer) + ' with ' + str(new_node))
                    feed2.info('Vanti: accuracy restored to ' + str(np.round(vanti_val * 100)) + '%')


# app functions
def paint_shop_app(stream):
    st.title('In-line Paint Shop Defect Detection')
    st.image('assets/Images/ferrari-cropped.png')
    sbc1, sbc2 = st.columns(2)
    sensitivity = sbc1.slider('model sensitivity', 0, 100, 50)
    speed = sbc1.slider('select path size', 16, 64, 32)
    with sbc2.expander('What is model sensitivity?'):
        st.write("_sensitivity 100 --> alert me on **everything**_")
        st.write("_sensitivity 0 --> alert me on **critical things only**_")
    with sbc2.expander('What is Patch Size?'):
        st.write(
            'Patch size is the # of pixels in each dimension that the images is broken down to for defect detections')
        st.write('a smaller patch will find smaller defects, but will take longer to run')
        st.write('a bigger patch will find bigger defects, but will take faster to run')

    # image_number = 146
    # zoom_names = []
    defect_list = {}

    alerts = pd.DataFrame()

    for file in os.listdir('assets/Images'):
        if '_zoom' in file:
            it = file.split('_')[0]
            dif = file.split('_')[-1].split('.')[0]
            defect_list[it] = dif

    c1, c2 = st.columns(2)

    pl = st.empty()
    p2 = st.empty()
    image_number = 146
    cls = ''
    is_error = False

    if stream:

        for i in range(1000):
            if stop_stream:
                stream = False
                break

            with pl.container():
                st.image('assets/Images/' + str(i % image_number) + '_rect.png')

                defect = defect_list[str(i % image_number)]
                print(i, defect)

                if defect == 'no-defect':
                    st.success('no defect!')
                    is_error = False
                else:
                    is_error = True
                    q = pd.DataFrame({
                        'section': [i % image_number],
                        'defect': [defect],
                    })
                    alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
            if is_error:
                with st.expander(defect + '  ::  defect-alert zoom-in @ section' + str(i % image_number)):
                    st.image('assets/Images/' + str(i % image_number) + '_zoom_' + defect + '.png')


def rt_sensors_app(stream):
    st.title('Real Time Anomaly Detection')
    df = files[0]
    if 'prog' in df.columns:
        df.drop(columns=['prog'], inplace=True)
        df = (df - df.mean()) / df.std()
    df['sen_alert'] = 0
    df['sit_alert'] = 0

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
        st.write('a **situation** anomaly is when all sensors are tracked together by Vanti''s model and the model '
                 'decides to alert the user')

    a = 1

    if mode == 'alert me only when there''s a situation anomaly':
        MODE = 0
    elif mode == 'alert me only when there''s a sensor anomaly':
        MODE = 1
    elif mode == 'I want all alerts':
        MODE = 2
    else:
        MODE = 3
    print(MODE)

    sensitivity = c1.slider('alert sensitivity', 0.0, 100.0, 50.0)
    with c2.expander("what is model sensitivity?"):
        st.write("_sensitivity 100 --> alert me on **everything**_")
        st.write("_sensitivity 0 --> alert me on **critical things only**_")

    ms = {i: df[i].mean() for i in feats}
    ss = {i: df[i].std() for i in feats}

    c1, c2, c3 = st.columns(3)
    sensitivity = (100 - sensitivity) / 10

    temp = df[feats].iloc[:2].copy()

    window = 300
    pl = st.empty()
    pl2 = st.empty()
    alerts = pd.DataFrame()

    if stream:
        for i in range(df.shape[0]):
            if stop_stream:
                stream = False
                break

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
                            eee = min(i + 5, df.shape[0])
                            temp2 = df[f].iloc[sss:eee]
                            fig3 = px.line(temp2)
                            fig3.update_layout(plot_bgcolor='#ffffff')
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
                        eee = min(i + 5, df.shape[0])
                        temp2 = df[feats].iloc[sss:eee]
                        fig3 = px.line(temp2)
                        fig3.update_layout(plot_bgcolor='#ffffff')
                    with st.expander('situation-alert zoom-in @' + str(df.index[i])):
                        st.write(fig3, title=str(df.index[i]))

            with pl.container():
                # st.text(str(np.round(i / df.shape[0] * 100, 2)) + ' %')
                fig = px.line(data_frame=temp)
                fig.update_layout(plot_bgcolor='#ffffff')
                st.write(fig)
                # time.sleep(0.1)
                st.dataframe(alerts.style.apply(highlight_survived, axis=1))
                fig2 = px.line(temp[['sen_alert', 'sit_alert']].cumsum())
                fig2.update_layout(plot_bgcolor='#ffffff')
                st.write(fig2)

    with st.expander('see training data'):
        st.dataframe(df)


def medical_device_app(stream):
    st.title('Medical Device Early Fault Detection')

    df = files[0]
    KPI = files[1]
    # --------------------------------------
    clist = ['All Measurements']

    for i in df.columns.to_list():
        clist.append(i)
    feats = st.multiselect("Select Data", clist)

    st.text("You selected: {}".format(", ".join(feats)))
    if 'All Measurements' in feats:
        feats = df.columns.to_list()

    c1, c2 = st.columns(2)
    mode = c1.radio('select alert mode',
                    ['alert me only when there''s a situation anomaly',
                     'alert me only when there''s a sensor anomaly',
                     'I want all alerts'])
    with c2.expander('what are these alerts?'):
        st.write('a **sensor** anomaly is when a single sensor is tracked by Vanti''s model and the model decides to '
                 'alert the user')
        st.write('a **situation** anomaly is when all sensors are tracked together by Vanti''s model and the model '
                 'decides to alert the user')

    a = 1

    if mode == 'alert me only when there''s a situation anomaly':
        selected_mode = 0
    elif mode == 'alert me only when there''s a sensor anomaly':
        selected_mode = 1
    elif mode == 'I want all alerts':
        selected_mode = 2
    else:
        selected_mode = 3
    print(selected_mode)

    sensitivity = c1.slider('alert sensitivity', 0.0, 100.0, 50.0)
    with c2.expander("what is model sensitivity?"):
        st.write("_sensitivity 100 --> alert me on **everything**_")
        st.write("_sensitivity 0 --> alert me on **critical things only**_")

    sbc1, sbc2 = st.columns(2)

    ms = {i: df[i].mean() for i in feats}
    ss = {i: df[i].std() for i in feats}

    data_graph = st.empty()
    error_inv = st.empty()
    metrics = st.empty()
    if stream:

        feed1, feed2 = st.columns([1, 4])
        fail_counter = 0
        i = 0

        for i in range(df.shape[0]):
            if stop_stream:
                stream = False
                break
            if KPI[i] == 1:
                fail_counter = fail_counter + 1
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
                    feed2.info(get_reason_medical(df, ms, ss))


def adaptive_ai_demo(stream):
    st.title('Adaptive AI Demo')
    st.header('Run Experiment')
    models(vanti_app_url, h2o_app_url)
    uploaded_file = files[0]
    dont_care = files[1]
    if st.button('Run!'):
        run_exp(uploaded_file, dont_care, BASE_PERF, BETA, VS)
    else:
        a = 1

    st.markdown("""---""")
    with st.expander('what is drift?'):
        st.image("assets/Images/drift sketch black copy.png")
        st.markdown(
            '> Data drift is unexpected and undocumented changes to data structure, semantics, and infrastructure '
            'that is '
            'a result of modern data architectures. Data drift breaks processes and corrupts data, but can also reveal '
            'new opportunities for data use.')

    with st.expander('6 easy steps'):
        st.title('6 easy steps')
        st.image('6 easy step copy.png')

    with st.expander('reach out to our CTO'):
        ro1, ro2 = st.columns(2)
        st.title('Reach out!')
        ro1.write("sub: [ADAPTIVE-AI DEMO] →")
        ro2.write("niro@vanti.ai")
        ro1.write('vanti.ai')
        ro2.write('app.vanti.ai')

    with st.expander('How does adaptive AI work?'):
        st.title('Self Wiring Networks')
        st.image('assets/Images/ezgif.com-gif-maker (3).gif')
        st.image('assets/Images/ezgif.com-gif-maker (2).gif')

    with st.expander('Visit Vanti.AI'):
        components.iframe('http://vanti.ai', height=900)


def video_assembly_app(stream):
    st.title('Defect Detection in Video Assembly')
    st.write(' ')
    col1, col2 = st.columns((1, 4))

    with col1:
        st.write(' ')

    with col2:
        st.image('assets/Data/assembly-movie-small.gif', caption='assemly video')

    # with col3:
    #     st.write(' ')

    df = files[0]
    KPI = files[1]
    N = df.shape[0]
    metrics = col1.empty()
    error_inv = st.empty()
    st.subheader('train vs real time root cause distribution')
    graph_inv = st.empty()
    st.subheader('Root Cause per Unit')

    v = df['reason'].value_counts(normalize=True) * 100
    v = v.reset_index(level=0)
    v.columns = ['reason', 'train_count']
    v['predict_count'] = 0
    # st.write(v)

    if stream:
        feed1, feed2 = st.columns([1, 4])
        fail_counter = 0
        # i = 0

        for i in range(df.shape[0] * 5):
            if stop_stream:
                stream = False
                break
            time.sleep(0.2)
            if KPI[i % N] == 'Fail':
                fail_counter = fail_counter + 1
                v['predict_count'].loc[v['reason'] == df['reason'].iloc[i % N]] = v['predict_count'].loc[
                                                                                      v['reason'] == df['reason'].iloc[
                                                                                          i % N]] + 1
            with metrics.container():
                st.metric(label="Predictions", value=i)
                st.metric(label="Fails", value=fail_counter)
                st.metric(label="Ratio", value=str(np.round(fail_counter / (i + 1) * 100, 1)) + "%")
            with error_inv.container():
                if KPI[i % N] == 'Fail':
                    feed1.error('FAIL @' + str(df.index[i % N]))
                    feed2.info(df['reason'].iloc[i % N])
            with graph_inv.container():
                q = v.copy()
                q['predict_count'] = q['predict_count'] / fail_counter * 100
                fig = px.bar(q,
                             x='reason',
                             y='predict_count',
                             barmode='group',
                             color_discrete_sequence=["#00818A"])
                fig.add_trace(px.bar(v,
                                     x='reason',
                                     y='train_count',
                                     barmode='group',
                                     color_discrete_sequence=["#52de97", "#00818A"]).data[0])
                fig.update_layout(plot_bgcolor='#ffffff')
                fig.update_layout(showlegend=True)
                fig.update_xaxes(type='category')
                fig.update_layout(yaxis_visible=True, yaxis_showticklabels=True)
                fig.update_layout(xaxis_visible=True, xaxis_showticklabels=True)
                fig.update_layout(
                    width=800,
                    height=400,
                    margin=dict(l=5, r=5, t=5, b=5),
                )
                st.write(fig)


def pre_paint_app(stream):
    st.title('Pre Paint Metal Defects')
    st.write(' ')

    col1, col2 = st.columns((2, 2))
    image_cont = col1.empty()
    class_cont = col2.empty()
    seen_cont = st.empty()

    runner, names, classes, seen_names, seen_class = [], [], [], [], []

    for folder in os.listdir(os.path.join('assets', 'Data', 'paint-data')):
        if "." not in folder:
            for file in os.listdir(os.path.join('assets', 'Data', 'paint-data', folder)):
                names.append(os.path.join('assets', 'Data', 'paint-data', folder, file))
                classes.append(folder)

    N = len(names)

    if stream:

        for j in range(N * 10):
            if stop_stream:
                stream = False
                break

            i = np.random.randint(0, N - 1, 1)[0]
            runner.append(classes[i % N])
            q = pd.DataFrame(runner)

            v = q[0].value_counts(normalize=False)
            v = v.reset_index(level=0)
            v.columns = ['class', 'count']

            with image_cont.container():
                time.sleep(1)
                st.image(names[i % N], use_column_width=True)  # , caption = names[i%N])
                seen_names.append(names[i % N])
                seen_class.append(classes[i % N])
            with class_cont.container():
                conf = np.random.randint(85, 100, 1)[0]
                st.info(classes[i % N] + ' with ' + str(conf) + '% confidence')
                fig = px.bar(v, y='class', x='count',
                             color='count',
                             color_discrete_sequence=['#00818A', '#52DE97', '#395243', '#ff3c78', '#f3f4d1', '#bada55'],
                             orientation='h')
                fig.update_layout(plot_bgcolor='#ffffff')
                fig.update_layout(
                    width=500,
                    # height=400,
                    # margin=dict(l=5, r=5, t=5, b=5),
                )
                st.write(fig)

            unique_list = (list(set(seen_class)))
            with seen_cont.container():
                for u in unique_list:
                    with st.expander(u):
                        g1, g2, g3, g4 = st.columns(4)
                        count = 0
                        for im in range(len(seen_names)):
                            if seen_class[im] == u:
                                if count == 0:
                                    g1.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                if count == 1:
                                    g2.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                if count == 2:
                                    g3.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                if count == 3:
                                    g4.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                count = count + 1
                                count = count % 4


def ask_for_files(app_type):
    if app_type == 'pre paint metal defects':
        return None
    if app_type == 'paint shop defect detection':
        # df = pd.read_csv('assets/Data/Images/car-pano.pn
        return None
    if app_type == 'real-time sensor anomaly detection':
        df = pd.read_csv('assets/Data/anomaly.csv', index_col=0)
        batch = st.file_uploader("upload batch file")
        st.write(batch)
        if batch is not None:
            df = pd.read_csv(batch)

            if 'prog' in df.columns:
                df.drop(columns=['prog'], inplace=True)
                df = (df - df.mean()) / df.std()

            df['sen_alert'] = 0
            df['sit_alert'] = 0
        loaded_files = [df]
        return loaded_files
    if app_type == 'adaptive AI demo':
        data_file = st.file_uploader("upload `good' file", accept_multiple_files=False)
        dont_file = st.file_uploader("upload `drift' file", accept_multiple_files=False)
        if data_file is not None:
            uploaded_file_int = pd.read_csv(data_file)
        else:
            uploaded_file_int = pd.read_csv('assets/Data/adaptive-ai-demo-data.csv')
        if dont_file is not None:
            dont_care_int = pd.read_csv(dont_file)
        else:
            dont_care_int = pd.read_csv('assets/Data/adaptive-ai-demo-drifted.csv')

        loaded_files = [uploaded_file_int, dont_care_int]
        st.write(loaded_files)
        return loaded_files
    if app_type == 'medical device early fault detection':
        batch = st.file_uploader('upload medical device data', accept_multiple_files=False)
        if batch is not None:
            raw = pd.read_csv(batch)
        else:
            raw = pd.read_csv('assets/Data/medical-data.csv')

        raw = raw.sample(frac=1).reset_index(drop=True)
        kpi = 'S_Scrap'
        kpi_col = raw[kpi].copy()
        df = raw.copy()
        df.drop(columns=[kpi], inplace=True)
        loaded_files = [df, kpi_col]
        st.write(loaded_files)
        return loaded_files
    if app_type == 'manual assembly with video':
        batch = st.file_uploader('upload assembly videos', accept_multiple_files=False)
        if batch is not None:
            raw = pd.read_csv(batch)
        else:
            raw = pd.read_csv('assets/Data/flex-results.csv', index_col=0)

        df = raw.copy()
        kpi_col = df['result'].copy()
        df.drop(columns=['result'], inplace=True)
        loaded_files = [df, kpi_col]
        st.write(loaded_files)
        return loaded_files

    st.error('app type not supported')


# sidebar
with st.sidebar:
    st.image('assets/Images/Vanti - Main Logo@4x copy.png')
    stream = False
    app_type = st.selectbox('select application', ['paint shop defect detection',
                                                   "pre paint metal defects",
                                                   'real-time sensor anomaly detection',
                                                   'adaptive AI demo',
                                                   'manual assembly with video',
                                                   'medical device early fault detection'])
    b1, b2 = st.columns(2)

    stream = b1.button('Start')
    stop_stream = b2.button('Stop')
    token = st.text_input('Vanti Model id', "####-production")

    connect = st.button('connect')
    if connect:
        for i in range(10000000):
            a = 1
        st.success('connected to to model')

    files = ask_for_files(app_type)

# main loop

# tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])
if app_type == 'paint shop defect detection':
    paint_shop_app(stream)

if app_type == 'real-time sensor anomaly detection':
    rt_sensors_app(stream)

if app_type == 'adaptive AI demo':
    adaptive_ai_demo(stream)

if app_type == 'manual assembly with video':
    video_assembly_app(stream)

if app_type == 'medical device early fault detection':
    medical_device_app(stream)

if app_type == 'pre paint metal defects':
    pre_paint_app(stream)
