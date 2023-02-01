# from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components
import time
import plotly.express as px
import os
import toml
from assets.helpful_stuff.auxFunctions import *
import plotly.graph_objects as go

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
# stream = False

nodes = ['anomaly remover', 'formatter', 'mini decision tree', 'local regression', 'local classifier', 'SVM',
         'perceptron', 'nan filler', 'normalizer', 'encoder', 'balancer']

st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

primaryColor = toml.load(".streamlit/config.toml")['theme']['primaryColor']
style_description = f"""
<style>
div.stButton > button:first-child {{ border: 2px solid {primaryColor}; border-radius:10px 10px 10px 10px; }}
div.stButton > button:hover {{ background-color: {primaryColor}; color:#000000;}}
footer {{ visibility: hidden;}}
# header {{ visibility: hidden;}}
<style>
"""
st.markdown(style_description, unsafe_allow_html=True)


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
        for j in range(1, thr1 + thr2):
            # time.sleep(0.05)

            error_val = -1 if j >= thr1 else 0

            h_score, v_score, v_pred, h_pred = get_pred(base_perf, beta, vs, thr1, tar_concat[j], j)

            v_acc = calc_perf(df_predictions, 'Vanti')
            h_acc = calc_perf(df_predictions, 'Standard Model')

            new_p = pd.DataFrame({
                'Ground Truth': [inv_dic[tar_concat[j]]],
                'Vanti': [inv_dic[v_pred]],
                'Standard Model': [inv_dic[h_pred]],
                'Vanti ACC': [v_acc],
                'Standard Model ACC': [h_acc]
            })
            new_p.index = [j]

            df_predictions = pd.concat([df_predictions, new_p], axis=0)

            v_score_w = v_score_w * alpha + v_score * (1 - alpha)
            h_score_w = h_score_w * alpha + h_score * (1 - alpha)
            new_predictions = pd.DataFrame({'Vanti': [v_score_w],
                                            'Standard Model': [h_score_w],
                                            'error': [error_val],
                                            'x': [j]},
                                           index=[j])
            predictions = pd.concat([predictions, new_predictions], axis=0)

            h20_val = np.round(h_score_w, 2)
            vanti_val = np.round(predictions['Vanti'].iloc[j], 2)
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

                data = df_pca.iloc[:j].copy()

                data['pred'] = 'old'
                data['pred'].iloc[-1] = 'new'
                sct = px.scatter(data, x=n1, y=n2, color='pred')
                sct.update_layout(plot_bgcolor='#ffffff')
                # sct.update_xaxes(range=[0, 13])
                # sct.update_yaxes(range=[-13, 0])

                # st.write(sct)

                if j == 1:
                    feed1.success('@index :: ' + str(j))
                    feed2.success('all is good!')
                if not drop and h20_val < 0.7 and j > 100:
                    feed1.success('@index :: ' + str(j))
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
                    feed1.success('@index :: ' + str(j))
                    feed1.info('notice')
                    feed1.info('notice')
                    feed2.success('updating Vanti')
                    feed2.info('replacing node ' + str(node) + ' in layer ' + str(layer) + ' with ' + str(new_node))
                    feed2.info('Vanti: accuracy restored to ' + str(np.round(vanti_val * 100)) + '%')


# app functions
def paint_shop_app(ferrari_stream):
    st.title('In-line Paint Shop Defect Detection')
    st.image('assets/Images/ferrari-cropped.png')
    sbc1, sbc2 = st.columns(2)
    sensitivity = sbc1.slider('model sensitivity', 0, 100, 50)
    speed = sbc1.slider('select path size', 16, 64, 32)
    print(sensitivity, speed)

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

    seen_names, seen_class = [], []

    for file in os.listdir('assets/Images'):
        if '_zoom' in file:
            it = file.split('_')[0]
            dif = file.split('_')[-1].split('.')[0]
            defect_list[it] = dif

    _, c2 = st.columns(2)

    pl = st.empty()
    # p2 = st.empty()
    image_number = 146
    # is_error = False
    seen_cont = st.empty()

    if ferrari_stream:
        for j in range(image_number):
            if stop_stream:
                # ferrari_stream = False
                break

            with pl.container():
                im_name = 'assets/Images/' + str(j % image_number) + '_rect.png'
                st.image(im_name)

                defect = defect_list[str(j % image_number)]
                # print(i, defect)

                seen_names.append('assets/Images/' + str(j % image_number) + '_rect_thumb.png')
                seen_class.append(defect)

                if defect == 'no-defect':
                    st.success('no defect!')
                    is_error = False
                else:
                    is_error = True
                    q = pd.DataFrame({
                        'section': [j % image_number],
                        'defect': [defect],
                    })
                    alerts = pd.concat([alerts, q], axis=0, ignore_index=True)

            unique_list = (list(set(seen_class)))

            with seen_cont.container():
                for u in unique_list:
                    with st.expander(u):
                        selected = []
                        for name, cls in zip(seen_names, seen_class):
                            if cls == u:
                                selected.append(name)
                        st.image(selected)
            if is_error:
                with st.expander(defect + '  ::  defect-alert zoom-in @ section' + str(j % image_number)):
                    st.image('assets/Images/' + str(j % image_number) + '_zoom_' + defect + '.png')


def rt_sensors_app(sensor_stream):
    st.title('Real Time Anomaly Detection')
    df = files[0]
    if 'prog' in df.columns:
        df.drop(columns=['prog'], inplace=True)
        df = (df - df.mean()) / df.std()

    orig_col_list = df.columns
    df['sen_alert'] = 0
    df['sit_alert'] = 0

    col_list = ['All Sensors']

    for col in df.columns.to_list():
        col_list.append(col)
    feats = st.multiselect("Select Sensors", col_list)

    st.text("You selected: {}".format(", ".join(feats)))
    if 'All Sensors' in feats:
        feats = df.columns.to_list()

    c1, c2 = st.columns(2)
    mode = c1.radio('select alert mode',
                    ['alert me only when there''s a situation anomaly',
                     'alert me only when there''s a sensor anomaly',
                     'I want all alerts'])
    normalize = st.checkbox('Scale Sensors', value=True)
    auto_clean_up = st.checkbox('Automated Data Cleaning', value=True)
    if auto_clean_up:
        for col in orig_col_list:
            df[col][df[col] == 0] = df[col].mean()

    if normalize:
        df = df / df.max()
    else:
        df = df

    max_val = df.max().max() * 1.1
    # min_val = df.min().min() * 1.1
    with c2.expander('what are these alerts?'):
        st.write('a **sensor** anomaly is when a single sensor is tracked by Vanti''s model and the model decides to '
                 'alert the user')
        st.write('a **situation** anomaly is when all sensors are tracked together by Vanti''s model and the model '
                 'decides to alert the user')

    if mode == 'alert me only when there''s a situation anomaly':
        alert_mode = 0
    elif mode == 'alert me only when there''s a sensor anomaly':
        alert_mode = 1
    elif mode == 'I want all alerts':
        alert_mode = 2
    else:
        alert_mode = 3
    print(alert_mode)

    sensitivity = c1.slider('alert sensitivity', 0.0, 100.0, 50.0)
    with c2.expander("what is model sensitivity?"):
        st.write("_sensitivity 100 --> alert me on **everything**_")
        st.write("_sensitivity 0 --> alert me on **critical things only**_")

    ms = {feat: df[feat].mean() for feat in feats}
    ss = {feat: df[feat].std() for feat in feats}

    _, c2, c3 = st.columns(3)
    sensitivity = (100 - sensitivity) / 20

    sensor_window = 50
    zoom_window = 2
    # pl = st.empty()
    pl2 = st.empty()
    alerts = pd.DataFrame()

    alert_highlights = {}
    local_count = {}
    alert_start, alert_end, alert_enable = [], [], []
    graph_col, alert_col = st.columns((3, 2))
    graph_cont = graph_col.empty()
    alert_cont = alert_col.empty()

    debug = st.empty()

    if sensor_stream:
        for col in range(df.shape[0]):
            if stop_stream:
                break

            start_index = max(0, col - sensor_window)
            e = min(col, df.shape[0])
            temp = df[feats].iloc[start_index:e]
            # temp['sen_alert'] = df['sen_alert'].iloc[start_index:e]
            # temp['sit_alert'] = df['sit_alert'].iloc[start_index:e]
            count = 0
            for f in feats:
                if np.abs(df[f].iloc[col] - ms[f]) > (ss[f] * sensitivity):
                    count = count + 1
                    rr = get_reason('sensor')
                    q = pd.DataFrame({
                        'time stamp': [df.index[col]],
                        'sensor': [f],
                        'reason': [rr],
                        'alert type': ['sensor']
                    })

                    if alert_mode == 1 or alert_mode == 2:
                        alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
                        colorh = '#ff3c78'
                        sss = max(0, col - zoom_window)
                        eee = min(col + zoom_window, df.shape[0])

                        alert_num = df['sit_alert'].cumsum().max()
                        alert_start.append(df.index[sss])
                        alert_end.append(df.index[eee])
                        alert_enable.append(0)
                        alert_highlights['alert_' + str(alert_num)] = go.layout.Shape(
                            type="rect",
                            x0=df.index[sss],
                            y0=-0.1,
                            x1=df.index[eee],
                            y1=max_val,
                            fillcolor=colorh,
                            line=dict(color=colorh, width=3),
                            opacity=0.25,
                            layer="below")

                        with pl2.container():
                            sss = max(0, col - zoom_window)
                            eee = min(col + zoom_window, df.shape[0])
                            temp2 = df[f].iloc[sss:eee]
                            fig3 = px.line(temp2)
                            fig3.update_layout(plot_bgcolor='#ffffff')
                        with st.expander('sensor-alert zoom-in @' + str(df.index[col])):
                            st.write(fig3, title=str(df.index[col]))

            if alert_mode == 0 or alert_mode == 2:
                if count > 3:
                    rr = get_reason('situation')
                    q = pd.DataFrame({
                        'time stamp': [df.index[col]], 'sensor': ['combination'],
                        'reason': [rr], 'alert type': ['Situation']})
                    alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
                    df['sit_alert'].iloc[col] = 1
                    colorh = '#00818A'
                    sss = max(col - sensor_window + 1, col - zoom_window)
                    eee = min(col + zoom_window, col + sensor_window - 1)
                    alert_num = df['sit_alert'].cumsum().max()
                    alert_start.append(df.index[sss])
                    alert_end.append(df.index[eee])
                    alert_enable.append(0)

                    alert_highlights['alert_' + str(int(alert_num))] = go.layout.Shape(
                        type="rect", x0=df.index[sss], y0=-0.1, x1=df.index[eee], y1=max_val,
                        # type="rect", x0=0, y0=-0.1, x1=4, y1=max_val,
                        fillcolor=colorh, line=dict(color=colorh, width=3), opacity=0.25, layer="below")

                    with pl2.container():
                        sss = max(col - sensor_window + 1, col - zoom_window)
                        eee = min(col + zoom_window, col + sensor_window - 1)
                        temp2 = df[feats].iloc[sss:eee]
                        fig3 = px.line(temp2)
                        fig3.update_layout(plot_bgcolor='#ffffff')

                    with st.expander('situation-alert zoom-in @' + str(df.index[col])):
                        st.write(fig3, title=str(df.index[col]))

            with alert_cont.container():
                for idx in range(alerts.shape[0]):
                    ts = alerts['time stamp'].iloc[idx]
                    sens = alerts['sensor'].iloc[idx]
                    reason = alerts['reason'].iloc[idx]
                    if len(reason) > 20:
                        reason = reason[:19] + '\n ' + reason[19:]
                        st.code(f'{ts} :: {sens} : {reason}')  # : {reason[:20]}')
                    else:
                        st.code(f'{ts} :: {sens} : {reason}')

            with graph_cont.container():
                alert_keys = list(alert_highlights.keys())
                local_alerts = {}
                local_start, local_end = [], []
                tries = 5
                show_count = 40
                for alert_idx in range(len(alert_start)):
                    # for each alert see if it's enabled or not
                    # if not enabled toggle it to enable in next iteration
                    if alert_enable[alert_idx] <= tries:
                        alert_enable[alert_idx] = alert_enable[alert_idx] + 1
                    # if enabled, copy alert details and start to local list
                    else:
                        local_alerts[alert_keys[alert_idx]] = alert_highlights[alert_keys[alert_idx]]
                        local_start.append(alert_start[alert_idx])
                        local_end.append(alert_end[alert_idx])
                        if alert_keys[alert_idx] not in local_count.keys():
                            local_count[alert_keys[alert_idx]] = 0

                    if alert_start[alert_idx] not in temp.index and alert_end[alert_idx] not in temp.index:
                        del alert_highlights[alert_keys[alert_idx]]
                        del alert_start[alert_idx]
                        del alert_end[alert_idx]

                    # for all alerts to shown count the number of times it was shown
                    local_keys = list(local_alerts.keys())
                    for local_idx in range(len(local_alerts)):
                        local_count[local_keys[local_idx]] = local_count[local_keys[local_idx]] + 1
                        if local_count[local_keys[local_idx]] >= show_count:
                            del local_alerts[local_keys[local_idx]]
                            del local_start[local_idx]
                            del local_end[local_idx]

                if temp.shape[0] < sensor_window:
                    pad = pd.DataFrame(index=df.index[temp.shape[1]:sensor_window], columns=temp.columns)
                    temp = pd.concat([temp, pad], axis=0)

                fig = px.line(data_frame=temp)
                fig.update_layout(plot_bgcolor='#ffffff')
                fig.update_yaxes(range=[-0.1, 1.1], fixedrange=True)

                # lst_shapes = list(alert_highlights.values())
                lst_shapes = list(local_alerts.values())
                fig.update_layout(shapes=lst_shapes)
                st.write(fig)

            # with debug.container():
            #     st.write(alert_highlights)
            #     st.write(local_count)
            #     st.write(local_alerts)
            #     st.write(pad.shape)
            #     # for alert_idx in range(len(alert_start_end)):
            #     #     if alert_start_end[alert_idx] in temp.index:
            #     #         st.write(alert_start_end[alert_idx])
            #     st.write(alert_highlights)

    with st.expander('see training data'):
        st.dataframe(df)


def medical_device_app(medical_stream):
    st.title('Medical Device Early Fault Detection')

    df = files[0]
    kpi_file = files[1]
    fi = files[2]
    # --------------------------------------
    fi['Feature Importance'] = fi['Feature Importance'].apply(lambda x: float(x.split('%')[0]))
    fi.sort_values(by=['Feature Importance'], ascending=False, inplace=True)

    # --------------------------------------
    col_list = ['All Measurements']

    for col in df.columns.to_list():
        col_list.append(col)
    feats = st.multiselect("Select Data", col_list)

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
    print(sensitivity)

    with c2.expander("what is model sensitivity?"):
        st.write("_sensitivity 100 --> alert me on **everything**_")
        st.write("_sensitivity 0 --> alert me on **critical things only**_")

    with st.expander('Driving Factors'):
        st.bar_chart(fi)
    _, sbc2 = st.columns(2)

    ms = {feat: df[feat].mean() for feat in feats}
    ss = {feat: df[feat].std() for feat in feats}

    data_graph = st.empty()
    error_inv = st.empty()
    metrics = st.empty()
    if medical_stream:

        feed1, feed2 = st.columns([1, 4])
        fail_counter = 0

        for col in range(df.shape[0]):
            if stop_stream:
                # medical_stream = False
                break
            if kpi_file[col] == 1:
                fail_counter = fail_counter + 1
            start_index = 0
            e = min(col, df.shape[0])
            temp = df[feats].iloc[start_index:e]

            with data_graph.container():
                # sss = max(0, col - 10)
                # eee = min(col, df.shape[0])
                # temp2 = df.iloc[sss:eee]

                fig3 = px.line(temp, markers=True)
                fig3.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
                # hide and lock down axes
                fig3.update_xaxes(visible=True, fixedrange=True)
                fig3.update_yaxes(visible=True, fixedrange=True)
                # remove facet/subplot labels
                fig3.update_layout(annotations=[], overwrite=True)
                st.write(fig3)
                with metrics.container():
                    st.metric(label="Predictions", value=col)
                    st.metric(label="Fails", value=fail_counter)
                    st.metric(label="Ratio", value=str(np.round(fail_counter / (col + 1) * 100, 1)) + "%")
            with error_inv.container():
                if kpi_file[col] == 1:
                    feed1.error('FAIL @SN = ' + str(df.index[col]))
                    # feed2.error('@index :: ' + str(i))
                    feed2.info(get_reason_medical(df, ms, ss))


def adaptive_ai_demo():
    st.title('Adaptive AI Demo')
    st.header('Run Experiment')
    models(vanti_app_url, h2o_app_url)
    uploaded_file = files[0]
    dont_care = files[1]
    if st.button('Run!'):
        run_exp(uploaded_file, dont_care, BASE_PERF, BETA, VS)
    else:
        print('not run')

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
        # st.image('as6 easy step copy.png')
        st.image('assets/helpful_stuff/6 easy step copy.png')

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


def video_assembly_app(assembly_stream):
    st.title('Defect Detection in Video Assembly')
    st.write(' ')
    col1, col2 = st.columns((1, 4))

    with col1:
        st.write(' ')

    with col2:
        st.image('assets/Data/assembly-movie-small.gif', caption='assembly video')

    # with col3:
    #     st.write(' ')

    df = files[0]
    kpi_file = files[1]
    video_num = df.shape[0]
    metrics = col1.empty()
    error_inv = st.empty()
    with st.expander('Operator Annotation'):
        with st.form("what was the error?"):
            ui1, ui2 = st.columns(2)
            user_sn = ui1.text_input('Unit SN')
            user_reason = ui2.text_input('what was the cause of failure?')
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write("Unit", user_sn, "root cause", user_reason)

    st.subheader('train vs real time root cause distribution')
    graph_inv = st.empty()

    st.subheader('Root Cause per Unit')

    v = df['reason'].value_counts(normalize=True) * 100
    v = v.reset_index(level=0)
    v.columns = ['reason', 'train_count']
    v['predict_count'] = 0
    # st.write(v)

    if assembly_stream:
        feed1, feed2 = st.columns([1, 4])
        fail_counter = 0
        # i = 0

        for j in range(df.shape[0] * 5):
            if stop_stream:
                # assembly_stream = False
                break
            time.sleep(0.2)
            if kpi_file[j % video_num] == 'Fail':
                fail_counter = fail_counter + 1
                v['predict_count'].loc[v['reason'] == df['reason'].iloc[j % video_num]] = v['predict_count'].loc[
                                                                                              v['reason'] ==
                                                                                              df['reason'].iloc[
                                                                                                  j % video_num]] + 1
            with metrics.container():
                st.metric(label="Predictions", value=j)
                st.metric(label="Fails", value=fail_counter)
                st.metric(label="Ratio", value=str(np.round(fail_counter / (j + 1) * 100, 1)) + "%")
            with error_inv.container():
                if kpi_file[j % video_num] == 'Fail':
                    feed1.error('FAIL @' + str(df.index[j % video_num]))
                    feed2.info(df['reason'].iloc[j % video_num])
                    # if df['reason'].iloc[i % N] == 'General Error':
                    #     new_reason = st.text_input(str(i%N)+'_what was the reason?')
                    #     df['reason'].iloc[i % N] = new_reason
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


def pre_paint_app(paint_stream):
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

    names_len = len(names)

    if paint_stream:

        for j in range(names_len * 10):
            if stop_stream:
                # paint_stream = False
                break

            k = np.random.randint(0, names_len - 1, 1)[0]
            runner.append(classes[k % names_len])
            q = pd.DataFrame(runner)

            v = q[0].value_counts(normalize=False)
            v = v.reset_index(level=0)
            v.columns = ['class', 'count']

            with image_cont.container():
                time.sleep(1)
                st.image(names[k % names_len], use_column_width=True)  # , caption = names[i%N])
                seen_names.append(names[k % names_len])
                seen_class.append(classes[k % names_len])
            with class_cont.container():
                conf = np.random.randint(85, 100, 1)[0]
                st.info(classes[k % names_len] + ' with ' + str(conf) + '% confidence')
                fig = px.bar(v, y='class', x='count',
                             color='count',
                             color_discrete_sequence=['#00818A', '#52DE97', '#395243', '#ff3c78', '#f3f4d1', '#bada55'],
                             orientation='h')
                fig.update_layout(plot_bgcolor='#ffffff')
                fig.update_layout(width=500)
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


def textile_app(textile_stream):
    st.title('Textile Defects')
    st.write(' ')

    col1, col2 = st.columns((2, 2))
    image_cont = col1.empty()
    class_cont = col2.empty()
    seen_cont = st.empty()

    runner, names, classes, seen_names, seen_class = [], [], [], [], []

    for folder in os.listdir(os.path.join('assets', 'Data', 'textile-data')):
        if "." not in folder:
            for file in os.listdir(os.path.join('assets', 'Data', 'textile-data', folder)):
                names.append(os.path.join('assets', 'Data', 'textile-data', folder, file))
                classes.append(folder)

    names_len = len(names)

    if textile_stream:

        for j in range(names_len * 10):
            if stop_stream:
                break

            k = np.random.randint(0, names_len - 1, 1)[0]
            runner.append(classes[k % names_len])
            q = pd.DataFrame(runner)

            v = q[0].value_counts(normalize=False)
            v = v.reset_index(level=0)
            v.columns = ['class', 'count']

            with image_cont.container():
                time.sleep(1)
                st.image(names[k % names_len], use_column_width=True)  # , caption = names[i%N])
                seen_names.append(names[k % names_len])
                seen_class.append(classes[k % names_len])
            with class_cont.container():
                conf = np.random.randint(85, 100, 1)[0]
                st.info(classes[k % names_len] + ' with ' + str(conf) + '% confidence')
                fig = px.bar(v, y='class', x='count',
                             color='count',
                             color_discrete_sequence=['#00818A', '#52DE97', '#395243', '#ff3c78', '#f3f4d1', '#bada55'],
                             orientation='h')
                fig.update_layout(plot_bgcolor='#ffffff')
                fig.update_layout(width=500)
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


def rt_test_reorder(test_order_stream):
    st.title('Real Time Process Optimization')
    st.subheader('cycle time reduction with dynamic test reordering')
    st.write('---------------------------------------------------------')
    df = files[0]
    df.set_index('time', drop=True, inplace=True)
    df = df.astype(np.int8)

    nominal = 60

    col1, dummy, col2 = st.columns((4, 1, 2))
    metrics = dummy.empty()
    data_graph = col1.empty()
    list_cont = col2.empty()

    if test_order_stream:
        for j in range(df.shape[0]):
            if stop_stream:
                # test_order_stream = False
                break
            optimized = nominal - np.random.randint(10, 15)
            tp = int((nominal / optimized - 1) * 100)
            with metrics.container():
                st.metric(label="Nominal", value=nominal)
                st.metric(label="Optimized Test Time", value=optimized)
                st.metric(label="Throughput", value='+' + str(tp) + "%")

            with data_graph.container():
                # sss = max(0, j - test_reorder_window)
                sss = 0
                eee = min(j, df.shape[0])
                temp2 = df.iloc[sss:eee]

                fig3 = px.line(temp2, markers=True)
                fig3.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
                fig3.update_xaxes(visible=True, fixedrange=True)
                fig3.update_yaxes(visible=True, fixedrange=True)
                fig3.update_layout(annotations=[], overwrite=True)
                st.write(fig3)
            with list_cont.container():
                local = df.iloc[j].copy()
                local = pd.DataFrame(local)
                # title = local.columns
                st.code(local.columns[0] + ' top 5 tests in order')
                local.columns = ['order']
                local.sort_values(by='order', ascending=True, inplace=True)
                st.code(''.join(['* ' + q + '\n' for q in local.index.to_list()]))
        with st.expander('full data'):
            st.line_chart(df)
    return None


def si_demo(si_stream):
    st.title('Standard Industries Demo')
    st.subheader('event prediction')
    fi = files[2]
    predictions = files[0]
    df = files[1]
    si_window = 10
    norm = st.checkbox('normalize values?', value=True)
    if norm:
        df = (df - df.min()) / (df.max() - df.min())

    fi['importance'] = fi['importance'].astype('float') * 100
    fi.set_index('feature', drop=True, inplace=True)
    fi.sort_values(by=['importance'], ascending=False, inplace=True)
    fi = fi.iloc[:10]
    # st.write(fi)
    with st.expander('Driving Factors'):
        st.bar_chart(fi)
    cm_cont = st.empty()
    tit, col1, col2, col3, col4 = st.columns((1, 1, 1, 1, 5))
    gt = tit.empty()
    conf_mat_1 = col1.empty()
    conf_mat_2 = col2.empty()
    conf_mat_3 = col3.empty()
    graph_mat = col4.empty()
    n = predictions.shape[0]
    cm = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    if si_stream:
        for si_idx in range(n):
            if stop_stream:
                break
            test = predictions['tusCoatingLine.MES.UtilizationState'].iloc[si_idx]
            pred = predictions['predictions'].iloc[si_idx]
            if test == 'Running':
                if pred == 'Running':
                    cm[0][0] = cm[0][0] + 1
                if pred == 'Running Slow':
                    cm[0][1] = cm[0][1] + 1
                if pred == 'Downtime':
                    cm[0][2] = cm[0][2] + 1
            if test == 'Running Slow':
                if pred == 'Running':
                    cm[1][1] = cm[1][0] + 1
                if pred == 'Running Slow':
                    cm[1][1] = cm[1][1] + 1
                if pred == 'Downtime':
                    cm[1][2] = cm[1][2] + 1
            if test == 'Downtime':
                if pred == 'Running':
                    cm[2][0] = cm[2][0] + 1
                if pred == 'Running Slow':
                    cm[2][1] = cm[2][1] + 1
                if pred == 'Downtime':
                    cm[2][2] = cm[2][2] + 1
            # st.write(test, pred)
            with cm_cont.container():
                if test == pred:
                    st.success(f'{df.index[si_idx]} : the model predicted {pred} -- the real result was also {test}')
                else:
                    st.success(f'{df.index[si_idx]} : the model predicted {pred} -- but the real result was  {test}')

                # st.info(test)
                # st.success(pred)
            st.subheader('')
            with graph_mat.container():
                sss = max(0, si_idx - si_window)
                eee = min(df.shape[0], si_idx + 1)

                st.line_chart(df.iloc[sss:eee])
            with gt.container():
                st.write('GROUND TRUTH')
                st.write(' ')
                st.write(' ')
                st.write('Running')
                st.write('')
                st.write('')
                st.write('')
                st.write('Running Slow')
                st.write('')
                st.write('')
                st.write('')
                st.write('Downtime')
            with conf_mat_1.container():
                st.write('Running')
                st.metric(label='', value=cm[0][0])
                st.metric(label='', value=cm[1][0])
                st.metric(label='', value=cm[2][0])
            with conf_mat_2.container():
                st.write('Running Slow')
                st.metric(label='', value=cm[1][0])
                st.metric(label='', value=cm[1][1])
                st.metric(label='', value=cm[1][2])
            with conf_mat_3.container():
                st.write('Downtime')
                st.metric(label='', value=cm[2][0])
                st.metric(label='', value=cm[2][1])
                st.metric(label='', value=cm[2][2])


def cpc(cpc_stream):
    st.title('Continuous Process Optimization Demo')
    st.subheader('closed loop power consumption reduction in real-time')
    st.write('---------------------------------------------------------')
    df = files[0]
    df.set_index('time', drop=True, inplace=True)
    df = df.astype(np.int8)

    nominal = 60

    random_dict = {i:np.random.choice([1,2,3,4,5]) for i in range(5)}
    df.replace(random_dict, inplace=True)

    col1, dummy, col2 = st.columns((4, 1, 2))
    metrics = dummy.empty()
    data_graph = col1.empty()
    list_cont = col2.empty()

    repeat_factor = 1

    if cpc_stream:
        for jj in range(df.shape[0] * repeat_factor):

            j = jj % df.shape[0]

            if stop_stream:
                # test_order_stream = False
                break
            optimized = nominal - np.random.randint(10, 15)
            tp = int((nominal / optimized - 1) * 100)
            with metrics.container():
                st.metric(label="Nominal Consumption", value=f'{nominal} [MW/h]')
                st.metric(label="Optimized Consumption", value=f'{optimized} [MW/h]')
                st.metric(label="Savings", value='+' + str(tp) + "%")

            with data_graph.container():
                sss = 0
                eee = min(j, df.shape[0] * repeat_factor)
                temp2 = df.iloc[sss:eee]

                fig3 = px.line(temp2, markers=True)
                fig3.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
                fig3.update_xaxes(visible=True, fixedrange=True)
                fig3.update_yaxes(visible=True, fixedrange=True)
                fig3.update_layout(annotations=[], overwrite=True)
                st.write(fig3)
            with list_cont.container():
                local = df.iloc[j].copy()
                local = pd.DataFrame(local)
                # title = local.columns
                st.code(local.columns[0] + ' instructions')
                local.columns = ['order']
                # local.sort_values(by='order', ascending=True, inplace=True)
                instructions = {
                    1: f' {np.round(np.random.randint(-500,-100)/100,2)}%',
                    2: f' {np.round(np.random.randint(-100,0)/100,2)}%',
                    4: f' {np.round(np.random.randint(0, 100)/100,2)}%',
                    5: f' {np.round(np.random.randint(100, 500)/100,2)}%',
                    3: ' no change',
                }
                st.code(''.join(['* ' + q + instructions[local['order'][idx]]+'\n' for idx, q in enumerate(local.index.to_list())]))
        with st.expander('full data'):
            st.line_chart(df)
    return None


def ask_for_files(app_type_file):
    if app_type_file == 'real time process optimization':
        df = pd.read_csv('assets/Data/test-reorder-data.csv')
        loaded_files = [df]
        return loaded_files
    if app_type_file == 'pre paint metal defects':
        return None
    if app_type_file == 'textile defects':
        return None

    if app_type_file == 'continuous process optimization demo':
        df = pd.read_csv('assets/Data/test-reorder-data.csv')
        df.columns = ['time', 'Env Temperature', 'H1 Pressure', 'H2 Pressure', 'M1 motor velocity', 'Valve Release']
        loaded_files = [df]
        return loaded_files

    if app_type_file == 'paint shop defect detection':
        # df = pd.read_csv('assets/Data/Images/car-pano.png')
        return None
    if app_type_file == 'real-time sensor anomaly detection':
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
    if app_type_file == 'adaptive AI demo':
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
    if app_type_file == 'medical device early fault detection':
        batch = st.file_uploader('upload medical device data', accept_multiple_files=False)
        fe = st.file_uploader('upload model feature importance', accept_multiple_files=False)
        if batch is not None:
            raw = pd.read_csv(batch)
        else:
            raw = pd.read_csv('assets/Data/medical-data.csv')
            fe = pd.read_csv('assets/Data/medical_device_feature_importance.csv', index_col=0)

        raw = raw.sample(frac=1).reset_index(drop=True)
        kpi = 'S_Scrap'
        kpi_col = raw[kpi].copy()
        df = raw.copy()
        df.drop(columns=[kpi], inplace=True)
        loaded_files = [df, kpi_col, fe]
        st.write(loaded_files)
        return loaded_files
    if app_type_file == 'manual assembly with video':
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
    if app_type_file == 'Standard Industries Demo':
        batch = st.file_uploader('upload data file', accept_multiple_files=False)
        if batch is not None:
            raw = pd.read_csv(batch)
        else:
            raw = pd.read_csv('assets/Data/SI-results.csv', index_col=0)
        loaded_files = [raw,
                        pd.read_csv('assets/Data/top-10-feats.csv', index_col=0),
                        pd.read_csv('assets/Data/SI_feat_imp.csv', index_col=0)]
        return loaded_files

    st.error('app type not supported sdsdf')


# sidebar
with st.sidebar:
    st.image('assets/Images/Vanti - Main Logo@4x copy.png')
    app_type = st.selectbox('select application', ['continuous process optimization demo',
                                                   'textile defects',
                                                   'Standard Industries Demo',
                                                   'real time process optimization',
                                                   'paint shop defect detection',
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
if app_type == 'continuous process optimization demo':
    cpc(stream)

if app_type == 'Standard Industries Demo':
    si_demo(stream)

if app_type == 'real time process optimization':
    rt_test_reorder(stream)

if app_type == 'paint shop defect detection':
    paint_shop_app(stream)

if app_type == 'textile defects':
    textile_app(stream)

if app_type == 'real-time sensor anomaly detection':
    rt_sensors_app(stream)

if app_type == 'adaptive AI demo':
    adaptive_ai_demo()

if app_type == 'manual assembly with video':
    video_assembly_app(stream)

if app_type == 'medical device early fault detection':
    medical_device_app(stream)

if app_type == 'pre paint metal defects':
    pre_paint_app(stream)
