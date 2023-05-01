# import streamlit.components.v1 as components
import time
import plotly.express as px
import os
# import streamlit as st
# import numpy as np
# import pandas as pd
from assets.helpful_stuff.auxFunctions import *
import plotly.graph_objects as go


def roadmap():
    # source = 'https://sharing.clickup.com/5712158/b/h/5ea8y-1402/c3da18542cfe989'
    # source = 'https://sharing.clickup.com/5712158/l/h/5ea8y-2262/ec42799aa27a3e3'
    source = 'https://sharing.clickup.com/5712158/l/h/5ea8y-2362/621790f9c8c8d11'
    st.components.v1.iframe(src=source,
                            width=1500,
                            height=900,
                            scrolling=True)


def ts_app(sensor_stream, stop_stream, files, header, sub_header, classification=False):
    st.title(header)
    st.subheader(sub_header)
    st.write('---------------------------------------------------------')
    df = files[0]
    if 'prog' in df.columns:
        df.drop(columns=['prog'], inplace=True)
        df = (df - df.mean()) / df.std()
    orig_col_list = df.columns

    if len(files)==4:
        jnj_file = files[3]
        with st.expander('full data'):
            st.dataframe(jnj_file)

    if classification:
        kpi_file = files[1]
        fi = files[2]
        # --------------------------------------
        fi['Feature Importance'] = fi['Feature Importance'].apply(lambda x: float(x.split('%')[0]))
        fi.sort_values(by=['Feature Importance'], ascending=False, inplace=True)
        fail_counter = 0
    else:
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
    # st.subheader('asdfsdf')
    alert_cont = st.empty()

    # debug = st.empty()

    if sensor_stream:
        for col in range(df.shape[0]):
            if stop_stream:
                break
            start_index = max(0, col - sensor_window)
            e = min(col, df.shape[0])
            temp = df[feats].iloc[start_index:e]

            if classification:
                if kpi_file[col] == 1:
                    fail_counter = fail_counter + 1
                if kpi_file[col] == 1:
                    rr = 'Fail'
                    detailed_reason = get_reason_medical(df, ms, ss)
                    q = pd.DataFrame({
                        'time stamp': [df.index[col]], 'sensor': '',
                        'reason': [rr], 'alert type': 'fail', 'detailed_reason': [detailed_reason]})
                    alerts = pd.concat([alerts, q], axis=0, ignore_index=True)
            else:
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
                        detailed_reason = get_reason_medical(df, ms, ss)
                        q = pd.DataFrame({
                            'time stamp': [df.index[col]], 'sensor': ['combination'],
                            'reason': [rr], 'alert type': ['Situation'], 'detailed_reason': [detailed_reason]})
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
                    detailed_reason = alerts['detailed_reason'].iloc[idx]
                    a = '\n and'.join(detailed_reason.split('and'))
                    a = ' and'.join(detailed_reason.split('and'))
                    if len(reason) > 20:
                        reason = reason[:19] + '\n ' + reason[19:]  # + '\n' + a[0] + '\n' + 'and ' + a[1]
                        st.code(f'{ts} :: {sens} : {reason} : \n {a}')
                    else:
                        st.code(f'{ts} :: {sens} : {reason} : \n {a}')

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


def visual_inspection_app(stream, stop_stream, title, sub_header, folder_name,
                          header_image=None,
                          moving_thumb=None,
                          scan_mode=False):
    st.title(title)
    st.subheader(sub_header)
    st.write('---------------------------------------------------------')
    if header_image is not None:
        with st.expander('full image'):
            st.image(header_image)
    if moving_thumb is not None:
        thumb_cont = st.empty()
    col1, col2 = st.columns((2, 2))
    image_cont = col1.empty()
    class_cont = col2.empty()
    seen_cont = st.empty()

    runner, names, classes, seen_names, seen_class = [], [], [], [], []

    for folder in os.listdir(os.path.join('assets', 'Data', folder_name)):
        if "." not in folder:
            for file in os.listdir(os.path.join('assets', 'Data', folder_name, folder)):
                names.append(os.path.join('assets', 'Data', folder_name, folder, file))
                classes.append(f'{folder}')

    names_len = len(names)

    with st.expander('OK/NG UI'):
        # st.text('asdfasdf')
        unique_classes = (list(set(classes)))
        q_sign = {}
        for idx, q in enumerate(unique_classes):
            # ans[idx] = 
            # q_sign.append(st.radio(q, ('OK','NG'), horizontal=True, key=idx))
            q_sign[q] = st.radio(q, ('OK','NG','Other'), horizontal=True, key=idx, index=2)
        st.write(q_sign)

    if stream:

        for j in range(names_len * 10):
            if stop_stream:
                break
            if scan_mode:
                k = j
            else:
                k = np.random.randint(0, names_len - 1, 1)[0]
            runner.append(classes[k % names_len])
            q = pd.DataFrame(runner)

            v = q[0].value_counts(normalize=False)
            v = v.reset_index(level=0)
            v.columns = ['class', 'count']
            if moving_thumb is not None:
                with thumb_cont.container():
                    zoom_img_name = names[k % names_len]
                    img_num = zoom_img_name.split('_')[0].split('/')[-1]
                    thumb_name = f'{img_num}_rect.png'
                    st.image(f'assets/Data/{folder_name}-thumb/{thumb_name}')

            with image_cont.container():
                time.sleep(1)
                st.image(names[k % names_len], use_column_width='auto')  # , caption = names[i%N])
                seen_names.append(names[k % names_len])
                seen_class.append(classes[k % names_len])
            with class_cont.container():
                conf = np.random.randint(85, 100, 1)[0]
                this_class = classes[k % names_len]
                if q_sign[this_class] == 'OK':
                    st.success(classes[k % names_len] + ' with ' + str(conf) + '% confidence')
                elif q_sign[this_class] == 'NG':
                    st.error(classes[k % names_len] + ' with ' + str(conf) + '% confidence')
                else:
                    st.info(classes[k % names_len] + ' with ' + str(conf) + '% confidence')
                fig = px.bar(v, y='class', x='count',
                             color='count',
                             color_discrete_sequence=['#00818A', '#52DE97', '#395243', '#ff3c78', '#f3f4d1',
                                                      '#bada55'],
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
