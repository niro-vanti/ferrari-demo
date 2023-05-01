import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def si_demo(si_stream, stop_stream, files):
    st.title('Machine Speed 3 minute prediction Demo')
    st.subheader('event prediction')
    st.write('---------------------------------------------------------')
    kpi = 'tusCoatingLine.MES.UtilizationState'
    fi = files[2]
    predictions = files[0]
    predictions  = pd.read_csv('assets/Data/standard-inds/shift_6.csv', index_col=0)
    predictions.drop(columns=['predict_result'], inplace=True)
    predictions.index = pd.to_datetime(predictions.index)

    df = files[1]
    df = pd.read_csv('assets/Data/standard-inds/cols.csv', usecols=df.columns)
    # df.to_csv('assets/Data/standard-inds/cols.csv')
    df = df.iloc[2892:]
    df.index = predictions.index
    valid_locations = predictions[kpi] != 'Unknown'
    predictions = predictions.loc[valid_locations]
    df = df.loc[valid_locations]

    si_window = 10

    qs, es = st.columns((2, 4))
    norm = qs.checkbox('normalize values?', value=True)
    event_log = qs.checkbox('show event log?', value=True)
    supervised = qs.checkbox('supervised?', value=True)
    skip_rows = qs.checkbox('skip first predictions?', value=True)
    if skip_rows:
        start_loop = 870
    else:
        start_loop = 0
    with es.expander('what is value normalization?'):
        st.write('value normalization is taking each feature and scaling its value to 0--1 range')
    with es.expander('what is event log?'):
        st.write('the event log is a running log for each non-normal event')
    with es.expander('what is supervised?'):
        st.write('a supervised app is when ground truth labels are used to estimate model'
                 ' performance in classification')
        st.write('an unsupervised app does not use ground truth labels and assigns each data point to a cluster')

    if norm:
        df = (df - df.min()) / (df.max() - df.min())

    fi['importance'] = fi['importance'].astype('float') * 100
    fi.set_index('feature', drop=True, inplace=True)
    fi.sort_values(by=['importance'], ascending=False, inplace=True)
    fi = fi.iloc[:10]
    # st.write(fi)
    with st.expander('Driving Factors'):
        st.bar_chart(fi)
    pred_col1, pred_col2 = st.columns(2)
    pred_cont = pred_col1.empty()
    gauge_cont = pred_col2.empty()
    log_cont = st.empty()

    cm_cont = st.empty()
    sct_cont = st.empty()
    tit, col1, col2, col3, col4 = st.columns((1, 1, 1, 1, 5))
    gt = tit.empty()
    conf_mat_1 = col1.empty()
    conf_mat_2 = col2.empty()
    conf_mat_3 = col3.empty()
    graph_mat = col4.empty()
    n = predictions.shape[0]
    cm = [
        [start_loop, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    log_str = []
    prev_pred = False

    error_counter = 0
    df_u = pd.concat([df, predictions], axis=1)
    sns = []

    prev_update = None

    with st.expander('batch results demo'):
        # dfb = pd.read_csv('assets/Data/standard-inds/blind_test_19_2-results.csv', index_col=1)
        dfb = pd.read_csv('assets/Data/standard-inds/shift_6.csv', index_col=0)
        valid_locations = dfb[kpi]!='Unknown'
        # fig3 = px.line(dfb['predict_result'], markers=True, width=1200)
        fig3 = px.line(dfb[['predictions',kpi]].loc[valid_locations], markers=True, width=1200)
        fig3.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
        fig3['data'][0]['line']['color'] = '#394253'
        fig3['data'][1]['line']['color'] = '#52de97'
        # fig3['data'][1]['line']['dash'] = 'dot'
        fig3['data'][0]['mode'] = 'markers'
        fig3['data'][1]['mode'] = 'markers'
        fig3['data'][0]['marker']['symbol'] = "x-thin-open"
        fig3['data'][1]['marker']['symbol'] = "octagon-open"
        fig3['data'][1]['marker']['size'] = 5
        # fig3['data'][0]['line']['width'] = 5
        fig3.update_xaxes(visible=True, fixedrange=True)
        fig3.update_yaxes(visible=True, fixedrange=True)
        fig3.update_layout(annotations=[], overwrite=True)
        st.write(fig3)

    if si_stream:
        for si_idx in range(start_loop,n):
            if stop_stream:
                break
            if supervised:
                test = predictions['tusCoatingLine.MES.UtilizationState'].iloc[si_idx]
                pred = predictions['predictions'].iloc[si_idx]
                if pred != test:
                    error_counter = error_counter + 1
                if pred in ['Running Slow', 'Downtime']:
                    if pred != prev_pred:
                        sns = [np.random.choice(df.columns.to_list(), replace=False) for _ in
                               range(np.random.choice([1, 2, 3]))]
                        log_str.append(f'{df.index[si_idx]} : the model predicted {pred} -- check {sns}')
                    prev_pred = pred
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
                        st.success(
                            f'{df.index[si_idx]} : the model predicted {pred} -- the real result was also {test}')
                    else:
                        st.error(f'{df.index[si_idx]} : the model predicted {pred} -- but the real result was  {test}')
                    # st.subheader(f'Accuracy = {np.round((cm[0][0] + cm[1][1] + cm[2][2]) / np.sum(cm)*100,2)} %')
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
                    st.metric(label='', value=cm[0][1])
                    st.metric(label='', value=cm[0][2])
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
                if event_log:
                    with log_cont.container():
                        new_update = pred
                        if new_update != prev_update:
                            st.code(''.join(['* ' + q + '\n' for idx, q in enumerate(log_str)]))
                            prev_update = new_update
                    with pred_cont.container():
                        st.subheader('event log')
                        st.write('---------------------------------------------------------')
                        temp2 = predictions.iloc[:si_idx]
                        fig_running = px.line(temp2, markers=True)
                        fig_running.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
                        fig_running.update_xaxes(visible=True, fixedrange=True)
                        fig_running.update_yaxes(visible=True, fixedrange=True)
                        fig_running.update_layout(annotations=[], overwrite=True)
                        st.write(fig_running)
                    with gauge_cont.container():
                        # st.write('---------------------------------------------------------')
                        g_map = {'Downtime': 0, 'Running Slow': 50, 'Running': 100}
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge",
                            value=g_map[pred],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': 'machine speed'},
                            gauge={'axis': {'range': [-10, 200]},
                                   'bar': {'color': "#52de97"},
                                   'steps': [
                                       {'range': [-10, 25], 'color': "#ff3c78"},
                                       {'range': [25, 75], 'color': "#00818A"}],
                                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75,
                                                 'value': 75}}))
                        st.write(fig_gauge)
                        # st.line_chart(predictions['predictions'].iloc[:si_idx])
            if not supervised:
                pred = predictions['predictions'].iloc[si_idx]
                if pred in ['Running Slow', 'Downtime']:
                    if pred != prev_pred:
                        sns = [np.random.choice(df.columns.to_list(), replace=False) for _ in
                               range(np.random.choice([1, 2, 3]))]

                        log_str.append(f'{df.index[si_idx]} : the model predicted {pred} -- check {sns}')
                    prev_pred = pred
                with sct_cont.container():
                    st.success(f'{df.index[si_idx]} : the model predicted {pred}')
                    # with cm_cont.container():
                    sct = px.scatter(df_u.iloc[:si_idx], x=df.columns[0], y=df.columns[1], color='predictions',
                                     symbol='predictions')
                    sct.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
                    sct.update_xaxes(visible=True, fixedrange=True)
                    sct.update_yaxes(visible=True, fixedrange=True)
                    sct.update_layout(annotations=[], overwrite=True)
                    st.write(sct)
                if event_log:
                    with log_cont.container():
                        st.code(''.join(['* ' + q + '\n' for idx, q in enumerate(log_str)]))
                    with pred_cont.container():
                        st.subheader('event log')
                        st.write('---------------------------------------------------------')
                        temp2 = predictions['predictions'].iloc[:si_idx]
                        fig3 = px.line(temp2, markers=True)
                        fig3.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
                        fig3.update_xaxes(visible=True, fixedrange=True)
                        fig3.update_yaxes(visible=True, fixedrange=True)
                        fig3.update_layout(annotations=[], overwrite=True)
                        st.write(fig3)