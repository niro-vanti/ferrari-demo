# from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components
# import time
# import plotly.express as px
# import os
import toml
# from assets.helpful_stuff.auxFunctions import *
import plotly.graph_objects as go
from assets.apps import *

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

def adaptive_ai_demo():
    st.title('Adaptive AI Demo')
    st.subheader('Run Experiment')
    st.write('---------------------------------------------------------')

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
    st.subheader('video based manual assembly defect detection')
    st.write('---------------------------------------------------------')

    col1, col2 = st.columns((1, 4))

    with col1:
        st.write(' ')
    with col2:
        st.image('assets/Data/assembly-movie-small.gif', caption='assembly video')

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

    if assembly_stream:
        feed1, feed2 = st.columns([1, 4])
        fail_counter = 0
        # i = 0

        for j in range(df.shape[0] * 5):
            if stop_stream:
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
    pred_col1, pred_col2 = st.columns(2)
    pred_cont = pred_col1.empty()
    gauge_cont = pred_col2.empty()
    log_cont = st.empty()
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
        fig3['data'][1]['line']['dash'] = 'dot'
        fig3['data'][0]['mode'] = 'markers'
        fig3['data'][0]['marker']['symbol'] = "x-thin-open"
        fig3['data'][1]['marker']['symbol'] = "octagon-open"
        fig3['data'][1]['marker']['size'] = 12
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


def cpc(cpc_stream):
    st.title('Continuous Process Control Demo')
    st.subheader('closed loop power consumption reduction in real-time')
    st.write('---------------------------------------------------------')
    df = files[0]
    df.set_index('time', drop=True, inplace=True)
    df = df.astype(np.int8)
    df['Valve 2 control'] = [np.random.choice([1, 2, 3, 4, 5], p=[0.005, 0.05, 1 - 2 * 0.05 - 2 * 0.005, 0.05, 0.005])
                             for _ in range(df.shape[0])]

    nominal = 60

    random_dict = {ii: np.random.choice([1, 2, 3, 4, 5]) for ii in range(df.shape[1])}
    df.replace(random_dict, inplace=True)

    col1, dummy, col2 = st.columns((4, 1, 2))
    metrics = dummy.empty()
    data_graph = col1.empty()
    list_cont = col2.empty()
    # tp_cont = st.empty()
    df_tp = pd.DataFrame()
    df_tp['nominal'] = [nominal for _ in range(df.shape[0])]
    df_tp['with Vanti'] = nominal
    df_tp.index = df.index

    repeat_factor = 1

    if cpc_stream:
        for jj in range(df.shape[0] * repeat_factor):

            j = jj % df.shape[0]

            if stop_stream:
                # test_order_stream = False
                break
            # optimized = nominal - np.random.randint(10, 15)
            optimized = nominal - int(np.random.normal(7, 1, 1)[0])
            tp = int((nominal / optimized - 1) * 100)
            df_tp['with Vanti'].iloc[jj] = tp
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
                    1: f' {np.round(np.random.randint(-500, -100) / 100, 2)}%',
                    2: f' {np.round(np.random.randint(-100, 0) / 100, 2)}%',
                    4: f' {np.round(np.random.randint(0, 100) / 100, 2)}%',
                    5: f' {np.round(np.random.randint(100, 500) / 100, 2)}%',
                    3: ' no change',
                }
                st.code(''.join(['* ' + q + instructions[local['order'][idx]] + '\n' for idx, q in
                                 enumerate(local.index.to_list())]))
        with st.expander('power consumption gains'):
            # with tp_cont.container():
            tp_fig = px.bar(df_tp['with Vanti'].iloc[sss:eee], color_discrete_sequence=["#52de97"])
            tp_fig.update_layout(plot_bgcolor='#ffffff', margin=dict(t=10, l=10, b=10, r=10))
            tp_fig.update_xaxes(visible=True, fixedrange=True)
            tp_fig.update_yaxes(visible=True, fixedrange=True, range=[0, 50])
            tp_fig.update_layout(annotations=[], overwrite=True)
            tp_fig.update_layout(
                xaxis_title="Date", yaxis_title="Gains"
            )
            st.write(tp_fig)
            # st.line_chart(df_tp)
        with st.expander('full data'):
            st.line_chart(df)
    return None


def roadmap():
    # source = 'https://sharing.clickup.com/5712158/b/h/5ea8y-1402/c3da18542cfe989'
    # source = 'https://sharing.clickup.com/5712158/l/h/5ea8y-2262/ec42799aa27a3e3'
    source = 'https://sharing.clickup.com/5712158/l/h/5ea8y-2362/621790f9c8c8d11'
    st.components.v1.iframe(src=source,
                            width=1500,
                            height=900,
                            scrolling=True)


def packages(package_stream):
    st.title('Packaging Inspection')
    st.subheader('image based visual defect detection in packages')
    st.write('---------------------------------------------------------')

    col1, col2 = st.columns((2, 2))
    image_cont = col1.empty()
    class_cont = col2.empty()
    seen_cont = st.empty()

    runner, names, classes, seen_names, seen_class = [], [], [], [], []

    for folder in os.listdir(os.path.join('assets', 'Data', 'packages')):
        if "." not in folder:
            for sub_folder in os.listdir(os.path.join('assets', 'Data', 'packages', folder)):
                if "." not in sub_folder:
                    for file in os.listdir(os.path.join('assets', 'Data', 'packages', folder, sub_folder)):
                        names.append(os.path.join('assets', 'Data', 'packages', folder, sub_folder, file))
                        classes.append(f'{folder}_{sub_folder}')

    names_len = len(names)

    if package_stream:

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


def ask_for_files(app_type_file):
    if app_type_file == 'paint shop visual inspection':
        return None
    if app_type_file == 'package visual inspection':
        return None
    if app_type_file == 'real time process optimization':
        df = pd.read_csv('assets/Data/test-reorder-data.csv')
        loaded_files = [df]
        return loaded_files
    if app_type_file == 'pre paint metal defects':
        return None
    if app_type_file == 'textile defects':
        return None

    if app_type_file == 'continuous process control demo':
        df = pd.read_csv('assets/Data/test-reorder-data.csv')
        df.columns = ['time', 'Env Temperature', 'H1 Pressure', 'H2 Pressure', 'M1 motor velocity', 'Valve Release']
        loaded_files = [df]
        return loaded_files

    if app_type_file == 'Ferrari paint shop defect detection':
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
            raw = pd.read_csv('assets/Data/standard-inds/SI-results.csv', index_col=0)
        loaded_files = [raw,
                        pd.read_csv('assets/Data/standard-inds/top-10-feats.csv', index_col=0),
                        pd.read_csv('assets/Data/standard-inds/SI_feat_imp.csv', index_col=0)]
        return loaded_files
    if app_type_file == 'roadmap':
        return

    st.error('app type not supported')


# sidebar
app_list = ['paint shop visual inspection',
            'package visual inspection',
            'continuous process control demo',
            'textile defects',
            'Standard Industries Demo',
            'real time process optimization',
            'Ferrari paint shop defect detection',
            "pre paint metal defects",
            'real-time sensor anomaly detection',
            'adaptive AI demo',
            'manual assembly with video',
            'medical device early fault detection',
            'roadmap']

with st.sidebar:
    st.image('assets/Images/Vanti - Main Logo@4x copy.png')
    app_type = st.selectbox('select application', app_list)

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
if app_type == 'paint shop visual inspection':
    # paint_defects(stream)
    visual_inspection_app(stream, stop_stream,
                          'Automotive Paint Shop',
                          'image based visual defect detection in paint',
                          'paint_photos')

if app_type == 'package visual inspection':
    packages(stream)

if app_type == 'continuous process control demo':
    cpc(stream)

if app_type == 'Standard Industries Demo':
    si_demo(stream)

if app_type == 'real time process optimization':
    rt_test_reorder(stream)

if app_type == 'Ferrari paint shop defect detection':
    # paint_shop_app(stream)
    visual_inspection_app(stream, stop_stream,
                          'In-line Paint Shop Defect Detection',
                          'image based paint defect detection in automotive assembly',
                          'ferrari',
                          header_image='assets/Images/ferrari-cropped.png',
                          moving_thumb=True,
                          scan_mode=True)

if app_type == 'textile defects':
    # textile_app(stream)
    visual_inspection_app(stream, stop_stream,
                          'Textile Defects',
                          'image based visual defect detection in textile',
                          'textile-data')

if app_type == 'real-time sensor anomaly detection':
    # rt_sensors_app(stream)
    ts_app(stream, stop_stream, files,
           'Real Time Anomaly Detection',
           'sensor based real time anomaly detection')

if app_type == 'adaptive AI demo':
    adaptive_ai_demo()

if app_type == 'manual assembly with video':
    video_assembly_app(stream)

if app_type == 'medical device early fault detection':
    # medical_device_app(stream)
    ts_app(stream, stop_stream, files,
           'Real Time Early Fault Detection',
           'tabular based real time early fault detection',
           classification=True)

if app_type == 'pre paint metal defects':
    # pre_paint_app(stream)
    visual_inspection_app(stream, stop_stream,
                          'Pre Paint Metal Defects',
                          'image based visual defect detection on pre paint metal automotive parts',
                          'paint-data')

if app_type == 'roadmap':
    roadmap()
