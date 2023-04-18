import streamlit as st
import pandas as pd
import numpy as np
from assets.helpful_stuff.auxFunctions import *
import plotly.express as px
import streamlit.components.v1 as components


# constants
vanti_app_url = 'https://app.vanti.ai'
h2o_app_url = 'https://cloud.h2o.ai/apps/6ab8bf64-9bc5-4a68-9a7e-7251909c8d47'
window = 30
BASE_PERF = [0.88, 0.89]
GAMMA = BASE_PERF[0] - 0.25
BETA = 1 - GAMMA / (BASE_PERF[0])
# gamma = ()
VS = 0.01
# stream = False
nodes = ['anomaly remover', 'formatter', 'mini decision tree', 'local regression', 'local classifier', 'SVM',
         'perceptron', 'nan filler', 'normalizer', 'encoder', 'balancer']


def run_exp(up_file, dc_file, base_perf, beta, vs):
    # pl = st.empty()
    # pl2 = st.empty()
    # pca_plot = st.empty()

    event_log = []

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
                st.code(''.join(['* ' + q + '\n' for idx, q in enumerate(event_log)]))
                if j == 1:
                    # feed1.success('@index :: ' + str(j))
                    # feed2.success('all is good!')

                    event_log.append('@index :: ' + str(j))
                    event_log.append('all is good!')
                    event_log.append(' ')
                if not drop and h20_val < 0.7 and j > 100:
                    # feed1.success('@index :: ' + str(j))
                    # feed1.error('alert')
                    # feed1.info('notice')
                    # feed2.error('drift detected! - 3 missing columns')
                    # feed2.error('standard model accuracy -->  50%')
                    # feed2.info('Vanti: analyzing affected nodes')
                    # st.info('This is a purely informational message', icon="ℹ️")
                    drop = True
                    recovery = True
                    event_log.append('@index :: ' + str(j))
                    event_log.append('alert - drift detected! - 3 missing columns')
                    event_log.append('notice - standard model accuarcy --> 50%')
                    event_log.append('Vanti: analyzing affected nodes')
                    event_log.append(' ')

                if vanti_val > 0.7 and recovery and np.random.rand() < 0.1:
                    node = np.random.randint(0, 10, 1)[0]
                    node = nodes[node]
                    new_node = np.random.randint(0, 10, 1)[0]
                    new_node = nodes[new_node]
                    layer = np.random.randint(0, 10, 1)
                    # feed1.success('@index :: ' + str(j))
                    # feed1.info('notice')
                    # feed1.info('notice')
                    # feed2.success('updating Vanti')
                    # feed2.info('replacing node ' + str(node) + ' in layer ' + str(layer) + ' with ' + str(new_node))
                    # feed2.info('Vanti: accuracy restored to ' + str(np.round(vanti_val * 100)) + '%')
                    event_log.append('@index :: ' + str(j))
                    event_log.append('notice - updating Vanti')
                    event_log.append('replacing node ' + str(node) + ' in layer ' + str(layer) + ' with ' + str(new_node))
                    event_log.append('Vanti: accuracy restored to ' + str(np.round(vanti_val * 100)) + '%')
                    event_log.append(' ')


def adaptive_ai_demo(files):
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

    # with st.expander('6 easy steps'):
    #     st.title('6 easy steps')
    #     # st.image('as6 easy step copy.png')
    #     st.image('assets/helpful_stuff/6 easy step copy.png')

    # with st.expander('reach out to our CTO'):
    #     ro1, ro2 = st.columns(2)
    #     st.title('Reach out!')
    #     ro1.write("sub: [ADAPTIVE-AI DEMO] →")
    #     ro2.write("niro@vanti.ai")
    #     ro1.write('vanti.ai')
    #     ro2.write('app.vanti.ai')

    # with st.expander('How does adaptive AI work?'):
    #     st.title('Self Wiring Networks')
    #     st.image('assets/Images/ezgif.com-gif-maker (3).gif')
    #     st.image('assets/Images/ezgif.com-gif-maker (2).gif')

    # with st.expander('Visit Vanti.AI'):
    #     components.iframe('http://vanti.ai', height=900)
