import pandas as pd
import streamlit as st
import numpy as np
import time
import plotly.express as px



def video_assembly_app(assembly_stream, stop_stream, files):
    st.title('Defect Detection in Video Assembly')
    st.subheader('video based manual assembly defect detection')
    st.write('---------------------------------------------------------')
    event_log = []
    col1, col2 = st.columns((1, 4))

    with col1:
        st.write(' ')
    with col2:
        st.image('assets/Data/video_assembly/assembly-movie-small.gif', caption='assembly video')

    df = files[0]
    kpi_file = files[1]
    video_num = df.shape[0]
    metrics = col1.empty()
    # error_inv = st.empty()
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
    error_inv = st.empty()

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
                    event_log.append(f'Fail @ {str(df.index[j % video_num])} -- {df["reason"].iloc[j % video_num]}')
                    # st.code(''.join(['* ' + q + '\n' for idx, q in enumerate(event_log)]))
                    # feed1.error('FAIL @' + str(df.index[j % video_num]))
                    # feed2.info(df['reason'].iloc[j % video_num])
                st.code(''.join(['* ' + q + '\n' for idx, q in enumerate(event_log)]))

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
