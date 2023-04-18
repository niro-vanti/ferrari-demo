import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import time


def harmonic_demo(stream, stop_stream, files):
    st.title('Process Calibration Demo')
    st.subheader('optimal configuration prediction')
    st.write('---------------------------------------------------------')

    df = files[0]
    raw = files[1]
    raw.drop(columns=['kpi'], inplace=True)
    joint = files[2]

    data_cont = st.empty()
    calib_cont = st.empty()
    log_cont = st.empty()
    events = []

    with st.expander('raw_data'):
        st.dataframe(joint)

    with st.expander('all plots'):
        st.line_chart(df)

    if stream:
        for idx in range(df.shape[0]):
            time.sleep(1)
            if stop_stream:
                break
            with data_cont.container():
                a = pd.DataFrame(raw.iloc[idx])
                st.dataframe(a.T)
                # st.table(raw.iloc[idx])
            with calib_cont.container():
                st.text(f'calibrating unit #{df.index[idx]}')
                y1 = df.iloc[idx]
                x = df.columns
                so_far = df.iloc[:idx]
                up = so_far.mean(axis=0)+so_far.std(axis=0)
                down = so_far.mean(axis=0)-so_far.std(axis=0)
                fig = go.Figure()
                fig.update_layout(
                    xaxis_title="Freq [MHz]",
                    yaxis_title="Return Loss [dB]",)
                fig.add_trace(go.Scatter(
                    x=x, y=y1,
                    line = dict(color='#52DE97', width=6),
                    # line_color='#52de97',
                    name='Unit Calibration',
                    # width=5
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=up,
                    line=dict(color="#00818A", dash='dot'),
                    # line_color = '#00818A',
                    name='typical upper value'
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=down,
                    line=dict(color="#00818A", dash='dot'),
                    # line_color = '#00818A',
                    name='typical lower value'
                ))
                fig.add_trace(go.Scatter(
                    x=x,
                    y=so_far.mean(axis=0),
                    line=dict(color='#ff3c78', dash='dot', width=1),
                    name='the optimal value'
                ))
                # fig.add_trace(go.Scatter(
                #     x=x,
                #     y=up+down-so_far.mean(axis=0),
                #     fill='toself'
                # ))
                fig.update_yaxes(range=[-50, -10])
                fig.update_layout(plot_bgcolor='#ffffff')
                fig.update_traces(mode='lines')

                st.plotly_chart(fig, use_container_width=True)
            with log_cont.container():
                up_diff = np.sum(y1 > up)
                down_diff = np.sum(y1 < down)
                limit = df.shape[1] * 0.5
                if (up_diff + down_diff) > limit:
                    events.append(f'unit {df.index[idx]} is significantly out of typical range\n')
                st.code(''.join(['* ' + q + '\n' for idx, q in enumerate(events)]))