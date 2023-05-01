import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px



def cpc(cpc_stream, stop_stream, files):
    st.title('Continuous Process Control Demo')
    st.subheader('closed loop power consumption reduction in real-time')
    st.write('---------------------------------------------------------')
    df = files[0]
    df.set_index('time', drop=True, inplace=True)
    df = df.astype(np.int8)
    df['Valve 2 control'] = [np.random.choice([1, 2, 3, 4, 5], p=[0.005, 0.05, 1 - 2 * 0.05 - 2 * 0.005, 0.05, 0.005])
                             for _ in range(df.shape[0])]

    nominal = 60
    window = 50
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
                sss = max(0, eee-window)
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
            tp_fig.update_layout(xaxis_title="Date", yaxis_title="Gains")
            st.write(tp_fig)
            # st.line_chart(df_tp)
        with st.expander('full data'):
            st.line_chart(df)
    return None
