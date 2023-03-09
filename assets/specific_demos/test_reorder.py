import pandas as pd
import streamlit as st
import numpy as np
import time
import plotly.express as px


def rt_test_reorder(test_order_stream, stop_stream, files):
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