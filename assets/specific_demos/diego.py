import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px



def diego(diego_strem, stop_stream, files):
    st.title('Assembly Yield <--> Vendor Yield')
    st.text('learning and using the relationship between the Vendor Yield and the Assembly Yield')
    st.write('---------------------------------------------------------')

    df = files[0]
    with st.expander('curated data'):
        st.dataframe(df)

    with st.expander('relationship exploration'):
        col_list = []
        for col in df.columns.to_list():
            col_list.append(col)
        x_label, y_label, type_select = st.columns(3)
        x_col = x_label.selectbox("Select X axis", col_list)
        y_col = y_label.multiselect("Select y axis", col_list)

        if x_col != None and y_col != None:
            chart_type = type_select.selectbox("select chart type", ['line','bar','area'])
            y = [i for i in y_col]
            q = pd.DataFrame(df[y])
            q.index = df[x_col]
            if chart_type == 'line':
                st.line_chart(q, use_container_width=True)
            if chart_type == 'bar':
                st.bar_chart(q, use_container_width=True)
            if chart_type == 'area':
                st.area_chart(q, use_container_width=True)


    