import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px



def diego(diego_strem, stop_stream, files):
    st.title('Vendor_Philips yield')
    st.subheader('Curated Data')
    st.text('Learning and using the relationship between the vendor yield and the assembly yield')
    st.write('---------------------------------------------------------')

    df = files[0]
    df.index.name='Vendor Batch'
    # st.text(df.index)
    with st.expander('Data Preview'):
        st.dataframe(df)
        st.code(f'# of joint files: 4\n# of records: {df.shape[0]}\n# of columns: {df.shape[1]}')

    with st.expander('relationship exploration'):
        col_list = []
        for col in df.columns.to_list():
            col_list.append(col)
        x_label, y_label, type_select = st.columns(3)
        x_col = x_label.selectbox("Select X axis", col_list)
        y_col = y_label.multiselect("Select Y axis", col_list)

        if len(x_col)>0  and len(y_col)>0:
            chart_type = type_select.selectbox("Select chart type", ['line','bar','area'])
            y = [i for i in y_col]
            q = pd.DataFrame(df[y])
            q.index = df[x_col]
            if chart_type == 'line':
                st.line_chart(q, use_container_width=True)
            if chart_type == 'bar':
                st.bar_chart(q, use_container_width=True)
            if chart_type == 'area':
                st.area_chart(q, use_container_width=True)

    with st.expander('target modeling'):
        if len(x_col)>0  and len(y_col)>0:
            target = st.number_input(f'What is the target at {y_col[0]}', value=100)
            cal = st.button('answer')
            start_units = 134234
            if cal:
                st.code(f'Based on the data and model \nIn order to manufacture {target} units at step {y_col} \nYou need to start production with {start_units} units')


    