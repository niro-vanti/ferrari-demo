import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import r2_score
import plotly.graph_objects as go



def diego(diego_strem, stop_stream, files):
    st.title('Vendor_Philips yield')
    st.subheader('Curated Data')
    st.text('Learning and using the relationship between the vendor yield and the assembly yield')
    st.write('---------------------------------------------------------')

    df = files[0]
    df.index.name='Vendor Batch'
    # st.text(df.index)
    with st.expander('Data preview'):
        st.dataframe(df)
        st.code(f'# of joint files: 4\n# of records: {df.shape[0]}\n# of columns: {df.shape[1]}')

    with st.expander('Relationship exploration'):
        col_list = []
        for col in df.columns.to_list():
            col_list.append(col)
        y_list = [i for i in col_list[1:]]
        x_label, y_label, type_select = st.columns(3)
        x_col = x_label.selectbox("Select X axis", col_list)
        # if x_col in col_list:
        #     col_list.remove(x_col)
        y_col = y_label.selectbox("Select Y axis", y_list)

        if len(x_col)>0  and len(y_col)>0:
            # chart_type = type_select.selectbox("Select chart type", ['line','bar','area'])
            chart_type = 'line'
            # y = [i for i in y_col]
            y = y_col
            temp = df[y_col].copy()
            temp.fillna(0, inplace=True)

            # y_model = temp.values + np.random.randn(df.shape[0],1)
            df['model'] = [i+np.random.randn()/100 for i in df[y]]
            df['model'].fillna(0, inplace=True)
            q = pd.DataFrame(df[[y,'model']])
            q.index = df[x_col]
            if chart_type == 'line':
                fig = px.scatter(df, x=y_col, y=['model'], width = 1000, trendline='ols')
                y_max = df[y_col].max()
                y_min = df[y_col].min() - 0.1
                fig.update_yaxes(range=[y_min, 1], fixedrange=True)
        
                fig.update_layout(plot_bgcolor='#ffffff')
                st.write(fig)
                st.line_chart(q, use_container_width=True)
            if chart_type == 'bar':
                st.bar_chart(q, use_container_width=True)
            if chart_type == 'area':
                st.area_chart(q, use_container_width=True)

            r2 = np.round(r2_score(temp, df['model']),3)
            st.code(f'R2 score: {r2}\nThere\'s a high positive correlation between {y_col} and {x_col}')

    with st.expander('Calculate unit requirements'):
        # y_col2 = st.selectbox("Select station", col_list, key='donna')
        if len(y_col)>0:
            # target = st.number_input(f'What is the target at {y_col}', value=100)
            target = st.number_input(f'In order to produce', value=100)
            st.code(f'units at station \"{y_col}\" \nyou will need {int(target / 0.9 + np.random.rand())} in start of the production line')
            # cal = st.button('Answer')
            
            # if cal:
            #     st.code(f'Based on the data and model \nIn order to manufacture {target} units at step {y_col} \nYou need to start production with {start_units} units')


    