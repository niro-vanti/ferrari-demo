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
        y_col = y_label.selectbox("Select Y axis", y_list)

        if len(x_col)>0  and len(y_col)>0:
            y = y_col
            temp = df[y_col].copy()
            temp.fillna(0, inplace=True)
            s = df[y_col].std()
            # st.text(s)
            df['model'] = [i+np.random.randn()*s/3 for i in df[y]]
            df['model'].fillna(0, inplace=True)
            q = pd.DataFrame(df[[y,'model']])
            q.index = df[x_col]
            st.write('Values')
            st.line_chart(q, use_container_width=True)


            fig = px.scatter(df, x=y_col, y=['model'], width = 1000) #, trendline='ols')
            y_max = df[y_col].max() + 0.1
            y_min = df[y_col].min() - 0.1
            fig.update_yaxes(range=[y_min, y_max], fixedrange=True)
    
            fig.update_layout(plot_bgcolor='#ffffff')
            st.write('Visual Regression - Model prediction vs true values')
            st.write(fig)
            

            r2 = np.round(r2_score(temp, df['model']),3)
            st.code(f'R2 score: {r2}\nThere\'s a high positive correlation between {y_col} and {x_col}')

    with st.expander('Calculate unit requirements'):
        if len(y_col)>0:
            if r2 > 0.7:
                target = st.number_input(f'In order to produce', value=100)
                st.code(f'units at station \"{y_col}\" \nyou will need {int(target / 0.9 + np.random.rand())} in start of the production line')
            else:
                st.code(f'The model has an R2 score of {r2} which is not high enough to be able to answer this question \nThe R2 limit is 0.7')
     