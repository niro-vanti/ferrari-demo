import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import r2_score
import plotly.graph_objects as go
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype




def diego(diego_strem, stop_stream, files):
    st.title('Vendor_Philips yield')
    st.subheader('Curated Data')
    st.text('Learning and using the relationship between the vendor yield and the assembly yield')
    st.write('---------------------------------------------------------')


    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    r2_limit = 0.6



    df = files[0]
    df.index.name='Vendor Batch'
    with st.expander('Data preview'):
        data_source = st.radio('Select data source',['Use application data','Upload my own'])
        if data_source == 'Use application data':
            qqq = 1
        if data_source == 'Upload my own':
            qq = st.file_uploader('Upload file',type='csv', accept_multiple_files=False)
            if qq is not None:
                df = pd.read_csv(qq, index_col=0)
        if 'vendor_batch' in df.columns:
            df.drop(columns='vendor_batch', inplace=True)
        st.dataframe(df)
        st.code(f'# of joint files: 4\n# of records: {df.shape[0]}\n# of columns: {df.shape[1]}')
    with st.expander('Relationship exploration'):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        tt = df.select_dtypes(include=numerics)
        col_list = []
        for col in tt.columns.to_list():
            col_list.append(col)
        y_list = [i for i in col_list[1:]]
        x_label, y_label, type_select = st.columns(3)
        x_col = x_label.selectbox("Select X axis", col_list)
        y_col = y_label.selectbox("Select Y axis", y_list)

        if len(x_col)>0  and len(y_col)>0:
            y = y_col
            temp = df[y_col].copy()
            temp.fillna(0, inplace=True)
            Y = temp
            X = df.copy()
            X.drop(columns=y_col, inplace=True)
            X = pd.get_dummies(data=X)
            

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
            regr = LinearRegression()
 
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X)
            df['model'] = y_pred
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
            if r2 >r2_limit:
                st.code(f'R2 score: {r2}\nThere\'s a high positive correlation between {y_col} and {x_col}')
            else:
                st.code(f'R2 score: {r2}\nThere correlation between {y_col} and {x_col} is not significant enough to ask further questions')

    with st.expander('Calculator'):
        if len(y_col)>0:
            if r2 > r2_limit:
                V = []
                max_cols = 5
                col0, col1, col2, col3, col4 = st.columns(max_cols)
                cols = [col0, col1, col2, col3, col4]
                col_index = 0


                col_list = df.columns.to_list()
                for name in ['model',y_col]:
                    if name in col_list:
                        col_list.remove(name)
                new_data = pd.DataFrame(columns=col_list, index=[0])
                for idx, col in enumerate(col_list):
                    if is_numeric_dtype(df[col]):
                        v = cols[col_index].number_input(f'enter {col}', value=df[col].mean())
                    if is_string_dtype(df[col]):
                        v = cols[col_index].selectbox(f'enter {col}',df[col].unique())
                    new_data[col] = v
                    col_index += 1 
                    col_index = col_index % max_cols
                    V.append(v)

                new_data.fillna(0, inplace=True)
                new_data = pd.get_dummies(data=new_data)
                new_data = pd.concat([X, new_data], axis=0, join='outer', ignore_index=True)
                new_data = pd.DataFrame(new_data.iloc[-1]).T
                new_data.fillna(0, inplace=True)
                out = regr.predict(new_data)
                st.code(f'With these inputs:\n {y_col} = {out[0]}')
            else:
                st.code(f'The model has an R2 score of {r2} which is not high enough to be able to answer this question \nThe R2 limit is {r2_limit}')

        else:
            st.text('sdfsd')
        
    with st.expander('Calculate unit requirements'):
        if len(y_col)>0:
            if r2 > r2_limit:
                target = st.number_input(f'In order to produce', value=100)
                st.code(f'units at station \"{y_col}\" \nyou will need {int(target / 0.9 + np.random.rand())} in start of the production line')
            else:
                st.code(f'The model has an R2 score of {r2} which is not high enough to be able to answer this question \nThe R2 limit is {r2_limit}')
     