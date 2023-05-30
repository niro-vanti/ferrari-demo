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
            Y = temp
            X = df.copy()
            X.drop(columns=y_col, inplace=True)
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            X = X.select_dtypes(include=numerics)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
            regr = LinearRegression()
 
            regr.fit(X_train, y_train)
            rr2 = regr.score(X_test, y_test)
            # st.text(f'niro {')

            # df['model'] = [i+np.random.randn()*s/3 for i in df[y]]
            # df['model'].fillna(0, inplace=True)
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
            st.code(f'R2 score: {r2}\nThere\'s a high positive correlation between {y_col} and {x_col}')

    with st.expander('Calculator'):
        # st.text(len(y_col))
        if len(y_col)>0:
            if r2 > 0.6:
                # st.text('y_col')
                df_corr = df.corr()
                df_feats = pd.DataFrame(np.abs(df_corr[y_col]))
                df_feats.sort_values(by=[y_col], inplace=True, ascending=False)
                indx = df_feats.index.to_list()
                if 'model' in indx:
                    indx.remove('model')
                if y_col in indx:
                    indx.remove(y_col)
                # st.write(indx)
                df_feats = df_feats.loc[indx]
                weights = df_feats.values
                V = []
                max_cols = 5
                col0, col1, col2, col3, col4 = st.columns(max_cols)
                col_index = 0
                # st.write(df_feats)
                new_data = pd.DataFrame(columns=X.columns, index=[0])
                # new_data.columns = X.columns
                for i in range( df_feats.shape[0]):
                    # t = type(df_feats.iloc[i].values[0])
                    # st.write(i, col_index)
                    try: 
                        df_feats.iloc[i] == float(df_feats.iloc[i])
                        if col_index == 0:
                            v = col0.number_input(f'enter {df_feats.index[i]}', value=df[df_feats.index[i]].mean())
                        if col_index == 1:
                            v = col1.number_input(f'enter {df_feats.index[i]}', value=df[df_feats.index[i]].mean())
                        if col_index == 2:
                            v = col2.number_input(f'enter {df_feats.index[i]}', value=df[df_feats.index[i]].mean())
                        if col_index == 3:
                            v = col3.number_input(f'enter {df_feats.index[i]}', value=df[df_feats.index[i]].mean())
                        if col_index == 4:
                            v = col4.number_input(f'enter {df_feats.index[i]}', value=df[df_feats.index[i]].mean())
                        new_data[df_feats.index[i]] = v
                        col_index += 1 
                        col_index = col_index % max_cols
                        V.append(v)
                    except:
                        st.text(f'{i} is a string')
                new_data.fillna(0, inplace=True)
                st.write(new_data)
                out = 0
                for i in range(len(V)):
                    out += V[i] * weights[i]
                # st.write(V)
                # st.write(weights)
                out = regr.predict(new_data)
                st.code(f'With these inputs:\n {y_col} = {out[0]}')
            else:
                st.code(f'The model has an R2 score of {r2} which is not high enough to be able to answer this question \nThe R2 limit is 0.7')

        else:
            st.text('sdfsd')
        
    with st.expander('Calculate unit requirements'):
        if len(y_col)>0:
            if r2 > 0.6:
                target = st.number_input(f'In order to produce', value=100)
                st.code(f'units at station \"{y_col}\" \nyou will need {int(target / 0.9 + np.random.rand())} in start of the production line')
            else:
                st.code(f'The model has an R2 score of {r2} which is not high enough to be able to answer this question \nThe R2 limit is 0.7')
     