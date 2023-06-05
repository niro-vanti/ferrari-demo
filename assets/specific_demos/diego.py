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
import plotly.figure_factory as ff

# from assets.specific_demos.bot import vanti_gpt




def diego(diego_strem, stop_stream, files):
    st.title('Vendor_Philips yield')
    st.subheader('Curated Data')
    st.text('Learning and using the relationship between the vendor yield and the assembly yield')
    st.write('---------------------------------------------------------')

    suplier = st.radio('Choose Supplier to analyze',['Amphenol','Ergo','Upload my own'], horizontal=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    r2_limit = 0.6
    enable = False

    if suplier == 'Amphenol':
        df = files[1]
        enable = True
        df.index.name='Vendor Batch'
        df.sort_index(ascending=True, inplace=True)
    if suplier == 'Ergo':
        df = files[0]
        enable = True
        
    if suplier == 'Upload my own':
        qq = st.file_uploader('Upload file',type='csv', accept_multiple_files=False)
        if qq is not None:
            df = pd.read_csv(qq, index_col=0)
            if df is not None:
                enable = True
    # df.sort_index(ascending=True, inplace=True)
    # with st.expander('GPT'):
    #     vanti_gpt

    if enable:
        df.index.name='Vendor Batch'
        df.index = df.index.astype('str')
        df.sort_index(ascending=True, inplace=True)
        with st.expander('Data preview'):
            if 'vendor_batch' in df.columns:
                df.drop(columns='vendor_batch', inplace=True)
            st.dataframe(df)
            st.code(f'# of joint files: 4\n# of records: {df.shape[0]}\n# of columns: {df.shape[1]}')
        
        
        with st.expander('Value anlyzer'):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            tt = df.select_dtypes(include=numerics)
            target = st.selectbox('Select value to analyze',tt.columns.to_list(), index=0)
            ss1, dc, ss2, dc2 = st.columns([4,1,4,1])
            ss1.code(f'{target} statistics:\nmean - {np.round(df[target].mean(),2)}\nstd - {np.round(df[target].std(),2)}\nmaximal value - {np.round(df[target].max(),2)}\nminimal value - {np.round(df[target].min(),2)}\nmissing values - {np.round(df[target].isna().sum(),2)}')
            target_min = df[target].min()
            target_max = df[target].max()
            target_mean = df[target].mean()
            local = df.copy()
            
            filter_type = ss2.radio('Select filter method',['Range','Cut Off'], index=0, horizontal=True)
            if filter_type == 'Range':
                show_range = ss2.slider(f'Select {target} range', min_value=0.0, value = (0.0,1.0), step=0.01)
                local = local[local[target]<=show_range[1]]
                local = local[local[target]>=show_range[0]]
            if filter_type == 'Cut Off':
                cut_off = ss2.number_input('Enter cut off', min_value=0)
                sort_direction = ss2.radio('Select range',['Top','Bottom'], index=0, horizontal=True)
                if sort_direction == 'Top':
                    local.sort_values(by=target, ascending=False, inplace=True)
                if sort_direction == 'Bottom':
                    local.sort_values(by=target, ascending=True, inplace=True)
                local = local.iloc[:cut_off]
            
            

            # st.write(show_range)
            if local.shape[0] > 1:
                s1, s2  = st.columns(2)
                fig2 = px.line(local, y=target,x=local.index)
                fig2.update_traces(line_color='#00818A')
                fig2.update_layout(plot_bgcolor="white")
                fig2.update_yaxes(automargin=True)
                fig2.update_xaxes(automargin=True)
                s1.write(fig2)
                fig33 = ff.create_distplot([local[target]], [target], bin_size=.01,
                                        curve_type='kde', # override default 'kde'
                                        colors=['#52DE97'])

                # Add title
                fig33.update_layout(title_text=f'Distribution of {target}')
                fig33.update_layout(plot_bgcolor="white")
                fig33.update_yaxes(automargin=True)
                fig33.update_xaxes(automargin=True)
                s2.write(fig33)
                st.code(f'# of rows: {local.shape[0]}')
                # local.sort_values(by=[target], ascending=False, inplace=True)
                st.dataframe(local[target])
            else:
                st.write('There are no results in your selection')
            # df_hist = pd.DataFrame({'bins':bins,'vals':hist_values})
            # hist = df[target].hist(bins=50)
            # s2.bar_chart(df_hist, x='bins', y='vals')
        
        with st.expander('Relationship exploration'):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            tt = df.select_dtypes(include=numerics)

            x_label, y_label, type_select = st.columns(3)
            x_col = x_label.selectbox("Select X axis", tt.columns.to_list(), index=0)
            y_col = y_label.selectbox("Select Y axis", tt.columns.to_list(), index=1)
            model_type = type_select.radio('Select Model Type',['1 to 1','Many to 1'], index=0, horizontal=True)

            if len(x_col)>0  and len(y_col)>0:
                y = y_col
                temp = df[y_col].copy()
                temp.fillna(0, inplace=True)
                Y = temp
                if model_type == 'Many to 1':
                    X = df.copy()
                    X.drop(columns=y_col, inplace=True)
                else:
                    X = df[x_col].copy()
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

                res = pd.DataFrame({y_col:y_test, 'model':regr.predict(X_test)})
                fig = px.scatter(res, x=y_col, y=['model'], width = 1000) #, trendline='ols')
                y_max = df[y_col].max() + 0.1
                y_min = df[y_col].min() - 0.1
                fig.update_yaxes(range=[y_min, y_max], fixedrange=True)
        
                fig.update_layout(plot_bgcolor='#ffffff')
                st.write('Visual Regression - Model prediction vs true values')
                st.write(fig) 
                

                # r2 = np.round(r2_score(temp, df['model']),3)
                r2 = np.round(r2_score(y_test, regr.predict(X_test)))
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


                    # col_list = df.columns.to_list()
                    col_list = X.columns.to_list()
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
        