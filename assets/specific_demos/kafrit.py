import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import r2_score 
import plotly.graph_objects as go
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years
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
from xgboost import XGBRegressor

# from assets.specific_demos.bot import vanti_gpt


def get_date_columns(df):
    l = df.select_dtypes(include=['datetime64'])
    if l.shape[1] == 0:
        # st.write('niro')
        # st.write(df.index.dtype)
        try:
            
            vals = [i for i in df.index]
            v = pd.DataFrame(vals, columns=['date'])
            v['date'] = pd.to_datetime(v['date'])
            return v, 'index'
        except:
            print('fail date time')
            return l, None
    elif l.shape[1] == 1:
        return l, l.columns
    else: 
        return False, None

def stats_block(df, t, title=None):
    a = str(np.round(df[t].mean(),2))
    s = str(np.round(df[t].std(),2))
    m = str(np.round(df[t].min(),2))
    M = str(np.round(df[t].max(),2))
    v = str(np.round(df[t].isna().sum(),2))
    if title is not None:
        lines = [
            '*** '+title+' ***',
            'mean - '+a,
            'std - '+s, 
            'maximal value - '+M,
            'minimal value - '+m,
            'missing values - '+v
        ]


    else:
        lines = [
            'mean - '+a,
            'std - '+s, 
            'maximal value - '+M,
            'minimal value - '+m,
            'missing values - '+v
        ]
    out = '\n'.join(lines)
    return out


def feat_imp(df, model):
    res = pd.DataFrame({
        "importance":np.abs(model.feature_importances_)},
        index=df.columns)
    res.index.name='feature'
    res.sort_values(by='importance', ascending=False, inplace=True)
    # st.write(res)
    return res

def find_max(df, model, N=3):
    res = feat_imp(df, model)

    df_mean = df.iloc[:1]
    for col in df_mean.columns:
        df_mean[col] = df[col].median()
    df_mean.index = ["max"]
    max_val = np.inf * -1
    f_range = []
    span = 5
    N = min(10, df.shape[1]) # hard coded for now
    top_feats = res.index[:N]
    for f in top_feats:
        n = df[f].nunique()
        v = df[f].unique()
        if n < 10:
            f_range.append([i for i in v])
        else:
            min_val = df[f].min()
            max_val = df[f].max()
            tick = (max_val - min_val)/span
            f_range.append( [i*tick  for i in range(span)] )
        # st.write(f'{f} -- {v}')
    # st.write(f_range)
    max_config = df_mean.copy()
    for v0 in f_range[0]:
        for v1 in f_range[1]:
            for v2 in f_range[2]:
                    for v3 in f_range[3]:
                        for v4 in f_range[4]:
                            for v5 in f_range[5]:
                                local = df_mean.copy()
                                local[top_feats[0]] = v0
                                local[top_feats[1]] = v1
                                local[top_feats[2]] = v2
                                local[top_feats[3]] = v3
                                local[top_feats[4]] = v4
                                local[top_feats[5]] = v5
                                pred = model.predict(local)
                                val = pred[0]
                                if val > max_val:
                                    print("found max")
                                    max_val = val
                                    max_config = local
    return np.round(max_val,2),max_config

def kafrit(diego_strem, stop_stream, files):
    st.title('Kafrit yield')
    st.subheader('Demo')
    st.text('Maximizing output based on live machine data')
    st.write('---------------------------------------------------------')

    suplier = st.radio('Choose Data to analyze',['Demo','Upload my own'], horizontal=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    r2_limit = 0.6
    enable = False
    comp_enable = False

    if suplier == 'Demo':
        df = files[0]
        enable = True
        df.index.name='Date_Time'
        df.sort_index(ascending=True, inplace=True)
        
    if suplier == 'Upload my own':
        qq = st.file_uploader('Upload file',type='csv', accept_multiple_files=False)
        if qq is not None:
            df = pd.read_csv(qq, index_col=0)
            if df is not None:
                enable = True


    if enable:
        df.index.name='Vendor Batch'
        df.index = df.index.astype('str')
        df.sort_index(ascending=True, inplace=True)
        with st.expander('Data preview'):
            # df_time, col_time = get_date_columns(df)
            st.dataframe(df)
            st.code(f'# of records: {df.shape[0]}\n# of columns: {df.shape[1]}')   
        
        with st.expander('Value anlyzer'):
            tt = df.select_dtypes(include=numerics)
            target = st.selectbox('Select value to analyze',tt.columns.to_list(), index=0)

            compare = st.file_uploader('Choose file for comparison',type='csv', accept_multiple_files=False)
            if compare is not None:
                df_comp = pd.read_csv(compare, index_col=0)
                cc = df_comp.select_dtypes(include=numerics)
                target_comp = st.selectbox('Select value to compare', cc.columns.to_list(), index=0)
                local_comp = df_comp.copy()
                comp_enable=True

            ss1, sb2, dc, ss2, dc2 = st.columns([2,2,1,4,1])
            if comp_enable:
                sb2.code(stats_block(df_comp,target_comp, title='Compared file'))
            ss1.code(stats_block(df,target, title='Original'))
                
            local = df.copy()
            
            
            filter_type = ss2.radio('Select filter method',['Range','Cut Off'], index=0, horizontal=True)
            if filter_type == 'Range':
                min_val = float(local[target].min())
                max_val = float(local[target].max())
                show_range = ss2.slider(f'Select {target} range', min_value=0.0, value = (min_val,max_val), step=0.01)
                local = local[local[target]<=show_range[1]]
                local = local[local[target]>=show_range[0]]
                if comp_enable:
                    local_comp = local_comp[local_comp[target_comp]<=show_range[1]]
                    local_comp = local_comp[local_comp[target_comp]>=show_range[0]]

            if filter_type == 'Cut Off':
                cut_off = ss2.number_input('Enter cut off', min_value=0, value=10)
                sort_direction = ss2.radio('Select range',['Top','Bottom'], index=0, horizontal=True)
                if sort_direction == 'Top':
                    local.sort_values(by=target, ascending=False, inplace=True)
                    if comp_enable:
                        local_comp.sort_values(by=target_comp, ascending=False, inplace=True)
                if sort_direction == 'Bottom':
                    local.sort_values(by=target, ascending=True, inplace=True)
                    if comp_enable:
                        local_comp.sort_values(by=target_comp, ascending=True, inplace=True)
                local = local.iloc[:cut_off]
                if comp_enable:
                    local_comp = local_comp.iloc[:cut_off]
                    

            # st.write(show_range)
            if local.shape[0] > 1:

                # regular value plots
                s1, s2  = st.columns(2)
                fig2 = px.line(local, y=target,x=local.index)
                fig2.update_traces(line_color='#00818A')
                fig2.update_layout(plot_bgcolor="white")
                fig2.update_yaxes(automargin=True)
                fig2.update_xaxes(automargin=True)
                if comp_enable: 
                    fig2.add_scatter(x=local_comp.index, y=local_comp[target_comp])
                s1.write(fig2)


                # histogram plots
                fig33 = ff.create_distplot([local[target]], [target], bin_size=.01,
                                        curve_type='kde', # override default 'kde'
                                        colors=['#52DE97'])

                # Add title
                fig33.update_layout(title_text=f'Distribution of {target}')
                fig33.update_layout(plot_bgcolor="white")
                fig33.update_yaxes(automargin=True)
                fig33.update_xaxes(automargin=True)

                if df[target].nunique() > 3:
                    s1.write('-------------------------')

                    x_max = local.copy()
                    x_max.drop(columns=target, inplace=True)
                    x_max = pd.get_dummies(x_max)
                    y_max = local[target].copy()

                    x_train, x_test, y_train, y_test = train_test_split(x_max, y_max, test_size = 0.25)
                    reg = XGBRegressor()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    r2 = np.round(r2_score(y_pred, y_test),2)
                    # st.write(r2)
                    res_max = pd.DataFrame(data={'original':local[target],'predicitons':y_pred}, index=x_test.index)
                    fig_max = px.line(res_max)
                    s1.code(f'We can use your data to predict {target} with R2 = {r2}')
                    s1.write(fig_max)

                    max_val, max_config = find_max(x_train, reg)
                    s1.code(f'Max Value = {max_val} \nthe configuration that will give you the max value is ')
                    s1.dataframe(max_config)
                    # s1.write(px.line(res))
                    

                if comp_enable:
                    x0 = local[target]
                    x1 = local_comp[target_comp]
                    hist_data = [x0,x1]
                    fig33 = ff.create_distplot(hist_data, [target, target_comp], bin_size=0.01,
                                               curve_type='kde',colors=['#52de97','#ff3c78'])
                s2.write(fig33)
                s2.write('-------------------------')

                # r1 = pd.DataFrame(data={'predictions':y_pred}, index=y_test.values)
                
                f2 = px.scatter(x=y_test, y=y_pred)
                f2.add_trace(go.Scatter(x=y_test, y=y_test))
                # fit_range = s2.slider(f'Select {target} fit range', min_value=0.0, value = (min_val,max_val), step=0.01, key='fit_range')
                s2.code(f'an R2 score >= {r2_limit} is considered enough to build a good model')
                s2.write(f2)
                res = feat_imp(x_train, reg)
                s2.code('feature importance')
                s2.write(px.bar(res))

                # info data frames
                if comp_enable:
                    cdf1, cdf2 = st.columns(2)
                    cdf1.write('original file')
                    cdf1.code(f'# of rows: {local.shape[0]}')
                    cdf1.dataframe(local[target])
                    cdf2.write('compared file')
                    cdf2.code(f'# of rows: {local_comp.shape[0]}')
                    cdf2.dataframe(local_comp[target_comp])
            else:
                st.write('There are no results in your selection')




        with st.expander('Relationship exploration'):
            
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
                # X = pd.get_dummies(data=X)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
                regr = XGBRegressor()
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
        