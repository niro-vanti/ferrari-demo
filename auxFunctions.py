import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import webbrowser

from sklearn.decomposition import PCA


def my_pcs(df, n_comp):
    pca = PCA(n_components=n_comp, random_state=22)
    pca.fit(df)
    x = pca.transform(df)
    x = pd.DataFrame(x)
    x.index = df.index
    col_names = ['component_' + str(i) for i in range(x.shape[1])]
    x.columns = col_names
    return x


def get_pred(base_perf, beta, vs, thr1, gt, i):
    if i >= thr1:
        h_score = 0.5 + np.random.randn() * vs / 4
        exp_factor = (1 - beta * np.exp(-0.01 * (i - thr1)))
        noise_factor = np.random.randn() * vs
        sine_factor = np.sin((i - thr1) * 0.025) * 0.025 * 0
        v_score = base_perf[0] * exp_factor + sine_factor + noise_factor
    else:
        h_score = base_perf[1] + np.random.randn() * vs
        v_score = base_perf[0] + np.random.randn() * vs

    v_flip = np.random.rand()
    h_flip = np.random.rand()

    v_prediction = gt if v_flip <= v_score else 1 - gt
    h_prediction = gt if h_flip <= h_score else 1 - gt

    return h_score, v_score, v_prediction, h_prediction


def parse_files(up_file, dc_file):
    # df = pd.read_csv(up_file)
    # dc = pd.read_csv(dc_file)
    df = up_file
    dc = dc_file

    df_concat = pd.concat([df, dc], axis=0)

    thr1 = df.shape[0]
    thr2 = dc.shape[0]
    b1 = df.shape[1]
    b2 = dc.shape[1]
    b = [b1, b2]

    df_concat.index = [i for i in range(thr1 + thr2)]

    kpi = df.columns[-1]
    dic = {'Passed': 1, 'Failed': 0}
    inv_dic = {v: k for k, v in dic.items()}

    tar = df[kpi].copy()
    tar_concat = df_concat[kpi].copy()
    tar.replace(dic, inplace=True)
    tar_concat.replace(dic, inplace=True)

    return tar, tar_concat, df, df_concat, dc, kpi, thr1, thr2, b, dic, inv_dic


def calc_perf(df, name, window=50):
    if df.shape[0] < window:
        gt = df['Ground Truth']
        pr = df[name]
        acc = accuracy_score(gt, pr)
    else:
        gt = df['Ground Truth'].iloc[-window:]
        pr = df[name].iloc[-window:]
        acc = accuracy_score(gt, pr)
    return acc


def get_cols_diff(up_file, dc_file):
    cols1 = up_file.columns
    diff = []
    [diff.append(i) for i in cols1 if i not in dc_file.columns]
    # [st.write(i) for i in cols1 if i not in dc_file.columns]

    return diff


def get_reason_medical(df, ms, ss):
    ff = np.random.randint(1, 4, 1)[0]
    n = df.shape[1]
    cols = df.columns.to_list()
    phrases = []
    for j in range(ff):
        feat = cols[np.random.randint(0, n - 1, 1)[0]]
        s = " > " if np.random.rand() < 0.5 else " < "
        m = np.round(ms[feat] + np.random.randn(), 2)
        d = np.round(ss[feat], 2)
        if s == " > ":
            v = m + d
        else:
            v = m - d
        v = str(np.round(v, 2))
        phrases.append(feat + s + v)
    phrase = ' and '.join(phrases)

    return phrase


def models(vanti_app_url, h2o_app_url):
    st.subheader("models")
    # ap1, ap2 = st.beta_columns(2)

    with st.expander('build models'):
        cc1, cc2 = st.columns((2, 2))
        if cc1.button('app.vanti'):
            webbrowser.open_new_tab(vanti_app_url)
        if cc2.button('app competitor'):
            webbrowser.open_new_tab(h2o_app_url)
    z1, z2, z3 = st.columns((1, 1, 3))
    dc2 = z3.file_uploader('mojo-file', accept_multiple_files=False)

    dc1r = z1.radio('Vanti error handling', ['flip coin', "auto"])
    dc2r = z2.radio('Standard model error handling', ['flip coin', 'auto'])
    print(dc2, dc2r, dc1r)
    with z3.expander('what does error handling mean?'):
        st.write('traditional models throw an error when structural drifts occur. ')
        st.write('flip coin - instead of an error the model will return a random pass/fail')
        st.write('auto - the model returns is default answer of error / response')


def get_reason(reason_type):
    ind = np.random.randint(0, 2, 1)[0]
    sensor_reasons = [
        'Sensor value is ok, but trend is unlikely',
        'Sensor value is unlikely',
        'Sensor is showing a slow change over time that is indicative of an issue'
    ]

    situation_reasons = [
        'All sensors are ok but there''s an unlikely combination of values',
        '2 or more sensors are registering an unlikely value'
        ''
    ]

    if reason_type == 'sensor':
        return sensor_reasons[ind]
    if reason_type == 'situation':
        return situation_reasons[ind]
    return 'no clear cut root cause'


def highlight_survived(s):
    return ['color: #52de97'] * len(s) if s['alert type'] == 'Situation' else ['color: #000000'] * len(s)
