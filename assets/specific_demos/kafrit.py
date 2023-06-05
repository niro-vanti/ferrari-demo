import streamlit as st
import pandas as pd

def kafrit(stream, stop_stream, files):
    df = files[0]


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