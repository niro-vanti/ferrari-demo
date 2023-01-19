import streamlit as st
import pandas as pd



# -----------
df = pd.read_csv('assets/Data/adaptive-ai-demo-data.csv')

# -----------
st.title('analyze data')

prompt = st.text_input('what\'s your question?')
st.write(prompt)
st.write(df)