import numpy as np
import streamlit as st
# from PIL import Image
import pandas as pd
import altair as alt

import calendar  # Core Python Module
from datetime import datetime  # Core Python Module

import plotly.graph_objects as go  # pip install plotly
import streamlit as st  # pip install streamlit
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu

import database as db  # local import
from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components
import webbrowser


@st.cache(allow_output_mutation=True)
def get_data():
    return []


vanti_app_url = 'https://app.vanti-analytics.com'


# -------------- SETTINGS --------------
incomes = ["Salary", "Blog", "Other Income"]
expenses = ["Rent", "Utilities", "Groceries", "Car", "Other Expenses", "Saving"]
currency = "USD"


page_title = "Vanti-DataFen"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"
# --------------------------------------

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)


# vanti_banner = Image.open('assets/Images/Vanti - Main Logo@4x copy.png')
# vanti_banner = Image.open('assets/Images/Vanti - Logo White L Green Icon & Dark Blue B@4x.png')
run_en = False

# st.image(vanti_banner)
st.title('Dataset Generator')

color_scale = alt.Scale(range=['#FAFA37', '#52de97', '#c9c9c9'])

st.markdown("""---""")





