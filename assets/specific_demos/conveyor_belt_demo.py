import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def cb_demo(si_stream, stop_stream, files):
    st.title('Video Based Object Detection')
    st.subheader('running detection on conveyor belt')
    st.write('---------------------------------------------------------')

    orange_video = files[0]

    st.video(orange_video)