import streamlit.components.v1 as components
import time
import plotly.express as px
import os
import streamlit as st
import numpy as np
import pandas as pd


def visual_inspection_app(stream, stop_stream, title, subheader, folder_name):
    st.title(title)
    st.subheader(subheader)
    st.write('---------------------------------------------------------')

    col1, col2 = st.columns((2, 2))
    image_cont = col1.empty()
    class_cont = col2.empty()
    seen_cont = st.empty()

    runner, names, classes, seen_names, seen_class = [], [], [], [], []

    for folder in os.listdir(os.path.join('assets', 'Data', folder_name)):
        if "." not in folder:
            for file in os.listdir(os.path.join('assets', 'Data', folder_name, folder)):
                names.append(os.path.join('assets', 'Data', folder_name, folder, file))
                classes.append(f'{folder}')

    names_len = len(names)

    if stream:

        for j in range(names_len * 10):
            if stop_stream:
                break

            k = np.random.randint(0, names_len - 1, 1)[0]
            runner.append(classes[k % names_len])
            q = pd.DataFrame(runner)

            v = q[0].value_counts(normalize=False)
            v = v.reset_index(level=0)
            v.columns = ['class', 'count']

            with image_cont.container():
                time.sleep(1)
                st.image(names[k % names_len], use_column_width=True)  # , caption = names[i%N])
                seen_names.append(names[k % names_len])
                seen_class.append(classes[k % names_len])
            with class_cont.container():
                conf = np.random.randint(85, 100, 1)[0]
                st.info(classes[k % names_len] + ' with ' + str(conf) + '% confidence')
                fig = px.bar(v, y='class', x='count',
                             color='count',
                             color_discrete_sequence=['#00818A', '#52DE97', '#395243', '#ff3c78', '#f3f4d1',
                                                      '#bada55'],
                             orientation='h')
                fig.update_layout(plot_bgcolor='#ffffff')
                fig.update_layout(width=500)
                st.write(fig)

            unique_list = (list(set(seen_class)))
            with seen_cont.container():
                for u in unique_list:
                    with st.expander(u):
                        g1, g2, g3, g4 = st.columns(4)
                        count = 0
                        for im in range(len(seen_names)):
                            if seen_class[im] == u:
                                if count == 0:
                                    g1.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                if count == 1:
                                    g2.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                if count == 2:
                                    g3.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                if count == 3:
                                    g4.image(seen_names[im], use_column_width=True, caption=u + "_" + str(im))
                                count = count + 1
                                count = count % 4
