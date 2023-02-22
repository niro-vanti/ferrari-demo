import streamlit.components.v1 as components
import time
import plotly.express as px
import os
import streamlit as st
import numpy as np
import pandas as pd


def paint_shop_app(ferrari_stream):
    st.title('In-line Paint Shop Defect Detection')
    st.subheader('image based paint defect detection in automotive assembly')
    st.write('---------------------------------------------------------')
    st.image('assets/Images/ferrari-cropped.png')
    sbc1, sbc2 = st.columns(2)
    sensitivity = sbc1.slider('model sensitivity', 0, 100, 50)
    speed = sbc1.slider('select path size', 16, 64, 32)
    print(sensitivity, speed)

    with sbc2.expander('What is model sensitivity?'):
        st.write("_sensitivity 100 --> alert me on **everything**_")
        st.write("_sensitivity 0 --> alert me on **critical things only**_")
    with sbc2.expander('What is Patch Size?'):
        st.write(
            'Patch size is the # of pixels in each dimension that the images is broken down to for defect detections')
        st.write('a smaller patch will find smaller defects, but will take longer to run')
        st.write('a bigger patch will find bigger defects, but will take faster to run')

    # image_number = 146
    # zoom_names = []
    defect_list = {}

    alerts = pd.DataFrame()

    seen_names, seen_class = [], []

    for file in os.listdir('assets/Images'):
        if '_zoom' in file:
            it = file.split('_')[0]
            dif = file.split('_')[-1].split('.')[0]
            defect_list[it] = dif

    _, c2 = st.columns(2)

    pl = st.empty()
    # p2 = st.empty()
    image_number = 146
    # is_error = False
    seen_cont = st.empty()

    if ferrari_stream:
        for j in range(image_number):
            if stop_stream:
                # ferrari_stream = False
                break

            with pl.container():
                im_name = 'assets/Images/' + str(j % image_number) + '_rect.png'
                st.image(im_name)

                defect = defect_list[str(j % image_number)]
                # print(i, defect)

                seen_names.append('assets/Images/' + str(j % image_number) + '_rect_thumb.png')
                seen_class.append(defect)

                if defect == 'no-defect':
                    st.success('no defect!')
                    is_error = False
                else:
                    is_error = True
                    q = pd.DataFrame({
                        'section': [j % image_number],
                        'defect': [defect],
                    })
                    alerts = pd.concat([alerts, q], axis=0, ignore_index=True)

            unique_list = (list(set(seen_class)))

            with seen_cont.container():
                for u in unique_list:
                    with st.expander(u):
                        selected = []
                        for name, cls in zip(seen_names, seen_class):
                            if cls == u:
                                selected.append(name)
                        st.image(selected)
            if is_error:
                with st.expander(defect + '  ::  defect-alert zoom-in @ section' + str(j % image_number)):
                    st.image('assets/Images/' + str(j % image_number) + '_zoom_' + defect + '.png')


def visual_inspection_app(stream, stop_stream, title, subheader, folder_name,
                          header_image=None,
                          moving_thumb=None,
                          scan_mode=False):
    st.title(title)
    st.subheader(subheader)
    st.write('---------------------------------------------------------')
    if header_image is not None:
        with st.expander('full image'):
            st.image(header_image)
    if moving_thumb is not None:
        thumb_cont = st.empty()
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
            if scan_mode:
                k = j
            else:
                k = np.random.randint(0, names_len - 1, 1)[0]
            runner.append(classes[k % names_len])
            q = pd.DataFrame(runner)

            v = q[0].value_counts(normalize=False)
            v = v.reset_index(level=0)
            v.columns = ['class', 'count']
            if moving_thumb is not None:
                with thumb_cont.container():
                    zoom_img_name = names[k % names_len]
                    img_num = zoom_img_name.split('_')[0].split('/')[-1]
                    thumb_name = f'{img_num}_rect.png'
                    st.image(f'assets/Data/{folder_name}-thumb/{thumb_name}')

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
