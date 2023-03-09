# from sklearn.metrics import accuracy_score
import streamlit.components.v1 as components
# import time
# import plotly.express as px
# import os
import toml
# from assets.helpful_stuff.auxFunctions import *
import plotly.graph_objects as go
from assets.apps import *
from assets.specific_demos.cpc_demo import cpc
from assets.specific_demos.standard_industries_demo import si_demo
from assets.specific_demos.adaptive_ai_demo import adaptive_ai_demo
from assets.specific_demos.video_assembly import video_assembly_app
from assets.specific_demos.test_reorder import rt_test_reorder


page_title = "Vanti Apps"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/

st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

primaryColor = toml.load(".streamlit/config.toml")['theme']['primaryColor']
style_description = f"""
<style>
div.stButton > button:first-child {{ border: 2px solid {primaryColor}; border-radius:10px 10px 10px 10px; }}
div.stButton > button:hover {{ background-color: {primaryColor}; color:#000000;}}
footer {{ visibility: hidden;}}
# header {{ visibility: hidden;}}
<style>
"""
st.markdown(style_description, unsafe_allow_html=True)


def packages(package_stream):
    st.title('Packaging Inspection')
    st.subheader('image based visual defect detection in packages')
    st.write('---------------------------------------------------------')

    col1, col2 = st.columns((2, 2))
    image_cont = col1.empty()
    class_cont = col2.empty()
    seen_cont = st.empty()

    runner, names, classes, seen_names, seen_class = [], [], [], [], []

    for folder in os.listdir(os.path.join('assets', 'Data', 'packages')):
        if "." not in folder:
            for sub_folder in os.listdir(os.path.join('assets', 'Data', 'packages', folder)):
                if "." not in sub_folder:
                    for file in os.listdir(os.path.join('assets', 'Data', 'packages', folder, sub_folder)):
                        names.append(os.path.join('assets', 'Data', 'packages', folder, sub_folder, file))
                        classes.append(f'{folder}_{sub_folder}')

    names_len = len(names)

    if package_stream:

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


def ask_for_files(app_type_file):
    if app_type_file == 'paint shop visual inspection':
        return None
    if app_type_file == 'package visual inspection':
        return None
    if app_type_file == 'real time process optimization':
        df = pd.read_csv('assets/Data/test-reorder-data.csv')
        loaded_files = [df]
        return loaded_files
    if app_type_file == 'pre paint metal defects':
        return None
    if app_type_file == 'textile defects':
        return None

    if app_type_file == 'continuous process control demo':
        df = pd.read_csv('assets/Data/test-reorder-data.csv')
        df.columns = ['time', 'Env Temperature', 'H1 Pressure', 'H2 Pressure', 'M1 motor velocity', 'Valve Release']
        loaded_files = [df]
        return loaded_files

    if app_type_file == 'Ferrari paint shop defect detection':
        # df = pd.read_csv('assets/Data/Images/car-pano.png')
        return None
    if app_type_file == 'real-time sensor anomaly detection':
        df = pd.read_csv('assets/Data/anomaly.csv', index_col=0)
        batch = st.file_uploader("upload batch file")
        st.write(batch)
        if batch is not None:
            df = pd.read_csv(batch)

            if 'prog' in df.columns:
                df.drop(columns=['prog'], inplace=True)
                df = (df - df.mean()) / df.std()

            df['sen_alert'] = 0
            df['sit_alert'] = 0
        loaded_files = [df]
        return loaded_files
    if app_type_file == 'adaptive AI demo':
        data_file = st.file_uploader("upload `good' file", accept_multiple_files=False)
        dont_file = st.file_uploader("upload `drift' file", accept_multiple_files=False)
        if data_file is not None:
            uploaded_file_int = pd.read_csv(data_file)
        else:
            uploaded_file_int = pd.read_csv('assets/Data/adaptive-ai-demo-data.csv')
        if dont_file is not None:
            dont_care_int = pd.read_csv(dont_file)
        else:
            dont_care_int = pd.read_csv('assets/Data/adaptive-ai-demo-drifted.csv')

        loaded_files = [uploaded_file_int, dont_care_int]
        st.write(loaded_files)
        return loaded_files
    if app_type_file == 'medical device early fault detection':
        batch = st.file_uploader('upload medical device data', accept_multiple_files=False)
        fe = st.file_uploader('upload model feature importance', accept_multiple_files=False)
        if batch is not None:
            raw = pd.read_csv(batch)
        else:
            raw = pd.read_csv('assets/Data/medical-data.csv')
            fe = pd.read_csv('assets/Data/medical_device_feature_importance.csv', index_col=0)

        raw = raw.sample(frac=1).reset_index(drop=True)
        kpi = 'S_Scrap'
        kpi_col = raw[kpi].copy()
        df = raw.copy()
        df.drop(columns=[kpi], inplace=True)
        loaded_files = [df, kpi_col, fe]
        st.write(loaded_files)
        return loaded_files
    if app_type_file == 'manual assembly with video':
        batch = st.file_uploader('upload assembly videos', accept_multiple_files=False)
        if batch is not None:
            raw = pd.read_csv(batch)
        else:
            raw = pd.read_csv('assets/Data/flex-results.csv', index_col=0)

        df = raw.copy()
        kpi_col = df['result'].copy()
        df.drop(columns=['result'], inplace=True)
        loaded_files = [df, kpi_col]
        st.write(loaded_files)
        return loaded_files
    if app_type_file == 'Standard Industries Demo':
        batch = st.file_uploader('upload data file', accept_multiple_files=False)
        if batch is not None:
            raw = pd.read_csv(batch)
        else:
            raw = pd.read_csv('assets/Data/standard-inds/SI-results.csv', index_col=0)
        loaded_files = [raw,
                        pd.read_csv('assets/Data/standard-inds/top-10-feats.csv', index_col=0),
                        pd.read_csv('assets/Data/standard-inds/SI_feat_imp.csv', index_col=0)]
        return loaded_files
    if app_type_file == 'roadmap':
        return

    st.error('app type not supported')


# sidebar
app_list = ['paint shop visual inspection',
            'package visual inspection',
            'continuous process control demo',
            'textile defects',
            'Standard Industries Demo',
            'real time process optimization',
            'Ferrari paint shop defect detection',
            "pre paint metal defects",
            'real-time sensor anomaly detection',
            'adaptive AI demo',
            'manual assembly with video',
            'medical device early fault detection',
            'roadmap']

with st.sidebar:
    st.image('assets/Images/Vanti - Main Logo@4x copy.png')
    app_type = st.selectbox('select application', app_list)

    b1, b2 = st.columns(2)

    stream = b1.button('Start')
    stop_stream = b2.button('Stop')
    token = st.text_input('Vanti Model id', "####-production")

    connect = st.button('connect')
    if connect:
        for i in range(10000000):
            a = 1
        st.success('connected to to model')

    files = ask_for_files(app_type)

# main loop

# tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])
if app_type == 'paint shop visual inspection':
    # paint_defects(stream)
    visual_inspection_app(stream, stop_stream,
                          'Automotive Paint Shop',
                          'image based visual defect detection in paint',
                          'paint_photos')

if app_type == 'package visual inspection':
    packages(stream)

if app_type == 'continuous process control demo':
    cpc(stream, stop_stream, files)

if app_type == 'Standard Industries Demo':
    si_demo(stream, stop_stream, files)

if app_type == 'real time process optimization':
    rt_test_reorder(stream, stop_stream, files)

if app_type == 'Ferrari paint shop defect detection':
    # paint_shop_app(stream)
    visual_inspection_app(stream, stop_stream,
                          'In-line Paint Shop Defect Detection',
                          'image based paint defect detection in automotive assembly',
                          'ferrari',
                          header_image='assets/Images/ferrari-cropped.png',
                          moving_thumb=True,
                          scan_mode=True)

if app_type == 'textile defects':
    # textile_app(stream)
    visual_inspection_app(stream, stop_stream,
                          'Textile Defects',
                          'image based visual defect detection in textile',
                          'textile-data')

if app_type == 'real-time sensor anomaly detection':
    # rt_sensors_app(stream)
    ts_app(stream, stop_stream, files,
           'Real Time Anomaly Detection',
           'sensor based real time anomaly detection')

if app_type == 'adaptive AI demo':
    adaptive_ai_demo(files)

if app_type == 'manual assembly with video':
    video_assembly_app(stream, stop_stream, files)

if app_type == 'medical device early fault detection':
    # medical_device_app(stream)
    ts_app(stream, stop_stream, files,
           'Real Time Early Fault Detection',
           'tabular based real time early fault detection',
           classification=True)

if app_type == 'pre paint metal defects':
    # pre_paint_app(stream)
    visual_inspection_app(stream, stop_stream,
                          'Pre Paint Metal Defects',
                          'image based visual defect detection on pre paint metal automotive parts',
                          'paint-data')

if app_type == 'roadmap':
    roadmap()
