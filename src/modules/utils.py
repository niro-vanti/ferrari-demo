import os
import pandas as pd
import streamlit as st
from io import StringIO
# import json
# from json2table import convert

from src.modules.chatbot import Chatbot_txt, Chatbot, Chatbot_ledger
from src.modules.embedder import Embedder_txt, Embedder


def ledger_to_dataframe(df_d):
    # st.write(ledger_csv_path)
    # df_d = pd.read_csv(ledger_csv_path)
    data_string = df_d.iloc[0]['fullLedger'][1:-1]
    temp = data_string
    temp = temp.replace("{\"date\":{\"$", '\"').replace("}", "")
    result = dict((a.strip(), b.strip())
                  for a, b in (element.split(':', 1)
                               for element in temp.split(',')))
    columns = [i.replace("\"", '') for i in list(result.keys())]

    row_count = 0
    for idx, element in enumerate(temp.split(',')):
        q = element.split(':')
        command = q[0]
        if command == '"date"':
            row_count = row_count + 1

    out = pd.DataFrame(columns=columns, index=range(row_count))

    row = -1

    for idx, element in enumerate(temp.split(',')):
        q = element.split(':')
        command = q[0].replace("\"", '')
        if len(q) > 2:
            value = ''.join(q[1:])
        else:
            value = q[1]
            try:
                value = float(value)
            except:
                value = value

        if command == 'date':
            #         print(row, command, value)
            row = row + 1
        out.iloc[row][command] = value
    out.index.name = 'transaction_id'
    return out


class Utilities:

    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or from the user's input
        and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Your OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
            )
            if user_api_key:
                st.sidebar.success("API key loaded", icon="🚀")
        return user_api_key

    @staticmethod
    def handle_upload_txt():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type="txt", label_visibility="collapsed")
        if uploaded_file is not None:

            def show_user_file(uploaded_file):
                file_container = st.expander("Your TXT file :")
                uploaded_file_content = StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = uploaded_file_content.read()
                file_container.write(string_data)

                try:
                    dict1 = {}
                    dict1 = json.loads(string_data)
                    st.write(dict1)
                    # creating dictionary
                    # st.write(string_data)
                    # for line in string_data:
                    #     st.write(line)
                    # with open(uploaded_file) as fh:
                    #
                    #     a = 1
                    #     for line in fh:
                    #         command, description = line.strip().split(None, 1)
                    #         dict1[command] = description.strip()
                    # file_container.write(dict1)

                    # # creating json file
                    # # the JSON file is named as test1
                    # out_file = open("test1.json", "w")
                    # json.dump(dict1, out_file, indent=4, sort_keys=False)
                    # out_file.close()
                    # #
                    # # # first load the json file
                    # file_path = 'test1.json'
                    # with open(file_path, 'r') as f:
                    #     data = json.load(f)
                    # df = pd.DataFrame(dict1)
                    df = pd.json_normalize(dict1, record_path=['date'])

                    st.DataFrame(df)
                    # build_direction = "TOP_TO_BOTTOM"
                    # table_attributes = {"style": "width:100%", "class": "table table-striped"}
                    # html = convert(dict1, build_direction=build_direction, table_attributes=table_attributes)
                    # st.markdown(html)
                except:
                    print('not json')
                    st.error('not a json')

            show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your TXT file to get started, "
                # "sample for try : [fishfry-locations.csv](https://drive.google.com/file/d/1TpP3thVnTcDO1_lGSh99EKH2iF3GDE7_/view?usp=sharing)"
            )
            st.session_state["reset_chat"] = True
        return uploaded_file

    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="collapsed")
        if uploaded_file is not None:

            def show_user_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                shows = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                file_container.write(shows)

            show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your CSV file to get started, "
                "sample for try : [fishfry-locations.csv](https://drive.google.com/file/d/1TpP3thVnTcDO1_lGSh99EKH2iF3GDE7_/view?usp=sharing)"
            )
            st.session_state["reset_chat"] = True
        return uploaded_file

    @staticmethod
    def handle_upload_ledger():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="collapsed")
        if uploaded_file is not None:

            def show_user_file(uploaded_file):
                file_container = st.expander("Your Ledger :")
                shows = pd.read_csv(uploaded_file)
                out = ledger_to_dataframe(shows)
                out.to_csv('ledger.csv')
                uploaded_file.seek(0)
                file_container.write(out)

            show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your CSV file to get started, "
                "sample for try : [fishfry-locations.csv](https://drive.google.com/file/d/1TpP3thVnTcDO1_lGSh99EKH2iF3GDE7_/view?usp=sharing)"
            )
            st.session_state["reset_chat"] = True
        return uploaded_file

    @staticmethod
    def setup_chatbot_txt(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder_txt()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True
        return chatbot

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeds = Embedder_txt()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            chatbot = Chatbot(model, temperature, vectors)
        st.session_state["ready"] = True
        return chatbot

    @staticmethod
    def setup_chatbot_ledger(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        # embeds = Embedder()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            shows = pd.read_csv(uploaded_file)
            out = ledger_to_dataframe(shows)
            out.to_csv('ledger.csv')
            # file = uploaded_file.read()
            # vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            chatbot = Chatbot_ledger(model, temperature, 'ledger.csv')
        st.session_state["ready"] = True
        return chatbot
