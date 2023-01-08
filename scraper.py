import openai
import streamlit as st
import pyperclip

# Set the API key
openai.api_key = "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"

# Set the model
model = "text-davinci-002"

page_title = "SEO blog post generator"
page_icon = ":fax:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")


def copy_blog():
    pyperclip.copy(st.session_state.blog)

# @st.cache
def get_response(query, depth):
    # Set the prompt
    prompt = f'give me a list of the top {depth} trends in {query}?'
    # Generate a response
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=1024)
    # Print the response
    a = response.to_dict()
    seo_list = a['choices'][0]['text']

    # response = 3
    # seo_list = ['ai in manufacturing', 'ai in autonomous vehicles', 'leo messi']
    print(seo_list)
    # st.write(seo_list)
    st.session_state.seo_list = seo_list
    st.session_state.query = query
    st.session_state.depth = depth
    # st.write(st.session_state)
    return response, seo_list


# @st.cache
def write_blog(title, word_num, style):
    # set the prompt
    prompt = f'write a {word_num} word blog post about {title} in a {style} style'

    # generate a response
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=1024)

    # print response
    a = response.to_dict()
    blog = a['choices'][0]['text']
    st.session_state.blog = blog
    print(blog)
    print(type(blog))
    return blog


###################################################
# init session_state
if 'seo_list' not in st.session_state:
    st.session_state.seo_list = None
if 'depth' not in st.session_state:
    st.session_state.depth = 5
if 'word_num' not in st.session_state:
    st.session_state.word_num = 500
if 'query' not in st.session_state:
    st.session_state.query = None
if 'blog' not in st.session_state:
    st.session_state.blog = None
if 'style' not in st.session_state:
    st.session_state.style = None

st.title('Niro\'s SEO magic')
with st.sidebar:
    st.write('\n')
    st.write('this is simple.')
    st.write('enter a topic, and some seo optimized blog topics will appear.')
    st.write('then select a topic, style, and word count and generate your SEO optimized blog post. ')
    st.write('\n')

col1, col2 = st.columns(2)
query = col1.text_input('enter a topic')
# depth = st.slider('how many SEO words', 1, 10, step=1, key='depth')
depth = col1.number_input('how many SEO words', format='%d', step=1, min_value=1, max_value=10, key='depth')
style_options = ['thought leadership', 'informative', 'playful']
submit = col1.button('generate SEO optimized content topics', on_click=get_response, args=(query, depth))

if st.session_state.seo_list is not None:
    col1.write(st.session_state.seo_list)
    options = st.session_state.seo_list.split('. ')[1:]

    options = [i.split('\n')[0] for i in options]

    title = col1.selectbox('what would you like to write a blog about?', options=options)
    word_num = col1.number_input('how many words in the blog?', min_value=100, max_value=2500, step=100, key='word_num')
    style = col1.radio('what style should the blog be in?',
                       style_options,
                       key='sytle',
                       horizontal=True)
    gen_blog = st.button('generate blog!',
                         on_click=write_blog,
                         args=(title, word_num, style))
    if st.session_state.blog is not None:
        col2.subheader('here\'s your post:')
        col2.write(st.session_state.blog)
        col2.button('copy to clipboard', on_click=copy_blog)

hide_menu_style = """
        <style>
        # MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
