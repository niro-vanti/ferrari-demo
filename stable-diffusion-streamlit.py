import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import streamlit as st
import os
from authtoken import auth_token

page_title = "Vanti-DataGen"
page_icon = ":money_with_wings:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"
# --------------------------------------

st.set_page_config(page_title=page_title, page_icon=page_icon)  # , layout=layout)
st.title(page_title + " " + page_icon)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"


def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: #00818A;
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        color: #52de97;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# path = 'assets/Images/pcb
def get_images_from_path(path):
    imgs = []
    # print(path)
    for file in os.listdir(path):
        # print(file)
        imgs.append(os.path.join(path, file))
    return imgs


with st.sidebar:
    col1, col2, col3 = st.columns(3)
    st.title('config')
    prompt = st.text_area('describe your image')
    img_num = st.number_input('enter number of images to generate', min_value=1)
    var = st.slider('variability', min_value=0.0, max_value=10.0, value=7.5, step=0.1)

    c1, c2 = st.columns(2)
    style = c1.radio('what is the image style?', ('x-ray', 'PCB', 'picasso', 'impressionsm'))

    defects = c2.multiselect('what defects to add to the image?',['scratch','hole','niro'])
    print(defects)

    prompt = prompt + ' ' + style
    for defect in defects:
        prompt = prompt + ' with ' + defect



    c1, c2 = st.columns(2)
    go_button = c1.button('generate')
    # with c2:
    #     st.button("show examples 👌", on_click=style_button_row, kwargs={
    #         'clicked_button_ix': 2, 'n_buttons': 4
    #     })
    stock_button = c2.button('show examples', on_click=style_button_row, kwargs={
        'clicked_button_ix': 1, 'n_buttons': 1
    })


def generate(img_name, var):
    with autocast(device):
        image = pipe(prompt, guidance_scale=var)["sample"][0]

    image.save(img_name)
    return image


if go_button:
    st.success(str(img_num) + ' of ' + prompt)
    # pipe = StableDiffusionPipeline.from_pretrained(modelid,
    #                                                revision="fp16",
    #                                                torch_dtype=torch.float16,
    #                                                use_auth_token=auth_token)
    imgs = []
    col1, col2 = st.columns(2)
    for i in range(img_num):
        col1.progress((i + 1) / img_num)
        col2.text('finished ' + str(i + 1) + ' of ' + str(img_num))
        print('running image', i)
        name = 'img_' + str(i) + '.png'
        imgs.append(name)
        generate(name, var=var)
    print('done gen')
    # st.snow()
    st.image(imgs,
             caption=["image"] * len(imgs),
             width=200)

if stock_button:
    st.success(str(img_num) + ' of ' + prompt)
    if style == 'x-ray':
        # def_img = ['xray_0.png', 'xray_1.png','xray_2.png', 'xray_3.png']
        def_img = get_images_from_path('assets/Images/xray')
        print(len(def_img))
        st.image(def_img, width=350, caption=['xray_image_'] * len(def_img))
    if style == 'PCB':
        # def_img = ['hr_0.png', 'hr_1.png', 'hr_2.png', 'hr_3.png']
        def_img = get_images_from_path('assets/Images/pcb')
        st.image(def_img, width=350, caption=['PCB_image_'] * len(def_img))
