##### PREPARATIONS

# libraries
import gc
import os
import pickle
import sys
import urllib.request
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import json

# custom libraries
sys.path.append('src')
from model import get_model
from tokenizer import get_tokenizer

# Load the configuration file
with open('../config.json') as config_file:
    config_model = json.load(config_file)
    
# download with progress bar
mybar = None
def show_progress(block_num, block_size, total_size):
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)
        
        
##### CONFIG

# page config
st.set_page_config(page_title            = "Detection of AI-generated essays", 
                   page_icon             = ":books:", 
                   layout                = "centered", 
                   initial_sidebar_state = "collapsed", 
                   menu_items            = None)


##### HEADER

# title
st.title('AI Generated text detection')

# image cover
image = Image.open(requests.get('https://i.postimg.cc/zfmZQFGy/human-vs-ai-essay.png', stream = True).raw)
st.image(image)

# description
st.write('This app uses deep learning to estimate the probability that an essay is generated by AI, like ChatGPT. Enter your text below, and we will run it through a deep learning model and display the result.')


##### PARAMETERS

# title
st.header('What is the probability that your essay is AI-generated?')



# input text
input_text = st.text_area('Which essay would you like to pass through the model?', 'Please enter the essay in this field.')


##### MODELING

# compute probability that text is generated by AI
if st.button('Compute prediction'):


    folder_path = 'outputs/'
    weight_path = 'https://github.com/Anas-Badaoui/Detect-AI-Generated-Essays/releases/download/v1.0/weights_finetuned_ep0'

    # download model weights
    if not os.path.isfile(folder_path + 'pytorch_model.bin'):
        with st.spinner('Downloading model weights. This is done once and can take a minute...'):
            urllib.request.urlretrieve(weight_path, folder_path + 'pytorch_model.bin', show_progress)

    # compute predictions
    with st.spinner('Computing prediction...'):

        # clear memory
        gc.collect()

        # initialize model
        model = get_model(config_model)
        model.eval()

        # load tokenizer
        tokenizer = get_tokenizer(config_model)

        # tokenize text
        text = tokenizer(text                  = input_text,
                         truncation            = True,
                         add_special_tokens    = True,
                         max_length            = config_model['max_len'],
                         padding               = False,
                         return_attention_mask = True,
                         return_tensors        = 'pt')
        inputs, masks = text['input_ids'], text['attention_mask']

        # clear memory
        del tokenizer, text, config_model
        gc.collect()

        # compute prediction
        if input_text != '':
            logits = model(inputs, masks)
            prediction = logits.sigmoid().cpu().data.numpy().squeeze()
            prediction = 100 * (prediction) # scale to [0,100]

        # clear memory
        del model, inputs, masks
        gc.collect()

        # print output
        st.metric('AI score:', '{:.2f}%'.format(prediction, 2))
        st.write('**Note:** AI scores are scaled to [0, 100%]. A higher score means that the essay is very likely generated by AI.')
        st.success('Success! Thanks for scoring your essay :)')


##### DOCUMENTATION

# header
st.header('More information')

# example texts
#with st.expander('Show example texts'):
#    st.table(pd.DataFrame({
#        'Text':  ['', '', ''],
#        'Score': ['', '', ''],
#    }))
    
 # models
with st.expander('Read about the model'):
    st.write("The transformer model used in this project is DeBERTa-v3-base from Microsoft. You can read more about this model [here](https://huggingface.co/microsoft/deberta-v3-base). Additionally, you can find the source code used to pre-train and fine-tune the model in my GitHub repository [here](https://github.com/Anas-Badaoui/Detect-AI-Generated-Essays).")
       
# metric
with st.expander('Read about the metric'):
    st.write("The readability metric is calculated on the basis of a Bradley-Terry analysis of more than 111,000 pairwise comparisons between excerpts. Teachers spanning grades 3-12 (a majority teaching between grades 6-10) served as the raters for these comparisons. The raw scores vary in [-4, 2] and are scaled to [0, 100%] for convenience. More details on the used reading complexity metric are available [here](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/240886).")
    
    
##### CONTACT

# header
st.header("Contact")

# profile links
st.write("[![Linkedin](https://img.shields.io/badge/-LinkedIn-306EA8?style=flat&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/anas-badaoui/)](https://www.linkedin.com/in/anas-badaoui/) [![GitHub](https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white&link=https://github.com/Anas-Badaoui)](https://github.com/Anas-Badaoui)")
# copyright
st.text("© 2024 Anas Badaoui")