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
import certifi
import ssl

# custom libraries
sys.path.append('src')
from model import get_model
from tokenizer import get_tokenizer

# Load the configuration file
with open('config.json') as config_file:
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
input_text = st.text_area('Which essay would you like to pass through the model?', 'Please enter the essay in this field. (We recommend at least 200 words)')


##### MODELING

@st.cache_data
def run_model(inputs, masks):
    return model(inputs, masks)

@st.cache_resource
def get_model_cached(config_model, path_to_model):
    return get_model(config_model, path_to_model)
    

# compute probability that text is generated by AI
if st.button('Compute prediction'):


    folder_path = 'outputs/'
    weight_path = 'https://github.com/Anas-Badaoui/Detect-AI-Generated-Essays/releases/download/v1.0/weights_finetuned_ep0'

    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # download model weights
    if not os.path.isfile(folder_path + 'pytorch_model.bin'):
        with st.spinner('Downloading model weights. This is done once and can take a minute...'):
            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations(certifi.where())
            with urllib.request.urlopen(weight_path, context=ssl_context) as u, open(folder_path + 'pytorch_model.bin', 'wb') as f:
                f.write(u.read())

    # compute predictions
    with st.spinner('Computing prediction...'):

        # clear memory
        gc.collect()

        # initialize model
        model = get_model_cached(config_model, folder_path + 'pytorch_model.bin')
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
            logits = run_model(inputs, masks)
            prediction = logits.sigmoid().cpu().data.numpy().squeeze()
            prediction = 100 * (prediction) # scale to [0,100]

        # clear memory
        del model, inputs, masks
        gc.collect()

        # print output
        st.metric('AI score:', '{:.2f}%'.format(prediction, 2))
        st.write('**Note:** AI scores are scaled to [0, 100%]. A higher score means that the essay is very likely generated by AI.')
        st.success('The estimation was computed successfully!')


##### DOCUMENTATION

# header
st.header('More information')

# example texts
with st.expander('Show example texts'):
    st.table(pd.DataFrame({
        'Text':  ["""Did you always wanted to know what it was like to be on a big boat? Well if it was you might be interested in this job. This job is called "Seagoing Cowboy". What is a Seagoing Cowboy you may ask? A Seagoing Cowboy is you guessed it, it's an animal taking care taker. The animals they care for are animals from countries that are affected by World War II or they were also affected by World War II.

I, Luke Bomberger, was a Seagoing Cowboy. I had just graduated from high school when I had gotten drafted. But I didn't have to go to the Army because my friend, Don Reist invited me to go to Europe with him. On his invition, I couldn't pass up it was an opportunity of a lifetime. In a way it was good because it helped me make new friends. The best part about being a Seagoing Cowboy was the trips around the world. I had the best of times when I had signed up for this job. On the return trips I played baseball, volleyball, and table-tennis tournaments.

I believe that this is a great job to have because it's a way to visit places you will never go to again, you get to see the whole world if you want to! You get to be an animal taker! And you get to go to Europe (or places similar to Europe) and China and Greece, and Crete, and just about anything. If I were you I'd take this job in heartbeat.

On my second trip to Europe, I had the opportunity to go up the narrow streets of Venice, Italy, with water only. And to take a ride on a gondola. There were a few occasions, when we got to have fun with the animals that were in that boat, the animals I was taking care off.

My favorite part of the job was when I took care of the horses. We also had to clean their stalls. Also, we had to fill food buckets with oats each day. I also had to take care of the animals to make sure they were safe and healthy.

Another good thing about that job and just being a Seagoing Cowboy is once you are done being a Seagoing Cowboy your life will be changed forever. Your life will be different if you are a Seagoing Cowboy. Your whole life will change for the better if you just take the chance to be one!

I say take the chance and be the new Seagoing Cowboy. Be the one that takes the change. Take the chance at a whole new life! Be one with the animals and make it the best experience you ever had. Be a Seagoing Cowboy today!!!!!!!!!!""",
"""NASA. The United States form of a dominating space academy. The Viking 1, was in fact NASA's spacecraft that was launched in 2001. The spacecraft was set course for the small, red planet that is simillar to Earth. The planet is called Mars. Viking 1's "sister" Viking 2 was supposed to be landing on Mars, and that's why Viking 1 was sent up in space to begin with. While Viking 1 was snapping photos, its camera picked up a weird face-like structure on the surface of Mars. Some believe its a mysterious face that "so called aliens" built. I believe that their reasoning is just a bunch of garbage. What I think that is, is just a simple and natural landform made by the underground volcanoes, sandy surface, and rocks that landed there.

One reason I believe that this is a natural landform is from looking at the photos. The "face" isn't any perticular shape that any man or alien would want. The landform would be shaped as a circle, square or any other simple shape, and it would not be so ridgid and cracked. If you were to look at the photo furthest to the left, you would assume that it resembled a face, but If you compair that photo to the one taken twenty-five years later, you would begin to understand that to photo was out-dated and that the camera was not as clear as the one taken later. In addition, why would an alien make a structure of a human's face.

That part of the theory makes more sense if you take into consideration the angle the camera could have been taken at. Notice photo two? Its seems to be taken at an angle where there looks like there is a small mountain range looming over the rest of the area. This is key evidence explaining how this is just a natural landmark on Mars.

If you still do not believe me, then this next part will. If you think about it, why would NASA keep an amazing discover like this a secret? The answer is that they would not. If NASA went worldwide with their "face" they would make more money than millionaires or even other space academies. It does not make sense that the academy would keep an alien encounter hidden.

In conclusion, NASA is either brave or very stupid not to tell people about their "face", or people just want to get their word out to make other people critisized, make other people feel stupid, or they just want to feel like they are correct when they are clearly not. Therefore; This "so called face" is not a face, but it is just a land mark that was naturally constructed by the elements and not alien life-forms."""],
        'Score': ['99.73%', '12.41%'],
    }))
    
 # models
with st.expander('Read about the model'):
    st.write("The transformer model used in this project is DeBERTa-v3-base from Microsoft. You can read more about this model [here](https://huggingface.co/microsoft/deberta-v3-base). Additionally, you can find the source code used to pre-train and fine-tune the model in my GitHub repository [here](https://github.com/Anas-Badaoui/Detect-AI-Generated-Essays).")
        
    
##### CONTACT

# header
st.header("Contact")

# profile links
st.write("[![Linkedin](https://img.shields.io/badge/-LinkedIn-306EA8?style=flat&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/anas-badaoui/)](https://www.linkedin.com/in/anas-badaoui/) [![GitHub](https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white&link=https://github.com/Anas-Badaoui)](https://github.com/Anas-Badaoui)")
# copyright
st.text("© 2024 Anas Badaoui")