## list of general imports ##
import streamlit as st  #streamlit ofcourse
import base64           #download function
import pandas as pd     #dataframes and such
import numpy as np      #computing and stuffs

## the actual app ##
st.write('it works!')

## file uploader ##
uploaded_file = st.file_uploader('',type="csv")
if uploaded_file is not None:
    st.write(uploaded_file)


## show and interact with the data TODO
        # - choice a number of (random?) examples from the text
no_of_examples = st.sidebar.text_input("hoeveel voorbeelden om testen", None)
if no_of_examples is not None:
        ## change length of the dataset to desired number ##
        pass


## tokenize data with the tokenizer of choice, also choice? 
        # - cache the tokenizers
## imports for the language models ##
from flair.embeddings import FlairEmbeddings, WordEmbeddings 
from flair.data import Sentence

st.write('hieronder laten we het verschil zien tussen woord- en karaktermodellen')
empty = ''
user_input = st.text_input('vul hier een zin in', empty)
if user_input is not empty:
    ## initialize models for the embeddings
    flair_embedding_forward = FlairEmbeddings('nl-forward')
    word_embedding_fasttext = WordEmbeddings('nl')
    # tokenize the sentence
    sentence = Sentence(user_input)

    st.write('dit zijn de vectoren voor de woorden in de zin')
    # show the words and accompagnied embeddings
    st.write('karaktermodel')
    flair_embedding_forward.embed(sentence)
    for token in sentence:
        st.write(token)
        st.write(token.embedding)

    st.write('woordmodel')
    word_embedding_fasttext.embed(sentence)
    for token in sentence:
        st.write(token)
        st.write(token.embedding)


## create a choice for PCA plot or a similarity score 

## divergence score for a dataset?


## download function
csv = None
if csv is not None: 
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)
else: st.write('hier kan je wat downloaden als het klaarstaat')

