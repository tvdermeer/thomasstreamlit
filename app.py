## list of general imports ##
import streamlit as st  #streamlit ofcourse
import base64           #download function
import pandas as pd     #dataframes and such
import numpy as np      #computing and stuffs
import torch

## model imports ##
from transformers import AutoModel, AutoTokenizer   #getting all the cool NLP models
from flair.embeddings import FlairEmbeddings, WordEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.data import Sentence

@st.cache
def createTransformerEmbeddings(modelName, data):

    embeddings = []

    model = AutoModel.from_pretrained(modelName)
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    sentences = data['Interest_Name'].values
    for sent in sentences:
        if __name__ == "__main__":
            input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0)
            #input_ids = tokenizer.encode(sent, add_special_tokens=True)
            #unsqueezed = torch.tensor(input_ids)
            #test = unsqueezed.unsqueeze(0)
            output = model(input_ids)
            final_output = output[0]
            resized =torch.reshape(final_output,(1,-1))
            array = resized.detach().numpy()
            embeddings.append(array)  
       
    return embeddings

@st.cache
def createFlairEmbeddings(embedding_list, data):

    embeddings = []

    sentences = data['Interest_Name'].values

    model = DocumentPoolEmbeddings(embedding_list, fine_tune_mode='nonlinear')
    if __name__ == "__main__":
        for sent in sentences:
            sentence = Sentence(sent)
            model.embed(sentence)
            modeled_embedding = sentence.get_embedding()
            array = modeled_embedding.cpu().detach().numpy()
            embeddings.append(array)

    return embeddings

def createDataFrame(embeddings, data):
    sentences = data['Interest_Name']
    sentences_df = pd.DataFrame(sentences)
    embeddings_df = pd.DataFrame(embeddings)
    dataframe = sentences_df.join(embeddings_df)

    return dataframe


## the actual app ##
st.markdown('# embeddings maken #')
st.markdown('__testversie__ 0.1')

st.markdown('deze app is bedoeld voor het omzetten van data als platte tekst naar embeddings :1234:, die later kunnen worden gebruikt om de echte analyses mee te doen :sunglasses:. dit gedeelte bestaat uit ongeveer de volgende stappen.  \n1. upload je data in het uploadveld :writing_hand:.  \n2. kies hoe jouw data eruit moet zien; alleen een interessenaam of ook de beschrijving erbij.  \n3. kies welk model je wilt gebruiken voor het creeëren van de embeddings.  \n4. Download de embeddings. :+1:')
## file uploader ##
st.markdown(' ### upload hieronder je .csv bestand ###')
uploaded_file = st.file_uploader('',type="csv")
if uploaded_file is not None:
    with st.spinner('data verwerken'):
            data = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1', delimiter=';')
    st.dataframe(data)


### choice of only interest names or also the analysis of the other text fields
### TODO: concatenating the different columns 

st.markdown('### Wat wil je analyseren? Kies een van de onderstaande mogelijkheden ###')

text_choice = st.radio(label='', options=['Interest_Name', 
                            'Interest_Name + Comment1',
                            'Interest_Name + Comment2', 
                            'Interest_Name + Comment1 & 2'])

if text_choice == 'Interest_Name': 
    text_input = data["Interest_Name"]
    st.write(text_input)

if text_choice == 'Interest_Name + Comment1':
    
   #text_input = data['combined'] = data.iloc[['Interest_Name', 'Comment1']].agg('.'.join, axis=1) #data['Interest_Name'].astype(str) + data['Comment1'].astype(str)   
    #data.iloc[['Interest_Name'.astype(str), 'Comment1'.astype(str)]].agg('.'.join, axis=1)
    st.write(text_input)



## show and interact with the data TODO
        # - choice a number of (random?) examples from the text

#### model selection ####
st.markdown('### Selecteer welk model je wilt gebruiken om embeddings te maken ###')

model_choice = st.radio(label='', options=['FastText (woord)', 
                            'Flair (karakter)',
                            'RobBERT (zin (RoBERTa))', 
                            'BERTje (zin, (BERT))'])

start_computation = st.button(label= 'start berekening')

if model_choice == 'FastText (woord)' and start_computation == True:
    with st.spinner('berekenen..'):
        fastext_embedding = WordEmbeddings('nl')
        st.write('model geladen, nu nog de embeddings...')
        embedding_list = [fastext_embedding]
        embeddings = createFlairEmbeddings(embedding_list, data)
        st.write('embeddings binnen, nu mooi maken') 
        dataframe = createDataFrame(embeddings, data)               
        st.write(dataframe)
        csv = dataframe.to_csv(sep=';')
        st.success('Het is gelukt!')

if model_choice == 'Flair (karakter)' and start_computation == True:

    flair_forward = FlairEmbeddings('nl-forward')
    flair_backward = FlairEmbeddings('nl-backward')
    embedding_list = [flair_forward, flair_backward]
    embeddings = createFlairEmbeddings(embedding_list, data) 
    dataframe = createDataFrame(embeddings, data) 
    st.write(dataframe)              
    csv = dataframe.to_csv(sep=';')   

if model_choice == 'RobBERT (zin (RoBERTa))' and start_computation == True:
    modelName ='pdelobelle/robBERT-base'

   
    embeddings =createTransformerEmbeddings(modelName, data)
    
    st.write(embeddings[0])
    print(embeddings[0][0][0])
    list_embeddings = []
    for i in embeddings: 
        list_embeddings.append(embeddings[i][0])
        
    #embeddings = embeddings[0]
    dataframe = createDataFrame(list_embeddings, data)
    st.write(dataframe)
    st.write('er zijn in totaal ' + str(dataframe.count()[0]) + ' embeddings gemaakt')
    st.balloons()

    st.write(dataframe)

    csv = dataframe.to_csv(sep=';')


if model_choice == 'BERTje (zin, (BERT))'  and start_computation == True:
    modelName ='bert-base-dutch-cased'
    
    embeddings = createTransformerEmbeddings(modelName, data)
    dataframe = createDataFrame(embeddings, data)

    st.write('er zijn in totaal ' + str(dataframe.count()[0]) + ' embeddings gemaakt')
    st.balloons()
    csv = dataframe.to_csv(sep=';')

## create a choice for PCA plot or a similarity score 
## TODO choose UMAP and for clustering HDBSCAN

## divergence score for a dataset?

## download function

if csv is not None: 
    st.markdown('## Download hier nu je gemaakte embeddings ##')
    st.markdown(' _let op!_ Klik met je rechtermuistoets op \' download csv file \' en klik daarna op opslaan als. Geef het hier een naam zoals \' embedding.csv \' hierna is het opgeslagen en kan je het gebruiken voor de volgende stap! :monocle: ')
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)
else: st.write('hier kan je wat downloaden als het klaarstaat')

