from transformers import AutoModel, AutoTokenizer   #getting all the cool NLP models
import numpy as np 
import pandas as pd 

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