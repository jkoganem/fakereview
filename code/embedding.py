print("Importing packages")
import numpy as np
import pandas as pd
import torch
import nltk
import os.path

from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
print("Done")


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} to compute")

# Load sentence transformer
sentence_model_light = SentenceTransformer("all-MiniLM-L6-v2").to(device)
sentence_model = SentenceTransformer("all-mpnet-base-v2").to(device)


# Boolean for specifying model
LIGHT_MODEL = True

# Define function for transforming text
def get_sentence_embedding(text):
    if isinstance(text,str):
        if not text.strip(): 
            print("Attempted to get embedding for empty text.")
            return []
        if LIGHT_MODEL == True:
            return sentence_model_light.encode(text)
        if LIGHT_MODEL == False:
            return sentence_model.encode(text)
    
    elif isinstance(text,list):
        return [get_sentence_embedding(x) for x in text]

# Load our data
DATA_DIR = "../raw data/"
df = pd.read_csv(f"{DATA_DIR}combined_data.csv")

# Map each text entry to a tensor
print("Mapping text to tensors using SBERT all-MiniLM-L6-v2")
tqdm.pandas()
embeddings_light = df['Text'].progress_apply(get_sentence_embedding)

# Stick the tensors on the end of our dataframe
df['embedding_light']  = embeddings_light
print("Done")

print("Mapping text to tensors using SBERT all-mpnet-base-v2")
LIGHT_MODEL = False
embeddings_full = df['Text'].progress_apply(get_sentence_embedding)
# Stick the tensors on the end of our dataframe
df['embedding_full']  = embeddings_full
print("Done")

# Save the result
print("Saving embeddings")
df.to_parquet('embeddings-SBERT.parquet')
print("Done")
