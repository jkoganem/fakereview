print("Importing packages")
import numpy as np
import pandas as pd
import torch
import nltk
import os.path

from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer 
print("Done")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} to compute")

# Load sentence transformer
sentence_model = SentenceTransformer("thenlper/gte-large").to(device)

# Define function for transforming text
def get_sentence_embedding(text):
    if isinstance(text,str):
        if not text.strip(): 
            print("Attempted to get embedding for empty text.")
            return []
        
        return sentence_model.encode(text)
    
    elif isinstance(text,list):
        return [get_sentence_embedding(x) for x in text]

# Load our data
DATA_DIR = ""
df = pd.read_csv(f"{DATA_DIR}combined_data.csv")

# Map each text entry to a tensor
print("Mapping text to tensors")
tqdm.pandas()
embeddings = df['Text'].progress_apply(get_sentence_embedding)

# Stick the tensors on the end of our dataframe
df['embedding']  = embeddings

# Save the result
df.to_parquet('combined_data_with_embeddings.parquet')
print("Done")
