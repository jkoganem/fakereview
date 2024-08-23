################################################################################
########## IMPORTS
################################################################################

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, Subset, random_split

################################################################################
########## MODEL LOADING
################################################################################


# The following class wraps an instance of the GTE model and attaches a linear
# projection at the end of it. The actual GTE model is a bit hard to use,
# so this class strips away many of the parts that extraneous to our purposes,
# such as the attention masks.
# Code based on usage example here: https://huggingface.co/thenlper/gte-large
class GTEtuner(nn.Module):
    def __init__(self):
        super(GTEtuner, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.gte = AutoModel.from_pretrained("thenlper/gte-large")
        self.linear_layer = nn.Linear(1024, 2)

    # Takes a string and tokenizes it
    # to a tensor of size (L, 512, 3) where third dimension
    # is indexed as 0: input_ids, 1: token_type_ids, 2: attention_mask
    def tokenize(self, x):
        if type(x) == str:
            x = [x]
        batch_dict = self.tokenizer(x, max_length=512, padding=True, truncation=True, return_tensors='pt')
        input_ids = batch_dict['input_ids']
        token_type_ids = batch_dict['token_type_ids']
        attention_mask = batch_dict['attention_mask']
        t = torch.stack([input_ids, token_type_ids, attention_mask], dim = 2)
        pad_length = 512-t.shape[1]
        t = F.pad(t, (0,0,0, pad_length), 'constant', 0)
        return t.squeeze(0)

    # Takes a tokenized string (as a tensor) and embeds it using GTE
    def embed(self, t):
        x = self.gte(input_ids = t[:,:, 0],
                     token_type_ids = t[:,:,1],
                     attention_mask = t[:,:,2])
        x = x.last_hidden_state.sum(dim = 1)
        x = F.normalize(x, p = 2, dim = 1)
        return x

    # Applies GTE embedding, then the linear layer
    def forward(self, t):
        x = self.embed(t)
        proj = self.linear_layer(x)
        return proj

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GTEtuner().to(device)


################################################################################
########## DATALOADING
################################################################################

tqdm.pandas()
DATA_DIR = "../../raw data"
df = pd.read_csv(f"{DATA_DIR}/combined_data.csv")
# df = df.sample(100)

# TODO: train test split

df['Label'] = df['Label'].progress_apply(lambda x: Tensor([0,1]) if x == 'Machine' else Tensor([1,0]))
df['Tokens'] = df['Text'].progress_apply(model.tokenize)

class EmbeddingDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tokens, label = list(df[['Tokens', 'Label']].iloc[index])
        return [tokens, label]

data_loader = torch.utils.data.DataLoader(EmbeddingDataset(df), batch_size = 4, shuffle = True)

################################################################################
########## SETUP LOSS AND OPTIMIZER
################################################################################

criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

################################################################################
########## TRAINING LOOP
################################################################################

for inputs, labels in tqdm(data_loader):
	print(model(inputs)) # TODO 

################################################################################
########## SAVE AND REPORT RESULTS
################################################################################
