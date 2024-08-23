################################################################################
########## IMPORTS
################################################################################

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, Subset, random_split

################################################################################
########## HYPERPARAMETERS
################################################################################

BATCH_SIZE = 4 # Note: got out of GPU memory error with batchsize 8 on cluster
NUM_EPOCHS = 4

################################################################################
########## MODEL LOADING
################################################################################


# The following class wraps an instance of the GTE model and attaches a linear
# projection at the end of it. The input to this model comes with shape
# (batchsize, 512, 3) where each input has 512 tokens, 512 length token types
# and 512 length attention mask.
# Code based on usage example here: https://huggingface.co/thenlper/gte-large
class GTEtuner(nn.Module):
    def __init__(self):
        super(GTEtuner, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.gte = AutoModel.from_pretrained("thenlper/gte-large")
        self.linear_layer = nn.Linear(1024, 1)

    # Takes a string and tokenizes it.
    # Returns a tensor of size (batchsize, 512, 3) where third dimension
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
        return t

    # Applies GTE to a tokenized string of shape (batchsize, 512, 3)
    def embed(self, t):
        tokens = t[:,:,2].sum(dim = 1) # Number of coordinates to include
        outputs = []
        # Annoyingly, due to different size inputs we need to iterate
        for i, num_tokens in enumerate(tokens):
            gte_inputs = t[i, :num_tokens, :].unsqueeze(0)
            x = self.gte(input_ids = gte_inputs[:,:,0],
                         token_type_ids = gte_inputs[:,:,1],
                         attention_mask = gte_inputs[:,:,2])
            average_pooled = x.last_hidden_state.sum(dim = 1) / num_tokens
            x = F.normalize(average_pooled, p = 2, dim = 1)
            outputs += [x]
        return torch.cat(outputs, dim = 0)

    # Applies GTE then a linear layer to a tokenized string
    # of shape (batchsize, 512, 3)
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
DATA_DIR = "."
df = pd.read_csv(f"{DATA_DIR}/combined_data.csv")
# df = df.sample(n = 500, random_state = 406) # For quicker tests

df['Stratify'] = df[['Label', 'Original dataset']].apply(lambda x: x['Label'] + " " + x['Original dataset'], axis = 1)
df['Label'] = df['Label'].progress_apply(lambda x: Tensor([1]) if x == 'Machine' else Tensor([0]))
df['Tokens'] = df['Text'].progress_apply(lambda x: model.tokenize(x).squeeze(0))

train, val = train_test_split(df, test_size = 0.2, stratify=df['Stratify'], random_state = 406)

################################################################################
########## DATASET CLASS
################################################################################

class EmbeddingDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tokens, label = list(df[['Tokens', 'Label']].iloc[index])
        return [tokens, label]

train_data_loader = torch.utils.data.DataLoader(EmbeddingDataset(train), batch_size = BATCH_SIZE, shuffle = True)
val_data_loader = torch.utils.data.DataLoader(EmbeddingDataset(val), batch_size = BATCH_SIZE, shuffle = True)

################################################################################
########## SETUP LOSS AND OPTIMIZER
################################################################################

criterion = nn.BCEWithLogitsLoss(reduction='mean')

# TODO choose an optimizer setup that focused on updating weights in
# later layers of the model
optimizer = optim.Adam(model.parameters(), lr=0.1) 

################################################################################
########## TRAINING LOOP
################################################################################

for epoch_num in range(NUM_EPOCHS):
    running_loss = 0
    for inputs, labels in tqdm(train_data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Training loss for epoch {epoch_num}: {running_loss / len(train_data_loader)}")
    with torch.no_grad():
        model.eval()
        answers_correct = 0
        for inputs,labels in tqdm(val_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.nn.Sigmoid()(model(inputs))
            # Avoiding rounding errors in floats
            guesses = torch.round(outputs)
            labels = torch.round(labels)
            answers_correct += int((guesses == labels).sum())
    acc = answers_correct / len(val)
    print(f"Validation accuracy for epoch {epoch_num}: {acc}")

    model.train()

################################################################################
########## SAVE AND REPORT RESULTS
################################################################################

torch.save(model.state_dict(), 'embedding_tuner.pt')
