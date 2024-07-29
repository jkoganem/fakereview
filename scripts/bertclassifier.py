################################################################################
# SETUP
################################################################################

# Convenience and saving flags
#ABRIDGED_RUN = True
ABRIDGED_RUN = False
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after ever epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training

# Training hyperparameters
BATCH_SIZE = 16 # Number of samples per batch while training our network
NUM_EPOCHS = 50 # Number of epochs to train our network
LEARNING_RATE = 0.00001 # Learning rate for our optimizer

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "../data/"
MODEL_NAME = "bert"

NUM_WORKERS = 16
################################################################################
# IMPORTS
################################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import AdamW

from tqdm import tqdm

data = pd.read_csv('../raw data/combined_data.csv')
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Human' else 1)
data.head()

train_X, test_X, train_Y, test_Y = train_test_split(
    data['Text'], 
    data['Label'], 
    train_size=0.7, 
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
train_tokens = tokenizer(list(train_X), padding = True, truncation=True)
test_tokens = tokenizer(list(test_X), padding = True, truncation=True)

class TokenData(Dataset):
    def __init__(self, train = False):
        if train:
            self.text_data = train_X
            self.tokens = train_tokens
            self.labels = list(train_Y)
        else:
            self.text_data = test_X
            self.tokens = test_tokens
            self.labels = list(test_Y)

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.tokens.items():
            sample[k] = torch.tensor(v[idx])
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample
    
batch_size = BATCH_SIZE
train_dataset = TokenData(train = True)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = TokenData(train = False)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

bert_model = BertForSequenceClassification.from_pretrained('bert-base-cased') # Pre-trained model
optimizer = AdamW(bert_model.parameters(), lr=LEARNING_RATE) # Optimization function
loss_fn = torch.nn.CrossEntropyLoss() # Loss function
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False)

num_epochs = NUM_EPOCHS
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
bert_model.to(device) # Transfer model to GPU if available

for epoch in tqdm(range(num_epochs)):
    print("Epoch: ",(epoch + 1))
    # TRAINING BLOCK STARTS
    bert_model.train()
    for i,batch in enumerate(train_loader):    
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Setting the gradients to zero
        optimizer.zero_grad()
        
        # Passing the data to the model
        outputs = bert_model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
        
        # The logits will be used for measuring the loss
        pred = outputs.logits
        loss = loss_fn(pred, batch['labels'])

        # Calculating the gradient for the loss function
        loss.backward()
        
        # Optimizing the parameters of the bert model
        optimizer.step()

        # Calculating the running loss for logging purposes
        train_batch_loss = loss.item()
        train_last_loss = train_batch_loss / batch_size
        
        if SAVE_CHECKPOINTS == True:
            torch.save(bert_model.state_dict(), f"{CHECKPOINT_DIR}{MODEL_NAME}/checkpoint_{num_epochs+1}.pt")

        print('Training batch {} last loss: {}'.format(i + 1, train_last_loss))
    # Logging epoch-wise training loss
    print(f"\nTraining epoch {epoch + 1} loss: ",train_last_loss)
    # TRAINING BLOCK ENDS 

    # TESTING BLOCK STARTS
    bert_model.eval()
    correct = 0
    test_pred = []
    for i, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # We don't need gradients for testing
        with torch.no_grad():
            outputs = bert_model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
        
        # Logits act as predictions
        logits = outputs.logits
        
        # Calculating total batch loss using the logits and labels
        loss = loss_fn(logits, batch['labels'])
        test_batch_loss = loss.item()
        
        # Calculating the mean batch loss
        test_last_loss = test_batch_loss / batch_size
        print('Testing batch {} loss: {}'.format(i + 1, test_last_loss))
        
        # Comparing the predicted target with the labels in the batch
        correct += (logits.argmax(1) == batch['labels']).sum().item()
        print("Testing accuracy: ",correct/((i + 1) * batch_size))
    
    print(f"\nTesting epoch {epoch + 1} last loss: ",test_last_loss)
    # TESTING BLOCK ENDS
    