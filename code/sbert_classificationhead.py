################################################################################
# IMPORTS
################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path


################################################################################
# SETUP
################################################################################

# Convenience and saving flags
ABRIDGED_RUN = False
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after ever epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training
REPORT_VALIDATION_ACCURACY_PER_EPOCH = True # We also want to monitor the accuracy score on the validation set

# Training hyperparameters
BATCH_SIZE = 32 # Number of samples per batch while training our network
NUM_EPOCHS = 100 # Number of epochs to train our network
LEARNING_RATE = 1e-3 # Learning rate for our optimizer
NUM_CLASSES = 2

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "../data/"

MODEL_NAME = "NAIVE_CLASSIFIER"


device = None
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device}")

################################################################################
# LOAD DATA
################################################################################

data = pd.read_parquet('../raw data/embeddings-SBERT-all-MiniLM-L6-v2.parquet')
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Human' else 1)


data.head()
stratify = data['Label']
data_train, data_validation = train_test_split(data, test_size = 0.2, stratify=stratify)


################################################################################
# DATASET CLASS
################################################################################

class embeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)  # Convert embeddings to torch tensor
        self.labels = torch.tensor(labels, dtype=torch.int)  # Convert labels to torch tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label
    
train_data = embeddingDataset(data_train['embedding'].to_list(), data_train['Label'].to_list())
val_data = embeddingDataset(data_validation['embedding'].to_list(), data_validation['Label'].to_list())


################################################################################
# DATALOADER
################################################################################

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

validation_dataloader = DataLoader(val_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

################################################################################
# ARCHITECTURE
################################################################################

class classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
# Initialize classification head
model = classifier(384, NUM_CLASSES).to(device)


################################################################################
# TRAINING SETUP
################################################################################

# Create a saving directory if needed
output_dir = Path(f'{CHECKPOINT_DIR}{MODEL_NAME}')
output_dir.mkdir(parents=True, exist_ok=True)
output_dir = f'{CHECKPOINT_DIR}{MODEL_NAME}'

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Training loop
print(f"Training on {len(train_data)} samples with {BATCH_SIZE} samples per batch.")
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    print(f"Validating on {len(validation_dataloader)} samples at the end of each epoch.")

training_losses = [None]*NUM_EPOCHS
validation_losses = [None]*NUM_EPOCHS
validation_accuracies = [None]*NUM_EPOCHS

torch.enable_grad() # Turn on the gradient

################################################################################
# TRAINING LOOP
################################################################################

for epoch_num, epoch in enumerate(tqdm(range(NUM_EPOCHS), leave = False)):

    running_loss = 0.0
    
    for i, data in enumerate(tqdm(train_dataloader, leave = False)):
        
        # Get batch of inputs and true labels, push to device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass on batch of inputs
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Update weights
        optimizer.step()

        # Update loss
        running_loss += loss.item()

    # Save checkpoint
    if SAVE_CHECKPOINTS == True:
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}{MODEL_NAME}/checkpoint_{epoch_num+1}.pt")

    # Compute training loss
    if REPORT_TRAINING_LOSS_PER_EPOCH == True:    
        training_losses[epoch_num] = running_loss/len(train_dataloader)
        
    # Compute validation loss
    if REPORT_VALIDATION_LOSS_PER_EPOCH == True or REPORT_VALIDATION_ACCURACY_PER_EPOCH == True:
        validation_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        model.eval()
        with torch.no_grad():
            for validation_data in validation_dataloader:
                inputs, labels = validation_data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                validation_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        validation_losses[epoch_num] = validation_loss/len(validation_dataloader)
        validation_accuracies[epoch_num] = correct_predictions / total_predictions
        model.train()
        torch.enable_grad()

    # Save losses
    losses = pd.DataFrame({"training_losses":training_losses, "validation_losses":validation_losses, "validation_accuracies": validation_accuracies})
    cols = []
    if REPORT_TRAINING_LOSS_PER_EPOCH == True:
        print(f"Training loss at epoch {epoch_num+1}: {training_losses[epoch_num]}")
        cols += ["training_losses"]
    if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
        print(f"Validation loss at epoch {epoch_num+1}: {validation_losses[epoch_num]}")
        cols += ["validation_losses"]
    if REPORT_VALIDATION_ACCURACY_PER_EPOCH == True:
        print(f"Validation accuracy at epoch {epoch_num+1}: {validation_accuracies[epoch_num]}")
        cols += ["validation_losses"]
    if len(cols) > 0:
        losses[cols].to_csv(f'{output_dir}/losses.csv', index = False)

print('Finished Training')


################################################################################
# SAVE AND REPORT 
################################################################################

# Save model
if SAVE_AFTER_TRAINING == True:
    torch.save(model.state_dict(), f'{output_dir}/final.pt')

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################