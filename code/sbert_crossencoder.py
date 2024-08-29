################################################################################
# IMPORTS
################################################################################

# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random
import os
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import logging
import glob

# Torch imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

# SBERT imports 
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, CrossEncoder, LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.readers import InputExample

device = None
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device}")

################################################################################
# SETUP
################################################################################

# Convenience and saving flags
ABRIDGED_RUN = False

# Training hyperparameters
BATCH_SIZE = 32 # Number of samples per batch while training our network
NUM_EPOCHS = 20 # Number of epochs to train our network

if ABRIDGED_RUN == True:
    BATCH_SIZE = 16
    NUM_EPOCHS = 50

LEARNING_RATE = 1e-6 # Learning rate for our optimizer

# Model name
MODEL_NAME = "SBERT_CrossEncoder"

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "../raw data/"

output_dir = Path(f'{CHECKPOINT_DIR}{MODEL_NAME}')
output_dir.mkdir(parents=True, exist_ok=True)
output_dir = f'{CHECKPOINT_DIR}{MODEL_NAME}'
output_dir = output_dir + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Logging configuration
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

################################################################################
# LOAD DATA
################################################################################

data = pd.read_csv("../raw data/combined_data.csv")

if ABRIDGED_RUN == True:
    data = data.sample(320)

# Set integer labels
# Human = 0, 1 = Machine
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Human' else 1)

# Train test split, stratified by binary labels
data_train, data_test = train_test_split(data, test_size = 0.2, stratify=data['Label'])
train_tt, train_vv = train_test_split(data_train, test_size = 0.2, stratify=data_train['Label'])

################################################################################
# DATASETS
################################################################################

train_dataset = Dataset.from_dict({
    "text": train_tt['Text'],
    "label": train_tt['Label'],
})

val_dataset = Dataset.from_dict({
    "text": train_vv['Text'],
    "label": train_vv['Label'],
})

test_dataset =  Dataset.from_dict({
    "text": data_test['Text'],
    "label": data_test['Label'],
})

train_samples = []
val_samples = []
test_samples = []

for train_sample in train_dataset:
     train_samples.append(InputExample(texts=[train_sample['text']], label=train_sample['label'])) 
for val_sample in val_dataset:
     val_samples.append(InputExample(texts=[val_sample['text']], label=val_sample['label']))
for test_sample in test_dataset:
     test_samples.append(InputExample(texts=[test_sample['text']], label=test_sample['label']))

################################################################################
# MODEL INITIALIZATION
################################################################################

# The CrossEncoder class is a wrapper around Huggingface AutoModelForSequenceClassification
if len(os.listdir(CHECKPOINT_DIR)) == 0:
    model = CrossEncoder("sentence-transformers/all-MiniLM-L6-v2", 
                     num_labels=1,
                     device = device)
else: 
    model = CrossEncoder(CHECKPOINT_DIR + '/SBERT_CrossEncoder2024-08-29_05-23-36')
    #model = CrossEncoder(max(glob.glob(os.path.join(CHECKPOINT_DIR, '*/')), key=os.path.getmtime)[:-1])


################################################################################
# TRAINING SETUP
################################################################################
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
accuracy_evaluator = CEBinaryAccuracyEvaluator.from_input_examples(val_samples)


################################################################################
# TRAINING 
################################################################################
model.fit(
    train_dataloader=train_dataloader,
    evaluator=accuracy_evaluator,
    epochs=NUM_EPOCHS,
    loss_fct = nn.CrossEntropyLoss(),
    scheduler = "warmuplinear",
    optimizer_class = torch.optim.AdamW,
    output_path=output_dir,
    save_best_model = True,
    show_progress_bar = True,
)

################################################################################
# REPORT ON TEST DATA
################################################################################

model = CrossEncoder(output_dir)
evaluator = CEBinaryAccuracyEvaluator.from_input_examples(test_samples)
evaluator(model)

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