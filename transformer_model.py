#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import jsonlines
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import datetime
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


# In[4]:


# Set the path to the downloaded EMBER dataset directory
data_path = "data/ember2018"

train_pickle_file = "data/training_dump.pkl"
test_pickle_file = "data/test_dump.pkl"


# In[5]:


def printf(*args, **kwargs):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add the timestamp to the message
    message = f"[{timestamp}] " + ' '.join(map(str, args))

    # Call the original print function with the modified message
    print(message, **kwargs)


# In[6]:


def load_features(file_path):
    features = []
    with jsonlines.open(os.path.join(data_path, file_path)) as reader:
        for sample in reader:
            features.append(sample)
    return features


# In[7]:


class MalwareDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label


# In[8]:


if os.path.exists(train_pickle_file):
    printf("Loading data from train_dump.pkl...")
    # Load the train_features from the pickle file
    with open(train_pickle_file, "rb") as f:
        train_features1 = pickle.load(f)
else:
    # Load the train_features from the JSONL files
    train_features1 = []
    for i in {0, 4}:
        printf("Collecting data from train_features1_" + str(i) + ".jsonl...")
        train_features1.extend(load_features("train_features_" + str(i) + ".jsonl"))

    # Save the train_features to the pickle file
    with open(train_pickle_file, "wb") as f:
        pickle.dump(train_features1, f)
printf("Completed loading training feature data, train_features1.len() = ", len(train_features1))


# In[9]:


if os.path.exists(test_pickle_file):
    printf("Loading data from test_dump.pkl...")
    # Load the test_features from the pickle file
    with open(test_pickle_file, "rb") as f:
        test_features = pickle.load(f)
else:
    printf("Collecting data from test_features.jsonl...")
    # Load the test_features from the JSONL file
    test_features = load_features("test_features.jsonl")

    # Save the test_features to the pickle file
    with open(test_pickle_file, "wb") as f:
        pickle.dump(test_features, f)
printf("Completed loading test feature data: ", len(test_features))


# In[10]:


# Split the dataset into training and validation sets
train_features, val_features = train_test_split(train_features1, test_size=0.2, random_state=42)

# Extract labels from the features
try:
    train_labels = [sample['label'] for sample in train_features]
    val_labels = [sample['label'] for sample in val_features]
except:
    train_labels = [sample['label'] for sample in train_features[0]]
    val_labels = [sample['label'] for sample in val_features[0]]
test_labels = [sample['label'] for sample in test_features]

print("train_features shape:", len(train_features))
print("Train labels shape:", len(train_labels))
print("val_labels shape:", len(val_labels))
print("val_features shape:", len(val_features))
print("test_labels shape:", len(test_labels))

# Check the class distribution in the training, validation, and testing sets
train_class_counts = dict(zip(*np.unique(train_labels, return_counts=True)))
val_class_counts = dict(zip(*np.unique(val_labels, return_counts=True)))
test_class_counts = dict(zip(*np.unique(test_labels, return_counts=True)))

printf("Training set class distribution:", train_class_counts)
printf("Validation set class distribution:", val_class_counts)
printf("Testing set class distribution:", test_class_counts)


# In[11]:


# Determine the dimensionality of numerical features
numerical_dim = 100  # Replace with the actual dimensionality

# Define the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[12]:


# Encode numerical features
def encode_numerical_features(features):
    encoded_features = []
    for sample in features:
        encoded_sample = []
        for feature_name, feature_value in sample.items():
            if isinstance(feature_value, float) or isinstance(feature_value, int):
                # Normalize numerical values to [0, 1] range
                encoded_sample.append(feature_value / numerical_dim)
        encoded_features.append(encoded_sample)
    return np.array(encoded_features)


# In[13]:


# Convert features to sequential format
def convert_to_sequences(features, tokenizer):
    sequences = []
    for sample in features:
        sequence = []
        for feature_name, feature_value in sample.items():
            if isinstance(feature_value, str):
                # Tokenize string features
                tokens = tokenizer.tokenize(feature_value)
                sequence.extend(tokens)
        sequences.append(sequence)
    return sequences


# In[14]:


# Encode numerical features in train, validation, and test sets
train_encoded = encode_numerical_features(train_features)
val_encoded = encode_numerical_features(val_features)
test_encoded = encode_numerical_features(test_features)

# Convert features to sequential format
train_sequences = convert_to_sequences(train_features, tokenizer)
printf("Training set sequence encoding:", len(train_sequences))
val_sequences = convert_to_sequences(val_features, tokenizer)
printf("Validation set sequence encoding:", len(val_sequences))
test_sequences = convert_to_sequences(test_features, tokenizer)
printf("Test set sequence encoding:", len(test_sequences))

# Check the class distribution in the sequence data
train_sequence_class_counts = dict(zip(*np.unique(train_labels, return_counts=True)))
val_sequence_class_counts = dict(zip(*np.unique(val_labels, return_counts=True)))
test_sequence_class_counts = dict(zip(*np.unique(test_labels, return_counts=True)))

printf("Training set sequence class distribution:", train_sequence_class_counts)
printf("Validation set sequence class distribution:", val_sequence_class_counts)
printf("Testing set sequence class distribution:", test_sequence_class_counts)


# In[15]:


from transformers import BertModel, BertConfig

# Load pre-trained model
model_name = "bert-base-uncased"
config = BertConfig.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)


# In[16]:


import torch

# Convert sequences back to string representation
printf("Converting train_strings sequences back to string representation...")
train_strings = [tokenizer.decode(tokenizer.convert_tokens_to_ids(seq), skip_special_tokens=False) for seq in train_sequences]
printf("Converting val_strings sequences back to string representation...")
val_strings = [tokenizer.decode(tokenizer.convert_tokens_to_ids(seq), skip_special_tokens=False) for seq in val_sequences]
printf("Converting test_strings sequences back to string representation...")
test_strings = [tokenizer.decode(tokenizer.convert_tokens_to_ids(seq), skip_special_tokens=False) for seq in test_sequences]

# Tokenize the sequences
tokenizer = BertTokenizer.from_pretrained(model_name)
printf("Tokenizing the train_encodings sequences...")
train_encodings = tokenizer.batch_encode_plus(train_strings, padding=True, truncation=True, max_length=128, return_tensors="pt")
printf("Tokenizing the val_encodings sequences...")
val_encodings = tokenizer.batch_encode_plus(val_strings, padding=True, truncation=True, max_length=128, return_tensors="pt")
printf("Tokenizing the test_encodings sequences...")
test_encodings = tokenizer.batch_encode_plus(test_strings, padding=True, truncation=True, max_length=128, return_tensors="pt")

printf("Converting the encodings to tensors...")
# Convert the encodings to tensors
train_input_ids = train_encodings["input_ids"]
train_attention_mask = train_encodings["attention_mask"]
val_input_ids = val_encodings["input_ids"]
val_attention_mask = val_encodings["attention_mask"]
test_input_ids = test_encodings["input_ids"]
test_attention_mask = test_encodings["attention_mask"]

print("Train labels shape:", len(train_labels))
print("Val labels  shape:", len(val_labels))
print("Test labels shape:", len(test_labels))

printf("Converting the labels to tensors...")
# Convert the labels to tensors
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

printf("Done converting the labels to tensors!")
print("Train labels shape:", len(train_labels))
print("Val labels  shape:", len(val_labels))
print("Test labels shape:", len(test_labels))


# In[17]:


print("Train input ids shape:", train_input_ids.shape)
print("Train attention mask shape:", train_attention_mask.shape)
print("Train labels shape:", train_labels.shape)

print("Val input ids shape:", val_input_ids.shape)
print("Val attention mask shape:", val_attention_mask.shape)
print("Val labels shape:", val_labels.shape)

print("Test input ids shape:", test_input_ids.shape)
print("Test attention mask shape:", test_attention_mask.shape)
print("Test labels shape:", test_labels.shape)

import torch
from torch.utils.data import TensorDataset, DataLoader

# Check the shape of the label tensors
print("Train labels shape:", train_labels.shape)
print("Val labels shape:", val_labels.shape)
print("Test labels shape:", test_labels.shape)

# Define the number of classes excluding the unlabeled class (-1)
num_classes = 2  # Set the number of classes for your classification task

# Filter out unlabeled instances (-1)
train_mask = train_labels != -1
val_mask = val_labels != -1
test_mask = test_labels != -1

# Apply the masks to exclude unlabeled instances
train_input_ids = train_input_ids[train_mask]
train_attention_mask = train_attention_mask[train_mask]
train_labels = train_labels[train_mask]
val_input_ids = val_input_ids[val_mask]
val_attention_mask = val_attention_mask[val_mask]
val_labels = val_labels[val_mask]
test_input_ids = test_input_ids[test_mask]
test_attention_mask = test_attention_mask[test_mask]
test_labels = test_labels[test_mask]

# Convert the labels to one-hot encoding
train_labels_onehot = torch.nn.functional.one_hot(train_labels, num_classes=num_classes).float()
val_labels_onehot = torch.nn.functional.one_hot(val_labels, num_classes=num_classes).float()
test_labels_onehot = torch.nn.functional.one_hot(test_labels, num_classes=num_classes).float()

# Create TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels_onehot)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels_onehot)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels_onehot)

# Create DataLoaders
batch_size = 4  # Set the batch size to your desired value
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# In[18]:


class MalwareDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MalwareDetectionModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.view(-1, input_ids.size(-1))  # Reshape input_ids tensor
        embedded = self.embedding(input_ids.to(torch.float))
        embedded = self.dropout(embedded)
        logits = self.classifier(embedded)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        else:
            return logits


# In[19]:


# Instantiate the model
input_size = 90  # Set the input size according to your data
hidden_size = 768  # Adjust the hidden size as per your requirements
num_classes = 2  # Adjust the number of classes as per your requirements

model = MalwareDetectionModel(input_size, hidden_size, num_classes)
# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
learning_rate = 0.001  # Set your desired learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

# Create DataLoaders
batch_size = 8  # Set your desired batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
num_epochs = 10  # Set your desired number of epochs
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


# In[20]:


# Testing on test set
model.eval()
test_loss = 0.0
correct_predictions = 0
total_predictions = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        test_loss += loss.item()

        _, predicted_labels = torch.max(logits, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

test_loss = test_loss / len(test_dataloader)
test_accuracy = correct_predictions / total_predictions

print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")


# In[21]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        logits = model(input_ids, attention_mask)
        _, predicted_labels = torch.max(logits, 1)

        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')
auc_roc = roc_auc_score(true_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

