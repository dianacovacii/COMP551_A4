from LSTM_model import LSTMModel
import save_load_models as slm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from data_loader import get_lstm_loaders


# Contant variables
LEARNING_RATE = [0.1, 0.05, 0.01, 0.005, 0.001]
EPOCHS = 20
train_loader, val_loader, test_loader, embedding, vocab_size = get_lstm_loaders()

# Set device
# quick device preference check used by your scripts
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

NUM_CLASSES_DOMAIN = 7
NUM_CLASSES_SUBDOMAIN = 34

def train_lstm_experiment(target_name, target_idx, num_classes):
    print(f"\n=== Training LSTM for {target_name} ===")
    for lr in LEARNING_RATE:
        model_name = f"lstm_{target_name}_lr_{lr:.3f}_{EPOCHS}"
        model = slm.load_model(model_name)
        if model is None:
            print(f"Training {model_name}...")
            model = LSTMModel(hidden_size=64, output_size=num_classes, embedding_matrix=embedding)
            model.to(device)
            optimizer = Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            model.train_model(train_loader, val_loader, test_loader, EPOCHS, optimizer, loss_fn, device, target_idx=target_idx)
            slm.save_model(model, model_name)
            print(f"Training time: {model.train_time:.2f}s")
        else:
            print(f"Loaded existing model: {model_name}")

# Train Domain (y1 is at index 1 in LSTM loader)
train_lstm_experiment("domain", 1, NUM_CLASSES_DOMAIN)

# Train Subdomain (y2 is at index 2 in LSTM loader)
train_lstm_experiment("subdomain", 2, NUM_CLASSES_SUBDOMAIN)