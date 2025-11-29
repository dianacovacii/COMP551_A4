from LSTM_model import LSTMModel
import save_load_models as slm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from data_loader import get_lstm_loaders


# Contant variables
LEARNING_RATE = 0.001
EPOCHS = 20
model_name = f"lstm_model_lr_{LEARNING_RATE}_{EPOCHS}"
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

# Try to load existing model
model = slm.load_model(model_name)
if model is None:
    model = LSTMModel(hidden_size=64, output_size=7, embedding_matrix=embedding)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    model.train_model(train_loader, val_loader, test_loader, EPOCHS, optimizer, loss_fn, device, target_idx=1)
    slm.save_model(model, model_name)
    print(model.train_time)