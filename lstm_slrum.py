from LSTM_model import LSTMModel
import save_load_models as slm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from data_loader import get_lstm_loaders


# Contant variables
LEARNING_RATE = 0.001
model_name = f"lstm_model_lr_{LEARNING_RATE}"
train_loader, val_loader, test_loader, embedding_matrix, vocab_size = get_lstm_loaders()

# Set device
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
if device == "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to load existing model
model = slm.load_model(model_name)
if model is None:
    model = LSTMModel(input_size=10, hidden_size=20, output_size=2)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    model.train_model(train_loader, val_loader, test_loader, 5, optimizer, loss_fn, device)
    slm.save_model(model, model_name)