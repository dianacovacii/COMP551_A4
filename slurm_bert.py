from BERT_model import BERTClassifier
import save_load_models as slm
import torch
from torch.optim import AdamW
from data_loader import get_bert_loaders
import os

# Constant variables
LEARNING_RATES = [0.00001, 0.00002, 0.00005] # BERT usually needs smaller LR
EPOCHS = 5 # BERT converges faster
bert_train_loader, bert_val_loader, bert_test_loader = get_bert_loaders()


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

def train_bert_experiment(target_name, target_idx, num_classes):
    print(f"\n=== Training BERT for {target_name} ===")
    
    for lr in LEARNING_RATES:
        print(f"\nTraining with Learning Rate: {lr}")
        model_name = f"bert_{target_name}_lr_{lr}"
        
        # Check if model exists
        model = slm.load_model(model_name)
        
        if model is None:
            print("Initializing new BERT model...")
            model = BERTClassifier(output_size=num_classes)
            model.to(device)
            
            optimizer = AdamW(model.parameters(), lr=lr)
            
            model.train_model(bert_train_loader, bert_val_loader, bert_test_loader, EPOCHS, optimizer, device, target_idx=target_idx)
            
            slm.save_model(model, model_name)
        else:
            print(f"Loaded existing model: {model_name}")

# Train for Domain (y1) -> index 2 in batch
train_bert_experiment("domain", 2, NUM_CLASSES_DOMAIN)

# Train for Subdomain (y2) -> index 3 in batch
train_bert_experiment("subdomain", 3, NUM_CLASSES_SUBDOMAIN)


