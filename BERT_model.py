import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
import time
import numpy as np

class BERTClassifier(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=output_size,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Variables for tracking learning
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.train_time = 0

    def forward(self, input_ids, attention_mask, labels=None, output_attentions=False):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=output_attentions)
        return output

    def train_model(self, train_loader, val_loader, test_loader, epochs, optimizer, device, target_idx):
        """
        target_idx: index of the label in the batch.
        For BERT loader: batch = (input_ids, attention_mask, y1, y2)
        So target_idx=2 for y1 (domain), target_idx=3 for y2 (subdomain)
        """
        start_time = time.time()
        total_epochs = len(self.train_losses) + epochs

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[target_idx].to(device)

                self.zero_grad()
                
                # BERT forward pass with labels returns loss and logits
                outputs = self(input_ids, attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                preds = torch.argmax(logits, dim=1)
                total_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += len(preds)

            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    labels = batch[target_idx].to(device)
                    
                    outputs = self(input_ids, attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    preds = torch.argmax(logits, dim=1)
                    val_loss += loss.item()
                    val_correct += (preds == labels).sum().item()
                    val_total += len(preds)

            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_acc = val_correct / val_total
            self.val_losses.append(val_epoch_loss)
            self.val_accuracies.append(val_epoch_acc)

            # Test accuracy
            self.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    labels = batch[target_idx].to(device)
                    
                    outputs = self(input_ids, attention_mask)
                    logits = outputs.logits

                    preds = torch.argmax(logits, dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_total += len(preds)
            
            test_acc = test_correct / test_total
            self.test_accuracies.append(test_acc)

            print(f'Epoch {len(self.train_losses)}/{total_epochs}: Train loss={epoch_loss:.4f}, Train acc={epoch_acc*100:.2f}%, Val loss={val_epoch_loss:.4f}, Val acc={val_epoch_acc*100:.2f}%, Test acc={test_acc*100:.2f}%')

        self.train_time = time.time() - start_time