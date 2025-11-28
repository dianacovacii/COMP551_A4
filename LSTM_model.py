import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import time

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Put all parameters in one matrix for efficiency
        # (nn.Linear contains weights and bias)
        self.input2gates = nn.Linear(input_size, hidden_size * 4) 
        self.hidden2gates = nn.Linear(hidden_size, hidden_size * 4)

        # The Final Prediction Layer (Hidden -> Output Class)
        self.fc = nn.Linear(hidden_size, output_size)

        self.__inititalize_weights()

        # Variables for tracking learning
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_accuracies = []

    def __inititalize_weights(self):
        nn.init.xavier_uniform_(self.input2gates.weight)
        nn.init.xavier_uniform_(self.hidden2gates.weight)
        nn.init.zeros_(self.input2gates.bias)
        nn.init.zeros_(self.hidden2gates.bias)



    def lstm_cell(self, input, short_term_memory, long_term_memory):
        '''
        short_term_memory: hidden state
        long_term_memory: cell state
        '''
        # Compute gates all at once
        gates = self.input2gates(input) + self.hidden2gates(short_term_memory)

        # Split the gates into their own tensors
        forget_chuck, input_chunk, memory_chunk, output_chunk = gates.chunk(4, 1)
        
        # Apply activations
        long_remember_percent = torch.sigmoid(forget_chuck)
        potential_memory_percent = torch.sigmoid(input_chunk)
        potential_memory = torch.tanh(memory_chunk)
        output_percent = torch.sigmoid(output_chunk)

        # Update long term memory
        updated_long_term_memory = long_remember_percent * long_term_memory
        updated_long_term_memory += potential_memory_percent * potential_memory

        # Update short term memory
        updated_short_term_memory = output_percent * torch.tanh(updated_long_term_memory)

        return updated_short_term_memory, updated_long_term_memory 
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Process the input sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)

        # Pass last hidden state through the output layer
        out = self.fc(h_t)
        return out

    def train_model(self, train_loader, val_loader, test_loader, epochs, optimizer, loss_fn, device):
        start_time = time.time()

        total_epochs = len(self.train_losses) + epochs

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # Clear gradients from previous batch
                
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, dim=1)
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
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = self(inputs)
                    loss = loss_fn(outputs, labels)

                    preds = torch.argmax(outputs, dim=1)
                    val_loss += loss.item()
                    val_correct += (preds == labels).sum().item()
                    val_total += len(preds)

            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_acc = val_correct / val_total
            self.val_losses.append(val_epoch_loss)
            self.val_accuracies.append(val_epoch_acc)

            # Test accuracy (on the full test set)
            self.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = self(inputs)

                    preds = torch.argmax(outputs, dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_total += len(preds)
            test_acc = test_correct / test_total
            self.test_accuracies.append(test_acc)

            print(f'Epoch {len(self.train_losses)}/{total_epochs}: Train loss={epoch_loss:.4f}, Train acc={epoch_acc*100:.2f}%, Val loss={val_epoch_loss:.4f}, Val acc={val_epoch_acc*100:.2f}%, Test acc={test_acc*100:.2f}%')

        self.train_time = time.time() - start_time
            

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            preds = torch.argmax(outputs, dim=1)
        return preds

    def evaluate_acc(self, true, pred):
        correct = (true == pred).sum().item()
        total = len(true)
        return correct / total
    

    
    