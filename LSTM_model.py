import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import time

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Put all parameters in one matrix for efficiency
        # Input to hidden (nn.Linear contains weights and bias)
        self.input2gates = nn.Linear(input_size, hidden_size * 4) 
        # Hidden to hidden
        self.hidden2gates = nn.Linear(hidden_size, hidden_size * 4)

        self.__inititalize_weights()

    def __inititalize_weights(self):
        nn.init.xavier_uniform_(self.input2gates.weight)
        nn.init.xavier_uniform_(self.hidden2gates.weight)
        nn.init.zeros_(self.input2gates.bias)
        nn.init.zeros_(self.hidden2gates.bias)



    def forward(self, input, short_term_memory, long_term_memory):
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

    def predict(self, x):
        pass

    def evaluate_acc(self, true, pred):
        pass
    

    
    