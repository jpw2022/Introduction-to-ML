import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, seq_length, activation=F.relu):
        """
        
        parameters:
        input_size (int): imput
        hidden_sizes (list of int): the list of hiddn layers
        output_size (int): output
        activation (callable, optional): default ReLU
        """
        super(CustomMLP, self).__init__()
        self.real_input_size = input_size * seq_length
        
        # from imput to the first hidden layer
        self.layers = nn.ModuleList([nn.Linear(self.real_input_size, hidden_sizes[0])])
        
        # all hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # from the last hidden layer to the output
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.activation = activation

    def forward(self, x):
        
        x = x.reshape(-1, self.real_input_size)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # no activation for the last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
