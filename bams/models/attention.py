#source: https://spotintelligence.com/2023/01/31/self-attention/

import torch
import torch.nn as nn
import torch.nn.functional as F

# class SelfAttention(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SelfAttention, self).__init__()
#         self.input_dim = input_dim
#         self.query = nn.Linear(input_dim, output_dim)
#         self.key = nn.Linear(input_dim, output_dim)
#         self.value = nn.Linear(input_dim, output_dim)
#         self.softmax = nn.Softmax(dim=2)
        
#     def forward(self, x):
#         queries = self.query(x)
#         keys = self.key(x)
#         values = self.value(x)
#         #scaled dot product of queries, keys
#         scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5) 
#         attention = self.softmax(scores)
#         weighted = torch.bmm(attention, values)
#         return weighted


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Linear layers for queries, keys, and values
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Output linear layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Compute queries, keys, and values
        queries = self.query(x)    # Shape: (batch, seq_len, hidden_dim)
        keys = self.key(x)         # Shape: (batch, seq_len, hidden_dim)
        values = self.value(x)     # Shape: (batch, seq_len, hidden_dim)
        
        # Compute scaled dot-product attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # Shape: (batch, seq_len, seq_len)
        
        # Apply softmax to get attention weights
        attention = self.softmax(scores)  # Shape: (batch, seq_len, seq_len)
        
        # Compute the weighted sum of values
        weighted = torch.bmm(attention, values)  # Shape: (batch, seq_len, hidden_dim)
        
        # Pass through the output linear layer
        out = self.fc_out(weighted)  # Shape: (batch, seq_len, output_dim)
        
        # Add residual connection and apply layer normalization
        out = self.layer_norm(out + x)  # Assuming input and output dims match; adjust if necessary
        
        return out

class SelfAttentionStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SelfAttentionStack, self).__init__()
        self.num_layers = num_layers
        
        # Create a list of self-attention layers
        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                input_dim=input_dim if i == 0 else output_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            ) for i in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x