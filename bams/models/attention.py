#source: https://spotintelligence.com/2023/01/31/self-attention/

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        #scaled dot product of queries, keys
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5) 
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted