import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.attention = nn.Softmax(dim=-1)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Calculate attention scores (scaled dot-product attention)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = self.attention(attention_scores)
        
        # Compute the weighted sum of values based on attention scores
        output = torch.matmul(attention_weights, value)
        
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, input_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.attention = nn.Softmax(dim=-1)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Calculate attention scores (scaled dot-product attention)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = self.attention(attention_scores)
        
        # Compute the weighted sum of values based on attention scores
        output = torch.matmul(attention_weights, value)
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
        self.attention = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = self.attention(attention_scores)
        
        output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.fc_out(output)
        
        return output

class AdditiveAttention(nn.Module):
    def __init__(self, input_dim):
        super(AdditiveAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.v = nn.Parameter(torch.rand(input_dim))
        self.attention = nn.Softmax(dim=-1)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        
        # Compute attention scores via additive mechanism
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = torch.tanh(scores)
        scores = torch.matmul(scores, self.v)
        attention_weights = self.attention(scores)
        
        # Compute the weighted sum of values based on attention scores
        output = torch.matmul(attention_weights, x)
        
        return output

class GlobalContextAttention(nn.Module):
    def __init__(self, input_dim):
        super(GlobalContextAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.attention = nn.Softmax(dim=-1)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Calculate attention scores based on global context
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.attention(attention_scores)
        
        # Compute the weighted sum of values based on attention scores
        output = torch.matmul(attention_weights, value)
        
        return output
