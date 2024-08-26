import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from torch.utils.data import DataLoader
from tqdm import tqdm



class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128, 64], dropout_prob=0.2):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.LayerNorm(h_dim))  # Use LayerNorm instead of BatchNorm1d
            encoder_layers.append(nn.LeakyReLU(0.1))      
            encoder_layers.append(nn.Dropout(dropout_prob)) 
            current_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoder_hidden_dims = hidden_dims[::-1]
        for h_dim in decoder_hidden_dims[1:]:
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.LayerNorm(h_dim))  # Use LayerNorm instead of BatchNorm1d
            decoder_layers.append(nn.LeakyReLU(0.1))      
            decoder_layers.append(nn.Dropout(dropout_prob)) 
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded) + x  
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)







    