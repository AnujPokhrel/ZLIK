import torch
from pathlib import Path
from torch import nn, Tensor
from typing import Sequence, Optional, Tuple, Any
import math
import pdb
import os
import sys
sys.path.append(str(Path(".").resolve()))
from TrainingPipeline.model.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
    TimePositionalEmbedding
)

from TrainingPipeline.utils.helpers import load_damage_encoder_weights

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, flip=False):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(0.1)
        self.flip = flip

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.flip:
            x = x + torch.flip(self.pe[:x.size(1)], [0])
        else:
            x = x + self.pe[:x.size(1)]
        return self.dropout(x)
    
class CleanDynamicsEncoderDecoder(nn.Module):
    def __init__(self, model_cfg): 
        super().__init__()
        dmv_transformer = model_cfg["damaged_dynamics"]


        #configs initialization
        self.state_dim = dmv_transformer["state_dim"]
        self.action_dim = dmv_transformer["action_dim"]
        self.output_dim = dmv_transformer["output_dim"]
        self.latent_dim = dmv_transformer["latent_dim"]
        self.d_model = dmv_transformer["text_projector_latent_dim"]
        self.num_heads = dmv_transformer["num_heads"]
        self.num_encoder_layers = dmv_transformer["num_encoder_layers"]
        self.num_decoder_layers = dmv_transformer["num_decoder_layers"]
        self.feedforward_dim_encoder = dmv_transformer["feedforward_dim_encoder"]
        self.feedforward_dim_decoder = dmv_transformer["feedforward_dim_decoder"]
        self.history_length = dmv_transformer["horizon_length"]
        self.prediction_length = dmv_transformer["prediction_horizon"]
        self.data_frequency = dmv_transformer["data_frequency"]
        self.dropout = dmv_transformer["dropout"]

        self.data_length = int(self.history_length * self.data_frequency)

        # INPUT Projections
        self.history_projector = nn.Linear(self.state_dim + self.action_dim, self.d_model)
        self.action_embedding = nn.Linear(self.action_dim, self.d_model)

        self.output_embedding = nn.Linear(self.d_model, self.output_dim)

        #damage signature projector
        self.damage_projector = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            # nn.Linear(128, self.d_model),
            nn.Linear(64, self.d_model),
        )

        #Defining positional embeddings for both history and future actions
            #Fun fact: it was just self.data_length * 2 - 1 when not including the damage embedding 
            # self.data_length * 2 - 1 when we are not concating with the whole thing,
        self.history_pos_emb = SinusoidalPositionalEncoding(self.d_model, self.data_length, dropout=0.3)      
        self.action_pos_emb = SinusoidalPositionalEncoding(self.d_model, self.prediction_length, dropout=0.3)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.num_heads,
            dim_feedforward = 256,
            dropout = self.dropout,
            batch_first = True,
            norm_first = True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = self.num_encoder_layers
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=256,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=self.num_decoder_layers
        )

    def forward(self, history, future_actions, 
                history_padding_mask=None, action_padding_mask=None, tgt_mask=None):
        B = history.size(0)

        #project history
        history_embedding = self.history_projector(history)     #[B, H_len, d_model]

        # Add positional embedding
        history_w_pos_emb = self.history_pos_emb(history_embedding)

        # Prepare encoder padding mask
        if history_padding_mask is not None:
            encoder_padding_mask = ~history_padding_mask.bool()
        else:
            encoder_padding_mask = None
        
        # Encode History with damage context
        memory = self.transformer_encoder(
            history_w_pos_emb,
            src_key_padding_mask=encoder_padding_mask
        )  # [B, H_len + 1, d_model]


        # Future action, input to the decoder
        action_emb = self.action_embedding(future_actions)
        action_emb_w_pos = self.action_pos_emb(action_emb)

        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                future_actions.size(1), 
                device=future_actions.device)
        
        tgt_key_padding_mask = ~action_padding_mask.bool() if action_padding_mask is not None else None


        # Decode future poses with history context
        decoder_output = self.transformer_decoder(
            tgt=action_emb_w_pos,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
    
        #Pass it through another projector to get from 10x128 -> 10x6
        predicted_poses = self.output_embedding(decoder_output)

        return predicted_poses