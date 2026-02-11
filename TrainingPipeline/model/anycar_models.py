import torch
from torch import nn, Tensor
from typing import Sequence, Optional, Tuple, Any
import math
import pdb

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
        seq_len = x.size(1)
        if self.flip:
            x = x + torch.flip(self.pe[:seq_len], [0])
        else:
            x = x + self.pe[:seq_len]
        return self.dropout(x)
    
class AnyCarTransformerDecoder(nn.Module):
    def __init__(self, cfg, device): 
        super().__init__()
        anycar_model_cfg = cfg["anycar_transformer"]

        self.state_dim = anycar_model_cfg["state_dim"]
        self.action_dim = anycar_model_cfg["action_dim"]
        self.output_dim = anycar_model_cfg["output_dim"]
        self.latent_dim = anycar_model_cfg["latent_dim"]
        self.num_heads = anycar_model_cfg["num_heads"]
        self.num_layers = anycar_model_cfg["num_layers"]
        self.history_length = anycar_model_cfg["horizon_length"]
        self.prediction_length = anycar_model_cfg["prediction_horizon"]
        self.data_frequency = anycar_model_cfg["data_frequency"]
        self.dropout = anycar_model_cfg["dropout"]

        self.data_length = int(self.history_length * self.data_frequency)
        self.context_seq_len = self.data_length * 2 - 1

        self.state_embedding = nn.Linear(self.state_dim, self.latent_dim)
        self.action_embedding = nn.Linear(self.action_dim, self.latent_dim)
        self.output_embedding = nn.Linear(self.latent_dim, self.output_dim)
        
        self.history_pos_emb = LearnedPositionalEncoding(self.latent_dim, self.context_seq_len, flip=True)
        self.action_pos_emb = LearnedPositionalEncoding(self.latent_dim, self.prediction_length)


        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=512,    
                                                                dropout=self.dropout, 
                                                                batch_first=True)

        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, 
                                                        num_layers=self.num_layers)

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None):
        B = history.size(0)
        T = self.history_length
        state_emb = self.state_embedding(history[:, :, :self.state_dim])

        hist_action_emb = self.action_embedding(history[:, :, self.state_dim:])

        #Manual interleaving
        history_emb = torch.zeros(B, self.context_seq_len, self.latent_dim, device=history.device)
        history_emb[:, ::2] = state_emb

        if T > 1:
            history_emb[:, 1::2] = hist_action_emb[:, :T - 1, :]

        #positional encoding for history
        history_emb = self.history_pos_emb(history_emb)

        # Future action, input to the decoder
        action_emb = self.action_embedding(action)
        action_emb = self.action_pos_emb(action_emb)

        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(action.size(1), device=action.device)
        
        tgt_key_padding_mask = None
        if action_padding_mask is not None:
            tgt_key_padding_mask = (action_padding_mask == 0).bool()

        memory_key_padding_mask = None
        if history_padding_mask is not None:
            # history_padding_mask is [B, T]. True = Masked (Ignore).
            interleaved_mask = torch.zeros(B, self.context_seq_len, dtype=torch.bool, device=history.device)
            
            # Mask states corresponding to masked steps
            interleaved_mask[:, ::2] = history_padding_mask.bool()
            
            # Mask actions corresponding to masked steps (up to T-1)
            if T > 1:
                interleaved_mask[:, 1::2] = history_padding_mask[:, :-1].bool()
            
            memory_key_padding_mask = interleaved_mask
        
        x = self.transformer_decoder(
            tgt=action_emb,
            memory=history_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        x = self.output_embedding(x)
        return x