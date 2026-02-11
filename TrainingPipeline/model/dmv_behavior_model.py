import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from sentence_transformers import SentenceTransformer
import pdb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pdb
from TrainingPipeline.model.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
    TimePositionalEmbedding
)

class DamageEncoder(nn.Module):
    """Project a frozen text embedding to a representation space (proj_dim)
    This is g_θ(DSR).
    """

    def __init__(self, cfg):
        super().__init__()
        text_dim = cfg["model"]["text_dim"]
        proj_dim = cfg["model"]["signature_dim"]

        self.net = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier/Glorot initialization is stable for projection heads
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, text_emb):
        z = self.net(text_emb)
        return z

class TransformerBehaviorEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        tf_cfg = cfg["transformer_encoder"]

        hz = int(data_cfg["data_frequency"])
        window_secs = data_cfg["horizon_length"]
        pose_dim = int(model_cfg["pose_dim"])
        act_dim = int(model_cfg["action_dim"])
        
        self.time_step = float(1 / hz)
        self.L = hz * window_secs
        self.feature_dim= pose_dim + 2 * act_dim
        self.d_model = int(tf_cfg["hidden_dimension"])
        self.n_head = int(tf_cfg["attention_heads"])
        self.n_layers = int(tf_cfg["transformer_encoder_layers"])
        self.dim_feedforward = int(tf_cfg["feedforward_network_dimension"])
        self.dropout_p = float(tf_cfg["dropout"])
        self.positional_encoding_type = str(tf_cfg["positional_encoding_type"])

        # # F_in = pose_dim + act_dim * 2
        in_dim = self.L * self.feature_dim 
        proj_dim = int(cfg["model"]["signature_dim"])

        self.input_projection = nn.Linear(self.feature_dim, self.d_model)

        # 2. Positional Encoding
        # self.pos_encoder = nn.Parameter(torch.zeros(1, self.L, self.d_model))
        # nn.init.xavier_uniform_(self.pos_encoder)

        if self.positional_encoding_type == "sinusoidal":
            self.positional_embedding = SinusoidalPositionalEncoding(
                max_len=self.L,
                d_model=self.d_model,
                dropout=self.dropout_p
            )
        elif self.positional_encoding_type == "learnable":
            self.positional_embedding = LearnablePositionalEncoding(
                max_len=self.L,
                d_model=self.d_model,
                dropout=self.dropout_p
            )
        elif self.positional_encoding_type == "time":
            self.positional_embedding = TimePositionalEmbedding(
                max_len=window_secs,
                d_model=self.d_model
            )
        else:
            self.positional_embedding = None

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.n_head,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=self.dropout_p
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.n_layers
        )

        # 4. Final Projection (Aggregation and MLP Head)
        # Use a simple Average Pooling across the sequence dimension (L)
        self.pooling = nn.AdaptiveAvgPool1d(1) 
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, proj_dim)
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, delta):
        x = self.input_projection(delta)  # [B, L, d_model]

        if self.positional_encoding_type == "time":
            time_sequence = torch.arange(
                0, self.L * self.time_step, self.time_step, device=delta.device
            ).unsqueeze(0).repeat(delta.size(0), 1)
        
            pos_emb = self.positional_embedding(time_sequence)
            x = x + pos_emb

        elif self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # x = x + self.pos_encoder         # [B, L, d_model] (Adding learned positional embeddings)

        # 2. Pass through Transformer
        x = self.transformer_encoder(x)  # [B, L, d_model]

        # 3. Aggregate sequence information (Pooling)
        # Permute for 1D pooling: [B, d_model, L]
        x = x.permute(0, 2, 1) 
        x = self.pooling(x).squeeze(-1)  # [B, d_model]

        # 4. Final Projection Head
        z = self.projection_head(x)      # [B, proj_dim]

        # return F.normalize(z, dim=-1) # [B, proj_dim]
        return z

class DamagedVehicleBehaviorModel(nn.Module):
    """Model for pretraining with contrastive loss.
    Consists of:
    - DamageEncoder: text embedding -> proj_dim
    - BehaviorEncoder: window of [Δpose||action] tokens -> proj_dim
    """

    def __init__(self, cfg):
        super().__init__()
        self.damage_encoder = DamageEncoder(cfg)
        self.behavior_encoder = TransformerBehaviorEncoder(cfg)

    def forward(self, delta, text_emb):
        # delta: [B, L, pose_dim + del_act_dim + act_dim]
        # text_emb: [B, text_dim]
        z_sig = self.behavior_encoder(delta)      # [B, proj_dim]
        z_txt = self.damage_encoder(text_emb)     # [B, proj_dim]
        return z_sig, z_txt
    

class TextEmbeddings():
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        if self.cfg['model']["embedding_model"] == "embedding_gemma":
            self.embedding_model = SentenceTransformer("google/embeddinggemma-300m").to(device=device) 


    def batch_encode(self, texts, device, batch_size=32):
        embs = []
        if self.cfg['model']["embedding_model"] == "embedding_gemma":
            # pdb.set_trace()
            for i in range(0, len(texts), batch_size):
                chunk = texts[i: i+batch_size]
                with torch.no_grad():
                    out = self.embedding_model.encode(chunk)

        return out