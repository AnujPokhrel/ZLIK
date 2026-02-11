import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from einops import rearrange, repeat

from TrainingPipeline.model.cross_models.cross_encoder import Encoder
from TrainingPipeline.model.cross_models.cross_decoder import Decoder
from TrainingPipeline.model.cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from TrainingPipeline.model.cross_models.cross_embed import DSW_embedding

from math import ceil

class CrossformerWithControl(nn.Module):
    def __init__(self, 
                data_dim,
                in_len, 
                out_len, 
                seg_len, 
                damage_dim = 128,   #size of the damage embedding
                action_dim = 2,
                win_size = 4,
                factor=10, 
                d_model=256, 
                d_ff = 521, 
                n_heads=4, 
                e_layers=3, 
                dropout=0.2, 
                baseline = False, 
                device=torch.device('cuda:0')):

        super(CrossformerWithControl, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.baseline = baseline
        self.device = device
        self.merge_win = win_size



        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        self.damage_projection = nn.Linear(damage_dim, d_model)

        self.dec_action_proj = nn.Linear(action_dim * seg_len, d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout, in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
    def forward(self, x_seq, future_actions, damage_embedding):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0

        batch_size = x_seq.shape[0]

        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        
        #inject damage
        damage_emb = self.damage_projection(damage_embedding)  # (B, D_model)
        damage_emb = damage_emb.unsqueeze(1).unsqueeze(1)  

        x_seq = x_seq + damage_emb
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        #inject future actions
        out_len_add = self.pad_out_len - self.out_len
        if out_len_add > 0:
            pad_action = future_actions[:, -1:, :].expand(-1, out_len_add, -1)
            future_actions = torch.cat((future_actions, pad_action), dim = 1)
        
        actions_segmented = rearrange(future_actions, 'b (seg_num seg_len) d -> b seg_num (seg_len d)', seg_len=self.seg_len)
        action_emb = self.dec_action_proj(actions_segmented)  # (B, L, D_model)

        action_emb = action_emb.unsqueeze(1)  # (B, 1, L, D_model)
        dec_in = dec_in + action_emb


        predict_y = self.decoder(dec_in, enc_out)
        return base + predict_y[:, :self.out_len, :]