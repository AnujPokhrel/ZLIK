import torch
import torch.nn as nn
from functools import partial
from einops import rearrange

from TrainingPipeline.model.dmv_behavior_model import DamagedVehicleBehaviorModel
from TrainingPipeline.utils.helpers import load_damage_encoder_weights
from TrainingPipeline.model.cross_models.cross_former import CrossformerWithControl

class DamagedCrossformer(nn.Module):
    def __init__(self, encoder_cfg, model_cfg, device):
        super(DamagedCrossformer, self).__init__()
        self.cross_cfg = model_cfg["crossformer_params"]
        print(f"Initializing Damage Encoder from: {self.cross_cfg['pre_train_chk_path']}")
        damage_behavior_model = DamagedVehicleBehaviorModel(encoder_cfg)
        self.damage_encoder = damage_behavior_model.damage_encoder.to(device)

        damage_encoder_state_dict = load_damage_encoder_weights(self.damage_encoder, 
                                        self.cross_cfg['pre_train_chk_path'], 
                                        device) 
        self.damage_encoder.load_state_dict(damage_encoder_state_dict)

        for param in self.damage_encoder.parameters():
            param.requires_grad = False

        self.crossformer = CrossformerWithControl(
            data_dim = self.cross_cfg['input_dim'], 
            in_len = self.cross_cfg['in_len'],
            out_len = self.cross_cfg['out_len'],
            seg_len = self.cross_cfg['seg_len'],
            damage_dim = self.cross_cfg['damage_dim'],
            action_dim = self.cross_cfg['action_dim'],
            win_size = self.cross_cfg['win_size'],
            factor = self.cross_cfg['factor'],
            d_model = self.cross_cfg['d_model'],
            d_ff = self.cross_cfg['d_ff'],
            n_heads = self.cross_cfg['n_heads'],
            e_layers = self.cross_cfg['e_layers'],
            dropout = self.cross_cfg['dropout'],
            device = device
        )

    def forward(self, history, future_actions, damage_embedding):
        damage_token = self.damage_encoder(damage_embedding)

        prediction_all_dims = self.crossformer(history, future_actions, damage_token)
        predicted_pose = prediction_all_dims[:, :, :self.cross_cfg['output_dim']]

        return predicted_pose