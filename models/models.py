# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .help_layers import TransformerEncoderLayer
from data_loading.pretrained_extractors import CustomMambaBlock

class EmotionPersonalityModel(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, mamba_layer_number=2, mamba_d_model=256, positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.emo_proj = nn.Sequential(
            nn.Linear(input_dim_emotion, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.per_proj = nn.Sequential(
            nn.Linear(input_dim_personality, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.emotion_encoder = nn.ModuleList([
            CustomMambaBlock(hidden_dim, mamba_d_model, dropout=dropout)
            for _ in range(mamba_layer_number)
        ])

        self.personality_encoder = nn.ModuleList([
            CustomMambaBlock(hidden_dim, mamba_d_model, dropout=dropout)
            for _ in range(mamba_layer_number)
        ])

        self.emotion_to_personality_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.personality_to_emotion_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])
        
        self.tr_layer_number = tr_layer_number
        self.mamba_layer_number = mamba_layer_number

        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

        self.personality_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )

        self.emotion_personality_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits+num_emotions)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, emotion_input=None, personality_input=None, mode='emotion'):
        """
        mode: 'emotion', 'personality', or 'fusion'
        Returns:
            - mode == 'emotion': emotion logits (B, T, num_emotions)
            - mode == 'personality': personality outputs (B, T, num_traits)
            - mode == 'fusion': (emotion_logits, personality_preds, multitask_output)
        """
        if emotion_input is not None:
            emo = self.emo_proj(emotion_input)  # (B, T, hidden_dim)
        if personality_input is not None:
            per = self.per_proj(personality_input)

        if mode == 'emotion':
            for layer in self.emotion_encoder:
                emo = layer(emo)
            out_emo = self.emotion_fc_out(emo.mean(dim=1))  # (B, num_emotions)
            return {'emotion_logits': out_emo}

        elif mode == 'personality':
            for layer in self.personality_encoder:
                per = layer(per)
            out_per = self.personality_fc_out(per.mean(dim=1))  # (B, num_traits)
            return {'personality_scores': self.sigmoid(out_per)}

        elif mode == 'fusion':
            # with torch.no_grad():
            for layer in self.emotion_encoder:
                emo = layer(emo)
            for layer in self.personality_encoder:
                per = layer(per)

            for layer in self.emotion_to_personality_attn:
                emo_att = layer(emo, per, per)
                emo += emo_att

            for layer in self.personality_to_emotion_attn:
                per_att = layer(per, emo, emo)
                per += per_att

            fused = torch.cat([emo, per], dim=-1)   # (B, T, 2 * hidden_dim)
            multitask_out = self.emotion_personality_fc_out(fused.mean(dim=1))  # (B, num_emotions + num_traits)
            return {'emotion_logits': multitask_out[:, :7], 'personality_scores': self.sigmoid(multitask_out[:, -5:])}

        else:
            raise ValueError(f"Invalid mode: {mode}")
