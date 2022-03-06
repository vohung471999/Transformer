from .attention import MultiHeadAttention
from .embedding import NormalPositionalEmbedding, PositionalEmbedding
from .transformer_config import TransformerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attention = MultiHeadAttention(config.encoder_attention_heads, config.model_dim, config.device, dropout=config.attention_dropout)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps, device=config.device, dtype=torch.float32)

        self.linear_1 = nn.Linear(config.model_dim, config.encoder_ffn_dim, device=config.device, dtype=torch.float32)
        self.activation = F.gelu
        self.activation_dropout = nn.Dropout(config.activation_dropout)

        self.linear_2 = nn.Linear(config.encoder_ffn_dim, config.model_dim, device=config.device, dtype=torch.float32)
        self.ff_dropout = nn.Dropout(config.encoder_layer_dropout)

        self.ff_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps, device=config.device, dtype=torch.float32)

    def forward(self, hidden_states, encoder_attention_mask):
        # attention block
        residual = hidden_states
        hidden_states = self.attention(hidden_states, hidden_states, hidden_states, encoder_attention_mask)
        hidden_states = self.attention_dropout(hidden_states)

        # residual + normalization block
        hidden_states = residual + hidden_states
        hidden_states = self.attention_norm(hidden_states)

        # feed forward block
        residual = hidden_states
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ff_dropout(hidden_states)

        # residual + norm block
        hidden_states = residual + hidden_states
        hidden_states = self.ff_norm(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig, embed_tokens):
        super().__init__()
        self.num_encoder_layers = config.num_encoder_layers
        self.word_embedding = embed_tokens
        self.positional_embedding = NormalPositionalEmbedding(config.model_dim, config.device, config.max_position_embeddings)
        # self.positional_embedding = PositionalEmbedding(embed_dim, dropout=dropout)
        self.norm_embedding = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps, device=config.device, dtype=torch.float32)
        self.layers = nn.ModuleList(
            [copy.deepcopy(EncoderLayer(config=config)) for _ in range(self.num_encoder_layers)])

    def forward(self, encoder_inputs, encoder_attention_mask):
        input_shape = encoder_inputs.size()

        # # embedding layer
        # word_embed = self.word_embedding(encoder_inputs)
        # hidden_states = self.positional_embedding(word_embed)
        # hidden_states = self.norm_embedding(hidden_states)

        # embedding layer
        word_embed = self.word_embedding(encoder_inputs)
        pos_embed = self.positional_embedding(input_shape)
        hidden_states = word_embed + pos_embed
        hidden_states = self.norm_embedding(hidden_states)

        # encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_attention_mask=encoder_attention_mask)

        return hidden_states
