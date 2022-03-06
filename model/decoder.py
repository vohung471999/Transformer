from .attention import MultiHeadAttention
from .embedding import NormalPositionalEmbedding, PositionalEmbedding
from .transformer_config import TransformerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.self_attention = MultiHeadAttention(config.decoder_attention_heads, config.model_dim, config.device, dropout=config.attention_dropout)
        self.self_attention_dropout = nn.Dropout(config.dropout)
        self.self_attention_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps, device=config.device, dtype=torch.float32)

        self.cross_attention = MultiHeadAttention(config.decoder_attention_heads, config.model_dim, config.device, dropout=config.attention_dropout)
        self.cross_attention_dropout = nn.Dropout(config.dropout)
        self.cross_attention_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps, device=config.device, dtype=torch.float32)

        self.linear_1 = nn.Linear(config.model_dim, config.decoder_ffn_dim, device=config.device, dtype=torch.float32)
        self.activation = F.gelu
        self.activation_dropout = nn.Dropout(config.activation_dropout)

        self.linear_2 = nn.Linear(config.decoder_ffn_dim, config.model_dim, device=config.device, dtype=torch.float32)
        self.ff_dropout = nn.Dropout(config.decoder_layer_dropout)

        self.ff_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps, device=config.device, dtype=torch.float32)

    def forward(self, hidden_states, encoder_outputs, decoder_self_attention_mask, decoder_cross_attention_mask):
        # self attention block
        residual = hidden_states
        hidden_states = self.self_attention(hidden_states, hidden_states, hidden_states, decoder_self_attention_mask)
        hidden_states = self.self_attention_dropout(hidden_states)

        # residual + normalization block
        hidden_states = residual + hidden_states
        hidden_states = self.self_attention_norm(hidden_states)

        # cross attention block
        residual = hidden_states
        hidden_states = self.cross_attention(hidden_states, encoder_outputs, encoder_outputs,
                                             decoder_cross_attention_mask)
        hidden_states = self.cross_attention_dropout(hidden_states)

        # residual + normalization block
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attention_norm(hidden_states)

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


class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig, embed_tokens):
        super().__init__()
        self.num_decoder_layers = config.num_decoder_layers
        self.word_embedding = embed_tokens
        self.positional_embedding = NormalPositionalEmbedding(config.model_dim, config.device, config.max_position_embeddings)
        # self.positional_embedding = PositionalEmbedding(embed_dim, dropout=dropout)
        self.norm_embedding = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps, device=config.device, dtype=torch.float32)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DecoderLayer(config=config)) for _ in range(self.num_decoder_layers)])

    def forward(self, decoder_input, encoder_hidden_states, decoder_self_attention_mask, decoder_cross_attention_mask):
        input_shape = decoder_input.size()

        # # embedding layer
        # word_embed = self.word_embedding(decoder_input)
        # hidden_states = self.positional_embedding(word_embed)
        # hidden_states = self.norm_embedding(hidden_states)

        # embedding layer
        word_embed = self.word_embedding(decoder_input)
        pos_embed = self.positional_embedding(input_shape)
        hidden_states = word_embed + pos_embed
        hidden_states = self.norm_embedding(hidden_states)

        # decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, decoder_self_attention_mask,
                                  decoder_cross_attention_mask)

        return hidden_states
