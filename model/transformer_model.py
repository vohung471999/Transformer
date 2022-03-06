from .encoder import Encoder
from .decoder import Decoder
from .embedding import WordEmbedding
from .transformer_config import TransformerConfig
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=config.pad_token_id, device=config.device, dtype=torch.float32)
        self.encoder = Encoder(config, self.word_embedding)
        self.decoder = Decoder(config, self.word_embedding)
        self.final_output = nn.Linear(config.model_dim, config.vocab_size, device=config.device, dtype=torch.float32)

    def forward(self, encoder_inputs, decoder_inputs, encoder_attention_mask, decoder_self_attention_mask):

        encoder_hidden_states = self.encoder(encoder_inputs, encoder_attention_mask)
        final_hidden_states = self.decoder(decoder_inputs, encoder_hidden_states,  decoder_self_attention_mask, encoder_attention_mask)

        output = self.final_output(final_hidden_states)
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
