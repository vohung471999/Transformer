import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from transformer_config import TransformerConfig


class WordEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=config.pad_token_id, device=config.device, dtype=torch.float32)

    def forward(self, x):
        return self.embed(x)


class NormalPositionalEmbedding(nn.Embedding):

    def __init__(self, embedding_dim: int, device, num_embeddings=1024):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, device=device, dtype=torch.float32)

    def forward(self, input_ids_shape: torch.Size):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class PositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.model_dim = config.model_dim
        self.dropout = nn.Dropout(config.dropout)

        positional_embedding = torch.zeros(config.max_position_embeddings, config.model_dim)
        position = torch.arange(0, config.max_position_embeddings).unsqueeze(1)
        w = torch.exp(torch.arange(0, config.model_dim, 2) * (-math.log(10000) / config.model_dim))

        positional_embedding[:, 0::2] = torch.sin(position * w)
        positional_embedding[:, 1::2] = torch.cos(position * w)

        positional_embedding = positional_embedding.unsqueeze(0)
        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, embedding):
        embedding = embedding * math.sqrt(self.model_dim)
        seq_len = embedding.size(1)

        positional_embedding = Variable(self.positional_embedding[:, :seq_len], requires_grad=False)
        embedding = embedding + positional_embedding

        return self.dropout(embedding)
