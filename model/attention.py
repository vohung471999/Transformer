import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, device, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attention_weight = None

        self.query_project = nn.Linear(embed_dim, embed_dim, device=device, dtype=torch.float32)
        self.key_project = nn.Linear(embed_dim, embed_dim, device=device, dtype=torch.float32)
        self.value_project = nn.Linear(embed_dim, embed_dim, device=device, dtype=torch.float32)
        self.out_matrix = nn.Linear(embed_dim, embed_dim, device=device, dtype=torch.float32)

        self.dropout = nn.Dropout(dropout)

    def _self_attention(self, query, key, value, attention_mask=None, dropout=None):
        """
        q: batch_size x heads x seq_length x d_model
        k: batch_size x heads x seq_length x d_model
        v: batch_size x heads x seq_length x d_model
        attention_mask: batch_size x 1 x seq_length
        output: batch_size x head x seq_length x d_model
        """

        batch_size, num_of_heads, seq_length, dim_head = query.shape
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * (dim_head ** -0.5)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_scores = nn.functional.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        attention_output = torch.matmul(attention_scores, value)
        return attention_output, attention_scores

    def _shape(self, tensor: torch.Tensor, sequence_length: int, batch_size: int):
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                query,
                key,
                value,
                attention_mask=None):

        batch_size, tgt_length, _ = query.shape
        _, src_length, _ = key.shape

        q = self.query_project(query)
        k = self.key_project(key)
        v = self.value_project(value)

        # change shape to (batch_size, number_of_heads, sequence_length, dim_head)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output, self.attention_weight = self._self_attention(q, k, v, attention_mask, self.dropout)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, tgt_length, self.embed_dim)
        attention_output = self.out_matrix(attention_output)

        return attention_output

