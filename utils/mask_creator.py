import numpy as np
import torch
from torch.autograd import Variable

def create_padding_mask(sequence, padding_token, device):
    batch_size, inputs_len = sequence.size()
    mask = (sequence != padding_token)
    mask = mask[:, None, None, :].expand(batch_size, 1, 1, inputs_len)
    mask = mask.to(device)
    return mask


def create_casual_mask(sequence, device):
    batch_size, input_len = sequence.size()
    casual_mask = np.triu(np.ones((batch_size, input_len, input_len)), k=1).astype('uint8')
    casual_mask = Variable(torch.from_numpy(casual_mask) == 0)
    casual_mask = casual_mask.unsqueeze(1)
    casual_mask = casual_mask.to(device)
    return casual_mask


def create_mask(encoder_inputs, decoder_inputs, padding_token, device):
    encoder_attention_mask = create_padding_mask(encoder_inputs, padding_token, device)

    decoder_padding_mask = create_padding_mask(decoder_inputs, padding_token, device)
    decoder_casual_mask = create_casual_mask(decoder_inputs, device)

    decoder_self_attention_mask = decoder_casual_mask.logical_and(decoder_padding_mask)

    return encoder_attention_mask, decoder_self_attention_mask
