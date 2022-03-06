from .mask_creator import create_casual_mask, create_padding_mask, create_mask
import torch
import torch.nn.functional as F


def init_beam(model, text, text_mask, max_seq_len, start_token_id, num_beams, device):
    text = text.to(device)
    text_mask = text_mask.to(device)

    memory_state = model.encoder(text, text_mask)
    batch_size, text_length, model_dim = memory_state.shape

    summary = torch.LongTensor([[start_token_id]]).to(device)
    summary_casual_mask = create_casual_mask(summary, device)
    summary_padding_mask = create_padding_mask(summary, 1, device)
    summary_mask = summary_casual_mask.logical_and(summary_padding_mask)

    model_outputs = model.final_output(model.decoder(summary, memory_state, summary_mask, text_mask))
    log_scores, index = F.log_softmax(model_outputs, dim=-1).topk(num_beams)

    model_outputs = torch.zeros((num_beams, max_seq_len), dtype=torch.int32, device=device)
    model_outputs[:, 0] = start_token_id
    model_outputs[:, 1] = index[0]
    memory_state = memory_state.expand(num_beams, text_length, model_dim)

    return model_outputs, log_scores, memory_state


def select_k_top_candidate(model_outputs, prob, log_scores, i, num_beams):
    log_outputs = F.log_softmax(prob, dim=-1)

    log_probs, index = log_outputs[:, -1].topk(num_beams)
    log_probs = log_probs + log_scores.transpose(0, 1)
    log_probs, k_index = log_probs.view(-1).topk(num_beams)

    rows = torch.div(k_index, num_beams, rounding_mode='floor')
    cols = k_index % num_beams
    model_outputs[:, :i] = model_outputs[rows, :i]
    model_outputs[:, i] = index[rows, cols]

    log_scores = log_probs.unsqueeze(0)

    return model_outputs, log_scores


def beam(model, text, text_mask, max_seq_len, start_token_id, end_token_id, num_beams, device):
    max_seq_len = 1024 if max_seq_len > 1024 else max_seq_len
    chosen_text_index = 0
    model_outputs, log_scores, memory_state = init_beam(model, text, text_mask, max_seq_len, start_token_id, num_beams,
                                                        device)

    for i in range(2, max_seq_len):
        summary_casual_mask = create_casual_mask(model_outputs[:, :i], device)
        summary_padding_mask = create_padding_mask(model_outputs[:, :i], 1, device)
        summary_mask = summary_casual_mask.logical_and(summary_padding_mask)

        prob = model.final_output(model.decoder(model_outputs[:, :i], memory_state, summary_mask, text_mask))
        model_outputs, log_scores = select_k_top_candidate(model_outputs, prob, log_scores, i, num_beams)

        finished_sentences = (model_outputs == end_token_id).nonzero()
        mark_end_tokens = torch.zeros(num_beams, dtype=torch.int64, device=device)
        num_finished_sentences = 0

        for end_token in finished_sentences:
            sentence_ind, end_token_location = end_token
            if mark_end_tokens[sentence_ind] == 0:
                mark_end_tokens[sentence_ind] = end_token_location
                num_finished_sentences += 1

        if num_finished_sentences == num_beams:
            alpha = 0.7
            division = mark_end_tokens.type_as(log_scores) ** alpha
            _, chosen_text_index = torch.max(log_scores / division, 1)
            chosen_text_index = chosen_text_index[0]
            break

    text_length = (model_outputs[chosen_text_index] == end_token_id).nonzero()
    text_length = text_length[0] if len(text_length) > 0 else -1
    return model_outputs[chosen_text_index][:text_length + 1]


def beam_summarize(model: torch.nn.Module, tokenizer, text: str, device, num_beams: int = 5):
    model.eval()
    with torch.no_grad():
        text_encodings = tokenizer.batch_encode_plus([text], padding=True)
        text_ids = torch.tensor(text_encodings.get('input_ids'))
        num_tokens = text_ids.shape[1]
        text_mask = create_padding_mask(text_ids, 1, device)

        summary_tokens = beam(
            model, text_ids, text_mask, max_seq_len=int(num_tokens * 0.8), start_token_id=0, end_token_id=2,
            num_beams=num_beams, device=device).flatten()
        summary = tokenizer.decode(summary_tokens.tolist()).replace('<s>', '').replace('</s>', '').replace('<unk>', '')
        return summary