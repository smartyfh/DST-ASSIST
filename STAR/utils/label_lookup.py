import torch
import torch.nn as nn
from .data_utils import slot_recovery


def get_label_ids(labels, tokenizer):
    label_ids = []
    label_lens = []
    max_len = 0
    for label in labels:
        if "-" in label:
            label = slot_recovery(label)
            
        label_token_ids = tokenizer(label)["input_ids"] # special tokens added automatically
        label_len = len(label_token_ids)
        max_len = max(max_len, label_len)
  
        label_ids.append(label_token_ids)
        label_lens.append(label_len)
    
    label_ids_padded = []
    for label_item_ids in label_ids:
        item_len = len(label_item_ids)
        padding = [0] * (max_len - item_len)
        label_ids_padded.append(label_item_ids + padding)
    label_ids_padded = torch.tensor(label_ids_padded, dtype=torch.long)

    return label_ids_padded, label_lens 


def get_label_lookup_from_first_token(labels, tokenizer, sv_encoder, device, use_layernorm=False):
    model_output_dim = sv_encoder.config.hidden_size
    
    sv_encoder.eval() 
    LN = nn.LayerNorm(model_output_dim, elementwise_affine=False)
    
    # get label ids
    label_ids, label_lens = get_label_ids(labels, tokenizer)

    # encoding
    label_type_ids = torch.zeros(label_ids.size(), dtype=torch.long)
    label_mask = (label_ids > 0)
    hid_label = sv_encoder(label_ids, label_mask, label_type_ids)[0]
    hid_label = hid_label[:, 0, :]
    hid_label = hid_label.detach()
    if use_layernorm:
        hid_label = LN(hid_label)
    label_lookup = hid_label.to(device)

    return label_lookup