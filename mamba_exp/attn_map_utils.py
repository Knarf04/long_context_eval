import os
import torch
import torch.nn.functional as F
import functools

import mamba_ssm.ops.triton.ssd_combined as ssd_combined
import matplotlib.pyplot as plt

# TODO: Set storing directories for the logits
LOGIT_DIR = '/logit/dir'

def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum

def ssd_attn_map(dt, A, B, C, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    if dt_limit:
        dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

    A = (A*dt).permute([0, 2, 1])
    L = torch.exp(segment_sum(A))
    M = torch.einsum("blhn, bshn, bhls, bsh -> blhs", C, B, L, dt)    # Full Attention Map of Mamba2

    # Step-wise Einsum
    # TODO: convert to a chunk-wise algorithm to reduce memory pressure
    # CB = torch.einsum("blhn, bshn -> blhs", C, B)
    # CBL = torch.einsum("blhs, bhls -> blhs", CB, L)
    # M_split = torch.einsum("blhs, bsh -> blhs", CBL, dt)
    
    return M, L, dt

def _ssd_attn_map_decorator(func, store_logits=False, compute_attn_map=False):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if store_logits:
            logit_dict = {}
            logit_dict['dt'] = args[1]
            logit_dict['A'] = args[2]
            logit_dict['B'] = args[3]
            logit_dict['C'] = args[4]
            logit_dict['dt_bias'] =kwargs['dt_bias']
            seq_len = args[1].shape[1]
            # print("Saving logits...")
            torch.save(logit_dict, os.path.join(LOGIT_DIR, f'logit_dict_{seq_len}_{decorator.layer_idx}.pt'))
        if compute_attn_map:
            M, _, _ = ssd_attn_map(args[1], args[2], args[3], args[4], kwargs['dt_bias'], kwargs['dt_softplus'], kwargs['dt_limit'])
            decorator.ssd_attn_map = M
        decorator.layer_idx += 1
        if decorator.layer_idx in [9, 18, 27]:
            decorator.layer_idx += 1 # skip attention layers
        return func(*args, **kwargs)
    decorator.layer_idx = 0
    decorator.ssd_attn_map = None
    return decorator

def toggle_decorator(output_attentions=False, store_logits=False, compute_attn_map=False):
    func = ssd_combined._mamba_chunk_scan_combined_fwd
    if output_attentions:
        if not hasattr(func, '__wrapped__'):
            ssd_combined._mamba_chunk_scan_combined_fwd = _ssd_attn_map_decorator(func, store_logits, compute_attn_map)
        else:
            ssd_combined._mamba_chunk_scan_combined_fwd.layer_idx = 0
            ssd_combined._mamba_chunk_scan_combined_fwd.ssd_attn_map = None
    elif (not output_attentions) and hasattr(func, '__wrapped__'):
        ssd_combined._mamba_chunk_scan_combined_fwd = func.__wrapped__

def get_ssd_attn_map():
    if hasattr(ssd_combined._mamba_chunk_scan_combined_fwd, "ssd_attn_map"):
        return ssd_combined._mamba_chunk_scan_combined_fwd.ssd_attn_map
    else: 
        return None
    
# print the attention map of transformer
def gqa_attn_map(query_states, key_states, attention_mask, scaling):
    with torch.no_grad():
        B, H_q, L, d = query_states.shape
        H_k = key_states.shape[1]
        group_size = H_q // H_k
        query_grouped = query_states.view(B, H_k, group_size, L, d)
        key_expanded_t = key_states.unsqueeze(2).transpose(-1, -2)
        scores_scaled = torch.einsum("bghid, bghdj -> bghij", query_grouped, key_expanded_t) * scaling
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.float32, device=query_states.device)).unsqueeze(0)

        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.float32)
            causal_mask *= attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2) 

        scores_scaled = scores_scaled.masked_fill(causal_mask==0, float('-inf'))
        attn_map_grouped = F.softmax(scores_scaled, dim=-1)
        attn_weights = attn_map_grouped.view(B, H_q, L, L)
    return attn_weights
