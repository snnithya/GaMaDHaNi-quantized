import torch
from torch import nn
import logging

def pseudo_quantize_tensor_conv1d(w, n_bit=4):
    org_w_shape = w.shape  # Original shape: (out_channels, in_channels, kernel_size)

    # Calculate the maximum (\alpha) and minimum (\beta) in the tensor
    max_val = w.amax() 
    min_val = w.amin()  

    # Calculate scale factor and zero point
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)

    # Quantize weights
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)

    # Dequantize weights (pseudo-quantization)
    w = (w - zeros) * scales

    # Reshape back to original shape
    assert w.shape == org_w_shape, "Weight shape should match the original shape after quantization; Expected: {}, Got: {}".format(org_w_shape, w.shape)
    return w

@torch.no_grad()
def pseudo_quantize_model_weight_conv1d(model, w_bit, q_group_size):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data = pseudo_quantize_tensor_conv1d(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)
