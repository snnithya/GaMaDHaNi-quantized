from random import randint
from typing import Dict, Optional

import numpy as np
import torch

import pdb

TensorDict = Dict[str, torch.Tensor]

def pitch_read_downsample(inputs: TensorDict,
                          seq_len: int, 
                          decoder_key: str, 
                          min_norm_pitch: int,  
                          time_downsample: int=1, 
                          pitch_downsample: int=1,
                          base_tonic: float=440.,
                          transpose_pitch: Optional[int]=0,
                          start: Optional[int]=None):
        data = inputs[decoder_key]["data"]
        if seq_len is not None:
            start = start if start is not None else randint(0, max(0, data.shape[0] - seq_len * time_downsample - 1))
            end = start + seq_len * time_downsample
            
            f0 = data[start : end+1 : time_downsample].clone()
        else:
            f0 = data.clone()

        # normalizing pitch contour from hertz to cents
        f0[f0 == 0] = np.nan
        norm_f0 = f0.clone().numpy()
        norm_f0[~np.isnan(norm_f0)] = (1200) * np.log2(norm_f0[~np.isnan(norm_f0)] / base_tonic)
        del f0

        #discretizing and making sure the values are positive
        norm_f0[~np.isnan(norm_f0)] = np.around(norm_f0[~np.isnan(norm_f0)])
        norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] - (min_norm_pitch)

        #pitch downsampling
        norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] // pitch_downsample + 1    # adding 1 to preserve 0 as a silence token 
        
        # data augmentation, if apply_transform is set to True in config
        if transpose_pitch:
            transposed_values = norm_f0[~np.isnan(norm_f0)] + (transpose_pitch//pitch_downsample)
            norm_f0[~np.isnan(norm_f0)] = transposed_values
        
        # add silence token of 0
        norm_f0[np.isnan(norm_f0)] = 0

        input_tokens = norm_f0[:-1, None].copy()
        target_tokens = norm_f0[1:, None].copy()
        return {
            "decoder_inputs": input_tokens,
            "decoder_targets": target_tokens,
            "sampled_sequence": norm_f0
        }

def invert_pitch_read_downsample(f0,
                          min_norm_pitch: int,  
                          time_downsample: int=1, 
                          pitch_downsample: int=1,
                          base_tonic: float=440.,
                          seq_len: int=None, 
                          decoder_key: str=None):
    f0[f0 == 0] = np.nan
    f0[~np.isnan(f0)] = ((f0[~np.isnan(f0)] - 1) * pitch_downsample)
    f0[~np.isnan(f0)] = f0[~np.isnan(f0)] + min_norm_pitch
    # Unnormalize the pitch contours
    f0[~np.isnan(f0)] = base_tonic * (2**(f0[~np.isnan(f0)] / 1200))
    
    return f0