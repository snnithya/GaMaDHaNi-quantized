from src.dataset import Task
from src.model_diffusion import UNet, UNetPitchConditioned
from src.model_transformer import XTransformerPrior

from functools import partial
import joblib
import gin
import json

import torch
import numpy as np


def load_pitch_model(config, ckpt, qt = None, prime_file=None, dataset_split_file=None, model_type = None, number_of_samples=None):
    gin.parse_config_file(config)
    assert model_type is not None, 'model_type argument is not passed for the pitch generator model, choose either diffusion or transformer'
    if model_type=="diffusion":
        model = UNet()
    elif model_type=="transformer":
        model = XTransformerPrior()
    
    ckpt = torch.load(ckpt, map_location="cuda", weights_only=True)
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)  
    model.to('cuda')

    # quantile transform file for the diffusion model
    if qt is not None:
        qt = joblib.load(qt)

    # primed generation using the primes from prime_file
    if prime_file is not None:
        primes = np.load(prime_file, allow_pickle=True)['concatenated_array'][:, 0]
        if number_of_samples < len(primes):
            primes = primes[:number_of_samples]
    # unprimed generation 
    else:
        primes = None
    return model, qt, primes

def load_audio_model(config, ckpt, qt = None):
    gin.parse_config_file(config)
    model = UNetPitchConditioned() # there are no gin parameters for some reason
    ckpt = torch.load(ckpt, weights_only=True)
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)  
    model.to('cuda')
    if qt is not None:
        qt = joblib.load(qt)

    return model, qt
