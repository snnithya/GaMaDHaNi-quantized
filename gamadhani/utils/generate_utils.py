from gamadhani.src.model_diffusion import UNet, UNetPitchConditioned
from gamadhani.src.model_transformer import XTransformerPrior
import gamadhani.utils.pitch_to_audio_utils as p2a

import joblib
import gin

import torch
import numpy as np
from typing import Optional, Callable, Tuple, Any
from functools import partial
import os


def load_pitch_model(config, ckpt, qt = None, prime_file=None, model_type = None, number_of_samples=None):
    gin.parse_config_file(config)
    if model_type is None:
        raise ValueError('model_type argument is not passed for the pitch generator model, choose either diffusion or transformer')
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

def load_pitch_fns(pitch_path: str, model_type: str, prime: bool = False, prime_file: Optional[str] = None, qt_path: Optional[str] = None, number_of_samples: int = 16, config_path: Optional[str] = None) -> Tuple[Any, Optional[Any], Callable, Callable, Optional[torch.Tensor]]:
    config_path = os.path.join(pitch_path, 'config.gin') if not config_path else config_path
    ckpt = os.path.join(pitch_path, 'models', 'last.ckpt') if os.path.isdir(pitch_path) else pitch_path
    
    if prime and not prime_file:
        raise ValueError("Error: If 'prime' is True, 'prime_file' must be provided.")
    gin.parse_config_file(config_path)  

    seq_len = gin.query_parameter("%SEQ_LEN")
    time_downsample = gin.query_parameter('src.dataset.Task.kwargs').get("time_downsample")
    seq_len_cache = int((1/3) * seq_len)
    
    pitch_model, pitch_qt, primes = load_pitch_model(
                                            config = config_path, 
                                            ckpt = ckpt, 
                                            qt = qt_path,
                                            prime_file = prime_file,
                                            model_type = model_type,
                                            number_of_samples=number_of_samples
                                            )
    
    if not prime:
        #unprimed generation
        if model_type=="transformer":
            #get start token (sampling a start token from a popular range in validation set) -> primes
            primes = np.random.randint(low=154,high=222, size=(number_of_samples,1)).astype(np.float64)
        else:
            primes = None

    Task_ = gin.get_configurable('src.dataset.Task')
    task_obj = Task_()
    pitch_task_fn = partial(task_obj.read_) 
    invert_pitch_task_fn = partial(task_obj.invert_) 
    processed_primes = None
    if prime:
        if model_type=="diffusion":
            pitch_task_fn = partial(pitch_task_fn, qt_transform = pitch_qt, add_noise_to_silence = True,
            seq_len = None)
            invert_pitch_task_fn = partial(invert_pitch_task_fn, qt_transform = pitch_qt)
        
        processed_primes = [torch.tensor(pitch_task_fn(
                    **{"inputs": {"pitch": {"data": torch.tensor(p[:seq_len_cache*time_downsample:])}}})["sampled_sequence"]) for p in primes]    
        primes = torch.stack(processed_primes)

    else:
        if model_type=="transformer":
            processed_primes = [torch.tensor(pitch_task_fn(
                **{"inputs": {"pitch": {"data": torch.tensor(p)}}})["sampled_sequence"]) for p in primes]
            primes = torch.stack(processed_primes)
        else:
            invert_pitch_task_fn = partial(invert_pitch_task_fn, qt_transform = pitch_qt)
            primes = None
    return pitch_model, pitch_qt, pitch_task_fn, invert_pitch_task_fn, primes

def load_audio_fns(audio_path, qt_path: Optional[str] = None, config_path=None):
    ckpt = os.path.join(audio_path, 'models', 'last.ckpt') if os.path.isdir(audio_path) else audio_path
    config = config_path if config_path else os.path.join(audio_path, 'config.gin')
    qt = qt_path if qt_path else os.path.join(audio_path, 'qt.joblib')

    audio_model, audio_qt = load_audio_model(config, ckpt, qt)
    audio_seq_len = gin.query_parameter('%AUDIO_SEQ_LEN')

    invert_audio_fn = partial(
        p2a.normalized_mels_to_audio,
        qt=audio_qt,
        n_iter=200,
    )

    return audio_model, audio_qt, audio_seq_len, invert_audio_fn