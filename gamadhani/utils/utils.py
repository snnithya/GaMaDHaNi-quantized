import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path
import os
import random
import torch
import torchaudio
import numpy as np
import gin
import pdb


def download_models(model_repo_id, pitch_model_type):
    if pitch_model_type is not None:
        PITCH_MODEL_FILENAME = f"{pitch_model_type}_pitch_model-model.ckpt"
        pitch_path = hf_hub_download(repo_id=model_repo_id, filename=PITCH_MODEL_FILENAME)
        if pitch_model_type=="diffusion":
            qt_path_pitch = hf_hub_download(repo_id=model_repo_id, filename=f"{pitch_model_type}_pitch_model-qt.joblib")
        else:
            qt_path_pitch = None
    AUDIO_MODEL_FILENAME = "pitch_to_audio_model-model.ckpt"
    audio_path = hf_hub_download(repo_id=model_repo_id, filename=AUDIO_MODEL_FILENAME)
    qt_path_p2a = hf_hub_download(repo_id=model_repo_id, filename="pitch_to_audio_model-qt.joblib")
    return pitch_path, qt_path_pitch, audio_path, qt_path_p2a

def download_data(data_repo_id):
    PRIME_FILE_PATH = "listening_study_primes.npz"
    prime_file = hf_hub_download(repo_id=data_repo_id, filename=PRIME_FILE_PATH, repo_type='dataset') if data_repo_id else None
    return prime_file

def search_for_run(run_path, mode="last"):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in os.path.basename(str(e)), ckpts)
    ckpts = sorted(ckpts)
    if len(ckpts): 
        if len(ckpts) > 1 and 'last.ckpt' in ckpts:
            return ckpts[-2]    # last.ckpt is always at the end, so we take the second last
        else:
            return ckpts[-1]
    else: return None

def set_seed(seed: int):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

@gin.configurable
def build_warmed_exponential_lr_scheduler(
        optim: torch.optim.Optimizer, start_factor: float, peak_iteration: int,
        decay_factor: float=None, cycle_length: int=None, eta_min: float=None, eta_max: float=None) -> torch.optim.lr_scheduler._LRScheduler:
    linear = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=start_factor,
        end_factor=1.,
        total_iters=peak_iteration,
    )
    if decay_factor:
        exp = torch.optim.lr_scheduler.ExponentialLR(
            optim, 
            gamma=decay_factor,
        )
        return torch.optim.lr_scheduler.SequentialLR(optim, [linear, exp],
                                                    milestones=[peak_iteration])
    if cycle_length:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=cycle_length,
            eta_min = eta_min * eta_max
        )
        return torch.optim.lr_scheduler.SequentialLR(optim, [linear, cosine],
                                                    milestones=[peak_iteration])
    
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Script is running on: {'GPU' if device.type == 'cuda' else 'CPU'}")
    return device 
          
def plot(f0_array: np.ndarray=None, time_array: np.ndarray=None, prime: bool=True):
    fig, ax = plt.subplots()
    # to plot silences as gaps in the contour
    f0_array = np.where(f0_array == 0, np.nan, f0_array)
    time_array = np.arange(len(f0_array)) / 100  #time downsampling
    if prime:
        split_index = len(f0_array) // 3
        ax.plot(time_array[:split_index], f0_array[:split_index], color='blue', label='Prime')
        ax.plot(time_array[split_index:], f0_array[split_index:], color='red', label='Generated Pitch')
    else:
        ax.plot(time_array, f0_array, color='red', label='Generated Pitch')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Pitch Contour')
    ax.grid(True)
    ax.legend()
    plt.close(fig)  
    return fig

def save_figure(figure: Figure, dest_path: str) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    figure.savefig(dest_path)

def save_csv(df: pd.DataFrame, dest_path: str) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df.to_csv(dest_path, index=False)

def save_audio(audio_array: torch.Tensor, dest_path: str, sample_rate: int) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    torchaudio.save(dest_path, audio_array, sample_rate)
