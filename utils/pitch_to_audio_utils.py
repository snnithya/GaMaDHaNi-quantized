import math
import librosa as li
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm
import numpy as np
import gin
import logging
import matplotlib.pyplot as plt
import pdb

@gin.configurable
def torch_stft(x, nfft):
    window = torch.hann_window(nfft).to(x)
    x = torch.stft(
        x,
        n_fft=nfft,
        hop_length=nfft // 4,
        win_length=nfft,
        window=window,
        center=True,
        return_complex=True,
    )
    x = 2 * x / torch.mean(window)
    return x

@gin.configurable
def torch_istft(x, nfft):
    # pdb.set_trace()
    window = torch.hann_window(nfft).to(x.device)
    x = x / 2 * torch.mean(window)
    return torch.istft(
        x,
        n_fft=nfft,
        hop_length=nfft // 4,
        win_length=nfft,
        window=window,
        center=True,
    )

@gin.configurable
def to_mels(stft, nfft, num_mels, sr, eps=1e-2):
    mels = li.filters.mel(
        sr=sr,
        n_fft=nfft,
        n_mels=num_mels,
        fmin=40,
    )
    # pdb.set_trace()
    mels = torch.from_numpy(mels).to(stft)
    mel_stft = torch.einsum("mf,bft->bmt", mels, stft)
    mel_stft = torch.log(mel_stft + eps)
    return mel_stft

@gin.configurable
def from_mels(mel_stft, nfft, num_mels, sr, eps=1e-2):
    mels = li.filters.mel(
        sr=sr,
        n_fft=nfft,
        n_mels=num_mels,
        fmin=40,
    )
    mels = torch.from_numpy(mels).to(mel_stft)
    mels = torch.pinverse(mels)
    mel_stft = torch.exp(mel_stft) - eps
    stft = torch.einsum("fm,bmt->bft", mels, mel_stft)
    return stft

@gin.configurable
def torch_gl(stft, nfft, sr, n_iter):

    def _gl_iter(phase, xs, stft):
        del xs
        # pdb.set_trace()
        c_stft = stft * torch.exp(1j * phase)
        rec = torch_istft(c_stft, nfft)
        r_stft = torch_stft(rec, nfft)
        phase = torch.angle(r_stft)
        return phase, None

    phase = torch.rand_like(stft) * 2 * torch.pi

    for _ in tqdm(range(n_iter)):
        phase, _ = _gl_iter(phase, None, stft)

    c_stft = stft * torch.exp(1j * phase)
    audio = torch_istft(c_stft, nfft)

    return audio

@gin.configurable
def normalize(x, qt=None):
    x_flat = x.reshape(-1, 1)
    if qt is None:
        logging.warning('No quantile transformer found, returning input')
        return x
    return torch.Tensor(qt.transform(x_flat).reshape(x.shape))

@gin.configurable
def unnormalize(x, qt=None):
    x_flat = x.reshape(-1, 1)
    if qt is None:
        logging.warning('No quantile transformer found, returning input')
        return x
    if isinstance(x_flat, torch.Tensor):
        x_flat = x_flat.detach().cpu().numpy()
    return torch.Tensor(qt.inverse_transform(x_flat).reshape(x.shape))

@gin.configurable
def audio_to_normalized_mels(x, nfft, num_mels, sr, qt):
    # pdb.set_trace()
    stfts = torch_stft(x, nfft=nfft).abs()[..., :-1]
    mel_stfts = to_mels(stfts, nfft, num_mels, sr)
    return normalize(mel_stfts, qt).to(x)

@gin.configurable
def normalized_mels_to_audio(x, nfft, num_mels, sr, qt, n_iter=20):
    x = unnormalize(x, qt).to(x)
    x = from_mels(x, nfft, num_mels, sr)
    x = torch.clamp(x, 0, nfft)
    x = torch_gl(x, nfft, sr, n_iter=n_iter)
    return x

def interpolate_pitch(pitch, audio_seq_len):
    pitch = torch.Tensor(pitch)
    pitch = interpolate(pitch, size=audio_seq_len, mode='linear')
    # plt.plot(pitch[0].squeeze(0).detach().cpu().numpy())
    # plt.savefig(f"./temp/interpolated_pitch.png")
    # plt.close()
    return pitch