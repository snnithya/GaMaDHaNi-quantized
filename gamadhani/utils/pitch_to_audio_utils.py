import math
import librosa as li
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm
import numpy as np
import gin
import logging
import matplotlib.pyplot as plt
import logging
import pdb


class GPUQuantileTransformer:
    '''
    Class to perform inverse quantile transform on GPU
    '''
    def __init__(self, qt, device):
        # Initialize the sklearn QuantileTransformer
        self.cpu_transformer = qt
        self.quantiles_ = torch.Tensor(qt.quantiles_).to(device) # shape (n_quantiles, n_features)
        self.references_ = torch.Tensor(qt.references_).to(device) # shape (n_quantiles,)
        self.device = device

    def transform(self, X):
        return self.cpu_transformer.transform(X)

    def inverse_transform(self, X, threshold_bs=32):
        '''
        Beyond threshold_bs, the computation is faster moves to CPU
        '''
        # X shape (bs, 1, seq_len)
        assert X.dim() == 3
        assert X.size(1) == 1
        bs = X.shape[0]
        if bs > threshold_bs:
            # move everything to cpu
            logging.warning('Batch size greater than 8, moving to CPU for faster computation')
            self.move_to_cpu()
            X = X.cpu()
        # add a bs dimension to references and quantiles for faster computation
        self.references_ = self.references_.unsqueeze(0).unsqueeze(-1).repeat(bs, 1, 1)   # shape (bs, n_quantiles, 1)
        self.quantiles_ = self.quantiles_.unsqueeze(0).repeat(bs, 1, 1)
        # convert distribution to uniform
        X = 0.5 * (1 + torch.erf(X / torch.sqrt(torch.tensor(2.0)))) 
        # Interpolate using the quantiles and references on GPU
        idxs = torch.searchsorted(self.references_.view(bs, 1, -1), X.view(bs, 1, -1)).to(torch.int64)  # X shape (bs, 1, seq_len)
        idxs = idxs.view(bs, -1, 1) # to match with the shape of quantiles, shape (bs, seq_len, 1)
        quantiles_low = torch.gather(self.quantiles_, 1, (idxs - 1).clamp(min=0)).reshape(bs, 1, -1) # shape (bs, 1, seq_len)
        quantiles_high = torch.gather(self.quantiles_, 1, idxs.clamp(max=self.quantiles_.size(1) - 1)).reshape(bs, 1, -1) # shape (bs, 1, seq_len)

        # Linear interpolation between quantiles
        batch_index = torch.arange(bs).unsqueeze(1).expand(bs, X.size(2)).to(self.device).flatten()
        idxs = idxs.squeeze(-1).flatten() # shape (bs, seq_len)
        t = (X - self.references_[batch_index, idxs - 1].reshape(X.shape)) / (self.references_[batch_index, idxs, 0].reshape(X.shape) - self.references_[batch_index, idxs - 1, 0].reshape(X.shape) + 1e-10)
        X_inv = quantiles_low + t * (quantiles_high - quantiles_low)
        # update shape of references and quantiles
        self.references_ = self.references_[0]
        self.quantiles_ = self.quantiles_[0]

        return X_inv.reshape(bs, 1, -1)
    
    def move_to_cpu(self):
        self.references_ = self.references_.cpu()
        self.quantiles_ = self.quantiles_.cpu()
        self.device = 'cpu'

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
    if not isinstance(pitch, torch.Tensor):
        pitch = torch.Tensor(pitch)
    pitch = interpolate(pitch, size=audio_seq_len, mode='linear')
    return pitch