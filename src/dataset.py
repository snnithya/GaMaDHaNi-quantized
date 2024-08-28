from typing import Any, Callable, Dict, Optional, Tuple
from functools import partial

import gin
import inspect
import lmdb
import torch
import pdb
import numpy as np
from torch.utils.data import Dataset, Subset
from GaMaDHaNi.src.protobuf.data_example import AudioExample
from sklearn.preprocessing import QuantileTransformer


TensorDict = Dict[str, torch.Tensor]

@gin.configurable
class Task:
    """
    read_fn: function to read and transform the data sequence in the format the training script accepts
    invert_fn: applying the inverse of the transform applied in read_fn
    common_args: the common arguments required by both the read_fn and the invert_fn
    kwargs in read and invert functions are to pass the dynamic entities i.e. Data
    """
    def __init__(self, read_fn: Callable[..., Any], invert_fn: Callable[..., Any], **kwargs):
        self.read_fn = read_fn
        self.invert_fn = invert_fn
        self.extra_args = kwargs["kwargs"] #because of gin file?

    def read_(self, **kwargs):
        kwargs.update(self.extra_args)
        return self.read_fn(**kwargs)
        
    def invert_(self, **kwargs):
        kwargs.update(self.extra_args)
        return self.invert_fn(**kwargs)


def ordered_dataset_split(dataset: Dataset,
                          sizes: Tuple[int, ...]) -> Tuple[Dataset]:
    assert sum(sizes) == len(dataset)
    datasets = []
    start_idx = 0
    for size in sizes:
        datasets.append(Subset(dataset, range(start_idx, start_idx + size)))
        start_idx += size
    return tuple(datasets)


@gin.configurable
class SequenceDataset(Dataset):
    def __init__(
            self,
            db_path: str,
            task: Optional[Task] = None,
            apply_transform: bool = False) -> None:
        super().__init__()
        self._env = None
        self._keys = None
        self._db_path = db_path
        self._task = task
        self._apply_transform = apply_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))
        ae = ae.as_dict()
        if self._task is not None:
            if self._apply_transform:
                if "train" in self._db_path:
                    transpose_pitch = np.random.randint(-400,400)
                    ae = self._task.read_(**{"inputs": ae, "transpose_pitch": transpose_pitch})
                else:
                    ae = self._task.read_(**{"inputs": ae})
            else:
                ae = self._task.read_(**{"inputs": ae})
           
        if ae:
            return ae

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(
                self._db_path,
                lock=False,
                readahead=False,
            ) 
        return self._env

    @property
    def keys(self):
        if self._keys is None:
            with self.env.begin(write=False) as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys

def hz_to_cents(f0, ref=440, min_norm_pitch=0, pitch_downsample=1, min_clip=200, max_clip=600, silence_token=None):
    # pdb.set_trace()
    f0[f0 == 0] = np.nan
    norm_f0 = f0.copy()
    norm_f0[~np.isnan(norm_f0)] = (1200) * np.log2(norm_f0[~np.isnan(norm_f0)] / ref)
    # descretize pitch
    norm_f0[~np.isnan(norm_f0)] = np.around(norm_f0[~np.isnan(norm_f0)])
    norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] - (min_norm_pitch)
    norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] // pitch_downsample + 1 # reserve 0 for silence
    norm_f0[~np.isnan(norm_f0)] = np.clip(norm_f0[~np.isnan(norm_f0)], min_clip, max_clip) #HACK
    if silence_token is not None:
        norm_f0[np.isnan(norm_f0)] = silence_token



    return norm_f0

@gin.configurable
def mel_pitch(
        inputs: TensorDict, 
        min_norm_pitch: int, 
        audio_seq_len: int=None, 
        pitch_downsample: int = 1, 
        qt_transform: Optional[QuantileTransformer] = None,
        min_clip: int = 200,
        max_clip: int = 600,
        nfft: int = 2048,
        convert_audio_to_mel: bool = False
        ):
    hop_size = nfft // 4
    audio_data = inputs['audio']['data']
    audio_sr = inputs['audio']['sampling_rate']
    pitch_data = inputs['pitch']['data']
    pitch_sr = inputs['pitch']['sampling_rate']
    # pdb.set_trace()
    if audio_seq_len is not None:
        # if audio_seq_len is given, cuts audio/pitch else returns the entire chunk
        pitch_seq_len = np.around((audio_seq_len/audio_sr) * pitch_sr ).astype(int)
        pitch_start = randint(0, pitch_data.shape[0] - pitch_seq_len - 1)
        pitch_end = pitch_start + pitch_seq_len
        pitch_data = pitch_data[pitch_start:pitch_end]
        audio_start = np.around(pitch_start * audio_sr // pitch_sr).astype(int)
        audio_end = np.around(audio_start + audio_seq_len).astype(int)
        # pdb.set_trace()
        audio_data = audio_data[audio_start:audio_end]
    else:
        pitch_seq_len = np.around((audio_data.shape[0]/audio_sr) * pitch_sr ).astype(int)
    audio_data = p2a.audio_to_normalized_mels(torch.Tensor(audio_data).unsqueeze(0), qt=qt_transform).numpy()[0]

    pitch_data = hz_to_cents(pitch_data, min_norm_pitch=min_norm_pitch, pitch_downsample=pitch_downsample, min_clip=min_clip, max_clip=max_clip)

    if audio_seq_len is not None:
        # linearly interpolate pitch data to match audio sequence length, if audio_seq_len is given
        pitch_inds = np.linspace(0, pitch_data.shape[0], num=audio_seq_len//hop_size, endpoint=False) #check here
        pitch_data = np.interp(pitch_inds, np.arange(0, pitch_data.shape[0]), pitch_data)

    # replace nan (aka silences) with min_clip - 4
    pitch_data[np.isnan(pitch_data)] = min_clip - 4

    return audio_data, pitch_data
def running_average(signal, window_size):
    
    weights = np.ones(int(window_size)) / window_size
    pad_width = len(weights) // 2
    padded_signal = np.pad(signal, pad_width, mode='symmetric')
    # Perform the convolution
    smoothed_signal = np.convolve(padded_signal, weights, mode='valid')
    if window_size % 2 == 0:
        smoothed_signal = smoothed_signal[:-1]
    return smoothed_signal

@gin.configurable
def pitch_coarse_condition(
        inputs: TensorDict, 
        min_norm_pitch: int, 
        pitch_seq_len: int=None, 
        pitch_downsample: int = 1, 
        time_downsample: int = 1,
        qt_transform: Optional[QuantileTransformer] = None,
        min_clip: int = 200,
        max_clip: int = 600,
        add_noise: bool = True,
        avg_window_size: float = 1 # window size in seconds
        ):
    
    pitch_data = inputs['pitch']['data']
    if pitch_seq_len is not None:
        pitch_start = randint(0, pitch_data.shape[0] - pitch_seq_len*time_downsample - 1)
        pitch_end = pitch_start + pitch_seq_len*time_downsample
        pitch_data = pitch_data[pitch_start:pitch_end:time_downsample]
    pitch_data = hz_to_cents(pitch_data, min_norm_pitch=min_norm_pitch, pitch_downsample=pitch_downsample, min_clip=min_clip, max_clip=max_clip)

    # extract coarse pitch condition
    pitch_sr = inputs['pitch']['sampling_rate'] // time_downsample
    avg_pitch = running_average(pitch_data, np.around(pitch_sr * avg_window_size).astype(int))
    # replace nan (aka silences) with min_clip - 4
    if add_noise:
        pitch_data[np.isnan(pitch_data)] = min_clip - 4 + np.clip(np.random.normal(size=pitch_data[np.isnan(pitch_data)].shape), -3, 3) # making sure noise is between -3 and 3 and thus won't spill into pitched values
        avg_pitch[np.isnan(avg_pitch)] = min_clip - 4 + np.clip(np.random.normal(size=avg_pitch[np.isnan(avg_pitch)].shape), -3, 3) # making sure noise is between -3 and 3 and thus won't spill into pitched values
    else:
        pitch_data[np.isnan(pitch_data)] = min_clip - 4

    if qt_transform:
        # apply qt transform
        qt_inp = pitch_data.reshape(-1, 1)
        pitch_data = qt_transform.transform(qt_inp).reshape(-1)
        avg_qt_inp = avg_pitch.reshape(-1, 1)
        avg_pitch = qt_transform.transform(avg_qt_inp).reshape(-1)
    # pdb.set_trace()
    return pitch_data, avg_pitch

@gin.configurable
def mel_pitch_coarse_condition(
        inputs: TensorDict,
        min_norm_pitch: int, 
        audio_seq_len: int=None, 
        pitch_downsample: int = 1, 
        qt_transform: Optional[QuantileTransformer] = None,
        min_clip: int = 200,
        max_clip: int = 600,
        nfft: int = 2048,
        avg_window_size: float = 1 # duration of avg window in seconds
):
    mel, pitch = mel_pitch(inputs, min_norm_pitch, audio_seq_len, pitch_downsample, qt_transform, min_clip, max_clip, nfft)
    silence_token = min_clip - 4
    avg_pitch = pitch.copy()
    avg_pitch[pitch == silence_token] = np.nan
    
    time = mel.shape[1]/inputs['audio']['sampling_rate']
    pitch_sr = pitch.shape[0]/time

    avg_pitch = running_average(avg_pitch, np.around(pitch_sr*avg_window_size))
    avg_pitch[np.isnan(avg_pitch)] = silence_token

    return mel, pitch, avg_pitch

def load_cached_audio(
        inputs: TensorDict, 
        audio_len: Optional[float] = None,
    ) -> torch.Tensor: 

    audio_data = inputs['audio']['data']
    if audio_len is not None:
        audio_start = randint(0, audio_data.shape[1] - audio_len - 1)
        audio_end = audio_start + audio_len
        audio_data = audio_data[:, audio_start:audio_end]
    return torch.Tensor(audio_data)

# need to add a silence token / range, calculate pitch avg
def load_cached_dataset(
        inputs: TensorDict, 
        audio_len: float,
        return_singer: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
    # pdb.set_trace()
    audio_sr = inputs['audio']['sampling_rate']
    audio_data = inputs['audio']['data']
    audio_start = randint(0, audio_data.shape[1] - audio_len - 1)
    audio_end = audio_start + audio_len
    audio_data = audio_data[:, audio_start:audio_end]

    pitch_sr = inputs['pitch']['sampling_rate']
    pitch_len = np.floor(audio_len / audio_sr * pitch_sr).astype(int)
    pitch_data = inputs['pitch']['data']
    pitch_start = np.floor(audio_start * pitch_sr / audio_sr).astype(int)
    pitch_end = pitch_start + pitch_len
    pitch_data = pitch_data[pitch_start:pitch_end]
    
    # interpolate data to match audio length
    pitch_inds = np.linspace(0, pitch_data.shape[0], num=audio_len, endpoint=False) #check here
    pitch_data = np.interp(pitch_inds, np.arange(0, pitch_data.shape[0]), pitch_data)

    if return_singer:
        singer = torch.Tensor([inputs['global_conditions']['singer']])
    else:
        singer = None
    
    # print(audio_data.shape, pitch_data.shape, singer.shape if singer is not None else None)
    return torch.Tensor(audio_data), torch.Tensor(pitch_data), singer