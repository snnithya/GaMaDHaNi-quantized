import numpy as np
import torch
import subprocess
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import os
from functools import partial
import gin
import sys
sys.path.append('./')
from utils.generate_utils import load_pitch_model, load_audio_model
import utils.pitch_to_audio_utils as p2a
import torchaudio
from absl import app
from torch.nn.functional import interpolate
import pdb
import logging
import time
import soundfile as sf

"""
Generate script flow:

1. make sure to set_grad = False
2. take in args - pitch
    a.transformer
    b. diffusion
3. take in args - audio
4. load data
5. load model
6. prime or not
7. model(data)
8. plot/save/
9. generate audio
19. plot/save
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('pitch_run', default=None, required=True, help='Path to the pitch generation model folder')
flags.DEFINE_string('audio_run', default=None, required=True, help='Path to the pitch to audio generation model folder')
flags.DEFINE_string('db_path_audio', default=None, required=True, help='Path to the audio DB')

def load_pitch_fns(pitch_path):
    pitch_model, pitch_qt, _, pitch_task_fn = load_pitch_model(
        os.path.join(pitch_path, 'config.gin'), 
        os.path.join(pitch_path, 'models', 'last.ckpt'), 
        os.path.join(pitch_path, 'qt.joblib')
        )
    invert_pitch_fn = partial(
        invert_pitch_read,
        min_norm_pitch=gin.query_parameter('dataset.pitch_read_w_downsample.min_norm_pitch'),
        time_downsample=gin.query_parameter('dataset.pitch_read_w_downsample.time_downsample'),
        pitch_downsample=gin.query_parameter('dataset.pitch_read_w_downsample.pitch_downsample'),
        qt_transform=pitch_qt,
        min_clip=gin.query_parameter('dataset.pitch_read_w_downsample.min_clip'),
        max_clip=gin.query_parameter('dataset.pitch_read_w_downsample.max_clip')
    )
    return pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn

def load_audio_fns(audio_path, db_path_audio):
    ckpt = os.path.join(audio_path, 'models', 'last.ckpt')
    config = os.path.join(audio_path, 'config.gin')
    qt = os.path.join(db_path_audio, 'qt.joblib')

    audio_model, audio_qt = load_audio_model(config, ckpt, qt)
    audio_seq_len = gin.query_parameter('%AUDIO_SEQ_LEN')

    invert_audio_fn = partial(
        p2a.normalized_mels_to_audio,
        qt=audio_qt,
        n_iter=200
    )

    return audio_model, audio_qt, audio_seq_len, invert_audio_fn

def interpolate_pitch(pitch, audio_seq_len):
    pitch = interpolate(pitch, size=audio_seq_len, mode='linear')
    plt.plot(pitch[0].squeeze(0).detach().cpu().numpy())
    plt.savefig(f"./temp/interpolated_pitch.png")
    plt.close()
    return pitch

def generate_pitch(pitch_model, 
                   invert_pitch_fn, 
                   num_samples, 
                   seq_len=2400,
                   temperature=1.0,
                   num_steps=None, 
                   outfolder=None, 
                   processed_primes=None):
    samples = pitch_model.sample_fn(num_samples, num_steps, prime=processed_primes) # keep only 4s
    inverted_pitches = [invert_pitch_fn(sample.detach().cpu().numpy())[0] for sample in samples]

    if outfolder is not None:
        os.makedirs(outfolder, exist_ok=True)
        for i, pitch in enumerate(inverted_pitches):
            flattened_pitch = pitch.flatten()
            pd.DataFrame({'f0': flattened_pitch}).to_csv(f"{outfolder}/{i}.csv", index=False)
            plt.plot(np.where(flattened_pitch == 0, np.nan, flattened_pitch))
            plt.savefig(f"{outfolder}/{i}.png")
            plt.close()
    return samples, inverted_pitches

def generate_audio(audio_model, f0s, invert_audio_fn, outfolder, singers=[3], num_steps=100):
    singer_tensor = torch.tensor(np.repeat(singers, repeats=f0s.shape[0])).to('cuda')
    samples, _, singers = audio_model.sample_cfg(f0s.shape[0], f0=f0s, num_steps=num_steps, singer=singer_tensor, strength=3)
    audio = invert_audio_fn(samples)
    
    # if outfolder is not None:
    #     os.makedirs(outfolder, exist_ok=True)
    #     for i, a in enumerate(audio):
    #         logging.log(logging.INFO, f"Saving audio {i}")
    #         torchaudio.save(f"{outfolder}/{i}.wav", torch.tensor(a).detach().unsqueeze(0).cpu(), 16000)
    return audio

def generate(audio_model=None, pitch_model=None, num_samples=2, num_steps=100, singers=[3], outfolder='temp', audio_seq_len=750, pitch_qt=None):
    logging.log(logging.INFO, 'Generate function')
    pitch, inverted_pitch = generate_pitch(pitch_model, invert_pitch_fn, 2, 100, outfolder=outfolder, processed_primes=selected_prime if global_ind != 0 else None)
    preprocessed_primes = pitch[:, :, 200:400]
    if pitch_qt is not None:
        def undo_qt(x, min_clip=200):
            pitch= pitch_qt.inverse_transform(x.reshape(-1, 1)).reshape(1, -1)
            pitch = np.around(pitch) # round to nearest integer, done in preprocessing of pitch contour fed into model
            pitch[pitch < 200] = np.nan
            return pitch
        pitch = torch.tensor(np.array([undo_qt(x) for x in pitch.detach().cpu().numpy()])).to(pitch_model.device)
    interpolated_pitch = interpolate_pitch(pitch=pitch, audio_seq_len=audio_seq_len)
    interpolated_pitch = torch.nan_to_num(interpolated_pitch, nan=196)
    interpolated_pitch = interpolated_pitch.squeeze(1) # to match input size by removing the extra dimension
    audio = generate_audio(audio_model, interpolated_pitch, invert_audio_fn, singers=singers, num_steps=100, outfolder=outfolder)[:, :16000*4]
    return pitch, audio
    
def main(argv):
    pitch_path = FLAGS.pitch_run
    audio_path = FLAGS.audio_run
    db_path_audio = FLAGS.db_path_audio
    pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn = load_pitch_fns(pitch_path)
    audio_model, audio_qt, audio_seq_len, invert_audio_fn = load_audio_fns(audio_path, db_path_audio)
