import os
import torch
import numpy as np
import soundfile as sf

def split_and_save_channels(input_dir, sample_rate=16000):
    """
    Converts multi-channel .pt files into separate .wav files, placing each channel
    into corresponding subfolders (1, 2, ..., N) under the 'wavs/' directory.

    Parameters:
    - input_dir (str): Path to the directory containing .pt files.
    - sample_rate (int): Sampling rate for the output .wav files (default: 16,000 Hz).
    """
    input_dir = os.path.abspath(input_dir)
    wavs_dir = os.path.join(input_dir, "wavs")

    os.makedirs(wavs_dir, exist_ok=True)

    # tterate over .pt files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pt"):
            pt_file_path = os.path.join(input_dir, file_name)
            print(f"Processing {pt_file_path}...")

            try:
                obj = torch.load(pt_file_path, map_location=torch.device('cpu'))

                # sxtract audio tensor
                if isinstance(obj, dict):
                    # Replace 'audio' with the correct key if different
                    audio_tensor = obj.get('audio') or next(iter(obj.values()))
                elif isinstance(obj, torch.Tensor):
                    audio_tensor = obj
                else:
                    raise TypeError("Unsupported data format in the .pt file")

                audio_data = audio_tensor.cpu().numpy()

                # transpose if the first dimension is the number of channels
                # if audio_data.shape[0] == 16:
                audio_data = audio_data.T

                # normalize each channel
                max_vals = np.max(np.abs(audio_data), axis=0)
                audio_data = audio_data / max_vals[np.newaxis, :]
                audio_data = np.nan_to_num(audio_data)

                # save each channel separately into corresponding subfolders
                for i in range(audio_data.shape[1]):
                    channel_data = audio_data[:, i]
                    channel_dir = os.path.join(wavs_dir, str(i + 1))
                    os.makedirs(channel_dir, exist_ok=True)
                    channel_wav_path = os.path.join(channel_dir, f"{file_name.replace('.pt', '')}_channel_{i+1}.wav")

                    # save .wav file
                    sf.write(channel_wav_path, channel_data, sample_rate)
                    print(f"Saved channel {i+1} to {channel_wav_path}")

            except Exception as e:
                print(f"Failed to process {pt_file_path}: {e}")


if __name__ == "__main__":
    input_directory = "./examples"
    split_and_save_channels(input_directory)
