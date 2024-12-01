"""
NOT WORKING YET. Please see convert_to_wav.ipynb instead.
"""

import os
import torch
import soundfile as sf

def convert_to_wav(input_dir, sample_rate=16000):
    """
    Converts all .pt files in a directory to .wav files inside a 'wavs/' subfolder.
    - Skips conversion if the corresponding .wav file already exists.
    - Handles .pt files saved on CUDA devices by mapping them to CPU.

    Parameters:
    - input_dir (str): Directory containing subfolders with .pt files.
    - sample_rate (int): Sampling rate for the output .wav files (default: 16,000 Hz).
    """
    input_dir = os.path.abspath(input_dir)

    # Create a 'wavs/' subfolder inside the input directory
    wavs_dir = os.path.join(input_dir, "wavs")
    print("WAVS DIR:", wavs_dir)
    os.makedirs(wavs_dir, exist_ok=True)

    # Process .pt files in the folder
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pt"):
            pt_file_path = os.path.join(input_dir, file_name)
            wav_file_name = file_name.replace(".pt", ".wav")
            wav_file_path = os.path.join(wavs_dir, wav_file_name)

            # Skip conversion if .wav already exists
            if os.path.exists(wav_file_path):
                print(f"Skipping {wav_file_path}, already exists.")
                continue

            try:
                # Load tensor and map to CPU
                audio_tensor = torch.load(pt_file_path, map_location=torch.device('cpu'))

                # Ensure tensor is on CPU and convert to NumPy
                audio_data = audio_tensor.cpu().numpy()

                # Normalize audio to range [-1, 1] if needed
                if audio_data.max() > 1 or audio_data.min() < -1:
                    audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

                # Save as .wav
                sf.write(wav_file_path, audio_data, sample_rate)
                print(f"Converted {pt_file_path} to {wav_file_path}")
            except Exception as e:
                print(f"Failed to process {pt_file_path}: {e}")

if __name__ == "__main__":
    input_directory = "./examples"
    convert_to_wav(input_directory)
