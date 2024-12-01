import os
import torch
import soundfile as sf

def convert_to_wav(input_dir, sample_rate=16000):
    """
    Converts all .pt files in a directory to .wav files inside a 'wavs/' subfolder.
    - Skips conversion if the corresponding .wav file already exists.
    
    Parameters:
    - input_dir (str): Directory containing subfolders with .pt files.
    - sample_rate (int): Sampling rate for the output .wav files (default: 16,000 Hz).
    """
    input_dir = os.path.abspath(input_dir)

    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # skip non-folder entries

        # create subfolder inside the current folder
        wavs_dir = os.path.join(folder_path, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)

        # process .pt files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pt"):
                pt_file_path = os.path.join(folder_path, file_name)
                wav_file_name = file_name.replace(".pt", ".wav")
                wav_file_path = os.path.join(wavs_dir, wav_file_name)

                # skip conversion if .wav already exists
                if os.path.exists(wav_file_path):
                    print(f"Skipping {wav_file_path}, already exists.")
                    continue

                # load tensor and convert to .wav
                audio_tensor = torch.load(pt_file_path)

                # ensure tensor is on CPU and convert to NumPy
                audio_data = audio_tensor.cpu().numpy()

                # normalize audio to range [-1, 1] if needed
                if audio_data.max() > 1 or audio_data.min() < -1:
                    audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

                # save as .wav
                sf.write(wav_file_path, audio_data, sample_rate)
                print(f"Converted {pt_file_path} to {wav_file_path}")

if __name__ == "__main__":
    input_directory = "./examples"
    convert_pt_to_wav_with_check(input_directory)
