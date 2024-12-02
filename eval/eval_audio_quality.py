import os
import csv
import librosa
import numpy as np
from pathlib import Path
from scipy.linalg import sqrtm
import openl3
import tensorflow as tf

# Ensure compatibility with TensorFlow 2.x
tf.get_logger().setLevel('ERROR')

# 1. Compute MCD
def compute_mcd(ref_audio, quant_audio, sample_rate=16000):
    ref_wave, _ = librosa.load(ref_audio, sr=sample_rate, mono=True)
    quant_wave, _ = librosa.load(quant_audio, sr=sample_rate, mono=True)
    ref_mfcc = librosa.feature.mfcc(ref_wave, sr=sample_rate, n_mfcc=13).T
    quant_mfcc = librosa.feature.mfcc(quant_wave, sr=sample_rate, n_mfcc=13).T

    # Compute DTW between MFCCs
    D, wp = librosa.sequence.dtw(ref_mfcc, quant_mfcc, metric='euclidean')
    distance = D[-1, -1]
    mcd = distance / len(wp)
    return mcd

# 2. Compute LSD
def compute_lsd(ref_audio, quant_audio, sample_rate=16000):
    ref_wave, _ = librosa.load(ref_audio, sr=sample_rate, mono=True)
    quant_wave, _ = librosa.load(quant_audio, sr=sample_rate, mono=True)
    ref_spectrum = np.abs(librosa.stft(ref_wave, n_fft=1024, hop_length=512)) ** 2
    quant_spectrum = np.abs(librosa.stft(quant_wave, n_fft=1024, hop_length=512)) ** 2
    log_diff = 10 * (np.log10(ref_spectrum + 1e-10) - np.log10(quant_spectrum + 1e-10))
    lsd = np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0)))
    return lsd

# 3. Compute FAD
def extract_openl3_embeddings(audio_path, sample_rate=16000):
    """
    Extract OpenL3 embeddings from an audio file using OpenL3's get_audio_embedding function.
    """
    # Load audio manually using librosa
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Extract embeddings using openl3.get_audio_embedding
    try:
        embeddings, timestamps = openl3.get_audio_embedding(
            waveform,
            sr,
            input_repr="mel256",
            content_type="music",
            embedding_size=512
        )
        print(f"Extracted embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error during embedding extraction: {e}")
        raise e

    return embeddings

def compute_fad(ref_audio, quant_audio, sample_rate=16000):
    ref_embedding = extract_openl3_embeddings(ref_audio, sample_rate)
    quant_embedding = extract_openl3_embeddings(quant_audio, sample_rate)

    # Compute mean and covariance of embeddings
    ref_mu = np.mean(ref_embedding, axis=0)
    ref_sigma = np.cov(ref_embedding, rowvar=False)
    quant_mu = np.mean(quant_embedding, axis=0)
    quant_sigma = np.cov(quant_embedding, rowvar=False)

    diff = ref_mu - quant_mu
    covmean, _ = sqrtm(ref_sigma @ quant_sigma, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad_score = np.trace(ref_sigma + quant_sigma - 2 * covmean) + diff @ diff
    return fad_score

# 4. Evaluate and Save Results
def evaluate_channel_quality(base_dir, sample_rate=16000):
    """
    Evaluate the quality metrics for ground truth vs quantized audio pairs and save results to CSV.
    """
    base_dir = Path(base_dir)

    # Iterate over each channel folder (1, 2, ..., 16)
    for channel_folder in sorted(base_dir.iterdir()):
        if not channel_folder.is_dir():
            continue  # Skip non-folder entries

        print(f"\nEvaluating channel: {channel_folder.name}")

        # Paths for ground truth and quantized files
        ground_truth_files = list(channel_folder.glob("*ground_truth*.wav"))
        if not ground_truth_files:
            print(f"No ground truth file found in {channel_folder}")
            continue
        ground_truth = ground_truth_files[0]

        quantized_files = [f for f in channel_folder.glob("*.wav") if "ground_truth" not in f.name]

        # Collect results
        results = []
        for quant_file in quantized_files:
            try:
                mcd = compute_mcd(ground_truth, quant_file, sample_rate)
                lsd = compute_lsd(ground_truth, quant_file, sample_rate)
                fad = compute_fad(ground_truth, quant_file, sample_rate)
                results.append([quant_file.name, mcd, lsd, fad])
                print(f"Processed {quant_file.name}: MCD={mcd:.4f}, LSD={lsd:.4f}, FAD={fad:.4f}")
            except Exception as e:
                print(f"Failed to process {quant_file.name}: {e}")

        # Write results to CSV
        csv_file = channel_folder / "evaluation_results.csv"
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "MCD", "LSD", "FAD"])
            writer.writerows(results)
        print(f"Saved results to {csv_file}")

if __name__ == "__main__":
    base_directory = "./examples/wavs"
    evaluate_channel_quality(base_directory)
