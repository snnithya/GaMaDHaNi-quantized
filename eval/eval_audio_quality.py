"""
The goal of this script is to evaluate the quality of audio generated at different quantization levels.
The results will help compare the quantized audio to the corresponding high-quality full-precision baseline.

This is @asmi's preliminary code before writing test cases with real audio examples. A sample test case is further below.
Testing will help decide whether to use and plot all three metrics in this script, only one, or some composite metric.
"""

import openl3
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.linalg import sqrtm
from pathlib import Path

# 1. Compute MCD
def compute_mcd(ref_audio, quant_audio, sample_rate=16000):
    """
    Mel Cepstral Distortion (MCD): measures spectral distortion between audio signals.

    - Aligns MFCCs (Mel-Frequency Cepstral Coefficients) of reference and quantized audio using DTW.
    - Lower MCD = better timbral and pitch preservation.
    """
    ref_wave, _ = librosa.load(ref_audio, sr=sample_rate, mono=True)
    quant_wave, _ = librosa.load(quant_audio, sr=sample_rate, mono=True)
    ref_mfcc = librosa.feature.mfcc(ref_wave, sr=sample_rate, n_mfcc=13).T
    quant_mfcc = librosa.feature.mfcc(quant_wave, sr=sample_rate, n_mfcc=13).T
    _, cost, _, path = librosa.sequence.dtw(ref_mfcc, quant_mfcc, metric="euclidean")
    return cost / len(path[0])


# 2. Compute LSD
def compute_lsd(ref_audio, quant_audio, sample_rate=16000):
    """
    Log Spectral Distance (LSD): measures frequency-domain distortion between audio signals.

    - Compares log-scaled power spectra of reference and quantized audio.
    - Lower LSD = better preservation of spectral content.
    """
    ref_wave, _ = librosa.load(ref_audio, sr=sample_rate, mono=True)
    quant_wave, _ = librosa.load(quant_audio, sr=sample_rate, mono=True)
    ref_spectrum = np.abs(librosa.stft(ref_wave)) ** 2
    quant_spectrum = np.abs(librosa.stft(quant_wave)) ** 2
    log_diff = np.log10(np.maximum(ref_spectrum, 1e-10)) - np.log10(np.maximum(quant_spectrum, 1e-10))
    lsd = np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0)))
    return lsd


# 3. Compute FAD
def compute_fad(reference_dir, quantized_dir, model, sample_rate=16000):
    """
    Frechet Audio Distance (FAD): measures overall perceptual similarity between audio distributions.

    - Compares mean and covariance of OpenL3 embeddings for reference and quantized audio.
    - Lower FAD = better overall perceptual similarity.
    """
    ref_embeddings = []
    quant_embeddings = []

    for ref_audio in Path(reference_dir).iterdir():
        if ref_audio.suffix == ".wav":
            ref_embeddings.append(extract_openl3_embeddings(str(ref_audio), model, sample_rate))

    for quant_audio in Path(quantized_dir).iterdir():
        if quant_audio.suffix == ".wav":
            quant_embeddings.append(extract_openl3_embeddings(str(quant_audio), model, sample_rate))

    ref_mu, ref_sigma = np.mean(ref_embeddings, axis=0), np.cov(np.array(ref_embeddings).T)
    quant_mu, quant_sigma = np.mean(quant_embeddings, axis=0), np.cov(np.array(quant_embeddings).T)
    diff = ref_mu - quant_mu
    covmean, _ = sqrtm(ref_sigma @ quant_sigma, disp=False)
    covmean = covmean.real if np.isfinite(covmean).all() else np.zeros_like(ref_sigma)
    fad_score = np.trace(ref_sigma + quant_sigma - 2 * covmean) + np.dot(diff, diff)
    return fad_score


def extract_openl3_embeddings(audio_path, model, sample_rate=16000):
    """
    OpenL3 embeddings: Extracts perceptual embeddings from audio using a pre-trained OpenL3 model.
    - Embeddings capture timbral and tonal features for perceptual quality evaluation.
    """
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    embeddings, _ = openl3.get_audio_embedding(waveform, sr, content_type="music", model=model)
    return np.mean(embeddings, axis=0)


# 4. Combined test function
def test_audio_quality(ref_dir, quant_dir, model, sample_rate=16000):
    """
    Evaluates quantization levels using FAD, MCD, and LSD.
    - FAD: measures overall perceptual similarity
    - MCD: evaluates timbral and pitch preservation
    - LSD: assesses spectral fidelity
    """
    ref_dir = Path(ref_dir)
    quant_dir = Path(quant_dir)
    results = []

    for quantization_level in quant_dir.iterdir():
        if quantization_level.is_dir():
            print(f"Evaluating quantization level: {quantization_level.name}")
            fad_score = compute_fad(ref_dir, quantization_level, model, sample_rate)
            # print(f"FAD Score for {quantization_level.name}: {fad_score}")

            for ref_file, quant_file in zip(ref_dir.iterdir(), quantization_level.iterdir()):
                if ref_file.suffix == ".wav" and quant_file.suffix == ".wav":
                    mcd_score = compute_mcd(ref_file, quant_file)
                    lsd_score = compute_lsd(ref_file, quant_file)
                    results.append({
                        "quantization_level": quantization_level.name,
                        "file": quant_file.name,
                        "FAD": fad_score,
                        "MCD": mcd_score,
                        "LSD": lsd_score
                    })
                    print(f"File: {quant_file.name}, MCD: {mcd_score}, LSD: {lsd_score}")
    return results


# Sample test case
if __name__ == "__main__":
    from openl3.models import load_audio_embedding_model

    # Load the OpenL3 model
    embedding_model = load_audio_embedding_model(content_type="music", input_repr="mel256")

    # Example directories
    reference_audio_dir = "reference_audio"  # path to high-quality baseline audio
    quantized_audio_dir = "quantized_audio"  # path to quantized audio organized by level

    # Run the test
    results = test_audio_quality(reference_audio_dir, quantized_audio_dir, embedding_model)
