#!/usr/bin/env python3
"""
Acoustic Feature Extraction Pipeline (Cleaned-Up Version)

- Preserves clinical feature logic from the uploaded script.
- Removes extraneous commented code and debug prints.
- Adds a filter to keep only clinically relevant openSMILE features.
- Uses structured logging.
"""

from __future__ import annotations
from clinical_denoise_pipeline import clinical_denoise_audio
from clinical_audio_config import CLINICAL_AUDIO_CONFIG
import os
import logging
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import soundfile as sf
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from dotenv import load_dotenv

load_dotenv()

output_audio_dir = os.getenv("PROCESSED_AUDIO_DIR", "datasets/output/processed_audio")

save_processed = os.getenv("SAVE_PROCESSED_AUDIO", "False").lower() =="true"
max_amplification = float(os.getenv("MAX_AMPLIFICATION", "3.0"))

# Optional dependency
try:
    import opensmile
    _OPENSML_AVAILABLE = True
except Exception:
    opensmile = None
    _OPENSML_AVAILABLE = False

# clinical config import (must exist in project)
from clinical_audio_config import ClinicalAudioConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("acoustic_feature_extraction")
warnings.filterwarnings("ignore")


# ---------------------------
# Configuration: openSMILE features to keep
# ---------------------------
USEFUL_OPENSML_FEATURES = {
    # Pitch (semitone) statistics
    "opensmile_F0semitoneFrom27.5Hz_sma3nz_amean",
    "opensmile_F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "opensmile_F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "opensmile_F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "opensmile_F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "opensmile_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    # Loudness/intensity
    #"opensmile_loudness_sma3_amean",
    #"opensmile_loudness_sma3_stddevNorm",
    #"opensmile_loudness_sma3_percentile20.0",
    #"opensmile_loudness_sma3_percentile50.0",
    #"opensmile_loudness_sma3_percentile80.0",
    "opensmile_loudness_sma3_pctlrange0-2",
    # Jitter/shimmer/HNR
    "opensmile_jitterLocal_sma3nz_amean",
    "opensmile_jitterLocal_sma3nz_stddevNorm",
    "opensmile_shimmerLocaldB_sma3nz_amean",
    "opensmile_shimmerLocaldB_sma3nz_stddevNorm",
    "opensmile_HNRdBACF_sma3nz_amean",
    "opensmile_HNRdBACF_sma3nz_stddevNorm",
    # Voicing / segment statistics
    #"opensmile_loudnessPeaksPerSec",
    "opensmile_VoicedSegmentsPerSec",
    "opensmile_MeanVoicedSegmentLengthSec",
    "opensmile_StddevVoicedSegmentLengthSec",
    "opensmile_MeanUnvoicedSegmentLength",
    "opensmile_StddevUnvoicedSegmentLength",
    # Equivalent sound level (overall energy)
    #"opensmile_equivalentSoundLevel_dBp",
}


def filter_useful_features(all_features: Dict[str, float]) -> Dict[str, float]:
    """
    Keep only clinically relevant features. Non-openSMILE features are kept by default.
    For openSMILE features, only the subset defined in USEFUL_OPENSML_FEATURES is retained.
    """
    filtered: Dict[str, float] = {}
    for k, v in all_features.items():
        if not k.startswith("opensmile_"):
            filtered[k] = v
        elif k in USEFUL_OPENSML_FEATURES:
            filtered[k] = v
    return filtered


class AcousticFeatureExtractor:
    """
    Cleaner, modular feature extractor focused on dementia-relevant acoustic biomarkers.
    """

    def __init__(self, config: Optional[Dict] = None):

        self.config = config if config is not None else CLINICAL_AUDIO_CONFIG
        self.sr = int(self.config.get("sample_rate", 16000))
        self.opensmile_available = _OPENSML_AVAILABLE and opensmile is not None
        if self.opensmile_available:
            # initialize opensmile instance lazily (functional set similar to original)
            try:
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
            except Exception:
                self.smile = None
                self.opensmile_available = False

    # --------------------------
    # Loading & preprocessing
    # --------------------------
    def _load_audio(self, audio_path: str, use_preproc_sr: bool = True) -> Tuple[np.ndarray, int]:
        sr = self.sr if use_preproc_sr else None
        audio, sr_loaded = librosa.load(audio_path, sr=sr)
        return audio, sr_loaded

    def preprocess_for_features(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Clinical-safe preprocessing:
        - load audio at target sample rate
        - clinical-grade denoising (no silence removal)
        - light RMS normalization
        """
        audio, sr = self._load_audio(audio_path)
        if audio.size == 0:
            return audio, sr

        # === Clinical denoising (preserves pauses, pitch, jitter) ===
        audio = clinical_denoise_audio(audio, sr, self.config)

        # === Conditional RMS normalization  ===
        norm = self.config.get("normalization", {})
        if norm.get("method", "rms") == "rms":
            target = norm.get("target_rms", 0.05)
            max_gain = norm.get("max_amplification", 3.0)
            cur = np.sqrt(np.mean(audio ** 2)) if audio.size > 0 else 0.0
            # Only normalize if audio is VERY quiet
            if cur > target * 0.6:
                pass  # preserve original levels if the current is loud enough 
            elif  cur>0:                
                gain = min(target / cur, max_gain)
                audio = audio *gain
            # Otherwise, preserve original levels

        # === SAVE PROCESSED AUDIO (NEW) ===
        if save_processed:
            processed_path = Path(output_audio_dir)
            processed_path.mkdir(parents=True, exist_ok=True)

            src = Path(audio_path)
            out_name = f"{src.stem}_processed.wav"
            out_file = processed_path / out_name

            # write as PCM16 for clinical safety
            sf.write(out_file, audio, sr, subtype="PCM_16")
                

        return audio, sr


    # --------------------------
    # Prosodic & voice quality features
    # --------------------------
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features: Dict[str, float] = {}
        # Ensure parselmouth sound creation
        try:
            sound = parselmouth.Sound(audio, sr)
        except Exception:
            sound = parselmouth.Sound(audio.astype(np.float64), int(sr))

        # Pitch extraction (Praat "To Pitch")


        pitch_values = np.array([])
        pitch_obj = None
        try:
            pitch_obj = call(sound, "To Pitch", 0.0,
                             self.config.get("prosody", {}).get("pitch_floor", 75),
                             self.config.get("prosody", {}).get("pitch_ceiling", 600))
            pitch_values = pitch_obj.selected_array["frequency"]
            pitch_values = pitch_values[pitch_values > 0]

            if pitch_obj is not None:
                total_frames = len(pitch_obj.selected_array["frequency"])
                voiced_frames = np.sum(pitch_obj.selected_array["frequency"] > 0)
                logger.info(
                    "[PITCH DIAG] Voiced frames: %d / %d (%.2f%%)",
                    voiced_frames,
                    total_frames,
                    100.0 * voiced_frames / total_frames if total_frames > 0 else 0.0
                )
        except Exception:
            pitch_values = np.array([])

        if pitch_values.size > 0:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
            features["pitch_min"] = float(np.min(pitch_values))
            features["pitch_max"] = float(np.max(pitch_values))
            features["pitch_range"] = float(features["pitch_max"] - features["pitch_min"])
            features["pitch_percentile_25"] = float(np.percentile(pitch_values, 25))
            features["pitch_percentile_75"] = float(np.percentile(pitch_values, 75))
            features["pitch_iqr"] = float(features["pitch_percentile_75"] - features["pitch_percentile_25"])
            if pitch_values.size > 1:
                t = np.arange(pitch_values.size)
                slope, _ = np.polyfit(t, pitch_values, 1)
                features["pitch_slope"] = float(slope)
            else:
                features["pitch_slope"] = 0.0
        else:
            for k in ["pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range",
                      "pitch_percentile_25", "pitch_percentile_75", "pitch_iqr", "pitch_slope"]:
                features[k] = 0.0

        # Voice breaks (count, rate, degree) - robust approach using pitch array
        features["voice_breaks_count"] = 0.0
        features["voice_breaks_rate"] = 0.0
        features["voice_break_degree"] = 0.0
        try:
            if pitch_obj is not None:
                duration = sound.get_total_duration() if hasattr(sound, "get_total_duration") else len(audio) / sr
                all_pitch = pitch_obj.selected_array["frequency"]
                if len(all_pitch) > 0:
                    voiced_mask = all_pitch > 0
                    transitions = np.diff(voiced_mask.astype(int))
                    voice_to_unvoice = int(np.sum(transitions == -1))
                    features["voice_breaks_count"] = float(voice_to_unvoice)
                    features["voice_breaks_rate"] = float(voice_to_unvoice / duration) if duration > 0 else 0.0
                    total_frames = len(all_pitch)
                    voiced_frames = int(np.sum(voiced_mask))
                    unvoiced_fraction = (total_frames - voiced_frames) / total_frames if total_frames > 0 else 0.0
                    features["voice_break_degree"] = float(unvoiced_fraction)
                    # Praat fallbacks (non-fatal)
                    try:
                        vb_praat = call(pitch_obj, "Count voice breaks", 0.0, 0.0)
                        if vb_praat is not None:
                            features["voice_breaks_count"] = float(vb_praat)
                            features["voice_breaks_rate"] = float(vb_praat / duration) if duration > 0 else 0.0
                        vb_degree_praat = call(pitch_obj, "Get fraction of locally unvoiced frames", 0.0, 0.0)
                        if vb_degree_praat is not None and not np.isnan(vb_degree_praat):
                            features["voice_break_degree"] = float(vb_degree_praat)
                    except Exception:
                        pass
        except Exception:
            features["voice_breaks_count"] = 0.0
            features["voice_breaks_rate"] = 0.0
            features["voice_break_degree"] = 0.0

        # Intensity
        
        try:
            intensity = call(sound, "To Intensity", 75, 0.0)
            intensity_values = intensity.values[0]
            features["intensity_mean"] = float(np.mean(intensity_values))
            features["intensity_std"] = float(np.std(intensity_values))
            features["intensity_min"] = float(np.min(intensity_values))
            features["intensity_max"] = float(np.max(intensity_values))
            features["intensity_range"] = float(features["intensity_max"] - features["intensity_min"])
        except Exception:
            features["intensity_mean"] = 0.0
            features["intensity_std"] = 0.0
            features["intensity_min"] = 0.0
            features["intensity_max"] = 0.0
            features["intensity_range"] = 0.0
        

        # Harmonicity (HNR)
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_vals = harmonicity.values[0]
            hnr_vals = hnr_vals[hnr_vals != -200]
            features["hnr_mean"] = float(np.mean(hnr_vals)) if hnr_vals.size > 0 else 0.0
            features["hnr_std"] = float(np.std(hnr_vals)) if hnr_vals.size > 0 else 0.0
        except Exception:
            features["hnr_mean"] = 0.0
            features["hnr_std"] = 0.0

        # Jitter & Shimmer (via PointProcess)
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["jitter"] = float(jitter) if jitter is not None and not np.isnan(jitter) else 0.0
            features["shimmer"] = float(shimmer) if shimmer is not None and not np.isnan(shimmer) else 0.0
        except Exception:
            features["jitter"] = 0.0
            features["shimmer"] = 0.0

        return features

    # --------------------------
    # Temporal / pause / voiced features
    # --------------------------
    def extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features: Dict[str, float] = {}
        try:
            energy = librosa.feature.rms(y=audio)[0]
            threshold_pct = self.config.get("temporal", {}).get("energy_percentile", 30)
            threshold = np.percentile(energy, threshold_pct)
            is_speech = energy > threshold

            all_pauses: List[float] = []
            short_pauses: List[float] = []
            long_pauses: List[float] = []

            in_pause = False
            pause_start = 0
            hop_length = 512
            frame_duration = hop_length / sr

            for i, speaking in enumerate(is_speech):
                if not speaking and not in_pause:
                    in_pause = True
                    pause_start = i
                elif speaking and in_pause:
                    pause_duration = (i - pause_start) * frame_duration
                    if pause_duration > 0.1:
                        all_pauses.append(pause_duration)
                        if 0.15 <= pause_duration <= 0.5:
                            short_pauses.append(pause_duration)
                        elif pause_duration > 2.0:
                            long_pauses.append(pause_duration)
                    in_pause = False

            total_duration = len(audio) / sr if sr > 0 else 0.0

            if all_pauses:
                features["pause_count"] = float(len(all_pauses))
                features["pause_mean"] = float(np.mean(all_pauses))
                features["pause_std"] = float(np.std(all_pauses))
                features["pause_total"] = float(np.sum(all_pauses))
                features["pause_ratio"] = float(features["pause_total"] / total_duration) if total_duration > 0 else 0.0
                features["pause_variability"] = float(features["pause_std"] / features["pause_mean"]) if features["pause_mean"] > 0 else 0.0
            else:
                features["pause_count"] = 0.0
                features["pause_mean"] = 0.0
                features["pause_std"] = 0.0
                features["pause_total"] = 0.0
                features["pause_ratio"] = 0.0
                features["pause_variability"] = 0.0

            features["long_pause_count"] = float(len(long_pauses))
            features["long_pause_total"] = float(np.sum(long_pauses)) if long_pauses else 0.0
            features["hesitation_count"] = float(len(short_pauses))
            features["hesitation_rate"] = float(len(short_pauses) / total_duration) if total_duration > 0 else 0.0

            voiced_frames = np.sum(is_speech)
            total_frames = len(is_speech)
            features["voiced_ratio"] = float(voiced_frames / total_frames) if total_frames > 0 else 0.0
            phonation_time = voiced_frames * frame_duration
            features["phonation_time_ratio"] = float(phonation_time / total_duration) if total_duration > 0 else 0.0

        except Exception:
            # default zero-filled temporal features
            temporal_keys = [
                "pause_count", "pause_mean", "pause_std", "pause_total", "pause_ratio", "pause_variability",
                "long_pause_count", "long_pause_total", "hesitation_count", "hesitation_rate",
                "voiced_ratio", "phonation_time_ratio"
            ]
            for k in temporal_keys:
                features[k] = 0.0

        return features

    # --------------------------
    # Formants (F1-F3 frequency & bandwidth)
    # --------------------------
    def extract_formants(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features: Dict[str, float] = {}
        audio = audio.astype(np.float64)
        maxval = np.max(np.abs(audio)) if audio.size > 0 else 1.0
        if maxval > 1.0:
            audio = audio / maxval

        try:
            sound = parselmouth.Sound(audio, sr)
            pitch = call(sound, "To Pitch", 0.0, 75, 400)
            freqs = pitch.selected_array["frequency"]
            voiced_mask = freqs > 0
            voiced_times = pitch.xs()[voiced_mask]

            if len(voiced_times) == 0:
                raise Exception("No voiced frames for formant extraction")

            max_formant = min(int(0.40 * sr), 5000)
            formant = call(sound, "To Formant (burg)", 0.0, 5, max_formant, 0.025, 50)

            num_samples = min(50, len(voiced_times))
            idx = np.linspace(0, len(voiced_times) - 1, num_samples).astype(int)
            times = voiced_times[idx]

            for i in range(1, 4):
                freq_vals: List[float] = []
                bw_vals: List[float] = []
                for t in times:
                    try:
                        freq = call(formant, "Get value at time", i, float(t), "Hertz", "Linear")
                        bw = call(formant, "Get bandwidth at time", i, float(t), "Hertz", "Linear")
                    except Exception:
                        freq = None
                        bw = None
                    if freq is not None and np.isfinite(freq) and freq > 0:
                        freq_vals.append(freq)
                    if bw is not None and np.isfinite(bw) and bw > 0:
                        bw_vals.append(bw)

                features[f"F{i}_mean"] = float(np.mean(freq_vals)) if freq_vals else 0.0
                features[f"F{i}_std"] = float(np.std(freq_vals)) if freq_vals else 0.0
                features[f"F{i}_bandwidth_mean"] = float(np.mean(bw_vals)) if bw_vals else 0.0
                features[f"F{i}_bandwidth_std"] = float(np.std(bw_vals)) if bw_vals else 0.0

        except Exception:
            for i in range(1, 4):
                features[f"F{i}_mean"] = 0.0
                features[f"F{i}_std"] = 0.0
                features[f"F{i}_bandwidth_mean"] = 0.0
                features[f"F{i}_bandwidth_std"] = 0.0

        return features

    # --------------------------
    # CPP (Cepstral Peak Prominence) - keep mean, median, count
    # --------------------------
    def extract_cpp(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features: Dict[str, float] = {}
        try:
            from scipy.stats import linregress

            audio = audio.astype(np.float64)
            maxval = np.max(np.abs(audio)) if audio.size > 0 else 0.0
            if maxval > 0:
                audio = audio / maxval

            frame_length = int(0.04 * sr)
            hop_length = int(0.01 * sr)
            if frame_length <= 0 or hop_length <= 0 or len(audio) < frame_length:
                features["cpp_mean"] = 0.0
                features["cpp_median"] = 0.0
                features["cpp_count"] = 0.0
                return features

            min_quefrency = 1.0 / 330
            max_quefrency = 1.0 / 60
            min_quefrency_idx = int(min_quefrency * sr)
            max_quefrency_idx = int(max_quefrency * sr)

            cpp_values: List[float] = []
            num_frames = (len(audio) - frame_length) // hop_length + 1

            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio[start:end]
                if len(frame) < 3:
                    continue
                window = np.hamming(len(frame))
                windowed_frame = frame * window
                n_fft = 2 ** int(np.ceil(np.log2(len(windowed_frame) * 4)))
                spectrum = np.fft.fft(windowed_frame, n=n_fft)
                log_spectrum = np.log(np.abs(spectrum) + 1e-10)
                cepstrum = np.fft.ifft(log_spectrum).real
                cepstrum = cepstrum[: n_fft // 2]

                scale_factor = n_fft / frame_length
                search_start = max(1, int(min_quefrency_idx * scale_factor))
                search_end = min(len(cepstrum) - 1, int(max_quefrency_idx * scale_factor))
                if search_end <= search_start:
                    continue
                search_region = cepstrum[search_start:search_end]
                if len(search_region) < 3:
                    continue
                peak_idx_local = np.argmax(search_region)
                peak_idx = search_start + peak_idx_local
                peak_value = cepstrum[peak_idx]

                trend_start = max(1, int(0.001 * sr * scale_factor))
                trend_end = search_end
                if trend_end <= trend_start + 2:
                    continue

                x_trend = np.arange(trend_start, trend_end)
                y_trend = cepstrum[trend_start:trend_end]
                try:
                    slope, intercept, _, _, _ = linregress(x_trend, y_trend)
                    trend_value_at_peak = slope * peak_idx + intercept
                    cpp = peak_value - trend_value_at_peak
                    if np.isfinite(cpp) and cpp > 0:
                        cpp_values.append(cpp)
                except Exception:
                    continue

            if len(cpp_values) > 0:
                features["cpp_mean"] = float(np.mean(cpp_values))
                features["cpp_median"] = float(np.median(cpp_values))
                features["cpp_count"] = float(len(cpp_values))
            else:
                features["cpp_mean"] = 0.0
                features["cpp_median"] = 0.0
                features["cpp_count"] = 0.0

        except Exception:
            features["cpp_mean"] = 0.0
            features["cpp_median"] = 0.0
            features["cpp_count"] = 0.0

        return features

    # --------------------------
    # Speaking & articulation rate
    # --------------------------
    def extract_speaking_rate(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features: Dict[str, float] = {}
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.2, wait=3)
            duration = len(audio) / sr if sr > 0 else 0.0
            syllables = len(peaks)
            features["speaking_rate"] = float(syllables / duration) if duration > 0 else 0.0

            energy = librosa.feature.rms(y=audio)[0]
            speech_frames = energy > np.percentile(energy, 40)
            hop_length = 512
            articulation_time = np.sum(speech_frames) * (hop_length / sr) if sr > 0 else 0.0
            features["articulation_rate"] = float(syllables / articulation_time) if articulation_time > 0 else 0.0

            if len(peaks) > 1:
                window_size = int(sr * 2)
                hop_size = int(sr * 0.5)
                local_rates: List[float] = []
                peak_samples = peaks * hop_length
                for start in range(0, max(1, len(audio) - window_size), hop_size):
                    end = start + window_size
                    peaks_in_window = np.sum((peak_samples >= start) & (peak_samples < end))
                    window_duration = window_size / sr
                    local_rate = peaks_in_window / window_duration
                    local_rates.append(local_rate)
                if local_rates:
                    features["speech_rate_variability"] = float(np.std(local_rates))
                    features["speech_rate_mean_local"] = float(np.mean(local_rates))
                else:
                    features["speech_rate_variability"] = 0.0
                    features["speech_rate_mean_local"] = features["speaking_rate"]
            else:
                features["speech_rate_variability"] = 0.0
                features["speech_rate_mean_local"] = features["speaking_rate"]

        except Exception:
            features["speaking_rate"] = 0.0
            features["articulation_rate"] = 0.0
            features["speech_rate_variability"] = 0.0
            features["speech_rate_mean_local"] = 0.0

        return features

    # --------------------------
    # Main extraction driver
    # --------------------------
    def extract_all_features(self, audio_path: str, use_preprocessing: bool = True) -> Dict[str, float]:
        if use_preprocessing:
            audio_raw, sr = self._load_audio(audio_path)
            audio_proc, sr = self.preprocess_for_features(audio_path)
        else:
            audio, sr = self._load_audio(audio_path)

        all_features: Dict[str, float] = {}

        # Prosodic & voice quality
        # Pitch & voice quality MUST use raw audio
        pros = self.extract_prosodic_features(audio_raw, sr)
        all_features.update(pros)

        # Temporal / pauses
        temp = self.extract_temporal_features(audio_proc, sr)
        all_features.update(temp)

        # Formants
        form = self.extract_formants(audio_raw, sr)
        all_features.update(form)

        # CPP
        cpp = self.extract_cpp(audio_proc, sr)
        all_features.update(cpp)

        # Speaking & articulation rate
        rate = self.extract_speaking_rate(audio_raw, sr)
        all_features.update(rate)

        # OpenSMILE: only if available
        if self.opensmile_available and self.smile is not None:
            try:
                smile_df = self.smile.process_file(audio_path)
                for col in smile_df.columns:
                    all_features[f"opensmile_{col}"] = float(smile_df[col].values[0])
            except Exception:
                # keep going without opensmile features
                pass

        # duration & sample rate metadata
        all_features["duration"] = float(len(audio_raw) / sr if sr > 0 else 0.0)
        all_features["sample_rate"] = int(sr)

        # Apply useful-feature filter (removes irrelevant openSMILE functionals)
        filtered = filter_useful_features(all_features)
        return filtered

    # --------------------------
    # Batch processing helper
    # --------------------------
    def batch_extract(self, input_dir: str, output_csv: str, use_preprocessing: bool = True) -> pd.DataFrame:
        input_path = Path(input_dir)
        exts = self.config.get("audio_extensions", ["mp3", "wav"])
        audio_files = []
        for ext in exts:
            audio_files.extend(sorted(input_path.glob(f"*.{ext}")))

        logger.info("Found %d audio files to extract features", len(audio_files))
        all_results: List[Dict[str, float]] = []

        for i, audio_file in enumerate(audio_files, start=1):
            logger.info("[%d/%d] Processing %s", i, len(audio_files), audio_file.name)
            try:
                feats = self.extract_all_features(str(audio_file), use_preprocessing=use_preprocessing)
                feats["filename"] = audio_file.name
                # Try robust participant_id extraction
                try:
                    feats["participant_id"] = audio_file.name[:-8]
                except Exception:
                    feats["participant_id"] = audio_file.name[:-8]
                all_results.append(feats)
                logger.info("Extracted features for %s", audio_file.name)
            except Exception as exc:
                logger.exception("Error processing %s: %s", audio_file.name, exc)

        df = pd.DataFrame(all_results)
        meta_cols = ["filename", "participant_id", "duration", "sample_rate"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + feature_cols] if len(df) > 0 else df

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("Saved features to %s", output_csv)
        return df


# --------------------------
# CLI entrypoint
# --------------------------
def main():
    # Load environment-aware defaults if available
    default_input = os.getenv("DEFAULT_INPUT_AUDIO_DIR", "datasets/output/patient_audio")
    default_output = os.getenv("DEFAULT_OUTPUT_DIR", "datasets/output/acoustic_features/features.csv")
    sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    parser = argparse.ArgumentParser(description="Acoustic feature extraction (cleaned)")
    parser.add_argument("--input", type=str, default=default_input, help="Input directory or single audio file")
    parser.add_argument("--output", type=str, default=default_output, help="Output CSV file path")
    parser.add_argument("--no-preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--batch", action="store_true", help="Process a directory of audio files")
    parser.add_argument("--config", type=str, default="dementia",
                        choices=["dementia", "voice", "therapy", "psychiatric", "custom"],
                        help="Configuration preset")
    args = parser.parse_args()

    # Load config presets
    if args.config == "dementia":
        cfg = ClinicalAudioConfig.get_dementia_assessment_config()
    elif args.config == "voice":
        cfg = ClinicalAudioConfig.get_voice_disorder_config()
    elif args.config == "therapy":
        cfg = ClinicalAudioConfig.get_speech_therapy_config()
    elif args.config == "psychiatric":
        cfg = ClinicalAudioConfig.get_psychiatric_interview_config()
    elif args.config == "custom":
        cfg = ClinicalAudioConfig.get_custom_config()
    else:
        cfg = None

    if cfg is None:
        cfg = CLINICAL_AUDIO_CONFIG #use default one without too much preset

    # Ensure sample rate is set consistently
    cfg["sample_rate"] = sample_rate
    # Provide default audio extensions if not present
    cfg.setdefault("audio_extensions", ["mp3", "wav"])

    extractor = AcousticFeatureExtractor(cfg)
    if args.batch:
        extractor.batch_extract(args.input, args.output, use_preprocessing=not args.no_preprocessing)
    elif os.path.isfile(args.input):
        # Single file mode: input is a path & filename to an audio file
        audio_input = args.input
        if 'mp3' in audio_input.lower() or 'wav' in audio_input.lower():
            logger.warning("openSMILE is not available; some features may be missing for MP3 files.")
            feats = extractor.extract_all_features(audio_input, use_preprocessing=not args.no_preprocessing)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(args.output, "w") as f:
                json.dump(feats, f, indent=2)
            logger.info("Saved features to %s", args.output)
        else:
            logger.error("Unsupported audio file format: %s", audio_input)
    else:
        logger.error("Input path is not a valid file, please use --batch for directory: %s", args.input)


if __name__ == "__main__":
    main()
