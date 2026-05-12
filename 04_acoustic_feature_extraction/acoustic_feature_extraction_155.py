#!/usr/bin/env python3
"""
Acoustic Feature Extraction Pipeline for Clinical Audio Analysis
Now reads useful parameters from clinical_audio_config by default.
Includes prosodic, spectral, temporal, formant and clinical features.

This code can extract 155 features for each given audio.

"""
import os
import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import opensmile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# clinical config
from clinical_audio_config import ClinicalAudioConfig

# Try to initialize opensmile; if not present, set flag
try:
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    _opensmile_available = True
except Exception:
    smile = None
    _opensmile_available = False


class AcousticFeatureExtractor:
    """
    Extract acoustic features for clinical analysis using config-driven parameters.
    """

    def __init__(self, config: Optional[Dict] = None):

        if config is None:
            config = ClinicalAudioConfig.get_dementia_assessment_config()
            config['name'] = 'dementia_assessment'
        self.config = config
        self.sr = config.get("sample_rate", 16000)
        self.opensmile_available = _opensmile_available and (smile is not None)
        if self.opensmile_available:
            self.smile = smile

    def _load_audio(self, audio_path: str, use_preproc_sr: bool = True) -> Tuple[np.ndarray, int]:
        sr = self.sr if use_preproc_sr else None
        audio, sr_loaded = librosa.load(audio_path, sr=sr)
        return audio, sr_loaded

    def preprocess_for_features(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Conservative preprocessing optimized for feature extraction.
        Uses config parameters for filtering, silence handling, and normalization.
        """
        audio, sr = self._load_audio(audio_path)
        # Remove DC offset if configured
        if self.config.get("filtering", {}).get("remove_dc_offset", True):
            audio = audio - np.mean(audio)

        # Use short silence trimming (but preserve pauses < threshold)
        sh = self.config.get("silence_handling", {})
        preserve_short = sh.get("preserve_short_pauses", True)
        min_pause = sh.get("min_silence_duration", 2.0)
        top_db = sh.get("silence_threshold_db", 30)
        intervals = librosa.effects.split(audio, top_db=top_db)
        non_silent = []
        last_end = 0
        for start, end in intervals:
            if last_end > 0:
                gap = (start - last_end) / sr
                if gap < min_pause and preserve_short:
                    # keep the pause segment between last_end and start
                    non_silent.append(audio[last_end:start])
                else:
                    # add short padding to indicate long gap
                    padding = int(sh.get("pause_padding", 0.1) * sr)
                    non_silent.append(np.zeros(padding))
            non_silent.append(audio[start:end])
            last_end = end
        if non_silent:
            audio_trimmed = np.concatenate(non_silent)
        else:
            audio_trimmed = audio

        # Bandpass filter if configured (gentle)
        fcfg = self.config.get("filtering", {})
        if fcfg.get("bandpass", True):
            from scipy import signal as _sig
            nyq = sr / 2.0
            low = max(0.0001, fcfg.get("lowcut", 70) / nyq)
            high = min(0.9999, fcfg.get("highcut", 8000) / nyq)
            if low < high:
                sos = _sig.butter(fcfg.get("filter_order", 3), [low, high], btype='band', output='sos')
                try:
                    audio_filtered = _sig.sosfiltfilt(sos, audio_trimmed)
                except Exception:
                    audio_filtered = audio_trimmed
            else:
                audio_filtered = audio_trimmed
        else:
            audio_filtered = audio_trimmed

        # Light RMS normalization based on config (preserve dynamics)
        norm = self.config.get("normalization", {})
        if norm.get("method", "rms") == "rms":
            target = norm.get("target_rms", 0.05)
            cur = np.sqrt(np.mean(audio_filtered**2)) if audio_filtered.size > 0 else 0.0
            if cur > 0:
                audio_normalized = audio_filtered * (target / cur)
            else:
                audio_normalized = audio_filtered
        else:
            audio_normalized = audio_filtered

        return audio_normalized, sr

    # --------------------------
    # Prosodic features
    # --------------------------
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict:
        features = {}
        try:
            sound = parselmouth.Sound(audio, sr)
        except Exception:
            # fallback: cast sampling rate to int and create Sound
            sound = parselmouth.Sound(audio.astype(np.float64), int(sr))

        # Pitch - store the pitch object for voice breaks analysis
        pitch = None
        try:
            pitch = call(sound, "To Pitch", 0.0,
                         self.config.get("prosody", {}).get("pitch_floor", 75),
                         self.config.get("prosody", {}).get("pitch_ceiling", 600))
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
        except Exception as e:
            print(f"Pitch extraction warning: {e}")
            pitch_values = np.array([])

        if pitch_values.size > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_min'] = float(np.min(pitch_values))
            features['pitch_max'] = float(np.max(pitch_values))
            features['pitch_range'] = float(features['pitch_max'] - features['pitch_min'])
            features['pitch_percentile_25'] = float(np.percentile(pitch_values, 25))
            features['pitch_percentile_75'] = float(np.percentile(pitch_values, 75))
            features['pitch_iqr'] = float(features['pitch_percentile_75'] - features['pitch_percentile_25'])
            features['pitch_cv'] = float(features['pitch_std'] / features['pitch_mean']) if features['pitch_mean'] > 0 else 0.0
            if pitch_values.size > 1:
                t = np.arange(pitch_values.size)
                slope, _ = np.polyfit(t, pitch_values, 1)
                features['pitch_slope'] = float(slope)
                # NEW: pitch_variation_rate - rate of pitch change over time
                pitch_diff = np.abs(np.diff(pitch_values))
                features['pitch_variation_rate'] = float(np.mean(pitch_diff)) if len(pitch_diff) > 0 else 0.0
            else:
                features['pitch_slope'] = 0.0
                features['pitch_variation_rate'] = 0.0
        else:
            for k in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',
                      'pitch_percentile_25', 'pitch_percentile_75', 'pitch_iqr', 'pitch_cv', 
                      'pitch_slope', 'pitch_variation_rate']:
                features[k] = 0.0

        # Voice breaks analysis - requires valid pitch object
        # Voice breaks are gaps/interruptions in the pitch contour (associated with AD)
        # Initialize all voice break features first to ensure they always exist
        features['voice_breaks_count'] = 0.0
        features['voice_breaks_rate'] = 0.0
        features['voice_break_degree'] = 0.0
        features['voice_breaks_count_alt'] = 0.0
        features['unvoiced_fraction'] = 0.0
        
        if pitch is not None:
            # Get duration for rate calculations
            try:
                duration = sound.get_total_duration()
            except:
                duration = len(audio) / sr if sr > 0 else 1.0
            
            # ALTERNATIVE METHOD (reliable): calculate from pitch array directly
            # This should always work if we have a pitch object
            try:
                all_pitch = pitch.selected_array['frequency']
                if len(all_pitch) > 0:
                    voiced_mask = all_pitch > 0
                    
                    # Count transitions (voiced -> unvoiced) = voice breaks
                    transitions = np.diff(voiced_mask.astype(int))
                    voice_to_unvoice = int(np.sum(transitions == -1))  # voiced to unvoiced
                    features['voice_breaks_count_alt'] = float(voice_to_unvoice)
                    features['voice_breaks_count'] = float(voice_to_unvoice)  # Use as primary too
                    
                    # Calculate voice breaks per second
                    features['voice_breaks_rate'] = float(voice_to_unvoice / duration) if duration > 0 else 0.0
                    
                    # Fraction of unvoiced frames
                    total_frames = len(all_pitch)
                    voiced_frames = int(np.sum(voiced_mask))
                    unvoiced_fraction = (total_frames - voiced_frames) / total_frames
                    features['unvoiced_fraction'] = float(unvoiced_fraction)
                    features['voice_break_degree'] = float(unvoiced_fraction)  # Use as primary too
                    
                    print(f"Voice breaks: count={voice_to_unvoice}, rate={features['voice_breaks_rate']:.2f}/s, unvoiced={unvoiced_fraction:.4f}")
            except Exception as e:
                print(f"Voice breaks alternative method error: {e}")
            
            # PRAAT METHOD (may not work in all versions): try to get official Praat values
            # These will overwrite the alternative values if successful
            try:
                voice_breaks_praat = call(pitch, "Count voice breaks", 0.0, 0.0)
                if voice_breaks_praat is not None:
                    features['voice_breaks_count'] = float(voice_breaks_praat)
                    features['voice_breaks_rate'] = float(voice_breaks_praat / duration) if duration > 0 else 0.0
                    print(f"Praat voice breaks count: {voice_breaks_praat}")
            except Exception as e:
                # Praat method not available, keep alternative values
                pass
            
            try:
                voice_break_degree_praat = call(pitch, "Get fraction of locally unvoiced frames", 0.0, 0.0)
                if voice_break_degree_praat is not None and not np.isnan(voice_break_degree_praat):
                    features['voice_break_degree'] = float(voice_break_degree_praat)
                    print(f"Praat voice break degree: {voice_break_degree_praat}")
            except Exception as e:
                # Praat method not available, keep alternative values
                pass

        # Intensity
        try:
            intensity = call(sound, "To Intensity", 75, 0.0)
            intensity_values = intensity.values[0]
            features['intensity_mean'] = float(np.mean(intensity_values))
            features['intensity_std'] = float(np.std(intensity_values))
            features['intensity_min'] = float(np.min(intensity_values))
            features['intensity_max'] = float(np.max(intensity_values))
            features['intensity_range'] = float(features['intensity_max'] - features['intensity_min'])
        except Exception:
            features['intensity_mean'] = 0.0
            features['intensity_std'] = 0.0
            features['intensity_min'] = 0.0
            features['intensity_max'] = 0.0
            features['intensity_range'] = 0.0

        # HNR (harmonicity)
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_vals = harmonicity.values[0]
            hnr_vals = hnr_vals[hnr_vals != -200]
            features['hnr_mean'] = float(np.mean(hnr_vals)) if hnr_vals.size > 0 else 0.0
            features['hnr_std'] = float(np.std(hnr_vals)) if hnr_vals.size > 0 else 0.0
        except Exception:
            features['hnr_mean'] = 0.0
            features['hnr_std'] = 0.0

        # jitter / shimmer
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter'] = float(jitter) if not np.isnan(jitter) else 0.0
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer'] = float(shimmer) if not np.isnan(shimmer) else 0.0
        except Exception:
            features['jitter'] = 0.0
            features['shimmer'] = 0.0

        return features

    # --------------------------
    # Spectral features
    # --------------------------
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        features = {}

        # spectral centroid, according to literature, not very useful.
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        except Exception:
            features['spectral_centroid_mean'] = 0.0
            features['spectral_centroid_std'] = 0.0      


        # rolloff / bandwidth
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        except Exception:
            features['spectral_rolloff_mean'] = 0.0
            features['spectral_rolloff_std'] = 0.0
            features['spectral_bandwidth_mean'] = 0.0
            features['spectral_bandwidth_std'] = 0.0

        # zcr
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
        except Exception:
            features['zcr_mean'] = 0.0
            features['zcr_std'] = 0.0


        # MFCCs  Originally, there are 13 features, however, only the first 6 are more useful
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=6)
            for i in range(6):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            mfcc_delta = librosa.feature.delta(mfccs)
            for i in range(6):
                features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
        except Exception:
            for i in range(6):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
                features[f'mfcc_delta_{i}_mean'] = 0.0
                features[f'mfcc_delta_{i}_std'] = 0.0

        return features

    # --------------------------
    # Temporal & Rhythm features
    # --------------------------
    def extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict:
        features = {}
        # onset rate
        try:
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['onset_rate'] = float(len(onset_frames) / (len(audio) / sr)) if len(audio) > 0 else 0.0
        except Exception:
            features['onset_rate'] = 0.0

        # tempo - kept but noted as less useful for dementia detection
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
        except Exception:
            features['tempo'] = 0.0

        # energy-based pause detection (config-driven) with enhanced pause metrics
        try:
            energy = librosa.feature.rms(y=audio)[0]
            threshold_pct = self.config.get("temporal", {}).get("energy_percentile", 30)
            threshold = np.percentile(energy, threshold_pct)
            is_speech = energy > threshold
            
            # Collect all pauses with their durations
            all_pauses = []
            short_pauses = []  # hesitations: 0.15-0.5s
            long_pauses = []   # clinically significant: > 2s
            
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
                    if pause_duration > 0.1:  # minimum pause threshold
                        all_pauses.append(pause_duration)
                        # Categorize pauses
                        if 0.15 <= pause_duration <= 0.5:
                            short_pauses.append(pause_duration)
                        elif pause_duration > 2.0:
                            long_pauses.append(pause_duration)
                    in_pause = False
            
            total_duration = len(audio) / sr
            
            if all_pauses:
                features['pause_count'] = float(len(all_pauses))
                features['pause_mean'] = float(np.mean(all_pauses))
                features['pause_std'] = float(np.std(all_pauses))
                features['pause_total'] = float(np.sum(all_pauses))
                features['pause_ratio'] = float(features['pause_total'] / total_duration)
                
                # NEW: pause_variability - Coefficient of variation in pause durations
                features['pause_variability'] = float(features['pause_std'] / features['pause_mean']) if features['pause_mean'] > 0 else 0.0
                
                # NEW: speech_to_pause_ratio - Inverse of pause_ratio for interpretability
                speech_time = total_duration - features['pause_total']
                features['speech_to_pause_ratio'] = float(speech_time / features['pause_total']) if features['pause_total'] > 0 else float('inf')
                # Cap at a reasonable maximum
                if features['speech_to_pause_ratio'] > 100:
                    features['speech_to_pause_ratio'] = 100.0
            else:
                features['pause_count'] = 0.0
                features['pause_mean'] = 0.0
                features['pause_std'] = 0.0
                features['pause_total'] = 0.0
                features['pause_ratio'] = 0.0
                features['pause_variability'] = 0.0
                features['speech_to_pause_ratio'] = 100.0  # Maximum when no pauses
            
            # NEW: long_pause_count - Pauses > 2 seconds (clinically significant)
            features['long_pause_count'] = float(len(long_pauses))
            features['long_pause_total'] = float(np.sum(long_pauses)) if long_pauses else 0.0
            
            # NEW: hesitation_rate - Short pauses (0.15-0.5s) indicating word-finding difficulty
            features['hesitation_count'] = float(len(short_pauses))
            features['hesitation_rate'] = float(len(short_pauses) / total_duration) if total_duration > 0 else 0.0

            # voiced_ratio and phonation_time_ratio
            voiced_frames = np.sum(is_speech)
            total_frames = len(is_speech)
            features['voiced_ratio'] = float(voiced_frames / total_frames) if total_frames > 0 else 0.0
            
            # NEW: phonation_time_ratio - Proportion of time spent phonating
            phonation_time = voiced_frames * frame_duration
            features['phonation_time_ratio'] = float(phonation_time / total_duration) if total_duration > 0 else 0.0
            
        except Exception as e:
            print(f"Temporal feature extraction warning: {e}")
            features['pause_count'] = 0.0
            features['pause_mean'] = 0.0
            features['pause_std'] = 0.0
            features['pause_total'] = 0.0
            features['pause_ratio'] = 0.0
            features['pause_variability'] = 0.0
            features['speech_to_pause_ratio'] = 100.0
            features['long_pause_count'] = 0.0
            features['long_pause_total'] = 0.0
            features['hesitation_count'] = 0.0
            features['hesitation_rate'] = 0.0
            features['voiced_ratio'] = 0.0
            features['phonation_time_ratio'] = 0.0

        return features

    # --------------------------
    # Clinical extras: formants, CPP, speaking/articulation rate
    # --------------------------
    
    def extract_formants(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract formant frequencies and bandwidths (F1-F3).
        Formant bandwidths are more discriminative than mean values for dementia detection.
        """
        features = {}

        #print("\n=== FORMANT DEBUG START ===")
        #print("audio dtype:", audio.dtype)
        #print("sr:", sr)
        #print("audio max abs:", np.max(np.abs(audio)))
        #print("audio length sec:", len(audio)/sr)

        # Praat requires float64
        audio = audio.astype(np.float64)

        # Scaling
        maxval = np.max(np.abs(audio))
        if maxval > 1.0:
            print("Normalizing audio because max abs >", maxval)
            audio = audio / maxval
        #print("new max abs:", np.max(np.abs(audio)))

        try:
            sound = parselmouth.Sound(audio, sr)
            print("Created Sound OK")

            # Detect voiced frames
            pitch = call(sound, "To Pitch", 0.0, 75, 400)
            freqs = pitch.selected_array['frequency']
            voiced_mask = freqs > 0
            voiced_times = pitch.xs()[voiced_mask]

            #print("Total pitch frames:", len(freqs))
            #print("Voiced frames:", np.sum(voiced_mask))

            if len(voiced_times) == 0:
                print("❌ No voiced frames detected → cannot compute formants")
                raise Exception("No voiced frames")

            # Build Formant object
            max_formant = min(0.40 * sr, 5000)
            #print("Using max_formant:", max_formant)

            formant = call(sound, "To Formant (burg)", 
                        0.0, 5, max_formant, 0.025, 50)
            #print("Formant object created")

            # Sample 50 voiced timestamps
            num_samples = min(50, len(voiced_times))
            idx = np.linspace(0, len(voiced_times)-1, num_samples).astype(int)
            times = voiced_times[idx]

            # Extract formant frequencies and bandwidths for F1, F2, F3
            for i in range(1, 4):
                freq_vals = []
                bw_vals = []
                
                for t in times:
                    # Get formant frequency
                    freq = call(formant, "Get value at time", i, float(t), "Hertz", "Linear")
                    # Get formant bandwidth
                    bw = call(formant, "Get bandwidth at time", i, float(t), "Hertz", "Linear")

                    # Print first few raw values for debugging
                    #if len(freq_vals) < 3:
                    #    print(f"F{i} t={t:.4f}, freq={freq}, bw={bw}")

                    if freq is not None and np.isfinite(freq) and freq > 0:
                        freq_vals.append(freq)
                    if bw is not None and np.isfinite(bw) and bw > 0:
                        bw_vals.append(bw)

                #print(f"Formant {i}: collected {len(freq_vals)} freq values, {len(bw_vals)} bandwidth values")

                # Formant frequency statistics
                if len(freq_vals) > 0:
                    features[f'F{i}_mean'] = float(np.mean(freq_vals))
                    features[f'F{i}_std'] = float(np.std(freq_vals))
                else:
                    features[f'F{i}_mean'] = 0.0
                    features[f'F{i}_std'] = 0.0
                
                # NEW: Formant bandwidth statistics
                if len(bw_vals) > 0:
                    features[f'F{i}_bandwidth_mean'] = float(np.mean(bw_vals))
                    features[f'F{i}_bandwidth_std'] = float(np.std(bw_vals))
                else:
                    features[f'F{i}_bandwidth_mean'] = 0.0
                    features[f'F{i}_bandwidth_std'] = 0.0

        except Exception as e:
            print("❌ Exception during formant extraction:", e)
            for i in range(1, 4):
                features[f'F{i}_mean'] = 0.0
                features[f'F{i}_std'] = 0.0
                features[f'F{i}_bandwidth_mean'] = 0.0
                features[f'F{i}_bandwidth_std'] = 0.0

        print("=== FORMANT DEBUG END ===\n")
        return features



    def extract_cpp(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract Cepstral Peak Prominence (CPP) using a native Python implementation.
        CPP is a robust measure of voice periodicity and quality.
        
        CPP is calculated as the difference between the cepstral peak amplitude
        and the corresponding value on the regression line (trend) at the same quefrency.
        """
        features = {}
        try:
            from scipy.signal import find_peaks, peak_prominences
            from scipy.stats import linregress
            
            # Ensure float64
            audio = audio.astype(np.float64)
            
            # Normalize audio
            maxval = np.max(np.abs(audio))
            if maxval > 0:
                audio = audio / maxval
            
            # Parameters for CPP calculation
            frame_length = int(0.04 * sr)  # 40ms frames
            hop_length = int(0.01 * sr)    # 10ms hop
            
            # Quefrency range for pitch (60-330 Hz typical human voice range)
            min_quefrency = 1.0 / 330  # ~3ms (330 Hz)
            max_quefrency = 1.0 / 60   # ~16.7ms (60 Hz)
            
            # Convert to sample indices
            min_quefrency_idx = int(min_quefrency * sr)
            max_quefrency_idx = int(max_quefrency * sr)
            
            cpp_values = []
            
            # Process audio in frames
            num_frames = (len(audio) - frame_length) // hop_length + 1
            
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio[start:end]
                
                # Apply Hamming window
                window = np.hamming(len(frame))
                windowed_frame = frame * window
                
                # Zero-pad for better frequency resolution
                n_fft = 2 ** int(np.ceil(np.log2(len(windowed_frame) * 4)))
                
                # Compute power spectrum
                spectrum = np.fft.fft(windowed_frame, n=n_fft)
                log_spectrum = np.log(np.abs(spectrum) + 1e-10)
                
                # Compute real cepstrum (IFFT of log spectrum)
                cepstrum = np.fft.ifft(log_spectrum).real
                
                # Take only the first half (real cepstrum is symmetric)
                cepstrum = cepstrum[:n_fft // 2]
                
                # Adjust quefrency indices for zero-padded FFT
                scale_factor = n_fft / frame_length
                search_start = max(1, int(min_quefrency_idx * scale_factor))
                search_end = min(len(cepstrum) - 1, int(max_quefrency_idx * scale_factor))
                
                if search_end <= search_start:
                    continue
                
                # Find the peak in the quefrency range of interest
                search_region = cepstrum[search_start:search_end]
                
                if len(search_region) < 3:
                    continue
                
                # Find the maximum peak in the search region
                peak_idx_local = np.argmax(search_region)
                peak_idx = search_start + peak_idx_local
                peak_value = cepstrum[peak_idx]
                
                # Compute the regression line (trend) over the cepstrum
                # Use quefrency range from 0.001s to max_quefrency for trend calculation
                trend_start = max(1, int(0.001 * sr * scale_factor))
                trend_end = search_end
                
                if trend_end <= trend_start + 2:
                    continue
                
                x_trend = np.arange(trend_start, trend_end)
                y_trend = cepstrum[trend_start:trend_end]
                
                # Linear regression to find the trend line
                try:
                    slope, intercept, _, _, _ = linregress(x_trend, y_trend)
                    
                    # Calculate expected value at peak quefrency from trend line
                    trend_value_at_peak = slope * peak_idx + intercept
                    
                    # CPP is the difference between peak and trend
                    cpp = peak_value - trend_value_at_peak
                    
                    if np.isfinite(cpp) and cpp > 0:
                        cpp_values.append(cpp)
                except Exception:
                    continue
            
            # Calculate statistics
            if len(cpp_values) > 0:
                features['cpp_mean'] = float(np.mean(cpp_values))
                features['cpp_std'] = float(np.std(cpp_values))
                features['cpp_min'] = float(np.min(cpp_values))
                features['cpp_max'] = float(np.max(cpp_values))
                features['cpp_median'] = float(np.median(cpp_values))
                features['cpp_count'] = float(len(cpp_values))
                # CPPS approximation (smoothed CPP - just use mean as proxy)
                features['cpps'] = features['cpp_mean']
                print(f"CPP extraction successful: {len(cpp_values)} frames, mean={features['cpp_mean']:.4f}")
            else:
                print("Warning: No valid CPP values extracted")
                features['cpp_mean'] = 0.0
                features['cpp_std'] = 0.0
                features['cpp_min'] = 0.0
                features['cpp_max'] = 0.0
                features['cpp_median'] = 0.0
                features['cpp_count'] = 0.0
                features['cpps'] = 0.0
                
        except Exception as e:
            print(f"CPP extraction error: {e}")
            import traceback
            traceback.print_exc()
            features['cpp_mean'] = 0.0
            features['cpp_std'] = 0.0
            features['cpp_min'] = 0.0
            features['cpp_max'] = 0.0
            features['cpp_median'] = 0.0
            features['cpp_count'] = 0.0
            features['cpps'] = 0.0
            
        return features

    def extract_speaking_rate(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract speaking rate features including variability measures.
        """
        features = {}
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.2, wait=3)
            duration = len(audio) / sr if sr > 0 else 0.0
            syllables = len(peaks)
            features['speaking_rate'] = float(syllables / duration) if duration > 0 else 0.0

            # articulation rate: count only voiced frames (energy > percentile)
            energy = librosa.feature.rms(y=audio)[0]
            speech_frames = energy > np.percentile(energy, 40)
            hop_length = 512
            articulation_time = np.sum(speech_frames) * (hop_length / sr) if sr > 0 else 0.0
            features['articulation_rate'] = float(syllables / articulation_time) if articulation_time > 0 else 0.0
            
            # NEW: speech_rate_variability - Standard deviation of local speech rates
            # Calculate local speech rates in sliding windows
            if len(peaks) > 1:
                window_size = int(sr * 2)  # 2-second windows
                hop_size = int(sr * 0.5)   # 0.5-second hop
                local_rates = []
                
                # Convert peak indices to sample positions
                peak_samples = peaks * hop_length
                
                for start in range(0, len(audio) - window_size, hop_size):
                    end = start + window_size
                    # Count peaks within this window
                    peaks_in_window = np.sum((peak_samples >= start) & (peak_samples < end))
                    window_duration = window_size / sr
                    local_rate = peaks_in_window / window_duration
                    local_rates.append(local_rate)
                
                if local_rates:
                    features['speech_rate_variability'] = float(np.std(local_rates))
                    features['speech_rate_mean_local'] = float(np.mean(local_rates))
                else:
                    features['speech_rate_variability'] = 0.0
                    features['speech_rate_mean_local'] = features['speaking_rate']
            else:
                features['speech_rate_variability'] = 0.0
                features['speech_rate_mean_local'] = features['speaking_rate']
                
        except Exception as e:
            print(f"Speaking rate extraction warning: {e}")
            features['speaking_rate'] = 0.0
            features['articulation_rate'] = 0.0
            features['speech_rate_variability'] = 0.0
            features['speech_rate_mean_local'] = 0.0
        return features

    # --------------------------
    # Main extraction
    # --------------------------
    def extract_all_features(self, audio_path: str, use_preprocessing: bool = True) -> Dict:
        if use_preprocessing:
            audio, sr = self.preprocess_for_features(audio_path)
        else:
            audio, sr = self._load_audio(audio_path)

        all_features = {}
        # prosodic
        pros = self.extract_prosodic_features(audio, sr)
        all_features.update(pros)

        # spectral, not that useful according to literature
        #spec = self.extract_spectral_features(audio, sr)
        #all_features.update(spec)

        # temporal
        temp = self.extract_temporal_features(audio, sr)
        all_features.update(temp)

        # formants
        form = self.extract_formants(audio, sr)
        all_features.update(form)

        # cpp
        cpp = self.extract_cpp(audio, sr)
        all_features.update(cpp)

        # speaking / articulation rate
        rate = self.extract_speaking_rate(audio, sr)
        all_features.update(rate)

        # OpenSMILE functional features if available
        if self.opensmile_available:
            try:
                smile_df = self.smile.process_file(audio_path)
                for col in smile_df.columns:
                    all_features[f'opensmile_{col}'] = float(smile_df[col].values[0])
            except Exception:
                pass

        all_features['duration'] = float(len(audio) / sr if sr > 0 else 0.0)
        all_features['sample_rate'] = int(sr)

        return all_features

    def batch_extract(self, input_dir: str, output_csv: str, use_preprocessing: bool = True) -> pd.DataFrame:
        input_path = Path(input_dir)
        audio_files = list(input_path.glob("*.mp3"))
        print(f"Found {len(audio_files)} audio files to extract features")

        all_results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_file.name}")
            try:
                feats = self.extract_all_features(str(audio_file), use_preprocessing=use_preprocessing)
                feats['filename'] = audio_file.name
                feats['participant_id'] = audio_file.name[:-8]
                all_results.append(feats)
                print(f"✓ Extracted features for {audio_file.name}")
            except Exception as e:
                print(f"✗ Error {audio_file.name}: {e}")
            if i>10:
                break  # for testing purposes, remove this line for full batch processing

        df = pd.DataFrame(all_results)
        meta_cols = ['filename', 'participant_id', 'duration', 'sample_rate']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + feature_cols] if len(df) > 0 else df
        
        # Create output directory if it doesn't exist
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_csv, index=False)
        print(f"Saved features to {output_csv}")
        return df


if __name__ == "__main__":
    import argparse


    # Load environment variables
    load_dotenv()

    # Load settings from .env
    DEFAULT_INPUT_AUDIO_DIR = os.getenv("DEFAULT_INPUT_AUDIO_DIR", "datasets/output/patient_audio")
    DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR", "datasets/output/acoustic_features/features.json")
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_EXTENSIONS = [ext.strip() for ext in os.getenv("AUDIO_EXTENSIONS", "mp3,wav").split(",")]
    input_dir = DEFAULT_INPUT_AUDIO_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    audio_sample_rate = AUDIO_SAMPLE_RATE
    audio_channels = AUDIO_CHANNELS
    
    parser = argparse.ArgumentParser(description="Acoustic feature extraction (config-driven)")
    parser.add_argument("--input", type=str, default=input_dir) #provide a filename if processing a single file.
    parser.add_argument("--output", type=str, default=output_dir) 
    parser.add_argument("--no-preprocessing", action="store_true") # Skip preprocessing step if explicitly set this parameter in command line.
    parser.add_argument("--batch", action="store_true") #please always include this flag for batch processing, otherwise, it will try to process a single file.
    parser.add_argument("--config", type=str, default="dementia",
                        choices=["dementia", "voice", "therapy", "psychiatric", "custom"])
    args = parser.parse_args()

    if args.config == "dementia":
        cfg = ClinicalAudioConfig.get_dementia_assessment_config()
    elif args.config == "voice":
        cfg = ClinicalAudioConfig.get_voice_disorder_config()
    elif args.config == "therapy":
        cfg = ClinicalAudioConfig.get_speech_therapy_config()
    elif args.config == "psychiatric":
        cfg = ClinicalAudioConfig.get_psychiatric_interview_config()
    else:
        cfg = ClinicalAudioConfig.get_custom_config()

    extractor = AcousticFeatureExtractor(cfg)
    if args.batch:
        df = extractor.batch_extract(args.input, args.output, use_preprocessing=not args.no_preprocessing)
        print(df.head())
    else:
        feats = extractor.extract_all_features(args.input, use_preprocessing=not args.no_preprocessing)
        import json
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(feats, f, indent=2)
        print(f"Saved features to {args.output}")