"""
Clinical-grade, CPU-friendly denoising pipeline
Preserves pitch / jitter / pauses for clinical speech analysis

Features:
- Automatic noise profile detection (clean, mild, heavy rumble, hiss, etc.)
- Adaptive processing based on detected noise type
- Preserves speech frequencies while targeting specific noise bands
- Compatible with batch processing of varied recording conditions
"""

from __future__ import annotations
import numpy as np
import librosa
from typing import Dict, Any, Tuple, Optional
from scipy import signal
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import noisereduce as nr
    _NR_AVAILABLE = True
except Exception:
    _NR_AVAILABLE = False

# Toggle WebRTC VAD usage
USE_WEBRTC_VAD = False

if USE_WEBRTC_VAD:
    import webrtcvad
    print("WebRTC VAD is enabled for ClinicalDenoiser.")

output_audio_dir = os.getenv("PROCESSED_AUDIO_DIR", "datasets/output/processed_audio")
max_amplification = float(os.getenv("MAX_AMPLIFICATION", "3.0"))

# =============================================================================
# Noise Profile Detection
# =============================================================================

class NoiseProfile(Enum):
    """Detected noise profile categories"""
    CLEAN = "clean"                    # Minimal noise, light processing
    MILD_NOISE = "mild_noise"          # Some background noise
    HEAVY_RUMBLE = "heavy_rumble"      # Strong low-frequency noise
    HIGH_HISS = "high_hiss"            # High-frequency noise
    MIXED_NOISE = "mixed_noise"        # Multiple noise types
    VERY_NOISY = "very_noisy"          # Severe noise issues


@dataclass
class NoiseAnalysis:
    """Results of noise analysis"""
    profile: NoiseProfile
    snr_estimate: float
    lf_ratio: float          # Low-freq to speech ratio
    hf_ratio: float          # High-freq to speech ratio
    peak_amplitude: float
    rms: float
    recommended_highpass: float
    recommended_lowpass: float
    band_energy: Dict[str, float]


# =============================================================================
# Main Denoiser Class
# =============================================================================

class ClinicalDenoiser:
    """
    Adaptive clinical denoiser that automatically detects noise profile
    and applies appropriate processing.
    
    Preserves speech characteristics important for clinical assessment:
    - Pitch and pitch variability
    - Jitter and shimmer
    - Pauses and hesitations
    - Voice quality markers
    """
    
    # Frequency bands for analysis (Hz)
    BANDS = {
        'sub_bass': (0, 60),
        'bass': (60, 150),
        'low_mid': (150, 300),
        'mid': (300, 2000),      # Primary speech frequencies
        'high_mid': (2000, 4000),
        'high': (4000, 8000)
    }
    
    # Thresholds for noise classification
    THRESHOLDS = {
        'lf_ratio_mild': 1.5,      # LF/speech ratio for mild rumble
        'lf_ratio_heavy': 3.0,     # LF/speech ratio for heavy rumble
        'hf_ratio_mild': 0.15,     # HF/speech ratio for mild hiss
        'hf_ratio_heavy': 0.3,     # HF/speech ratio for heavy hiss
        'snr_clean': 20,           # SNR threshold for "clean" audio
        'snr_noisy': 10,           # SNR threshold for "noisy" audio
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.sr = int(config.get("sample_rate", 16000))
        self.max_amplification = max_amplification

        # --- VAD ---
        if USE_WEBRTC_VAD:            
            vad_mode = config.get("vad", {}).get("mode", 2)
            self.vad = webrtcvad.Vad(vad_mode)

        # --- Noise reduction parameters ---
        nr_cfg = config.get("noise_reduction", {})
        self.aggressiveness = nr_cfg.get("aggressiveness", 0.15)
        self.use_wiener = nr_cfg.get("use_wiener", False)
        
        # --- Adaptive processing settings ---
        self.adaptive_cfg = config.get("adaptive_processing", {})
        self.enable_adaptive = self.adaptive_cfg.get("enabled", True)
        self.min_highpass = self.adaptive_cfg.get("min_highpass", 70)
        self.max_highpass = self.adaptive_cfg.get("max_highpass", 200)
        self.default_lowpass = self.adaptive_cfg.get("lowpass", 7500)
        
        # --- Verbose logging ---
        self.verbose = config.get("verbose", False)

    # -------------------------------------------------------------------------
    # Noise Analysis
    # -------------------------------------------------------------------------
    
    def analyze_noise(self, audio: np.ndarray) -> NoiseAnalysis:
        """
        Comprehensive noise analysis to determine processing strategy.
        """
        if audio.size == 0:
            return NoiseAnalysis(
                profile=NoiseProfile.CLEAN,
                snr_estimate=0, lf_ratio=0, hf_ratio=0,
                peak_amplitude=0, rms=0,
                recommended_highpass=self.min_highpass,
                recommended_lowpass=self.default_lowpass,
                band_energy={}
            )
        
        # Basic stats
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        # Compute STFT for spectral analysis
        n_fft = 2048
        stft = np.abs(librosa.stft(audio, n_fft=n_fft))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # Energy per band
        band_energy = {}
        for name, (low, high) in self.BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy[name] = np.mean(stft[mask, :]) if np.any(mask) else 0
        
        # Calculate ratios relative to speech band
        speech_energy = band_energy['mid']
        if speech_energy < 1e-10:
            speech_energy = 1e-10  # Avoid division by zero
            
        lf_energy = band_energy['sub_bass'] + band_energy['bass']
        hf_energy = band_energy['high']
        
        lf_ratio = lf_energy / speech_energy
        hf_ratio = hf_energy / speech_energy
        
        # Estimate SNR from quiet vs loud sections
        frame_rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if len(frame_rms) > 10:
            noise_floor = np.percentile(frame_rms, 10)
            signal_level = np.percentile(frame_rms, 90)
            snr_estimate = 20 * np.log10(signal_level / noise_floor) if noise_floor > 0 else 40
        else:
            snr_estimate = 20
        
        # Determine recommended highpass based on rumble
        recommended_hp = self._calculate_adaptive_highpass(
            stft, freqs, lf_ratio, speech_energy
        )
        
        # Determine recommended lowpass based on hiss
        if hf_ratio > self.THRESHOLDS['hf_ratio_heavy']:
            recommended_lp = 6000
        elif hf_ratio > self.THRESHOLDS['hf_ratio_mild']:
            recommended_lp = 7000
        else:
            recommended_lp = self.default_lowpass
        
        # Classify noise profile
        profile = self._classify_noise_profile(lf_ratio, hf_ratio, snr_estimate)
        
        return NoiseAnalysis(
            profile=profile,
            snr_estimate=snr_estimate,
            lf_ratio=lf_ratio,
            hf_ratio=hf_ratio,
            peak_amplitude=peak,
            rms=rms,
            recommended_highpass=recommended_hp,
            recommended_lowpass=recommended_lp,
            band_energy=band_energy
        )
    
    def _calculate_adaptive_highpass(self, stft: np.ndarray, freqs: np.ndarray,
                                      lf_ratio: float, speech_energy: float) -> float:
        """Calculate optimal highpass cutoff based on noise analysis."""
        if lf_ratio > self.THRESHOLDS['lf_ratio_heavy']:
            # Find where LF energy drops below speech energy
            freq_energy = np.mean(stft, axis=1)
            
            # Find highest frequency where LF energy exceeds 3x speech energy
            lf_mask = freqs < 300
            if np.any(lf_mask):
                lf_freqs = freqs[lf_mask]
                lf_energy_curve = freq_energy[lf_mask]
                
                # Find where energy drops to acceptable level relative to speech
                exceeds_threshold = lf_energy_curve > speech_energy * 3
                
                if np.any(exceeds_threshold):
                    # Get the highest frequency that still has excessive energy
                    last_high_idx = np.where(exceeds_threshold)[0][-1]
                    # Set cutoff above that frequency
                    recommended_hp = min(lf_freqs[last_high_idx] * 1.2, self.max_highpass)
                else:
                    recommended_hp = self.min_highpass
            else:
                recommended_hp = self.min_highpass
                
            # Ensure minimum for heavy rumble
            recommended_hp = max(recommended_hp, self.min_highpass + 30)
            
        elif lf_ratio > self.THRESHOLDS['lf_ratio_mild']:
            recommended_hp = self.min_highpass + 20
        else:
            recommended_hp = self.min_highpass
        
        # Clamp to configured limits
        return np.clip(recommended_hp, self.min_highpass, self.max_highpass)
    
    def _classify_noise_profile(self, lf_ratio: float, hf_ratio: float, 
                                 snr: float) -> NoiseProfile:
        """Classify the type of noise present."""
        has_rumble = lf_ratio > self.THRESHOLDS['lf_ratio_mild']
        has_heavy_rumble = lf_ratio > self.THRESHOLDS['lf_ratio_heavy']
        has_hiss = hf_ratio > self.THRESHOLDS['hf_ratio_mild']
        has_heavy_hiss = hf_ratio > self.THRESHOLDS['hf_ratio_heavy']
        is_clean = snr > self.THRESHOLDS['snr_clean']
        is_noisy = snr < self.THRESHOLDS['snr_noisy']
        
        if is_clean and not has_rumble and not has_hiss:
            return NoiseProfile.CLEAN
        elif has_heavy_rumble and has_heavy_hiss:
            return NoiseProfile.VERY_NOISY
        elif has_heavy_rumble:
            return NoiseProfile.HEAVY_RUMBLE
        elif has_heavy_hiss:
            return NoiseProfile.HIGH_HISS
        elif has_rumble or has_hiss:
            return NoiseProfile.MIXED_NOISE
        elif is_noisy:
            return NoiseProfile.MILD_NOISE
        else:
            return NoiseProfile.CLEAN

    # -------------------------------------------------------------------------
    # Filtering Methods
    # -------------------------------------------------------------------------
    
    def _apply_bandpass(self, audio: np.ndarray, lowcut: float, 
                        highcut: float, order: int = 4) -> np.ndarray:
        """Apply bandpass filter with safety checks."""
        nyq = self.sr / 2
        low = np.clip(lowcut / nyq, 1e-4, 0.99)
        high = np.clip(highcut / nyq, low + 0.01, 0.999)
        
        if not (0 < low < high < 1):
            return audio
            
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, audio)
    
    def _apply_highpass(self, audio: np.ndarray, cutoff: float, 
                        order: int = 4) -> np.ndarray:
        """Apply highpass filter."""
        nyq = self.sr / 2
        normalized = np.clip(cutoff / nyq, 1e-4, 0.99)
        sos = signal.butter(order, normalized, btype='high', output='sos')
        return signal.sosfiltfilt(sos, audio)

    # -------------------------------------------------------------------------
    # Advanced Processing Methods
    # -------------------------------------------------------------------------
    
    def _multiband_process(self, audio: np.ndarray, 
                           analysis: NoiseAnalysis) -> np.ndarray:
        """
        Apply frequency-dependent gain based on noise analysis.
        More aggressive in noisy bands, preserves speech.
        """
        n_fft = 2048
        hop_length = 512
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # Build gain curve based on noise profile
        gain = np.ones(len(freqs))
        agg = self.aggressiveness
        
        # Sub-bass: always reduce significantly
        sub_bass_gain = 0.1 + (1 - agg) * 0.2
        gain[freqs < 60] = sub_bass_gain
        
        # Bass: reduce based on LF ratio
        if analysis.lf_ratio > self.THRESHOLDS['lf_ratio_heavy']:
            bass_gain = 0.2 + (1 - agg) * 0.2
        elif analysis.lf_ratio > self.THRESHOLDS['lf_ratio_mild']:
            bass_gain = 0.5 + (1 - agg) * 0.2
        else:
            bass_gain = 0.8 + (1 - agg) * 0.2
        gain[(freqs >= 60) & (freqs < 150)] = bass_gain
        
        # Low-mid: gentle transition
        lowmid_gain = 0.7 + (1 - agg) * 0.2
        if analysis.lf_ratio > self.THRESHOLDS['lf_ratio_heavy']:
            lowmid_gain = 0.6 + (1 - agg) * 0.2
        gain[(freqs >= 150) & (freqs < 300)] = lowmid_gain
        
        # Speech band: preserve
        gain[(freqs >= 300) & (freqs < 3500)] = 1.0
        
        # High-mid: mostly preserve
        gain[(freqs >= 3500) & (freqs < 5000)] = 0.95
        
        # High: reduce based on HF ratio
        if analysis.hf_ratio > self.THRESHOLDS['hf_ratio_heavy']:
            high_gain = 0.5 + (1 - agg) * 0.2
        elif analysis.hf_ratio > self.THRESHOLDS['hf_ratio_mild']:
            high_gain = 0.7 + (1 - agg) * 0.15
        else:
            high_gain = 0.9
        gain[freqs >= 5000] = high_gain
        
        # Smooth gain curve to avoid artifacts
        gain = uniform_filter1d(gain, size=5)
        
        # Apply gain
        gain_matrix = gain[:, np.newaxis]
        magnitude_processed = magnitude * gain_matrix
        
        # Reconstruct
        stft_processed = magnitude_processed * np.exp(1j * phase)
        return librosa.istft(stft_processed, hop_length=hop_length, length=len(audio))
    
    def _spectral_subtraction(self, audio: np.ndarray,
                               noise_percentile: float = 8,
                               subtraction_factor: float = 0.3) -> np.ndarray:
        """Gentle spectral subtraction using noise estimated from quiet frames."""
        n_fft = 2048
        hop_length = 512
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from quietest frames
        frame_energy = np.sum(magnitude, axis=0)
        threshold = np.percentile(frame_energy, noise_percentile)
        quiet_frames = frame_energy <= threshold
        
        if np.sum(quiet_frames) > 3:
            noise_spectrum = np.mean(magnitude[:, quiet_frames], axis=1, keepdims=True)
        else:
            noise_spectrum = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Subtract with floor
        floor = magnitude * 0.1
        magnitude_clean = np.maximum(magnitude - subtraction_factor * noise_spectrum, floor)
        
        # Reconstruct
        stft_clean = magnitude_clean * np.exp(1j * phase)
        return librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))

    # -------------------------------------------------------------------------
    # Original Methods (preserved for compatibility)
    # -------------------------------------------------------------------------
    
    def _frame_generator(self, audio, frame_ms=30):
        """Generate frames for VAD processing."""
        frame_len = int(self.sr * frame_ms / 1000)
        for i in range(0, len(audio) - frame_len, frame_len):
            yield audio[i:i + frame_len]

    def _is_speech(self, frame):
        """Check if frame contains speech using VAD."""
        if not USE_WEBRTC_VAD or not hasattr(self, "vad"):
            return True  # assume speech, conservative choice
        else:
            pcm16 = (frame * 32768).astype(np.int16).tobytes()
            return self.vad.is_speech(pcm16, self.sr)

    def estimate_noise(self, audio):
        """
        Estimate noise conservatively.
        If WebRTC VAD is disabled, fall back to energy-based estimation.
        """
        if not USE_WEBRTC_VAD:
            return self.estimate_noise_energy_based(audio, self.sr)  

        # VAD-based estimation
        noise_frames = []
        for frame in self._frame_generator(audio):
            if not self._is_speech(frame):
                noise_frames.append(frame)

        if noise_frames:
            return np.concatenate(noise_frames)

        return audio
    
    def estimate_noise_energy_based(self, audio, sr, frame_ms=30):
        """
        Estimate noise using low-energy frames.
        No silence removal, no time compression.
        """
        frame_len = int(sr * frame_ms / 1000)
        hop_len = frame_len

        if audio.size < frame_len:
            return audio

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_len,
            hop_length=hop_len,
            center=False
        )[0]

        # Conservative: bottom 5% energy only
        thr = np.percentile(rms, 5)

        noise_frames = []
        for i, r in enumerate(rms):
            if r <= thr:
                start = i * hop_len
                end = start + frame_len
                if end <= len(audio):
                    noise_frames.append(audio[start:end])

        # Require minimum noise sample duration (at least 0.5s)
        min_noise_duration = int(sr * 0.5)

        if not noise_frames:
            return np.zeros(int(sr * 0.1))
            
        noise = np.concatenate(noise_frames)
        if len(noise) < min_noise_duration:
            return np.zeros(int(sr * 0.1))
        
        return noise
    
    def spectral_denoise_without_package(self, audio):
        """Apply noisereduce without VAD."""
        noise = self.estimate_noise_energy_based(audio, self.sr)

        if not _NR_AVAILABLE:
            return audio
            
        if noise.size < int(self.sr * 0.3) or np.max(np.abs(noise)) < 1e-6:
            return audio

        return nr.reduce_noise(
            y=audio,
            sr=self.sr,
            y_noise=noise,
            stationary=False,
            prop_decrease=min(self.aggressiveness, 0.2),
            n_fft=2048,
            n_std_thresh_stationary=1.5
        )

    def spectral_denoise(self, audio, noise):
        """Apply noisereduce with provided noise profile."""
        if not _NR_AVAILABLE:
            return audio
        
        if noise.size < self.sr * 0.3 or np.max(np.abs(noise)) < 1e-6:
            return audio
        
        return nr.reduce_noise(
            y=audio,
            sr=self.sr,
            y_noise=noise,
            stationary=False,
            prop_decrease=min(self.aggressiveness, 0.2),
            n_fft=2048,
            n_std_thresh_stationary=1.5
        )

    def wiener_filter(self, audio):
        """Apply Wiener filter if enabled."""
        if not self.use_wiener:
            return audio
        return signal.wiener(audio, mysize=51)

    # -------------------------------------------------------------------------
    # Main Denoise Method
    # -------------------------------------------------------------------------
    
    def denoise(self, audio: np.ndarray, 
                return_analysis: bool = False) -> np.ndarray | Tuple[np.ndarray, NoiseAnalysis]:
        """
        Main denoising method. Automatically adapts to noise profile if enabled.
        
        Args:
            audio: Input audio array
            return_analysis: If True, also return the NoiseAnalysis object
            
        Returns:
            Processed audio array, or tuple of (audio, analysis) if return_analysis=True
        """
        if audio.size == 0:
            if return_analysis:
                return audio, NoiseAnalysis(
                    profile=NoiseProfile.CLEAN, snr_estimate=0, lf_ratio=0,
                    hf_ratio=0, peak_amplitude=0, rms=0,
                    recommended_highpass=self.min_highpass,
                    recommended_lowpass=self.default_lowpass, band_energy={}
                )
            return audio

        # Remove DC offset
        audio = audio - np.mean(audio)
        original_rms = np.sqrt(np.mean(audio**2))
        
        # Analyze noise profile if adaptive processing is enabled
        if self.enable_adaptive:
            analysis = self.analyze_noise(audio)
            
            if self.verbose:
                print(f"  Noise profile: {analysis.profile.value}")
                print(f"  LF ratio: {analysis.lf_ratio:.2f}, HF ratio: {analysis.hf_ratio:.2f}")
                print(f"  Recommended HP: {analysis.recommended_highpass:.0f}Hz")
            
            # Apply profile-specific processing
            audio = self._denoise_adaptive(audio, analysis)
        else:
            analysis = None
            # Use original fixed processing
            audio = self._denoise_fixed(audio)
        
        # Final RMS normalization - preserve original level, don't amplify
        new_rms = np.sqrt(np.mean(audio**2))
        if new_rms > 0:
            if new_rms > original_rms * 1.1:
                audio = audio * (original_rms / new_rms)
            elif new_rms < original_rms * 0.5:
                # Restore some level if too quiet (but not beyond original)
                # limit the amplification factor
                gain = original_rms / new_rms
                gain = min(gain, self.max_amplification)
                audio = audio * gain
        
        if return_analysis:
            return audio, analysis
        return audio
    
    def _denoise_adaptive(self, audio: np.ndarray, 
                          analysis: NoiseAnalysis) -> np.ndarray:
        """Apply adaptive denoising based on noise profile."""
        fcfg = self.cfg.get("filtering", {})
        use_multiband = self.adaptive_cfg.get("use_multiband", True)
        use_spectral_sub = self.adaptive_cfg.get("use_spectral_subtraction", True)
        
        if analysis.profile == NoiseProfile.CLEAN:
            # Minimal processing - just gentle bandpass
            if self.verbose:
                print("  Processing: Minimal (clean audio)")
            lowcut = fcfg.get("lowcut", self.min_highpass)
            highcut = fcfg.get("highcut", self.default_lowpass)
            audio = self._apply_bandpass(audio, lowcut, highcut, order=2)
            
        elif analysis.profile == NoiseProfile.MILD_NOISE:
            # Light processing
            if self.verbose:
                print("  Processing: Light denoising")
            audio = self._apply_bandpass(
                audio, 
                analysis.recommended_highpass,
                analysis.recommended_lowpass, 
                order=3
            )
            if use_spectral_sub:
                audio = self._spectral_subtraction(audio, noise_percentile=5, 
                                                    subtraction_factor=0.2)
                                                    
        elif analysis.profile == NoiseProfile.HEAVY_RUMBLE:
            # Aggressive low-frequency removal
            if self.verbose:
                print(f"  Processing: Heavy rumble removal (HP={analysis.recommended_highpass:.0f}Hz)")
            # Higher order filter for steeper rolloff
            audio = self._apply_highpass(audio, analysis.recommended_highpass, order=5)
            if use_multiband:
                audio = self._multiband_process(audio, analysis)
            audio = self._apply_bandpass(
                audio,
                analysis.recommended_highpass,
                analysis.recommended_lowpass, 
                order=3
            )
            if use_spectral_sub:
                audio = self._spectral_subtraction(audio, noise_percentile=8,
                                                    subtraction_factor=0.35)
                                                    
        elif analysis.profile == NoiseProfile.HIGH_HISS:
            # Focus on high-frequency reduction
            if self.verbose:
                print(f"  Processing: Hiss reduction (LP={analysis.recommended_lowpass:.0f}Hz)")
            audio = self._apply_bandpass(
                audio,
                analysis.recommended_highpass,
                analysis.recommended_lowpass, 
                order=4
            )
            if use_multiband:
                audio = self._multiband_process(audio, analysis)
            if use_spectral_sub:
                audio = self._spectral_subtraction(audio, noise_percentile=8,
                                                    subtraction_factor=0.3)
                                                    
        elif analysis.profile in [NoiseProfile.MIXED_NOISE, NoiseProfile.VERY_NOISY]:
            # Full processing pipeline
            if self.verbose:
                print("  Processing: Full pipeline (mixed/heavy noise)")
            audio = self._apply_highpass(audio, analysis.recommended_highpass, order=5)
            if use_multiband:
                audio = self._multiband_process(audio, analysis)
            if use_spectral_sub:
                audio = self._spectral_subtraction(audio, noise_percentile=10,
                                                    subtraction_factor=0.4)
            # Apply noisereduce for very noisy
            if analysis.profile == NoiseProfile.VERY_NOISY and _NR_AVAILABLE:
                noise = self.estimate_noise(audio)
                if noise.size >= self.sr * 0.3:
                    audio = self.spectral_denoise(audio, noise)
            audio = self._apply_bandpass(
                audio,
                analysis.recommended_highpass,
                analysis.recommended_lowpass, 
                order=3
            )
        
        # Apply Wiener filter if enabled
        audio = self.wiener_filter(audio)
        
        return audio
    
    def _denoise_fixed(self, audio: np.ndarray) -> np.ndarray:
        """Original fixed denoising (non-adaptive mode)."""
        fcfg = self.cfg.get("filtering", {})
        
        if fcfg.get("bandpass", True):
            lowcut = fcfg.get("lowcut", 80)
            highcut = fcfg.get("highcut", 7500)
            audio = self._apply_bandpass(
                audio, lowcut, highcut, 
                order=fcfg.get("filter_order", 2)
            )

        noise = self.estimate_noise(audio)

        # Check if denoising is even needed
        audio_rms = np.sqrt(np.mean(audio ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2)) if noise.size > 0 else 0

        # If noise is too similar to audio, skip spectral denoising
        if noise_rms > audio_rms * 0.4:
            return self.wiener_filter(audio)
        
        if USE_WEBRTC_VAD:
            audio = self.spectral_denoise(audio, noise)
        else:
            audio = self.spectral_denoise_without_package(audio)
        
        audio = self.wiener_filter(audio)

        return audio


# =============================================================================
# Public API Functions
# =============================================================================

def clinical_denoise_audio(audio: np.ndarray, sr: int, 
                           config: Dict[str, Any]) -> np.ndarray:
    """
    Main entry point for clinical audio denoising.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        config: Configuration dictionary
        
    Returns:
        Processed audio array
    """
    # Ensure sample rate is in config
    config_with_sr = config.copy()
    config_with_sr["sample_rate"] = sr
    
    denoiser = ClinicalDenoiser(config_with_sr)
    return denoiser.denoise(audio)


def clinical_denoise_audio_with_analysis(audio: np.ndarray, sr: int,
                                          config: Dict[str, Any],
                                          verbose: bool = False) -> Tuple[np.ndarray, Optional[NoiseAnalysis]]:
    """
    Denoise with full analysis returned.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        config: Configuration dictionary
        verbose: Print processing steps
        
    Returns:
        Tuple of (processed_audio, noise_analysis)
    """
    config_with_sr = config.copy()
    config_with_sr["sample_rate"] = sr
    config_with_sr["verbose"] = verbose
    
    denoiser = ClinicalDenoiser(config_with_sr)
    return denoiser.denoise(audio, return_analysis=True)
