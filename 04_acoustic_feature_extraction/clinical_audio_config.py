#!/usr/bin/env python3

"""
Clinical Audio Processing Configuration
Optimized settings for medical/diagnostic audio analysis
Preserves speech characteristics important for clinical assessment

Updated to include adaptive processing settings for the unified denoising pipeline.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Default Clinical Audio Configuration
# =============================================================================

CLINICAL_AUDIO_CONFIG = {
    "sample_rate": 16000,

    "filtering": {
        "bandpass": True,
        "lowcut": 80,
        "highcut": 7500,
        "filter_order": 2
    },

    "vad": {
        "mode": 2   # 0–3, recommended: 2 for clinical speech
    },

    "noise_reduction": {
        "aggressiveness": 0.2,
        "use_wiener": False,
        "max_amplification": 3.0
    },

    "normalization": {
        "method": "rms",
        "target_rms": 0.05,
        "max_amplification": 3.0
    },
    
    # NEW: Adaptive processing settings
    "adaptive_processing": {
        "enabled": True,              # Enable automatic noise profile detection
        "min_highpass": 70,           # Minimum highpass cutoff (Hz)
        "max_highpass": 200,          # Maximum highpass cutoff (Hz)
        "lowpass": 7500,              # Default lowpass cutoff (Hz)
        "use_multiband": True,        # Enable multiband processing for heavy noise
        "use_spectral_subtraction": True  # Enable spectral subtraction
    }
}


# =============================================================================
# Configuration Presets for Different Clinical Scenarios
# =============================================================================

class ClinicalAudioConfig:
    """
    Configuration presets for different clinical audio scenarios
    """
    
    @staticmethod
    def get_dementia_assessment_config() -> Dict[str, Any]:
        """
        Settings optimized for dementia/cognitive assessment
        Preserves: pauses, hesitations, voice quality changes
        """
        return {
            # Audio parameters
            "sample_rate": 16000,
            "bit_depth": 16,
            
            # Normalization settings
            "normalization": {
                "method": "rms",
                "target_rms": 0.05,
                "preserve_dynamics": True,
                "local_normalization": False,
                "max_amplification": 3.0
            },
            
            # Noise reduction settings - CONSERVATIVE
            "noise_reduction": {
                "method": "adaptive",
                "aggressiveness": 0.15,  # Very conservative for clinical
                "stationary": False,
                "preserve_speech_threshold": 0.7,
                "noise_gate_threshold": -45,
                "use_wiener": False,
                "max_amplification": 3.0
            },
            
            # Filtering settings
            "filtering": {
                "bandpass": True,
                "lowcut": 70,
                "highcut": 8000,
                "filter_order": 2,
                "remove_dc_offset": True
            },
            
            # Adaptive processing - enabled for varied recording conditions
            "adaptive_processing": {
                "enabled": True,
                "min_highpass": 70,
                "max_highpass": 200,
                "lowpass": 8000,
                "use_multiband": True,
                "use_spectral_subtraction": True
            },
            
            # Silence handling - CRITICAL for clinical analysis
            "silence_handling": {
                "remove_long_silence": True,
                "silence_threshold_db": 30,
                "min_silence_duration": 2.0,
                "preserve_short_pauses": True,
                "pause_padding": 0.1
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.95,
                "min_speech_duration": 0.5,
                "snr_threshold": 10
            }
        }
    
    @staticmethod
    def get_voice_disorder_config() -> Dict[str, Any]:
        """
        Settings for voice disorder assessment
        Preserves: breathiness, hoarseness, pitch irregularities
        """
        return {
            # Audio parameters
            "sample_rate": 44100,
            "bit_depth": 24,
            
            # Normalization settings
            "normalization": {
                "method": "peak",
                "target_peak": 0.9,
                "preserve_dynamics": True,
                "local_normalization": False
            },
            
            # Noise reduction - MINIMAL
            "noise_reduction": {
                "method": "spectral",
                "aggressiveness": 0.1,  # Very low for voice disorder
                "stationary": True,
                "preserve_speech_threshold": 0.9,
                "noise_gate_threshold": -50,
                "use_wiener": False
            },
            
            # Filtering - MINIMAL
            "filtering": {
                "bandpass": True,
                "lowcut": 50,
                "highcut": 12000,
                "filter_order": 2,
                "remove_dc_offset": True
            },
            
            # Adaptive processing - conservative for voice quality
            "adaptive_processing": {
                "enabled": True,
                "min_highpass": 50,
                "max_highpass": 100,  # Very conservative
                "lowpass": 12000,
                "use_multiband": False,  # Disabled to preserve voice quality
                "use_spectral_subtraction": False
            },
            
            # Silence handling
            "silence_handling": {
                "remove_long_silence": False,
                "silence_threshold_db": 40,
                "min_silence_duration": 5.0,
                "preserve_short_pauses": True,
                "pause_padding": 0.2
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.99,
                "min_speech_duration": 0.1,
                "snr_threshold": 5
            }
        }
    
    @staticmethod
    def get_speech_therapy_config() -> Dict[str, Any]:
        """
        Settings for speech therapy sessions
        Balanced between clarity and preservation
        """
        return {
            # Audio parameters
            "sample_rate": 16000,
            "bit_depth": 16,
            
            # Normalization settings
            "normalization": {
                "method": "rms",
                "target_rms": 0.06,
                "preserve_dynamics": True,
                "local_normalization": False
            },
            
            # Noise reduction - MODERATE
            "noise_reduction": {
                "method": "adaptive",
                "aggressiveness": 0.25,
                "stationary": False,
                "preserve_speech_threshold": 0.75,
                "noise_gate_threshold": -40,
                "use_wiener": False
            },
            
            # Filtering
            "filtering": {
                "bandpass": True,
                "lowcut": 80,
                "highcut": 7500,
                "filter_order": 3,
                "remove_dc_offset": True
            },
            
            # Adaptive processing - moderate
            "adaptive_processing": {
                "enabled": True,
                "min_highpass": 80,
                "max_highpass": 180,
                "lowpass": 7500,
                "use_multiband": True,
                "use_spectral_subtraction": True
            },
            
            # Silence handling
            "silence_handling": {
                "remove_long_silence": True,
                "silence_threshold_db": 35,
                "min_silence_duration": 1.5,
                "preserve_short_pauses": True,
                "pause_padding": 0.1
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.95,
                "min_speech_duration": 0.3,
                "snr_threshold": 15
            }
        }
    
    @staticmethod
    def get_psychiatric_interview_config() -> Dict[str, Any]:
        """
        Settings for psychiatric interview recordings
        Preserves: emotional prosody, speech patterns, pauses
        """
        return {
            # Audio parameters
            "sample_rate": 16000,
            "bit_depth": 16,
            
            # Normalization settings
            "normalization": {
                "method": "rms",
                "target_rms": 0.05,
                "preserve_dynamics": True,
                "local_normalization": False
            },
            
            # Noise reduction - CONSERVATIVE
            "noise_reduction": {
                "method": "adaptive",
                "aggressiveness": 0.15,
                "stationary": False,
                "preserve_speech_threshold": 0.8,
                "noise_gate_threshold": -45,
                "use_wiener": False
            },
            
            # Filtering
            "filtering": {
                "bandpass": True,
                "lowcut": 70,
                "highcut": 8000,
                "filter_order": 2,
                "remove_dc_offset": True
            },
            
            # Adaptive processing
            "adaptive_processing": {
                "enabled": True,
                "min_highpass": 70,
                "max_highpass": 180,
                "lowpass": 8000,
                "use_multiband": True,
                "use_spectral_subtraction": True
            },
            
            # Silence handling - preserve pauses for psychological analysis
            "silence_handling": {
                "remove_long_silence": False,
                "silence_threshold_db": 30,
                "min_silence_duration": 3.0,
                "preserve_short_pauses": True,
                "pause_padding": 0.15
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.95,
                "min_speech_duration": 0.5,
                "snr_threshold": 10
            }
        }
    
    @staticmethod
    def get_noisy_environment_config() -> Dict[str, Any]:
        """
        Settings for recordings in noisy environments
        More aggressive noise reduction while still preserving clinical features
        """
        return {
            # Audio parameters
            "sample_rate": 16000,
            "bit_depth": 16,
            
            # Normalization settings
            "normalization": {
                "method": "rms",
                "target_rms": 0.05,
                "preserve_dynamics": True,
                "local_normalization": False
            },
            
            # Noise reduction - MORE AGGRESSIVE
            "noise_reduction": {
                "method": "adaptive",
                "aggressiveness": 0.35,  # Higher for noisy environments
                "stationary": False,
                "preserve_speech_threshold": 0.6,
                "noise_gate_threshold": -35,
                "use_wiener": True  # Enable Wiener for additional smoothing
            },
            
            # Filtering
            "filtering": {
                "bandpass": True,
                "lowcut": 100,  # Higher to cut more rumble
                "highcut": 7000,  # Lower to cut more hiss
                "filter_order": 4,  # Steeper rolloff
                "remove_dc_offset": True
            },
            
            # Adaptive processing - full processing enabled
            "adaptive_processing": {
                "enabled": True,
                "min_highpass": 100,
                "max_highpass": 250,  # Allow higher cutoff for heavy rumble
                "lowpass": 7000,
                "use_multiband": True,
                "use_spectral_subtraction": True
            },
            
            # Silence handling
            "silence_handling": {
                "remove_long_silence": True,
                "silence_threshold_db": 25,
                "min_silence_duration": 1.0,
                "preserve_short_pauses": True,
                "pause_padding": 0.1
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.95,
                "min_speech_duration": 0.5,
                "snr_threshold": 5  # Lower threshold acceptable for noisy recordings
            }
        }
    
    @staticmethod
    def get_custom_config() -> Dict[str, Any]:
        """
        Base template for custom configurations
        """
        return CLINICAL_AUDIO_CONFIG.copy()


# =============================================================================
# Audio Processing Helper Classes
# =============================================================================

class ClinicalAudioProcessor:
    """
    Main audio processor for clinical recordings
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize processor with configuration
        
        Args:
            config: Configuration dictionary, defaults to CLINICAL_AUDIO_CONFIG
        """
        self.config = config if config is not None else CLINICAL_AUDIO_CONFIG
        self.sr = self.config.get("sample_rate", 16000)
    
    def validate_audio(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Validate audio quality for clinical use
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Validation report dictionary
        """
        report = {
            "passed": True,
            "issues": [],
            "metrics": {}
        }
        
        # Check duration
        duration = len(audio) / sr
        report["metrics"]["duration"] = duration
        
        min_duration = self.config.get("quality_control", {}).get("min_speech_duration", 0.5)
        if duration < min_duration:
            report["issues"].append(f"Duration too short: {duration:.2f}s < {min_duration}s")
            report["passed"] = False
        
        # Check amplitude
        max_amp = np.max(np.abs(audio))
        report["metrics"]["max_amplitude"] = max_amp
        
        max_allowed = self.config.get("quality_control", {}).get("max_amplitude", 0.95)
        if max_amp > max_allowed:
            report["issues"].append(f"Possible clipping: max amplitude {max_amp:.3f}")
        
        # Check for silence
        rms = np.sqrt(np.mean(audio**2))
        report["metrics"]["rms"] = rms
        
        if rms < 0.001:
            report["issues"].append("Audio appears to be mostly silent")
            report["passed"] = False
        
        # Estimate SNR
        frame_rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if len(frame_rms) > 10:
            noise_floor = np.percentile(frame_rms, 10)
            signal_level = np.percentile(frame_rms, 90)
            snr = 20 * np.log10(signal_level / noise_floor) if noise_floor > 0 else 40
        else:
            snr = 20
        
        report["metrics"]["snr_estimate"] = snr
        
        snr_threshold = self.config.get("quality_control", {}).get("snr_threshold", 10)
        if snr < snr_threshold:
            report["issues"].append(f"Low SNR: {snr:.1f}dB < {snr_threshold}dB")
        
        return report
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, output_path: str, 
                   sr: int = None, subtype: str = "PCM_16"):
        """
        Save audio to file
        
        Args:
            audio: Audio array
            output_path: Output file path
            sr: Sample rate (uses config if not provided)
            subtype: Audio subtype for soundfile
        """
        if sr is None:
            sr = self.sr
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(output_path), audio, sr, subtype=subtype)


# =============================================================================
# Utility Functions
# =============================================================================

def get_config_for_scenario(scenario: str) -> Dict[str, Any]:
    """
    Get appropriate configuration for a given scenario
    
    Args:
        scenario: One of 'dementia', 'voice', 'therapy', 'psychiatric', 'noisy', 'custom'
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'dementia': ClinicalAudioConfig.get_dementia_assessment_config,
        'voice': ClinicalAudioConfig.get_voice_disorder_config,
        'therapy': ClinicalAudioConfig.get_speech_therapy_config,
        'psychiatric': ClinicalAudioConfig.get_psychiatric_interview_config,
        'noisy': ClinicalAudioConfig.get_noisy_environment_config,
        'custom': ClinicalAudioConfig.get_custom_config,
    }
    
    if scenario.lower() in configs:
        return configs[scenario.lower()]()
    else:
        print(f"Unknown scenario '{scenario}', using default config")
        return CLINICAL_AUDIO_CONFIG.copy()


def print_config_summary(config: Dict[str, Any], name: str = "Configuration"):
    """
    Print a summary of configuration settings
    
    Args:
        config: Configuration dictionary
        name: Name to display
    """
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    print(f"Sample rate: {config.get('sample_rate', 'N/A')} Hz")
    
    nr = config.get('noise_reduction', {})
    print(f"Noise reduction aggressiveness: {nr.get('aggressiveness', 'N/A')}")
    print(f"Wiener filter: {nr.get('use_wiener', False)}")
    
    filt = config.get('filtering', {})
    print(f"Bandpass: {filt.get('lowcut', 'N/A')}-{filt.get('highcut', 'N/A')} Hz")
    print(f"Filter order: {filt.get('filter_order', 'N/A')}")
    
    adapt = config.get('adaptive_processing', {})
    print(f"Adaptive processing: {adapt.get('enabled', False)}")
    if adapt.get('enabled', False):
        print(f"  Highpass range: {adapt.get('min_highpass', 'N/A')}-{adapt.get('max_highpass', 'N/A')} Hz")
        print(f"  Multiband: {adapt.get('use_multiband', False)}")
        print(f"  Spectral subtraction: {adapt.get('use_spectral_subtraction', False)}")
    
    print('='*60)
