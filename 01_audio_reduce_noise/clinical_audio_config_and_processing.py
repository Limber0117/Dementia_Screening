#!/usr/bin/env python3

"""
Clinical Audio Processing Configuration
Optimized settings for medical/diagnostic audio analysis
Preserves speech characteristics important for clinical assessment

The main() function allows to compare the different clinical configurations, which generates four different audio files for the same input audio.

"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


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
            "sample_rate": 16000,  # Standard for speech analysis
            "bit_depth": 16,       # Sufficient for speech
            
            # Normalization settings
            "normalization": {
                "method": "rms",  # RMS better preserves dynamics than LUFS for clinical
                "target_rms": 0.05,  # Lower than default to preserve quiet speech
                "preserve_dynamics": True,
                "local_normalization": False  # Don't normalize locally - preserve natural variations
            },
            
            # Noise reduction settings - CONSERVATIVE
            "noise_reduction": {
                "method": "adaptive",  # Adapts to changing conditions
                "aggressiveness": 0.5,  # Low aggression (0-1 scale, default is 0.9)
                "stationary": False,    # Non-stationary for natural environments
                "preserve_speech_threshold": 0.7,  # High threshold to preserve speech
                "noise_gate_threshold": -45,  # dB - very low to keep quiet speech
            },
            
            # Filtering settings
            "filtering": {
                "bandpass": True,
                "lowcut": 70,  # Lower than typical to preserve voice characteristics
                "highcut": 8000,  # Standard speech upper limit
                "filter_order": 3,  # Lower order = gentler filtering
                "remove_dc_offset": True
            },
            
            # Silence handling - CRITICAL for clinical analysis
            "silence_handling": {
                "remove_long_silence": True,
                "silence_threshold_db": 30,  # Higher threshold to preserve quiet speech
                "min_silence_duration": 2.0,  # Only remove silences > 2 seconds
                "preserve_short_pauses": True,  # Keep pauses < 2 seconds
                "pause_padding": 0.1  # Keep 100ms padding around speech
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,  # Usually skip for clinical to preserve natural speech
                "pre_emphasis": 0.0,  # No pre-emphasis for clinical
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.95,
                "min_speech_duration": 0.5,  # Minimum speech segment to keep
                "snr_threshold": 10  # Minimum SNR in dB
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
            "sample_rate": 44100,  # Higher for voice quality analysis
            "bit_depth": 24,      # Higher bit depth for subtle changes
            
            # Normalization settings
            "normalization": {
                "method": "peak",  # Preserve exact amplitude relationships
                "target_peak": 0.9,
                "preserve_dynamics": True,
                "local_normalization": False
            },
            
            # Noise reduction - MINIMAL
            "noise_reduction": {
                "method": "spectral",
                "aggressiveness": 0.3,  # Very low
                "stationary": True,
                "preserve_speech_threshold": 0.9,  # Very high preservation
                "noise_gate_threshold": -50,
            },
            
            # Filtering - MINIMAL
            "filtering": {
                "bandpass": True,
                "lowcut": 50,  # Very low to preserve voice fundamental
                "highcut": 12000,  # Higher to preserve harmonics
                "filter_order": 2,  # Very gentle filtering
                "remove_dc_offset": True
            },
            
            # Silence handling
            "silence_handling": {
                "remove_long_silence": False,  # Keep all audio
                "silence_threshold_db": 40,
                "min_silence_duration": 5.0,
                "preserve_short_pauses": True,
                "pause_padding": 0.2
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,  # Never for voice disorders
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.99,
                "min_speech_duration": 0.1,
                "snr_threshold": 5  # Lower threshold for pathological voices
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
                "method": "lufs",  # Perceptual loudness
                "target_lufs": -20.0,  # Slightly louder for clarity
                "preserve_dynamics": True,
                "local_normalization": False
            },
            
            # Noise reduction - MODERATE
            "noise_reduction": {
                "method": "adaptive",
                "aggressiveness": 0.6,
                "stationary": False,
                "preserve_speech_threshold": 0.75,
                "noise_gate_threshold": -40,
            },
            
            # Filtering
            "filtering": {
                "bandpass": True,
                "lowcut": 80,
                "highcut": 8000,
                "filter_order": 4,
                "remove_dc_offset": True
            },
            
            # Silence handling
            "silence_handling": {
                "remove_long_silence": True,
                "silence_threshold_db": 25,
                "min_silence_duration": 1.5,
                "preserve_short_pauses": True,
                "pause_padding": 0.15
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": True,  # Mild enhancement for clarity
                "pre_emphasis": 0.95,
                "de_essing": False,
                "compressor": True
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
        Settings for psychiatric interviews
        Preserves: emotional prosody, speaking rate changes, hesitations
        """
        return {
            # Audio parameters
            "sample_rate": 16000,
            "bit_depth": 16,
            
            # Normalization settings
            "normalization": {
                "method": "rms",
                "target_rms": 0.07,
                "preserve_dynamics": True,
                "local_normalization": False  # Preserve emotional dynamics
            },
            
            # Noise reduction - CONSERVATIVE
            "noise_reduction": {
                "method": "wiener",  # Good for preserving prosody
                "aggressiveness": 0.4,
                "stationary": False,
                "preserve_speech_threshold": 0.8,
                "noise_gate_threshold": -45,
            },
            
            # Filtering
            "filtering": {
                "bandpass": True,
                "lowcut": 60,  # Lower to preserve emotional pitch changes
                "highcut": 10000,  # Higher for emotional prosody
                "filter_order": 3,
                "remove_dc_offset": True
            },
            
            # Silence handling - IMPORTANT
            "silence_handling": {
                "remove_long_silence": True,
                "silence_threshold_db": 35,
                "min_silence_duration": 3.0,  # Longer pauses may be significant
                "preserve_short_pauses": True,
                "pause_padding": 0.2
            },
            
            # Speech enhancement
            "speech_enhancement": {
                "apply": False,  # Preserve natural emotional expression
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            # Quality control
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.95,
                "min_speech_duration": 0.2,
                "snr_threshold": 12
            }
        }
    
    @staticmethod
    def get_custom_config(
            sample_rate: int = 16000,
            normalization_method: str = "rms",
            noise_reduction_aggressiveness: float = 0.5,
            preserve_pauses_under: float = 2.0,
            bandpass_range: tuple = (70, 8000)
        ) -> Dict[str, Any]:
        """
        Create a custom configuration
        
        Args:
            sample_rate: Target sample rate in Hz
            normalization_method: 'peak', 'rms', 'lufs', or 'none'
            noise_reduction_aggressiveness: 0-1, where 0 is no reduction
            preserve_pauses_under: Preserve pauses shorter than this (seconds)
            bandpass_range: (lowcut, highcut) in Hz
        
        Returns:
            Configuration dictionary
        """
        return {
            "sample_rate": sample_rate,
            "bit_depth": 16,
            
            "normalization": {
                "method": normalization_method,
                "target_rms": 0.05,
                "target_lufs": -23.0,
                "target_peak": 0.95,
                "preserve_dynamics": True,
                "local_normalization": False
            },
            
            "noise_reduction": {
                "method": "adaptive",
                "aggressiveness": noise_reduction_aggressiveness,
                "stationary": False,
                "preserve_speech_threshold": 1.0 - (noise_reduction_aggressiveness * 0.3),
                "noise_gate_threshold": -40 - (10 * noise_reduction_aggressiveness),
            },
            
            "filtering": {
                "bandpass": True,
                "lowcut": bandpass_range[0],
                "highcut": bandpass_range[1],
                "filter_order": 3,
                "remove_dc_offset": True
            },
            
            "silence_handling": {
                "remove_long_silence": True,
                "silence_threshold_db": 30,
                "min_silence_duration": preserve_pauses_under,
                "preserve_short_pauses": True,
                "pause_padding": 0.1
            },
            
            "speech_enhancement": {
                "apply": False,
                "pre_emphasis": 0.0,
                "de_essing": False,
                "compressor": False
            },
            
            "quality_control": {
                "check_clipping": True,
                "max_amplitude": 0.95,
                "min_speech_duration": 0.5,
                "snr_threshold": 10
            }
        }


class ClinicalAudioProcessor:
    """
    Audio processor with clinical-specific settings
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with clinical configuration
        
        Args:
            config: Configuration dictionary from ClinicalAudioConfig
        """
        self.config = config
        self.sample_rate = config["sample_rate"]
        
    def validate_audio(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Validate audio meets clinical quality standards
        
        Returns:
            Dictionary with validation results and metrics
        """
        validation_results = {
            "passed": True,
            "issues": [],
            "metrics": {}
        }
        
        # Check duration
        duration = len(audio) / sr
        validation_results["metrics"]["duration"] = duration
        
        if duration < self.config["quality_control"]["min_speech_duration"]:
            validation_results["issues"].append(f"Audio too short: {duration:.2f}s")
            validation_results["passed"] = False
        
        # Check for clipping
        max_amp = np.max(np.abs(audio))
        validation_results["metrics"]["max_amplitude"] = max_amp
        
        if max_amp >= 0.999:
            validation_results["issues"].append("Audio appears to be clipped")
            validation_results["passed"] = False
        
        # Estimate SNR
        noise_floor = np.percentile(np.abs(audio), 10)
        signal_peak = np.percentile(np.abs(audio), 90)
        
        if noise_floor > 0:
            snr_estimate = 20 * np.log10(signal_peak / noise_floor)
            validation_results["metrics"]["snr_estimate"] = snr_estimate
            
            if snr_estimate < self.config["quality_control"]["snr_threshold"]:
                validation_results["issues"].append(f"Low SNR: {snr_estimate:.1f} dB")
                validation_results["passed"] = False
        
        # Check for DC offset
        dc_offset = np.mean(audio)
        validation_results["metrics"]["dc_offset"] = dc_offset
        
        if abs(dc_offset) > 0.1:
            validation_results["issues"].append(f"Significant DC offset: {dc_offset:.3f}")
        
        # Check dynamic range
        dynamic_range = 20 * np.log10(max_amp / (noise_floor + 1e-10))
        validation_results["metrics"]["dynamic_range"] = dynamic_range
        
        return validation_results
    
    def apply_clinical_preprocessing(self, audio_path: str, 
                                    output_path: Optional[str] = None) -> tuple:
        """
        Apply clinical-grade preprocessing based on configuration
        
        Args:
            audio_path: Path to input audio
            output_path: Optional path to save processed audio
            
        Returns:
            Tuple of (processed_audio, sample_rate, validation_report)
        """
        import noisereduce as nr
        from scipy import signal
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.config["sample_rate"])
        original_audio = audio.copy()
        
        print(f"Loaded: {Path(audio_path).name}")
        print(f"Duration: {len(audio)/sr:.2f}s, Sample rate: {sr} Hz")
        
        # Step 1: Remove DC offset if needed
        if self.config["filtering"]["remove_dc_offset"]:
            audio = audio - np.mean(audio)
        
        # Step 2: Apply bandpass filter
        if self.config["filtering"]["bandpass"]:
            nyquist = sr / 2
            low = self.config["filtering"]["lowcut"] / nyquist
            high = min(self.config["filtering"]["highcut"] / nyquist, 0.99)
            
            if low < high:
                sos = signal.butter(
                    self.config["filtering"]["filter_order"], 
                    [low, high], 
                    btype='band', 
                    output='sos'
                )
                audio = signal.sosfiltfilt(sos, audio)
        
        # Step 3: Noise reduction (if specified)
        if self.config["noise_reduction"]["aggressiveness"] > 0:
            # Scale aggressiveness to noisereduce's prop_decrease (0-1)
            prop_decrease = self.config["noise_reduction"]["aggressiveness"]
            
            audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.config["noise_reduction"]["stationary"],
                prop_decrease=prop_decrease,
                n_std_thresh_stationary=2.0
            )
        
        # Step 4: Handle silence
        if self.config["silence_handling"]["remove_long_silence"]:
            # Detect non-silent intervals
            intervals = librosa.effects.split(
                audio,
                top_db=self.config["silence_handling"]["silence_threshold_db"],
                frame_length=2048,
                hop_length=512
            )
            
            # Process intervals to preserve short pauses
            processed_segments = []
            last_end = 0
            
            for start, end in intervals:
                # Check gap from last segment
                gap_duration = (start - last_end) / sr
                
                # Add padding or gap based on configuration
                if last_end > 0:  # Not the first segment
                    if gap_duration < self.config["silence_handling"]["min_silence_duration"]:
                        # Preserve this pause
                        if self.config["silence_handling"]["preserve_short_pauses"]:
                            processed_segments.append(audio[last_end:start])
                    else:
                        # Add configured padding
                        padding_samples = int(self.config["silence_handling"]["pause_padding"] * sr)
                        processed_segments.append(np.zeros(padding_samples))
                
                # Add the speech segment
                processed_segments.append(audio[start:end])
                last_end = end
            
            if processed_segments:
                audio = np.concatenate(processed_segments)
        
        # Step 5: Normalization
        norm_method = self.config["normalization"]["method"]
        
        if norm_method == "rms":
            current_rms = np.sqrt(np.mean(audio**2))
            if current_rms > 0:
                target_rms = self.config["normalization"]["target_rms"]
                audio = audio * (target_rms / current_rms)
        
        elif norm_method == "peak":
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                target_peak = self.config["normalization"]["target_peak"]
                audio = audio * (target_peak / max_val)
        
        elif norm_method == "lufs":
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            target_lufs = self.config["normalization"]["target_lufs"]
            audio = pyln.normalize.loudness(audio, loudness, target_lufs)
        
        # Step 6: Apply final amplitude limit
        max_amplitude = self.config["quality_control"]["max_amplitude"]
        audio = np.clip(audio, -max_amplitude, max_amplitude)
        
        # Step 7: Validate processed audio
        validation_report = self.validate_audio(audio, sr)
        
        # Add processing metrics
        validation_report["processing_metrics"] = {
            "original_duration": len(original_audio) / sr,
            "processed_duration": len(audio) / sr,
            "duration_change": (len(audio) - len(original_audio)) / sr,
            "amplitude_change": np.max(np.abs(audio)) / (np.max(np.abs(original_audio)) + 1e-10)
        }
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, audio, sr, subtype='PCM_16')
            print(f"Saved to: {output_path}")
        
        return audio, sr, validation_report
    
    def batch_process_clinical(self, input_dir: str, output_dir: str, 
                              report_path: str = None) -> Dict[str, Any]:
        """
        Process multiple clinical audio files with validation
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory for processed files
            report_path: Optional path to save processing report
            
        Returns:
            Dictionary with processing results and statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_files = list(input_path.glob("*.mp3")) + list(input_path.glob("*.wav")) 
        
        results = {
            "total_files": len(audio_files),
            "processed": 0,
            "failed": 0,
            "warnings": 0,
            "file_reports": {}
        }
        
        print(f"Found {len(audio_files)} audio files to process")
        print(f"Using configuration: {self.config.get('name', 'custom')}")
        print("=" * 60)
        
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            
            try:
                output_file = output_path / audio_file.name.replace(audio_file.suffix, '.wav')
                
                # Process with validation
                _, _, validation = self.apply_clinical_preprocessing(
                    str(audio_file),
                    str(output_file)
                )
                
                # Store results
                results["file_reports"][audio_file.name] = validation
                
                if validation["passed"]:
                    results["processed"] += 1
                    print(f"✓ Processed successfully")
                else:
                    results["warnings"] += 1
                    print(f"⚠ Processed with warnings: {', '.join(validation['issues'])}")
                
                # Print key metrics
                metrics = validation.get("metrics", {})
                print(f"  Duration: {metrics.get('duration', 0):.2f}s")
                print(f"  SNR: {metrics.get('snr_estimate', 0):.1f} dB")
                print(f"  Max amplitude: {metrics.get('max_amplitude', 0):.3f}")
                
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
                results["failed"] += 1
                results["file_reports"][audio_file.name] = {
                    "passed": False,
                    "error": str(e)
                }
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files: {results['total_files']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Processed with warnings: {results['warnings']}")
        print(f"Failed: {results['failed']}")
        
        # Save report if requested
        if report_path:
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {report_path}")
        
        return results


def compare_clinical_configs(audio_path: str, output_dir: str):
    """
    Compare different clinical configurations on the same audio
    """
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    configs = {
        "Dementia Assessment": ClinicalAudioConfig.get_dementia_assessment_config(),
        "Voice Disorder": ClinicalAudioConfig.get_voice_disorder_config(),
        "Speech Therapy": ClinicalAudioConfig.get_speech_therapy_config(),
        "Psychiatric Interview": ClinicalAudioConfig.get_psychiatric_interview_config()
    }
    
    results = {}
    
    # Process with each configuration
    for name, config in configs.items():
        print(f"\nProcessing with {name} configuration...")
        processor = ClinicalAudioProcessor(config)
        
        audio, sr, validation = processor.apply_clinical_preprocessing(
            audio_path,
            str(output_path / f"{Path(audio_path).stem}_{name.lower().replace(' ', '_')}.wav")
        )
        
        results[name] = {
            "audio": audio,
            "sr": sr,
            "validation": validation,
            "config": config
        }
    
    # Create comparison visualization
    fig, axes = plt.subplots(len(configs), 3, figsize=(15, 4*len(configs)))
    
    for idx, (name, result) in enumerate(results.items()):
        audio = result["audio"]
        sr = result["sr"]
        
        # Waveform
        time = np.arange(len(audio)) / sr
        axes[idx, 0].plot(time[:sr*5], audio[:sr*5], alpha=0.7)  # First 5 seconds
        axes[idx, 0].set_title(f'{name} - Waveform')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                       ax=axes[idx, 1], cmap='viridis')
        axes[idx, 1].set_title(f'{name} - Spectrogram')
        axes[idx, 1].set_ylim(0, 8000)  # Focus on speech frequencies
        
        # Configuration summary
        config_text = f"Normalization: {result['config']['normalization']['method']}\n"
        config_text += f"Noise reduction: {result['config']['noise_reduction']['aggressiveness']:.1f}\n"
        config_text += f"Bandpass: {result['config']['filtering']['lowcut']}-{result['config']['filtering']['highcut']} Hz\n"
        config_text += f"Min pause: {result['config']['silence_handling']['min_silence_duration']}s\n"
        config_text += f"\nValidation: {'PASSED' if result['validation']['passed'] else 'WARNINGS'}\n"
        
        if result['validation'].get('issues'):
            config_text += f"Issues: {', '.join(result['validation']['issues'][:2])}\n"
        
        metrics = result['validation'].get('metrics', {})
        config_text += f"\nDuration: {metrics.get('duration', 0):.2f}s\n"
        config_text += f"SNR: {metrics.get('snr_estimate', 0):.1f} dB\n"
        config_text += f"Max amp: {metrics.get('max_amplitude', 0):.3f}"
        
        axes[idx, 2].text(0.1, 0.5, config_text, transform=axes[idx, 2].transAxes,
                         fontsize=9, verticalalignment='center', fontfamily='monospace')
        axes[idx, 2].axis('off')
        axes[idx, 2].set_title(f'{name} - Settings & Metrics')
    
    plt.suptitle(f'Clinical Configuration Comparison\n{Path(audio_path).name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'clinical_config_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Comparison saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clinical audio preprocessing")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input audio file or directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file or directory")
    parser.add_argument("--config", type=str, default="dementia",
                       choices=["dementia", "voice", "therapy", "psychiatric", "custom"],
                       help="Clinical configuration preset")
    parser.add_argument("--batch", action="store_true",
                       help="Process directory of files")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all clinical configurations")
    parser.add_argument("--report", type=str,
                       help="Path to save processing report (JSON)")
    
    # Custom configuration parameters
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Sample rate for custom config")
    parser.add_argument("--noise-reduction", type=float, default=0.5,
                       help="Noise reduction aggressiveness (0-1)")
    parser.add_argument("--preserve-pauses", type=float, default=2.0,
                       help="Preserve pauses shorter than this (seconds)")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare different configurations
        compare_clinical_configs(args.input, args.output)
    else:
        # Select configuration
        if args.config == "dementia":
            config = ClinicalAudioConfig.get_dementia_assessment_config()
        elif args.config == "voice":
            config = ClinicalAudioConfig.get_voice_disorder_config()
        elif args.config == "therapy":
            config = ClinicalAudioConfig.get_speech_therapy_config()
        elif args.config == "psychiatric":
            config = ClinicalAudioConfig.get_psychiatric_interview_config()
        else:  # custom
            config = ClinicalAudioConfig.get_custom_config(
                sample_rate=args.sample_rate,
                noise_reduction_aggressiveness=args.noise_reduction,
                preserve_pauses_under=args.preserve_pauses
            )
        
        # Process audio
        processor = ClinicalAudioProcessor(config)
        
        if args.batch:
            # Batch processing
            results = processor.batch_process_clinical(
                args.input,
                args.output,
                args.report
            )
        else:
            # Single file
            audio, sr, validation = processor.apply_clinical_preprocessing(
                args.input,
                args.output
            )
            
            # Print validation report
            print("\nValidation Report:")
            print("=" * 40)
            print(f"Status: {'PASSED' if validation['passed'] else 'FAILED/WARNING'}")
            
            if validation.get('issues'):
                print(f"Issues: {', '.join(validation['issues'])}")
            
            print("\nMetrics:")
            for key, value in validation.get('metrics', {}).items():
                print(f"  {key}: {value:.3f}")