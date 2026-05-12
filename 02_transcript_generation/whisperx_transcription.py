#!/usr/bin/env python3


'''
whisperx_transcription.py - Local transcription using WhisperX

Completely offline processing
High accuracy with speaker diarization (if the keywords are well defined)
Word-level timestamps
GPU acceleration support

Medical Conversation Transcription with WhisperX
Provides speaker diarization, word-level timestamps, and formatted output

!!! The performance is lower than online LLMs, however,  it is cheaper, more private, and more efficient.

'''

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables only.")
    print("Install with: pip install python-dotenv")

try:
    import whisperx
    import torch
    import pandas as pd
except ImportError as e:
    print(f"Required package not installed: {e}")
    print("Please install required packages: pip install whisperx torch")
    sys.exit(1)


class MedicalTranscriber:
    """Handles transcription of medical conversations with speaker diarization"""
    
    def __init__(self, device: str = "cuda", compute_type: str = "float16", 
                 batch_size: int = 16, model_size: str = "large-v2"):
        """
        Initialize the transcriber with WhisperX
        
        Args:
            device: "cuda" for GPU or "cpu" for CPU processing
            compute_type: "float16" for GPU, "int8" for CPU
            batch_size: Batch size for processing
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.batch_size = batch_size
        self.model_size = model_size
        
        print(f"Initializing WhisperX with device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        self.model = whisperx.load_model(
            self.model_size, 
            self.device, 
            compute_type=self.compute_type
        )
        
        # Get HF token from environment (.env file or system environment)
        self.hf_token = os.getenv("HF_TOKEN", None)
        if not self.hf_token:
            print("Warning: HF_TOKEN not set. Speaker diarization may not work.")
            print("Please set HF_TOKEN in your .env file or environment variables")
            print("Get your token from: https://huggingface.co/settings/tokens")
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file with WhisperX
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription results
        """
        print(f"Transcribing: {audio_path}")
        
        # Load and transcribe audio
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(
            audio, 
            batch_size=self.batch_size,
            language="en"
        )
        
        # Align whisper output
        print("Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=self.device
        )
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        # Perform speaker diarization with proper API handling
        if self.hf_token:
            try:
                print("Performing speaker diarization...")
                
                # Try the newer API first (for whisperx >= 3.0)
                try:
                    # Import the diarization module
                    from whisperx import DiarizationPipeline
                    diarize_model = DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                except (ImportError, AttributeError):
                    # Try alternative import paths for different whisperx versions
                    try:
                        from whisperx.diarize import DiarizationPipeline
                        diarize_model = DiarizationPipeline(
                            use_auth_token=self.hf_token,
                            device=self.device
                        )
                    except (ImportError, AttributeError):
                        # Try the older API (for whisperx < 3.0)
                        try:
                            import whisperx.diarize as diarize
                            diarize_model = diarize.DiarizationPipeline(
                                use_auth_token=self.hf_token,
                                device=self.device
                            )
                        except (ImportError, AttributeError):
                            # Last resort: try loading diarization directly
                            print("Warning: Could not import DiarizationPipeline using standard methods.")
                            print("Attempting alternative diarization approach...")
                            
                            # Alternative approach using pyannote directly
                            try:
                                from pyannote.audio import Pipeline
                                diarize_model = Pipeline.from_pretrained(
                                    "pyannote/speaker-diarization@2.1",
                                    use_auth_token=self.hf_token
                                )
                                if self.device == "cuda":
                                    diarize_model.to(torch.device("cuda"))
                                    
                                # Run diarization
                                diarization = diarize_model({"waveform": torch.from_numpy(audio).unsqueeze(0), 
                                                            "sample_rate": 16000})
                                
                                # Convert pyannote output to whisperx format
                                diarize_segments = []
                                for turn, _, speaker in diarization.itertracks(yield_label=True):
                                    diarize_segments.append({
                                        "start": turn.start,
                                        "end": turn.end,
                                        "speaker": speaker
                                    })
                                
                                # Assign speakers to words
                                result = self.assign_word_speakers_fallback(diarize_segments, result)
                                return result
                                
                            except Exception as e:
                                print(f"Could not load diarization model: {e}")
                                print("Proceeding without speaker diarization.")
                                return result
                
                # Run diarization with the loaded model
                diarize_segments = diarize_model(audio)
                
                # Assign speakers to words
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
            except Exception as e:
                print(f"Warning: Speaker diarization failed: {e}")
                print("Continuing without speaker labels...")
        
        return result
    
    def assign_word_speakers_fallback(self, diarize_segments: List[Dict], result: Dict) -> Dict:
        """
        Fallback method to assign speakers to words when whisperx.assign_word_speakers is not available
        
        Args:
            diarize_segments: List of diarization segments
            result: Transcription result
            
        Returns:
            Result with assigned speakers
        """
        segments = result.get("segments", [])
        
        for segment in segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            segment_mid = (segment_start + segment_end) / 2
            
            # Find the speaker for this segment
            assigned_speaker = None
            for diar_seg in diarize_segments:
                if diar_seg["start"] <= segment_mid <= diar_seg["end"]:
                    assigned_speaker = diar_seg["speaker"]
                    break
            
            if assigned_speaker:
                segment["speaker"] = assigned_speaker
                
                # Assign speaker to words if they exist
                if "words" in segment:
                    for word in segment["words"]:
                        word["speaker"] = assigned_speaker
        
        return result
    
    def identify_speakers(self, segments: List[Dict]) -> Dict[str, str]:
        """
        Identify which speaker is doctor and which is patient
        Based on medical keywords and question patterns
        
        Args:
            segments: List of transcription segments with speaker labels
            
        Returns:
            Mapping of speaker IDs to roles (DOC/PAT)
        """
        # Medical question keywords commonly used by doctors
        # !!!! The following keywords can be adjusted based on context, used to use third-party libraries proposed by other papers to cover all possible terms.
        doctor_keywords = [
            "symptoms", "pain", "medication", "allergy", "allergies",
            "medical history", "how long", "when did", "describe",
            "scale of", "any other", "taking any", "feel", "experiencing",
            "diagnosis", "prescribe", "recommend", "treatment"
        ]
        
        speaker_scores = {}
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").lower()
            
            if speaker not in speaker_scores:
                speaker_scores[speaker] = {"doctor_score": 0, "word_count": 0}
            
            # Count doctor keywords
            for keyword in doctor_keywords:
                if keyword in text:
                    speaker_scores[speaker]["doctor_score"] += 1
            
            # Count question marks (doctors tend to ask more questions)
            speaker_scores[speaker]["doctor_score"] += text.count("?") * 2
            speaker_scores[speaker]["word_count"] += len(text.split())
        
        # Identify doctor as speaker with highest doctor score
        if len(speaker_scores) == 0:
            return {}
        elif len(speaker_scores) == 1:
            # Only one speaker detected, assume it's the doctor
            speaker = list(speaker_scores.keys())[0]
            return {speaker: "DOC"}
        else:
            # Multiple speakers, identify based on scores
            doctor_speaker = max(
                speaker_scores.keys(), 
                key=lambda x: speaker_scores[x]["doctor_score"]
            )
            
            speaker_mapping = {}
            for speaker in speaker_scores:
                if speaker == doctor_speaker:
                    speaker_mapping[speaker] = "DOC"
                else:
                    speaker_mapping[speaker] = "PAT"
            
            return speaker_mapping
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS.mmm format"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    def generate_transcripts(self, result: Dict, audio_filename: str) -> Tuple[str, str]:
        """
        Generate both timestamped and non-timestamped transcripts
        
        Args:
            result: WhisperX transcription result
            audio_filename: Name of the audio file
            
        Returns:
            Tuple of (timestamped_transcript, simple_transcript)
        """
        segments = result.get("segments", [])
        
        # Identify speakers
        speaker_mapping = self.identify_speakers(segments)
        
        timestamped_lines = []
        simple_lines = []
        
        timestamped_lines.append(f"Transcript for: {audio_filename}")
        timestamped_lines.append("=" * 60)
        timestamped_lines.append("Format: [START_TIME - END_TIME] SPEAKER: text")
        timestamped_lines.append("Word timestamps: (word|start-end)")
        timestamped_lines.append("=" * 60)
        timestamped_lines.append("")
        
        simple_lines.append(f"Transcript for: {audio_filename}")
        simple_lines.append("=" * 60)
        simple_lines.append("")
        
        for segment in segments:
            speaker_id = segment.get("speaker", "UNKNOWN")
            speaker_label = speaker_mapping.get(speaker_id, "UNKNOWN")
            
            # Get sentence timestamps
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            # Format sentence with timestamp
            sentence_header = f"[{self.format_timestamp(start_time)} - {self.format_timestamp(end_time)}] {speaker_label}: {text}"
            timestamped_lines.append(sentence_header)
            
            # Add word-level timestamps if available
            words = segment.get("words", [])
            if words:
                word_timestamps = []
                for word_data in words:
                    word = word_data.get("word", "*")
                    word_start = word_data.get("start", 0)
                    word_end = word_data.get("end", 0)
                    
                    # Handle unrecognized words
                    if word.strip() == "":
                        word = "*"
                    
                    word_timestamp = f"({word}|{self.format_timestamp(word_start)}-{self.format_timestamp(word_end)})"
                    word_timestamps.append(word_timestamp)
                
                # Add word timestamps in chunks for readability
                chunk_size = 5
                for i in range(0, len(word_timestamps), chunk_size):
                    chunk = word_timestamps[i:i+chunk_size]
                    timestamped_lines.append("  " + " ".join(chunk))
            
            timestamped_lines.append("")  # Empty line between segments
            
            # Simple transcript
            simple_lines.append(f"{speaker_label}: {text}")
        
        timestamped_transcript = "\n".join(timestamped_lines)
        simple_transcript = "\n".join(simple_lines)
        
        return timestamped_transcript, simple_transcript
    
    def process_file(self, input_path: Path, output_dir: Path) -> bool:
        """
        Process a single audio file
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save transcripts
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Transcribe audio
            result = self.transcribe_audio(str(input_path))
            
            # Generate transcripts
            timestamped, simple = self.generate_transcripts(result, input_path.name)
            
            # Save transcripts
            base_name = input_path.stem
            
            # Save timestamped transcript
            timestamped_path = output_dir / f"{base_name}.txt"
            with open(timestamped_path, 'w', encoding='utf-8') as f:
                f.write(timestamped)
            
            # Save simple transcript
            simple_path = output_dir / f"{base_name}_simple.txt"
            with open(simple_path, 'w', encoding='utf-8') as f:
                f.write(simple)
            
            # Save raw JSON for debugging/further processing
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Processed: {input_path.name}")
            print(f"  - Timestamped transcript: {timestamped_path}")
            print(f"  - Simple transcript: {simple_path}")
            print(f"  - Raw data: {json_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing {input_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def check_whisperx_version():
    """Check WhisperX version and available features"""
    try:
        import whisperx
        print(f"WhisperX version: {whisperx.__version__ if hasattr(whisperx, '__version__') else 'Unknown'}")
        
        # Check for diarization support
        has_diarization = False
        diarization_method = None
        
        if hasattr(whisperx, 'DiarizationPipeline'):
            has_diarization = True
            diarization_method = "whisperx.DiarizationPipeline"
        else:
            try:
                from whisperx.diarize import DiarizationPipeline
                has_diarization = True
                diarization_method = "whisperx.diarize.DiarizationPipeline"
            except ImportError:
                try:
                    import whisperx.diarize as diarize
                    if hasattr(diarize, 'DiarizationPipeline'):
                        has_diarization = True
                        diarization_method = "whisperx.diarize.DiarizationPipeline"
                except ImportError:
                    pass
        
        if has_diarization:
            print(f"✓ Diarization support found: {diarization_method}")
        else:
            print("✗ Diarization support not found - will try pyannote.audio directly")
            
    except Exception as e:
        print(f"Error checking WhisperX version: {e}")


def main():
    """Main function to execute the transcription program"""
    parser = argparse.ArgumentParser(
        description="Transcribe medical conversations with speaker diarization"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.getenv("DEFAULT_INPUT_DIR", None),
        required=False,
        help="Directory containing audio files (default: from .env or current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("DEFAULT_OUTPUT_DIR", None),
        required=False,
        help="Directory to save transcripts (default: from .env or ./output)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=os.getenv("WHISPER_MODEL", "large-v2"),
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: from .env or large-v2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("WHISPER_DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Device to use for processing (default: from .env or cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("WHISPER_BATCH_SIZE", "16")),
        help="Batch size for processing (default: from .env or 16)"
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        default=os.getenv("AUDIO_EXTENSIONS", "mp3,wav,m4a,flac,ogg,mp4,webm").split(","),
        help="Audio file extensions to process"
    )
    parser.add_argument(
        "--check-version",
        action="store_true",
        help="Check WhisperX version and available features"
    )
    
    args = parser.parse_args()
    
    # Check version if requested
    if args.check_version:
        check_whisperx_version()
        sys.exit(0)
    
    # Check required arguments
    if not args.input_dir:
        print("Error: --input-dir is required")
        print("You can also set DEFAULT_INPUT_DIR in your .env file")
        sys.exit(1)
    
    if not args.output_dir:
        args.output_dir = "./output"
        print(f"No output directory specified, using: {args.output_dir}")
    
    # Convert to absolute paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check WhisperX configuration
    print("\n" + "=" * 60)
    check_whisperx_version()
    print("=" * 60 + "\n")
    
    # Find audio files
    audio_files = []
    for ext in args.file_extensions:
        audio_files.extend(input_dir.glob(f"*.{ext}"))
        #audio_files.extend(input_dir.glob(f"*.{ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        print(f"Supported extensions: {', '.join(args.file_extensions)}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio file(s) to process")
    
    # Initialize transcriber
    transcriber = MedicalTranscriber(
        device=args.device,
        model_size=args.model_size,
        batch_size=args.batch_size
    )
    
    # Process each file
    success_count = 0
    for audio_file in audio_files:
        if transcriber.process_file(audio_file, output_dir):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(audio_files)} files")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()