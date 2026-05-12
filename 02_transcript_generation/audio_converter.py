"""
This is  a simple utility script for converting audio formats
"""

#!/usr/bin/env python3
"""
Audio Format Converter Utility
Converts audio files to formats suitable for transcription
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
from typing import List


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def convert_audio(input_path: Path, output_path: Path, 
                 output_format: str = "mp3", 
                 sample_rate: int = 16000,
                 channels: int = 1) -> bool:
    """
    Convert audio file to specified format
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        output_format: Output format (mp3, wav, etc.)
        sample_rate: Sample rate in Hz (16000 recommended for speech)
        channels: Number of channels (1 for mono, 2 for stereo)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-ar", str(sample_rate),  # Sample rate
            "-ac", str(channels),      # Audio channels
            "-c:a", "libmp3lame" if output_format == "mp3" else "pcm_s16le",
            "-y",  # Overwrite output file
            str(output_path)
        ]
        
        print(f"Converting: {input_path.name} -> {output_path.name}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ Converted successfully")
            return True
        else:
            print(f"  ✗ Conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Conversion timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False


def get_audio_info(audio_path: Path) -> dict:
    """
    Get information about an audio file
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with audio information
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(audio_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        import json
        info = json.loads(result.stdout)
        
        # Extract relevant information
        audio_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break
        
        if audio_stream:
            return {
                "duration": float(info.get("format", {}).get("duration", 0)),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "codec": audio_stream.get("codec_name", "unknown"),
                "bitrate": info.get("format", {}).get("bit_rate", "unknown"),
                "size_mb": float(info.get("format", {}).get("size", 0)) / (1024 * 1024)
            }
        
        return {}
        
    except Exception as e:
        print(f"Could not get audio info: {e}")
        return {}


def batch_convert(input_dir: Path, output_dir: Path, 
                 input_extensions: List[str],
                 output_format: str = "mp3",
                 sample_rate: int = 16000) -> None:
    """
    Batch convert audio files
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        input_extensions: List of input file extensions to process
        output_format: Output format
        sample_rate: Output sample rate
    """
    # Find all audio files
    audio_files = []
    for ext in input_extensions:
        audio_files.extend(input_dir.glob(f"*.{ext}"))
        audio_files.extend(input_dir.glob(f"*.{ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found with extensions: {', '.join(input_extensions)}")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to convert")
    print(f"Output format: {output_format}, Sample rate: {sample_rate} Hz")
    print("-" * 60)
    
    success_count = 0
    total_input_size = 0
    total_output_size = 0
    
    for audio_file in audio_files:
        # Get input file info
        info = get_audio_info(audio_file)
        if info:
            total_input_size += info.get("size_mb", 0)
            duration_min = info.get("duration", 0) / 60
            print(f"\nFile: {audio_file.name}")
            print(f"  Duration: {duration_min:.1f} minutes")
            print(f"  Current: {info.get('sample_rate', 0)} Hz, {info.get('channels', 0)} channels")
        
        # Create output path
        output_path = output_dir / f"{audio_file.stem}.{output_format}"
        
        # Convert
        if convert_audio(audio_file, output_path, output_format, sample_rate, channels=1):
            success_count += 1
            
            # Get output file info
            out_info = get_audio_info(output_path)
            if out_info:
                total_output_size += out_info.get("size_mb", 0)
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print(f"  Files processed: {success_count}/{len(audio_files)}")
    print(f"  Input total size: {total_input_size:.1f} MB")
    print(f"  Output total size: {total_output_size:.1f} MB")
    print(f"  Size reduction: {(1 - total_output_size/max(total_input_size, 0.1)) * 100:.1f}%")
    print(f"  Output directory: {output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert audio files for optimal transcription"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save converted files"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="mp3",
        choices=["mp3", "wav", "flac", "ogg"],
        help="Output format (default: mp3)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[8000, 16000, 22050, 44100, 48000],
        help="Sample rate in Hz (default: 16000, recommended for speech)"
    )
    parser.add_argument(
        "--input-extensions",
        type=str,
        nargs="+",
        default=["mp3", "wav", "m4a", "flac", "ogg", "wma", "aac", "opus"],
        help="Input file extensions to process"
    )
    
    args = parser.parse_args()
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed!")
        print("Please install FFmpeg first:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org")
        sys.exit(1)
    
    # Convert to absolute paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run batch conversion
    batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        input_extensions=args.input_extensions,
        output_format=args.format,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()