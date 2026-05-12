#!/usr/bin/env python3
"""
Test script to verify the transcription setup
"""


import sys
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    dotenv_available = True
except ImportError:
    dotenv_available = False


def test_environment_file():
    """Test .env file configuration"""
    env_path = Path(".env")
    
    if env_path.exists():
        print("✓ .env file exists")
        
        if dotenv_available:
            # Check which keys are set
            hf_token = os.getenv("HF_TOKEN")
            google_key = os.getenv("GOOGLE_API_KEY")
            
            if hf_token:
                print("  ✓ HF_TOKEN is configured")
            else:
                print("  ⚠ HF_TOKEN not set (WhisperX speaker diarization won't work)")
            
            if google_key:
                print("  ✓ GOOGLE_API_KEY is configured")
            else:
                print("  ⚠ GOOGLE_API_KEY not set (Gemini transcription won't work)")
            
            # Check directories
            input_dir = os.getenv("DEFAULT_INPUT_DIR", "./audio_files")
            output_dir = os.getenv("DEFAULT_OUTPUT_DIR", "./transcripts")
            print(f"  - Input directory: {Path(input_dir).absolute()}")
            print(f"  - Output directory: {Path(output_dir).absolute()}")
        else:
            print("  ⚠ python-dotenv not installed")
            print("    Install with: pip install python-dotenv")
    else:
        print("⚠ No .env file found")
        print("  Run: python setup_environment.py")
        return False
    
    return True


def test_whisperx():
    """Test WhisperX installation"""
    try:
        import whisperx
        import torch
        print("✓ WhisperX is installed")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
        
        # Check HF token from environment
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("✓ Hugging Face token is set")
        else:
            print("⚠ Hugging Face token not set (required for speaker diarization)")
            print("  Set it in your .env file or run: python setup_environment.py")
        
        return True
    except ImportError as e:
        print(f"✗ WhisperX not installed: {e}")
        return False


def test_gemini():
    """Test Gemini setup"""
    try:
        import google.generativeai as genai
        from pydub import AudioSegment
        print("✓ Google Gemini libraries installed")
        
        # Check API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print("✓ Google API key is set")
            # Test the API connection
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Say 'API connection successful'")
                print("✓ API connection successful")
            except Exception as e:
                print(f"⚠ API connection failed: {e}")
        else:
            print("⚠ Google API key not set")
            print("  Set it in your .env file or run: python setup_environment.py")
        
        return True
    except ImportError as e:
        print(f"✗ Gemini libraries not installed: {e}")
        return False


def test_ffmpeg():
    """Test FFmpeg installation"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg installed: {version_line}")
            return True
        else:
            print("✗ FFmpeg not working properly")
            return False
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ FFmpeg not installed")
        print("  Install with:")
        print("    Ubuntu/Debian: sudo apt install ffmpeg")
        print("    macOS: brew install ffmpeg")
        print("    Windows: Download from https://ffmpeg.org")
        return False


def create_test_structure():
    """Create test directory structure"""
    test_dir = Path("./test_transcription")
    audio_dir = test_dir / "audio"
    output_dir = test_dir / "output"
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created test directories:")
    print(f"  - Audio input: {audio_dir.absolute()}")
    print(f"  - Output: {output_dir.absolute()}")
    
    # Create a sample usage script
    sample_script = test_dir / "run_test.sh"
    with open(sample_script, 'w') as f:
        f.write("""#!/bin/bash
# Sample usage script

echo "WhisperX Example:"
python whisperx_transcription.py \\
    --input-dir ./test_transcription/audio \\
    --output-dir ./test_transcription/output \\
    --model-size base \\
    --device cpu

echo ""
echo "Gemini Example:"
python gemini_transcription.py \\
    --input-dir ./test_transcription/audio \\
    --output-dir ./test_transcription/output \\
    --model gemini-1.5-flash
""")
    
    sample_script.chmod(0o755)
    print(f"✓ Created sample script: {sample_script.absolute()}")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Medical Transcription System - Setup Test")
    print("=" * 60)
    print()
    
    # Test environment file
    print("Testing environment configuration...")
    env_ok = test_environment_file()
    print()
    
    # Test FFmpeg
    print("Testing FFmpeg installation...")
    ffmpeg_ok = test_ffmpeg()
    print()
    
    # Test WhisperX
    print("Testing WhisperX setup...")
    whisperx_ok = test_whisperx()
    print()
    
    # Test Gemini
    print("Testing Gemini setup...")
    gemini_ok = test_gemini()
    print()
    
    # Create test structure
    print("Creating test directory structure...")
    structure_ok = create_test_structure()
    print()
    
    # Summary
    print("=" * 60)
    print("Setup Summary:")
    print("=" * 60)
    
    if env_ok:
        print("✓ Environment: Configured")
    else:
        print("✗ Environment: Not configured (run setup_environment.py)")
    
    if ffmpeg_ok:
        print("✓ FFmpeg: Ready")
    else:
        print("✗ FFmpeg: Not installed (required)")
    
    if whisperx_ok:
        print("✓ WhisperX: Ready for local transcription")
    else:
        print("⚠ WhisperX: Not ready (optional)")
    
    if gemini_ok:
        print("✓ Gemini: Ready for cloud transcription")
    else:
        print("⚠ Gemini: Not ready (optional)")
    
    print()
    
    if not env_ok:
        print("⚠ First step: Run 'python setup_environment.py' to configure your environment")
        sys.exit(1)
    
    if not ffmpeg_ok:
        print("⚠ Critical: FFmpeg must be installed for audio processing")
        sys.exit(1)
    
    if not whisperx_ok and not gemini_ok:
        print("⚠ Warning: Neither WhisperX nor Gemini is fully configured")
        print("  Please set up at least one transcription method")
        sys.exit(1)
    
    print("✓ System is ready for transcription!")
    print()
    print("Next steps:")
    print("1. Place your audio files in: ./test_transcription/audio/")
    print("2. Run one of the transcription scripts:")
    
    if whisperx_ok:
        print("   python whisperx_transcription.py --input-dir ./test_transcription/audio --output-dir ./test_transcription/output")
    
    if gemini_ok:
        print("   python gemini_transcription.py --input-dir ./test_transcription/audio --output-dir ./test_transcription/output")
    
    print()
    print("For detailed instructions, see SETUP_GUIDE.md")


if __name__ == "__main__":
    main()