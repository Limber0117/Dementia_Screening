#!/usr/bin/env python3
"""
Environment Setup Helper for Medical Transcription System
Helps users create and configure their .env file
"""

import os
import sys
from pathlib import Path
import getpass


def setup_environment():
    """Interactive setup for environment configuration"""
    
    print("=" * 60)
    print("Medical Transcription System - Environment Setup")
    print("=" * 60)
    print()
    
    env_path = Path(".env")
    
    # Check if .env already exists
    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Setup cancelled. Using existing .env file.")
            return
    
    print("This wizard will help you set up your API keys and configuration.")
    print("Leave blank to skip optional settings.\n")
    
    config = {}
    
    # Choose which service to configure
    print("Which transcription service do you want to use?")
    print("1. WhisperX (Local processing, no API costs)")
    print("2. Gemini (Cloud processing, easier setup)")
    print("3. Both")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    # WhisperX Configuration
    if choice in ['1', '3']:
        print("\n" + "-" * 40)
        print("WhisperX Configuration (Local Processing)")
        print("-" * 40)
        
        print("\nFor speaker diarization, you need a Hugging Face token.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Make sure to accept the user agreements for:")
        print("  - https://huggingface.co/pyannote/speaker-diarization")
        print("  - https://huggingface.co/pyannote/segmentation")
        
        hf_token = getpass.getpass("\nEnter your Hugging Face token (hidden): ").strip()
        if hf_token:
            config['HF_TOKEN'] = hf_token
            print("✓ Hugging Face token configured")
        else:
            print("⚠ Skipping Hugging Face token (speaker diarization won't work)")
        
        # WhisperX model settings
        print("\nWhisperX Model Settings (press Enter for defaults):")
        
        model = input("Model size (tiny/base/small/medium/large/large-v2) [large-v2]: ").strip()
        config['WHISPER_MODEL'] = model if model else "large-v2"
        
        device = input("Device (cuda/cpu) [cuda]: ").strip()
        config['WHISPER_DEVICE'] = device if device else "cuda"
        
        batch_size = input("Batch size [16]: ").strip()
        config['WHISPER_BATCH_SIZE'] = batch_size if batch_size else "16"
    
    # Gemini Configuration
    if choice in ['2', '3']:
        print("\n" + "-" * 40)
        print("Google Gemini Configuration (Cloud Processing)")
        print("-" * 40)
        
        print("\nYou need a Google API key for Gemini.")
        print("Get your key from: https://aistudio.google.com/app/apikey")
        
        google_key = getpass.getpass("\nEnter your Google API key (hidden): ").strip()
        if google_key:
            config['GOOGLE_API_KEY'] = google_key
            print("✓ Google API key configured")
        else:
            print("⚠ Skipping Google API key (Gemini transcription won't work)")
        
        # Gemini model settings
        print("\nGemini Model Settings:")
        print("1. gemini-2.5-flash-lite (Faster, cheaper)")
        print("2. gemini-2.0-flash (Old standard)")
        print("3. gemini-3-pro-preview (More accurate)")
        
        model_choice = input("Choose model (1/2/3) [1]: ").strip()
        if model_choice == '2':
            config['GEMINI_MODEL'] = "gemini-2.0-flash"
        elif model_choice == '3':
            config['GEMINI_MODEL'] = "gemini-3-pro-preview"
        else:
            config['GEMINI_MODEL'] = "gemini-2.5-flash-lite"
        
    
    # General Configuration
    print("\n" + "-" * 40)
    print("General Configuration")
    print("-" * 40)
    
    input_dir = input("\nDefault input directory [./audio_files]: ").strip()
    config['DEFAULT_INPUT_DIR'] = input_dir if input_dir else "./audio_files"
    
    output_dir = input("Default output directory [./transcripts]: ").strip()
    config['DEFAULT_OUTPUT_DIR'] = output_dir if output_dir else "./transcripts"
    
    # Audio settings
    print("\nAudio Settings (press Enter for defaults):")
    
    sample_rate = input("Sample rate in Hz [16000]: ").strip()
    config['AUDIO_SAMPLE_RATE'] = sample_rate if sample_rate else "16000"
    
    extensions = input("Audio extensions (comma-separated) [mp3,wav,m4a,flac,ogg]: ").strip()
    config['AUDIO_EXTENSIONS'] = extensions if extensions else "mp3,wav,m4a,flac,ogg,mp4,webm"
    
    # Advanced settings
    advanced = input("\nConfigure advanced settings? (y/n) [n]: ").lower()
    if advanced == 'y':
        print("\nAdvanced Settings:")
        
        log_level = input("Log level (DEBUG/INFO/WARNING/ERROR) [INFO]: ").strip()
        config['LOG_LEVEL'] = log_level if log_level else "INFO"
        
        max_size = input("Max audio file size in MB [500]: ").strip()
        config['MAX_AUDIO_SIZE_MB'] = max_size if max_size else "500"
        
        timeout = input("Processing timeout in seconds [600]: ").strip()
        config['PROCESS_TIMEOUT'] = timeout if timeout else "600"
        
        api_delay = input("API delay between files in seconds [2]: ").strip()
        config['API_DELAY_SECONDS'] = api_delay if api_delay else "2"
        
        max_retries = input("Max API retries [3]: ").strip()
        config['MAX_RETRIES'] = max_retries if max_retries else "3"
    else:
        # Set defaults for advanced settings
        config['LOG_LEVEL'] = "INFO"
        config['MAX_AUDIO_SIZE_MB'] = "500"
        config['PROCESS_TIMEOUT'] = "600"
        config['API_DELAY_SECONDS'] = "2"
        config['MAX_RETRIES'] = "3"
    
    # Feature toggles
    config['ENABLE_WORD_TIMESTAMPS'] = "true"
    config['ENABLE_SPEAKER_DIARIZATION'] = "true"
    config['SAVE_JSON_OUTPUT'] = "true"
    config['AUDIO_CHANNELS'] = "1"
    config['WHISPER_COMPUTE_TYPE'] = "float16"
    
    # Write .env file
    print("\n" + "-" * 40)
    print("Saving Configuration")
    print("-" * 40)
    
    with open(env_path, 'w') as f:
        f.write("# Medical Transcription System Configuration\n")
        f.write("# Generated by setup_environment.py\n")
        f.write("# " + "=" * 60 + "\n\n")
        
        # Group configurations
        if 'HF_TOKEN' in config:
            f.write("# WhisperX Configuration\n")
            f.write(f"HF_TOKEN={config.get('HF_TOKEN', '')}\n")
            f.write(f"WHISPER_MODEL={config.get('WHISPER_MODEL', 'large-v2')}\n")
            f.write(f"WHISPER_DEVICE={config.get('WHISPER_DEVICE', 'cuda')}\n")
            f.write(f"WHISPER_BATCH_SIZE={config.get('WHISPER_BATCH_SIZE', '16')}\n")
            f.write(f"WHISPER_COMPUTE_TYPE={config.get('WHISPER_COMPUTE_TYPE', 'float16')}\n")
            f.write("\n")
        
        if 'GOOGLE_API_KEY' in config:
            f.write("# Gemini Configuration\n")
            f.write(f"GOOGLE_API_KEY={config.get('GOOGLE_API_KEY', '')}\n")
            f.write(f"GEMINI_MODEL={config.get('GEMINI_MODEL', 'gemini-1.5-flash')}\n")
            f.write("\n")
        
        f.write("# General Configuration\n")
        f.write(f"DEFAULT_INPUT_DIR={config['DEFAULT_INPUT_DIR']}\n")
        f.write(f"DEFAULT_OUTPUT_DIR={config['DEFAULT_OUTPUT_DIR']}\n")
        f.write(f"AUDIO_SAMPLE_RATE={config['AUDIO_SAMPLE_RATE']}\n")
        f.write(f"AUDIO_CHANNELS={config['AUDIO_CHANNELS']}\n")
        f.write(f"AUDIO_EXTENSIONS={config['AUDIO_EXTENSIONS']}\n")
        f.write("\n")
        
        f.write("# Advanced Settings\n")
        f.write(f"LOG_LEVEL={config['LOG_LEVEL']}\n")
        f.write(f"MAX_AUDIO_SIZE_MB={config['MAX_AUDIO_SIZE_MB']}\n")
        f.write(f"PROCESS_TIMEOUT={config['PROCESS_TIMEOUT']}\n")
        f.write(f"API_DELAY_SECONDS={config['API_DELAY_SECONDS']}\n")
        f.write(f"MAX_RETRIES={config['MAX_RETRIES']}\n")
        f.write("\n")
        
        f.write("# Feature Toggles\n")
        f.write(f"ENABLE_WORD_TIMESTAMPS={config['ENABLE_WORD_TIMESTAMPS']}\n")
        f.write(f"ENABLE_SPEAKER_DIARIZATION={config['ENABLE_SPEAKER_DIARIZATION']}\n")
        f.write(f"SAVE_JSON_OUTPUT={config['SAVE_JSON_OUTPUT']}\n")
    
    print("\n✓ Configuration saved to .env file")
    
    # Create directories if they don't exist
    input_path = Path(config['DEFAULT_INPUT_DIR'])
    output_path = Path(config['DEFAULT_OUTPUT_DIR'])
    
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directories:")
    print(f"  - Input: {input_path.absolute()}")
    print(f"  - Output: {output_path.absolute()}")
    
    # Create .gitignore if it doesn't exist
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write("# Environment files\n")
            f.write(".env\n")
            f.write("*.env\n")
            f.write("\n# API Keys\n")
            f.write("*_key.txt\n")
            f.write("*_token.txt\n")
            f.write("\n# Output files\n")
            f.write("transcripts/\n")
            f.write("output/\n")
            f.write("*.json\n")
            f.write("\n# Python\n")
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
            f.write(".python-version\n")
            f.write("venv/\n")
            f.write("env/\n")
        print("✓ Created .gitignore file")
    
    # Show next steps
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print(f"1. Place your audio files in: {input_path.absolute()}")
    
    if 'HF_TOKEN' in config:
        print("2. Run WhisperX transcription:")
        print("   python whisperx_transcription.py")
    
    if 'GOOGLE_API_KEY' in config:
        print("2. Run Gemini transcription:")
        print("   python gemini_transcription.py")
    
    print("\nYour transcripts will be saved in:", output_path.absolute())
    print("\nTo test your setup, run: python test_setup.py")


def view_current_config():
    """Display current environment configuration"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("No .env file found. Run setup first.")
        return
    
    print("\n" + "=" * 60)
    print("Current Configuration (.env file)")
    print("=" * 60)
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Hide sensitive data
                if '=' in line:
                    key, value = line.split('=', 1)
                    if any(sensitive in key.upper() for sensitive in ['TOKEN', 'KEY', 'SECRET']):
                        if value:
                            masked = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
                            print(f"{key}={masked}")
                        else:
                            print(f"{key}=<not set>")
                    else:
                        print(line)


def main():
    """Main function"""
    print("Medical Transcription System - Environment Setup")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "view":
        view_current_config()
    else:
        setup_environment()


if __name__ == "__main__":
    main()