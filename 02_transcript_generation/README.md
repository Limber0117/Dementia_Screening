# 🏥 Medical Conversation Transcription System

A professional Python system for transcribing medical conversations between doctors and patients with speaker identification, timestamps, and multiple output formats.

## ✨ Features

- **Speaker Diarization**: Automatically identifies Doctor (DOC) vs Patient (PAT)
- **Dual Timestamps**: Both sentence-level and word-level timestamps
- **Multiple Outputs**: Timestamped, simple, and JSON formats
- **Two Processing Options**: Local (WhisperX) or Cloud (Google Gemini)
- **Batch Processing**: Handle multiple audio files automatically
- **Secure Configuration**: Environment-based API key management
- **Format Support**: MP3, WAV, M4A, FLAC, OGG, and more

## 🚀 Quick Start

### 1. Clone and Install

```bash
# Clone the repository (or download the files)
git clone <repository-url>
cd medical-transcription

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Run the interactive setup wizard:

```bash
python setup_environment.py
```

This will:
- Ask for your API keys (securely)
- Configure default settings
- Create necessary directories
- Set up a `.gitignore` for security

### 3. Test Your Setup

```bash
python test_setup.py
```

### 4. Transcribe Your Audio

```bash
# Place audio files in ./audio_files/ (or your configured input directory)

# For local processing (WhisperX)
python whisperx_transcription.py

# For cloud processing (Gemini)
python gemini_transcription.py
```

## 📁 Project Structure

```
medical-transcription/
├── whisperx_transcription.py    # Local transcription with WhisperX
├── gemini_transcription.py       # Cloud transcription with Gemini
├── setup_environment.py          # Interactive setup wizard
├── test_setup.py                 # Setup verification tool
├── audio_converter.py            # Audio format conversion utility
├── requirements.txt              # Python dependencies
├── .env.template                 # Environment template
├── .env                          # Your configuration (created by setup)
├── SETUP_GUIDE.md               # Detailed setup instructions
├── README.md                    # This file
├── audio_files/                 # Input directory (configurable)
└── transcripts/                 # Output directory (configurable)
```

## 🔧 Configuration Options

### WhisperX (Local Processing)

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `WHISPER_MODEL` | tiny, base, small, medium, large, large-v2 | large-v2 | Model size (accuracy vs speed) |
| `WHISPER_DEVICE` | cuda, cpu | cuda | Processing device |
| `WHISPER_BATCH_SIZE` | 1-32 | 16 | Batch size for processing |
| `HF_TOKEN` | Your token | - | Required for speaker diarization |

### Gemini (Cloud Processing)

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `GEMINI_MODEL` | gemini-1.5-flash, gemini-1.5-pro | gemini-1.5-flash | Model selection |
| `GOOGLE_API_KEY` | Your API key | - | Required for Gemini |
| `API_DELAY_SECONDS` | 0-10 | 2 | Delay between API calls |

## 📝 Output Formats

### 1. Timestamped Transcript (`filename.txt`)

```
[00:00.123 - 00:05.456] DOC: Good morning. How are you feeling today?
  (Good|00:00.123-00:00.456) (morning.|00:00.567-00:01.234) 
  (How|00:01.345-00:01.678) (are|00:01.789-00:02.123) 
  (you|00:02.234-00:02.567) (feeling|00:02.678-00:03.123) 
  (today?|00:03.234-00:03.567)

[00:05.789 - 00:10.123] PAT: I've been having headaches for the past week.
```

### 2. Simple Transcript (`filename_simple.txt`)

```
DOC: Good morning. How are you feeling today?
PAT: I've been having headaches for the past week.
DOC: Can you describe the pain on a scale of 1 to 10?
```

### 3. JSON Data (`filename.json`)

Complete structured data for further processing or integration.

## 🔐 Security Notes

- **Never commit `.env` to version control**
- API keys are stored securely in environment files
- The setup wizard creates a `.gitignore` automatically
- Use `python setup_environment.py view` to safely view config
- For HIPAA compliance, use WhisperX (local processing)

## 🏥 Medical Features

The system identifies medical professionals based on:
- Medical terminology usage
- Question patterns (symptoms, history, diagnosis)
- Professional language markers
- Interaction dynamics

## ⚡ Performance Comparison

| Feature | WhisperX | Gemini |
|---------|----------|---------|
| **Speed** | ~5x realtime (GPU) | ~2x realtime |
| **Accuracy** | 95%+ | 90%+ |
| **Speaker ID** | Excellent | Good |
| **Timestamps** | Precise | Estimated |
| **Cost** | Free (local) | Pay-per-use |
| **Internet** | Not required | Required |
| **Privacy** | Complete | Data to Google |

## 🛠️ Troubleshooting

### Common Issues

**No audio files found:**
- Check file extensions in `.env`
- Ensure files are in the input directory
- Try: `python audio_converter.py` to convert formats

**GPU out of memory (WhisperX):**
- Use smaller model: `WHISPER_MODEL=small`
- Reduce batch size: `WHISPER_BATCH_SIZE=8`
- Use CPU: `WHISPER_DEVICE=cpu`

**API errors (Gemini):**
- Verify API key in `.env`
- Check quota at https://console.cloud.google.com
- Increase `API_DELAY_SECONDS`

### Getting Help

1. Run `python test_setup.py` to diagnose issues
2. Check `SETUP_GUIDE.md` for detailed instructions
3. Review error messages in terminal output
4. Verify FFmpeg installation: `ffmpeg -version`

## 📚 API Documentation

### Command Line Arguments

Both transcription scripts support command-line overrides:

```bash
# Override input/output directories
python whisperx_transcription.py \
    --input-dir /custom/input \
    --output-dir /custom/output

# Override model settings
python gemini_transcription.py \
    --model gemini-1.5-pro \
    --api-key YOUR_KEY
```

### Python Integration

```python
from whisperx_transcription import MedicalTranscriber

# Initialize transcriber
transcriber = MedicalTranscriber(
    device="cuda",
    model_size="large-v2"
)

# Process a file
transcriber.process_file(
    input_path=Path("audio.mp3"),
    output_dir=Path("output/")
)
```

## 📜 License

This is a medical transcription system. Ensure compliance with:
- HIPAA regulations for patient data
- Medical record retention requirements
- Patient consent for recording
- Local privacy laws

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows Python PEP 8 standards
- API keys are never committed
- Tests pass with `python test_setup.py`
- Documentation is updated

## 🌟 Acknowledgments

- WhisperX by @m-bain for excellent local transcription
- Google Gemini for cloud AI capabilities
- Hugging Face for speaker diarization models
- OpenAI Whisper for speech recognition foundation

---

**Note**: This system is designed for medical professionals. Always verify transcriptions for accuracy and maintain appropriate patient data security measures.