# Medical Conversation Transcription System
## Setup and Usage Guide

This package provides two solutions for transcribing medical conversations between doctors and patients with speaker diarization and timestamps.

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg (for audio processing)
- 8GB+ RAM (16GB recommended for WhisperX with large models)
- NVIDIA GPU with CUDA support (optional but recommended for WhisperX)

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from: https://ffmpeg.org/download.html

---

## Quick Setup (Recommended)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
Run the interactive setup wizard:
```bash
python setup_environment.py
```

This will:
- Create a `.env` file with your API keys
- Set up default directories
- Configure model settings
- Create a `.gitignore` to protect your keys

### Step 3: Test Your Setup
```bash
python test_setup.py
```

### Step 4: Run Transcription
```bash
# For WhisperX (local processing)
python whisperx_transcription.py

# For Gemini (cloud processing)
python gemini_transcription.py
```

---

## Manual Setup (Alternative)

### Step 1: Create Environment File
Copy the template and edit with your keys:
```bash
cp .env.template .env
nano .env  # or use any text editor
```

### Step 2: Add Your API Keys

Edit `.env` and add your keys:

```env
# For WhisperX (get from https://huggingface.co/settings/tokens)
HF_TOKEN=your_huggingface_token_here

# For Gemini (get from https://aistudio.google.com/app/apikey)
GOOGLE_API_KEY=your_google_api_key_here

# Set your default directories
DEFAULT_INPUT_DIR=./audio_files
DEFAULT_OUTPUT_DIR=./transcripts
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Option 1: WhisperX Solution (Local Processing)

### Advantages
- Completely offline processing
- No API costs
- Better privacy (data stays local)
- More accurate timestamps
- Better speaker diarization

### Setup Instructions

1. **Configure environment (if not done already):**
```bash
python setup_environment.py
# Choose option 1 or 3 for WhisperX
```

2. **For GPU support (recommended):**
```bash
# Check your CUDA version
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Accept Hugging Face model agreements:**
   - Go to https://huggingface.co/pyannote/speaker-diarization
   - Click "Agree and access repository"
   - Go to https://huggingface.co/pyannote/segmentation
   - Click "Agree and access repository"

### Usage

**Basic usage (uses settings from .env):**
```bash
python whisperx_transcription.py
```

**Override .env settings:**
```bash
python whisperx_transcription.py \
    --input-dir /custom/path/to/audio \
    --output-dir /custom/output/path \
    --model-size large-v2 \
    --device cuda
```

**Available model sizes:**
- `tiny`: Fastest, least accurate (39M parameters)
- `base`: Fast, good for English (74M)
- `small`: Balanced (244M)
- `medium`: Good accuracy (769M)
- `large`: High accuracy (1550M)
- `large-v2`: Best accuracy (1550M, recommended)

---

## Option 2: Google Gemini Solution (Online Processing)

### Advantages
- No local GPU required
- Easier setup
- Good for smaller datasets
- Latest AI models

### Setup Instructions

1. **Configure environment (if not done already):**
```bash
python setup_environment.py
# Choose option 2 or 3 for Gemini
```

2. **That's it!** The setup wizard handles everything.

### Usage

**Basic usage (uses settings from .env):**
```bash
python gemini_transcription.py
```

**Override .env settings:**
```bash
python gemini_transcription.py \
    --input-dir /custom/path/to/audio \
    --output-dir /custom/output/path \
    --model gemini-1.5-pro
```

**Available models:**
- `gemini-1.5-flash`: Fast and cost-effective
- `gemini-1.5-pro`: More accurate but slower

### API Pricing (as of 2024)
- Gemini 1.5 Flash: ~$0.35 per 1M tokens
- Gemini 1.5 Pro: ~$3.50 per 1M tokens
- Audio processing typically uses ~1000 tokens per minute of audio

---

## Output Format

Both solutions generate three files for each audio:

### 1. Timestamped Transcript (filename.txt)
```
Transcript for: conversation1.mp3
============================================================
Format: [START_TIME - END_TIME] SPEAKER: text
Word timestamps: (word|start-end)
============================================================

[00:00.123 - 00:05.456] DOC: Good morning. How are you feeling today?
  (Good|00:00.123-00:00.456) (morning.|00:00.567-00:01.234) 
  (How|00:01.345-00:01.678) (are|00:01.789-00:02.123) 
  (you|00:02.234-00:02.567) (feeling|00:02.678-00:03.123) 
  (today?|00:03.234-00:03.567)

[00:05.789 - 00:10.123] PAT: I've been having headaches for the past week.
  (I've|00:05.789-00:06.123) (been|00:06.234-00:06.567)
  ...
```

### 2. Simple Transcript (filename_simple.txt)
```
Transcript for: conversation1.mp3
============================================================

DOC: Good morning. How are you feeling today?
PAT: I've been having headaches for the past week.
DOC: Can you describe the pain on a scale of 1 to 10?
PAT: It's about a 7, sometimes worse in the morning.
```

### 3. Raw JSON Data (filename.json)
Contains all detailed information for further processing.

---

## Usage Examples

### Process all MP3 files in a folder:
```bash
# WhisperX
python whisperx_transcription.py \
    --input-dir ~/medical_recordings \
    --output-dir ~/transcripts \
    --file-extensions mp3

# Gemini
python gemini_transcription.py \
    --input-dir ~/medical_recordings \
    --output-dir ~/transcripts \
    --file-extensions mp3
```

### Process multiple audio formats:
```bash
python whisperx_transcription.py \
    --input-dir ./audio \
    --output-dir ./output \
    --file-extensions mp3 wav m4a
```

---

## Troubleshooting

### WhisperX Issues

**CUDA out of memory:**
- Use a smaller model: `--model-size small`
- Reduce batch size: `--batch-size 8`
- Use CPU instead: `--device cpu`

**Speaker diarization not working:**
- Ensure HF_TOKEN is set correctly
- Check you've accepted the model agreements on Hugging Face

**Installation errors:**
```bash
# Clean install
pip uninstall whisperx torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install whisperx
```

### Gemini Issues

**Rate limiting:**
- The script adds 2-second delays between files
- For large batches, consider upgrading to Gemini Pro

**API errors:**
- Verify your API key is valid
- Check your quota at https://console.cloud.google.com

**Audio format not supported:**
```bash
# Convert to supported format using FFmpeg
ffmpeg -i input.opus -acodec mp3 output.mp3
```

---

## Performance Comparison

| Feature | WhisperX | Gemini |
|---------|----------|---------|
| Speed | ~5x realtime (GPU) | ~2x realtime |
| Accuracy | 95%+ | 90%+ |
| Speaker Diarization | Excellent | Good |
| Word Timestamps | Precise | Estimated |
| Cost | Free (local) | Pay-per-use |
| Internet Required | No | Yes |
| GPU Required | Recommended | No |
| Privacy | Full (local) | Data sent to Google |

---

## Advanced Configuration

### Custom Speaker Labels
Edit the `identify_speakers()` method in either script to customize speaker identification logic.

### Batch Processing with Parallel Jobs
```bash
# Split files into batches and process in parallel
find ./audio -name "*.mp3" | xargs -P 4 -I {} \
    python whisperx_transcription.py \
    --input-dir {} --output-dir ./output
```

### Integration with Other Systems
The JSON output can be easily integrated with:
- Electronic Health Record (EHR) systems
- Natural Language Processing pipelines
- Database storage systems

---

## Support and Contributions

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Consult the official documentation:
   - WhisperX: https://github.com/m-bain/whisperX
   - Gemini: https://ai.google.dev/

---

## Security Best Practices

### Protecting Your API Keys

1. **Never commit `.env` to version control:**
   - The setup wizard automatically creates a `.gitignore` file
   - Always verify `.env` is listed in `.gitignore`

2. **Use environment files properly:**
   ```bash
   # Good - keys in .env file
   echo "GOOGLE_API_KEY=your_key" >> .env
   
   # Bad - keys in code
   api_key = "your_key_here"  # Never do this!
   ```

3. **View your configuration safely:**
   ```bash
   # This command masks sensitive data
   python setup_environment.py view
   ```

4. **Rotate keys regularly:**
   - WhisperX: Regenerate HF token monthly
   - Gemini: Rotate API keys quarterly
   - Update `.env` file after rotation

5. **Set file permissions (Linux/Mac):**
   ```bash
   chmod 600 .env  # Only owner can read/write
   ```

---

## License and Privacy

- Ensure compliance with HIPAA and medical data privacy regulations
- WhisperX solution keeps all data local
- Gemini solution sends data to Google's servers
- Always obtain proper consent for recording and transcribing medical conversations

---

## Quick Start Guide

### Fastest Setup (Gemini):
```bash
# 1. Install dependencies
pip install google-generativeai pydub

# 2. Get API key from https://aistudio.google.com/app/apikey

# 3. Run transcription
python gemini_transcription.py \
    --input-dir ./audio \
    --output-dir ./transcripts \
    --api-key "YOUR_KEY"
```

### Most Accurate Setup (WhisperX with GPU):
```bash
# 1. Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install WhisperX
pip install whisperx

# 3. Get HF token from https://huggingface.co/settings/tokens
export HF_TOKEN="YOUR_TOKEN"

# 4. Run transcription
python whisperx_transcription.py \
    --input-dir ./audio \
    --output-dir ./transcripts \
    --model-size large-v2 \
    --device cuda
```

---

Happy transcribing! 