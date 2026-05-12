# Clinical Audio Processing Guide

## Quick Start for Clinical Audio

### 1. Choose Your Clinical Scenario

The script provides **4 pre-configured settings** optimized for different clinical use cases:

| Configuration | Use For | Key Features |
|--------------|---------|--------------|
| **dementia** | Cognitive assessment, AD/MCI detection | Preserves pauses, hesitations, minimal noise reduction |
| **voice** | Voice disorder assessment | Minimal processing, preserves breathiness/hoarseness |
| **therapy** | Speech therapy sessions | Balanced clarity and preservation |
| **psychiatric** | Mental health interviews | Preserves emotional prosody, speaking rate changes |

### 2. Basic Usage Examples

#### For Dementia/Cognitive Assessment:
```bash
# Single file
python clinical_audio_config.py --input patient_001.mp3 --output patient_001_processed.wav --config dementia

# Batch process all files in a directory
python clinical_audio_config.py --input ./raw_audios/ --output ./processed_audios/ --config dementia --batch

# Generate processing report
python clinical_audio_config.py --input ./raw_audios/ --output ./processed_audios/ --config dementia --batch --report processing_report.json
```

#### For Voice Disorder Analysis:
```bash
python clinical_audio_config.py --input voice_sample.wav --output voice_processed.wav --config voice
```

#### For Psychiatric Interviews:
```bash
python clinical_audio_config.py --input interview.mp3 --output interview_processed.wav --config psychiatric
```

#### Compare Different Settings:
```bash
# See how different clinical settings affect the same audio
python clinical_audio_config.py --input sample.wav --output ./comparison/ --compare
```

### 3. Custom Settings for Specific Needs

If the presets don't match your exact needs, you can customize:

```bash
python clinical_audio_config.py \
    --input audio.wav \
    --output processed.wav \
    --config custom \
    --sample-rate 16000 \
    --noise-reduction 0.3 \
    --preserve-pauses 1.5
```

Parameters explained:
- `--noise-reduction`: 0-1 scale (0 = no reduction, 1 = aggressive)
- `--preserve-pauses`: Keep pauses shorter than X seconds
- `--sample-rate`: Target sample rate (16000 for speech, 44100 for quality)

## Settings Comparison for Clinical Scenarios

### Dementia Assessment Settings
```python
# What's preserved:
✓ Pauses and hesitations (critical markers)
✓ Speech rate variations
✓ Voice quality changes
✓ Natural prosody

# Settings used:
- Normalization: RMS (0.05 target)
- Noise reduction: Low (0.5 aggressiveness)
- Bandpass: 70-8000 Hz
- Preserves pauses < 2 seconds
- Removes only silences > 2 seconds
```

### Voice Disorder Settings
```python
# What's preserved:
✓ Breathiness and hoarseness
✓ Pitch irregularities
✓ Voice breaks
✓ All harmonics

# Settings used:
- Normalization: Peak (minimal change)
- Noise reduction: Very low (0.3)
- Bandpass: 50-12000 Hz (wider range)
- Sample rate: 44100 Hz (higher quality)
- Keeps all audio (no silence removal)
```

### Psychiatric Interview Settings
```python
# What's preserved:
✓ Emotional prosody
✓ Speaking rate changes
✓ Long pauses (meaningful)
✓ Voice stress indicators

# Settings used:
- Normalization: RMS (0.07 target)
- Noise reduction: Low (0.4)
- Bandpass: 60-10000 Hz
- Preserves pauses < 3 seconds
- Wider frequency range for emotions
```

## Practical Workflow for Clinical Studies

### Step 1: Test on Sample Files
```bash
# Test on one file first
python clinical_audio_config.py --input sample.wav --output test.wav --config dementia

# Listen to the output and check if speech characteristics are preserved
```

### Step 2: Compare Configurations
```bash
# Compare all presets to find the best one
python clinical_audio_config.py --input sample.wav --output ./test_configs/ --compare
```

### Step 3: Process Your Dataset
```bash
# Process all files with chosen configuration
python clinical_audio_config.py \
    --input ./participant_audios/ \
    --output ./processed_audios/ \
    --config dementia \
    --batch \
    --report processing_report.json
```

### Step 4: Extract Features
```bash
# Use the acoustic feature extraction script on processed files
python acoustic_feature_extraction.py \
    --input ./processed_audios/ \
    --output features.csv \
    --batch
```

## Validation and Quality Control

The script automatically validates each audio file and reports:

1. **Duration**: Is the audio long enough?
2. **SNR (Signal-to-Noise Ratio)**: Is the audio quality sufficient?
3. **Clipping**: Is the audio distorted?
4. **Dynamic Range**: Is there enough variation?

Example validation output:
```
Processing: patient_001.wav
✓ Processed successfully
  Duration: 45.3s
  SNR: 18.5 dB
  Max amplitude: 0.912
```

## Important Clinical Considerations

### DO ✓
- **Test settings on representative samples** before batch processing
- **Listen to processed files** to ensure clinical markers are preserved
- **Keep original files** - never overwrite them
- **Document your settings** in research papers
- **Use consistent settings** across all participants
- **Check validation reports** for quality issues

### DON'T ✗
- **Don't over-process** - clinical markers can be subtle
- **Don't use aggressive noise reduction** - it removes important information
- **Don't remove all pauses** - they're diagnostically important
- **Don't use different settings** for different participants in same study
- **Don't ignore validation warnings** - they indicate potential issues

## Adjusting Settings for Your Specific Needs

### If Speech is Too Quiet After Processing:
Increase the normalization target:
```python
# In the config:
"target_rms": 0.1,  # Increase from 0.05
```

### If Too Much Background Noise Remains:
Slightly increase noise reduction:
```python
# In the config:
"aggressiveness": 0.6,  # Increase from 0.5 (but stay < 0.7 for clinical)
```

### If Important Pauses Are Being Removed:
Increase the pause preservation threshold:
```python
# In the config:
"min_silence_duration": 3.0,  # Increase from 2.0 seconds
"preserve_short_pauses": True,  # Always keep this True
```

### If Voice Quality Features Are Lost:
Reduce filtering and noise reduction:
```python
# In the config:
"lowcut": 50,  # Lower from 70 Hz
"highcut": 10000,  # Higher from 8000 Hz
"aggressiveness": 0.3,  # Reduce from 0.5
```

## Example: Complete Pipeline for Dementia Study

```bash
# 1. Create directories
mkdir raw_audios processed_audios features reports

# 2. Copy your audio files to raw_audios/

# 3. Test on one file
python clinical_audio_config.py \
    --input raw_audios/sample.wav \
    --output test_output.wav \
    --config dementia

# 4. Listen and verify quality
# If good, proceed. If not, adjust settings

# 5. Process all files
python clinical_audio_config.py \
    --input raw_audios/ \
    --output processed_audios/ \
    --config dementia \
    --batch \
    --report reports/preprocessing.json

# 6. Extract acoustic features
python acoustic_feature_extraction.py \
    --input processed_audios/ \
    --output features/acoustic_features.csv \
    --batch

# 7. Check the report for any issues
cat reports/preprocessing.json
```

## Interpreting the Processing Report

The JSON report contains:
```json
{
  "total_files": 50,
  "processed": 48,
  "warnings": 2,
  "failed": 0,
  "file_reports": {
    "patient_001.wav": {
      "passed": true,
      "metrics": {
        "duration": 45.3,
        "snr_estimate": 18.5,
        "max_amplitude": 0.912
      }
    }
  }
}
```

- **processed**: Files successfully processed
- **warnings**: Files processed but with quality concerns
- **failed**: Files that couldn't be processed
- **metrics**: Quality measurements for each file

## Getting Help

If you encounter issues:

1. Check the validation report for specific problems
2. Try the `--compare` option to see different configurations
3. Start with less aggressive settings and increase if needed
4. Test on a sample before batch processing


# Audio Preprocessing Best Practices for Acoustic Feature Analysis

## Quick Answers to Your Questions

### 1. Is Normalization Necessary?
**YES, absolutely!** Especially when processing audio from different participants with varying:
- Recording devices (different microphones, smartphones, etc.)
- Recording environments (quiet room vs. noisy environment)
- Speaking volumes (loud vs. soft speakers)
- Distance from microphone

### 2. Which Normalization Method to Use?

For your multi-participant acoustic analysis, I recommend:

1. **LUFS Normalization (BEST for your case)**
   - Industry standard (EBU R128)
   - Accounts for human perception of loudness
   - Consistent across different types of content
   - Preserves natural dynamics

2. **RMS Normalization (Good alternative)**
   - Normalizes average energy
   - Simple and effective
   - Good for speech analysis
   - Preserves relative amplitude variations

3. **Energy Normalization (For varying speaking volumes)**
   - Adapts to local energy changes
   - Good for speakers who vary their volume
   - Helps with feature consistency

**Avoid Peak Normalization** for acoustic features - it only looks at the loudest point and doesn't account for overall loudness.

## Recommended Preprocessing Pipeline

For your dementia/clinical audio analysis, use this order:

```python
1. Load audio (standardize sampling rate to 16kHz)
2. Remove long silences (keep short pauses - they're important features!)
3. Apply bandpass filter (80-8000 Hz for speech)
4. Reduce noise (adaptive filtering recommended)
5. Apply LUFS or RMS normalization
6. Extract acoustic features
```

## Noise Reduction Strategies

### Types of Noise and Solutions:

#### 1. **Stationary Noise** (fan, AC, computer hum)
- **Method**: Spectral Subtraction
- **How**: Estimates noise spectrum from quiet parts and subtracts it
- **Effectiveness**: Very good for constant background noise

#### 2. **Non-stationary Noise** (other people talking, traffic)
- **Method**: Adaptive Filtering
- **How**: Continuously adapts to changing noise conditions
- **Effectiveness**: Good, but may slightly affect speech quality

#### 3. **Microphone/Recording Artifacts**
- **Method**: Bandpass Filtering + De-clicking
- **How**: Removes frequencies outside speech range
- **Effectiveness**: Excellent for removing rumble and hiss

#### 4. **Background Music**
- **Method**: Source Separation (advanced)
- **Tools**: Spleeter, Demucs
- **Note**: More complex, may affect speech quality

