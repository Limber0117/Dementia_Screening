import os
import re
from pathlib import Path
from pydub import AudioSegment
from typing import List, Tuple
from dotenv import load_dotenv
from tqdm import tqdm



# Load environment variables
load_dotenv()

# Load settings from .env
DEFAULT_INPUT_AUDIO_DIR = os.getenv("DEFAULT_INPUT_AUDIO_DIR", "datasets/input")
DEFAULT_INPUT_TRANS_DIR = os.getenv("DEFAULT_INPUT_TRANS_DIR", "datasets/output/transcripts")
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR", "datasets/output/patient_audio")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_EXTENSIONS = [ext.strip() for ext in os.getenv("AUDIO_EXTENSIONS", "mp3,wav").split(",")]


def parse_time_to_ms(timestr: str) -> int:
    parts = re.match(r"(\d+):(\d+)\.(\d+)", timestr)
    if not parts:
        raise ValueError(f"Invalid timestamp format: {timestr}")
    minutes, seconds, millis = map(int, parts.groups())
    return (minutes * 60 + seconds) * 1000 + millis


import re
from pathlib import Path

def parse_transcript(transcript_path: Path) -> list[tuple[int, int]]:
    """
    Parse transcript to extract patient audio segments.
    
    Patient segments start from the end of the previous doctor's speech
    and end when the doctor speaks again.
    """
    pattern = r'\[(\d{2}):(\d{2})\.(\d{3})\s*-\s*(\d{2}):(\d{2})\.(\d{3})\]\s*(DOC|PAT):'
    
    # First, collect all segments with their speaker and timestamps
    all_segments = []
    
    with open(transcript_path, 'r') as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                start_min, start_sec, start_ms = int(match.group(1)), int(match.group(2)), int(match.group(3))
                end_min, end_sec, end_ms = int(match.group(4)), int(match.group(5)), int(match.group(6))
                speaker = match.group(7)
                
                start_time_ms = (start_min * 60 + start_sec) * 1000 + start_ms
                end_time_ms = (end_min * 60 + end_sec) * 1000 + end_ms
                
                all_segments.append({
                    'speaker': speaker,
                    'start': start_time_ms,
                    'end': end_time_ms
                })
    
    if not all_segments:
        return []
    
    # Now extract patient segments with adjusted timing
    patient_segments = []
    current_patient_start = None
    current_patient_end = None
    
    for i, segment in enumerate(all_segments):
        if segment['speaker'] == 'PAT':
            if current_patient_start is None:
                # Starting a new patient segment
                # Look back to find the end of the previous doctor's speech
                if i > 0 and all_segments[i - 1]['speaker'] == 'DOC':
                    current_patient_start = all_segments[i - 1]['end']
                else:
                    # No previous doctor speech, use patient's own start time
                    current_patient_start = segment['start']
            
            # Update the end time to this patient segment's end
            current_patient_end = segment['end']
        
        elif segment['speaker'] == 'DOC':
            # Doctor is speaking - close any open patient segment
            if current_patient_start is not None and current_patient_end is not None:
                patient_segments.append((current_patient_start, current_patient_end))
                current_patient_start = None
                current_patient_end = None
    
    # Don't forget to close the last patient segment if the transcript ends with patient
    if current_patient_start is not None and current_patient_end is not None:
        patient_segments.append((current_patient_start, current_patient_end))
    print(f"Extracted {len(patient_segments)} patient segments from transcript.")
    for seg in patient_segments:
        start_str = format_time(seg[0])
        end_str = format_time(seg[1])
        print(f" |--> Segment: [{start_str}] - [{end_str}]")
    return patient_segments


def parse_patient_transcript(transcript_path: Path) -> List[Tuple[int, int]]:
    """
    Parse transcript file and extract only patient's (PAT) segments with timestamps. The pauses between
    sentences are excluded from the patient's segments. 
    Good for transcripts where PAT and DOC segments are clearly marked.
    
    Handles:
    - Empty lines and whitespace-only lines
    - Invalid timestamp formats
    - Malformed lines
    - Invalid segment durations
    """
    segments = []
    total_lines = 0
    processed_lines = 0
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
            
            # Look for patient segments - allow flexible spacing around the dash
            match = re.match(r"\[(\d{2}:\d{2}\.\d{3})\s*-\s*(\d{2}:\d{2}\.\d{3})\]\s+PAT:", line)
            if match:
                try:
                    start_ms = parse_time_to_ms(match.group(1))
                    end_ms = parse_time_to_ms(match.group(2))
                    
                    # Validate segment duration
                    if start_ms >= end_ms:
                        print(f"Warning: Invalid segment duration at line {line_num}: start={start_ms}ms, end={end_ms}ms")
                        continue
                    
                    # Check for reasonable segment duration (optional - remove if not needed)
                    duration_ms = end_ms - start_ms
                    if duration_ms > 300000:  # 5 minutes - might be too long
                        print(f"Warning: Very long segment at line {line_num}: {duration_ms/1000:.1f} seconds")
                    
                    segments.append((start_ms, end_ms))
                    processed_lines += 1
                    
                except ValueError as e:
                    print(f"Warning: Error parsing timestamp at line {line_num}: {e}")
                    print(f"  Line content: {line[:100]}...")  # Show first 100 chars for debugging
                    continue
            else:
                # Check if line might be a malformed PAT line (for debugging)
                if "PAT:" in line and "[" in line:
                    print(f"Warning: Possible malformed PAT line at {line_num}: {line[:100]}...")
    
    print(f"Transcript parsing summary: {processed_lines} PAT segments found from {total_lines} total lines")
    return segments

def extract_patient_audio(audio_path: Path, transcript_path: Path, output_path: Path) -> bool:
    """Extract patient audio segments from the full audio file"""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(AUDIO_CHANNELS)
        print(f"Loaded audio: {len(audio)/1000:.1f} seconds, {audio.frame_rate}Hz, {audio.channels} channels")
    except Exception as e:
        print(f"Failed to load audio {audio_path.name}: {e}")
        return False

    segments = parse_transcript(transcript_path)
    if not segments:
        print(f"No PAT segments found in {transcript_path.name}. Skipping.")
        return False

    combined = AudioSegment.silent(duration=0)
    valid_segments = 0
    invalid_segments = 0
    
    for start_ms, end_ms in segments:
        if 0 <= start_ms < end_ms <= len(audio):
            combined += audio[start_ms:end_ms]
            valid_segments += 1
        else:
            print(f"Invalid segment [{start_ms}, {end_ms}] (audio length: {len(audio)}ms). Skipped.")
            invalid_segments += 1

    if valid_segments == 0:
        print(f"No valid segments found for {transcript_path.name}")
        return False

    print(f"Extracted {valid_segments} valid segments, skipped {invalid_segments} invalid segments")
    print(f"Total extracted audio duration: {len(combined)/1000:.1f} seconds")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        combined.export(output_path, format="mp3")
        print(f"Saved: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save audio to {output_path}: {e}")
        return False

def format_time(milliseconds: int) -> str:
    """Convert milliseconds to mm:ss.xxx format."""
    total_seconds = milliseconds // 1000
    ms = milliseconds % 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"

def process_all():
    """Process all audio files in the input directory"""

    input_audio_dir = Path(DEFAULT_INPUT_AUDIO_DIR)
    input_trans_dir = Path(DEFAULT_INPUT_TRANS_DIR)
    output_dir = Path(DEFAULT_OUTPUT_DIR)

    # Validate input directories
    if not input_audio_dir.exists():
        print(f"Error: Input audio directory not found: {input_audio_dir}")
        return

    if not input_trans_dir.exists():
        print(f"Error: Input transcript directory not found: {input_trans_dir}")
        return

    # Find audio files
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        for path in input_audio_dir.glob(f"*.{ext}"):
            audio_files.append((path, ext.lower()))
    
    if not audio_files:
        print(f"Error: No audio files found in {input_audio_dir}")
        print(f"Looking for extensions: {AUDIO_EXTENSIONS}")
        return

    print(f"Found {len(audio_files)} audio file(s) to process...")
    print(f"Audio directory: {input_audio_dir}")
    print(f"Transcript directory: {input_trans_dir}")
    print(f"Output directory: {output_dir}")
    print()

    success_count = 0
    '''
    ### output without tqdm for easier debugging
    for audio_file, extension in audio_files:
        base = audio_file.stem
        transcript_file = input_trans_dir / f"{base}.txt"
        #output_file = output_dir / f"{base}_pat.mp3"
        output_file = output_dir / f"{base}_pat.{extension}"

        print(f"Processing {audio_file.name}")

        if not transcript_file.exists():
            print(f"Warning: Transcript missing for {audio_file.name}. Skipping.")
            continue

        print(f"\nProcessing: {audio_file.name}")
        ok = extract_patient_audio(audio_file, transcript_file, output_file)
        
        if ok:
            print(f"Success: {output_file.name}")
            success_count += 1
            #if success_count>2:
            #    break
        else:
            print(f"Failed: {audio_file.name}")


    '''
    with tqdm(audio_files, desc="Processing", unit="file") as pbar:
        for audio_file, extension in pbar:
            base = audio_file.stem
            transcript_file = input_trans_dir / f"{base}.txt"
            #output_file = output_dir / f"{base}_pat.mp3"
            output_file = output_dir / f"{base}_pat.{extension}"

            pbar.set_postfix_str(f"Processing {audio_file.name}")

            if not transcript_file.exists():
                pbar.write(f"Warning: Transcript missing for {audio_file.name}. Skipping.")
                continue

            pbar.write(f"\nProcessing: {audio_file.name}")
            ok = extract_patient_audio(audio_file, transcript_file, output_file)
            
            if ok:
                pbar.write(f"Success: {output_file.name}")
                success_count += 1
                #if success_count>2:
                #    break
            else:
                pbar.write(f"Failed: {audio_file.name}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(audio_files)} audio files")
    if success_count < len(audio_files):
        print(f"Failed: {len(audio_files) - success_count} audio files")

if __name__ == "__main__":
    process_all()
