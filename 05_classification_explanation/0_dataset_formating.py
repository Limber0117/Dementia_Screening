"""
This script processes .txt transcript files in a specified folder, and extract the useful ground-truth for the evaluation.
It saves the extracted information into a CSV file.
"""

import os
import csv
import re
from dotenv import load_dotenv
from pathlib import Path
import argparse

try:
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE
    from mutagen import MutagenError
except ImportError:
    print("Error: 'mutagen' library is required.")
    print("Install it with: pip install mutagen")
    exit(1)

# Load folder path from .env
load_dotenv()
DATA_FOLDER = os.path.join(os.getcwd(),os.getenv("DEFAULT_INPUT_DIR", "datasets/input"))

# Output csv file
OUTPUT_CSV = os.path.join(os.getcwd(),os.getenv("GROUNDTRUTH_CSV", "datasets/groundtruth/groundtruth.csv"))


# Regex patterns
pid_pattern = re.compile(r"@PID:\s*\S+/(.+)")
media_pattern = re.compile(r"@Media:\s*([^,]+)")
topic_pattern = re.compile(r"@G:\s*(.+)")
id_pattern = re.compile(r"@ID:\s*(.+)")

def parse_id_line(id_line):
    """
    Example:
    @ID: eng|Pitt|PAR|71;|female|ProbableAD||Participant|13||
    @ID: eng|Pitt|PAR|66;00.|male|Control||Participant|30||
    Fields:
      0 = language
      1 = author/doctor
      3 = age "66;00."
      4 = gender
      5 = diagnosis
    """
    parts = id_line.split("|")
    
    language = parts[0] if len(parts) > 0 else ""
    doctor = parts[1] if len(parts) > 1 else ""
    raw_age = parts[3] if len(parts) > 3 else ""
    gender = parts[4] if len(parts) > 4 else ""
    diagnosis = parts[5] if len(parts) > 5 else ""
    mmse = parts[8] if len(parts) > 8 else ""

    # Extract numeric age before the semicolon
    age = ""
    if raw_age:
        match = re.match(r"(\d+)", raw_age)
        if match:
            age = match.group(1)

    return language, doctor, age, gender, diagnosis, mmse
    


def extract_from_file(filepath: str, audio_folder: str):
    """
    Extracts required values from one .txt file
    """
    pid = ""
    media_name = ""
    topic = ""
    topics=[]
    language = ""
    age = ""
    gender = ""
    mmse=""
    diagnosis = ""
    doctor=""
    duration=0.0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # PID extraction
            m = pid_pattern.search(line)
            if m:
                pid = m.group(1).strip()
                continue
            
            # Media line
            m = media_pattern.search(line)
            if m:
                media_name = m.group(1).strip()
                continue
            
            # Topic line
            m = topic_pattern.search(line)
            if m:
                topics.append(m.group(1).strip())
                continue
            
            # ID line
            if line.startswith("@ID:") and "|PAR|" in line:
                m = id_pattern.search(line)
                if m:
                    language, doctor, age, gender, diagnosis, mmse = parse_id_line(m.group(1).strip())
        if topics:
            topic = "#".join(topics)

        if pid:
            duration = get_audio_duration(audio_folder,pid)   

    return pid, media_name, language, age, gender, diagnosis, mmse, topic, doctor,duration

def load_existing_csv(csv_path):
    """
    Load existing CSV data and return a set of existing PIDs and list of existing rows.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        tuple: (set of existing PIDs, list of existing row dictionaries)
    """
    existing_pids = set()
    existing_rows = []
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_rows.append(row)
                    if row.get("PID"):
                        existing_pids.add(row["PID"])
            print(f"Loaded {len(existing_rows)} existing entries from CSV")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    
    print(f"\n+ Existing PIDs loaded: {len(existing_pids)}\n")

    return existing_pids, existing_rows


def get_audio_duration(file_path: str, pid:str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file (.mp3 or .wav)
        
    Returns:
        Duration in seconds, or None if file cannot be read
    """
    try:
        filename_mp3 = os.path.join(file_path,pid + ".mp3")
        filename_wav = os.path.join(file_path,pid + ".wav") 
        if os.path.exists(filename_mp3):
            print(f"  Reading MP3 file: {filename_mp3}")
            audio = MP3(filename_mp3)
        elif os.path.exists(filename_wav):
            audio = WAVE(filename_wav)
            print(f"  Reading WAV file: {filename_wav}")
        else:
            print(f"  Warning: Unsupported format for {pid}. Supported formats are .mp3 and .wav")
            return 0.0
        print(f"  Duration for {file_path}: {audio.info.length} seconds")
        return audio.info.length
    except MutagenError as e:
        print(f"  Error reading {file_path}: {e}")
        return 0.0
    except Exception as e:
        print(f"  Unexpected error reading {file_path}: {e}")
        return 0.0





def main():
    results = []
    
    # Load existing CSV data
    existing_pids, existing_rows = load_existing_csv(OUTPUT_CSV)

    # Get list of .txt files
    for root, dirs, files in os.walk(DATA_FOLDER):
        for filename in files:
            if filename.lower().endswith(".txt"):
                fullpath = os.path.join(root, filename)
                
                pid, media_name, language, age, gender, diagnosis, mmse, topic, doctor, duration = extract_from_file(fullpath,DATA_FOLDER)
                
                if not pid in existing_pids:
                    results.append({
                        "PID": pid,
                        "filename": filename,
                        "mediaName": media_name,
                        "language": language,
                        "age": age,
                        "gender": gender,
                        "diagnosis": diagnosis,
                        "MMSE":mmse,
                        "topic": topic,
                        "doctor": doctor,
                        "duration": duration
                    })
                else:
                    print(f"Skipping PID {pid} as it is in existing CSV.")



    # Combine existing rows with new results
    all_results = existing_rows + results
    
    # Write to CSV (preserving existing + adding new)
    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["PID", "filename", "mediaName", "language", "age", "gender", "diagnosis", "MMSE", "topic", "doctor", "duration"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return

    print(f"Extraction complete. CSV saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
