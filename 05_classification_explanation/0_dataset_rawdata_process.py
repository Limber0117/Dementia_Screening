"""
This script processes .txt transcript files from the RAWTEXT folder, extracts ground-truth 
information for evaluation, copies and renames audio and text files, and saves the extracted 
information into a CSV file.

!!!!! This script is designed to process the "DementiaBank" dataset, in which the groundtruth documents and audio files are separated and are manually downloaded.

NOTE: This script preserves existing data and only supplements new information.
- Existing files in the output directory will NOT be overwritten
- Existing entries in the CSV will be preserved, new entries will be appended
- Missing .mp3 files are handled gracefully (not all .txt files have corresponding audio)

Folder Structure:
    datasets/English/RAWMP3    - Raw .mp3 audio files
    datasets/English/RAWTEXT   - Raw .txt transcript files
    datasets/English/          - Processed audio and text files (output)
    datasets/groundtruth/      - Ground truth CSV file (output)
"""

import os
import csv
import re
import shutil
from dotenv import load_dotenv

# Load configuration from .env
load_dotenv()

# Input directories
RAW_MP3_DIR = os.path.join(os.getcwd(), os.getenv("RAW_MP3_DIR", "datasets/English/RAWMP3"))
RAW_TEXT_DIR = os.path.join(os.getcwd(), os.getenv("RAW_TEXT_DIR", "datasets/English/RAWTEXT"))

# Output directories
PROCESSED_DIR = os.path.join(os.getcwd(), os.getenv("PROCESSED_DIR", "datasets/English"))
GROUNDTRUTH_CSV = os.path.join(os.getcwd(), os.getenv("GROUNDTRUTH_CSV", "datasets/groundtruth/groundtruth.csv"))

# Regex patterns
pid_pattern = re.compile(r"@PID:\s*\S+/(.+)")
media_pattern = re.compile(r"@Media:\s*([^,]+)")
topic_pattern = re.compile(r"@G:\s*(.+)")
id_pattern = re.compile(r"@ID:\s*(.+)")


def parse_id_line(id_line):
    """
    Parse the @ID line to extract participant information.
    
    Example:
    @ID: eng|Pitt|PAR|78;|male|ProbableAD||Participant|16||
    Fields (pipe-separated):
      0 = language (eng)
      1 = corpus (Pitt)
      2 = role (PAR/INV)
      3 = age (78;)
      4 = gender (male)
      5 = diagnosis (ProbableAD)
      6 = (empty)
      7 = participant type (Participant)
      8 = MMSE score (16)
    
    Returns:
        tuple: (language, age, gender, diagnosis, mmse)
    """
    parts = id_line.split("|")
    
    language = parts[0] if len(parts) > 0 else ""
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

    return language, age, gender, diagnosis, mmse


def extract_from_file(filepath):
    """
    Extracts required values from one .txt transcript file.
    
    Args:
        filepath: Path to the .txt file
        
    Returns:
        tuple: (pid, media_name, language, age, gender, diagnosis, mmse, topic)
    """
    pid = ""
    media_name = ""
    topic = ""
    language = ""
    age = ""
    gender = ""
    diagnosis = ""
    mmse = ""

    try:
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
                    topic = m.group(1).strip()
                    continue
                
                # ID line - only process Participant lines (PAR), not Investigator (INV)
                if line.startswith("@ID:") and "|PAR|" in line:
                    m = id_pattern.search(line)
                    if m:
                        language, age, gender, diagnosis, mmse = parse_id_line(m.group(1).strip())
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    
    return pid, media_name, language, age, gender, diagnosis, mmse, topic


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
    
    return existing_pids, existing_rows


def copy_file_if_not_exists(src_path, dest_path):
    """
    Copy a file from source to destination only if destination doesn't exist.
    
    Args:
        src_path: Source file path
        dest_path: Destination file path
        
    Returns:
        str: 'copied' if file was copied, 'exists' if already exists, 'not_found' if source missing, 'error' if failed
    """
    # Check if destination already exists
    if os.path.exists(dest_path):
        return 'exists'
    
    # Check if source exists
    if not os.path.exists(src_path):
        return 'not_found'
    
    try:
        shutil.move(src_path, dest_path)
        return 'copied'
    except Exception as e:
        print(f"Error copying {src_path} to {dest_path}: {e}")
        return 'error'


def main():
    new_results = []
    files_processed = 0
    audio_copied = 0
    audio_skipped = 0
    audio_not_found = 0
    text_copied = 0
    text_skipped = 0
    text_errors = 0
    entries_skipped = 0
    
    print(f"Processing files from: {RAW_TEXT_DIR}")
    print(f"Looking for audio in: {RAW_MP3_DIR}")
    print(f"Output directory: {PROCESSED_DIR}")
    print(f"Groundtruth CSV: {GROUNDTRUTH_CSV}")
    print("-" * 50)
    
    # Load existing CSV data
    existing_pids, existing_rows = load_existing_csv(GROUNDTRUTH_CSV)
    
    # Process all .txt files in RAWTEXT folder
    try:
        for root, dirs, files in os.walk(RAW_TEXT_DIR):
            for filename in files:
                if filename.lower().endswith(".txt"):
                    fullpath = os.path.join(root, filename)
                    
                    try:
                        # Extract information from the file
                        pid, media_name, language, age, gender, diagnosis, mmse, topic = extract_from_file(fullpath)
                        
                        if not pid:
                            print(f"Warning: No PID found in {filename}, skipping...")
                            continue
                        
                        files_processed += 1
                        
                        # Check if this PID already exists in CSV
                        if pid in existing_pids:
                            print(f"Skipping PID {pid} - already exists in CSV")
                            entries_skipped += 1
                            continue
                        
                        # Generate the new filename based on PID
                        new_txt_filename = f"{pid}.txt"
                        new_mp3_filename = f"{pid}.mp3"

                        
                        src_mp3_path = os.path.join(RAW_MP3_DIR, f"{media_name}.mp3")
                        
                        if os.path.exists(fullpath) and os.path.exists(src_mp3_path):

                            # Copy and rename the text file (only if not exists)                        
                            try:
                                src_txt_path = fullpath                            
                                dest_txt_path = os.path.join(PROCESSED_DIR, new_txt_filename)
                                result = copy_file_if_not_exists(src_txt_path, dest_txt_path)
                                if result == 'copied':
                                    text_copied += 1
                                    print(f"Copied text: {filename} -> {new_txt_filename}")
                                elif result == 'exists':
                                    text_skipped += 1
                                    print(f"Text file already exists: {new_txt_filename}")
                                elif result == 'not_found':
                                    text_errors += 1
                                    print(f"Warning: Source text file not found: {filename}")
                                else:
                                    text_errors += 1
                            except Exception as e:
                                text_errors += 1
                                print(f"Error processing text file {filename}: {e}")
                            
                            # Copy and rename the audio file (only if not exists)
                            try:
                                if media_name:                                
                                    dest_mp3_path = os.path.join(PROCESSED_DIR, new_mp3_filename)
                                    result = copy_file_if_not_exists(src_mp3_path, dest_mp3_path)
                                    if result == 'copied':
                                        audio_copied += 1
                                        print(f"Copied audio: {media_name}.mp3 -> {new_mp3_filename}")
                                    elif result == 'exists':
                                        audio_skipped += 1
                                        print(f"Audio file already exists: {new_mp3_filename}")
                                    elif result == 'not_found':
                                        audio_not_found += 1
                                        print(f"Warning: Audio file not found: {media_name}.mp3 (PID: {pid})")
                                else:
                                    audio_not_found += 1
                                    print(f"Warning: No media name found for PID {pid}")
                            except Exception as e:
                                audio_not_found += 1
                                print(f"Error processing audio for PID {pid}: {e}")
                            
                            # Append to new results (even if audio is missing)
                            new_results.append({
                                "PID": pid,
                                "filename": new_txt_filename,
                                "mediaName": media_name,
                                "language": language,
                                "age": age,
                                "gender": gender,
                                "diagnosis": diagnosis,
                                "MMSE": mmse,
                                "topic": topic
                            })
                    
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
                        continue
    
    except Exception as e:
        print(f"Error walking directory {RAW_TEXT_DIR}: {e}")

    # Combine existing rows with new results
    all_results = existing_rows + new_results
    
    # Write to CSV (preserving existing + adding new)
    try:
        with open(GROUNDTRUTH_CSV, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["PID", "filename", "mediaName", "language", "age", "gender", "diagnosis", "MMSE", "topic"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return

    print("-" * 50)
    print(f"Processing complete!")
    print(f"  Files processed: {files_processed}")
    print(f"  Entries skipped (already in CSV): {entries_skipped}")
    print(f"  New entries added to CSV: {len(new_results)}")
    print(f"  Text files - copied: {text_copied}, skipped: {text_skipped}, errors: {text_errors}")
    print(f"  Audio files - copied: {audio_copied}, skipped: {audio_skipped}, not found: {audio_not_found}")
    print(f"  Total entries in CSV: {len(all_results)}")
    print(f"  CSV saved to: {GROUNDTRUTH_CSV}")


if __name__ == "__main__":
    main()