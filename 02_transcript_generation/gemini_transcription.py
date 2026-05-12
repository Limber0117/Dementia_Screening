#!/usr/bin/env python3
"""
Hybrid Asynchronous Gemini Transcription
- Sync upload + polling (SDK limitation)
- Async model invocation (supported)
- Sequential async calls for stability
- Retry logic for robustness
- Plain text transcription
"""

import os
import sys
import shutil
import argparse
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai
from pydub import AudioSegment


# ------------------------------------------------------------
# Gemini Transcriber
# ------------------------------------------------------------

class GeminiTranscriber:
    def __init__(self, api_key: str, model_name: str = "gemini-3-pro-preview"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Escaped prompt (important)
        
        self.transcription_prompt = """
        You are helping with medical research by transcribing de-identified conversations. This content is for legitimate research purposes only.
        You are transcribing a medical conversation between two speakers (Speaker A is a doctor and Speaker B is a patient), or just one speaker (Speaker B, the patient) to answers questions or describe something.        
        Please provide a detailed transcription with the following requirements:

        1. **Language Detection & Transcription**:
        - Automatically detect the spoken language and transcribe in that same language
        - Maintain the original language throughout the transcription
        - If multiple languages are spoken, transcribe each in its original language

        2. Identify the speaker(s) based on:
        - Terminology usage
        - Question patterns. For example, the Speaker A (doctor) used to ask questions, give instructions and/or elaborate requirements, etc., and Speaker B (patient) used to answer questions or describe something.
        - Professional language used in the conversation
        - use "DOC" for Speaker A (doctor) and "PAT" (patitent) for Speaker B in the transcript
        - each sentence in the transcript has a start timestamp and end timestamp in front of the sentence in the format [MM:SS.mmm - MM:SS.mmm]
        - after the transcript of a sentence, give each word in the sentence a start timestamp in the format (word|MM:SS.mmm). All the words in a sentence should be in one line, enclosed by {{ }} and separated by space.
        - transcribe accurately.

        3. Format the output as a plain text with the following exemplary structure:
        The first part consists of 1) the start and end timestamps for the entire sentence, in the format [MM:SS.mmm - MM:SS.mmm]; 2) the speaker label (DOC or PAT); and 3) the transcribed sentence. 
        The second part enclosed in {{ }} contains each word with its corresponding start timestamp, .
      
        Example:
            [00:00.000 - 00:03.300] DOC: Do you remember what those three strategies I taught you were? {{(Do|00:00.000) (you|00:00.300) (remember|00:00.600) (what|00:00.900) (those|00:01.200) (three|00:01.500) (strategies|00:01.800) (I|00:02.100) (taught|00:02.400) (you|00:02.700) (were?|00:03.000)}}
            [00:04.100 - 00:05.600] PAT: writing in your calendar. {{(writing|00:04.100) (in|00:04.400) (your|00:04.700) (calendar|00:05.000) (.|00:05.300)}}
     
        4. **Accuracy & Handling Uncertainty**:
        - Transcribe accurately in the detected language
        - If a word is unclear, use "*" as placeholder
        - Preserve medical terminology exactly as spoken        
        5. Estimate word-level timestamps based on speech pace and audio duration, including the placeholder.
        6. The duration of the entire audio is approximately {duration} seconds.
        7. Please ignore any background noises or music and focus solely on the spoken words.
        8. This transcription is for authorised medical research purposes.
      
        IMPORTANT: Maintain the original language of the conversation. Do not translate to English.

"""

    # ------------------------------------------------------------
    def get_audio_duration(self, audio_path: str) -> float:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0

    # ------------------------------------------------------------
    async def transcribe_with_gemini_async(self, audio_path: str) -> str:
        """Async inference only — upload must be synchronous."""

        print(f"Uploading audio: {audio_path}")

        duration = self.get_audio_duration(audio_path)

        try:
            # ---- SYNC UPLOAD (SDK limitation) ----
            audio_file = genai.upload_file(audio_path)

            # Polling MUST also be sync (SDK limitation)
            while audio_file.state.name == "PROCESSING":
                print("Processing audio on Gemini...")
                await asyncio.sleep(3)  # async-friendly wait
                audio_file = genai.get_file(audio_file.name)

            if audio_file.state.name == "FAILED":
                raise RuntimeError("Gemini failed to process the audio file")

            # ---- ASYNC GENERATION ----
            print(f"Transcribing with Gemini (async)...duration={duration:.2f}s")

            prompt = self.transcription_prompt.format(duration=duration)

            response = await self.model.generate_content_async(
                [audio_file, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192
                )
            )
            

            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")            
            raise


    # ------------------------------------------------------------
    async def transcribe_with_retry(self, audio_path: str, retries: int = 3) -> str:
        for attempt in range(1, retries + 1):
            try:
                result = await self.transcribe_with_gemini_async(audio_path)
                # Check if response is empty
                if not result or len(result.strip()) < 10:  # Very short response
                    raise RuntimeError("Empty response from Gemini")
                return result
                #return await self.transcribe_with_gemini_async(audio_path)
            except Exception as e:
                print(f"[Retry {attempt}/{retries}] Transcription error: {e}")
                if attempt == retries:
                    #raise
                    # Return a default error message instead of raising
                    return f"Transcription failed after {retries} attempts. Error: {e}"
                await asyncio.sleep(5)

    # ------------------------------------------------------------
    def transcribe(self, audio_path: str) -> str:
        """Runs async transcription synchronously."""
        return asyncio.run(self.transcribe_with_retry(audio_path))

# ------------------------------------------------------------
# File Processing (Async Version)
# ------------------------------------------------------------

async def process_file_async(transcriber: GeminiTranscriber, input_path: Path, input_dir: Path, output_dir: Path) -> bool:
    try:
        text_output = await transcriber.transcribe_with_retry(str(input_path))

        output_path = output_dir / f"{input_path.stem}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_output)
        
        # Create processed directory if it doesn't exist
        processed_dir = input_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        shutil.move(str(input_path), str(processed_dir / input_path.name))

        print(f"✓ Saved transcript: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {e}")
        return False

async def process_all_files_async(transcriber: GeminiTranscriber, audio_files: list, input_dir: Path, output_dir: Path) -> int:
    """Process all files in a single event loop"""
    success_count = 0
    
    # Process files sequentially to avoid rate limits and event loop issues
    for audio_file in audio_files:
        if await process_file_async(transcriber, audio_file, input_dir, output_dir):
            success_count += 1
        # Small delay between files to avoid overwhelming the API
        await asyncio.sleep(2)
    
    return success_count

# ------------------------------------------------------------
# Updated Main Program
# ------------------------------------------------------------

async def main_async():
    """Async main function that handles all files in one event loop"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Hybrid Async Gemini Transcription")
    parser.add_argument("--input-dir", default=os.getenv("DEFAULT_INPUT_DIR", "datasets/input"))
    parser.add_argument("--output-dir", default=os.getenv("DEFAULT_OUTPUT_DIR", "datasets/output/transcripts"))
    parser.add_argument("--api-key", default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument(
        "--file-extensions",
        nargs="+",
        default=os.getenv("AUDIO_EXTENSIONS", "mp3,wav").split(",")
    )

    args = parser.parse_args()

    if not args.input_dir:
        print("Error: --input-dir required.")
        sys.exit(1)
    if not args.api_key:
        print("Error: Missing GOOGLE_API_KEY.")
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = []
    for ext in args.file_extensions:
        print(f"Searching for *.{ext} files...")
        audio_files += list(input_dir.glob(f"*.{ext}"))

    if not audio_files:
        print("No audio files found.")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files.")

    transcriber = GeminiTranscriber(api_key=args.api_key, model_name=args.model)

    # Process all files in one event loop
    success = await process_all_files_async(transcriber, audio_files, input_dir, output_dir)

    print("\n====================================")
    print(f"Completed {success}/{len(audio_files)} files")
    print("====================================")

def main():
    """Run the async main function"""
    asyncio.run(main_async())


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()