#!/usr/bin/env python3
"""
Enhanced Gemini Transcription for Long Audio Files
- Multi-stage processing: semantic segmentation → clip transcription → combination
- Smart audio segmentation based on semantic boundaries
- Word-level timestamp correction
- Handles long audio files beyond Gemini's token limits
"""

import os
import sys
import shutil
import argparse
import asyncio
import tempfile
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json

from dotenv import load_dotenv
import google.generativeai as genai
from pydub import AudioSegment
import ffmpeg


# ------------------------------------------------------------
# Enhanced Gemini Transcriber with Segmentation
# ------------------------------------------------------------

class EnhancedGeminiTranscriber:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Prompt for initial sentence-level transcript (for segmentation)
        self.segmentation_prompt = """
        
        You are helping with medical research by transcribing de-identified conversations. This content is for legitimate research purposes only.
        You are transcribing a medical conversation. There might be two speakers in the conversation: Speaker A is a doctor, and Speaker B is a patient. Or, there is just one speaker (Speaker B, the patient).  
        
        Please diarise the conversation and  provide a detailed transcription with the following requirements:

        1. **Language Detection & Transcription**:
        - Automatically detect the spoken language and transcribe in that same language
        - Maintain the original language throughout the transcription
        - If multiple languages are spoken, transcribe each in its original language

        2. **Speaker Identification Strategies**: 
        Please follow these clues to identify speakers:
        - Identify speakers based on terminology usage, question patterns, and professional language
        - The doctor (DOC) typically asks questions, gives instructions, guide the patient, or elaborates a topic
        - Sometimes, the doctor helps the patient to clarify their answers, but mostly focuses on asking questions or giving instructions
        - The patient (PAT) typically answers questions or describes something, like describing the Cookie Theft picture or their life experiences, stories, or illness symptoms
        - Sometimes, the patient may repeat the words or phrases given by the doctor
        - Use "DOC" for Speaker A (doctor) and "PAT" for Speaker B (patient) in the transcript
        - If there are two speakers, the conversation may start with either speaker
        - If there is only one speaker, that speaker is the patient
        - Each sentence in the transcript has a start timestamp and end timestamp in the format [MM:SS.mmm - MM:SS.mmm]
        - Transcribe accurately.
        
        The generated transcript consists of 1) the start and end timestamps for the entire sentence, in the format [MM:SS.mmm - MM:SS.mmm]; 2) the speaker label (DOC or PAT); and 3) the transcribed sentence. 
        
        Example format is as follows:
        [00:00.000 - 00:03.300] DOC: How are you feeling today?
        [00:04.100 - 00:07.200] PAT: I've been having some headaches.
        [00:08.500 - 00:12.100] DOC: When did these headaches start?
        
        The audio duration is approximately {duration} seconds.
        """

        # Prompt for detailed word-level transcription (for clips)
        self.detailed_transcription_prompt = """
        You are helping with medical research by transcribing de-identified conversations. This content is for legitimate research purposes only.
        You are transcribing a medical conversation. There might be two speakers in the conversation: Speaker A is a doctor, and Speaker B is a patient. Or, there is just one speaker (Speaker B, the patient). 
        
        Please provide a detailed transcription with the following requirements:

        1. **Language Detection & Transcription**:
        - Automatically detect the spoken language and transcribe in that same language
        - Maintain the original language throughout the transcription
        - If multiple languages are spoken, transcribe each in its original language

        2. **Speaker Identification & Continuity**:
        Please follow these clues to identify speakers:
        - Identify speakers based on terminology usage, question patterns, and professional language
        - The doctor (DOC) typically asks questions, gives instructions, guide the patient, or elaborates a topic
        - Sometimes, the doctor helps the patient to clarify their answers, but mostly focuses on asking questions or giving instructions
        - The patient (PAT) typically answers questions or describes something, like describing the Cookie Theft picture or their life experiences, stories, or illness symptoms
        - Sometimes, the patient may repeat the words or phrases given by the doctor
        - If there are two speakers, the conversation may start with either speaker
        - If there is only one speaker, that speaker is the patient
        
        - Use "DOC" for Speaker A (doctor) and "PAT" for Speaker B (patient) in the transcript       
        - Each sentence in the transcript has a start timestamp and end timestamp in the format [MM:SS.mmm - MM:SS.mmm]
        - Right after the transcript of a sentence, give each word in the sentence a start timestamp in the format (word|MM:SS.mmm). All words in a sentence should be in one line, enclosed by {{ }} and separated by space.
        - The word-level transcript of a sentenc should be put in the same line as the sentence transcript, right after the sentence without line breaks.
            Example:
            [00:00.000 - 00:03.300] DOC: Do you remember what those three strategies I taught you were? {{(Do|00:00.000) (you|00:00.300) (remember|00:00.600) (what|00:00.900) (those|00:01.200) (three|00:01.500) (strategies|00:01.800) (I|00:02.100) (taught|00:02.400) (you|00:02.700) (were?|00:03.000)}}

        - Transcribe accurately.
        - Consider the speaker context provided for this clip to maintain continuity

        4. Format the output as a plain text with the following exemplary structure:
        The first part consists of 1) the start and end timestamps for the entire sentence, in the format [MM:SS.mmm - MM:SS.mmm]; 2) the speaker label (DOC or PAT); and 3) the transcribed sentence. 
        The second part enclosed in {{ }} contains each word with its corresponding start timestamp.
      
        Example:
            [00:00.000 - 00:03.300] DOC: Do you remember what those three strategies I taught you were? {{(Do|00:00.000) (you|00:00.300) (remember|00:00.600) (what|00:00.900) (those|00:01.200) (three|00:01.500) (strategies|00:01.800) (I|00:02.100) (taught|00:02.400) (you|00:02.700) (were?|00:03.000)}}
            [00:04.100 - 00:05.600] PAT: writing in your calendar. {{(writing|00:04.100) (in|00:04.400) (your|00:04.700) (calendar|00:05.000) (.|00:05.300)}}
     
        5. **Accuracy & Handling Uncertainty**:
        - Transcribe accurately in the detected language
        - If a word is unclear, use "*" as placeholder
        - Preserve medical terminology exactly as spoken        
        - Estimate word-level timestamps based on speech pace and audio duration, including the placeholder.
        - The duration of this audio clip is approximately {duration} seconds.
        - Please ignore any background noises or music and focus solely on the spoken words.
        - This transcription is for authorised medical research purposes.
      
        IMPORTANT: Maintain the original language of the conversation. Do not translate to English.
        """

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0

   

    async def transcribe_with_gemini_async(self, audio_path: str, prompt: str, is_segmentation: bool = False) -> str:
        """Async transcription with different prompts for segmentation vs detailed transcription"""
        print(f"Uploading audio: {audio_path}")

        # Check if file exists FIRST
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file does not exist: {audio_path}")
        
        try:
            # Then get duration (this will also validate the file is readable)
            duration = self.get_audio_duration(audio_path)
            print(f"Audio duration: {duration:.2f} seconds")
            
            # Sync upload
            print("Uploading to Gemini...")
            audio_file = genai.upload_file(audio_path)
            print(f"Upload successful. File state: {audio_file.state.name}")

            # Poll for processing completion
            poll_count = 0
            while audio_file.state.name == "PROCESSING":
                poll_count += 1
                print(f"Processing audio on Gemini... (poll #{poll_count})")
                await asyncio.sleep(10)
                audio_file = genai.get_file(audio_file.name)
                print(f"Current state: {audio_file.state.name}")

            if audio_file.state.name == "FAILED":
                raise RuntimeError(f"Gemini failed to process the audio file. State: {audio_file.state.name}")

            # Async generation
            task_type = "segmentation" if is_segmentation else "detailed transcription"
            print(f"Performing {task_type} with Gemini (async)...")

            #formatted_prompt = prompt.format(duration=str(duration))
            print(f"Prompt length: {len(prompt)} characters")
        
            print("Sending generation request...")
            response = await self.model.generate_content_async(
                [audio_file, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=65536 if is_segmentation else 65536
                )
            )
            print("Generation feedback received!")

            return response.text.strip()
            
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
            raise RuntimeError(f"Audio file not found or inaccessible: {audio_path}")
            
        except PermissionError as e:
            print(f"Permission error: {e}")
            raise RuntimeError(f"No permission to read audio file: {audio_path}")
            
        except Exception as e:
            print(f"Gemini API error: {type(e).__name__}: {e}")
            # Add more detailed error info
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def transcribe_with_retry(self, audio_path: str, prompt: str, is_segmentation: bool = False, retries: int = 0) -> str:
        """Retry logic for transcription"""
        for attempt in range(0, retries + 1):
            try:
                result = await self.transcribe_with_gemini_async(audio_path, prompt, is_segmentation)
                print(f"Transcription result length: {len(result)} characters")
                #print(result)
                if not result or len(result.strip()) < 10:
                    raise RuntimeError("Empty response from Gemini")
                return result
            except Exception as e:
                print(f"[Retry {attempt}/{retries}] Transcription error: {e}")
                if attempt == retries:
                    return f"Transcription failed after {retries} attempts. Error: {e}"
                await asyncio.sleep(5)

    def parse_sentence_level_transcript(self, transcript: str) -> List[Dict]:
        """Parse sentence-level transcript into structured data"""
        print("Parsing sentence-level transcript...")
        sentences = []
        pattern = r'\[(\d{2}:\d{2}\.\d{3}) - (\d{2}:\d{2}\.\d{3})\] (DOC|PAT): ([^\n]+)'
        
        if not transcript or len(transcript.strip()) < 10:
            print("Warning: Empty or too short transcript for parsing.")
            return sentences

        for line in transcript.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = re.match(pattern, line)
            if match:
                start_time, end_time, speaker, text = match.groups()
                sentences.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker': speaker,
                    'text': text,
                    'start_ms': self.timestamp_to_ms(start_time),
                    'end_ms': self.timestamp_to_ms(end_time)
                })
                
        return sentences

    def timestamp_to_ms(self, timestamp: str) -> int:
        """Convert MM:SS.mmm timestamp to milliseconds"""
        minutes, rest = timestamp.split(':')
        seconds, milliseconds = rest.split('.')
        return int(minutes) * 60000 + int(seconds) * 1000 + int(milliseconds)

    def ms_to_timestamp(self, ms: int) -> str:
        """Convert milliseconds to MM:SS.mmm format"""
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    async def get_optimal_segmentation_points(self, sentences: List[Dict], total_duration: float, max_clip_duration: int = 120) -> List[Tuple[int, str, int]]:
        """
        Use LLM to find optimal segmentation points that don't break sentences
        Returns: List of tuples (segmentation_point_ms, previous_speaker, speaker_change_flag:1 Yes & 0 No)
        """
            
        # Simple heuristic: segment at natural breaks (speaker changes or long pauses)
        # For production, you might want to use a more sophisticated LLM call here
        segmentation_points = []
        current_duration = 0
        
        for i, sentence in enumerate(sentences):
            sentence_duration = sentence['end_ms'] - sentence['start_ms']
            
            # Check if adding this sentence would exceed max duration
            if current_duration + sentence_duration > max_clip_duration * 1000 and i > 0:
                # Look for a good break point (preferably at speaker change)
                prev_sentence = sentences[i-1]
                # Prefer to break at speaker changes
                if prev_sentence['speaker'] != sentence['speaker']:
                    segmentation_points.append((prev_sentence['end_ms'], prev_sentence['speaker'],1))
                    current_duration = sentence_duration
                else:
                    # If no speaker change, continue to break before current sentence
                    segmentation_points.append((prev_sentence['end_ms'], prev_sentence['speaker'],0))
                    current_duration = sentence_duration
            else:
                current_duration += sentence_duration
                
        return segmentation_points

    def split_audio_ffmpeg(self, audio_path: str, split_points: List[Tuple[int, str,int]], output_dir: Path) -> List[Tuple[str, int, int]]:
        """Split audio using ffmpeg at specified points"""
        print("Splitting audio into clips...")
        clips = []
        audio_duration_ms = int(self.get_audio_duration(audio_path) * 1000)
        
        # Extract just the timestamps for splitting
        split_timestamps = [point[0] for point in split_points]
        split_timestamps = sorted(split_timestamps)
        split_timestamps.append(audio_duration_ms)  # Add end point
        
        # Create debug directory if it doesn't exist
        debug_dir = Path("datasets/output/transcripts/").resolve()
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create debug directory: {e}")

        start_ms = 0
        clip_index = 0
        
        for end_ms in split_timestamps:
            if end_ms - start_ms < 1000:  # Skip very short clips
                start_ms = end_ms
                continue
                
            output_path = output_dir / f"clip_{clip_index:03d}.wav"
            
            # Convert milliseconds to seconds for ffmpeg
            start_sec = start_ms / 1000.0
            duration_sec = (end_ms - start_ms) / 1000.0
            
            try:
                (
                    ffmpeg
                    .input(audio_path, ss=start_sec, t=duration_sec)
                    .output(str(output_path), acodec='pcm_s16le', ac=1, ar='16000')
                    .overwrite_output()
                    .run(quiet=True)
                )
                print(f"  ✓ Created clip: {output_path.name} [{self.ms_to_timestamp(start_ms)} - {self.ms_to_timestamp(end_ms)}]")

                # Copy to debug directory with error handling
                try:
                    debug_file = debug_dir / output_path.name
                    shutil.copyfile(output_path, debug_file)
                    os.unlink(output_path)  # Remove original temp file after copying   
                    print(f"  ✓ Copied clip to debug directory: {debug_file}")
                except Exception as copy_error:
                    print(f"  Warning: Could not copy to debug directory: {copy_error}")


                #clips.append((str(output_path), start_ms, end_ms))
                clips.append((str(debug_file), start_ms, end_ms))
                clip_index += 1
                
            except Exception as e:
                print(f"Error splitting audio: {e}")
                
            start_ms = end_ms
            
        return clips

    def correct_word_timestamps(self, transcript: str, clip_start_ms: int) -> str:
        """Correct word timestamps by adding clip offset"""
        print("Correcting word-level timestamps...")
        def correct_timestamp(match,i:int):
            timestamp = match.group(i)
            original_ms = self.timestamp_to_ms(timestamp)
            corrected_ms = original_ms + clip_start_ms
            return self.ms_to_timestamp(corrected_ms)
        
        # Correct sentence-level timestamps
        sentence_pattern = r'\[(\d{2}:\d{2}\.\d{3}) - (\d{2}:\d{2}\.\d{3})\]'
        corrected_transcript = re.sub(sentence_pattern, 
                                    lambda m: f"[{correct_timestamp(m,1)} - {correct_timestamp(m,2)}]", 
                                    transcript)
        
        # Correct word-level timestamps
        word_pattern = r'\(([^|]+)\|(\d{2}:\d{2}\.\d{3})\)'
        corrected_transcript = re.sub(word_pattern, 
                                    lambda m: f"({m.group(1)}|{correct_timestamp(m,2)})", 
                                    corrected_transcript)
        
        return corrected_transcript

    async def transcribe_long_audio(self, audio_path: str, max_clip_duration: int = 120) -> str:
        """Main method for transcribing long audio files"""
        print(f"Starting transcription of long audio: {audio_path}")
        
        # Step 1: Get sentence-level transcript for segmentation
        print("Step 1: Getting sentence-level transcript for segmentation...")
        sentence_transcript = await self.transcribe_with_retry(
            audio_path, self.segmentation_prompt, is_segmentation=True
        )
        
        sentences = self.parse_sentence_level_transcript(sentence_transcript)
        if not sentences:
            # there is no sentence-level transcript parsed
            print("Warning: No sentences parsed from initial transcript")
            # Fallback: use fixed intervals
            total_duration = self.get_audio_duration(audio_path)
            split_points = [(i * max_clip_duration * 1000, "UNKNOWN",0) for i in range(1, int(total_duration // max_clip_duration) + 1)]
        else:
            # Step 2: Find optimal segmentation points
            print("Step 2: Finding optimal segmentation points...")
            total_duration = self.get_audio_duration(audio_path)
            split_points = await self.get_optimal_segmentation_points(sentences, total_duration, max_clip_duration)
            print("Segmentation points (ms, speaker, change_flag):", split_points)

        # Step 3: Split audio into clips
        print("Step 3: Splitting audio into clips...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            #clips = self.split_audio_ffmpeg(audio_path, [point[0] for point in split_points], temp_path)
            clips = self.split_audio_ffmpeg(audio_path, split_points, temp_path)
            print(f"Created {len(clips)} audio clips")
            
            # Step 4: Transcribe each clip with word-level timestamps
            print("Step 4: Transcribing clips with word-level timestamps...")
            all_transcripts = [None] * len(clips)
            
            # Process clips concurrently but maintain order
            tasks = []
            for clip_index, (clip_path, clip_start_ms, clip_end_ms) in enumerate(clips):
                task = self.process_single_clip_with_index(
                    clip_index, clip_path, clip_start_ms, clip_end_ms, split_points, sentences, all_transcripts
                )
                tasks.append(task)
                #sleep(15)  # Slight delay to avoid overwhelming the API
            
            # Wait for all clips to complete
            await asyncio.gather(*tasks)
            
            # Filter out any failed transcriptions and combine
            successful_transcripts = [t for t in all_transcripts if t is not None]
            
            # Step 5: Combine all transcripts in correct order
            print("Step 5: Combining all transcripts in correct order...")
            final_transcript = "\n\n".join(successful_transcripts)
            
            return final_transcript
        

    async def process_single_clip_with_index(self, clip_index: int, clip_path: str, clip_start_ms: int, 
                                        clip_end_ms: int, split_points: List[Tuple[int, str]], 
                                        sentences: List[Dict], all_transcripts: List[Optional[str]]) -> None:
        """Process a single clip and store result at the correct index"""
        #try:
        print(f"Transcribing clip {clip_index + 1}: {clip_path} (offset: {clip_start_ms}ms)")
        
        # Get speaker context for this clip
        speaker_context = self.get_speaker_context_for_clip(clip_index, split_points, sentences)
        #speaker_context = {'previous_speaker':"PAT",'speaker_change_flag':0}
        clip_transcript = await self.transcribe_clip_with_context(clip_index, clip_path, clip_start_ms, speaker_context)
        
        
        # Correct timestamps for global timeline
        #clip_transcript = """
        #[00:25.290 - 00:36.590] PAT: Someone must have called the fire department and they are they've come to the rescue bringing a ladder and hoping to get the tree get maybe get the man out of the tree because his ladder fell.
        #[00:37.400 - 00:41.590] PAT: And um, get help hopefully get the the cat out of the tree as well.
        #[00:43.270 - 00:45.380] PAT: And there's a bird in the tree, but that's not part of the story.
        #"""
        corrected_transcript = self.correct_word_timestamps(clip_transcript, clip_start_ms)
        
        # Store at the correct index to maintain order
        print()
        all_transcripts[clip_index] = corrected_transcript
        
        # Clean up temporary clip file
        try:
            os.unlink(clip_path)
            pass
        except:
            pass
            
        print(f"✓ Completed clip {clip_index + 1}")
            
        #except Exception as e:
        #    print(f"✗ Error processing clip {clip_index + 1}: {e}")
        #    # Store error message to maintain index alignment
        #    all_transcripts[clip_index] = f"[ERROR: Failed to transcribe clip {clip_index + 1}]"    



    def get_speaker_context_for_clip(self, clip_index: int, split_points: List[Tuple[int, str,int]], 
                                    sentences: List[Dict]) -> Dict[str,int]:
        """Get speaker context information for a specific clip"""
        context = {
            "previous_speaker": "UNKNOWN",
            "speaker_change_flag": 0
        }
        
        # For the first clip, we don't have a previous speaker
        if clip_index == 0:
            if sentences:
                context["previous_speaker"] = sentences[0].get('speaker', 'UNKNOWN')
            return context
        
        # For subsequent clips, use the previous speaker from segmentation
        if clip_index > 0 and clip_index - 1 < len(split_points):
            context["previous_speaker"] = split_points[clip_index - 1][1] # the last speaker at previous segmentation point
            context["speaker_change_flag"] = split_points[clip_index - 1][2]
        
        return context


    async def transcribe_clip_with_context(self, clip_index:int, clip_path: str, clip_start_ms: int, 
                                        speaker_context: Dict[str, int]) -> str:
        """Transcribe a single clip with speaker context"""
        
        # Enhanced prompt with speaker context
        if(speaker_context['speaker_change_flag']):
            if(speaker_context["previous_speaker"]=="DOC"):
                current_speaker = "patient"
            else:
                current_speaker = "doctor"
        else:
            current_speaker = speaker_context["previous_speaker"]

        if(clip_index!=0):
            clip_context_prompt = f""" 
            Please note that this is an audio clip of a longer conversation.
            
            IMPORTANT SPEAKER CONTEXT FOR THIS CLIP:
            
            - The likely first speaker in this clip is: {current_speaker}. However, this might be incorrect. Please identify the speaker based on the conversation content.

            """
            print(f"Clip {clip_index}: Using context - Previous speaker: {speaker_context['previous_speaker']}, Current likely speaker: {current_speaker}")

        context_aware_prompt = f"""
        {self.detailed_transcription_prompt}    
        {clip_context_prompt if clip_index!=0 else ""}
        
        """
        
        duration = self.get_audio_duration(clip_path)
        #formatted_prompt = context_aware_prompt.format(duration=duration)
        
        print(f"Transcribing clip {clip_index + 1} with context-aware prompt...")
        return await self.transcribe_with_retry(
            clip_path, context_aware_prompt, is_segmentation=False
        )

    async def process_single_file(self, audio_path: str, output_path: str, max_clip_duration: int) -> bool:
        """Process a single audio file with appropriate method based on duration"""
        duration = self.get_audio_duration(audio_path)
        
        print(f"The current max clip duration setting is {max_clip_duration} seconds.")

        if duration > max_clip_duration:  # 2 minutes threshold for long audio processing
            print(f"Long audio detected ({duration:.2f}s), using segmentation approach...")
            transcript = await self.transcribe_long_audio(audio_path,max_clip_duration)
        else:
            print(f"Short audio detected ({duration:.2f}s), using direct approach...")
            transcript = await self.transcribe_with_retry(
                audio_path, self.detailed_transcription_prompt, is_segmentation=False
            )
        # predifined error message from LLM generated response
        ErrorMessage = "Transcription failed after"
        if ErrorMessage in  transcript:
            print(f"\n✗ Transcription failed for {audio_path}.\n")
            return False    
        else:
        # Save transcript
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            return True


# ------------------------------------------------------------
# File Processing
# ------------------------------------------------------------

async def process_file_async(transcriber: EnhancedGeminiTranscriber, input_path: Path, input_dir: Path, output_dir: Path, max_clip_duration: int) -> bool:
    """Process a single file"""
    try:
        output_path = output_dir / f"{input_path.stem}.txt"
        
        success = await transcriber.process_single_file(str(input_path), str(output_path), max_clip_duration)
        
        if success:
            # Move to processed directory
            processed_dir = input_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            shutil.move(str(input_path), str(processed_dir / input_path.name))
            print(f"\n✓ Saved transcript: {output_path}\n")
            return True
        else:
            return False

    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {e}")
        return False

async def process_all_files_async(transcriber: EnhancedGeminiTranscriber, audio_files: list, input_dir: Path, output_dir: Path, max_clip_duration: int) -> int:
    """Process all files sequentially"""
    success_count = 0
    
    for audio_file in audio_files:
        if await process_file_async(transcriber, audio_file, input_dir, output_dir, max_clip_duration):
            success_count += 1
        await asyncio.sleep(30)  # Rate limiting
    
    return success_count


# ------------------------------------------------------------
# Main Program
# ------------------------------------------------------------

async def main_async():
    """Async main function"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Enhanced Gemini Transcription for Long Audio")
    parser.add_argument("--input-dir", default=os.getenv("DEFAULT_INPUT_DIR", "datasets/input"))
    parser.add_argument("--output-dir", default=os.getenv("DEFAULT_OUTPUT_DIR", "datasets/output/transcripts"))
    parser.add_argument("--api-key", default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--max-clip-duration", type=int, default=300, help="Maximum clip duration in seconds (default: 120)")
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

    # Find audio files
    audio_files = []
    for ext in args.file_extensions:
        audio_files.extend(list(input_dir.glob(f"*.{ext}")))
    
    if not audio_files:
        print("No audio files found.")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files.")

    transcriber = EnhancedGeminiTranscriber(api_key=args.api_key, model_name=args.model)

    success = await process_all_files_async(transcriber, audio_files, input_dir, output_dir, max_clip_duration=args.max_clip_duration)

    print("\n" + "=" * 40)
    print(f"Completed {success}/{len(audio_files)} files")
    print("=" * 40)

def main():
    """Run the async main function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()