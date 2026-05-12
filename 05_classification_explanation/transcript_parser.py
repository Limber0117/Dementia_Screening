"""
Transcript Parser for Dementia Evaluation System.

Parses transcripts with sentence-level and word-level timestamps.
Format: [MM:SS.mmm - MM:SS.mmm] SPEAKER: text {{(word|MM:SS.mmm) ...}}
"""

import re
import os
import logging
from typing import List, Optional

from models import (
    WordTimestamp, 
    Utterance, 
    ParsedTranscript
)

logger = logging.getLogger(__name__)


class TranscriptParser:
    """Parser for timestamped transcripts."""
    
    # Regex patterns
    UTTERANCE_PATTERN = re.compile(
        r'\[(\d+):(\d+\.\d+)\s*-\s*(\d+):(\d+\.\d+)\]\s*(\w+):\s*(.+?)(?:\s*\{\{(.+?)\}\})?$'
    )
    
    WORD_PATTERN = re.compile(r'\(([^|]+)\|(\d+):(\d+\.\d+)\)')
    
    def __init__(self):
        pass
    
    def parse_time(self, minutes: str, seconds: str) -> float:
        """Convert MM:SS.mmm to seconds."""
        return int(minutes) * 60 + float(seconds)
    
    def parse_word_timestamps(self, word_section: str) -> List[WordTimestamp]:
        """Parse word timestamps from the {{...}} section."""
        words = []
        
        if not word_section:
            return words
        
        for match in self.WORD_PATTERN.finditer(word_section):
            word = match.group(1)
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            start_time = minutes * 60 + seconds
            
            words.append(WordTimestamp(word=word, start_time=start_time))
        
        return words
    
    def parse_line(self, line: str) -> Optional[Utterance]:
        """Parse a single line of the transcript."""
        line = line.strip()
        
        if not line:
            return None
        
        match = self.UTTERANCE_PATTERN.match(line)
        
        if not match:
            logger.debug(f"Failed to parse line: {line[:50]}...")
            return None
        
        start_min, start_sec, end_min, end_sec, speaker, text, word_section = match.groups()
        
        start_time = self.parse_time(start_min, start_sec)
        end_time = self.parse_time(end_min, end_sec)
        
        # Clean text (remove word timestamps if accidentally included)
        text = text.strip()
        
        # Parse word timestamps
        words = self.parse_word_timestamps(word_section) if word_section else []
        
        return Utterance(
            speaker=speaker,
            text=text,
            start_time=start_time,
            end_time=end_time,
            words=words
        )
    
    def parse_file(self, filepath: str, pid: Optional[str] = None) -> ParsedTranscript:
        """
        Parse a transcript file.
        
        Args:
            filepath: Path to the transcript file
            pid: Patient ID (extracted from filename if not provided)
            
        Returns:
            ParsedTranscript object
        """
        if pid is None:
            # Extract PID from filename
            pid = os.path.splitext(os.path.basename(filepath))[0]
        
        utterances = []
        raw_lines = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_lines.append(line)
                    utterance = self.parse_line(line)
                    if utterance:
                        utterances.append(utterance)
        
        except FileNotFoundError:
            logger.error(f"Transcript file not found: {filepath}")
            return ParsedTranscript(pid=pid, raw_text="")
        except Exception as e:
            logger.error(f"Error parsing transcript {filepath}: {e}")
            return ParsedTranscript(pid=pid, raw_text="")
        
        return ParsedTranscript(
            pid=pid,
            utterances=utterances,
            raw_text=''.join(raw_lines)
        )
    
    def parse_text(self, text: str, pid: str = "unknown") -> ParsedTranscript:
        """
        Parse transcript from text string.
        
        Args:
            text: Raw transcript text
            pid: Patient ID
            
        Returns:
            ParsedTranscript object
        """
        utterances = []
        
        for line in text.split('\n'):
            utterance = self.parse_line(line)
            if utterance:
                utterances.append(utterance)
        
        return ParsedTranscript(
            pid=pid,
            utterances=utterances,
            raw_text=text
        )


def load_transcript(filepath: str, pid: Optional[str] = None) -> ParsedTranscript:
    """
    Convenience function to load and parse a transcript.
    
    Args:
        filepath: Path to transcript file
        pid: Optional patient ID
        
    Returns:
        ParsedTranscript object
    """
    parser = TranscriptParser()
    return parser.parse_file(filepath, pid)



