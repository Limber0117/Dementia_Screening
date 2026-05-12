"""
Temporal Feature Calculator for Dementia Evaluation System.

Computes fluency and timing features from transcripts using timestamps.
These features are calculated locally without requiring LLM calls.
"""

import logging
from typing import List, Optional
import statistics

from models import TemporalFeatures, ParsedTranscript, Utterance

logger = logging.getLogger(__name__)


class TemporalFeatureCalculator:
    """
    Calculates temporal/fluency features from parsed transcripts.
    
    Features computed:
    - Response latency (time from doctor end to patient start)
    - Speaking rate (words per second)
    - Inter-word pause detection
    - Utterance patterns
    - Between-utterance gaps
    """
    
    # Thresholds
    LONG_PAUSE_THRESHOLD = 1  # seconds - pause considered "long"
    INCOMPLETE_SENTENCE_MARKERS = ['a', 'the', 'is', 'are', 'was', 'were', 
                                    'and', 'but', 'or', 'so', 'to', 'in', 'it']
    
    def __init__(self, long_pause_threshold: float = 0.5):
        """
        Initialize calculator.
        
        Args:
            long_pause_threshold: Seconds above which a pause is considered "long"
        """
        self.long_pause_threshold = long_pause_threshold
    
    def calculate_response_latencies(
        self, 
        transcript: ParsedTranscript
    ) -> List[float]:
        """
        Calculate response latencies from doctor to patient.
        
        Returns list of latency values in seconds.
        """
        latencies = []
        
        for i, utterance in enumerate(transcript.utterances):
            if utterance.speaker == 'PAT' and i > 0:
                prev = transcript.utterances[i-1]
                if prev.speaker == 'DOC':
                    latency = utterance.start_time - prev.end_time
                    if latency >= 0:  # Sanity check
                        latencies.append(latency)
        
        return latencies
    
    def calculate_speaking_rates(
        self, 
        utterances: List[Utterance]
    ) -> List[float]:
        """
        Calculate speaking rates for each utterance.
        
        Returns list of rates in words/second.
        """
        rates = []
        
        for u in utterances:
            if u.duration > 0 and u.word_count > 0:
                rate = u.word_count / u.duration
                rates.append(rate)
        
        return rates
    
    def count_long_pauses(
        self, 
        utterances: List[Utterance]
    ) -> tuple:
        """
        Count long pauses between words within utterances.
        
        Returns:
            (total_long_pauses, all_gaps, max_gap)
        """
        all_gaps = []
        long_pause_count = 0
        max_gap = 0.0
        
        for u in utterances:
            gaps = u.get_inter_word_gaps()
            all_gaps.extend(gaps)
            
            for gap in gaps:
                if gap > self.long_pause_threshold:
                    long_pause_count += 1
                if gap > max_gap:
                    max_gap = gap
        
        return long_pause_count, all_gaps, max_gap
    
    def detect_incomplete_utterances(
        self, 
        utterances: List[Utterance]
    ) -> int:
        """
        Detect utterances that appear incomplete (end mid-sentence).
        
        Returns count of incomplete utterances.
        """
        incomplete_count = 0
        
        for u in utterances:
            text = u.text.strip()
            
            # Check if ends without proper punctuation
            if not text:
                continue
                
            # Ends with incomplete word/article
            last_word = text.split()[-1].lower().rstrip('.,!?')
            if last_word in self.INCOMPLETE_SENTENCE_MARKERS:
                incomplete_count += 1
                continue
            
            # No ending punctuation and short utterance
            if not text[-1] in '.!?' and len(text.split()) < 3:
                incomplete_count += 1
        
        return incomplete_count
    
    def calculate_between_utterance_gaps(
        self, 
        utterances: List[Utterance]
    ) -> List[float]:
        """
        Calculate gaps between consecutive patient utterances.
        
        Returns list of gap durations in seconds.
        """
        gaps = []
        
        patient_utterances = [u for u in utterances if u.speaker == 'PAT']
        
        for i in range(1, len(patient_utterances)):
            prev = patient_utterances[i-1]
            curr = patient_utterances[i]
            
            gap = curr.start_time - prev.end_time
            if gap >= 0:  # Sanity check
                gaps.append(gap)
        
        return gaps
    
    def calculate(self, transcript: ParsedTranscript) -> TemporalFeatures:
        """
        Calculate all temporal features for a transcript.
        
        Args:
            transcript: Parsed transcript with timestamps
            
        Returns:
            TemporalFeatures object with all computed metrics
        """
        features = TemporalFeatures(pid=transcript.pid)
        
        patient_utterances = transcript.patient_utterances
        
        if not patient_utterances:
            logger.warning(f"No patient utterances found for {transcript.pid}")
            return features
        
        # Response latencies
        latencies = self.calculate_response_latencies(transcript)
        if latencies:
            features.avg_response_latency = statistics.mean(latencies)
            features.max_response_latency = max(latencies)
        
        # Speaking rates
        rates = self.calculate_speaking_rates(patient_utterances)
        if rates:
            features.avg_speaking_rate = statistics.mean(rates)
            features.min_speaking_rate = min(rates)
            features.max_speaking_rate = max(rates)
        
        # Long pauses
        long_pause_count, all_gaps, max_gap = self.count_long_pauses(patient_utterances)
        features.total_long_pauses = long_pause_count
        features.max_inter_word_gap = max_gap
        if all_gaps:
            features.avg_inter_word_gap = statistics.mean(all_gaps)
        
        # Utterance patterns
        features.total_patient_utterances = len(patient_utterances)
        word_counts = [u.word_count for u in patient_utterances]
        if word_counts:
            features.avg_utterance_length = statistics.mean(word_counts)
        features.incomplete_utterances = self.detect_incomplete_utterances(patient_utterances)
        
        # Between-utterance gaps
        between_gaps = self.calculate_between_utterance_gaps(transcript.utterances)
        if between_gaps:
            features.avg_between_utterance_gap = statistics.mean(between_gaps)
            features.max_between_utterance_gap = max(between_gaps)
        
        # Totals
        features.total_patient_words = transcript.total_patient_words
        features.total_patient_speaking_time = sum(u.duration for u in patient_utterances)
        
        return features



