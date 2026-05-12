"""
Lexical Feature Calculator for Dementia Evaluation System.

Computes lexical diversity metrics from transcripts.
These are computed locally without requiring LLM calls.
"""

import re
import logging
from typing import Set, List, Dict
from collections import Counter

from models import ParsedTranscript, LexicalFeatures

logger = logging.getLogger(__name__)


# Top 2000 most common English words (subset for demonstration)
# In production, use a full frequency list from COCA or similar
COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having', 'does',
    'did', 'doing', 'done', 'am', 'got', 'getting', 'here', 'there', 'where', 'why',
    'very', 'more', 'much', 'too', 'really', 'right', 'still', 'already', 'again', 'always',
    'never', 'ever', 'maybe', 'yes', 'no', 'yeah', 'ok', 'okay', 'oh', 'um',
    'uh', 'ah', 'well', 'just', 'like', 'thing', 'things', 'stuff', 'something', 'nothing',
    'everything', 'anything', 'someone', 'everyone', 'anyone', 'nobody', 'everybody', 'somebody',
    'put', 'let', 'try', 'keep', 'find', 'help', 'tell', 'ask', 'need', 'feel',
    'become', 'leave', 'call', 'should', 'may', 'might', 'must', 'shall', 'ought',
    'man', 'woman', 'child', 'kid', 'boy', 'girl', 'mother', 'father', 'mom', 'dad',
    'house', 'home', 'room', 'door', 'window', 'floor', 'wall', 'table', 'chair',
    'water', 'food', 'hand', 'head', 'face', 'eye', 'mouth', 'ear', 'body', 'foot',
    'big', 'small', 'little', 'old', 'young', 'long', 'short', 'high', 'low', 'great',
    'same', 'different', 'next', 'last', 'few', 'many', 'each', 'every', 'both', 'own',
    'down', 'off', 'away', 'around', 'through', 'between', 'under', 'before', 'after',
    'while', 'during', 'without', 'within', 'against', 'along', 'across', 'toward',
    'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred',
    'side', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question',
    'government', 'number', 'night', 'point', 'world', 'state', 'family', 'country',
    'problem', 'service', 'fact', 'group', 'percent', 'hand', 'school', 'lot',
    'she', 'her', 'hers', 'herself', 'he', 'him', 'his', 'himself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'i', 'me', 'my', 'mine',
    'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'sink', 'kitchen', 'cookie', 'jar', 'stool', 'climb', 'fall', 'reach', 'stand',
    'sit', 'run', 'walk', 'open', 'close', 'turn', 'hold', 'pick', 'together'
}


class LexicalCalculator:
    """
    Calculates lexical diversity metrics from transcripts.
    
    Metrics computed:
    - Type-Token Ratio (TTR)
    - Unique word count
    - Total word count  
    - Advanced vocabulary count (words outside common list)
    - Advanced vocabulary ratio
    """
    
    def __init__(self, common_words: Set[str] = None):
        """
        Initialize calculator.
        
        Args:
            common_words: Set of common words. Uses default if not provided.
        """
        self.common_words = common_words or COMMON_WORDS
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words.
        
        Args:
            text: Input text
            
        Returns:
            List of lowercase word tokens
        """
        # Remove punctuation and convert to lowercase
        text = text.lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Split on whitespace
        words = text.split()
        # Filter out empty strings and very short tokens
        words = [w for w in words if len(w) > 0]
        return words
    
    def calculate_ttr(self, words: List[str]) -> float:
        """
        Calculate Type-Token Ratio.
        
        TTR = unique_words / total_words
        
        Args:
            words: List of word tokens
            
        Returns:
            TTR value (0-1)
        """
        if not words:
            return 0.0
        
        unique = len(set(words))
        total = len(words)
        
        return unique / total
    
    def calculate_mattr(self, words: List[str], window_size: int = 50) -> float:
        """
        Calculate Moving Average Type-Token Ratio.
        
        More robust for varying text lengths than simple TTR.
        
        Args:
            words: List of word tokens
            window_size: Size of sliding window
            
        Returns:
            MATTR value (0-1)
        """
        if len(words) < window_size:
            return self.calculate_ttr(words)
        
        ttrs = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            ttrs.append(self.calculate_ttr(window))
        
        return sum(ttrs) / len(ttrs) if ttrs else 0.0
    
    def count_advanced_vocabulary(self, words: List[str]) -> tuple:
        """
        Count words outside the common vocabulary list.
        
        Args:
            words: List of word tokens
            
        Returns:
            (count, ratio) of advanced vocabulary
        """
        if not words:
            return 0, 0.0
        
        advanced_count = 0
        for word in words:
            if word not in self.common_words:
                advanced_count += 1
        
        ratio = advanced_count / len(words)
        
        return advanced_count, ratio
    
    def get_word_frequency_distribution(self, words: List[str]) -> Dict[str, int]:
        """
        Get frequency distribution of words.
        
        Args:
            words: List of word tokens
            
        Returns:
            Dictionary mapping words to their counts
        """
        return dict(Counter(words))
    
    def get_hapax_legomena(self, words: List[str]) -> List[str]:
        """
        Get words that appear only once (hapax legomena).
        
        Args:
            words: List of word tokens
            
        Returns:
            List of words appearing exactly once
        """
        freq = Counter(words)
        return [word for word, count in freq.items() if count == 1]
    
    def calculate(self, transcript: ParsedTranscript) -> LexicalFeatures:
        """
        Calculate all lexical features for a transcript.
        
        Args:
            transcript: Parsed transcript
            
        Returns:
            LexicalFeatures object with computed metrics
        """
        features = LexicalFeatures()
        
        # Get patient text only
        patient_text = transcript.patient_text
        words = self.tokenize(patient_text)
        
        if not words:
            logger.warning(f"No words found in patient text for {transcript.pid}")
            return features
        
        # Basic counts
        features.total_words = len(words)
        features.unique_words = len(set(words))
        
        # Type-Token Ratio
        features.type_token_ratio = self.calculate_ttr(words)
        
        # Advanced vocabulary
        adv_count, adv_ratio = self.count_advanced_vocabulary(words)
        features.advanced_vocab_count = adv_count
        features.advanced_vocab_ratio = adv_ratio
        
        return features




# Test
if __name__ == "__main__":
    from transcript_parser import TranscriptParser
    
    sample = """[00:18.500 - 00:19.400] PAT: It's a house. {{(It's|00:18.500) (a|00:18.700) (house.|00:19.400)}}
[00:21.600 - 00:25.000] PAT: The kid trying to climb that, look at the cookie jar. {{(The|00:21.600) (kid|00:21.900) (trying|00:22.200) (to|00:22.300) (climb|00:22.900) (that,|00:23.300) (look|00:24.000) (at|00:24.100) (the|00:24.200) (cookie|00:24.500) (jar.|00:25.000)}}
[00:25.000 - 00:27.900] PAT: And you got mom over here, she has that. {{(And|00:25.000) (you|00:25.200) (got|00:25.400) (mom|00:25.700) (over|00:25.900) (here,|00:26.200) (she|00:26.500) (has|00:26.800) (that.|00:27.900)}}
[00:27.900 - 00:31.800] PAT: And she has all this work in the sink and she's got it some on the floor. {{(And|00:27.900) (she|00:28.100) (has|00:28.300) (all|00:28.500) (this|00:28.700) (work|00:29.000) (in|00:29.100) (the|00:29.200) (sink|00:29.700) (and|00:29.800) (she's|00:30.000) (got|00:30.200) (it|00:30.300) (some|00:30.600) (on|00:30.700) (the|00:30.800) (floor.|00:31.800)}}
[00:32.200 - 00:34.700] PAT: And she is got this one, this kid over here over there. {{(And|00:32.200) (she|00:32.400) (is|00:32.600) (got|00:32.800) (this|00:33.000) (one,|00:33.300) (this|00:33.500) (kid|00:33.700) (over|00:33.900) (here|00:34.100) (over|00:34.300) (there.|00:34.700)}}
[00:35.000 - 00:36.000] PAT: You know, there's cookie jar. {{(You|00:35.000) (know,|00:35.200) (there's|00:35.500) (cookie|00:35.700) (jar.|00:36.000)}}"""
    
    parser = TranscriptParser()
    transcript = parser.parse_text(sample, pid="test-001")
    
    calculator = LexicalCalculator()
    features = calculator.calculate(transcript)
    
    print("\n" + "="*60)
    print("LEXICAL FEATURES ANALYSIS")
    print("="*60)
    
    print(f"\nPatient text: {transcript.patient_text[:100]}...")
    
    print("\n--- Basic Counts ---")
    print(f"  Total words: {features.total_words}")
    print(f"  Unique words: {features.unique_words}")
    
    print("\n--- Lexical Diversity ---")
    print(f"  Type-Token Ratio: {features.type_token_ratio:.4f}")
    
    print("\n--- Advanced Vocabulary ---")
    print(f"  Advanced vocab count: {features.advanced_vocab_count}")
    print(f"  Advanced vocab ratio: {features.advanced_vocab_ratio:.4f}")
    
    # Show word frequency
    words = calculator.tokenize(transcript.patient_text)
    freq = calculator.get_word_frequency_distribution(words)
    print("\n--- Word Frequency (top 10) ---")
    for word, count in sorted(freq.items(), key=lambda x: -x[1])[:10]:
        print(f"  {word}: {count}")
    
    # Show hapax legomena
    hapax = calculator.get_hapax_legomena(words)
    print(f"\n--- Hapax Legomena (words appearing once): {len(hapax)} ---")
    print(f"  {hapax[:10]}...")
