"""
Data models for Dementia Evaluation System.

This module defines the data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Optional, List,Dict, Any
from datetime import datetime
from enum import Enum
import os
from dotenv import load_dotenv
from acoustic_zscore_analyzer import DementiaAcousticAnalyzer

import re



# Load the .env file first
load_dotenv()

is_debug = os.getenv("DEBUG", "").lower() == "true"
use_detailed_transcript = os.getenv("DETAILEDTRANSCRIPT","True").lower()=="true"
feature_amount = int(os.getenv("FEATURE_AMOUNT","20"))

class DiagnosisResult(Enum):
    """Possible diagnosis results from the evaluation."""
    CONTROL = "Control"
    IMPAIRMENT = "Impairment"
    UNKNOWN = "Unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "DiagnosisResult":
        """Convert string to DiagnosisResult, handling various formats."""
        if not value:
            return cls.UNKNOWN
        
        value_lower = str(value).lower().strip()
        
        if value_lower in ["healthy", "health", "normal", "control", "hc"]:
            return cls.CONTROL
        elif value_lower in ["impairment","mci", "mild cognitive impairment","dementia", "ad", "alzheimer", "alzheimer's", "svppa", "lvppa", "ppa-nos", "nfappa", "memory", "probablead", "vascular","possiblead"]:
            return cls.IMPAIRMENT
        else:
            return cls.UNKNOWN
        
    @classmethod
    def get_binary_label(cls, value: "DiagnosisResult") -> Optional[int]:
        """
        Convert diagnosis to binary label for metrics calculation.
        Healthy = 0 (Negative), MCI/Dementia = 1 (Positive)
        """
        if value == cls.CONTROL:
            return 0
        elif value == cls.IMPAIRMENT:
            return 1
        return None

@dataclass
class PatientInfo:
    """Patient demographic and metadata information."""
    pid: str
    age: Optional[int] = None
    gender: Optional[str] = None
    language: Optional[str] = None
    topic: Optional[str] = None
    ground_truth_diagnosis: Optional[str] = None
    mmse_score: Optional[float] = None
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PatientInfo":
        """Create PatientInfo from a DataFrame row or dictionary."""
        # Handle age - might be empty, float, or string
        age = None
        try:
            age_val = row.get("age")
            if age_val is not None and str(age_val).strip():
                age = int(float(str(age_val)))
        except (ValueError, TypeError):
            pass
        
        # Handle MMSE score, not useful for evaluation. This is the ground-truth for regression tasks.
        
        mmse = None
        try:
            mmse_val = row.get("MMSE")
            if mmse_val is not None and str(mmse_val).strip():
                mmse = float(str(mmse_val))
        except (ValueError, TypeError):
            pass
        
        
        return cls(
            pid=str(row.get("PID", "")),
            age=age,
            gender=str(row.get("gender", "")) if row.get("gender") else "UNKNOWN",
            language=str(row.get("language", "")) if row.get("language") else "UNKNOWN",
            topic=str(row.get("topic", "")) if row.get("topic") else "UNKNOWN",
            ground_truth_diagnosis=str(row.get("diagnosis", "")) if row.get("diagnosis") else "UNKNOWN",
            mmse_score=mmse,
        )


@dataclass
class TranscriptData:
    """Transcript data from patient conversation."""
    pid: str
    raw_text: str
    patient_utterances: list[str] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    
    @classmethod
    def from_file(cls, pid: str, file_path: str) -> "TranscriptData":
        """Load transcript from a file."""
        # the returned patient utterances will include all lines, including non-patient lines and timestamps.
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except (FileNotFoundError, IOError):
            return cls(pid=pid, raw_text="")
        
        # Parse patient utterances (lines starting with PAT:)
        patient_utterances = []
        pure_utterances= [] #without timestamps and indicators, e.g., DOC, PAT
        pure_transcript = []
        lines = raw_text.strip().split("\n")
        
        for line in lines:
            # Extract patient speech from formatted transcript
            patient_utterances.append(line.strip())
            if "PAT:" in line:
                # Extract the text between PAT: and {{
                start = line.index("PAT:") + 4
            elif "DOC:" in line:
                start = line.index("DOC:") + 4
            try:
                end = line.index("{{") if "{{" in line else len(line)
                utterance = line[start:end].strip()
                tend = line.index("{{") if "{{" in line else len(line)
                pure_transcript.append(line[0:tend].strip())
                pure_utterances.append(utterance)
            except (ValueError, IndexError):
                pass

        # Calculate statistics
        all_text = " ".join(pure_utterances)
        words = all_text.split()
        
        if use_detailed_transcript:

            return cls(
                pid=pid,
                raw_text=raw_text,
                patient_utterances=patient_utterances,
                word_count=len(words),
                sentence_count=len(patient_utterances),
            )
        else:

            return cls(
                pid=pid,
                raw_text=raw_text,
                patient_utterances=pure_transcript,
                word_count=len(words),
                sentence_count=len(patient_utterances),
            )

@dataclass
class AcousticFeatures:
    """Acoustic features extracted from patient audio."""
    pid: str
    features: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "AcousticFeatures":
        """Create AcousticFeatures from a DataFrame row."""
        pid = str(row.get("participant_id", ""))
        
        # Store all features except the ID column
        features = {
            k: v for k, v in row.items() 
            if k != "participant_id" and k!="filename" and v is not None
        }
        
        return cls(pid=pid, features=features)
    
    def get_summary(self, max_features: int = feature_amount) -> Dict[str, Any]:
        """Get a summary of the most relevant acoustic features."""
       
        # Prioritize certain feature categories

        analyser = DementiaAcousticAnalyzer()
        priority_keywords = analyser.get_important_features()
        if is_debug:
            print(f"The retrieved priority keyfeatures are:{priority_keywords}")

        summary = {}
        
        keys = list(self.features.keys())
        
        # First, add priority features
        for kw in priority_keywords:
            if kw in keys:
                summary[kw] = format(float(self.features[kw]),".4f")
                if len(summary) >= max_features:
                    break

        if (len(summary)< max_features):
            # Fill remaining with other features
            for key, value in self.features.items():
                if key not in summary:
                    summary[key] = format(float(value),".4f")
                    if len(summary) >= max_features:
                        break
        if is_debug:
            print(f" The current selected features are: {summary}")

        return summary


@dataclass
class EvaluationResult:
    """Result of a dementia evaluation."""
    pid: str
    prediction: DiagnosisResult
    explanation: str
    impairment_conf: Optional[float] = None
    control_conf: Optional[float] = None
    model_name: str = ""
    provider: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None
    variant: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            "PID": self.pid,
            "Prediction": self.prediction.value,
            "Explanation": self.explanation,
            "impairment_conf": self.impairment_conf,
            "control_conf": self.control_conf,
            "Model": self.model_name,
            "Provider": self.provider,
            "Timestamp": self.timestamp.isoformat(),
            "ProcessingTime": self.processing_time_seconds,
            "Error": self.error,
        }


@dataclass 
class PatientEvaluationInput:
    """Complete input data for evaluating a patient."""
    patient_info: PatientInfo
    transcript: TranscriptData
    acoustic_features: AcousticFeatures
    
    def format_for_llm(self) -> tuple[str, str, str]:
        """Format all patient data for LLM prompt."""
        Demographic_sections = []
        
        # Patient information section
        info_lines = []
        if self.patient_info.age:
            info_lines.append(f"Age: {self.patient_info.age if self.patient_info.age else 'UNKNOWN'}")
        if self.patient_info.gender:
            info_lines.append(f"Gender: {self.patient_info.gender if self.patient_info.gender else 'UNKNOWN'}")
        if self.patient_info.language:
            info_lines.append(f"Language spoken in the conversation: {self.patient_info.language if self.patient_info.language else 'UNKNOWN'}")
        if self.patient_info.topic:
            info_lines.append(f"Topic discussed in the conversation: {self.patient_info.topic if self.patient_info.topic else 'UNKNOWN'}")
        
        Demographic_sections.append("\n\n".join(info_lines))
        
        # Transcript section
        Transcript_sections = []
        if self.transcript.patient_utterances:
            transcript_text = "\n\n".join(
                f"- {utt}" for utt in self.transcript.patient_utterances
            )
            Transcript_sections.append(f"{transcript_text}")

        # Acoustic features section
        Acoustic_sections = []
        if self.acoustic_features.features:
            feature_summary = self.acoustic_features.get_summary()
            feature_lines = [
                f"- {k}: {v}" for k, v in feature_summary.items()
            ]

            Acoustic_sections.append(
                "## Acoustic Features (from patient's speech analysis)\n" + 
                "\n\n".join(feature_lines)
            )
        
        if is_debug:
            amount = len(Acoustic_sections)
            print(f"- A total of {amount} patients are extracted with features. ")
        
        return (Demographic_sections, Transcript_sections, Acoustic_sections)

##----------- used for transcript analysis ------------------------
@dataclass
class WordTimestamp:
    """Represents a single word with its timestamp."""
    word: str
    start_time: float  # in seconds
    
    @classmethod
    def parse(cls, word_str: str) -> 'WordTimestamp':
        """Parse a word timestamp from format: (word|MM:SS.mmm)"""
        match = re.match(r'\(([^|]+)\|(\d+):(\d+\.\d+)\)', word_str)
        if match:
            word = match.group(1)
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            start_time = minutes * 60 + seconds
            return cls(word=word, start_time=start_time)
        return None


@dataclass
class Utterance:
    """Represents a single utterance (sentence) with timestamps."""
    speaker: str  # 'DOC' or 'PAT'
    text: str
    start_time: float  # in seconds
    end_time: float  # in seconds
    words: List[WordTimestamp] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Duration of the utterance in seconds."""
        return self.end_time - self.start_time
    
    @property
    def word_count(self) -> int:
        """Number of words in the utterance."""
        return len(self.words)
    
    @property
    def speaking_rate(self) -> float:
        """Words per second."""
        if self.duration > 0:
            return self.word_count / self.duration
        return 0.0
    
    def get_inter_word_gaps(self) -> List[float]:
        """Calculate gaps between consecutive words."""
        gaps = []
        for i in range(1, len(self.words)):
            gap = self.words[i].start_time - self.words[i-1].start_time
            gaps.append(gap)
        return gaps


@dataclass
class ParsedTranscript:
    """Complete parsed transcript with all utterances."""
    pid: str
    utterances: List[Utterance] = field(default_factory=list)
    raw_text: str = ""
    
    @property
    def patient_utterances(self) -> List[Utterance]:
        """Get only patient utterances."""
        return [u for u in self.utterances if u.speaker == 'PAT']
    
    @property
    def doctor_utterances(self) -> List[Utterance]:
        """Get only doctor utterances."""
        return [u for u in self.utterances if u.speaker == 'DOC']
    
    @property
    def patient_text(self) -> str:
        """Get concatenated patient text."""
        return ' '.join(u.text for u in self.patient_utterances)
    
    @property
    def total_patient_words(self) -> int:
        """Total words spoken by patient."""
        return sum(u.word_count for u in self.patient_utterances)
    
    @property
    def total_duration(self) -> float:
        """Total transcript duration."""
        if self.utterances:
            return self.utterances[-1].end_time - self.utterances[0].start_time
        return 0.0


@dataclass
class FeatureScore:
    """Score for a single feature with metadata."""
    feature_name: str
    score: float  # 1-5 for ratings, float for computed metrics
    confidence: float  # 0-1 confidence level
    summary: str = ""  # LLM-generated summary (max 200 words)
    is_computed: bool = False  # True if computed locally, False if LLM-rated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            f"{self.feature_name}_score": self.score,
            f"{self.feature_name}_confidence": self.confidence,
            f"{self.feature_name}_summary": self.summary
        }


@dataclass
class TemporalFeatures:
    """Computed temporal/fluency features (no LLM needed)."""
    pid: str
    
    # Response latency
    avg_response_latency: float = 0.0  # seconds
    max_response_latency: float = 0.0
    
    # Speaking rate
    avg_speaking_rate: float = 0.0  # words/second
    min_speaking_rate: float = 0.0
    max_speaking_rate: float = 0.0
    
    # Pause detection
    total_long_pauses: int = 0  # pauses > 0.5s between words
    avg_inter_word_gap: float = 0.0
    max_inter_word_gap: float = 0.0
    
    # Utterance patterns
    total_patient_utterances: int = 0
    avg_utterance_length: float = 0.0  # words per utterance
    incomplete_utterances: int = 0  # utterances ending mid-sentence
    
    # Between-utterance gaps
    avg_between_utterance_gap: float = 0.0
    max_between_utterance_gap: float = 0.0
    
    # Total metrics
    total_patient_words: int = 0
    total_patient_speaking_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "temporal_avg_response_latency": self.avg_response_latency,
            "temporal_max_response_latency": self.max_response_latency,
            "temporal_avg_speaking_rate": self.avg_speaking_rate,
            "temporal_min_speaking_rate": self.min_speaking_rate,
            "temporal_max_speaking_rate": self.max_speaking_rate,
            "temporal_total_long_pauses": self.total_long_pauses,
            "temporal_avg_inter_word_gap": self.avg_inter_word_gap,
            "temporal_max_inter_word_gap": self.max_inter_word_gap,
            "temporal_total_patient_utterances": self.total_patient_utterances,
            "temporal_avg_utterance_length": self.avg_utterance_length,
            "temporal_incomplete_utterances": self.incomplete_utterances,
            "temporal_avg_between_utterance_gap": self.avg_between_utterance_gap,
            "temporal_max_between_utterance_gap": self.max_between_utterance_gap,
            "temporal_total_patient_words": self.total_patient_words,
            "temporal_total_patient_speaking_time": self.total_patient_speaking_time,
        }


@dataclass 
class LexicalFeatures:
    """Lexical richness and diversity features (computed + LLM)."""
    # Computed metrics
    type_token_ratio: float = 0.0
    unique_words: int = 0
    total_words: int = 0
    advanced_vocab_count: int = 0  # words outside top 2000
    advanced_vocab_ratio: float = 0.0
    
    # LLM-rated features
    vocabulary_range: Optional[FeatureScore] = None
    lexical_accuracy: Optional[FeatureScore] = None
    specificity: Optional[FeatureScore] = None
    advanced_vocabulary: Optional[FeatureScore] = None


@dataclass
class SyntacticFeatures:
    """Syntactic structure features (LLM-rated)."""
    grammar_complexity: Optional[FeatureScore] = None
    structure_variety: Optional[FeatureScore] = None
    grammar_correctness: Optional[FeatureScore] = None


@dataclass
class PragmaticFeatures:
    """Pragmatic competence features (LLM-rated)."""
    referential_clarity: Optional[FeatureScore] = None
    state_of_mind_language: Optional[FeatureScore] = None
    implausible_details: Optional[FeatureScore] = None


@dataclass
class SemanticFeatures:
    """Semantic coherence and cohesion features (LLM-rated)."""
    topic_management: Optional[FeatureScore] = None
    logical_organization: Optional[FeatureScore] = None
    cohesion: Optional[FeatureScore] = None
    cause_and_effect: Optional[FeatureScore] = None
    repetition: Optional[FeatureScore] = None
    information_prioritization: Optional[FeatureScore] = None


@dataclass
class SemanticEvaluationResult:
    """Complete evaluation result for a patient."""
    pid: str
    
    # Feature categories
    temporal: Optional[TemporalFeatures] = None
    lexical: Optional[LexicalFeatures] = None
    syntactic: Optional[SyntacticFeatures] = None
    pragmatic: Optional[PragmaticFeatures] = None
    semantic: Optional[SemanticFeatures] = None
    prediction:str=None
    
    # Metadata
    model_name: str = ""
    provider: str = ""
    processing_time_seconds: float = 0.0
    error: Optional[str] = None
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        result = {"pid": self.pid}
        
        # Add temporal features
        if self.temporal:
            result.update(self.temporal.to_dict())
        
        # Add lexical computed features
        if self.lexical:
            result["lexical_type_token_ratio"] = self.lexical.type_token_ratio
            result["lexical_unique_words"] = self.lexical.unique_words
            result["lexical_total_words"] = self.lexical.total_words
            result["lexical_advanced_vocab_count"] = self.lexical.advanced_vocab_count
            result["lexical_advanced_vocab_ratio"] = self.lexical.advanced_vocab_ratio
            
            # Add LLM-rated lexical features
            for feat in [self.lexical.vocabulary_range, self.lexical.lexical_accuracy,
                        self.lexical.specificity, self.lexical.advanced_vocabulary]:
                if feat:
                    result.update(feat.to_dict())
        
        # Add syntactic features
        if self.syntactic:
            for feat in [self.syntactic.grammar_complexity, 
                        self.syntactic.structure_variety,
                        self.syntactic.grammar_correctness]:
                if feat:
                    result.update(feat.to_dict())
        
        # Add pragmatic features
        if self.pragmatic:
            for feat in [self.pragmatic.referential_clarity,
                        self.pragmatic.state_of_mind_language,
                        self.pragmatic.implausible_details]:
                if feat:
                    result.update(feat.to_dict())
        
        # Add semantic features
        if self.semantic:
            for feat in [self.semantic.topic_management,
                        self.semantic.logical_organization,
                        self.semantic.cohesion,
                        self.semantic.cause_and_effect,
                        self.semantic.repetition,
                        self.semantic.information_prioritization]:
                if feat:
                    result.update(feat.to_dict())
        
        # Add metadata
        result["model_name"] = self.model_name
        result["provider"] = self.provider
        result["processing_time_seconds"] = self.processing_time_seconds
        result["error"] = self.error or ""
        
        return result



