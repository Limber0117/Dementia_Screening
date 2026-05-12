"""
Data loading utilities for Dementia Evaluation System.

This module handles loading data from CSV files and transcript files.
"""

import os
import logging
from typing import Generator, Optional, Dict, Any
import pandas as pd

from models import (
    PatientInfo, 
    TranscriptData, 
    AcousticFeatures, 
    PatientEvaluationInput
)
from config import DatasetConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and merging patient data from various sources."""
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the data loader.
        
        Args:
            config: Dataset configuration with file paths
        """
        self.config = config
        self._groundtruth_df: Optional[pd.DataFrame] = None
        self._features_df: Optional[pd.DataFrame] = None
        self._merged_data: Optional[pd.DataFrame] = None
    
    def _load_groundtruth(self) -> pd.DataFrame:
        """Load the groundtruth CSV file."""
        if self._groundtruth_df is None:
            try:
                self._groundtruth_df = pd.read_csv(
                    self.config.groundtruth_csv,
                    dtype=str  # Load all as strings initially
                )
                logger.info(
                    f"Loaded groundtruth with {len(self._groundtruth_df)} records"
                )
            except FileNotFoundError:
                logger.error(f"Groundtruth file not found: {self.config.groundtruth_csv}")
                self._groundtruth_df = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading groundtruth: {e}")
                self._groundtruth_df = pd.DataFrame()
        
        return self._groundtruth_df
    


    def _load_merged_data_pids(self, merged_data_file) -> pd.DataFrame:
        """Load the groundtruth CSV file."""
        if self._merged_data is None:
            try:
                self._merged_data = pd.read_csv(
                    merged_data_file,
                    dtype=str  # Load all as strings initially
                )
                logger.info(
                    f"Loaded merged data with {len(self._merged_data)} records"
                )
            except FileNotFoundError:
                logger.error(f"Merged file not found: {merged_data_file}")
                self._merged_data = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading merged data: {e}")
                self._merged_data = pd.DataFrame()
        
        if self._merged_data.empty or "PID" not in self._merged_data.columns:
            return []
        
        return self._merged_data["PID"].dropna().unique().tolist()


    def _load_features(self) -> pd.DataFrame:
        """Load the acoustic features CSV file."""
        if self._features_df is None:
            try:
                self._features_df = pd.read_csv(self.config.acoustic_features_csv)
                logger.info(
                    f"Loaded acoustic features with {len(self._features_df)} records"
                )
            except FileNotFoundError:
                logger.error(
                    f"Acoustic features file not found: "
                    f"{self.config.acoustic_features_csv}"
                )
                self._features_df = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading acoustic features: {e}")
                self._features_df = pd.DataFrame()
        
        return self._features_df
    
    def _load_transcript(self, pid: str) -> TranscriptData:
        """
        Load transcript for a specific patient.
        
        Args:
            pid: Patient ID
            
        Returns:
            TranscriptData object
        """
        transcript_path = os.path.join(
            self.config.transcripts_dir, 
            f"{pid}.txt"
        )
        
        if os.path.exists(transcript_path):
            return TranscriptData.from_file(pid, transcript_path)
        else:
            logger.warning(f"Transcript not found for {pid}: {transcript_path}")
            return TranscriptData(pid=pid, raw_text="")
    
    def get_patient_ids(self) -> list[str]:
        """
        Get all patient IDs from the groundtruth file.
        
        Returns:
            List of patient ID strings
        """
        df = self._load_groundtruth()
        
        if df.empty or "PID" not in df.columns:
            return []
        
        return df["PID"].dropna().unique().tolist()

    def get_valid_patient_ids(self) -> list[str]:
        """
        Get patient IDs that have BOTH acoustic features and transcripts available.
        
        Returns:
            List of patient ID strings with complete data
        """
        valid_pids = []
        
        # Get patient IDs from acoustic features
        features_df = self._load_features()
        if features_df.empty or "participant_id" not in features_df.columns:
            logger.warning("No acoustic features available")
            feature_pids = set()
        else:
            feature_pids = set(features_df["participant_id"].dropna().astype(str).unique())
        
        # Get patient IDs from transcripts folder
        transcript_pids = set()
        if os.path.isdir(self.config.transcripts_dir):
            for filename in os.listdir(self.config.transcripts_dir):
                if filename.endswith(".txt"):
                    pid = filename[:-4]  # Remove .txt extension
                    transcript_pids.add(pid)
        else:
            logger.warning(f"Transcripts directory not found: {self.config.transcripts_dir}")
        
        # Find intersection - patients with both features and transcripts
        valid_pids = feature_pids.intersection(transcript_pids)
        
        logger.info(
            f"Found {len(feature_pids)} patients with acoustic features, "
            f"{len(transcript_pids)} with transcripts, "
            f"{len(valid_pids)} with both"
        )
        
        return sorted(list(valid_pids))


    def get_patient_data(self, pid: str) -> Optional[PatientEvaluationInput]:
        """
        Get complete data for a single patient.
        
        Args:
            pid: Patient ID
            
        Returns:
            PatientEvaluationInput or None if essential data not found
        """
        groundtruth_df = self._load_groundtruth()
        features_df = self._load_features()
        
        # Get patient info from groundtruth (optional - may not exist)
        patient_info = PatientInfo(pid=pid)
        
        if not groundtruth_df.empty and "PID" in groundtruth_df.columns:
            patient_rows = groundtruth_df[groundtruth_df["PID"] == pid]
            if not patient_rows.empty:
                patient_row = patient_rows.iloc[0].to_dict()
                patient_info = PatientInfo.from_row(patient_row)
        
        # Get transcript (required)
        transcript = self._load_transcript(pid)
        if not transcript.raw_text:
            logger.warning(f"No transcript found for patient {pid}")
            return None
        
        # Get acoustic features (required)
        acoustic_features = AcousticFeatures(pid=pid)
        
        if not features_df.empty and "participant_id" in features_df.columns:
            # Convert participant_id to string for comparison
            features_df_copy = features_df.copy()
            features_df_copy["participant_id"] = features_df_copy["participant_id"].astype(str)
            feature_rows = features_df_copy[features_df_copy["participant_id"] == pid]
            
            if not feature_rows.empty:
                acoustic_features = AcousticFeatures.from_row(
                    feature_rows.iloc[0].to_dict()
                )
            else:
                logger.warning(f"No acoustic features found for patient {pid}")
                return None
        else:
            logger.warning(f"No acoustic features available for patient {pid}")
            return None
        
        return PatientEvaluationInput(
            patient_info=patient_info,
            transcript=transcript,
            acoustic_features=acoustic_features
        )
    
    def iter_patients(
        self, 
        patient_ids: Optional[list[str]] = None,
        require_both: bool = True
    ) -> Generator[PatientEvaluationInput, None, None]:
        """
        Iterate over patients, yielding complete data for each.
        
        Args:
            patient_ids: Optional list of specific patient IDs to process.
                        If None, processes all valid patients.
            require_both: If True (default), only include patients with both
                         acoustic features and transcripts.
                        
        Yields:
            PatientEvaluationInput for each patient
        """
        if patient_ids is None:
            if require_both:
                patient_ids = self.get_valid_patient_ids()
            else:
                patient_ids = self.get_patient_ids()
        
        for pid in patient_ids:
            patient_data = self.get_patient_data(pid)
            if patient_data is not None:
                yield patient_data
            else:
                logger.warning(f"Skipping patient {pid}: incomplete data")
    
    def get_groundtruth_diagnosis(self, pid: str) -> Optional[str]:
        """
        Get the ground truth diagnosis for a patient.
        
        Args:
            pid: Patient ID
            
        Returns:
            Diagnosis string or None
        """
        df = self._load_groundtruth()
        
        if df.empty or "PID" not in df.columns:
            return None
        
        patient_rows = df[df["PID"] == pid]
        
        if patient_rows.empty:
            return None
        
        groundtruth = None

        if patient_rows.iloc[0].get("diagnosis"):
            match patient_rows.iloc[0].get("diagnosis").lower():
                case "impairment" | "alzheimer's" | "mci" | "dementia" | "pick's" | "probablead" | "possiblead" | "memory" | "vascular" | "ppa-nos" | "nfappa" | "lvppa" | "svppa":
                    groundtruth="Impairment"
                case "control":
                    groundtruth="Control"
        return groundtruth


