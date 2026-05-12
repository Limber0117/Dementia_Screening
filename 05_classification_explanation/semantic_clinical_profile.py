"""
semantic_clinical_profile.py

Updated Semantic Clinical Profiling with ML Bridge Integration
===============================================================

This module supports TWO modes:
1. ML Bridge mode (RECOMMENDED) - Uses pre-trained ensemble models from nested CV
2. Legacy z-score mode - Uses population-based z-score discretization

The ML Bridge mode is recommended because:
- Uses proper nested CV methodology (no data leakage)
- Inner CV optimizes threshold per fold
- Ensemble models capture complex feature interactions
- Categorical output is easier for LLMs to interpret

Usage:
    # The module automatically loads the appropriate fold's model
    # Set SEMANTIC_CV_FOLD environment variable or use set_current_fold()
    
    from semantic_clinical_profile import (
        set_current_fold,
        get_test_patient_ids,
        analyze_patient_for_llm,
    )
    
    set_current_fold(1)
    patient_ids = get_test_patient_ids()
    pre_diag, prompt = analyze_patient_for_llm(patient_features, pid)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Z-score thresholds (for legacy mode)
Z_DOMINANT = 1.0
CONF_DOMINANT = 0.6

# ML Bridge settings
USE_ML_SEMANTIC_BRIDGE = os.getenv("USE_ML_SEMANTIC_BRIDGE", "True").lower() == "true"
SEMANTIC_CV_FOLDS_DIR = os.path.join(
    os.getenv("RESULTS_DIR", "datasets/results"),
    "semantic_cv_folds"
)
CURRENT_SEMANTIC_FOLD = int(os.getenv("SEMANTIC_CV_FOLD", "1"))

# Legacy single model path (fallback)
ML_SEMANTIC_BRIDGE_PATH = os.getenv("ML_SEMANTIC_BRIDGE_PATH", "ml_semantic_bridge.pkl")

is_debug = os.getenv("DEBUG", "False").lower() == "true"

# Global bridge instance (lazy loaded)
_semantic_bridge_instance = None
_current_semantic_fold = CURRENT_SEMANTIC_FOLD


# =============================================================================
# FOLD MANAGEMENT
# =============================================================================

def set_current_fold(fold: int) -> None:
    """
    Set the current CV fold to use and reload the model.
    
    Args:
        fold: Fold number (1-5)
    """
    global _current_semantic_fold, _semantic_bridge_instance
    
    if fold < 1 or fold > 5:
        raise ValueError(f"Fold must be 1-5, got {fold}")
    
    _current_semantic_fold = fold
    _semantic_bridge_instance = None  # Reset to force reload
    
    # Immediately load the new model
    bridge = get_bridge()
    
    if bridge is not None:
        logger.info(f"Semantic Fold {fold}: Loaded model with threshold={bridge.classification_threshold}")
    else:
        logger.warning(f"Semantic Fold {fold}: Failed to load model")


def get_current_fold() -> int:
    """Get the current CV fold number."""
    return _current_semantic_fold


def get_bridge():
    """
    Get the ML Semantic Bridge instance for the current fold.
    
    Returns:
        MLSemanticBridge instance or None
    """
    global _semantic_bridge_instance
    
    if _semantic_bridge_instance is not None:
        return _semantic_bridge_instance
    
    if not USE_ML_SEMANTIC_BRIDGE:
        return None
    
    # Import MLSemanticBridge class
    try:
        from ml_semantic_bridge import MLSemanticBridge
    except ImportError:
        logger.warning("Could not import MLSemanticBridge class")
        return None
    
    # Try to load from CV folds directory first
    fold_model_path = os.path.join(
        SEMANTIC_CV_FOLDS_DIR,
        f"fold_{_current_semantic_fold}",
        "ml_semantic_bridge.pkl"
    )
    
    if os.path.exists(fold_model_path):
        logger.info(f"Loading ML Semantic Bridge from: {fold_model_path}")
        _semantic_bridge_instance = MLSemanticBridge.load(fold_model_path)
        return _semantic_bridge_instance
    
    # Fallback to single model path
    if os.path.exists(ML_SEMANTIC_BRIDGE_PATH):
        logger.info(f"Loading ML Semantic Bridge from fallback: {ML_SEMANTIC_BRIDGE_PATH}")
        _semantic_bridge_instance = MLSemanticBridge.load(ML_SEMANTIC_BRIDGE_PATH)
        return _semantic_bridge_instance
    
    logger.warning(
        f"ML Semantic Bridge not found at {fold_model_path} or {ML_SEMANTIC_BRIDGE_PATH}"
    )
    return None


def get_test_patient_ids(fold: int = None) -> List[str]:
    """
    Get the list of patient IDs in the test set for a given fold.
    
    Args:
        fold: Fold number (1-5). If None, uses current fold.
        
    Returns:
        List of patient IDs
    """
    if fold is None:
        fold = _current_semantic_fold
    
    test_csv_path = os.path.join(
        SEMANTIC_CV_FOLDS_DIR,
        f"fold_{fold}",
        "test.csv"
    )
    
    if not os.path.exists(test_csv_path):
        logger.warning(f"Test CSV not found: {test_csv_path}")
        return []
    
    df = pd.read_csv(test_csv_path)
    
    if 'PID' in df.columns:
        return df['PID'].tolist()
    elif 'participant_id' in df.columns:
        return df['participant_id'].tolist()
    else:
        logger.warning("No PID or participant_id column found in test CSV")
        return []


def get_cv_summary() -> Optional[Dict]:
    """
    Load the CV summary with results from all folds.
    
    Returns:
        Dictionary with CV summary or None if not found
    """
    summary_path = os.path.join(SEMANTIC_CV_FOLDS_DIR, "cv_summary.json")
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    return None


# =============================================================================
# UTILITIES
# =============================================================================

def safe_float(v) -> Optional[float]:
    """Safely convert value to float."""
    try:
        return float(v)
    except Exception:
        return None


def robust_z(value: float, median: float, iqr: float) -> float:
    """Calculate robust z-score using median and IQR."""
    if iqr <= 0:
        return 0.0
    return (value - median) / iqr


def directional_label(z: float) -> str:
    """Convert z-score to directional label."""
    if z >= 0.25:
        return "markedly increased"
    elif z >= 0.1:
        return "moderately increased"
    elif z >= -0.05:
        return "mildly increased"
    elif z <= -0.25:
        return "markedly reduced"
    elif z <= -0.20:
        return "moderately reduced"
    elif z <= -0.15:
        return "mildly reduced"
    return "within normal limits"


# =============================================================================
# FEATURE CONFIGURATION (for z-score mode)
# =============================================================================

class FeatureSelector:
    """Feature selection and grouping for semantic analysis."""
    
    FEATURES = {
        # Temporal
        "temporal_avg_response_latency",
        "temporal_avg_speaking_rate",
        "temporal_max_inter_word_gap",
        "temporal_avg_between_utterance_gap",
        
        # Lexical Richness and Diversity
        "vocabulary_range_score",
        "lexical_accuracy_score",
        "specificity_score",
        "advanced_vocabulary_score",
        
        # Syntactic Structure
        "grammar_complexity_score",
        "structure_variety_score",
        
        # Pragmatic Competence
        "referential_clarity_score",
        "state_of_mind_language_score",
        "implausible_details_score",
        
        # Semantic Coherence and Cohesion
        "topic_management_score",
        "logical_organization_score",
        "cohesion_score",
        "cause_and_effect_score",
        "repetition_score",
        "information_prioritization_score",
        
        # Demographics
        "age",
    }
    
    STRONG_GROUPS = {"Syntactic_Structure", "Lexical_Richness_and_Diversity"}
    WEAK_GROUPS = {"Pragmatic_Competence"}
    
    GROUP_WEIGHTS = {
        "Temporal_Features": 1.0,
        "Lexical_Richness_and_Diversity": 1.5,
        "Syntactic_Structure": 1.3,
        "Pragmatic_Competence": 1.0,
        "Semantic_Coherence_and_Cohesion": 1.5,
    }
    
    @classmethod
    def select(cls, patient: Dict) -> Dict[str, float]:
        """Select relevant features from patient data."""
        out = {}
        for k in cls.FEATURES:
            v = safe_float(patient.get(k))
            if v is not None:
                out[k] = v
        return out


# =============================================================================
# CLINICAL INDICES (for z-score mode)
# =============================================================================

class ClinicalIndices:
    """Weak-signal aggregation into neurological dimensions."""
    
    # Maps from feature groups to their summary fields
    ANALYSIS_MAP = {
        "Lexical_Richness_and_Diversity": [
            "vocabulary_range_summary",
            "lexical_accuracy_summary",
            "specificity_summary",
            "advanced_vocabulary_summary",
        ],
        "Syntactic_Structure": [
            "grammar_complexity_summary",
            "structure_variety_summary",
        ],
        "Pragmatic_Competence": [
            "referential_clarity_summary",
            "state_of_mind_language_summary",
            "implausible_details_summary",
        ],
        "Semantic_Coherence_and_Cohesion": [
            "topic_management_summary",
            "logical_organization_summary",
            "cohesion_summary",
            "cause_and_effect_summary",
            "repetition_summary",
            "information_prioritization_summary",
        ]
    }
    
    # Feature group mappings with direction info
    # Direction: -1 means higher value = more likely impairment
    MAP = {
        "Temporal_Features": [
            ("temporal_avg_response_latency", -1),
            ("temporal_avg_speaking_rate", 1),
            ("temporal_max_inter_word_gap", -1),
            ("temporal_avg_between_utterance_gap", -1),
        ],
        "Lexical_Richness_and_Diversity": [
            ("vocabulary_range_score", 1),
            ("lexical_accuracy_score", 1),
            ("specificity_score", 1),
            ("advanced_vocabulary_score", 1),
        ],
        "Syntactic_Structure": [
            ("grammar_complexity_score", 1),
            ("structure_variety_score", 1),
        ],
        "Pragmatic_Competence": [
            ("referential_clarity_score", 1),
            ("state_of_mind_language_score", 1),
            ("implausible_details_score", 1),
        ],
        "Semantic_Coherence_and_Cohesion": [
            ("topic_management_score", 1),
            ("logical_organization_score", 1),
            ("cohesion_score", 1),
            ("cause_and_effect_score", 1),
            ("repetition_score", 1),
            ("information_prioritization_score", 1),
        ]
    }
    
    @classmethod
    def compute(cls, z_features: Dict[str, float]) -> Dict[str, Dict]:
        """Compute clinical indices from z-scored features."""
        results = {}
        
        if is_debug:
            print(f" - The normalised raw feature length is {len(z_features)}.")
            print(f"{z_features}")
        
        for index, feature_defs in cls.MAP.items():
            values = []
            
            for fd in feature_defs:
                if isinstance(fd, tuple):
                    key, sign = fd
                    if key in z_features:
                        values.append(z_features[key] * sign)
                else:
                    if fd in z_features:
                        values.append(z_features[fd])
            
            if not values:
                continue
            
            values = np.array(values)
            mean_dev = float(np.mean(values))
            dispersion = float(np.std(values))
            support = len(values)
            
            # Improved aggregation for strong groups
            if index in FeatureSelector.STRONG_GROUPS:
                max_dev = float(np.max(values))
                group_z = 0.7 * mean_dev + 0.3 * max_dev
            else:
                group_z = mean_dev
            
            # Confidence calibration
            confidence = min(1.0, (1.0 / (1.0 + 1.5 * dispersion)))
            
            # Dominance flag
            dominant = (abs(group_z) >= Z_DOMINANT) and (confidence >= CONF_DOMINANT)
            
            results[index] = {
                "z": group_z,
                "direction": directional_label(group_z),
                "confidence": round(confidence, 3),
                "feature_support": support,
                "dispersion": round(float(dispersion), 3),
                "dominant": dominant
            }
        
        if is_debug:
            print(f"{results}")
        
        return results


# =============================================================================
# POPULATION STATISTICS BUILDER
# =============================================================================

class PopulationStatsBuilder:
    """Builds robust statistics for z-score normalization."""
    
    @staticmethod
    def build(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Build population statistics from a list of patient records."""
        features = FeatureSelector.FEATURES
        stats = {}
        
        for f in features:
            vals = [
                safe_float(r.get(f)) for r in rows
                if safe_float(r.get(f)) is not None
            ]
            if not vals:
                continue
            
            vals = np.array(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            stats[f] = {
                "median": float(np.median(vals)),
                "iqr": float(q3 - q1)
            }
        
        return stats


# =============================================================================
# SEMANTIC CLINICAL PROFILE CLASS
# =============================================================================

class SemanticClinicalProfile:
    """
    Dual-normalisation semantic analysis.
    
    Supports both ML Bridge mode and legacy z-score mode.
    """
    
    def __init__(
        self,
        healthy_stats: Dict[str, Dict[str, float]] = None,
        cohort_stats: Dict[str, Dict[str, float]] = None
    ):
        """
        Initialize the profiler.
        
        Args:
            healthy_stats: Statistics from healthy controls (for z-score mode)
            cohort_stats: Statistics from full cohort (for z-score mode)
        """
        self.healthy = healthy_stats or {}
        self.cohort = cohort_stats or {}
        self.bridge = get_bridge()
    
    def _normalise(
        self,
        features: Dict[str, float],
        stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Normalize features using z-scores."""
        z = {}
        for k, v in features.items():
            if k in stats:
                z[k] = robust_z(
                    v,
                    stats[k]["median"],
                    stats[k]["iqr"]
                )
        return z
    
    def analyse_patient(self, patient: Dict, pid: str = None) -> Dict:
        """
        Analyze a patient's semantic features.
        
        Uses ML Bridge if available, otherwise falls back to z-score mode.
        
        Args:
            patient: Dictionary of patient features
            pid: Patient ID
            
        Returns:
            Analysis dictionary
        """
        if self.bridge is not None:
            return self._analyse_with_ml_bridge(patient, pid)
        else:
            return self._analyse_with_zscore(patient, pid)
    
    def _analyse_with_ml_bridge(self, patient: Dict, pid: str) -> Dict:
        """Analyze using ML Bridge."""
        result = self.bridge.analyze_patient(patient, pid)
        
        profile = {
            "PID": pid,
            "mode": "ml_bridge",
            "overall_score": result['overall_score'],
            "overall_category": result['overall_category'],
            "ml_prediction": result['ml_prediction'],
            "threshold": result['threshold'],
            "model_agreement": result['model_agreement'],
            "group_scores": result['group_scores'],
            "group_z": {},  # For compatibility
        }
        
        return profile
    
    def _analyse_with_zscore(self, patient: Dict, pid: str) -> Dict:
        """Analyze using z-score mode."""
        selected = FeatureSelector.select(patient)
        
        z_healthy = self._normalise(selected, self.healthy)
        z_cohort = self._normalise(selected, self.cohort)
        
        indices_healthy = ClinicalIndices.compute(z_healthy)
        indices_cohort = ClinicalIndices.compute(z_cohort)
        
        if is_debug:
            print(f"indices_healthy: {indices_healthy}")
        
        # Extract group-level z for calibration
        group_z = {
            group: info["z"]
            for group, info in indices_healthy.items()
        }
        
        group_confidence = {
            group: info["confidence"]
            for group, info in indices_healthy.items()
        }
        
        dominant_healthy = {
            group: info["dominant"]
            for group, info in indices_healthy.items()
        }
        
        return {
            "raw_features": selected,
            "healthy_normalised": {
                "z_features": z_healthy,
                "indices": indices_healthy,
                "dominant": dominant_healthy
            },
            "cohort_normalised": {
                "z_features": z_cohort,
                "indices": indices_cohort
            },
            "group_z": group_z,
            "group_confidence": group_confidence,
            "PID": pid if pid else "unknown",
            "mode": "zscore",
        }
    
    def _zscore_to_category(self, z: float) -> str:
        """Convert z-score to risk category."""
        if z >= 0.6:
            return "HIGH RISK"
        elif z >= 0.3:
            return "MODERATE-HIGH RISK"
        elif z >= -0.1:
            return "MODERATE RISK"
        elif z >= -0.4:
            return "LOW-MODERATE RISK"
        else:
            return "LOW RISK"


# =============================================================================
# LLM PROMPT FORMATTING
# =============================================================================

def format_semantic_for_llm(profile: Dict, patient_features: Dict) -> str:
    """
    Format semantic analysis for LLM prompt.
    
    Args:
        profile: Analysis profile from SemanticClinicalProfile
        patient_features: Raw patient features (for summaries)
        
    Returns:
        Formatted string for LLM prompt
    """
    if profile.get("mode") == "ml_bridge":
        return _format_ml_bridge_for_llm(profile, patient_features)
    else:
        return _format_zscore_for_llm(profile, patient_features)


def _format_ml_bridge_for_llm(profile: Dict, patient_features: Dict) -> str:
    """Format ML Bridge analysis for LLM."""
    lines = []
    
    # Header with integrated score
    lines.append(f">>> INTEGRATED RISK SCORE: {profile['overall_category']} <<<")
    lines.append(f"    (Ensemble ML probability: {profile['overall_score']:.1%})")
    lines.append("")
    lines.append("*** USE THIS SCORE AS YOUR PRIMARY CLASSIFICATION BASIS ***")
    lines.append("")
    lines.append("-" * 60)
    lines.append("DOMAIN BREAKDOWN (for interpretation support):")
    lines.append("-" * 60)
    
    # Sort groups by score (highest risk first)
    group_scores = profile.get('group_scores', {})
    sorted_groups = sorted(
        group_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    # Display each group's risk level
    for group_name, gs in sorted_groups:
        score = gs['score']
        if score > 0.55:
            indicator = "↑"
        elif score < 0.45:
            indicator = "↓"
        else:
            indicator = "→"
        
        lines.append(
            f"  {indicator} {gs['display_name']}: {gs['level']} ({gs['confidence']} confidence)"
        )
    
    # Add detailed summaries section
    lines.append("")
    lines.append("-" * 60)
    lines.append("DETAILED ANALYSIS BY DOMAIN:")
    lines.append("-" * 60)
    
    # Import feature groups for summary field mapping
    try:
        from ml_semantic_bridge import SEMANTIC_FEATURE_GROUPS
    except ImportError:
        SEMANTIC_FEATURE_GROUPS = {}
    
    for group_name, gs in sorted_groups:
        group_info = SEMANTIC_FEATURE_GROUPS.get(group_name, {})
        summary_fields = group_info.get('summary_fields', [])
        
        if summary_fields:
            lines.append("")
            lines.append(f"**{gs['display_name']}:**")
            
            summaries_found = False
            for summary_field in summary_fields:
                if summary_field in patient_features:
                    summary_text = patient_features[summary_field]
                    if summary_text and str(summary_text).strip():
                        # Truncate long summaries
                        if len(str(summary_text)) > 500:
                            summary_text = str(summary_text)[:500] + "..."
                        lines.append(f"  - {summary_text}")
                        summaries_found = True
            
            if not summaries_found:
                lines.append("  (No detailed analysis available)")
    
    # Add interpretation guidelines
    lines.append("")
    lines.append("-" * 60)
    lines.append("INTERPRETATION GUIDELINES:")
    lines.append("-" * 60)
    lines.append("")
    lines.append("Key principles:")
    lines.append("1. The INTEGRATED RISK SCORE is your primary decision input")
    lines.append("2. Domain indicators provide interpretable breakdown for explanation")
    lines.append("3. 'ELEVATED' levels suggest patterns associated with cognitive impairment")
    lines.append("4. 'REDUCED' or 'TYPICAL' levels suggest patterns consistent with healthy controls")
    lines.append("5. LOW-confidence domains should be mentioned but not weighted heavily")
    lines.append("")
    lines.append("Feature direction notes:")
    lines.append("- Higher scores in lexical/syntactic/pragmatic/semantic features = healthier")
    lines.append("- Higher temporal latency/gaps = potential impairment indicator")
    lines.append("- Higher repetition_score = LESS problematic repetition (healthier)")
    lines.append("- Higher implausible_details_score = FEWER implausible details (healthier)")
    
    return "\n".join(lines)


def _format_zscore_for_llm(profile: Dict, patient_features: Dict) -> str:
    """Format z-score analysis for LLM."""
    lines = []
    
    indices = profile.get("healthy_normalised", {}).get("indices", {})
    has_dominant = False
    
    if is_debug:
        print("-" * 60)
        print(indices.items())
        print("-" * 60)
    
    for name, info in indices.items():
        if info["confidence"] < 0.3:
            if is_debug:
                print(f" Feature Group {name} -> current confidence is {info['confidence']}")
            continue
        
        z_val = info.get("z")
        
        if info.get('dominant', False):
            dominant = ", dominant"
            has_dominant = True
        else:
            dominant = ""
        
        category_name = name.replace('_', ' ').capitalize()
        lines.append(
            f" - {category_name} is "
            f"{info['direction']}. "
            f"(z={z_val:.2f}, confidence={info['confidence']:.2f}{dominant})."
        )
        
        # Find corresponding items in ANALYSIS_MAP and retrieve from LLM-generated feature analysis
        if name in ClinicalIndices.ANALYSIS_MAP:
            feature_names = ClinicalIndices.ANALYSIS_MAP[name]
            summaries = []
            for feature_name in feature_names:
                if feature_name in patient_features:
                    value = patient_features[feature_name]
                    if value is not None and str(value).strip():
                        summaries.append(str(value))
                    else:
                        if is_debug:
                            print(f"Warning: Cannot find the feature {feature_name}")
            
            if summaries:
                paragraph = " ".join(summaries)
                lines.append(f"Analysis about the {category_name}: {paragraph}\n\n")
        else:
            if is_debug:
                print(f"The feature group {name} is not in the predefined list - ANALYSIS_MAP")
    
    text = " ".join(lines)
    if has_dominant:
        return f"{text} If there are any dominant features listed above, they should be weighted more heavily than mild or inconsistent deviations."
    else:
        return text


# =============================================================================
# RULE-BASED PRE-CLASSIFIER
# =============================================================================

def severity_bin(z: float) -> int:
    """Define severity bins based on z-score."""
    if z <= -0.7:
        return 3  # severe
    if z <= -0.4:
        return 2  # moderate
    if z <= 0.2:
        return 1  # mild
    return 0  # normal


def weighted_evidence_score(indices: Dict[str, Dict]) -> float:
    """Calculate weighted evidence score across groups."""
    score = 0.0
    
    for group, info in indices.items():
        if info["confidence"] < 0.3:
            continue
        
        if len(FeatureSelector.GROUP_WEIGHTS) > 0:
            weight = FeatureSelector.GROUP_WEIGHTS.get(group, 1.0)
            score += weight * info["z"]
        else:
            score = info["z"]
    
    return score


def preclassify(indices: Dict[str, Dict]) -> str:
    """Pre-classify based on z-score indices."""
    severe = 0
    moderate = 0
    dominant_count = 0
    dominant_groups = []
    
    for group, info in indices.items():
        if info["confidence"] < 0.3:
            continue
        
        b = severity_bin(info["z"])
        if b == 3:
            severe += 1
        elif b == 2:
            moderate += 1
        
        if info["dominant"]:
            dominant_count += 1
            dominant_groups.append(group)
    
    # Rule-based classification
    if dominant_count >= 2:
        return "Impairment"
    
    if dominant_count == 1 and (severe >= 1 or moderate >= 2):
        return "Impairment"
    
    if severe >= 2 or (severe == 1 and moderate >= 2):
        return "Impairment"
    
    if dominant_count == 1 or (dominant_count < 1 and moderate >= 2):
        return "Impairment"
    
    return "Control"


def rule_based_semantic_diagnosis(profile: Dict) -> str:
    """Generate rule-based pre-diagnosis text."""
    if profile.get("mode") == "ml_bridge":
        return f"""
## Clinical Decision Support (ML Semantic Analysis)

**ML Ensemble Prediction: {profile['ml_prediction']}**
**Probability: {profile['overall_score']:.1%}**
**Classification Threshold: {profile['threshold']:.0%}**

The ensemble model (CatBoost + XGBoost + RandomForest) analyzes semantic/linguistic biomarkers.
Use the integrated score as your primary basis; domain breakdown supports explanation.
"""
    else:
        pre_diagnosis = preclassify(profile["healthy_normalised"]["indices"])
        return f"""Note: A rule-based dementia screening algorithm suggests that this patient's cognitive status is: **{pre_diagnosis}**. Please take this suggestion as a prior, but adjust if evidence strongly contradicts it."""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_patient_for_llm(
    patient_features: Dict[str, Any],
    pid: str = "unknown",
    healthy_stats: Dict = None,
    cohort_stats: Dict = None
) -> Tuple[str, str]:
    """
    Analyze patient semantic features and generate LLM prompt.
    
    Args:
        patient_features: Dictionary of patient features
        pid: Patient ID
        healthy_stats: Optional healthy population stats (for z-score mode)
        cohort_stats: Optional cohort stats (for z-score mode)
        
    Returns:
        Tuple of (pre_diagnosis_text, semantic_analysis_text)
    """
    bridge = get_bridge()
    
    if bridge is not None:
        return bridge.generate_llm_prompt(patient_features, pid)
    
    # Fallback to z-score mode
    profiler = SemanticClinicalProfile(healthy_stats, cohort_stats)
    profile = profiler.analyse_patient(patient_features, pid)
    
    pre_diagnosis = rule_based_semantic_diagnosis(profile)
    semantic_prompt = format_semantic_for_llm(profile, patient_features)
    
    return pre_diagnosis, semantic_prompt


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Testing SemanticClinicalProfile...")
    print(f"USE_ML_SEMANTIC_BRIDGE: {USE_ML_SEMANTIC_BRIDGE}")
    print(f"SEMANTIC_CV_FOLDS_DIR: {SEMANTIC_CV_FOLDS_DIR}")
    print(f"CURRENT_FOLD: {_current_semantic_fold}")
    
    bridge = get_bridge()
    if bridge:
        print(f"\nML Semantic Bridge loaded successfully")
        print(f"  Threshold: {bridge.classification_threshold}")
        print(f"  Features: {len(bridge.feature_names)}")
        print(f"  Models: {list(bridge.models.keys())}")
    else:
        print("\nML Semantic Bridge not available, using z-score mode")
    
    # Test with CV summary
    summary = get_cv_summary()
    if summary:
        print(f"\nCV Summary found:")
        print(f"  Folds: {summary.get('n_folds', 'N/A')}")
        print(f"  Mean AUC: {summary.get('mean_test_auc', 'N/A'):.4f}")
        print(f"  Mean Accuracy: {summary.get('mean_test_accuracy', 'N/A'):.4f}")
