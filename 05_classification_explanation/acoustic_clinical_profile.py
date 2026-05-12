"""
acoustic_clinical_profile.py

Updated Clinical Acoustic Profiling with Nested CV ML Bridge Integration
========================================================================

This module supports TWO modes:
1. ML Bridge mode (RECOMMENDED) - Uses pre-trained ensemble models from nested CV
2. Legacy z-score mode - Uses population-based z-score discretization

The ML Bridge mode is recommended because:
- Uses proper nested CV methodology (no data leakage)
- Inner CV optimizes threshold per fold
- Ensemble models capture complex feature interactions (AUC ~0.73)
- Categorical output is easier for LLMs to interpret

Usage:
    # The module automatically loads the appropriate fold's model
    # Set CV_FOLD environment variable or use set_current_fold()
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
import logging
from train_ml_acoustic_bridge_nested_cv import MLAcousticBridge

logger = logging.getLogger(__name__)
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Z-score thresholds (for legacy mode)
Z_DOMINANT = 1.2
CONF_DOMINANT = 0.6

# ML Bridge settings
USE_ML_BRIDGE = os.getenv("USE_ML_BRIDGE", "True").lower() == "true"
CV_FOLDS_DIR = os.getenv("CV_FOLDS_DIR", "cv_folds")
CURRENT_FOLD = int(os.getenv("CV_FOLD", "1"))

# Legacy single model path (fallback)
ML_BRIDGE_PATH = os.getenv("ML_BRIDGE_PATH", "ml_acoustic_bridge.pkl")

is_debug = os.getenv("DEBUG", "False").lower() == "true"

# Global bridge instance (lazy loaded)
_bridge_instance = None
_current_fold = CURRENT_FOLD


def set_current_fold(fold: int):
    """
    Set the current CV fold to use and reload the model.
    """
    global _current_fold, _bridge_instance
    if fold < 1 or fold > 5:
        raise ValueError(f"Fold must be 1-5, got {fold}")
    
    _current_fold = fold
    _bridge_instance = None  # Reset to force reload
    
    # Immediately load the new model (instead of lazy loading)
    bridge = get_bridge()
    
    if bridge is not None:
        logger.info(f"Fold {fold}: Loaded model with threshold={bridge.classification_threshold}")
    else:
        logger.warning(f"Fold {fold}: Failed to load model")


def get_current_fold() -> int:
    """Get the current CV fold number."""
    return _current_fold


def get_bridge():
    """
    Get the ML Bridge instance for the current fold.
    
    Returns:
        MLAcousticBridge instance
    """
    global _bridge_instance
    
    if _bridge_instance is not None:
        return _bridge_instance
    
    if not USE_ML_BRIDGE:
        return None
    
    # Import MLAcousticBridge class before unpickling
    # This ensures the class is available in the namespace
    try:
        from ml_acoustic_bridge import MLAcousticBridge
    except ImportError:
        try:
            from train_ml_acoustic_bridge_nested_cv import MLAcousticBridge
        except ImportError:
            logger.warning("Could not import MLAcousticBridge class")
            return None
    
    # Try to load from CV folds directory first
    fold_model_path = os.path.join(CV_FOLDS_DIR, f"fold_{_current_fold}", "ml_acoustic_bridge.pkl")
    
    if os.path.exists(fold_model_path):
        logger.info(f"Loading ML Bridge from: {fold_model_path}")
        _bridge_instance = MLAcousticBridge.load(fold_model_path)
        return _bridge_instance
    
    # Fallback to single model path
    if os.path.exists(ML_BRIDGE_PATH):
        logger.info(f"Loading ML Bridge from fallback: {ML_BRIDGE_PATH}")
        _bridge_instance = MLAcousticBridge.load(ML_BRIDGE_PATH)
        return _bridge_instance
    
    logger.warning(f"ML Bridge not found at {fold_model_path} or {ML_BRIDGE_PATH}")
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
        fold = _current_fold
    
    test_csv_path = os.path.join(CV_FOLDS_DIR, f"fold_{fold}", "test.csv")
    
    if not os.path.exists(test_csv_path):
        logger.warning(f"Test CSV not found: {test_csv_path}")
        return []
    
    import pandas as pd
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
    summary_path = os.path.join(CV_FOLDS_DIR, "cv_summary.json")
    
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
    if z >= 0.6:
        return "markedly increased"
    if z >= 0.3:
        return "moderately increased"
    if z >= 0.05:
        return "mildly increased"
    if z <= -0.8:
        return "markedly reduced"
    if z <= -0.4:
        return "moderately reduced"
    if z <= -0.1:
        return "mildly reduced"
    return "within normal limits"


# =============================================================================
# FEATURE SELECTION (for z-score mode)
# =============================================================================

class FeatureSelector:
    """Feature selection and grouping for acoustic analysis."""
    
    FEATURES = {
        'cpp_mean', 'pitch_std', 'pitch_range', 'pitch_iqr',
        'jitter', 'shimmer', 'hnr_mean',
        'pause_ratio', 'pause_variability', 'long_pause_ratio', 'hesitation_rate',
        'voice_breaks_rate', 'speech_rate_variability',
        'opensmile_jitterLocal_sma3nz_amean', 'opensmile_shimmerLocaldB_sma3nz_amean',
        'opensmile_HNRdBACF_sma3nz_amean', 'opensmile_MeanVoicedSegmentLengthSec',
        'opensmile_MeanUnvoicedSegmentLength',
        'age',
    }
    
    GROUP_WEIGHTS = {
        "Cepstral_Peak_Prominence": 2.0,
        "Phonatory_Instability": 1.5,
        "Prosodic_Flattening": 1.5,
        "Phonation_Discontinuity": 1.2,
        "Temporal_Pause_Disruption": 0.5,
        "Speech_Rate_Instability": 0.5,
        "Voice_Breaks": 0.3,
        "HNR_Reduction": 1.0,
    }
    
    FEATURE_GROUPS = {
        "Cepstral_Peak_Prominence": ["cpp_mean"],
        "Phonatory_Instability": [
            "jitter", "shimmer",
            "opensmile_jitterLocal_sma3nz_amean",
            "opensmile_shimmerLocaldB_sma3nz_amean"
        ],
        "Prosodic_Flattening": ["pitch_std", "pitch_range", "pitch_iqr"],
        "Phonation_Discontinuity": [
            "opensmile_MeanVoicedSegmentLengthSec",
            "opensmile_MeanUnvoicedSegmentLength"
        ],
        "Temporal_Pause_Disruption": [
            "pause_ratio", "pause_variability",
            "long_pause_ratio", "hesitation_rate"
        ],
        "Speech_Rate_Instability": ["speech_rate_variability"],
        "Voice_Breaks": ["voice_breaks_rate"],
        "HNR_Reduction": ["hnr_mean", "opensmile_HNRdBACF_sma3nz_amean"],
    }
    
    GROUP_DESCRIPTIONS = {
        "Cepstral_Peak_Prominence": "voice clarity and periodicity",
        "Phonatory_Instability": "vocal fold regularity",
        "Prosodic_Flattening": "speech melody variation",
        "Phonation_Discontinuity": "continuous voice production",
        "Temporal_Pause_Disruption": "pause patterns and timing",
        "Speech_Rate_Instability": "speaking tempo control",
        "Voice_Breaks": "voice continuity",
        "HNR_Reduction": "voice quality/breathiness",
    }

    @classmethod
    def get_features(cls) -> set:
        return cls.FEATURES

    @classmethod
    def get_group_weights(cls) -> Dict[str, float]:
        return cls.GROUP_WEIGHTS

    @classmethod
    def get_feature_groups(cls) -> Dict[str, List[str]]:
        return cls.FEATURE_GROUPS

    @classmethod
    def get_group_descriptions(cls) -> Dict[str, str]:
        return cls.GROUP_DESCRIPTIONS


# =============================================================================
# POPULATION STATS (for z-score mode)
# =============================================================================

class PopulationStatsBuilder:
    """Build population statistics for z-score normalization."""
    
    @staticmethod
    def build(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Build median and IQR for each feature from population data."""
        stats = {}
        features = FeatureSelector.get_features()
        
        for feature in features:
            vals = []
            for row in rows:
                v = safe_float(row.get(feature))
                if v is not None:
                    vals.append(v)
            
            if len(vals) >= 3:
                arr = np.array(vals)
                q1, med, q3 = np.percentile(arr, [25, 50, 75])
                stats[feature] = {"median": med, "iqr": q3 - q1}
            else:
                stats[feature] = {"median": 0.0, "iqr": 1.0}
        
        return stats


# =============================================================================
# ACOUSTIC CLINICAL PROFILE (Main Class)
# =============================================================================

class AcousticClinicalProfile:
    """
    Clinical profile generator supporting both ML Bridge and z-score modes.
    """
    
    def __init__(self, healthy_stats: Dict = None, cohort_stats: Dict = None, use_ml_bridge=False, ml_bridge_path=None):  
        self.healthy_stats = healthy_stats or {}
        self.cohort_stats = cohort_stats or {}
        self.use_ml_bridge = use_ml_bridge
        self.ml_bridge_path = ml_bridge_path
        self._bridge = None
    
    @property
    def bridge(self):
        if self._bridge is None and self.use_ml_bridge:
            self._bridge = get_bridge()
        return self._bridge
    
    def analyse_patient(self, patient: Dict, pid: str = "unknown") -> Dict:
        if self.use_ml_bridge and self.bridge is not None:
            return self._analyse_with_ml_bridge(patient, pid)
        else:
            return self._analyse_with_zscore(patient, pid)
    
    def _analyse_with_ml_bridge(self, patient: Dict, pid: str) -> Dict:
        result = self.bridge.analyze_patient(patient, pid)
        
        profile = {
            "PID": pid,
            "mode": "ml_bridge",
            "overall_score": result['overall_score'],
            "overall_category": result['overall_category'],
            "ml_prediction": result['ml_prediction'],
            "threshold": result['threshold'],
            "group_scores": result['group_scores'],
            "group_z": {},
        }
        
        return profile
    
    def _analyse_with_zscore(self, patient: Dict, pid: str) -> Dict:
        feature_groups = FeatureSelector.get_feature_groups()
        group_weights = FeatureSelector.get_group_weights()
        group_descriptions = FeatureSelector.get_group_descriptions()
        
        group_z = {}
        group_labels = {}
        
        for group_name, features in feature_groups.items():
            z_scores = []
            for feat in features:
                val = safe_float(patient.get(feat))
                stats = self.healthy_stats.get(feat)
                
                if val is not None and stats:
                    z = robust_z(val, stats['median'], stats['iqr'])
                    z_scores.append(z)
            
            if z_scores:
                avg_z = np.mean(z_scores)
                group_z[group_name] = avg_z
                group_labels[group_name] = directional_label(avg_z)
            else:
                group_z[group_name] = 0.0
                group_labels[group_name] = "data unavailable"
        
        total_weight = sum(group_weights.values())
        weighted_sum = sum(
            group_z.get(g, 0) * w 
            for g, w in group_weights.items()
        )
        overall_z = weighted_sum / total_weight if total_weight > 0 else 0
        
        profile = {
            "PID": pid,
            "mode": "zscore",
            "overall_z": overall_z,
            "overall_category": self._zscore_to_category(overall_z),
            "group_z": group_z,
            "group_labels": group_labels,
            "group_descriptions": group_descriptions,
        }
        
        return profile
    
    def _zscore_to_category(self, z: float) -> str:
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

def format_for_llm(profile: Dict) -> str:
    if profile.get("mode") == "ml_bridge":
        return _format_ml_bridge_for_llm(profile)
    else:
        return _format_zscore_for_llm(profile)


def _format_ml_bridge_for_llm(profile: Dict) -> str:
    lines = []
    
    lines.append(f">>> INTEGRATED RISK SCORE: {profile['overall_category']} <<<")
    lines.append(f"    (Ensemble ML probability: {profile['overall_score']:.1%})")
    lines.append("")
    lines.append("*** USE THIS SCORE AS YOUR PRIMARY CLASSIFICATION BASIS ***")
    lines.append("")
    lines.append("-" * 60)
    lines.append("DOMAIN BREAKDOWN (for interpretation support):")
    lines.append("-" * 60)
    
    group_scores = profile.get('group_scores', {})
    sorted_groups = sorted(
        group_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    for group_name, gs in sorted_groups:
        score = gs['score']
        if score > 0.55:
            indicator = "↑"
        elif score < 0.45:
            indicator = "↓"
        else:
            indicator = "→"
        
        lines.append(f"  {indicator} {gs['display_name']}: {gs['level']} ({gs['confidence']} confidence)")
    
    lines.append("")
    lines.append("Note: The integrated score captures complex feature interactions.")
    lines.append("")
    lines.append("## Interpretation Guidelines")
    lines.append("")
    lines.append("The acoustic analysis above was generated by validated machine learning models that")
    lines.append("have learned the complex relationships between voice features and cognitive status.")
    lines.append("")
    lines.append("Key principles:")
    lines.append("1. The INTEGRATED RISK SCORE is your primary decision input - it combines all features")
    lines.append("2. Domain indicators provide interpretable breakdown for your explanation")
    lines.append("3. \"Elevated\" levels suggest patterns associated with cognitive impairment")
    lines.append("4. \"Reduced\" or \"Typical\" levels suggest patterns consistent with healthy controls")
    lines.append("5. Low-confidence domains should be mentioned but not weighted heavily")
    lines.append("")
    lines.append("If patterns conflict:")
    lines.append("- The integrated score already accounts for feature interactions")
    lines.append("- Use domain breakdown to explain your reasoning, not override the integrated score")
    
    return "\n".join(lines)


def _format_zscore_for_llm(profile: Dict) -> str:
    lines = []
    
    lines.append(f">>> CLINICAL RISK ASSESSMENT: {profile['overall_category']} <<<")
    lines.append(f"    (Overall z-score: {profile.get('overall_z', 0):.2f})")
    lines.append("")
    lines.append("-" * 60)
    lines.append("DOMAIN BREAKDOWN:")
    lines.append("-" * 60)
    
    group_z = profile.get('group_z', {})
    group_labels = profile.get('group_labels', {})
    group_desc = profile.get('group_descriptions', {})
    
    for group_name, z in sorted(group_z.items(), key=lambda x: x[1], reverse=True):
        label = group_labels.get(group_name, "unknown")
        desc = group_desc.get(group_name, "")
        
        if z > 0.3:
            indicator = "↑"
        elif z < -0.3:
            indicator = "↓"
        else:
            indicator = "→"
        
        lines.append(f"  {indicator} {group_name}: {label} (z={z:.2f})")
        if desc:
            lines.append(f"      [{desc}]")
    
    return "\n".join(lines)


def rule_based_diagnosis(profile: Dict) -> str:
    if profile.get("mode") == "ml_bridge":
        return f"""
## Clinical Decision Support

**ML Ensemble Prediction: {profile['ml_prediction']}**
**Probability: {profile['overall_score']:.1%}**
**Classification Threshold: {profile['threshold']:.0%}**

The ensemble model (CatBoost + XGBoost + RandomForest) analyzes acoustic biomarkers.
Use the integrated score as your primary basis; domain breakdown supports explanation.
"""
    else:
        category = profile.get('overall_category', 'UNKNOWN')
        overall_z = profile.get('overall_z', 0)
        
        if overall_z >= 0.3:
            prediction = "Impairment"
        else:
            prediction = "Control"
        
        return f"""
## Clinical Assessment Summary

**Risk Category: {category}**
**Z-Score Prediction: {prediction}**
**Overall Z-Score: {overall_z:.2f}**

This assessment is based on population-normalized acoustic biomarkers.
"""



if __name__ == "__main__":
    import pandas as pd
    
    print("Testing AcousticClinicalProfile...")
    print(f"USE_ML_BRIDGE: {USE_ML_BRIDGE}")
    print(f"CV_FOLDS_DIR: {CV_FOLDS_DIR}")
    print(f"CURRENT_FOLD: {_current_fold}")
    
    bridge = get_bridge()
    if bridge:
        print(f"ML Bridge loaded successfully")
        print(f"  Threshold: {bridge.classification_threshold}")
        print(f"  Features: {len(bridge.feature_names)}")
        print(f"  Models: {list(bridge.models.keys())}")
    else:
        print("ML Bridge not available, using z-score mode")
