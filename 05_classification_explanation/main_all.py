#!/usr/bin/env python3
"""
main_all.py

Combined Multi-Modal Dementia Evaluation System
===============================================

This script combines acoustic and semantic features for cognitive impairment
screening using a multi-modal approach .

Architecture:
1. Load pre-trained ML bridges for both modalities (acoustic and semantic)
2. For each patient, generate analyses from both bridges
3. Combine both analyses into a single prompt using STANDARD_EVALUATION_PROMPT_TEMPLATE
4. Send to LLM for final classification with explanation

Prerequisites:
1. Run train_unified_cv_folds.py to create aligned CV folds
2. Run train_ml_acoustic_bridge_nested_cv.py --use-unified-folds
3. Run train_ml_semantic_bridge_nested_cv.py --use-unified-folds

Usage:
    python main_all.py                              # Run 5-fold CV evaluation
    python main_all.py --fold 1                     # Evaluate fold 1 only
    python main_all.py --evaluate-single P001       # Evaluate single patient
    python main_all.py --evaluate-single P001 --fold 2  # Evaluate single patient using fold 2
    python main_all.py --provider openai --model gpt-4o
    python main_all.py --check-bridges              # Check bridge status
    python main_all.py --recalculate                # Recalculate metrics from saved CSVs
    python main_all.py --recalculate --output-dir path/to/results  # Recalculate from specific dir

Environment:
    Set PROMPTCONTENT=standard in .env to use this script
"""

import argparse
import logging
import sys
import random
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get_active_llm_config
from llm_providers import get_provider, LLMProvider
from models import EvaluationResult, DiagnosisResult

# Import from unified CV folds
from train_unified_cv_folds import (
    load_unified_folds,
    get_fold_pids,
    DEFAULT_OUTPUT_DIR as UNIFIED_FOLDS_DIR
)

# Import ML bridges
try:
    from ml_acoustic_bridge import MLAcousticBridge
    ACOUSTIC_BRIDGE_AVAILABLE = True
except ImportError:
    ACOUSTIC_BRIDGE_AVAILABLE = False
    print("Warning: MLAcousticBridge not available")

try:
    from ml_semantic_bridge import MLSemanticBridge
    SEMANTIC_BRIDGE_AVAILABLE = True
except ImportError:
    SEMANTIC_BRIDGE_AVAILABLE = False
    print("Warning: MLSemanticBridge not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

is_debug = os.getenv("DEBUG", "").lower() == "true"
result_folder = os.getenv("RESULTS_DIR", "datasets/results")
stop_reasoning_env = os.getenv("STOPREASOING", "false").lower() == "true"
is_reasoning = not stop_reasoning_env 

# Data paths
ACOUSTIC_DATA_PATH = os.getenv(
    "ACOUSTIC_FEATURES_CSV",
    "datasets/output/acoustic_features/merged_data.csv"
)
SEMANTIC_DATA_PATH = os.path.join(
    os.getenv("SEMANTIC_FEATURE_FOLDER", "datasets/output/semantic_features"),
    "merged_semantic_data.csv"
)

# CV fold directories
ACOUSTIC_CV_DIR = os.getenv("CV_FOLDS_DIR", "cv_folds")
SEMANTIC_CV_DIR = os.path.join(
    os.getenv("RESULTS_DIR", "datasets/results"),
    "semantic_cv_folds"
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS FOR COMBINED EVALUATION
# =============================================================================

COMBINED_SYSTEM_PROMPT = """You are an expert clinical neuropsychologist specializing in cognitive assessment 
and dementia diagnosis through multi-modal analysis.

## Your Expertise
You analyze patient data from TWO complementary modalities:

1. **Acoustic Voice Biomarkers**: Captures speech motor control and neurological integrity
   - Phonatory Instability features (like jitter, shimmer)
   - Cepstral Peak Prominence features (like cpp_mean)
   - Prosodic Variation features (like pitch_std, pitch_range, and pitch_iqr)
   - Temporal Pause patterns (like pause_variability, long_pause_ratio, and hesitation_rate)
   - Phonation Continuity features (like mean_voiced_segment_length)
   - Speech Rate features (like speech rate variablility)
   - Voice Breaks (like voice_breaks_rate)
   - Harmonics-to-Noise Ratio (like hnr_mean, opensmile_HNRdBACF_sma3nz_amean)

2. **Semantic/Linguistic Features**: Captures language processing and cognitive organization
   - Lexical richness (vocabulary range, lexical accuracy, specificity score, and advance vocabulary usage)
   - Syntactic complexity (grammar complexity, sentence variety)
   - Pragmatic competence (referential clarity score, state_of_mind language, and implausible details)
   - Semantic coherence (topic management, local coherence, cause and effect relations, repetitions and information prioritization)
   - Temporal speech patterns (average response latency, average speaking rate, maximum inter word gap, and average between utterance gap)

## ML-Enhanced Analysis
Both modalities have been analyzed by ML ensemble models (XGBoost, CatBoost, RandomForest) trained on clinical data. Use these integrated risk scores as your PRIMARY decision basis.
However, as ML ensemble models can have limitations (sometimes the predictions are unreliable or inconsistent), you must also consider the detailed feature-level and/or feature group breakdowns provided in each section.

Key principles:
1. The INTEGRATED RISK SCORE is your decision-making guide
2. Domain indicators provide interpretable breakdowns for evidence-based decision-making
3. 'ELEVATED' levels suggest patterns associated with cognitive impairment
4. 'REDUCED' or 'TYPICAL' levels suggest patterns consistent with healthy controls
5. LOW-confidence domains should be mentioned but not weighted heavily


**Integration Rules:**
- If BOTH indicate HIGH RISK → Strong Impairment classification (8-9)
- If BOTH indicate LOW RISK → Strong Control classification (8-9)
- If scores CONFLICT or MODERATE or WEAK confidence or LOW agreement → Use breakdown explanations and facts to decide

## Output Format
Respond with a JSON object:
```json
{
    "Control": <integer 1-10>,
    "Impairment": <integer 1-10>,
    "explanation": "<string, max 400 words with facts as evidence from both modalities>"
}
```
The scores must sum to 10. Higher = more likely.


"""

COMBINED_USER_PROMPT_TEMPLATE = """
## Patient Demographics
{patient_data}

Use the following two "integrated risk scores" as your classification basis while considering the breakdowns from BOTH modalities below.

        
## MODALITY 1: Acoustic Voice Biomarker Analysis
{acoustic_section}

## MODALITY 2: Semantic/Linguistic Analysis
{semantic_section}

## Your Task

Analyze ALL available evidence (acoustic AND semantic) to assess cognitive status:

### Multi-Modal Integration Strategy

**PRIMARY Decision Basis**: Use BOTH integrated risk scores as basis:

1. **Acoustic ML Analysis** (above): Reflects motor speech control and neurological integrity
2. **Semantic ML Analysis** (above): Reflects language processing and cognitive organization


### Final Assessment

Provide your integrated assessment:
- Probability scores (1-10 scale, sum to 10)
- Explanation referencing BOTH acoustic AND semantic evidence
- Note any convergent or divergent patterns

**Respond with JSON only:**
```json
{{
    "Control": 3,
    "Impairment": 7,
    "explanation": "A maximum of 400 words integrating evidence from both modalities."
}}
```


Explanation Guidelines:

Your explanation is used by the GP (General Practitioner) and/or SP (Specialist) to support their clinical decision-making. Your explanation must satisfy all four criteria below:

| Criterion | Requirement |
|---|---|
| **Clinical Plausibility** | Reasoning must align with established neuropsychological knowledge — avoid contradictions with known clinical principles |
| **Evidence Grounding** | Cite specific features from the provided acoustic and semantic data to justify every claim — no unsupported assertions |
| **Clarity** | Use concise clinical language accessible to both GPs and SPs — define technical terms where necessary |
| **Usefulness** | Highlight the most diagnostically significant findings that support the predicted diagnosis |

Your explanation should make the diagnostic reasoning transparent and interpretable — providing the treating GP and/or SP with a clear clinical justification for why the patient received this diagnosis, based solely on the evidence provided.

An exemplary explanation might look like this:

Paragraph 1: final predition result from the integration of both modalities (discuss any convergent or divergent patterns and how they influenced your final assessment).
Paragraph 2: detailed analysis of acoustic features (mention specific elevated or reduced features and their clinical implications).
Paragraph 3: detailed analysis of semantic features (mention specific elevated or reduced features and their clinical implications).    
Paragraph 4: summmary of your prediction and key evidence from both modalities (1-2 sentences).

"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def calculate_probability_softmax(imp_conf: float, ctrl_conf: float) -> float:
    """Calculate probability of impairment using softmax normalization."""
    try:
        imp_exp = np.exp(float(imp_conf))
        ctrl_exp = np.exp(float(ctrl_conf))
        prob = imp_exp / (imp_exp + ctrl_exp)
        return max(0.0, min(1.0, prob))
    except (ValueError, TypeError):
        return 0.5


# =============================================================================
# BRIDGE MANAGEMENT
# =============================================================================

class CombinedBridgeManager:
    """
    Manages both acoustic and semantic ML bridges for combined evaluation.
    """
    
    def __init__(
        self,
        acoustic_cv_dir: str = ACOUSTIC_CV_DIR,
        semantic_cv_dir: str = SEMANTIC_CV_DIR,
        unified_folds_dir: str = UNIFIED_FOLDS_DIR
    ):
        self.acoustic_cv_dir = acoustic_cv_dir
        self.semantic_cv_dir = semantic_cv_dir
        self.unified_folds_dir = unified_folds_dir
        
        self.acoustic_bridge = None
        self.semantic_bridge = None
        self.current_fold = None
        
        # Load data
        self.acoustic_df = None
        self.semantic_df = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load acoustic and semantic feature datasets."""
        if os.path.exists(ACOUSTIC_DATA_PATH):
            self.acoustic_df = pd.read_csv(ACOUSTIC_DATA_PATH)
            logger.info(f"Loaded acoustic data: {len(self.acoustic_df)} patients")
        else:
            logger.warning(f"Acoustic data not found: {ACOUSTIC_DATA_PATH}")
        
        if os.path.exists(SEMANTIC_DATA_PATH):
            self.semantic_df = pd.read_csv(SEMANTIC_DATA_PATH)
            logger.info(f"Loaded semantic data: {len(self.semantic_df)} patients")
        else:
            logger.warning(f"Semantic data not found: {SEMANTIC_DATA_PATH}")
    
    def set_fold(self, fold: int) -> None:
        """Load bridges for a specific fold."""
        if fold < 1 or fold > 5:
            raise ValueError(f"Fold must be 1-5, got {fold}")
        
        self.current_fold = fold
        
        # Load acoustic bridge
        acoustic_path = os.path.join(
            self.acoustic_cv_dir, f"fold_{fold}", "ml_acoustic_bridge.pkl"
        )
        if os.path.exists(acoustic_path) and ACOUSTIC_BRIDGE_AVAILABLE:
            self.acoustic_bridge = MLAcousticBridge.load(acoustic_path)
            logger.info(f"Loaded acoustic bridge for fold {fold}")
        else:
            logger.warning(f"Acoustic bridge not found: {acoustic_path}")
            self.acoustic_bridge = None
        
        # Load semantic bridge
        semantic_path = os.path.join(
            self.semantic_cv_dir, f"fold_{fold}", "ml_semantic_bridge.pkl"
        )
        if os.path.exists(semantic_path) and SEMANTIC_BRIDGE_AVAILABLE:
            self.semantic_bridge = MLSemanticBridge.load(semantic_path)
            logger.info(f"Loaded semantic bridge for fold {fold}")
        else:
            logger.warning(f"Semantic bridge not found: {semantic_path}")
            self.semantic_bridge = None
    
    def get_test_pids(self, fold: int = None) -> List[str]:
        """Get test patient IDs for a fold from unified folds."""
        if fold is None:
            fold = self.current_fold
        
        _, test_pids = get_fold_pids(self.unified_folds_dir, fold)
        return test_pids
    
    def get_patient_acoustic_features(self, pid: str) -> Optional[Dict[str, Any]]:
        """Get acoustic features for a patient."""
        if self.acoustic_df is None:
            return None
        
        # Try different ID columns
        id_col = 'participant_id' if 'participant_id' in self.acoustic_df.columns else 'PID'
        patient_rows = self.acoustic_df[self.acoustic_df[id_col].astype(str) == str(pid)]
        
        if patient_rows.empty:
            return None
        
        return patient_rows.iloc[0].to_dict()
    
    def get_patient_semantic_features(self, pid: str) -> Optional[Dict[str, Any]]:
        """Get semantic features for a patient."""
        if self.semantic_df is None:
            return None
        
        id_col = 'PID' if 'PID' in self.semantic_df.columns else 'participant_id'
        patient_rows = self.semantic_df[self.semantic_df[id_col].astype(str) == str(pid)]
        
        if patient_rows.empty:
            return None
        
        return patient_rows.iloc[0].to_dict()
    
    def generate_combined_analysis(
        self, 
        pid: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Generate combined analysis for a patient.
        
        Returns:
            Tuple of (demographic_section, acoustic_section, semantic_section)
        """
        acoustic_features = self.get_patient_acoustic_features(pid)
        semantic_features = self.get_patient_semantic_features(pid)
        
        # Build demographic section
        age = None
        gender = None
        
        if semantic_features:
            age = semantic_features.get('age')
            gender = semantic_features.get('gender')
        if acoustic_features and (age is None or gender is None):
            age = age or acoustic_features.get('age')
            gender = gender or acoustic_features.get('gender')
        
        demographic_section = f"Age: {age or 'Unknown'}, Gender: {gender or 'Unknown'}"
        
        # Generate acoustic analysis
        acoustic_section = "Acoustic analysis not available."
        if acoustic_features and self.acoustic_bridge:
            try:
                _, acoustic_section = self.acoustic_bridge.generate_llm_prompt(
                    acoustic_features, pid
                )
            except Exception as e:
                logger.error(f"Acoustic analysis failed for {pid}: {str(e)}")
                print(f"Acoustic analysis failed for {pid}: {e}")
                acoustic_section = f"Acoustic analysis failed: {str(e)}"
        
        # Generate semantic analysis
        semantic_section = "Semantic analysis not available."
        if semantic_features and self.semantic_bridge:
            try:
                _, semantic_section = self.semantic_bridge.generate_llm_prompt(
                    semantic_features, pid
                )
            except Exception as e:
                logger.error(f"Semantic analysis failed for {pid}: {e}")
                semantic_section = f"Semantic analysis failed: {str(e)}"
        
        return demographic_section, acoustic_section, semantic_section
    
    def check_status(self) -> Dict[str, Any]:
        """Check the status of bridges and data."""
        status = {
            'acoustic_data_loaded': self.acoustic_df is not None,
            'acoustic_data_count': len(self.acoustic_df) if self.acoustic_df is not None else 0,
            'semantic_data_loaded': self.semantic_df is not None,
            'semantic_data_count': len(self.semantic_df) if self.semantic_df is not None else 0,
            'current_fold': self.current_fold,
            'acoustic_bridge_loaded': self.acoustic_bridge is not None,
            'semantic_bridge_loaded': self.semantic_bridge is not None,
            'unified_folds_available': os.path.exists(
                os.path.join(self.unified_folds_dir, "cv_fold_summary.json")
            ),
        }
        
        # Check each fold
        for fold in range(1, 6):
            acoustic_path = os.path.join(
                self.acoustic_cv_dir, f"fold_{fold}", "ml_acoustic_bridge.pkl"
            )
            semantic_path = os.path.join(
                self.semantic_cv_dir, f"fold_{fold}", "ml_semantic_bridge.pkl"
            )
            unified_path = os.path.join(
                self.unified_folds_dir, f"fold_{fold}", "test_pids.txt"
            )
            
            status[f'fold_{fold}_acoustic'] = os.path.exists(acoustic_path)
            status[f'fold_{fold}_semantic'] = os.path.exists(semantic_path)
            status[f'fold_{fold}_unified'] = os.path.exists(unified_path)
        
        return status


# =============================================================================
# COMBINED EVALUATOR CLASS
# =============================================================================

class CombinedEvaluator:
    """
    Evaluator that uses both acoustic and semantic ML bridges for classification.
    """
    
    def __init__(self, config=None):
        self.config = config or load_config()
        self.provider: Optional[LLMProvider] = None
        self.bridge_manager = CombinedBridgeManager()
        self.results: List[EvaluationResult] = []
    
    def setup_provider(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        """Set up the LLM provider."""
        if provider_name:
            self.config.active_provider = provider_name
        if model:
            self.config.active_model = model
        
        provider_name, llm_config = get_active_llm_config(self.config)
        
        logger.info(f"Setting up provider: {provider_name} with model: {llm_config.model}")
        
        self.provider = get_provider(
            provider_name,
            llm_config,
            self.config.processing
        )
    
    def set_fold(self, fold: int) -> None:
        """Set the current CV fold."""
        self.bridge_manager.set_fold(fold)
    
    def evaluate_patient(self, pid: str) -> EvaluationResult:
        """
        Evaluate a single patient using both modalities.
        
        Args:
            pid: Patient ID
            
        Returns:
            EvaluationResult
        """
        import time
        start_time = time.time()
        
        # Generate combined analysis
        demographic_section, acoustic_section, semantic_section = \
            self.bridge_manager.generate_combined_analysis(pid)
        
        # Check if we have at least some data
        if acoustic_section.startswith("Acoustic analysis not available") and \
           semantic_section.startswith("Semantic analysis not available"):
            return EvaluationResult(
                pid=pid,
                prediction=DiagnosisResult.UNKNOWN,
                explanation="",
                error="No feature data available for either modality"
            )
        
        # Build prompt
   

        user_prompt = COMBINED_USER_PROMPT_TEMPLATE.format(
            patient_data=demographic_section,
            acoustic_section=acoustic_section,
            semantic_section=semantic_section
        )
        
        if is_debug:
            print(f"\n{'='*60}")
            print(f"Patient: {pid}")
            print(f"{'='*60}")
            print(f"\nDemographics: {demographic_section}")
            print(f"\n--- Acoustic Section ---\n{acoustic_section[:500]}...")
            print(f"\n--- Semantic Section ---\n{semantic_section[:500]}...")
        
        # Check if we should skip LLM call
        if not is_reasoning or is_debug:
            logger.info(f"Skipping LLM call for {pid} (STOPREASOING=True)")
            self.provider.save_to_txt(f"System:\n{COMBINED_SYSTEM_PROMPT}\nUser:\n{user_prompt}", f"{pid}.txt", result_folder)
        
        if not is_reasoning:
            return EvaluationResult(
                pid=pid,
                prediction=DiagnosisResult.UNKNOWN,
                explanation="LLM call skipped (debug mode)",
                error=None,
                processing_time_seconds=time.time() - start_time
            )
        
        # Call LLM
        if self.provider is None:
            return EvaluationResult(
                pid=pid,
                prediction=DiagnosisResult.UNKNOWN,
                explanation="",
                error="LLM provider not configured"
            )
        
            
        try:
           
            result = self.provider.evaluate(
                pid=pid,
                system_prompt=COMBINED_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
        
            return result
            
        except Exception as e:
            logger.error(f"LLM evaluation failed for {pid}: {e}")
            return EvaluationResult(
                pid=pid,
                prediction=DiagnosisResult.UNKNOWN,
                explanation="",
                error=f"LLM evaluation failed: {str(e)}",
                processing_time_seconds=time.time() - start_time
            )
    
    def evaluate_all(
        self,
        patient_ids: List[str],
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate multiple patients."""
        results = []
        
        total = len(patient_ids)
        for i, pid in enumerate(patient_ids):
            if show_progress:
                print(f"Evaluating {i+1}/{total}: {pid}")
            
            result = self.evaluate_patient(pid)
            results.append(result)
            
            if is_debug and result.error:
                print(f"  Error: {result.error}")
            elif is_debug:
                print(f"  Prediction: {result.prediction.value}")
        
        self.results = results
        return results
    
    def save_results(
        self,
        results: List[EvaluationResult],
        output_path: str = None
    ) -> str:
        """Save results to CSV."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(result_folder, f"combined_results_{timestamp}.csv")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build rows
        rows = []
        for r in results:
            # Get ground truth
            groundtruth = self._get_groundtruth(r.pid)
            
            rows.append({
                'PID': r.pid,
                'GroundTruth': groundtruth,
                'Prediction': r.prediction.value if r.prediction else 'unknown',
                'control_conf': r.control_conf,
                'impairment_conf': r.impairment_conf,
                'probability': calculate_probability_softmax(
                    r.impairment_conf or 5,
                    r.control_conf or 5
                ),
                'explanation': r.explanation,
                'error': r.error,
                'processing_time': r.processing_time_seconds
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def _get_groundtruth(self, pid: str) -> str:
        """Get ground truth diagnosis for a patient."""
        # Try semantic data first
        if self.bridge_manager.semantic_df is not None:
            id_col = 'PID' if 'PID' in self.bridge_manager.semantic_df.columns else 'participant_id'
            rows = self.bridge_manager.semantic_df[
                self.bridge_manager.semantic_df[id_col].astype(str) == str(pid)
            ]
            if not rows.empty and 'diagnosis' in rows.columns:
                return rows.iloc[0]['diagnosis']
        
        # Try acoustic data
        if self.bridge_manager.acoustic_df is not None:
            id_col = 'participant_id' if 'participant_id' in self.bridge_manager.acoustic_df.columns else 'PID'
            rows = self.bridge_manager.acoustic_df[
                self.bridge_manager.acoustic_df[id_col].astype(str) == str(pid)
            ]
            if not rows.empty and 'diagnosis' in rows.columns:
                return rows.iloc[0]['diagnosis']
        
        return 'unknown'
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        y_true = []
        y_pred = []
        y_prob = []
        
        impairment_labels = [
            'impairment', 'mci', 'dementia', 'ad', "alzheimer's",
            'probablead', 'possiblead', 'memory', 'vascular'
        ]
        
        for r in results:
            if r.prediction == DiagnosisResult.UNKNOWN:
                continue
            
            # Get ground truth
            gt = self._get_groundtruth(r.pid).lower()
            
            if gt in impairment_labels or any(imp in gt for imp in impairment_labels):
                y_true.append(1)
            else:
                y_true.append(0)
            
            # Prediction
            y_pred.append(1 if r.prediction == DiagnosisResult.IMPAIRMENT else 0)
            
            # Probability
            prob = calculate_probability_softmax(
                r.impairment_conf or 5,
                r.control_conf or 5
            )
            y_prob.append(prob)
        
        if len(y_true) == 0:
            return {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'n_samples': int(len(y_true)),
        }
        
        # AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = float(roc_auc_score(y_true, y_prob))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            # Convert numpy int64 to Python int for JSON serialization
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
            metrics['sensitivity'] = float(metrics['tp'] / (metrics['tp'] + metrics['fn'])) if (metrics['tp'] + metrics['fn']) > 0 else 0.0
            metrics['specificity'] = float(metrics['tn'] / (metrics['tn'] + metrics['fp'])) if (metrics['tn'] + metrics['fp']) > 0 else 0.0
        
        return metrics


# =============================================================================
# CSV LOADING AND RECALCULATION FUNCTIONS
# =============================================================================

def load_results_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation results from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of result dictionaries
    """
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        results.append({
            'pid': str(row['PID']),
            'groundtruth': str(row['GroundTruth']),
            'prediction': str(row['Prediction']),
            'control_conf': float(row['control_conf']) if pd.notna(row['control_conf']) else 5.0,
            'impairment_conf': float(row['impairment_conf']) if pd.notna(row['impairment_conf']) else 5.0,
            'probability': float(row['probability']) if pd.notna(row.get('probability')) else 0.5,
        })
    
    return results


def calculate_metrics_from_dicts(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics from result dictionaries (loaded from CSV).
    
    Args:
        results: List of result dictionaries with keys: groundtruth, prediction, probability
        
    Returns:
        Dictionary with calculated metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    y_true = []
    y_pred = []
    y_prob = []
    
    impairment_labels = [
        'impairment', 'mci', 'dementia', 'ad', "alzheimer's",
        'probablead', 'possiblead', 'memory', 'vascular'
    ]
    
    for r in results:
        pred = r['prediction'].lower()
        if pred == 'unknown':
            continue
        
        # Get ground truth
        gt = r['groundtruth'].lower()
        
        if gt in impairment_labels or any(imp in gt for imp in impairment_labels):
            y_true.append(1)
        else:
            y_true.append(0)
        
        # Prediction
        y_pred.append(1 if pred == 'impairment' else 0)
        
        # Probability
        y_prob.append(r['probability'])
    
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'n_samples': int(len(y_true)),
    }
    
    # AUC if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return metrics


def recalculate_metrics_from_csv(
    input_dir: str,
    output_dir: str = None,
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Recalculate metrics from pre-saved CSV files.
    
    Args:
        input_dir: Directory containing results_fold_*.csv files
        output_dir: Output directory for recalculated metrics (defaults to input_dir)
        n_folds: Number of folds to look for
        
    Returns:
        Dictionary with CV summary
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_fold_metrics = []
    all_results = []
    
    print(f"\nRecalculating metrics from: {input_dir}")
    print("=" * 60)
    
    for fold in range(1, n_folds + 1):
        csv_path = os.path.join(input_dir, f"results_fold_{fold}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping fold {fold}")
            continue
        
        print(f"\nProcessing Fold {fold}: {csv_path}")
        
        # Load results from CSV
        results = load_results_from_csv(csv_path)
        print(f"  Loaded {len(results)} results")
        
        # Calculate metrics
        metrics = calculate_metrics_from_dicts(results)
        metrics['fold'] = fold
        all_fold_metrics.append(metrics)
        all_results.extend(results)
        
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
        print(f"  AUC: {metrics.get('auc', 0):.4f}")
        print(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
        print(f"  Specificity: {metrics.get('specificity', 0):.4f}")
    
    if not all_fold_metrics:
        return {'error': 'No fold results found'}
    
    # Aggregate metrics
    cv_summary = aggregate_cv_metrics(all_fold_metrics)
    
    # Save summary
    summary_path = os.path.join(output_dir, "cv_summary_recalculated.json")
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\nRecalculated summary saved to: {summary_path}")
    
    # Also calculate overall metrics (all folds combined)
    overall_metrics = calculate_metrics_from_dicts(all_results)
    overall_path = os.path.join(output_dir, "overall_metrics_recalculated.json")
    with open(overall_path, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    print(f"Overall metrics saved to: {overall_path}")
    
    return cv_summary


# =============================================================================
# CV EVALUATION FUNCTIONS
# =============================================================================

def run_cv_evaluation(
    evaluator: CombinedEvaluator,
    n_folds: int = 5,
    output_dir: str = None,
    fold: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run cross-validation evaluation.
    
    Args:
        evaluator: CombinedEvaluator instance
        n_folds: Number of folds (when fold is None)
        output_dir: Output directory for results
        fold: Optional specific fold to evaluate (1-5). If provided, only this fold is evaluated.
        
    Returns:
        Dictionary with CV summary
    """
    if output_dir is None:
        output_dir = os.path.join(result_folder, "combined_cv_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_fold_metrics = []
    all_results = []
    
    # Determine which folds to evaluate
    if fold is not None:
        # Evaluate only the specified fold
        folds_to_evaluate = [fold]
    else:
        # Evaluate all folds
        folds_to_evaluate = range(1, n_folds + 1)
    
    for fold in folds_to_evaluate:
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
        
        # Set fold (loads bridges)
        evaluator.set_fold(fold)
        
        # Get test patient IDs
        test_pids = evaluator.bridge_manager.get_test_pids(fold)

        # Shuffle test PIDs for randomness
        random.shuffle(test_pids)
        
        if is_debug:
            test_pids = test_pids[:2]
        
        print(f"Test patients: {len(test_pids)}")
        
        # Evaluate
        results = evaluator.evaluate_all(test_pids, show_progress=True)
        
        # Save fold results
        fold_output_path = os.path.join(output_dir, f"results_fold_{fold}.csv")
        evaluator.save_results(results, fold_output_path)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results)
        metrics['fold'] = fold
        all_fold_metrics.append(metrics)
        all_results.extend(results)
        
        print(f"\nFold {fold} Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
        print(f"  AUC: {metrics.get('auc', 0):.4f}")
    
    # Aggregate metrics
    cv_summary = aggregate_cv_metrics(all_fold_metrics)
    
    # Save summary
    summary_path = os.path.join(output_dir, "cv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    return cv_summary


def aggregate_cv_metrics(fold_metrics: List[Dict]) -> Dict[str, Any]:
    """Aggregate metrics across CV folds."""
    if not fold_metrics:
        return {'error': 'No fold metrics available'}
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'sensitivity', 'specificity']
    
    summary = {
        'n_folds': len(fold_metrics),
        'fold_metrics': fold_metrics,
    }
    
    for metric in metric_names:
        values = [fm.get(metric, 0) for fm in fold_metrics if metric in fm]
        if values:
            summary[f'mean_{metric}'] = float(np.mean(values))
            summary[f'std_{metric}'] = float(np.std(values))
    
    return summary


def print_cv_summary(cv_summary: Dict) -> None:
    """Print CV summary."""
    print("\n" + "=" * 60)
    print("COMBINED CV EVALUATION SUMMARY")
    print("=" * 60)
    
    if 'error' in cv_summary:
        print(f"\nError: {cv_summary['error']}")
        return
    
    print(f"\nFolds: {cv_summary.get('n_folds', 0)}")
    print(f"\nMetrics (mean ± std):")
    print(f"  Accuracy:    {cv_summary.get('mean_accuracy', 0):.4f} ± {cv_summary.get('std_accuracy', 0):.4f}")
    print(f"  Precision:   {cv_summary.get('mean_precision', 0):.4f} ± {cv_summary.get('std_precision', 0):.4f}")
    print(f"  Recall:      {cv_summary.get('mean_recall', 0):.4f} ± {cv_summary.get('std_recall', 0):.4f}")
    print(f"  F1-Score:    {cv_summary.get('mean_f1_score', 0):.4f} ± {cv_summary.get('std_f1_score', 0):.4f}")
    print(f"  AUC:         {cv_summary.get('mean_auc', 0):.4f} ± {cv_summary.get('std_auc', 0):.4f}")
    print(f"  Sensitivity: {cv_summary.get('mean_sensitivity', 0):.4f} ± {cv_summary.get('std_sensitivity', 0):.4f}")
    print(f"  Specificity: {cv_summary.get('mean_specificity', 0):.4f} ± {cv_summary.get('std_specificity', 0):.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    default_llm_provider = os.getenv("ACTIVE_LLM_PROVIDER", "google")
    default_llm_model = os.getenv("ACTIVE_LLM_MODEL", "gemini-2.5-pro")
    
    parser = argparse.ArgumentParser(
        description="Combined Multi-Modal Dementia Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_all.py                              # Run 5-fold CV evaluation
    python main_all.py --fold 1                     # Evaluate fold 1 only
    python main_all.py --evaluate-single P001       # Evaluate single patient
    python main_all.py --evaluate-single P001 --fold 2  # Evaluate single patient using fold 2
    python main_all.py --provider openai --model gpt-4o
    python main_all.py --check-bridges              # Check bridge status
    python main_all.py --recalculate-from datasets/results/combined_cv_results
                                                    # Recalculate metrics from saved CSVs
    
Prerequisites:
    1. Run: python train_unified_cv_folds.py
    2. Run: python train_ml_acoustic_bridge_nested_cv.py --use-unified-folds
    3. Run: python train_ml_semantic_bridge_nested_cv.py --use-unified-folds
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run 5-fold CV evaluation (default)"
    )
    mode_group.add_argument(
        "--evaluate-single",
        type=str,
        metavar="PID",
        help="Evaluate a single patient"
    )
    mode_group.add_argument(
        "--check-bridges",
        action="store_true",
        help="Check ML bridge status"
    )
    mode_group.add_argument(
        "--list-patients",
        action="store_true",
        help="List available patients"
    )
    mode_group.add_argument(
        "--recalculate",
        action="store_true",
        help="Recalculate metrics from pre-saved CSV files in output-dir (skip full evaluation)"
    )
    
    # LLM configuration
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "google", "deepseek", "ollama"],
        default=default_llm_provider,
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_llm_model,
        help="Model name to use"
    )
    
    # Other options
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(result_folder, "combined_cv_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Specific fold to evaluate (1-5). If not provided, all 5 folds are evaluated."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle check-bridges mode
    if args.check_bridges:
        print("\n" + "=" * 60)
        print("ML BRIDGE STATUS CHECK")
        print("=" * 60)
        
        manager = CombinedBridgeManager()
        status = manager.check_status()
        
        print(f"\nData Status:")
        print(f"  Acoustic data loaded: {status['acoustic_data_loaded']} ({status['acoustic_data_count']} patients)")
        print(f"  Semantic data loaded: {status['semantic_data_loaded']} ({status['semantic_data_count']} patients)")
        print(f"  Unified folds available: {status['unified_folds_available']}")
        
        print(f"\nFold Status:")
        for fold in range(1, 6):
            acoustic_ok = "✓" if status[f'fold_{fold}_acoustic'] else "✗"
            semantic_ok = "✓" if status[f'fold_{fold}_semantic'] else "✗"
            unified_ok = "✓" if status[f'fold_{fold}_unified'] else "✗"
            print(f"  Fold {fold}: Acoustic={acoustic_ok}, Semantic={semantic_ok}, Unified={unified_ok}")
        
        if not status['unified_folds_available']:
            print("\n⚠ Unified folds not found!")
            print("  Run: python train_unified_cv_folds.py")
        
        return
    
    # Handle list-patients mode
    if args.list_patients:
        manager = CombinedBridgeManager()
        
        print("\n" + "=" * 60)
        print("AVAILABLE PATIENTS")
        print("=" * 60)
        
        # Find common patients
        acoustic_pids = set()
        semantic_pids = set()
        
        if manager.acoustic_df is not None:
            id_col = 'participant_id' if 'participant_id' in manager.acoustic_df.columns else 'PID'
            acoustic_pids = set(manager.acoustic_df[id_col].astype(str).unique())
        
        if manager.semantic_df is not None:
            id_col = 'PID' if 'PID' in manager.semantic_df.columns else 'participant_id'
            semantic_pids = set(manager.semantic_df[id_col].astype(str).unique())
        
        common_pids = acoustic_pids.intersection(semantic_pids)
        
        print(f"\nAcoustic patients: {len(acoustic_pids)}")
        print(f"Semantic patients: {len(semantic_pids)}")
        print(f"Common patients (for combined evaluation): {len(common_pids)}")
        
        if len(common_pids) > 0:
            print(f"\nCommon patients:")
            for pid in sorted(common_pids)[:20]:
                print(f"  {pid}")
            if len(common_pids) > 20:
                print(f"  ... and {len(common_pids) - 20} more")
        
        return
    
    # Handle recalculate mode
    if args.recalculate:
        print("\n" + "=" * 60)
        print("RECALCULATE METRICS FROM CSV FILES")
        print("=" * 60)
        
        input_dir = args.output_dir
        
        if not os.path.isdir(input_dir):
            print(f"\nError: Directory not found: {input_dir}")
            print(f"Please ensure the results directory exists or specify with --output-dir")
            return
        
        # Check for CSV files
        csv_files = [f for f in os.listdir(input_dir) if f.startswith('results_fold_') and f.endswith('.csv')]
        if not csv_files:
            print(f"\nError: No results_fold_*.csv files found in {input_dir}")
            return
        
        print(f"\nFound {len(csv_files)} fold result files in {input_dir}:")
        for f in sorted(csv_files):
            print(f"  - {f}")
        
        # Recalculate metrics
        cv_summary = recalculate_metrics_from_csv(
            input_dir=input_dir,
            output_dir=input_dir,
            n_folds=5
        )
        
        print_cv_summary(cv_summary)
        
        return
    
    # Load config
    config = load_config(args.env)
    
    # Initialize evaluator
    evaluator = CombinedEvaluator(config=config)
    evaluator.setup_provider(args.provider, args.model)
    
    print(f"\nUsing: {args.provider}/{args.model}")
    
    # Handle single patient evaluation
    if args.evaluate_single:
        fold = args.fold or 1
        print(f"\nEvaluating patient: {args.evaluate_single} (using fold {fold})")
        
        evaluator.set_fold(fold)
        result = evaluator.evaluate_patient(args.evaluate_single)
        
        print(f"\nPrediction: {result.prediction.value}")
        print(f"Control confidence: {result.control_conf}")
        print(f"Impairment confidence: {result.impairment_conf}")
        
        if result.explanation:
            print(f"\nExplanation:\n{result.explanation}")
        if result.error:
            print(f"\nError: {result.error}")
        
        return
    
    # Default: Run CV evaluation
    print("\n" + "=" * 60)
    print("COMBINED MULTI-MODAL EVALUATION")
    print("=" * 60)
    
    # Check prerequisites
    status = evaluator.bridge_manager.check_status()
    
    if not status['unified_folds_available']:
        print("\n⚠ ERROR: Unified CV folds not found!")
        print("Please run the following first:")
        print("  1. python train_unified_cv_folds.py")
        print("  2. python train_ml_acoustic_bridge_nested_cv.py --use-unified-folds")
        print("  3. python train_ml_semantic_bridge_nested_cv.py --use-unified-folds")
        return
    
    # Run CV evaluation
    if args.fold:
        print(f"\nEvaluating fold {args.fold} only")
    else:
        print(f"\nEvaluating all 5 folds")
    
    cv_summary = run_cv_evaluation(
        evaluator=evaluator,
        n_folds=5,
        output_dir=args.output_dir,
        fold=args.fold
    )
    
    print_cv_summary(cv_summary)
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()