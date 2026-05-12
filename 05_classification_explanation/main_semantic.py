#!/usr/bin/env python3
"""
main_semantic.py

Semantic Feature Evaluation System - Main Entry Point (ML Bridge Version)
=========================================================================

This script provides a command-line interface for evaluating patients
for cognitive impairment using semantic/linguistic features with ML-enhanced
classification.

Two Modes:
1. Feature Extraction: Extract semantic features from transcripts using LLM
2. ML-Enhanced Evaluation: Use pre-trained ML models + LLM for final classification

Usage:
    # Feature extraction (first step)
    python main_semantic.py --extract-features
    
    # Train ML models (after feature extraction)
    python train_ml_semantic_bridge_nested_cv.py
    
    # Run evaluation with ML bridge (5-fold CV)
    python main_semantic.py --evaluate
    
    # Evaluate specific patients
    python main_semantic.py --evaluate --patient-ids P001 P002
    
    # Use specific LLM provider/model
    python main_semantic.py --evaluate --provider openai --model gpt-4o
"""

import argparse
import logging
import sys
import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get_active_llm_config
from data_loader import DataLoader
from llm_providers import get_provider, LLMProvider
from models import EvaluationResult, DiagnosisResult

# Import semantic-specific modules
from semantic_clinical_profile import (
    set_current_fold,
    get_current_fold,
    get_test_patient_ids,
    get_cv_summary,
    get_bridge,
    analyze_patient_for_llm,
    SemanticClinicalProfile,
    PopulationStatsBuilder,
)
from prompts import (
    get_system_prompt,
    create_evaluation_prompt,
    get_semantic_system_prompt,
)

# Configuration
is_debug = os.getenv("DEBUG", "").lower() == "true"
result_folder = os.getenv("RESULTS_DIR", "datasets/results")
semantic_feature_folder = os.getenv("SEMANTIC_FEATURE_FOLDER", "datasets/output/semantic_features/")


stop_reasoning_env = os.getenv("STOPREASOING", "false").lower()=="true"
is_reasoning = not stop_reasoning_env  # Send to LLM unless explicitly stopped

logger = logging.getLogger(__name__)


def calculate_probability_softmax(imp_conf: float, ctrl_conf: float) -> float:
    """
    Calculate probability of impairment using softmax normalization.
    
    This properly handles the case where we have both impairment_conf and control_conf
    on a 1-10 scale that sum to 10.
    
    Args:
        imp_conf: Impairment confidence score (1-10 scale)
        ctrl_conf: Control confidence score (1-10 scale)
        
    Returns:
        Probability of impairment (0-1 scale)
    """
    try:
        # SOFTMAX normalization for proper 0-1 probability
        # This properly normalizes to 0-1 and handles scale differences
        imp_exp = np.exp(float(imp_conf))
        ctrl_exp = np.exp(float(ctrl_conf))
        prob = imp_exp / (imp_exp + ctrl_exp)
        
        # Ensure valid probability
        prob = max(0.0, min(1.0, prob))
        return prob
        
    except (ValueError, TypeError):
        # Fallback if confidence values are invalid
        return 0.5
    

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# =============================================================================
# SEMANTIC EVALUATOR CLASS
# =============================================================================

class SemanticMLEvaluator:
    """
    Evaluator that uses ML Semantic Bridge for classification.
    
    Workflow:
    1. Load pre-extracted semantic features from CSV
    2. Generate ML-based risk scores and domain breakdown
    3. Send to LLM for final classification with explanation
    """
    
    def __init__(self, config=None):
        """Initialize the evaluator."""
        self.config = config or load_config()
        self.provider: Optional[LLMProvider] = None
        self.data_loader = DataLoader(self.config.dataset)
        self.results: List[EvaluationResult] = []
        
        # Load semantic features
        self.features_df = None
        self._load_semantic_features()
    
    def _load_semantic_features(self) -> None:
        """Load pre-extracted semantic features from CSV."""
        features_path = os.path.join(
            semantic_feature_folder,
            "merged_semantic_data.csv"
        )
        
        if os.path.exists(features_path):
            self.features_df = pd.read_csv(features_path)
            logger.info(f"Loaded {len(self.features_df)} patient records from {features_path}")
        else:
            logger.warning(f"Semantic features file not found: {features_path}")
            self.features_df = pd.DataFrame()
    
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
    
    def get_patient_features(self, pid: str) -> Optional[Dict[str, Any]]:
        """Get semantic features for a patient."""
        if self.features_df is None or self.features_df.empty:
            return None
        
        patient_rows = self.features_df[self.features_df['PID'] == pid]
        
        if patient_rows.empty:
            return None
        
        return patient_rows.iloc[0].to_dict()
    
    def evaluate_patient(self, pid: str, variant: str = "transcript") -> EvaluationResult:
        """
        Evaluate a single patient using ML bridge + LLM.
        
        Args:
            pid: Patient ID
            variant: Evaluation variant ("transcript" or "standard")
            
        Returns:
            EvaluationResult
        """
        import time
        start_time = time.time()
        
        # Get patient features
        patient_features = self.get_patient_features(pid)
        
        if patient_features is None:
            return EvaluationResult(
                pid=pid,
                prediction=DiagnosisResult.UNKNOWN,
                explanation="",
                error="Patient features not found"
            )
        
        # Generate ML bridge analysis
        try:
            pre_diagnosis, semantic_section = analyze_patient_for_llm(
                patient_features, pid
            )
        except Exception as e:
            logger.error(f"ML bridge analysis failed for {pid}: {e}")
            return EvaluationResult(
                pid=pid,
                prediction=DiagnosisResult.UNKNOWN,
                explanation="",
                error=f"ML bridge analysis failed: {str(e)}"
            )
        
        # Build demographic section
        age = patient_features.get('age', 'Unknown')
        gender = patient_features.get('gender', 'Unknown')
        demographic_section = f"Age: {age}, Gender: {gender}"
        
        # Create prompt
        user_prompt = create_evaluation_prompt(
            variant=variant,
            demographic_sections=[demographic_section],
            semantic_sections=semantic_section,
            pre_diagnosis=pre_diagnosis
        )
        
        system_prompt = get_semantic_system_prompt()
        

        if is_debug:
            print(f"\n{'='*60}")
            print(f"Patient: {pid}")
            print(f"{'='*60}")
            print(f"System Prompt: {system_prompt[:200]}...")
            print(f"User Prompt: {user_prompt[:500]}...")
        
        # Save prompt to file (for debugging or when not sending to LLM)
        if is_debug or not is_reasoning:
            save_to_txt(f"System: {system_prompt}\n\n User: \n{user_prompt}", f"{pid}.txt", result_folder)
            print(f"   Prompt saved to: {result_folder}/{pid}.txt")

        # Call LLM
        try:
            if is_reasoning:
                result = self.provider.evaluate(
                    pid=pid,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                result.processing_time_seconds = time.time() - start_time
            else:   
                result = EvaluationResult(
                    pid=pid,
                    prediction=DiagnosisResult.UNKNOWN,
                    explanation="",
                    processing_time_seconds=time.time() - start_time,
                    error="Reasoning disabled; no LLM call made."
                )
            return result
            
        except Exception as e:
            logger.error(f"LLM evaluation failed for {pid}: {e}")
            return EvaluationResult(
                pid=pid,
                prediction=DiagnosisResult.UNKNOWN,
                explanation="",
                processing_time_seconds=time.time() - start_time,
                error=f"LLM evaluation failed: {str(e)}"
            )
    
    def evaluate_all(
        self,
        patient_ids: List[str],
        variant: str = "transcript",
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple patients.
        
        Args:
            patient_ids: List of patient IDs
            variant: Evaluation variant
            show_progress: Whether to show progress bar
            
        Returns:
            List of EvaluationResult
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(patient_ids, desc="Evaluating")
            except ImportError:
                iterator = patient_ids
        else:
            iterator = patient_ids
        
        for pid in iterator:
            result = self.evaluate_patient(pid, variant)
            results.append(result)
            
            if is_debug:
                gt = self.data_loader.get_groundtruth_diagnosis(pid)
                print(f"  {pid}: Predicted={result.prediction.value}, GT={gt}")
        
        self.results = results
        return results
    
    def save_results(
        self,
        results: List[EvaluationResult] = None,
        output_path: str = None
    ) -> str:
        """Save evaluation results to CSV."""
        results = results or self.results
        
        if not results:
            logger.warning("No results to save")
            return ""
        
        # Build output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fold = get_current_fold()
            output_path = os.path.join(
                result_folder,
                f"semantic_results_fold_{fold}_{timestamp}.csv"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build dataframe
        rows = []
        for result in results:
            row = {
                'PID': result.pid,
                'Prediction': result.prediction.value,
                'impairment_conf': getattr(result, 'impairment_conf', None),
                'control_conf': getattr(result, 'control_conf', None),
                'Explanation': result.explanation[:500] if result.explanation else '',
                'ProcessingTime': result.processing_time_seconds,
                'Model': result.model_name,
                'Provider': result.provider,
                'Error': result.error,
            }
            
            # Add ground truth
            gt = self.data_loader.get_groundtruth_diagnosis(result.pid)
            row['GroundTruth'] = gt
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def calculate_metrics(
        self,
        results: List[EvaluationResult] = None
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics with corrected AUC using softmax."""
        results = results or self.results
        
        if not results:
            return {}
        
        y_true = []
        y_pred = []
        y_prob = []  # Changed from y_scores to y_prob for clarity
        
        for result in results:
            gt = self.data_loader.get_groundtruth_diagnosis(result.pid)
            
            if not gt or result.prediction.value == "Unknown":
                continue
            
            gt_binary = 1 if gt.lower() == "impairment" else 0
            pred_binary = 1 if result.prediction.value.lower() == "impairment" else 0
            
            y_true.append(gt_binary)
            y_pred.append(pred_binary)
            
            # **CRITICAL: Calculate PROBABILITY of impairment using SOFTMAX**
            # We have BOTH impairment_conf and control_conf (1-10 scale)
            try:
                imp_conf = float(getattr(result, 'impairment_conf', 5) or 5)
                ctrl_conf = float(getattr(result, 'control_conf', 5) or 5)
                
                # SOFTMAX normalization for proper 0-1 probability
                imp_exp = np.exp(imp_conf)
                ctrl_exp = np.exp(ctrl_conf)
                prob = imp_exp / (imp_exp + ctrl_exp)
                
                # Ensure valid probability
                prob = max(0.0, min(1.0, prob))
                
            except (ValueError, TypeError, AttributeError):
                # Fallback if confidence values are missing/invalid
                if pred_binary == 1:  # Predicted impairment
                    prob = 0.7
                else:  # Predicted control
                    prob = 0.3
            
            y_prob.append(prob)
        
        if not y_true:
            return {}
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate AUC with softmax probabilities

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = (sensitivity + specificity) / 2

        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total': len(y_true),
            'support_impairment': int(tp + fn),
            'support_control': int(tn + fp),
        }

def save_to_txt(content, filename, folder):
    """Save content to text file."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(str(content))

def recalculate_cv_summary_from_files(
    output_dir: str = None,
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Recalculate CV summary from previously saved fold CSV files.
    
    FIXED: Uses softmax with both impairment_conf and control_conf for AUC calculation.
    
    Args:
        output_dir: Directory containing semantic_results_fold_*.csv files
        n_folds: Number of folds to look for
        
    Returns:
        CV summary dictionary
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    
    output_dir = output_dir or result_folder
    fold_results = []
    
    for fold in range(1, n_folds + 1):
        csv_path = os.path.join(output_dir, f"semantic_results_fold_{fold}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: Fold {fold} CSV not found: {csv_path}")
            continue
        
        print(f"Loading fold {fold} from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Extract y_true, y_pred, and calculate y_prob from CSV
        y_true = []
        y_pred = []
        y_prob = []  # CHANGED from y_scores
        
        for _, row in df.iterrows():
            gt = row.get('GroundTruth', '')
            pred = row.get('Prediction', '')
            
            if pd.isna(gt) or pd.isna(pred) or pred == 'Unknown':
                continue
            
            gt_binary = 1 if gt == 'Impairment' else 0
            pred_binary = 1 if pred == 'Impairment' else 0
            
            y_true.append(gt_binary)
            y_pred.append(pred_binary)
            
            # **CRITICAL: Calculate probability using SOFTMAX**
            try:
                imp_conf = float(row.get('impairment_conf', 5))
                ctrl_conf = float(row.get('control_conf', 5))
                
                # Handle NaN values
                if pd.isna(imp_conf):
                    imp_conf = 5.0
                if pd.isna(ctrl_conf):
                    ctrl_conf = 5.0
                
                # SOFTMAX normalization
                imp_exp = np.exp(imp_conf)
                ctrl_exp = np.exp(ctrl_conf)
                prob = imp_exp / (imp_exp + ctrl_exp)
                
                # Ensure valid probability
                prob = max(0.0, min(1.0, prob))
                
            except (ValueError, KeyError, TypeError):
                # Fallback if confidence values are missing/invalid
                if pred_binary == 1:
                    prob = 0.7
                else:
                    prob = 0.3
            
            y_prob.append(prob)
        
        if not y_true:
            print(f"Warning: No valid predictions in fold {fold}")
            continue
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Use labels=[0, 1] to ensure 2x2 confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate AUC with SOFTMAX probabilities

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = (sensitivity + specificity) / 2

        
        fold_results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'total': len(y_true),
        })
        
        print(f"  Fold {fold}: Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Calculate CV summary
    if fold_results:
        cv_summary = {
            'n_folds': len(fold_results),
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_precision': np.mean([r['precision'] for r in fold_results]),
            'std_precision': np.std([r['precision'] for r in fold_results]),
            'mean_sensitivity': np.mean([r['sensitivity'] for r in fold_results]),
            'std_sensitivity': np.std([r['sensitivity'] for r in fold_results]),
            'mean_specificity': np.mean([r['specificity'] for r in fold_results]),
            'std_specificity': np.std([r['specificity'] for r in fold_results]),
            'mean_f1': np.mean([r['f1_score'] for r in fold_results]),
            'std_f1': np.std([r['f1_score'] for r in fold_results]),
            'mean_auc': np.mean([r['auc'] for r in fold_results]),
            'std_auc': np.std([r['auc'] for r in fold_results]),
            'fold_results': fold_results,
            'timestamp': datetime.now().isoformat(),
            'source': 'recalculated_from_csv_with_softmax_auc',
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "semantic_cv_evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
        
        return cv_summary
    
    return {}

# =============================================================================
# CV WORKFLOW FUNCTIONS
# =============================================================================

def run_nested_cv_evaluation(
    evaluator: SemanticMLEvaluator,
    variant: str = "transcript",
    n_folds: int = 5,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run 5-fold cross-validation evaluation.
    
    Args:
        evaluator: SemanticMLEvaluator instance
        variant: Evaluation variant
        n_folds: Number of folds
        output_dir: Output directory
        
    Returns:
        CV summary dictionary
    """
    output_dir = output_dir or result_folder
    
    fold_results = []
    all_results = []
    
    for fold in range(1, n_folds + 1):
        print(f"\n{'='*60}")
        print(f"EVALUATING FOLD {fold}")
        print(f"{'='*60}")
        
        # Set current fold (loads correct model)
        set_current_fold(fold)
        
        # Get test patient IDs for this fold
        patient_ids = get_test_patient_ids(fold)
        
        if not patient_ids:
            print(f"Warning: No test patients found for fold {fold}")
            continue
        
        if is_debug:
            patient_ids = patient_ids[:3]  # Limit for debugging
        
        print(f"Test patients: {len(patient_ids)}")
        
        # Run evaluation
        results = evaluator.evaluate_all(patient_ids=patient_ids, variant=variant, show_progress=True)
        
        # Calculate fold metrics
        metrics = evaluator.calculate_metrics(results)
        
        if metrics:
            fold_results.append({
                'fold': fold,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'f1_score': metrics['f1_score'],
                'auc': metrics['auc'],
                'tp': metrics['true_positives'],
                'tn': metrics['true_negatives'],
                'fp': metrics['false_positives'],
                'fn': metrics['false_negatives'],
                'total': metrics['total'],
            })
            
            print(f"\nFold {fold} Results:")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"  F1-Score:    {metrics['f1_score']:.4f}")
            print(f"  AUC:         {metrics['auc']:.4f}")
        
        # Save fold results
        output_path = os.path.join(output_dir, f"semantic_results_fold_{fold}.csv")
        evaluator.save_results(results, output_path)
        
        all_results.extend(results)
    
    # Calculate CV summary
    if fold_results:
        cv_summary = {
            'n_folds': len(fold_results),
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_precision': np.mean([r['precision'] for r in fold_results]),
            'std_precision': np.std([r['precision'] for r in fold_results]),
            'mean_sensitivity': np.mean([r['sensitivity'] for r in fold_results]),
            'std_sensitivity': np.std([r['sensitivity'] for r in fold_results]),
            'mean_specificity': np.mean([r['specificity'] for r in fold_results]),
            'std_specificity': np.std([r['specificity'] for r in fold_results]),
            'mean_f1': np.mean([r['f1_score'] for r in fold_results]),
            'std_f1': np.std([r['f1_score'] for r in fold_results]),
            'mean_auc': np.mean([r['auc'] for r in fold_results]),
            'std_auc': np.std([r['auc'] for r in fold_results]),
            'fold_results': fold_results,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 70)
        print(f"\nTest Performance (mean ± std across {len(fold_results)} folds):")
        print(f"  Accuracy:    {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"  Precision:   {cv_summary['mean_precision']:.4f} ± {cv_summary['std_precision']:.4f}")
        print(f"  Sensitivity: {cv_summary['mean_sensitivity']:.4f} ± {cv_summary['std_sensitivity']:.4f}")
        print(f"  Specificity: {cv_summary['mean_specificity']:.4f} ± {cv_summary['std_specificity']:.4f}")
        print(f"  F1-Score:    {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        print(f"  AUC:         {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "semantic_cv_evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
        
        return cv_summary
    
    return {}


def print_cv_summary(cv_summary: Dict[str, Any]) -> None:
    """Print CV summary in a formatted way."""
    if not cv_summary:
        print("No CV summary available")
        return
    
    print("\n" + "=" * 70)
    print("SEMANTIC EVALUATION CV SUMMARY")
    print("=" * 70)
    
    print(f"\nFolds evaluated: {cv_summary.get('n_folds', 'N/A')}")
    print(f"\nMetric          Mean ± Std")
    print("-" * 40)
    print(f"Accuracy:       {cv_summary.get('mean_accuracy', 0):.4f} ± {cv_summary.get('std_accuracy', 0):.4f}")
    print(f"Precision:      {cv_summary.get('mean_precision', 0):.4f} ± {cv_summary.get('std_precision', 0):.4f}")
    print(f"Sensitivity:    {cv_summary.get('mean_sensitivity', 0):.4f} ± {cv_summary.get('std_sensitivity', 0):.4f}")
    print(f"Specificity:    {cv_summary.get('mean_specificity', 0):.4f} ± {cv_summary.get('std_specificity', 0):.4f}")
    print(f"F1-Score:       {cv_summary.get('mean_f1', 0):.4f} ± {cv_summary.get('std_f1', 0):.4f}")
    print(f"AUC:            {cv_summary.get('mean_auc', 0):.4f} ± {cv_summary.get('std_auc', 0):.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    default_llm_provider = os.getenv("ACTIVE_LLM_PROVIDER", "google")
    default_llm_model = os.getenv("ACTIVE_LLM_MODEL", "gemini-2.5-pro")
    variant = os.getenv("PROMPTCONTENT", "transcript").lower()
    
    parser = argparse.ArgumentParser(
        description="Semantic Feature Evaluation System (ML Bridge Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 5-fold CV evaluation
    python main_semantic.py --evaluate
    
    # Evaluate specific patients
    python main_semantic.py --evaluate --patient-ids P001 P002
    
    # Use specific LLM
    python main_semantic.py --evaluate --provider openai --model gpt-4o
    
    # Check ML bridge status
    python main_semantic.py --check-bridge
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run ML-enhanced evaluation (5-fold CV)"
    )
    mode_group.add_argument(
        "--evaluate-single",
        type=str,
        metavar="PID",
        help="Evaluate a single patient"
    )
    mode_group.add_argument(
        "--check-bridge",
        action="store_true",
        help="Check ML bridge status and CV summary"
    )
    mode_group.add_argument(
        "--list-patients",
        action="store_true",
        help="List available patients in semantic features"
    )    
    mode_group.add_argument(
        "--recalculate",
        action="store_true",
        help="Recalculate CV metrics from saved fold CSV files"
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
    
    # Input/output
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=result_folder,
        help="Output directory for results"
    )
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        type=str,
        help="Specific patient IDs to evaluate"
    )
    
    # Options
    parser.add_argument(
        "--variant",
        type=str,
        default=variant,
        choices=["transcript", "standard"],
        help="Evaluation variant"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load config
    config = load_config(args.env)
    
    # Handle modes
    if args.check_bridge:
        print("\n" + "=" * 60)
        print("ML SEMANTIC BRIDGE STATUS")
        print("=" * 60)
        
        bridge = get_bridge()
        if bridge:
            print(f"\n✓ ML Bridge loaded successfully")
            print(f"  Threshold: {bridge.classification_threshold}")
            print(f"  Features: {len(bridge.feature_names)}")
            print(f"  Models: {list(bridge.models.keys())}")
        else:
            print(f"\n✗ ML Bridge not available")
            print(f"  Run train_ml_semantic_bridge_nested_cv.py first")
        
        summary = get_cv_summary()
        if summary:
            print(f"\n✓ CV Summary found")
            print(f"  Folds: {summary.get('n_folds', 'N/A')}")
            print(f"  Mean AUC: {summary.get('mean_test_auc', 'N/A'):.4f}")
            print(f"  Mean Accuracy: {summary.get('mean_test_accuracy', 'N/A'):.4f}")
        else:
            print(f"\n✗ CV Summary not found")
        
        return
    
    if args.list_patients:
        features_path = os.path.join(semantic_feature_folder, "merged_semantic_data.csv")
        if os.path.exists(features_path):
            df = pd.read_csv(features_path)
            print(f"\nPatients in semantic features: {len(df)}")
            print("-" * 40)
            for pid in sorted(df['PID'].tolist()):
                diag = df[df['PID'] == pid]['diagnosis'].iloc[0]
                print(f"  {pid}: {diag}")
        else:
            print(f"Semantic features file not found: {features_path}")
        return
    
    if args.recalculate:
        print("\n" + "=" * 60)
        print("RECALCULATING CV METRICS FROM SAVED FILES")
        print("=" * 60)
        
        cv_summary = recalculate_cv_summary_from_files(
            output_dir=args.output_dir,
            n_folds=5
        )
        
        print_cv_summary(cv_summary)
        return
    
    if args.evaluate_single:
        # Single patient evaluation
        print(f"\nEvaluating patient: {args.evaluate_single}")
        
        evaluator = SemanticMLEvaluator(config=config)
        evaluator.setup_provider(args.provider, args.model)
        
        result = evaluator.evaluate_patient(args.evaluate_single, args.variant)
        
        print(f"\nPrediction: {result.prediction.value}")
        if result.explanation:
            print(f"\nExplanation:\n{result.explanation}")
        if result.error:
            print(f"\nError: {result.error}")
        
        return
    
    # Default: Run 5-fold CV evaluation
    print("\n" + "=" * 60)
    print("SEMANTIC ML EVALUATION SYSTEM")
    print("=" * 60)
    
    # Check ML bridge
    bridge = get_bridge()
    if bridge is None:
        print("\n⚠ WARNING: ML Bridge not loaded!")
        print("Run train_ml_semantic_bridge_nested_cv.py first to train models.")
        print("Falling back to z-score mode (less accurate).")
    
    # Initialize evaluator
    evaluator = SemanticMLEvaluator(config=config)
    evaluator.setup_provider(args.provider, args.model)
    
    print(f"\nUsing: {args.provider}/{args.model}")
    print(f"Variant: {args.variant}")
    
    # Run evaluation
    if args.patient_ids:
        # Evaluate specific patients
        print(f"\nEvaluating {len(args.patient_ids)} specified patients...")
        
        results = evaluator.evaluate_all(
            patient_ids=args.patient_ids,
            variant=args.variant,
            show_progress=not args.no_progress
        )
        
        output_path = evaluator.save_results(results)
        metrics = evaluator.calculate_metrics(results)
        
        if metrics:
            print(f"\nAccuracy: {metrics['accuracy']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
    else:
        # Run 5-fold CV
        print("\nRunning 5-fold cross-validation...")
        
        cv_summary = run_nested_cv_evaluation(
            evaluator=evaluator,
            variant=args.variant,
            n_folds=5,
            output_dir=args.output_dir
        )
        
        print_cv_summary(cv_summary)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()