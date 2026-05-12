#!/usr/bin/env python3
"""
train_ml_semantic_bridge_nested_cv.py

Nested Cross-Validation Training for ML Semantic Bridge
========================================================

This script implements proper nested CV methodology for semantic features:
- Outer loop: 5-fold CV for final evaluation
- Inner loop: 4-fold CV within each training set for threshold optimization

This ensures NO data leakage - test data never influences threshold selection.

Usage:
    python train_ml_semantic_bridge_nested_cv.py                     # Run full nested CV
    python train_ml_semantic_bridge_nested_cv.py --fold 1            # Train only fold 1
    python train_ml_semantic_bridge_nested_cv.py --data custom.csv   # Use custom data

Output:
    - semantic_cv_folds/fold_1/train.csv, test.csv, ml_semantic_bridge.pkl
    - semantic_cv_folds/fold_2/...
    - semantic_cv_folds/cv_summary.json (thresholds and metrics per fold)
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from dotenv import load_dotenv
load_dotenv()

# Import from ml_semantic_bridge
from ml_semantic_bridge import (
    MLSemanticBridge,
    SEMANTIC_FEATURE_MAP,
    SEMANTIC_FEATURE_GROUPS,
    CANDIDATE_THRESHOLDS,
    IMPAIRMENT_LABELS,
    prepare_labels,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths
DEFAULT_DATA_PATH = os.path.join(
    os.getenv("SEMANTIC_FEATURE_FOLDER", "datasets/output/semantic_features"),
    "merged_semantic_data.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.getenv("RESULTS_DIR", "datasets/results"),
    "semantic_cv_folds"
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FoldResult:
    """Results for a single CV fold."""
    fold: int
    threshold: float
    train_size: int
    test_size: int
    train_auc: float
    test_auc: float
    test_accuracy: float
    test_sensitivity: float
    test_specificity: float
    test_f1: float
    test_precision: float
    test_tp: int
    test_tn: int
    test_fp: int
    test_fn: int
    inner_cv_thresholds: Dict[float, float]  # threshold -> inner CV F1


@dataclass
class CVSummary:
    """Summary of all CV folds."""
    n_folds: int
    timestamp: str
    data_path: str
    total_samples: int
    feature_count: int
    fold_results: List[FoldResult]
    mean_test_auc: float
    std_test_auc: float
    mean_test_accuracy: float
    std_test_accuracy: float
    mean_test_f1: float
    std_test_f1: float
    mean_test_sensitivity: float
    std_test_sensitivity: float
    mean_test_specificity: float
    std_test_specificity: float
    threshold_stability: Dict[float, int]  # threshold -> count of folds selecting it


# =============================================================================
# INNER CV FOR THRESHOLD OPTIMIZATION
# =============================================================================

def run_inner_cv_func(
    df_train: pd.DataFrame,
    label_column: str = 'diagnosis',
    n_inner_folds: int = 4,
    random_state: int = 42
) -> Tuple[float, Dict[float, float]]:
    """
    Run inner CV to find optimal classification threshold.
    
    Args:
        df_train: Training data
        label_column: Name of label column
        n_inner_folds: Number of inner CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (best_threshold, {threshold: f1_score})
    """
    y_train = prepare_labels(df_train, label_column)
    
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds,
        shuffle=True,
        random_state=random_state
    )
    
    # Store F1 scores for each threshold across all inner folds
    threshold_scores = {t: [] for t in CANDIDATE_THRESHOLDS}
    
    for inner_train_idx, inner_val_idx in inner_cv.split(df_train, y_train):
        inner_train = df_train.iloc[inner_train_idx]
        inner_val = df_train.iloc[inner_val_idx]
        
        y_inner_train = y_train[inner_train_idx]
        y_inner_val = y_train[inner_val_idx]
        
        # Train model on inner train set
        bridge = MLSemanticBridge()
        bridge.fit(inner_train, y_inner_train, train_group_models=False)
        
        # Get probabilities on validation set
        probas = bridge.predict_proba(inner_val)
        
        # Evaluate each threshold
        for thresh in CANDIDATE_THRESHOLDS:
            preds = (probas >= thresh).astype(int)
            f1 = f1_score(y_inner_val, preds, zero_division=0)
            threshold_scores[thresh].append(f1)
    
    # Average F1 for each threshold
    avg_scores = {t: np.mean(scores) for t, scores in threshold_scores.items()}
    
    # Find best threshold
    best_threshold = max(avg_scores.keys(), key=lambda t: avg_scores[t])
    
    return best_threshold, avg_scores


# =============================================================================
# FOLD TRAINING
# =============================================================================

def train_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fold_num: int,
    output_dir: str,
    label_column: str = 'diagnosis',
    id_column: str = 'PID',
    run_inner_cv: bool = True
) -> FoldResult:
    """
    Train a single fold and evaluate.
    
    Args:
        df_train: Training data
        df_test: Test data
        fold_num: Fold number (1-indexed)
        output_dir: Output directory
        label_column: Name of label column
        id_column: Name of ID column
        run_inner_cv: Whether to run inner CV for threshold optimization
        
    Returns:
        FoldResult with metrics
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num}")
    print(f"{'='*60}")
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")
    
    # Prepare labels
    y_train = prepare_labels(df_train, label_column)
    y_test = prepare_labels(df_test, label_column)
    
    print(f"Train distribution: Control={sum(y_train==0)}, Impairment={sum(y_train==1)}")
    print(f"Test distribution: Control={sum(y_test==0)}, Impairment={sum(y_test==1)}")
    
    # Run inner CV for threshold optimization
    inner_scores = {}
    if run_inner_cv:
        print("\nRunning inner CV for threshold optimization...")
        best_threshold, inner_scores = run_inner_cv_func(df_train, label_column)
        print(f"Best threshold from inner CV: {best_threshold:.2f}")
    else:
        best_threshold = 0.50
    
    # Train final model on full training set
    print("\nTraining final model...")
    bridge = MLSemanticBridge(classification_threshold=best_threshold)
    bridge.fit(df_train, y_train, train_group_models=True)
    
    # Evaluate on training set (for reference)
    train_probas = bridge.predict_proba(df_train)
    train_auc = roc_auc_score(y_train, train_probas)
    print(f"Train AUC: {train_auc:.4f}")
    
    # Evaluate on test set
    test_probas = bridge.predict_proba(df_test)
    test_preds = (test_probas >= best_threshold).astype(int)
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, test_probas)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)  # Sensitivity
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nTest Results:")
    print(f"  AUC:         {test_auc:.4f}")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  F1-Score:    {test_f1:.4f}")
    print(f"  Precision:   {test_precision:.4f}")
    print(f"  Sensitivity: {test_recall:.4f}")
    print(f"  Specificity: {test_spec:.4f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Create fold output directory
    fold_dir = os.path.join(output_dir, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Save train/test splits
    df_train.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
    
    # Save model
    model_path = os.path.join(fold_dir, "ml_semantic_bridge.pkl")
    bridge.save(model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save fold metadata
    metadata = {
        'fold': fold_num,
        'threshold': best_threshold,
        'train_size': len(df_train),
        'test_size': len(df_test),
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_sensitivity': float(test_recall),
        'test_specificity': float(test_spec),
        'inner_cv_thresholds': {str(k): float(v) for k, v in inner_scores.items()},
        'features_used': bridge.feature_names,
        'models_trained': list(bridge.models.keys()),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(fold_dir, "fold_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return FoldResult(
        fold=fold_num,
        threshold=best_threshold,
        train_size=len(df_train),
        test_size=len(df_test),
        train_auc=train_auc,
        test_auc=test_auc,
        test_accuracy=test_acc,
        test_sensitivity=test_recall,
        test_specificity=test_spec,
        test_f1=test_f1,
        test_precision=test_precision,
        test_tp=int(tp),
        test_tn=int(tn),
        test_fp=int(fp),
        test_fn=int(fn),
        inner_cv_thresholds=inner_scores,
    )


# =============================================================================
# MAIN NESTED CV
# =============================================================================

def run_nested_cv(
    data_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_outer_folds: int = 5,
    label_column: str = 'diagnosis',
    id_column: str = 'PID',
    random_state: int = 42,
    specific_fold: Optional[int] = None
) -> CVSummary:
    """
    Run full nested cross-validation for semantic features.
    
    Args:
        data_path: Path to merged_semantic_data.csv
        output_dir: Directory to save fold outputs
        n_outer_folds: Number of outer CV folds
        label_column: Name of diagnosis column
        id_column: Name of patient ID column
        random_state: Random seed
        specific_fold: If specified, only run this fold (1-indexed)
        
    Returns:
        CVSummary with all results
    """
    print("\n" + "=" * 70)
    print("NESTED CROSS-VALIDATION FOR ML SEMANTIC BRIDGE")
    print("=" * 70)
    print(f"\nData: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Outer folds: {n_outer_folds}")
    print(f"Inner folds: 4 (for threshold optimization)")
    
    # Load data
    df = pd.read_csv(data_path)
    y = prepare_labels(df, label_column)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"  Control: {sum(y == 0)}")
    print(f"  Impairment: {sum(y == 1)}")
    
    # Check available features
    available_features = [f for f in SEMANTIC_FEATURE_MAP if f in df.columns]
    print(f"\nAvailable features: {len(available_features)}/{len(SEMANTIC_FEATURE_MAP)}")
    
    missing_features = [f for f in SEMANTIC_FEATURE_MAP if f not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Outer CV
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds,
        shuffle=True,
        random_state=random_state
    )
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df, y)):
        fold_num = fold_idx + 1
        
        # Skip if specific fold requested
        if specific_fold is not None and fold_num != specific_fold:
            continue
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        result = train_fold(
            df_train=df_train,
            df_test=df_test,
            fold_num=fold_num,
            output_dir=output_dir,
            label_column=label_column,
            id_column=id_column,
            run_inner_cv=True
        )
        
        fold_results.append(result)
    
    # Summary statistics
    if len(fold_results) > 0:
        test_aucs = [r.test_auc for r in fold_results]
        test_accs = [r.test_accuracy for r in fold_results]
        test_f1s = [r.test_f1 for r in fold_results]
        test_sens = [r.test_sensitivity for r in fold_results]
        test_specs = [r.test_specificity for r in fold_results]
        
        # Threshold stability
        threshold_counts = {}
        for r in fold_results:
            t = r.threshold
            threshold_counts[t] = threshold_counts.get(t, 0) + 1
        
        summary = CVSummary(
            n_folds=len(fold_results),
            timestamp=datetime.now().isoformat(),
            data_path=data_path,
            total_samples=len(df),
            feature_count=len(available_features),
            fold_results=fold_results,
            mean_test_auc=float(np.mean(test_aucs)),
            std_test_auc=float(np.std(test_aucs)),
            mean_test_accuracy=float(np.mean(test_accs)),
            std_test_accuracy=float(np.std(test_accs)),
            mean_test_f1=float(np.mean(test_f1s)),
            std_test_f1=float(np.std(test_f1s)),
            mean_test_sensitivity=float(np.mean(test_sens)),
            std_test_sensitivity=float(np.std(test_sens)),
            mean_test_specificity=float(np.mean(test_specs)),
            std_test_specificity=float(np.std(test_specs)),
            threshold_stability=threshold_counts,
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("NESTED CV SUMMARY")
        print("=" * 70)
        print(f"\nTest Performance (mean ± std across {len(fold_results)} folds):")
        print(f"  AUC:         {summary.mean_test_auc:.4f} ± {summary.std_test_auc:.4f}")
        print(f"  Accuracy:    {summary.mean_test_accuracy:.4f} ± {summary.std_test_accuracy:.4f}")
        print(f"  F1-Score:    {summary.mean_test_f1:.4f} ± {summary.std_test_f1:.4f}")
        print(f"  Sensitivity: {summary.mean_test_sensitivity:.4f} ± {summary.std_test_sensitivity:.4f}")
        print(f"  Specificity: {summary.mean_test_specificity:.4f} ± {summary.std_test_specificity:.4f}")
        
        print(f"\nThreshold Stability:")
        for thresh, count in sorted(threshold_counts.items()):
            print(f"  {thresh:.2f}: selected by {count}/{len(fold_results)} folds")
        
        print(f"\nPer-Fold Results:")
        print(f"{'Fold':<6} {'Thresh':<8} {'AUC':<8} {'Acc':<8} {'Sens':<8} {'Spec':<8} {'F1':<8}")
        print("-" * 60)
        for r in fold_results:
            print(f"{r.fold:<6} {r.threshold:<8.2f} {r.test_auc:<8.4f} {r.test_accuracy:<8.4f} "
                  f"{r.test_sensitivity:<8.4f} {r.test_specificity:<8.4f} {r.test_f1:<8.4f}")
        
        # Save summary
        summary_dict = asdict(summary)
        # Convert FoldResult objects to dicts
        summary_dict['fold_results'] = [asdict(r) for r in fold_results]
        
        summary_path = os.path.join(output_dir, "cv_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        
        return summary
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nested CV Training for ML Semantic Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_ml_semantic_bridge_nested_cv.py                     # Run full 5-fold nested CV
    python train_ml_semantic_bridge_nested_cv.py --fold 1            # Train only fold 1
    python train_ml_semantic_bridge_nested_cv.py --data custom.csv   # Use custom data file
    python train_ml_semantic_bridge_nested_cv.py --output my_folds   # Custom output directory

Output Structure:
    semantic_cv_folds/
    ├── fold_1/
    │   ├── train.csv
    │   ├── test.csv
    │   ├── ml_semantic_bridge.pkl
    │   └── fold_metadata.json
    ├── fold_2/
    │   └── ...
    ├── ...
    └── cv_summary.json
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to merged_semantic_data.csv"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for fold data and models"
    )
    
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Train only a specific fold (1-5)"
    )
    
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of outer CV folds"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--label-column",
        type=str,
        default="diagnosis",
        help="Name of the diagnosis column"
    )
    
    parser.add_argument(
        "--id-column",
        type=str,
        default="PID",
        help="Name of the patient ID column"
    )
    
    args = parser.parse_args()
    
    # Check data exists
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"\nPlease ensure the semantic features have been extracted.")
        print(f"Expected path: {args.data}")
        sys.exit(1)
    
    # Run nested CV
    summary = run_nested_cv(
        data_path=args.data,
        output_dir=args.output,
        n_outer_folds=args.n_folds,
        label_column=args.label_column,
        id_column=args.id_column,
        random_state=args.seed,
        specific_fold=args.fold
    )
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nTo use a specific fold's model:")
    print(f"  from ml_semantic_bridge import MLSemanticBridge")
    print(f"  bridge = MLSemanticBridge.load('{args.output}/fold_1/ml_semantic_bridge.pkl')")
    print(f"  pre_diag, prompt = bridge.generate_llm_prompt(patient_features, pid)")


if __name__ == "__main__":
    main()
