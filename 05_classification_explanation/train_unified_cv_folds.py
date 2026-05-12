#!/usr/bin/env python3
"""
train_unified_cv_folds.py

Unified Cross-Validation Fold Generator for Multi-Modal Dementia Screening
===========================================================================

This script creates ALIGNED CV folds for both acoustic and semantic modalities.
It ensures that the same patients are in the same folds across both modalities,
which is essential for valid multi-modal evaluation.

The workflow is:
1. Load both acoustic and semantic feature datasets
2. Find common patients (by PID) that have BOTH feature types
3. Create 5-fold stratified CV splits based on these common PIDs
4. Save fold information that can be used by both training scripts

Usage:
    python train_unified_cv_folds.py                    # Generate unified folds
    python train_unified_cv_folds.py --check            # Check existing folds
    python train_unified_cv_folds.py --acoustic data1.csv --semantic data2.csv

Output:
    unified_cv_folds/
    ├── common_patients.csv          # List of patients with both modalities
    ├── fold_1/
    │   ├── train_pids.txt           # PIDs for training
    │   ├── test_pids.txt            # PIDs for testing
    │   └── fold_info.json           # Metadata
    ├── fold_2/
    │   └── ...
    └── cv_fold_summary.json         # Overall summary

After running this script:
1. Run train_ml_acoustic_bridge_nested_cv.py --use-unified-folds
2. Run train_ml_semantic_bridge_nested_cv.py --use-unified-folds
3. Run main_all.py for combined evaluation
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from sklearn.model_selection import StratifiedKFold

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths from .env
DEFAULT_ACOUSTIC_PATH = os.getenv(
    "ACOUSTIC_FEATURES_CSV",
    "datasets/output/acoustic_features/merged_data.csv"
)
DEFAULT_SEMANTIC_PATH = os.path.join(
    os.getenv("SEMANTIC_FEATURE_FOLDER", "datasets/output/semantic_features"),
    "merged_semantic_data.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.getenv("RESULTS_DIR", "datasets/results"),
    "unified_cv_folds"
)

# Impairment labels (consistent with other scripts)
IMPAIRMENT_LABELS = [
    'impairment', 'mci', 'dementia', 'ad', "alzheimer's",
    'probablead', 'possiblead', 'memory', 'vascular',
    'svppa', 'lvppa', 'ppa-nos', 'nfappa', "pick's"
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FoldInfo:
    """Information about a single CV fold."""
    fold: int
    train_pids: List[str]
    test_pids: List[str]
    train_control: int
    train_impairment: int
    test_control: int
    test_impairment: int


@dataclass
class UnifiedCVSummary:
    """Summary of unified CV fold generation."""
    n_folds: int
    timestamp: str
    acoustic_data_path: str
    semantic_data_path: str
    total_acoustic_patients: int
    total_semantic_patients: int
    common_patients: int
    control_count: int
    impairment_count: int
    random_state: int
    fold_info: List[FoldInfo]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def prepare_labels(df: pd.DataFrame, label_column: str = 'diagnosis') -> np.ndarray:
    """
    Convert diagnosis labels to binary (0=Control, 1=Impairment).
    
    Args:
        df: DataFrame with diagnosis column
        label_column: Name of diagnosis column
        
    Returns:
        Binary label array
    """
    labels = []
    
    for _, row in df.iterrows():
        diag = str(row.get(label_column, '')).lower().strip()
        
        if diag in IMPAIRMENT_LABELS or any(imp in diag for imp in IMPAIRMENT_LABELS):
            labels.append(1)
        else:
            labels.append(0)
    
    return np.array(labels)


def find_common_patients(
    acoustic_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    acoustic_id_col: str = 'participant_id',
    semantic_id_col: str = 'PID'
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Find patients that exist in both datasets.
    
    Args:
        acoustic_df: Acoustic features DataFrame
        semantic_df: Semantic features DataFrame
        acoustic_id_col: ID column name in acoustic data
        semantic_id_col: ID column name in semantic data
        
    Returns:
        Tuple of (set of common PIDs, merged info DataFrame)
    """
    # Get PIDs from both datasets
    acoustic_pids = set(acoustic_df[acoustic_id_col].astype(str).unique())
    semantic_pids = set(semantic_df[semantic_id_col].astype(str).unique())
    
    # Find intersection
    common_pids = acoustic_pids.intersection(semantic_pids)
    
    print(f"Acoustic patients: {len(acoustic_pids)}")
    print(f"Semantic patients: {len(semantic_pids)}")
    print(f"Common patients: {len(common_pids)}")
    
    # Create merged info DataFrame with diagnosis from semantic (or acoustic)
    merged_info = []
    for pid in common_pids:
        # Get diagnosis from semantic data (preferred) or acoustic
        sem_row = semantic_df[semantic_df[semantic_id_col] == pid]
        if not sem_row.empty and 'diagnosis' in sem_row.columns:
            diag = sem_row.iloc[0]['diagnosis']
        else:
            aco_row = acoustic_df[acoustic_df[acoustic_id_col] == pid]
            diag = aco_row.iloc[0].get('diagnosis', 'unknown') if not aco_row.empty else 'unknown'
        
        # Get age and gender if available
        age = None
        gender = None
        if not sem_row.empty:
            age = sem_row.iloc[0].get('age', None)
            gender = sem_row.iloc[0].get('gender', None)
        
        merged_info.append({
            'PID': pid,
            'diagnosis': diag,
            'age': age,
            'gender': gender
        })
    
    merged_df = pd.DataFrame(merged_info)
    
    return common_pids, merged_df


def generate_stratified_folds(
    patient_df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    label_column: str = 'diagnosis'
) -> List[FoldInfo]:
    """
    Generate stratified CV folds based on diagnosis labels.
    
    Args:
        patient_df: DataFrame with PID and diagnosis columns
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility
        label_column: Name of diagnosis column
        
    Returns:
        List of FoldInfo objects
    """
    # Prepare labels
    y = prepare_labels(patient_df, label_column)
    pids = patient_df['PID'].values
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_infos = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patient_df, y)):
        fold_num = fold_idx + 1
        
        train_pids = pids[train_idx].tolist()
        test_pids = pids[test_idx].tolist()
        
        train_labels = y[train_idx]
        test_labels = y[test_idx]
        
        fold_info = FoldInfo(
            fold=fold_num,
            train_pids=train_pids,
            test_pids=test_pids,
            train_control=int(sum(train_labels == 0)),
            train_impairment=int(sum(train_labels == 1)),
            test_control=int(sum(test_labels == 0)),
            test_impairment=int(sum(test_labels == 1)),
        )
        
        fold_infos.append(fold_info)
        
        print(f"Fold {fold_num}: Train={len(train_pids)} (C:{fold_info.train_control}, I:{fold_info.train_impairment}), "
              f"Test={len(test_pids)} (C:{fold_info.test_control}, I:{fold_info.test_impairment})")
    
    return fold_infos


def save_unified_folds(
    output_dir: str,
    fold_infos: List[FoldInfo],
    common_df: pd.DataFrame,
    acoustic_path: str,
    semantic_path: str,
    acoustic_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    random_state: int = 42
) -> UnifiedCVSummary:
    """
    Save unified CV folds to disk.
    
    Args:
        output_dir: Output directory
        fold_infos: List of FoldInfo objects
        common_df: DataFrame with common patient info
        acoustic_path: Path to acoustic data
        semantic_path: Path to semantic data
        acoustic_df: Full acoustic DataFrame
        semantic_df: Full semantic DataFrame
        random_state: Random seed used
        
    Returns:
        UnifiedCVSummary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save common patients list
    common_df.to_csv(os.path.join(output_dir, "common_patients.csv"), index=False)
    
    # Prepare labels for summary
    y = prepare_labels(common_df, 'diagnosis')
    
    # Save each fold
    for fold_info in fold_infos:
        fold_dir = os.path.join(output_dir, f"fold_{fold_info.fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save PID lists
        with open(os.path.join(fold_dir, "train_pids.txt"), 'w') as f:
            f.write('\n'.join(fold_info.train_pids))
        
        with open(os.path.join(fold_dir, "test_pids.txt"), 'w') as f:
            f.write('\n'.join(fold_info.test_pids))
        
        # Save fold metadata
        fold_meta = {
            'fold': fold_info.fold,
            'train_size': len(fold_info.train_pids),
            'test_size': len(fold_info.test_pids),
            'train_control': fold_info.train_control,
            'train_impairment': fold_info.train_impairment,
            'test_control': fold_info.test_control,
            'test_impairment': fold_info.test_impairment,
        }
        
        with open(os.path.join(fold_dir, "fold_info.json"), 'w') as f:
            json.dump(fold_meta, f, indent=2)
    
    # Create summary
    summary = UnifiedCVSummary(
        n_folds=len(fold_infos),
        timestamp=datetime.now().isoformat(),
        acoustic_data_path=acoustic_path,
        semantic_data_path=semantic_path,
        total_acoustic_patients=len(acoustic_df),
        total_semantic_patients=len(semantic_df),
        common_patients=len(common_df),
        control_count=int(sum(y == 0)),
        impairment_count=int(sum(y == 1)),
        random_state=random_state,
        fold_info=fold_infos,
    )
    
    # Save summary
    summary_dict = asdict(summary)
    summary_dict['fold_info'] = [asdict(fi) for fi in fold_infos]
    
    with open(os.path.join(output_dir, "cv_fold_summary.json"), 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    return summary


def load_unified_folds(folds_dir: str) -> Optional[Dict]:
    """
    Load unified CV folds from disk.
    
    Args:
        folds_dir: Directory containing unified folds
        
    Returns:
        Dictionary with fold information or None if not found
    """
    summary_path = os.path.join(folds_dir, "cv_fold_summary.json")
    
    if not os.path.exists(summary_path):
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load PID lists for each fold
    for fold_info in summary.get('fold_info', []):
        fold_num = fold_info['fold']
        fold_dir = os.path.join(folds_dir, f"fold_{fold_num}")
        
        train_pids_path = os.path.join(fold_dir, "train_pids.txt")
        test_pids_path = os.path.join(fold_dir, "test_pids.txt")
        
        if os.path.exists(train_pids_path):
            with open(train_pids_path, 'r') as f:
                fold_info['train_pids'] = [line.strip() for line in f if line.strip()]
        
        if os.path.exists(test_pids_path):
            with open(test_pids_path, 'r') as f:
                fold_info['test_pids'] = [line.strip() for line in f if line.strip()]
    
    return summary


def get_fold_pids(folds_dir: str, fold: int) -> Tuple[List[str], List[str]]:
    """
    Get train and test PIDs for a specific fold.
    
    Args:
        folds_dir: Directory containing unified folds
        fold: Fold number (1-5)
        
    Returns:
        Tuple of (train_pids, test_pids)
    """
    fold_dir = os.path.join(folds_dir, f"fold_{fold}")
    
    train_pids_path = os.path.join(fold_dir, "train_pids.txt")
    test_pids_path = os.path.join(fold_dir, "test_pids.txt")
    
    train_pids = []
    test_pids = []
    
    if os.path.exists(train_pids_path):
        with open(train_pids_path, 'r') as f:
            train_pids = [line.strip() for line in f if line.strip()]
    
    if os.path.exists(test_pids_path):
        with open(test_pids_path, 'r') as f:
            test_pids = [line.strip() for line in f if line.strip()]
    
    return train_pids, test_pids


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_unified_folds(
    acoustic_path: str,
    semantic_path: str,
    output_dir: str,
    n_folds: int = 5,
    random_state: int = 42
) -> UnifiedCVSummary:
    """
    Generate unified CV folds for both modalities.
    
    Args:
        acoustic_path: Path to acoustic features CSV
        semantic_path: Path to semantic features CSV
        output_dir: Output directory for fold data
        n_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        UnifiedCVSummary
    """
    print("\n" + "=" * 70)
    print("UNIFIED CV FOLD GENERATION")
    print("=" * 70)
    print(f"\nAcoustic data: {acoustic_path}")
    print(f"Semantic data: {semantic_path}")
    print(f"Output: {output_dir}")
    print(f"Folds: {n_folds}")
    print(f"Random state: {random_state}")
    
    # Load datasets
    print("\n" + "-" * 40)
    print("Loading datasets...")
    
    if not os.path.exists(acoustic_path):
        raise FileNotFoundError(f"Acoustic data not found: {acoustic_path}")
    if not os.path.exists(semantic_path):
        raise FileNotFoundError(f"Semantic data not found: {semantic_path}")
    
    acoustic_df = pd.read_csv(acoustic_path)
    semantic_df = pd.read_csv(semantic_path)
    
    print(f"Acoustic data shape: {acoustic_df.shape}")
    print(f"Semantic data shape: {semantic_df.shape}")
    
    # Find ID columns
    acoustic_id_col = 'PID' if 'PID' in acoustic_df.columns else 'participant_id'
    semantic_id_col = 'PID' if 'PID' in semantic_df.columns else 'participant_id'
    
    print(f"Acoustic ID column: {acoustic_id_col}")
    print(f"Semantic ID column: {semantic_id_col}")
    
    # Find common patients
    print("\n" + "-" * 40)
    print("Finding common patients...")
    
    common_pids, common_df = find_common_patients(
        acoustic_df, semantic_df,
        acoustic_id_col, semantic_id_col
    )
    
    if len(common_pids) == 0:
        raise ValueError("No common patients found between acoustic and semantic datasets!")
    
    # Check diagnosis distribution
    y = prepare_labels(common_df, 'diagnosis')
    print(f"\nDiagnosis distribution:")
    print(f"  Control: {sum(y == 0)}")
    print(f"  Impairment: {sum(y == 1)}")
    
    # Generate stratified folds
    print("\n" + "-" * 40)
    print("Generating stratified folds...")
    
    fold_infos = generate_stratified_folds(
        common_df,
        n_folds=n_folds,
        random_state=random_state
    )
    
    # Save folds
    print("\n" + "-" * 40)
    print("Saving fold data...")
    
    summary = save_unified_folds(
        output_dir=output_dir,
        fold_infos=fold_infos,
        common_df=common_df,
        acoustic_path=acoustic_path,
        semantic_path=semantic_path,
        acoustic_df=acoustic_df,
        semantic_df=semantic_df,
        random_state=random_state
    )
    
    print(f"\nSaved to: {output_dir}")
    
    return summary


def check_existing_folds(folds_dir: str) -> None:
    """Check and display existing unified folds."""
    print("\n" + "=" * 70)
    print("CHECKING EXISTING UNIFIED FOLDS")
    print("=" * 70)
    
    summary = load_unified_folds(folds_dir)
    
    if summary is None:
        print(f"\nNo unified folds found at: {folds_dir}")
        print("Run this script without --check to generate folds.")
        return
    
    print(f"\nFound unified folds created at: {summary.get('timestamp', 'unknown')}")
    print(f"Random state: {summary.get('random_state', 'unknown')}")
    print(f"\nDatasets:")
    print(f"  Acoustic: {summary.get('acoustic_data_path', 'unknown')}")
    print(f"  Semantic: {summary.get('semantic_data_path', 'unknown')}")
    print(f"\nPatient counts:")
    print(f"  Acoustic patients: {summary.get('total_acoustic_patients', 0)}")
    print(f"  Semantic patients: {summary.get('total_semantic_patients', 0)}")
    print(f"  Common patients: {summary.get('common_patients', 0)}")
    print(f"  Control: {summary.get('control_count', 0)}")
    print(f"  Impairment: {summary.get('impairment_count', 0)}")
    
    print(f"\nFold details:")
    print(f"{'Fold':<6} {'Train':<10} {'Test':<10} {'Train C/I':<15} {'Test C/I':<15}")
    print("-" * 60)
    
    for fold_info in summary.get('fold_info', []):
        train_size = len(fold_info.get('train_pids', []))
        test_size = len(fold_info.get('test_pids', []))
        train_ci = f"{fold_info.get('train_control', 0)}/{fold_info.get('train_impairment', 0)}"
        test_ci = f"{fold_info.get('test_control', 0)}/{fold_info.get('test_impairment', 0)}"
        
        print(f"{fold_info.get('fold', 0):<6} {train_size:<10} {test_size:<10} {train_ci:<15} {test_ci:<15}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Unified CV Folds for Multi-Modal Dementia Screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_unified_cv_folds.py                    # Generate with default paths
    python train_unified_cv_folds.py --check            # Check existing folds
    python train_unified_cv_folds.py --seed 123         # Use different random seed
    python train_unified_cv_folds.py --acoustic a.csv --semantic s.csv

After generating folds:
    1. Run: python train_ml_acoustic_bridge_nested_cv.py --use-unified-folds
    2. Run: python train_ml_semantic_bridge_nested_cv.py --use-unified-folds
    3. Run: python main_all.py
        """
    )
    
    parser.add_argument(
        "--acoustic", "-a",
        type=str,
        default=DEFAULT_ACOUSTIC_PATH,
        help="Path to acoustic features CSV"
    )
    
    parser.add_argument(
        "--semantic", "-s",
        type=str,
        default=DEFAULT_SEMANTIC_PATH,
        help="Path to semantic features CSV"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for unified folds"
    )
    
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check existing unified folds instead of generating new ones"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_existing_folds(args.output)
        return
    
    # Generate unified folds
    try:
        summary = generate_unified_folds(
            acoustic_path=args.acoustic,
            semantic_path=args.semantic,
            output_dir=args.output,
            n_folds=args.n_folds,
            random_state=args.seed
        )
        
        print("\n" + "=" * 70)
        print("UNIFIED FOLDS GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nTotal common patients: {summary.common_patients}")
        print(f"Control: {summary.control_count}, Impairment: {summary.impairment_count}")
        print(f"\nNext steps:")
        print(f"  1. Run: python train_ml_acoustic_bridge_unified.py")
        print(f"  2. Run: python train_ml_semantic_bridge_unified.py")
        print(f"  3. Run: python main_all.py")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
