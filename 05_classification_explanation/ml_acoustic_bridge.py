#!/usr/bin/env python3
"""
train_ml_bridge_nested_cv.py

Nested Cross-Validation Training for ML Acoustic Bridge
========================================================

This script implements proper nested CV methodology:
- Outer loop: 5-fold CV for final evaluation
- Inner loop: 4-fold CV within each training set for threshold optimization

This ensures NO data leakage - test data never influences threshold selection.

Usage:
    python train_ml_bridge_nested_cv.py                     # Run full nested CV
    python train_ml_bridge_nested_cv.py --fold 1            # Train only fold 1
    python train_ml_bridge_nested_cv.py --data custom.csv   # Use custom data

Output:
    - cv_folds/fold_1/train.csv, test.csv, ml_acoustic_bridge.pkl
    - cv_folds/fold_2/...
    - cv_folds/cv_summary.json (thresholds and metrics per fold)
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed")

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature map - consistent with main_ml_classifier.py
FEATURE_MAP = [
    'cpp_mean',
    'opensmile_jitterLocal_sma3nz_amean',
    'opensmile_shimmerLocaldB_sma3nz_amean',
    'pitch_std',
    'pitch_range',
    'pitch_iqr',
    'opensmile_MeanVoicedSegmentLengthSec',
    'opensmile_MeanUnvoicedSegmentLength',
    'pause_variability',
    'long_pause_ratio',
    'hesitation_rate',
    'speech_rate_variability',
    'voice_breaks_rate',
    'hnr_mean',
    'opensmile_HNRdBACF_sma3nz_amean',
    'age',
]

# Feature groups for interpretable domain breakdown
FEATURE_GROUPS = {
    'phonatory_instability': {
        'features': ['opensmile_jitterLocal_sma3nz_amean', 
                     'opensmile_shimmerLocaldB_sma3nz_amean'],
        'display_name': 'Phonatory Instability (Jitter/Shimmer)',
    },
    'cepstral': {
        'features': ['cpp_mean'],
        'display_name': 'Cepstral Peak Prominence',
    },
    'prosodic': {
        'features': ['pitch_std', 'pitch_range', 'pitch_iqr'],
        'display_name': 'Prosodic Variation (Pitch)',
    },
    'temporal_pause': {
        'features': ['pause_variability', 'long_pause_ratio', 'hesitation_rate'],
        'display_name': 'Temporal Pause Patterns',
    },
    'phonation_continuity': {
        'features': ['opensmile_MeanVoicedSegmentLengthSec', 
                     'opensmile_MeanUnvoicedSegmentLength'],
        'display_name': 'Phonation Continuity',
    },
    'speech_rate': {
        'features': ['speech_rate_variability'],
        'display_name': 'Speech Rate Control',
    },
    'voice_breaks': {
        'features': ['voice_breaks_rate'],
        'display_name': 'Voice Breaks',
    },
    'hnr': {
        'features': ['hnr_mean', 'opensmile_HNRdBACF_sma3nz_amean'],
        'display_name': 'Harmonics-to-Noise Ratio',
    },
    'demographics': {
        'features': ['age'],
        'display_name': 'Demographic Factors',
    },
}

# Candidate thresholds for inner CV optimization
CANDIDATE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

# Risk category thresholds (fixed, not tuned)
# RISK_THRESHOLDS = {
#     'high': 0.70,
#     'moderate_high': 0.55,
#     'moderate_low': 0.45,
#     'low': 0.30,
# }

# Impairment labels
IMPAIRMENT_LABELS = [
    'impairment', 'mci', 'dementia', 'ad', "alzheimer's",
    'probablead', 'possiblead', 'memory', 'vascular',
    'svppa', 'lvppa', 'ppa-nos', 'nfappa', "pick's"
]


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
    threshold_stability: Dict[float, int]  # threshold -> count of folds selecting it


# =============================================================================
# ML ACOUSTIC BRIDGE CLASS
# =============================================================================

class MLAcousticBridge:
    """
    ML Bridge with nested CV support.
    
    Uses CatBoost, XGBoost, and RandomForest (consistent with main_ml_classifier.py).
    """
    
    def __init__(self, classification_threshold: float = 0.50):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.group_models = {}
        self.feature_names = []
        self.classification_threshold = classification_threshold
        self.is_fitted = False
        self.training_metadata = {}
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            train_group_models: bool = True) -> 'MLAcousticBridge':
        """
        Train the ensemble models.
        
        Args:
            X: Feature dataframe
            y: Binary labels (0=Control, 1=Impairment)
            train_group_models: Whether to train group models for explanation
        """
        # Get available features
        available = [f for f in FEATURE_MAP if f in X.columns]
        self.feature_names = available
        
        X_train = X[available].copy()
        
        # Handle categorical features (gender)
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = self.label_encoder.fit_transform(X_train[col].astype(str))
        
        # Fill missing values
        X_train = X_train.fillna(X_train.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train ensemble models (CatBoost, XGBoost, RandomForest)
        self.models = {}
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_scaled, y)
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            self.models['xgboost'].fit(X_scaled, y)
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
            self.models['catboost'].fit(X_scaled, y)
        
        # Train group models for explanation
        if train_group_models:
            self._train_group_models(X_train, y)
        
        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(y),
            'n_features': len(self.feature_names),
            'n_positive': int(sum(y)),
            'n_negative': int(len(y) - sum(y)),
            'models_trained': list(self.models.keys()),
        }
        
        return self
    
    def _train_group_models(self, X: pd.DataFrame, y: np.ndarray):
        """Train logistic regression models for each feature group."""
        from sklearn.linear_model import LogisticRegression
        
        for group_name, group_info in FEATURE_GROUPS.items():
            features = group_info['features']
            available = [f for f in features if f in X.columns]
            
            if len(available) >= 1:
                X_group = X[available].fillna(0).values
                try:
                    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                    model.fit(X_group, y)
                    self.group_models[group_name] = {
                        'model': model,
                        'features': available,
                        'display_name': group_info['display_name'],
                    }
                except Exception as e:
                    print(f"Warning: Could not train group model {group_name}: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions."""
        X_pred = X[self.feature_names].copy()
        
        # Handle categorical
        for col in X_pred.select_dtypes(include=['object']).columns:
            X_pred[col] = self.label_encoder.transform(X_pred[col].astype(str))
        
        X_pred = X_pred.fillna(X_pred.median())
        X_scaled = self.scaler.transform(X_pred)
        
        # Average probabilities from all models
        probs = []
        for model in self.models.values():
            probs.append(model.predict_proba(X_scaled)[:, 1])
        
        return np.mean(probs, axis=0)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get binary predictions using current threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.classification_threshold).astype(int)
    
    def get_model_agreement(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate agreement between ensemble models.
        
        Args:
            X: Feature dataframe (single row or multiple rows)
            
        Returns:
            Dictionary with predictions, probabilities, agreement_ratio, and confidence
        """
        X_pred = X[self.feature_names].copy()
        
        # Handle categorical features
        for col in X_pred.select_dtypes(include=['object']).columns:
            X_pred[col] = self.label_encoder.transform(X_pred[col].astype(str))
        
        X_pred = X_pred.fillna(X_pred.median())
        X_scaled = self.scaler.transform(X_pred)
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[0, 1]
            pred = int(prob >= self.classification_threshold)
            predictions[name] = pred
            probabilities[name] = float(prob)
        
        # Calculate agreement ratio
        pred_values = list(predictions.values())
        agreement = sum(pred_values) / len(pred_values) if pred_values else 0.5
        
        # Confidence based on agreement level
        if agreement >= 0.9 or agreement <= 0.1:
            confidence = "HIGH"
        elif agreement >= 0.7 or agreement <= 0.3:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'agreement_ratio': agreement,
            'confidence': confidence,
        }
    
    def _get_dynamic_thresholds(self) -> Dict[str, float]:
        """
        Compute risk thresholds dynamically using proportional offsets.
        
        Offsets scale based on distance to boundaries (0 and 1) to avoid
        extreme values when threshold is far from 0.5.
        
        Returns:
            Dictionary with adjusted thresholds
        """
        thresh = self.classification_threshold
        
        # Calculate proportional offsets based on available space
        # Below threshold: use % of distance to 0
        # Above threshold: use % of distance to 1
        space_below = thresh           # Distance to lower bound (0)
        space_above = 1.0 - thresh     # Distance to upper bound (1)
        
        return {
            'high': thresh + (space_above * 0.40),         # 40% of space above
            'moderate_high': thresh + (space_above * 0.10),  # 10% of space above
            'moderate_low': thresh - (space_below * 0.10),   # 10% of space below
            'low': thresh - (space_below * 0.40),            # 40% of space below
        }
    
    def _score_to_category(self, score: float) -> str:
        """Convert probability to risk category."""
        thresholds = self._get_dynamic_thresholds()
        if score >= thresholds['high']:
            return "HIGH RISK"
        elif score >= thresholds['moderate_high']:
            return "MODERATE-HIGH RISK"
        elif score >= thresholds['moderate_low']:
            return "MODERATE RISK"
        elif score >= thresholds['low']:
            return "LOW-MODERATE RISK"
        else:
            return "LOW RISK"
    
    def _score_to_level(self, score: float) -> str:
        """
        Convert group score to level description using proportional thresholds.
        
        Uses 20% of available space above/below the classification threshold.
        """
        thresh = self.classification_threshold
        space_below = thresh
        space_above = 1.0 - thresh
        
        elevated_threshold = thresh + (space_above * 0.20)  # 20% of space above
        reduced_threshold = thresh - (space_below * 0.20)   # 20% of space below
        
        if score >= elevated_threshold:
            return "possibly elevated"
        elif score <= reduced_threshold:
            return "possibly reduced"
        else:
            return "inconclusive"
    
    def calculate_certainty(_self, prob, threshold=0.5):
        if prob >= threshold:
            # Distance from threshold to 1.0
            certainty = (prob - threshold) / (1 - threshold)
        else:
            # Distance from threshold to 0.0
            certainty = (threshold - prob) / threshold
        return certainty


    def analyze_patient(self, patient_features: Dict[str, float], 
                       pid: str = "unknown") -> Dict:
        """Analyze a single patient and return structured output."""
        df = pd.DataFrame([patient_features])
        overall_score = float(self.predict_proba(df)[0])
        
        ml_prediction = "Impairment" if overall_score >= self.classification_threshold else "Control"
        overall_category = self._score_to_category(overall_score)
        
        # Get model agreement
        agreement = self.get_model_agreement(df)
        
        # Group scores
        group_scores = {}
        for group_name, group_info in self.group_models.items():
            features = group_info['features']
            available = [f for f in features if f in df.columns]
            if available:
                X_group = df[available].fillna(0).values
                try:
                    group_prob = group_info['model'].predict_proba(X_group)[0, 1]
                except:
                    group_prob = 0.0
                
                confidence_score = self.calculate_certainty(group_prob, self.classification_threshold)  # Convert to confidence score (0-1)
                # Convert to categorical
                if confidence_score >= 0.7:
                    confidence_level = "HIGH"
                elif confidence_score >= 0.5:
                    confidence_level = "MODERATE"
                else:
                    confidence_level = "LOW"
        
                group_scores[group_name] = {
                    'display_name': group_info['display_name'],
                    'score': float(group_prob),
                    'level': self._score_to_level(group_prob),
                    'confidence': confidence_level,  # Individual groups have low AUC
                }
        
        return {
            'pid': pid,
            'overall_score': overall_score,
            'overall_category': overall_category,
            'ml_prediction': ml_prediction,
            'threshold': self.classification_threshold,
            'model_agreement': agreement,
            'group_scores': group_scores,
        }
    
    def generate_llm_prompt(self, patient_features: Dict[str, float],
                           pid: str = "unknown") -> Tuple[str, str]:
        """
        Generate LLM-ready prompt with rich domain breakdown.
        
        Returns:
            Tuple of (pre_diagnosis_text, acoustic_prompt)
        """
        output = self.analyze_patient(patient_features, pid)
        
        lines = []
        
        # Header

        lines.append(f"**ML Ensemble Prediction:  {output['ml_prediction']}**")
        lines.append(f"**Integrated risk score: {output['overall_category']} **")
        lines.append(f"**Probability: {output['overall_score']:.1%}**")
        lines.append(f"**Classification Threshold: {output['threshold']:.0%}**")
        lines.append(f"**Model Agreement: {output['model_agreement']['confidence']}**")
        lines.append("")
        lines.append("- The ensemble model (CatBoost + XGBoost + RandomForest) analyzes acoustic biomarkers.")
        lines.append("-" * 60)
        lines.append("DOMAIN BREAKDOWN (for interpretation support and evidence-based decision-marking)")
        lines.append("↑ = patterns associated with impairment risk; ↓ = patterns associated with healthy controls")
        lines.append("-" * 60)
        
        # Sort by score
        sorted_groups = sorted(
            output['group_scores'].items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        for group_name, gs in sorted_groups:
            thresh = self.classification_threshold
            space_below = thresh
            space_above = 1.0 - thresh
            
            elevated_threshold = thresh + (space_above * 0.10)  # 10% of space above
            reduced_threshold = thresh - (space_below * 0.10)   # 10% of space below
            
            if gs['score'] > elevated_threshold:
                indicator = "↑"  # Elevated risk
            elif gs['score'] < reduced_threshold:
                indicator = "↓"  # Reduced risk
            else:
                indicator = "→"  # Typical
            
            if gs['display_name'] != "Demographic Factors":
                lines.append(f"  {indicator} {gs['display_name']}: risk level {gs['level']} ({gs['confidence']} confidence)")
        
        lines.append("")
        lines.append("Note: The integrated score captures complex acoustic feature interactions.")
        lines.append("")

        acoustic_prompt = "\n".join(lines)
        
        # Pre-diagnosis
        pre_diagnosis = f"""
## Clinical Decision Support

**ML Ensemble Prediction: {output['ml_prediction']}**
**Probability: {output['overall_score']:.1%}**
**Classification Threshold: {output['threshold']:.0%}**

The ensemble model (CatBoost + XGBoost + RandomForest) analyzes {len(self.feature_names)} acoustic biomarkers.
Use the integrated score as your primary basis; domain breakdown supports explanation.
"""
        
        return pre_diagnosis, acoustic_prompt
    
    def save(self, path: str):
        """Save the bridge to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'MLAcousticBridge':
        """Load the bridge from disk, handling __main__ module issues."""
        import pickle
        
        class _BackwardsCompatibleUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # If pickled as __main__, redirect to the actual class
                if module == '__main__' and name == 'MLAcousticBridge':
                    return cls
                return super().find_class(module, name)
        
        with open(path, 'rb') as f:
            return _BackwardsCompatibleUnpickler(f).load()


# =============================================================================
# NESTED CV FUNCTIONS
# =============================================================================

def prepare_labels(df: pd.DataFrame, label_column: str = 'diagnosis') -> np.ndarray:
    """Convert diagnosis labels to binary."""
    return df[label_column].apply(
        lambda x: 1 if str(x).lower() in IMPAIRMENT_LABELS else 0
    ).values


def inner_cv_threshold_search(
    X: pd.DataFrame, 
    y: np.ndarray,
    candidate_thresholds: List[float] = CANDIDATE_THRESHOLDS,
    n_inner_folds: int = 4,
    random_state: int = 42
) -> Tuple[float, Dict[float, float]]:
    """
    Find optimal threshold using inner cross-validation.
    
    Args:
        X: Training features
        y: Training labels
        candidate_thresholds: List of thresholds to try
        n_inner_folds: Number of inner CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (best_threshold, {threshold: mean_f1_score})
    """
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
    
    threshold_scores = {t: [] for t in candidate_thresholds}
    
    for train_idx, val_idx in inner_cv.split(X, y):
        X_train_inner = X.iloc[train_idx]
        y_train_inner = y[train_idx]
        X_val_inner = X.iloc[val_idx]
        y_val_inner = y[val_idx]
        
        # Train a quick model
        bridge = MLAcousticBridge(classification_threshold=0.5)
        bridge.fit(X_train_inner, y_train_inner, train_group_models=False)
        
        # Get probabilities on validation set
        probs = bridge.predict_proba(X_val_inner)
        
        # Evaluate each threshold
        for thresh in candidate_thresholds:
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(y_val_inner, preds, zero_division=0)
            threshold_scores[thresh].append(f1)
    
    # Average scores
    mean_scores = {t: np.mean(scores) for t, scores in threshold_scores.items()}
    
    # Find best threshold
    best_threshold = max(mean_scores, key=mean_scores.get)
    
    return best_threshold, mean_scores


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
    Train model for a single fold with inner CV threshold optimization.
    
    Args:
        df_train: Training data
        df_test: Test data
        fold_num: Fold number (1-5)
        output_dir: Directory to save outputs
        label_column: Name of diagnosis column
        id_column: Name of patient ID column
        run_inner_cv: Whether to run inner CV for threshold optimization
        
    Returns:
        FoldResult with metrics
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num}")
    print(f"{'='*60}")
    
    # Prepare data
    y_train = prepare_labels(df_train, label_column)
    y_test = prepare_labels(df_test, label_column)
    
    available_features = [f for f in FEATURE_MAP if f in df_train.columns]
    X_train = df_train[available_features]
    X_test = df_test[available_features]
    
    print(f"Training samples: {len(df_train)} (Impairment: {sum(y_train)}, Control: {len(y_train)-sum(y_train)})")
    print(f"Test samples: {len(df_test)} (Impairment: {sum(y_test)}, Control: {len(y_test)-sum(y_test)})")
    print(f"Features: {len(available_features)}")
    
    # Inner CV for threshold optimization
    if run_inner_cv:
        print(f"\nRunning inner 4-fold CV for threshold optimization...")
        best_threshold, inner_scores = inner_cv_threshold_search(X_train, y_train)
        print(f"Inner CV results:")
        for thresh, score in sorted(inner_scores.items()):
            marker = " <-- SELECTED" if thresh == best_threshold else ""
            print(f"  Threshold {thresh:.2f}: F1 = {score:.4f}{marker}")
    else:
        best_threshold = 0.50
        inner_scores = {0.50: 0.0}
        print(f"Using default threshold: {best_threshold}")
    
    # Train final model on full training set
    print(f"\nTraining final model with threshold = {best_threshold}...")
    bridge = MLAcousticBridge(classification_threshold=best_threshold)
    bridge.fit(X_train, y_train, train_group_models=True)
    
    # Evaluate on training set (sanity check)
    train_probs = bridge.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, train_probs)
    
    # Evaluate on test set
    test_probs = bridge.predict_proba(X_test)
    test_preds = (test_probs >= best_threshold).astype(int)
    
    test_auc = roc_auc_score(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_sens = recall_score(y_test, test_preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nTest Results:")
    print(f"  AUC:         {test_auc:.4f}")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  Sensitivity: {test_sens:.4f}")
    print(f"  Specificity: {test_spec:.4f}")
    print(f"  F1-Score:    {test_f1:.4f}")
    print(f"  Confusion:   TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Save fold outputs
    fold_dir = os.path.join(output_dir, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Save train/test splits
    df_train.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
    
    # Save model
    bridge.save(os.path.join(fold_dir, "ml_acoustic_bridge.pkl"))
    
    # Save fold metadata
    fold_meta = {
        'fold': fold_num,
        'threshold': best_threshold,
        'inner_cv_scores': inner_scores,
        'train_size': len(df_train),
        'test_size': len(df_test),
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
    }
    with open(os.path.join(fold_dir, "fold_metadata.json"), 'w') as f:
        json.dump(fold_meta, f, indent=2)
    
    print(f"\nSaved to: {fold_dir}/")
    
    return FoldResult(
        fold=fold_num,
        threshold=best_threshold,
        train_size=len(df_train),
        test_size=len(df_test),
        train_auc=train_auc,
        test_auc=test_auc,
        test_accuracy=test_acc,
        test_sensitivity=test_sens,
        test_specificity=test_spec,
        test_f1=test_f1,
        test_tp=int(tp),
        test_tn=int(tn),
        test_fp=int(fp),
        test_fn=int(fn),
        inner_cv_thresholds=inner_scores,
    )


def run_nested_cv(
    data_path: str,
    output_dir: str = "cv_folds",
    n_outer_folds: int = 5,
    label_column: str = 'diagnosis',
    id_column: str = 'PID',
    random_state: int = 42,
    specific_fold: Optional[int] = None
) -> CVSummary:
    """
    Run full nested cross-validation.
    
    Args:
        data_path: Path to merged_data.csv
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
    print("NESTED CROSS-VALIDATION FOR ML ACOUSTIC BRIDGE")
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Outer CV
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    
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
            feature_count=len([f for f in FEATURE_MAP if f in df.columns]),
            fold_results=fold_results,
            mean_test_auc=float(np.mean(test_aucs)),
            std_test_auc=float(np.std(test_aucs)),
            mean_test_accuracy=float(np.mean(test_accs)),
            std_test_accuracy=float(np.std(test_accs)),
            mean_test_f1=float(np.mean(test_f1s)),
            std_test_f1=float(np.std(test_f1s)),
            threshold_stability=threshold_counts,
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("NESTED CV SUMMARY")
        print("=" * 70)
        print(f"\nTest Performance (mean ± std across {len(fold_results)} folds):")
        print(f"  AUC:      {summary.mean_test_auc:.4f} ± {summary.std_test_auc:.4f}")
        print(f"  Accuracy: {summary.mean_test_accuracy:.4f} ± {summary.std_test_accuracy:.4f}")
        print(f"  F1-Score: {summary.mean_test_f1:.4f} ± {summary.std_test_f1:.4f}")
        
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
        
        with open(os.path.join(output_dir, "cv_summary.json"), 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        print(f"\nSummary saved to: {output_dir}/cv_summary.json")
        
        return summary
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nested CV Training for ML Acoustic Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_ml_bridge_nested_cv.py                     # Run full 5-fold nested CV
    python train_ml_bridge_nested_cv.py --fold 1            # Train only fold 1
    python train_ml_bridge_nested_cv.py --data custom.csv   # Use custom data file
    python train_ml_bridge_nested_cv.py --output my_folds   # Custom output directory

Output Structure:
    cv_folds/
    ├── fold_1/
    │   ├── train.csv
    │   ├── test.csv
    │   ├── ml_acoustic_bridge.pkl
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
        default=os.getenv("NORMALISATION_DATASET_PATH", 
                         "datasets/output/acoustic_features/merged_data.csv"),
        help="Path to merged_data.csv"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="cv_folds",
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
    print(f"  from train_ml_bridge_nested_cv import MLAcousticBridge")
    print(f"  bridge = MLAcousticBridge.load('{args.output}/fold_1/ml_acoustic_bridge.pkl')")
    print(f"  pre_diag, prompt = bridge.generate_llm_prompt(patient_features, pid)")


if __name__ == "__main__":
    main()