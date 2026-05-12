#!/usr/bin/env python3
"""
ml_semantic_bridge.py

ML Semantic Bridge for Dementia Screening
==========================================

This module provides ML-based classification using semantic/linguistic features
extracted from patient transcripts. It mirrors the acoustic bridge architecture
but operates on semantic features.

Key Features:
- Ensemble models (XGBoost, CatBoost, RandomForest)
- Feature group analysis for interpretable explanations
- Nested CV support for proper evaluation
- LLM prompt generation with risk scores and domain breakdown

Usage:
    from ml_semantic_bridge import MLSemanticBridge
    
    bridge = MLSemanticBridge.load('semantic_cv_folds/fold_1/ml_semantic_bridge.pkl')
    pre_diag, prompt = bridge.generate_llm_prompt(patient_features, pid)
"""

import os
import json
import pickle
from matplotlib import lines
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Run: pip install catboost")

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature map - 20 semantic features (removed grammar_correctness_score)
SEMANTIC_FEATURE_MAP = [
    # Temporal Features (4) - continuous values
    "temporal_avg_response_latency",
    "temporal_avg_speaking_rate",
    "temporal_max_inter_word_gap",
    "temporal_avg_between_utterance_gap",
    
    # Lexical Richness and Diversity (4) - scores 0-10
    "vocabulary_range_score",
    "lexical_accuracy_score",
    "specificity_score",
    "advanced_vocabulary_score",
    
    # Syntactic Structure (2) - scores 0-10 (removed grammar_correctness_score)
    "grammar_complexity_score",
    "structure_variety_score",
    
    # Pragmatic Competence (3) - scores 0-10
    "referential_clarity_score",
    "state_of_mind_language_score",
    "implausible_details_score",
    
    # Semantic Coherence and Cohesion (6) - scores 0-10
    "topic_management_score",
    "logical_organization_score",
    "cohesion_score",
    "cause_and_effect_score",
    "repetition_score",
    "information_prioritization_score",
    
    # Demographics
    "age",
]

# Feature groups for interpretable domain breakdown
# Each group has: features, display_name, summary_fields, weights, and direction info
SEMANTIC_FEATURE_GROUPS = {
    "temporal_features": {
        "features": [
            "temporal_avg_response_latency",
            "temporal_avg_speaking_rate",
            "temporal_max_inter_word_gap",
            "temporal_avg_between_utterance_gap",
        ],
        "display_name": "Temporal Speech Patterns",
        "summary_fields": [],  # No LLM summaries for temporal features
        # Direction: -1 means higher value = more likely impairment
        # Weight: relative importance in the group
        "feature_info": {
            "temporal_avg_response_latency": {"direction": -1, "weight": 1.2},
            "temporal_avg_speaking_rate": {"direction": 1, "weight": 1.0},  # higher = healthier
            "temporal_max_inter_word_gap": {"direction": -1, "weight": 1.5},
            "temporal_avg_between_utterance_gap": {"direction": -1, "weight": 1.0},
        },
        "group_weight": 1.0,  # Overall weight of this group
    },
    "lexical_richness": {
        "features": [
            "vocabulary_range_score",
            "lexical_accuracy_score",
            "specificity_score",
            "advanced_vocabulary_score",
        ],
        "display_name": "Lexical Richness and Diversity",
        "summary_fields": [
            "vocabulary_range_summary",
            "lexical_accuracy_summary",
            "specificity_summary",
            "advanced_vocabulary_summary",
        ],
        "feature_info": {
            "vocabulary_range_score": {"direction": 1, "weight": 1.2},  # higher = healthier
            "lexical_accuracy_score": {"direction": 1, "weight": 1.5},
            "specificity_score": {"direction": 1, "weight": 1.0},
            "advanced_vocabulary_score": {"direction": 1, "weight": 0.8},
        },
        "group_weight": 1.5,  # Lexical features are important
    },
    "syntactic_structure": {
        "features": [
            "grammar_complexity_score",
            "structure_variety_score",
        ],
        "display_name": "Syntactic Structure and Correctness",
        "summary_fields": [
            "grammar_complexity_summary",
            "structure_variety_summary",
        ],
        "feature_info": {
            "grammar_complexity_score": {"direction": 1, "weight": 1.2},
            "structure_variety_score": {"direction": 1, "weight": 1.0},
        },
        "group_weight": 1.3,
    },
    "pragmatic_competence": {
        "features": [
            "referential_clarity_score",
            "state_of_mind_language_score",
            "implausible_details_score",
        ],
        "display_name": "Pragmatic Competence",
        "summary_fields": [
            "referential_clarity_summary",
            "state_of_mind_language_summary",
            "implausible_details_summary",
        ],
        "feature_info": {
            "referential_clarity_score": {"direction": 1, "weight": 1.3},
            "state_of_mind_language_score": {"direction": 1, "weight": 0.8},
            # Higher implausible_details_score = fewer implausible details = healthier
            "implausible_details_score": {"direction": 1, "weight": 1.2},
        },
        "group_weight": 1.0,
    },
    "semantic_coherence": {
        "features": [
            "topic_management_score",
            "logical_organization_score",
            "cohesion_score",
            "cause_and_effect_score",
            "repetition_score",
            "information_prioritization_score",
        ],
        "display_name": "Semantic Coherence and Cohesion",
        "summary_fields": [
            "topic_management_summary",
            "logical_organization_summary",
            "cohesion_summary",
            "cause_and_effect_summary",
            "repetition_summary",
            "information_prioritization_summary",
        ],
        "feature_info": {
            "topic_management_score": {"direction": 1, "weight": 1.2},
            "logical_organization_score": {"direction": 1, "weight": 1.3},
            "cohesion_score": {"direction": 1, "weight": 1.0},
            "cause_and_effect_score": {"direction": 1, "weight": 1.1},
            # Higher repetition_score = less problematic repetition = healthier
            "repetition_score": {"direction": 1, "weight": 1.4},
            "information_prioritization_score": {"direction": 1, "weight": 0.9},
        },
        "group_weight": 1.5,  # Semantic coherence is very important
    },
    "demographics": {
        "features": ["age"],
        "display_name": "Demographic Factors",
        "summary_fields": [],
        "feature_info": {
            "age": {"direction": -1, "weight": 2.0},  # Higher age = higher risk
        },
        "group_weight": 1.2,  # Age is quite important
    },
}

# Risk category thresholds
RISK_THRESHOLDS = {
    "high": 0.70,
    "moderate_high": 0.55,
    "moderate_low": 0.45,
    "low": 0.30,
}

# Candidate thresholds for inner CV optimization
CANDIDATE_THRESHOLDS = [round(x * 0.01, 2) for x in range(30, 66)]

# Impairment labels for diagnosis normalization
IMPAIRMENT_LABELS = [
    'impairment', 'mci', 'dementia', 'ad', "alzheimer's",
    'probablead', 'possiblead', 'memory', 'vascular',
    'svppa', 'lvppa', 'ppa-nos', 'nfappa', "pick's"
]


# =============================================================================
# ML SEMANTIC BRIDGE CLASS
# =============================================================================

class MLSemanticBridge:
    """
    ML Bridge for semantic feature classification.
    
    Uses ensemble of CatBoost, XGBoost, and RandomForest for:
    1. Overall impairment probability prediction
    2. Per-group probability for interpretable breakdown
    3. LLM prompt generation with risk levels
    """
    
    def __init__(self, classification_threshold: float = 0.50):
        """
        Initialize the ML Semantic Bridge.
        
        Args:
            classification_threshold: Threshold for binary classification
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.group_models = {}
        self.feature_names = []
        self.classification_threshold = classification_threshold
        self.is_fitted = False
        self.training_metadata = {}
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        train_group_models: bool = True
    ) -> 'MLSemanticBridge':
        """
        Train the ensemble models.
        
        Args:
            X: Feature dataframe with semantic features
            y: Binary labels (0=Control, 1=Impairment)
            train_group_models: Whether to train per-group models for explanations
            
        Returns:
            Self for chaining
        """
        # Get available features
        available = [f for f in SEMANTIC_FEATURE_MAP if f in X.columns]
        self.feature_names = available
        
        X_train = X[available].copy()
        
        # Handle any categorical features
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = self.label_encoder.fit_transform(X_train[col].astype(str))
        
        # Fill missing values with median
        X_train = X_train.fillna(X_train.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train ensemble models
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
                random_seed=42,
                verbose=False
            )
            self.models['catboost'].fit(X_scaled, y)
        
        # Train group models for interpretable breakdown
        if train_group_models:
            self._train_group_models(X, y)
        
        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(y),
            'n_features': len(available),
            'n_positive': int(sum(y)),
            'n_negative': int(len(y) - sum(y)),
            'models_trained': list(self.models.keys()),
            'timestamp': datetime.now().isoformat(),
        }
        
        return self
    
    def _train_group_models(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Train per-group models for interpretable explanations.
        
        Args:
            X: Feature dataframe
            y: Binary labels
        """
        self.group_models = {}
        
        for group_name, group_info in SEMANTIC_FEATURE_GROUPS.items():
            group_features = [f for f in group_info['features'] if f in X.columns]
            
            if len(group_features) < 1:
                continue
            
            X_group = X[group_features].copy()
            
            # Handle categorical
            for col in X_group.select_dtypes(include=['object']).columns:
                X_group[col] = self.label_encoder.fit_transform(X_group[col].astype(str))
            
            X_group = X_group.fillna(X_group.median())
            
            # Train a simple RandomForest for each group
            group_scaler = StandardScaler()
            X_group_scaled = group_scaler.fit_transform(X_group)
            
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_group_scaled, y)
            
            self.group_models[group_name] = {
                'model': model,
                'scaler': group_scaler,
                'features': group_features,
            }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability predictions.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of impairment probabilities (average of ensemble)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_pred = X[self.feature_names].copy()
        
        # Handle categorical
        for col in X_pred.select_dtypes(include=['object']).columns:
            X_pred[col] = self.label_encoder.transform(X_pred[col].astype(str))
        
        X_pred = X_pred.fillna(X_pred.median())
        X_scaled = self.scaler.transform(X_pred)
        
        # Collect predictions from all models
        probas = []
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[:, 1]
            probas.append(prob)
        
        # Return average probability
        return np.mean(probas, axis=0)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get binary predictions using threshold.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Binary predictions (0=Control, 1=Impairment)
        """
        probas = self.predict_proba(X)
        return (probas >= self.classification_threshold).astype(int)
    
    def get_model_agreement(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get individual model predictions to assess confidence.
        
        Args:
            X: Feature dataframe (single row)
            
        Returns:
            Dictionary with per-model predictions and agreement info
        """
        X_pred = X[self.feature_names].copy()
        
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
        
        # Calculate agreement
        pred_values = list(predictions.values())
        agreement = sum(pred_values) / len(pred_values) if pred_values else 0.5
        
        # Confidence based on agreement
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
    
    def get_group_scores(
        self, 
        X: pd.DataFrame,
        patient_features: Dict[str, Any] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get per-group probability scores for interpretable breakdown.
        
        Args:
            X: Feature dataframe (single row)
            patient_features: Optional dict with raw features (for LLM confidence)
            
        Returns:
            Dictionary with group scores and confidence
        """
        group_scores = {}
        
        for group_name, group_data in self.group_models.items():
            model = group_data['model']
            scaler = group_data['scaler']
            features = group_data['features']
            
            # Get available features
            available = [f for f in features if f in X.columns]
            if not available:
                continue
            
            X_group = X[available].copy()
            
            for col in X_group.select_dtypes(include=['object']).columns:
                X_group[col] = self.label_encoder.transform(X_group[col].astype(str))
            
            X_group = X_group.fillna(X_group.median())
            X_group_scaled = scaler.transform(X_group)
            
            # Get probability
            prob = float(model.predict_proba(X_group_scaled)[0, 1])
            
            # Determine risk level using dynamic thresholds
            thresholds = self._get_dynamic_thresholds()
            if prob >= thresholds['high']:
                level = "ELEVATED"
            elif prob >= thresholds['moderate_high']:
                level = "MODERATELY ELEVATED"
            elif prob <= thresholds['low']:
                level = "REDUCED"
            elif prob <= thresholds['moderate_low']:
                level = "SLIGHTLY REDUCED"
            else:
                level = "TYPICAL"
            
            # Calculate confidence based on model agreement and LLM confidence
            confidence = self._calculate_group_confidence(
                group_name, prob, patient_features
            )
            
            group_scores[group_name] = {
                'score': prob,
                'level': level,
                'confidence': confidence,
                'display_name': SEMANTIC_FEATURE_GROUPS[group_name]['display_name'],
                'features_used': available,
            }
        
        return group_scores
    
    def calculate_certainty(self, prob, threshold=0.5):
        if prob >= threshold:
            # Distance from threshold to 1.0
            certainty = (prob - threshold) / (1 - threshold)
        else:
            # Distance from threshold to 0.0
            certainty = (threshold - prob) / threshold
        return certainty


    def _calculate_group_confidence(
        self,
        group_name: str,
        prob: float,
        patient_features: Dict[str, Any] = None
    ) -> str:
        """
        Calculate confidence level for a group based on multiple factors.
        
        Args:
            group_name: Name of the feature group
            prob: ML probability score
            patient_features: Optional patient features dict
            
        Returns:
            Confidence level string (HIGH/MODERATE/LOW)
        """
        confidence_score = 0  # Base confidence
        
        # Factor 1: Probability certainty (extreme values = higher confidence)
        prob_certainty = self.calculate_certainty(prob, self.classification_threshold)  # 0 to 1
        confidence_score += prob_certainty * 0.5
        
        # Factor 2: LLM confidence scores if available
        if patient_features:
            group_info = SEMANTIC_FEATURE_GROUPS.get(group_name, {})
            features = group_info.get('features', [])
            
            llm_confidences = []
            for feat in features:
                conf_key = feat.replace('_score', '_confidence')
                if conf_key in patient_features:
                    try:
                        conf = float(patient_features[conf_key])
                        # Normalize to 0-1 (assuming 0-10 scale)
                        llm_confidences.append(conf / 10.0)
                    except (ValueError, TypeError):
                        pass
            
            if llm_confidences:
                avg_llm_conf = np.mean(llm_confidences)
                confidence_score += avg_llm_conf * 0.5 #equally weight LLM confidence and ML confidence
        
        # Convert to categorical
        if confidence_score >= 0.7:
            return "HIGH"
        elif confidence_score >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
    
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
    
    def analyze_patient(
        self,
        patient_features: Dict[str, Any],
        pid: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Complete analysis of a patient's semantic features.
        
        Args:
            patient_features: Dictionary of feature values
            pid: Patient ID
            
        Returns:
            Complete analysis dictionary
        """
        # Create dataframe from features
        X = pd.DataFrame([patient_features])
        
        # Get overall prediction
        prob = float(self.predict_proba(X)[0])
        pred = int(prob >= self.classification_threshold)
        
        # Determine overall risk category using dynamic thresholds
        thresholds = self._get_dynamic_thresholds()
        if prob >= thresholds['high']:
            category = "HIGH RISK"
        elif prob >= thresholds['moderate_high']:
            category = "MODERATE-HIGH RISK"
        elif prob <= thresholds['low']:
            category = "LOW RISK"
        elif prob <= thresholds['moderate_low']:
            category = "LOW-MODERATE RISK"
        else:
            category = "MODERATE RISK"
        
        # Get model agreement
        agreement = self.get_model_agreement(X)
        
        # Get group scores
        group_scores = self.get_group_scores(X, patient_features)
        
        return {
            'pid': pid,
            'overall_score': prob,
            'overall_category': category,
            'ml_prediction': "Impairment" if pred == 1 else "Control",
            'threshold': self.classification_threshold,
            'model_agreement': agreement,
            'group_scores': group_scores,
        }
    
    def generate_llm_prompt(
        self,
        patient_features: Dict[str, Any],
        pid: str = "unknown"
    ) -> Tuple[str, str]:
        """
        Generate LLM-ready prompt with ML analysis results.
        
        Args:
            patient_features: Dictionary of feature values
            pid: Patient ID
            
        Returns:
            Tuple of (pre_diagnosis_text, acoustic_section_text)
        """
        analysis = self.analyze_patient(patient_features, pid)
        
        # Build pre-diagnosis text
        pre_diagnosis = self._format_pre_diagnosis(analysis)
        
        # Build main analysis section
        analysis_section = self._format_analysis_section(analysis, patient_features)
        
        return pre_diagnosis, analysis_section
    
    def _format_pre_diagnosis(self, analysis: Dict[str, Any]) -> str:
        """Format the pre-diagnosis summary."""
        return None
    
    def _format_analysis_section(
        self, 
        analysis: Dict[str, Any],
        patient_features: Dict[str, Any]
    ) -> str:
        """Format the main analysis section with domain breakdown and summaries."""
        lines = []
        
        # Header with integrated score

        
        lines.append(f"**ML Ensemble Prediction: {analysis['ml_prediction']}**")
        lines.append(f"**Integrated risk score: {analysis['overall_category']} <<<")
        lines.append(f"**Probability: {analysis['overall_score']:.1%}**")
        lines.append(f"**Classification Threshold: {analysis['threshold']:.0%}**")
        lines.append(f"**Model Agreement: {analysis['model_agreement']['confidence']}**")
        lines.append("")
        lines.append("- The ensemble model (CatBoost + XGBoost + RandomForest) analyzes semantic/linguistic biomarkers.")
        lines.append("-" * 60)
        lines.append("DOMAIN BREAKDOWN (for interpretation support and evidence-based decision-marking):")
        lines.append("↑ = patterns associated with impairment risk; ↓ = patterns associated with healthy controls")
        lines.append("-" * 60)
        
        # Sort groups by score (highest risk first)
        group_scores = analysis.get('group_scores', {})
        sorted_groups = sorted(
            group_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Display each group's risk level
        for group_name, gs in sorted_groups:
            thresh = self.classification_threshold
            space_below = thresh
            space_above = 1.0 - thresh
            
            elevated_threshold = thresh + (space_above * 0.10)  # 10% of space above
            reduced_threshold = thresh - (space_below * 0.10)   # 10% of space below
            
            score = gs['score']
            if score > elevated_threshold:
                indicator = "↑"  # Elevated risk
            elif score < reduced_threshold:
                indicator = "↓"  # Reduced risk
            else:
                indicator = "→"  # Typical
            
            if gs['display_name'] != "Demographic Factors":
                lines.append(f"  {indicator} {gs['display_name']}: risk level {gs['level']} ({gs['confidence']} confidence)")
        
        # Add detailed summaries section
        lines.append("")
        lines.append("-" * 60)
        lines.append("DETAILED SEMANTIC FEATURE ANALYSIS BY DOMAIN:")
        lines.append("-" * 60)
        
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
                            #if len(str(summary_text)) > 500:
                            #    summary_text = str(summary_text)[:600] + "..."
                            lines.append(f"  - {summary_text}")
                            summaries_found = True
                
                if not summaries_found:
                    lines.append("  (No detailed analysis available)")
        
        # Add interpretation guidelines
        lines.append("")
        lines.append("-" * 60)
        # lines.append("INTERPRETATION GUIDELINES:")
        # lines.append("-" * 60)
        # lines.append("")
        # lines.append("The semantic analysis above was generated by validated ML models that")
        # lines.append("have learned the complex relationships between linguistic features and")
        # lines.append("cognitive status from clinical data.")
        # lines.append("")
        # lines.append("Key principles:")
        # lines.append("1. The INTEGRATED RISK SCORE is your decision-making guide")
        # lines.append("2. Domain indicators provide interpretable breakdowns")
        # lines.append("3. 'ELEVATED' levels suggest patterns associated with cognitive impairment")
        # lines.append("4. 'REDUCED' or 'TYPICAL' levels suggest patterns consistent with healthy controls")
        # lines.append("5. LOW-confidence domains should be mentioned but not weighted heavily")

        
        return "\n".join(lines)
    
    def save(self, filepath: str) -> None:
        """Save the trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MLSemanticBridge':
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


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


def get_feature_importance(bridge: MLSemanticBridge) -> pd.DataFrame:
    """
    Get feature importance from the trained models.
    
    Args:
        bridge: Trained MLSemanticBridge
        
    Returns:
        DataFrame with feature importance
    """
    importance_data = []
    
    for model_name, model in bridge.models.items():
        if hasattr(model, 'feature_importances_'):
            for feat, imp in zip(bridge.feature_names, model.feature_importances_):
                importance_data.append({
                    'model': model_name,
                    'feature': feat,
                    'importance': imp
                })
    
    df = pd.DataFrame(importance_data)
    
    if not df.empty:
        # Pivot to show all models
        pivot = df.pivot(index='feature', columns='model', values='importance')
        pivot['mean'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('mean', ascending=False)
        return pivot
    
    return df


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("ML Semantic Bridge Module")
    print("=" * 50)
    print(f"Features: {len(SEMANTIC_FEATURE_MAP)}")
    print(f"Feature groups: {len(SEMANTIC_FEATURE_GROUPS)}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"CatBoost available: {CATBOOST_AVAILABLE}")
    print()
    
    # Display feature groups
    for group_name, group_info in SEMANTIC_FEATURE_GROUPS.items():
        print(f"\n{group_info['display_name']}:")
        for feat in group_info['features']:
            info = group_info['feature_info'].get(feat, {})
            direction = "↑=healthy" if info.get('direction', 1) == 1 else "↑=impairment"
            weight = info.get('weight', 1.0)
            print(f"  - {feat} ({direction}, weight={weight})")