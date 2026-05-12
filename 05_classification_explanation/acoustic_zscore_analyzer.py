"""
Dementia Detection: Acoustic Feature Z-Score Normalizer
========================================================
Customized for your dataset with 76 features from 269 healthy controls.

Usage:
------
1. Load the pre-computed normative statistics
2. Transform patient features to z scores
3. Generate LLM-ready prompts for dementia screening

"""
from scipy.stats import percentileofscore
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os,sys
from pathlib import Path
from dotenv import load_dotenv
import acoustic_report_utils as aru

from config import Config, load_config, get_active_llm_config


# Load the .env file first
load_dotenv()
from pathlib import Path

# Optional: force cwd to script root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

z_score_feature_file = os.getenv("Z_SCORE_FEATURES")
normative_status_json = os.getenv("NORMATIVE_STATUS_JSON")
#print(f" Normative status JSON file: {normative_status_json} {z_score_feature_file}")
is_debug = os.getenv("DEBUG", "").lower() == "true"
use_key_features = os.getenv("USE_KEY_FEATURES_ONLY","False").lower() =="true"
use_plain_text_features = os.getenv("USE_PLAIN_TEXT_FEATURES","False").lower() =="true"
feature_definition = os.getenv("FEATURE_DEFINITIONS_CSV", "datasets/output/acoustic_features/audio_features_definitions.csv")
print(f" - Use key features only for analysis: {use_key_features} ")

@dataclass
class FeatureStats:
    """Statistics for a single acoustic feature from healthy controls"""
    mean: float
    std: float
    median: float
    p25: float
    p75: float
    p5: float
    p95: float
    min_val: float
    max_val: float
    n_samples: int


class DementiaAcousticAnalyzer:
    """
    Analyzes acoustic features for dementia screening using z-score normalization
    against healthy controls.
    """
    
    # Key features for dementia detection (based on literature)
    FULL_KEY_DEMENTIA_FEATURES = [
        # -------------------------------------------------------------------------
        # TIER 1 - Primary Features (Strongest Evidence)
        # -------------------------------------------------------------------------
        
        # Pause/Hesitation Features (TOP predictors per literature)
        'pause_count',
        'pause_mean',
        'pause_total',
        'pause_ratio',
        'pause_variability',
        'long_pause_count',
        'long_pause_total',
        'hesitation_count',
        'hesitation_rate',
        
        # Speech/Articulation Rate
        'speaking_rate',
        'articulation_rate',
        'speech_rate_mean_local',
        'speech_rate_variability',
        
        # Voice Quality - Jitter/Shimmer 
        'jitter', #Less reliable across recording conditions.
        'shimmer', #Less reliable across recording conditions.
        'opensmile_jitterLocal_sma3nz_amean',
        'opensmile_shimmerLocaldB_sma3nz_amean',
        
        # HNR
        'hnr_mean',
        'opensmile_HNRdBACF_sma3nz_amean',
        
        # Pitch Variability (Prosody)
        'pitch_std',
        'pitch_range',
        'pitch_iqr',
        
        # -------------------------------------------------------------------------
        # TIER 2 - Secondary Features (Supporting Evidence)
        # -------------------------------------------------------------------------
        
        # Voice Breaks
        'voice_breaks_count',
        'voice_breaks_rate',
        'voice_break_degree',
        
        # Voiced Segments
        'voiced_ratio',
        'phonation_time_ratio',
        'opensmile_MeanVoicedSegmentLengthSec',
        'opensmile_MeanUnvoicedSegmentLength',
        
        # Cepstral Peak Prominence
        'cpp_mean',
        'cpp_median',
        
        # Formants (weaker evidence, more relevant in later stages)
        'F1_mean',
        'F2_mean',
        'F3_mean',
        'F1_std',
        'F2_std',
        'F3_std',
        
        # Intensity/Loudness
        'opensmile_loudness_sma3_amean',
    ]

    KEY_DEMENTIA_FEATURES = set(FULL_KEY_DEMENTIA_FEATURES)
    
    # Features where HIGHER values suggest pathology (concerning when elevated)
    HIGHER_IS_ABNORMAL = {
        # Voice Quality - Perturbation measures
        'jitter',
        'shimmer',
        'opensmile_jitterLocal_sma3nz_amean',
        'opensmile_shimmerLocaldB_sma3nz_amean',
        
        # Pause/Hesitation Features
        'pause_count',
        'pause_mean',
        'pause_std',
        'pause_total',
        'pause_ratio',
        'pause_variability',
        'long_pause_count',
        'long_pause_total',
        'hesitation_count',
        'hesitation_rate',
        
        # Voice Breaks
        'voice_breaks_count',
        'voice_breaks_rate',
        'voice_break_degree',
        
        # Temporal - Task Duration
        'durationTotal',
        
        # Speech Variability (irregularity)
        'speech_rate_variability',
        
        # Unvoiced Segment Features
        'opensmile_MeanUnvoicedSegmentLength',
        'opensmile_StddevUnvoicedSegmentLength',
    }
    
    # Features where LOWER values suggest pathology (concerning when reduced)
    LOWER_IS_ABNORMAL = {
        # Speech/Articulation Rate
        'speaking_rate',
        'articulation_rate',
        'speech_rate_mean_local',
        
        # HNR (voice quality)
        'hnr_mean',
        'opensmile_HNRdBACF_sma3nz_amean',
        
        # Pitch Variability (reduced = monotonic speech)
        'pitch_std',
        'pitch_range',
        'pitch_iqr',
        
        # Cepstral Peak Prominence (voice periodicity)
        'cpp_mean',
        'cpp_median',
        
        # Voiced Ratio
        'voiced_ratio',
        'phonation_time_ratio',
        
        # Intensity/Loudness
        'intensity_mean',
        'opensmile_loudness_sma3_amean',
        
        # Voiced Segment Features
        'opensmile_MeanVoicedSegmentLengthSec',
        'opensmile_VoicedSegmentsPerSec',
        'opensmile_loudnessPeaksPerSec',
    }
    
    def __init__(self):
        self.stats: Dict[str, FeatureStats] = {}
        self.feature_names: List[str] = []
        self.n_healthy_controls: int = 0
        self.is_fitted = False
        self.gender: str = "male" # Placeholder for future demographic adjustments

    
    def fit_from_csv(self, healthy_csv_path: str, exclude_cols: List[str] = None, gender:str = "male") -> 'DementiaAcousticAnalyzer':
        """
        Compute normative statistics from healthy controls CSV.
        
        Args:
            healthy_csv_path: Path to CSV with healthy control acoustic features
            exclude_cols: Columns to exclude (e.g., ['sample_rate', 'participant_id'])
        """
        if gender.lower()=="male":
            df = pd.read_csv(Path(healthy_csv_path+"_male.csv"))
        else:
            df = pd.read_csv(Path(healthy_csv_path+"_female.csv"))
   

        if exclude_cols is None:
            exclude_cols = ['sample_rate','duration','filename','jitter', 'opensmile_jitterLocal_sma3nz_amean','shimmer','opensmile_shimmerLocaldB_sma3nz_amean','opensmile_shimmerLocaldB_sma3nz_stddevNorm',
                            'hnr_mean','hnr_std','opensmile_HNRdBACF_sma3nz_amean','opensmile_HNRdBACF_sma3nz_stddevNorm']  # Default: exclude constant columns
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        self.n_healthy_controls = len(df)
        self.feature_names = feature_cols
        
        for feature in feature_cols:
            values = df[feature].dropna().values
            if len(values) == 0:
                continue
                
            self.stats[feature] = FeatureStats(
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)),
                median=float(np.median(values)),
                p25=float(np.percentile(values, 25)),
                p75=float(np.percentile(values, 75)),
                p5=float(np.percentile(values, 5)),
                p95=float(np.percentile(values, 95)),
                min_val=float(np.min(values)),
                max_val=float(np.max(values)),
                n_samples=len(values)
            )
        
        self.is_fitted = True
        print(f"Fitted on {len(self.stats)} features from {self.n_healthy_controls} healthy controls.")
        return self
    
    def load_normative_stats(self, json_path: str, gender:str="male") -> 'DementiaAcousticAnalyzer':
        """Load pre-computed normative statistics from JSON."""

        if gender.lower()=="male":
            json_path = str(Path(json_path+"_male.json"))
        else:
            json_path = str(Path(json_path+"_female.json"))

        with open(json_path, 'r') as f:
            stats_dict = json.load(f)
        
        for feature, s in stats_dict.items():
            self.stats[feature] = FeatureStats(
                mean=s['mean'], std=s['std'], median=s['median'],
                p25=s['p25'], p75=s['p75'], p5=s['p5'], p95=s['p95'],
                min_val=s['min'], max_val=s['max'], n_samples=s['n_samples']
            )
        
        self.feature_names = list(self.stats.keys())
        self.n_healthy_controls = self.stats[self.feature_names[0]].n_samples
        self.is_fitted = True
        #print(f"Loaded {len(self.stats)} features (n={self.n_healthy_controls} healthy controls)")
        return self
    
    def save_normative_stats(self, json_path: str, gender:str="male"):
        """Save normative statistics to JSON for later use."""
        stats_dict = {
            feat: {
                'mean': s.mean, 'std': s.std, 'median': s.median,
                'p25': s.p25, 'p75': s.p75, 'p5': s.p5, 'p95': s.p95,
                'min': s.min_val, 'max': s.max_val, 'n_samples': s.n_samples
            }
            for feat, s in self.stats.items()
        }

        if gender.lower()=="male":
            json_path = str(Path(json_path+"_male.json"))
        else:
            json_path = str(Path(json_path+"_female.json"))

        with open(json_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Saved to {json_path}")
    
    def compute_zscore(self, feature: str, value: float) -> float:
        """Compute z-score for a single feature value."""
        if feature not in self.stats:
            return np.nan
        s = self.stats[feature]
        if s.std == 0:
            return 0.0
        return (value - s.mean) / s.std
    
    
    def get_category(self, feature: str, z: float) -> Tuple[str, bool]:
        """
        Get interpretive category and whether it's concerning.
        Returns: (category_string, is_concerning)
        """
        feat_lower = feature.lower()
        higher_bad = any(p in feat_lower for p in self.HIGHER_IS_ABNORMAL)
        lower_bad = any(p in feat_lower for p in self.LOWER_IS_ABNORMAL)
        
        is_concerning = False
        
        if higher_bad:
            if z > 2:
                return "markedly elevated", True
            elif z > 1:
                is_concerning = z > 1.5
                return "elevated", is_concerning
            elif z > -1:
                return "normal", False
            else:
                return "below typical", False
        elif lower_bad:
            if z < -2:
                return "markedly reduced", True
            elif z < -1:
                is_concerning = z < -1.5
                return "reduced", is_concerning
            elif z < 1:
                return "normal", False
            else:
                return "above typical", False
        else:
            if abs(z) > 2:
                return "markedly deviant", True
            elif abs(z) > 1:
                return "outside typical range", abs(z) > 1.5
            else:
                return "normal", False
    
    def analyze_patient(self, features: Dict[str, float], use_key_features_only: bool = use_key_features) -> Dict:
        """
        Analyze a patient's acoustic features.
        
        Args:
            features: Dict of {feature_name: value}
            use_key_features_only: If True, only analyze dementia-relevant features
        
        Returns:
            Dict with z-scores, categories, and summary
        """
      

        if use_key_features_only:
            features = {k: v for k, v in features.items() if k in self.KEY_DEMENTIA_FEATURES}

        # use DEBUG to test the received patient features in self.stats
        if is_debug:
            print(f"- Type of features: {type(features)} and amount: {len(features)} ")
            print(f"- First few entries in features: {list(features)[:10]}")
            print(f"- Last five entries in features: {list(features)[-5:]}")

        results = []
        concerning_features = []
        
        for feat, val in features.items():
            if feat not in self.stats:
                continue
            
            z = self.compute_zscore(feat, val)
            category, is_concerning = self.get_category(feat, z)
            
            results.append({
                'feature': feat,
                'value': val,
                'z_score': z,
                'category': category,
                'is_concerning': is_concerning
            })
            
            if is_concerning:
                concerning_features.append((feat, z, category))
        
        # Sort concerning features by absolute z-score
        #concerning_features.sort(key=lambda x: abs(x[1]), reverse=True)
        

        # use DEBUG to test the received patient features in self.stats
        if is_debug:
            print(f"\n -- given patient -- \nType of feature results: {type(results)}")
            print(f"First few entries in feature results: {list(results)[:10]}")
            print(f"Last five entries in feature results: {list(results)[-5:]}")


        return {
            'features': results,
            'concerning': concerning_features,
            'n_concerning': len(concerning_features),
            'n_analyzed': len(results)
        }
    
    def create_llm_prompt(
        self, 
        features: Dict[str, float],
        patient_id: str = None,
        use_key_features_only: bool = use_key_features,
        gender: str = "male"
    ) -> str:
        """
        Create a complete LLM prompt for dementia screening.
        
        Args:
            features: Dict of acoustic feature values
            patient_id: Optional patient identifier
            use_key_features_only: Focus on dementia-relevant features
            include_system_prompt: Include full system instructions
        
        Returns:
            Complete prompt string ready for LLM
        """
        print(f" - Amount of features used for prompt generation: {len(features)} ")
        analysis = self.analyze_patient(features, use_key_features)
        
        if use_plain_text_features:
            # Build feature analysis section with plain text features
            result = aru.process_patient_to_plain_text(
                patient_id=patient_id, # test patient ID
                patient_data=analysis['features'], # load sample patient data for this test
                definitions_csv=feature_definition,
                use_definitions=True
                )
            if result.success:
                plain_features = result.plain_text
            else:
                plain_features = "Error generating plain text features."

        else:
            # Build feature analysis section with z-scores features
            feature_lines = []
            for r in analysis['features']:
                feature_lines.append(f"- {r['feature']}: {r['value']:.4f} | z={r['z_score']:+.4f} | {r['category']}")
            feature_text = "\n\n".join(feature_lines)
            
            # Build concerning features section
            if analysis['concerning']:
                concerning_text = "\n\n###  Features Requiring Attention:\n\n"
                for feat, z, cat in analysis['concerning']:
                    concerning_text += f"  - {feat}: {cat} (z={z:+.2f})\n\n"
            else:
                concerning_text = "\n\n### All features within normal range:\n\n"
        
        # Build the prompt
        

        if use_plain_text_features:
            
            prompt = f"""
            ## Context
            The patient's acoustic features are discretised based on a large number of healthy controls. The analysis of those features are as follows.

            {plain_features}

            """
        else:
            prompt = f"""
            ## Context
            The patient's acoustic features are discretised  based on a large number of healthy controls.

            {feature_text}

            {concerning_text}

            """
        #if is_debug:
            #print(prompt)
        
        return prompt
    
    def process_patient_csv(
        self, 
        patient_csv_path: str,
        patient_id_col: str = None,
        output_format: str = "prompts"
    ) -> List[Dict]:
        """
        Process multiple patients from a CSV file.
        
        Args:
            patient_csv_path: Path to CSV with patient acoustic features
            patient_id_col: Column name for patient IDs
            output_format: "prompts" for LLM prompts, "analysis" for raw analysis
        
        Returns:
            List of dicts with patient_id and prompt/analysis
        """
        df = pd.read_csv(patient_csv_path)

        results = []
        
        for idx, row in df.iterrows():
            patient_id = row[patient_id_col] if patient_id_col else f"Patient_{idx}"
            
            # Get features (exclude ID column)
            features = {
                col: row[col] for col in df.columns 
                if col != patient_id_col and col in self.stats and pd.notna(row[col])
            }
            
            if output_format == "prompts":
                prompt = self.create_llm_prompt(features, patient_id=patient_id, use_key_features_only=use_key_features)
                results.append({'patient_id': patient_id, 'prompt': prompt})
            else:
                analysis = self.analyze_patient(features, use_key_features)
                results.append({'patient_id': patient_id, 'analysis': analysis})  

        if is_debug:
            amount = len(results)
            print(f"- A total of {amount} patients are extracted with features. ")

        return results


    def get_important_features(self) ->str:

        return self.FULL_KEY_DEMENTIA_FEATURES



