"""
evaluator.py

Main evaluator module for Dementia Evaluation System.
Updated to support ML Bridge mode for acoustic feature analysis.
"""

import os
import time
import json
import csv
import logging
from typing import Tuple, Dict, Optional, List
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from config import Config, load_config, get_active_llm_config
from models import EvaluationResult, PatientEvaluationInput, DiagnosisResult
from data_loader import DataLoader
from llm_providers import get_provider, LLMProvider
from prompts import (
    get_system_prompt,
    create_evaluation_prompt,
)
from acoustic_clinical_profile import (
    PopulationStatsBuilder,
    AcousticClinicalProfile,
    format_for_llm,
    rule_based_diagnosis,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
result_folder = os.getenv("RESULTS_DIR", "datasets/results")
is_debug = os.getenv("DEBUG", "False").lower() == "true"

# IMPORTANT: STOPREASOING=false means DO send to LLM (is_reasoning=True)
#            STOPREASOING=true means DON'T send to LLM (is_reasoning=False)
stop_reasoning_env = os.getenv("STOPREASOING", "false").lower() == "true"
is_reasoning = not stop_reasoning_env  # Send to LLM unless explicitly stopped

is_discrete = os.getenv("DISCRETE", "true").lower() == "true"
normalization_dataset_path = os.getenv(
    "NORMALISATION_DATASET_PATH", 
    "datasets/output/acoustic_features/merged_data.csv"
)
use_prediction = os.getenv("USE_PREPREDICTION", "True").lower() == "true"
USE_ML_BRIDGE = os.getenv("USE_ML_BRIDGE", "True").lower() == "true"
ML_BRIDGE_PATH = os.getenv("ML_BRIDGE_PATH", "ml_acoustic_bridge.pkl")

# Log configuration on startup
print(f"[Evaluator Config]")
print(f"  USE_ML_BRIDGE: {USE_ML_BRIDGE}")
print(f"  ML_BRIDGE_PATH: {ML_BRIDGE_PATH}")
print(f"  is_reasoning (send to LLM): {is_reasoning}")
print(f"  is_discrete: {is_discrete}")
print(f"  use_prediction: {use_prediction}")
print(f"  is_debug: {is_debug}")


def feature_normalization(dataset_path: str = normalization_dataset_path) -> Tuple[Dict, Dict]:
    """Load dataset and compute normative statistics for z-score mode."""
    rows_all, rows_healthy = [], []
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_all.append(row)
            if row.get("diagnosis", "").lower() == "control":
                rows_healthy.append(row)
    return PopulationStatsBuilder.build(rows_healthy), PopulationStatsBuilder.build(rows_all)


def ensure_ml_bridge_exists(data_path: str = normalization_dataset_path,
                            bridge_path: str = ML_BRIDGE_PATH) -> bool:
    """Ensure ML Bridge model exists."""
    if os.path.exists(bridge_path):
        print(f"  ML Bridge found at: {bridge_path}")
        return True
    print(f"  ML Bridge NOT found at: {bridge_path}")
    if not os.path.exists(data_path):
        print(f"  Training data NOT found at: {data_path}")
        return False
    try:
        print(f"  Training ML Bridge from: {data_path}")
        from acoustic_clinical_profile import train_ml_bridge
        train_ml_bridge(data_path, bridge_path)
        return True
    except Exception as e:
        logger.error(f"Failed to train ML Bridge: {e}")
        return False


class DementiaEvaluator:
    """Main class for evaluating patients for dementia."""
    
    def __init__(self, config: Optional[Config] = None, env_path: Optional[str] = None):
        self.config = config or load_config(env_path)
        self.data_loader = DataLoader(self.config.dataset)
        self.provider: Optional[LLMProvider] = None
        self.results: List[EvaluationResult] = []
        self.acoustic_analyzer = None
        self._init_acoustic_analyzer()
    
    def _init_acoustic_analyzer(self):
        """Initialize acoustic analyzer (ML Bridge or z-score mode)."""
        if USE_ML_BRIDGE:
            if ensure_ml_bridge_exists():
                self.acoustic_analyzer = AcousticClinicalProfile(
                    use_ml_bridge=True, ml_bridge_path=ML_BRIDGE_PATH)
                print(f"  Acoustic analyzer: ML Bridge mode")
            else:
                print(f"  Falling back to z-score mode")
                self._init_zscore_analyzer()
        else:
            self._init_zscore_analyzer()
    
    def _init_zscore_analyzer(self):
        """Initialize z-score based analyzer."""
        try:
            healthy_stats, cohort_stats = feature_normalization()
            self.acoustic_analyzer = AcousticClinicalProfile(
                healthy_stats=healthy_stats, cohort_stats=cohort_stats, use_ml_bridge=False, ml_bridge_path=None)
            print(f"  Acoustic analyzer: Z-score mode")
        except Exception as e:
            logger.error(f"Failed to initialize z-score analyzer: {e}")
            self.acoustic_analyzer = None
    
    def setup_provider(self, provider_name: Optional[str] = None, model: Optional[str] = None):
        """Setup LLM provider."""
        if provider_name: 
            self.config.active_provider = provider_name
        if model: 
            self.config.active_model = model
        provider_name, llm_config = get_active_llm_config(self.config)
        print(f"  LLM Provider: {provider_name}, Model: {llm_config.model}")
        self.provider = get_provider(provider_name, llm_config, self.config.processing)
    
    def parse_acoustic_features(self, acoustic_list: list) -> dict:
        """Parse acoustic features from formatted list."""
        patient = {}
        for text in acoustic_list:
            for line in text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or not line.startswith('- '):
                    continue
                content = line[2:]
                if ': ' in content:
                    key, value = content.split(': ', 1)
                    try: 
                        patient[key] = float(value)
                    except: 
                        patient[key] = value
        return patient
    
    def convert_to_plain_features(self, patient: dict) -> str:
        """Convert features dict to plain text."""
        return "\n\n".join([f"**{k}:** {v}" for k, v in patient.items()])
    
    def generate_acoustic_prompt(self, patient: dict, gender: str = "male", 
                                  pid: str = "") -> Tuple[str, str]:
        """Generate acoustic analysis prompt using ML Bridge or z-score."""
        if self.acoustic_analyzer is None:
            print(f"  WARNING: No acoustic analyzer, using plain features")
            return "", self.convert_to_plain_features(patient)
        
        profile = self.acoustic_analyzer.analyse_patient(patient, pid)
        prompt = format_for_llm(profile)
        pre_diagnosis = rule_based_diagnosis(profile) if use_prediction else ""
        
        # Debug output
        if is_debug:
            logger.info(f"\n[Acoustic Analysis for {pid}]")
            logger.info(f"  Mode: {profile.get('mode', 'unknown')}")
            if profile.get('mode') == 'ml_bridge':
                logger.info(f"  Overall Score: {profile.get('overall_score', 0):.2%}")
                logger.info(f"  Category: {profile.get('overall_category', 'unknown')}")
                logger.info(f"  ML Prediction: {profile.get('ml_prediction', 'unknown')}")
        
        self._log_analysis_results(profile, pid)
        return pre_diagnosis, prompt
    
    def _log_analysis_results(self, profile: Dict, pid: str):
        """Log analysis results to CSV."""
        if not is_debug: 
            return
        row = {"participant_id": pid}
        if profile.get("mode") == "ml_bridge":
            row.update({
                "overall_score": profile.get("overall_score", 0),
                "overall_category": profile.get("overall_category", ""),
                "ml_prediction": profile.get("ml_prediction", ""),
            })
        mode = profile.get("mode", "z_score")
        output_csv = os.path.join(result_folder, f"acoustic_analysis_{mode}.csv")
        os.makedirs(result_folder, exist_ok=True)
        file_exists = os.path.isfile(output_csv)
        with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists: 
                writer.writeheader()
            writer.writerow(row)
    
    def save_to_txt(self, content, filename, folder):
        """Save content to text file."""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(str(content))
    
    def evaluate_patient(self, variant: str, patient_data: PatientEvaluationInput) -> Optional[EvaluationResult]:
        """Evaluate a single patient."""
        if self.provider is None: 
            self.setup_provider()
        
        pid = patient_data.patient_info.pid
        gender = (patient_data.patient_info.gender or "unknown").lower()
        
        try:
            # Get prompts
            system_prompt = get_system_prompt()
            demo, trans, acou = patient_data.format_for_llm()
            acoustic_features = self.parse_acoustic_features(acou)
            
            # Generate acoustic analysis
            if is_discrete or USE_ML_BRIDGE:
                pre_diag, acoustic_prompt = self.generate_acoustic_prompt(acoustic_features, gender, pid)
                if not use_prediction: 
                    pre_diag = ""

                '''def create_evaluation_prompt(
                        variant: str,
                        demographic_sections: list,
                        transcript_sections: list,
                        acoustic_sections: str,
                        pre_diagnosis: str = None
                    ) 
                '''
                user_prompt = create_evaluation_prompt(variant, demo, trans, acoustic_prompt, pre_diag)
            else:
                plain = self.convert_to_plain_features(acoustic_features) + "\n\n"
                user_prompt = create_evaluation_prompt(variant, demo, trans, plain, None)
            
            # Debug: print prompt
            if is_debug: 
                print(f"\n{'='*60}")
                print(f"USER PROMPT FOR {pid}:")
                print(f"{'='*60}")
                print(user_prompt[:2000] + "..." if len(user_prompt) > 2000 else user_prompt)
            
            print(f" + Evaluating patient PID: {pid}")
            
            # Save prompt to file (for debugging or when not sending to LLM)
            if is_debug or not is_reasoning:
                self.save_to_txt(f"System: {system_prompt}\n\n{user_prompt}", f"{pid}.txt", result_folder)
                print(f"   Prompt saved to: {result_folder}/{pid}.txt")
            
            # Send to LLM if is_reasoning is True
            if is_reasoning:
                print(f"   Sending to LLM...")
                result = self.provider.evaluate(pid, system_prompt, user_prompt)
                print(f"   LLM Response: {result.prediction.value if result else 'None'}")
                return result
            else:
                print(f"   SKIPPED: is_reasoning=False (STOPREASOING=true in .env)")
                return None
            
        except Exception as e:
            logger.error(f"Error evaluating patient {pid}: {e}")
            import traceback
            traceback.print_exc()
            return EvaluationResult(
                pid=pid, 
                prediction=DiagnosisResult.UNKNOWN, 
                explanation="",
                model_name=self.provider.config.model if self.provider else "",
                provider=self.provider.provider_name if self.provider else "",
                error=str(e), 
                variant=variant
            )
    
    def evaluate_all(self, variant: str = "standard", patient_ids: Optional[List[str]] = None,
                    show_progress: bool = True) -> List[EvaluationResult]:
        """Evaluate all patients."""
        if self.provider is None: 
            self.setup_provider()
        
        self.results = []
        patients = list(self.data_loader.iter_patients(patient_ids))
        
        print(f"\nEvaluating {len(patients)} patients...")
        print(f"  Variant: {variant}")
        print(f"  Send to LLM: {is_reasoning}")
        
        if show_progress: 
            patients = tqdm(patients, desc="Evaluating")
        
        for p in patients:            
            result = self.evaluate_patient(variant, p)
            if result is not None:
                self.results.append(result)
            
            # Rate limiting
            if is_reasoning: 
                time.sleep(5)
        
        print(f"\nCompleted: {len(self.results)} results collected")
        return self.results
    
    def save_results(self, results: Optional[List[EvaluationResult]] = None,
                    output_path: Optional[str] = None) -> str:
        """Save evaluation results to CSV."""
        results = results or self.results
        if not results: 
            print("No results to save")
            return ""
        
        # Ensure results directory exists
        results_dir = self.config.dataset.results_dir if hasattr(self.config.dataset, 'results_dir') else result_folder
        os.makedirs(results_dir, exist_ok=True)
        
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "mlbridge" if USE_ML_BRIDGE else "zscore"
            provider = results[0].provider if results[0].provider else "unknown"
            model = results[0].model_name if hasattr(results[0], 'model_name') and results[0].model_name else "unknown"
            model_clean = model.replace("/", "-").replace(":", "-")
            filename = f"results_{provider}_{model_clean}_{mode}_{ts}.csv"
            output_path = os.path.join(results_dir, filename)
        
        # If output_path is just a filename without directory, add results_dir
        if not os.path.dirname(output_path):
            output_path = os.path.join(results_dir, output_path)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df = pd.DataFrame([r.to_dict() for r in results])
        df["GroundTruth"] = [self.data_loader.get_groundtruth_diagnosis(r.pid) for r in results]
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        return output_path
    
    def calculate_metrics(self, results: Optional[List[EvaluationResult]] = None) -> dict:
        """Calculate evaluation metrics."""
        results = results or self.results
        if not results: 
            return {}
        
        y_true, y_pred = [], []
        for r in results:
            gt = self.data_loader.get_groundtruth_diagnosis(r.pid)
            if not gt: 
                continue
            gt_n = DiagnosisResult.from_string(gt)
            if gt_n.value == "Unknown" or r.prediction.value == "Unknown": 
                continue
            y_true.append(gt_n.value)
            y_pred.append(r.prediction.value)
        
        if not y_true: 
            return {}
        
        yt = [1 if l == "Impairment" else 0 for l in y_true]
        yp = [1 if l == "Impairment" else 0 for l in y_pred]
        
        tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
        
        total = len(y_true)
        acc = (tp + tn) / total if total else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
        
        return {
            "accuracy": acc, 
            "precision": prec, 
            "sensitivity": sens, 
            "recall": sens,  # Same as sensitivity
            "specificity": spec,
            "f1_score": f1, 
            "auc": (sens + spec) / 2, 
            "total": total,
            "true_positives": tp, 
            "true_negatives": tn,
            "false_positives": fp, 
            "false_negatives": fn,
            "support_impairment": tp + fn,
            "support_control": tn + fp,
            "confusion_matrix": {
                "Control": {"Control": tn, "Impairment": fp},
                "Impairment": {"Control": fn, "Impairment": tp}
            }
        }


def run_evaluation(env_path: Optional[str] = None, provider: Optional[str] = None,
                  model: Optional[str] = None, patient_ids: Optional[List[str]] = None,
                  output_path: Optional[str] = None) -> str:
    """Convenience function to run a complete evaluation."""
    evaluator = DementiaEvaluator(env_path=env_path)
    if provider or model:
        evaluator.setup_provider(provider, model)
    evaluator.evaluate_all(patient_ids=patient_ids)
    return evaluator.save_results(output_path=output_path)