#!/usr/bin/env python3
"""
Dementia Evaluation System - Main Entry Point

This script provides a command-line interface for evaluating patients
for cognitive impairment using various LLM models.

Usage:
    python main.py                          # Run with default settings from .env
    python main.py --provider openai        # Use specific provider
    python main.py --model gpt-4o           # Use specific model
    python main.py --patient-ids P001 P002  # Evaluate specific patients
    python main.py --list-patients          # List available patient IDs
"""

import argparse
import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the .env file first
load_dotenv()

is_debug = os.getenv("DEBUG", "").lower() == "true"
result_folder = os.getenv("RESULTS_DIR", "datasets/results")
variant = os.getenv("PROMPTCONTENT", "standard").lower()

stop_reasoning_env = os.getenv("STOPREASOING", "false").lower()=="true"
is_reasoning = not stop_reasoning_env  # Send to LLM unless explicitly stopped

from config import load_config, validate_config, get_active_llm_config
from evaluator import DementiaEvaluator
from data_loader import DataLoader
from acoustic_clinical_profile import set_current_fold, get_test_patient_ids





def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def list_patients(env_path: str = None) -> None:
    """List all available patient IDs with data availability status."""
    config = load_config(env_path)
    loader = DataLoader(config.dataset)
    
    # Get different patient ID sets
    all_groundtruth_ids = set(loader.get_patient_ids())
    valid_ids = set(loader.get_valid_patient_ids())
    
    # Get IDs from features and transcripts separately
    features_df = loader._load_features()
    if not features_df.empty and "participant_id" in features_df.columns:
        feature_ids = set(features_df["participant_id"].dropna().astype(str).unique())
    else:
        feature_ids = set()
    
    transcript_ids = set()
    if os.path.isdir(config.dataset.transcripts_dir):
        for f in os.listdir(config.dataset.transcripts_dir):
            if f.endswith(".txt"):
                transcript_ids.add(f[:-4])
    
    # Combine all known IDs
    all_ids = all_groundtruth_ids.union(feature_ids).union(transcript_ids)
    
    if not all_ids:
        print("No patients found in the dataset.")
        return
    
    print(f"\n{'='*70}")
    print("Patient Data Availability")
    print(f"{'='*70}")
    print(f"Total unique patient IDs: {len(all_ids)}")
    print(f"With groundtruth: {len(all_groundtruth_ids)}")
    print(f"With acoustic features: {len(feature_ids)}")
    print(f"With transcripts: {len(transcript_ids)}")
    print(f"With BOTH features & transcripts (valid for evaluation): {len(valid_ids)}")
    
    print(f"\n{'-'*70}")
    print(f"{'PID':<20} | {'Features':^10} | {'Transcript':^10} | {'Groundtruth':^12} | {'Diagnosis':<12}")
    print(f"{'-'*70}")
    
    for pid in sorted(all_ids):
        has_features = "✓" if pid in feature_ids else "✗"
        has_transcript = "✓" if pid in transcript_ids else "✗"
        has_groundtruth = "✓" if pid in all_groundtruth_ids else "✗"
        
        diagnosis = ""
        if pid in all_groundtruth_ids:
            diagnosis = loader.get_groundtruth_diagnosis(pid) or ""
        
        # Highlight valid patients
        marker = " *" if pid in valid_ids else ""
        print(f"{pid:<20} |     {has_features}      |     {has_transcript}      |      {has_groundtruth}       | {diagnosis:<12}{marker}")
    
    print(f"\n* = Valid for evaluation (has both features and transcript)")
    print()


def show_config(env_path: str = None) -> None:
    """Display current configuration."""
    config = load_config(env_path)
    
    print("\n" + "=" * 50)
    print("Current Configuration")
    print("=" * 50)
    
    print("\nDataset Paths:")
    print(f"  Groundtruth CSV: {config.dataset.groundtruth_csv}")
    print(f"  Acoustic Features: {config.dataset.acoustic_features_csv}")
    print(f"  Transcripts Dir: {config.dataset.transcripts_dir}")
    print(f"  Results Dir: {config.dataset.results_dir}")
    
    print(f"\nActive LLM:")
    print(f"  Provider: {config.active_provider}")
    print(f"  Model: {config.active_model}")
    
    print("\nAvailable Providers:")
    providers = [
        ("OpenAI", config.openai),
        ("Anthropic", config.anthropic),
        ("Google", config.google),
        ("DeepSeek", config.deepseek),
        ("Ollama", config.ollama),
    ]
    
    for name, llm_config in providers:
        has_key = "✓" if llm_config.api_key else "✗"
        if name == "Ollama":
            has_key = "✓ (local)"
        print(f"  {name}: {has_key} (model: {llm_config.model})")
    
    # Validate
    warnings = validate_config(config)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    
    print()


def calculate_cv_metrics(evaluator, rootfolder: str = None, n_folds: int = 5) -> dict:
    """
    Calculate aggregated metrics across all CV folds.
    """
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    for fold in range(1, n_folds + 1):
        results_path = os.path.join(rootfolder, f"results_fold_{fold}.csv")
        print(f"Loading results from: {results_path}")
        
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found, skipping fold {fold}")
            continue
        
        try:
            results = pd.read_csv(results_path)
            print(f"  Successfully loaded fold {fold}: {results.shape[0]} rows, {results.shape[1]} columns")
            print(f"  Columns: {results.columns.tolist()}")
            
            y_true = []
            y_pred = []
            y_prob = []
            
            # Check which columns actually exist
            # Find columns - be more specific with your column names
            groundtruth_col = 'GroundTruth' if 'GroundTruth' in results.columns else None
            prediction_col = 'Prediction' if 'Prediction' in results.columns else None
            impairment_conf_col = 'impairment_conf' if 'impairment_conf' in results.columns else None
            control_conf_col = 'control_conf' if 'control_conf' in results.columns else None
            
            if not all([groundtruth_col, prediction_col, impairment_conf_col, control_conf_col]):
                print(f"  ⚠️ Missing required columns!")
                print(f"  Available columns: {results.columns.tolist()}")
                continue
            
            print(f"  Using columns:")
            print(f"    GroundTruth: {groundtruth_col}")
            print(f"    Prediction: {prediction_col}")
            print(f"    impairment_conf: {impairment_conf_col}")
            print(f"    control_conf: {control_conf_col}")
            
            print(f"  Using columns -> GroundTruth: {groundtruth_col}, Prediction: {prediction_col}, Im/Co Confidence: {impairment_conf_col}, {control_conf_col}")
            
            # Process each row
            for idx, row in results.iterrows():
                # Get ground truth
                if groundtruth_col and groundtruth_col in row:
                    gt = str(row[groundtruth_col]).lower().strip()
                else:
                    # Try alternative column names
                    gt = ''
                    for col in results.columns:
                        if any(keyword in col.lower() for keyword in ['ground', 'truth', 'diagnosis', 'actual']):
                            gt = str(row[col]).lower().strip()
                            break
                
                # Get prediction
                if prediction_col and prediction_col in row:
                    pred = str(row[prediction_col]).lower().strip()
                else:
                    pred = ''
                    for col in results.columns:
                        if any(keyword in col.lower() for keyword in ['predict', 'prediction', 'result']):
                            pred = str(row[col]).lower().strip()
                            break
                
                # Skip if we couldn't get valid values
                if not gt or not pred:
                    continue
                
                # Convert to binary (1 = Impairment, 0 = Control)
                if gt in ['impairment', 'mci', 'dementia', 'impaired', '1']:
                    y_true.append(1)
                elif gt in ['control', 'healthy', 'normal', '0']:
                    y_true.append(0)
                else:
                    continue  # Skip unknown labels
                
                if pred in ['impairment', 'impaired', '1', 'yes', 'positive']:
                    y_pred.append(1)
                elif pred in ['control', 'healthy', 'normal', '0', 'no', 'negative']:
                    y_pred.append(0)
                else:
                    # Default to control if unclear
                    y_pred.append(0)
                
                # Get probability/confidence
                # **CRITICAL: Calculate PROBABILITY of impairment (0-1)**
                # We have BOTH impairment_conf and control_conf (1-10 scale)
                # Calculate probability using softmax for proper 0-1 scale
                try:
                    # Get confidence scores
                    imp_conf = float(row['impairment_conf'])
                    ctrl_conf = float(row['control_conf'])
                    
                    # Method 1: SOFTMAX (best for probabilities)
                    # This properly normalizes to 0-1 and handles scale differences
                    imp_exp = np.exp(imp_conf)
                    ctrl_exp = np.exp(ctrl_conf)
                    prob = imp_exp / (imp_exp + ctrl_exp)
                    
                    # Method 2: SIMPLE RATIO (alternative)
                    # prob = imp_conf / (imp_conf + ctrl_conf)
                    
                    # Method 3: NORMALIZED DIFFERENCE (if you want more extreme probabilities)
                    # diff = imp_conf - ctrl_conf  # Range: -9 to +9
                    # prob = (diff + 9) / 18.0      # Normalize to 0-1
                    
                    # Method 4: DIRECT SCALING (if using 1-10 as probability)
                    # prob = imp_conf / 10.0  # WRONG - don't use this!
                    
                    # Ensure valid probability
                    prob = max(0.0, min(1.0, prob))
                    
                except (ValueError, KeyError) as e:
                    # Fallback if confidence values are missing/invalid
                    if y_pred[-1] == 1:  # Predicted impairment
                        prob = 0.8
                    else:  # Predicted control
                        prob = 0.2
                
                y_prob.append(prob)
            
            if len(y_true) == 0:
                print(f"  Warning: No valid samples found in fold {fold}")
                continue
            
            print(f"  Valid samples: {len(y_true)}")
            print(f"  Class distribution - Control: {y_true.count(0)}, Impairment: {y_true.count(1)}")
            
            # Calculate fold metrics
            try:
                if len(set(y_true)) > 1:  # Need both classes for AUC
                    auc = roc_auc_score(y_true, y_prob)
                else:
                    auc = 0
                    print(f"  Warning: Only one class present, AUC set to 0")
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                
                fold_result = {
                    'fold': fold,
                    'n_samples': len(y_true),
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'sensitivity': recall_score(y_true, y_pred, zero_division=0),
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'auc': auc,
                    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                }
                
                fold_metrics.append(fold_result)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                all_y_prob.extend(y_prob)
                
                print(f"  Fold {fold} metrics - Acc: {fold_result['accuracy']:.3f}, Sens: {fold_result['sensitivity']:.3f}, Spec: {fold_result['specificity']:.3f}")
                
            except Exception as e:
                print(f"  Error calculating metrics for fold {fold}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing fold {fold}: {e}")
            continue
    
    # Aggregate across folds
    if len(fold_metrics) == 0:
        print("\nERROR: No valid fold results found after processing all CSV files")
        return {'error': 'No valid fold results found'}
    
    metrics_to_aggregate = ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'auc']
    
    aggregated = {
        'n_folds': len(fold_metrics),
        'total_samples': sum(f['n_samples'] for f in fold_metrics),
        'fold_results': fold_metrics,
    }
    
    for metric in metrics_to_aggregate:
        values = [f[metric] for f in fold_metrics]
        aggregated[f'mean_{metric}'] = np.mean(values)
        aggregated[f'std_{metric}'] = np.std(values, ddof=1)  # Use ddof=1 for sample std dev
    
    # Overall confusion matrix
    aggregated['total_tp'] = sum(f['tp'] for f in fold_metrics)
    aggregated['total_tn'] = sum(f['tn'] for f in fold_metrics)
    aggregated['total_fp'] = sum(f['fp'] for f in fold_metrics)
    aggregated['total_fn'] = sum(f['fn'] for f in fold_metrics)
    
    # Calculate overall metrics from aggregated predictions
    if all_y_true and all_y_pred:
        try:
            aggregated['overall_accuracy'] = accuracy_score(all_y_true, all_y_pred)
            aggregated['overall_precision'] = precision_score(all_y_true, all_y_pred, zero_division=0)
            aggregated['overall_sensitivity'] = recall_score(all_y_true, all_y_pred, zero_division=0)
            aggregated['overall_f1'] = f1_score(all_y_true, all_y_pred, zero_division=0)
            
            overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
            overall_tn, overall_fp, overall_fn, overall_tp = overall_cm.ravel()
            aggregated['overall_specificity'] = overall_tn / (overall_tn + overall_fp) if (overall_tn + overall_fp) > 0 else 0
            
            if len(set(all_y_true)) > 1:
                aggregated['overall_auc'] = roc_auc_score(all_y_true, all_y_prob)
            else:
                aggregated['overall_auc'] = 0
        except Exception as e:
            print(f"Warning: Could not calculate overall metrics: {e}")
    
    print(f"\nAggregated total samples: {aggregated['total_samples']}")
    
    return aggregated


def print_cv_summary(cv_metrics: dict):
    """
    Print a formatted summary of CV results.
    """
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    
    # Handle error case
    if 'error' in cv_metrics:
        print(f"\nERROR: {cv_metrics['error']}")
        print("=" * 70)
        return
    
    print(f"\nTotal samples evaluated: {cv_metrics['total_samples']}")
    print(f"Number of folds: {cv_metrics['n_folds']}")
    
    print("\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'N':<6} {'Acc':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8} {'F1':<8} {'AUC':<8}")
    print("-" * 70)
    
    for f in cv_metrics['fold_results']:
        print(f"{f['fold']:<6} {f['n_samples']:<6} {f['accuracy']:<8.4f} "
              f"{f['precision']:<8.4f} {f['sensitivity']:<8.4f} {f['specificity']:<8.4f} "
              f"{f['f1']:<8.4f} {f['auc']:<8.4f}")
    
    print("\n--- Aggregated Results (Mean ± Std across folds) ---")
    print(f"Accuracy:    {cv_metrics['mean_accuracy']:.4f} ± {cv_metrics['std_accuracy']:.4f}")
    print(f"Precision:   {cv_metrics['mean_precision']:.4f} ± {cv_metrics['std_precision']:.4f}")
    print(f"Sensitivity: {cv_metrics['mean_sensitivity']:.4f} ± {cv_metrics['std_sensitivity']:.4f}")
    print(f"Specificity: {cv_metrics['mean_specificity']:.4f} ± {cv_metrics['std_specificity']:.4f}")
    print(f"F1-Score:    {cv_metrics['mean_f1']:.4f} ± {cv_metrics['std_f1']:.4f}")
    print(f"AUC:         {cv_metrics['mean_auc']:.4f} ± {cv_metrics['std_auc']:.4f}")
    
    # Print overall metrics if available
    if 'overall_accuracy' in cv_metrics:
        print("\n--- Overall Metrics (Pooled predictions) ---")
        print(f"Accuracy:    {cv_metrics['overall_accuracy']:.4f}")
        print(f"Precision:   {cv_metrics['overall_precision']:.4f}")
        print(f"Sensitivity: {cv_metrics['overall_sensitivity']:.4f}")
        print(f"Specificity: {cv_metrics['overall_specificity']:.4f}")
        print(f"F1-Score:    {cv_metrics['overall_f1']:.4f}")
        print(f"AUC:         {cv_metrics['overall_auc']:.4f}")
    
    print("\n--- Overall Confusion Matrix (Sum across folds) ---")
    print(f"TP: {cv_metrics['total_tp']}, TN: {cv_metrics['total_tn']}, "
          f"FP: {cv_metrics['total_fp']}, FN: {cv_metrics['total_fn']}")
    
    # Calculate metrics from confusion matrix
    total_positive = cv_metrics['total_tp'] + cv_metrics['total_fn']
    total_negative = cv_metrics['total_tn'] + cv_metrics['total_fp']
    
    if total_positive > 0:
        sensitivity = cv_metrics['total_tp'] / total_positive
    else:
        sensitivity = 0
        
    if total_negative > 0:
        specificity = cv_metrics['total_tn'] / total_negative
    else:
        specificity = 0
        
    total_predictions = cv_metrics['total_tp'] + cv_metrics['total_tn'] + cv_metrics['total_fp'] + cv_metrics['total_fn']
    if total_predictions > 0:
        accuracy = (cv_metrics['total_tp'] + cv_metrics['total_tn']) / total_predictions
    else:
        accuracy = 0
        
    print(f"\n--- Metrics from Confusion Matrix ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    print("=" * 70)


def evaluate_single_patient(
    pid: str, 
    env_path: str = None,
    provider: str = None,
    model: str = None
) -> None:
    """Evaluate a single patient and display results."""
    evaluator = DementiaEvaluator(env_path=env_path)
    
    if provider or model:
        evaluator.setup_provider(provider, model)
    
    # Load patient data
    patient_data = evaluator.data_loader.get_patient_data(pid)
    
    if patient_data is None:
        print(f"Error: Patient {pid} not found in the dataset.")
        return
    
    print(f"\nEvaluating patient: {pid}")
    print("-" * 40)
    
    # Show patient info
    info = patient_data.patient_info
    print(f"Age: {info.age or 'N/A'}")
    print(f"Gender: {info.gender or 'N/A'}")
    print(f"Language: {info.language or 'N/A'}")
    print(f"Topic: {info.topic or 'N/A'}")
    print(f"Ground Truth: {info.ground_truth_diagnosis or 'N/A'}")
    
    print("\nTranscript Preview:")
    if patient_data.transcript.patient_utterances:
        for i, utt in enumerate(patient_data.transcript.patient_utterances[:3]):
            print(f"  {i+1}. {utt[:80]}...")
    else:
        print("  (No transcript available)")
    
    print("\nEvaluating with LLM...")
    result = evaluator.evaluate_patient(variant, patient_data)
    
    print("\n" + "=" * 40)
    print("EVALUATION RESULT")
    print("=" * 40)
    print(f"Prediction: {result.prediction.value}")
    print(f"Model: {result.provider}/{result.model_name}")
    print(f"Processing Time: {result.processing_time_seconds:.2f}s")
    
    if result.error:
        print(f"Error: {result.error}")
    
    print(f"\nExplanation:")
    print(result.explanation)
    print()


def main():
    """
    ACTIVE_LLM_PROVIDER=google
    ACTIVE_LLM_MODEL=gemini-2.5-flash
    """
    default_llm_provider = os.getenv("ACTIVE_LLM_PROVIDER")
    default_llm_model = os.getenv("ACTIVE_LLM_MODEL")

    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dementia Evaluation System using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py                              # Run full evaluation
            python main.py --provider anthropic         # Use Claude
            python main.py --model gemini-2.5-pro       # Use specific model
            python main.py --evaluate-patient P001      # Evaluate single patient
            python main.py --list-patients              # Show available patients
            python main.py --show-config                # Show configuration
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--list-patients",
        action="store_true",
        help="List all available patient IDs"
    )
    mode_group.add_argument(
        "--show-config",
        action="store_true",
        help="Display current configuration"
    )
    mode_group.add_argument(
        "--evaluate-patient",
        type=str,
        metavar="PID",
        help="Evaluate a single patient by ID"
    )
    mode_group.add_argument(
        "--variant",
        type=str,
        default=variant,
        metavar="Data Variant",
        help="Options: standard, acoustic, transcript"
    )
    
    # LLM configuration
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "google", "deepseek", "ollama"],
        help="LLM provider to use (overrides .env)",
        default=default_llm_provider
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (overrides .env)",
        default=default_llm_model
    )
    
    # Input/output
    parser.add_argument(
        "--env",
        type=str,
        default="05_classification_explanation/.env",
        help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output file path for results"
    )
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        type=str,
        help="Specific patient IDs to evaluate (if not specified, evaluates all valid patients)"
    )
    
    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--noprogress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--nometrics",
        action="store_true",
        help="Not to calculate and display evaluation metrics"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle different modes
    if args.list_patients:
        list_patients(args.env)
        return
    
    if args.show_config:
        show_config(args.env)
        return
    
    if args.evaluate_patient:
        evaluate_single_patient(
            args.evaluate_patient,
            args.env,
            args.provider,
            args.model
        )
        return
    
    # Default: Run full evaluation
    print("\n" + "=" * 50)
    print("Dementia Evaluation System")
    print("=" * 50)
    
    # Load and validate config
    config = load_config(args.env)
    warnings = validate_config(config)
    
    for w in warnings:
        logging.warning(w)
    
    # Get active provider info
    provider_name, llm_config = get_active_llm_config(config)
    
    if args.provider:
        provider_name = args.provider
    if args.model:
        llm_config.model = args.model
    
    print(f"\nUsing: {provider_name}/{llm_config.model}")
    
    # Initialize evaluator
    evaluator = DementiaEvaluator(config=config)
    evaluator.setup_provider(args.provider, args.model)
    
    # Determine which patients to evaluate
    if args.patient_ids:
        # Use specified patient IDs
        patient_ids = args.patient_ids
        print(f"\nEvaluating specified patients: {len(patient_ids)}")
    else:
        # Use 5-fold cross-validation patient IDs
        print("\nRunning 5-fold cross-validation...")
    
    # Loop through all 5 folds
    for fold in range(1, 6):
        print(f"\n{'='*60}")
        print(f"EVALUATING FOLD {fold}")
        print(f"{'='*60}")
        
        # Set current fold (loads correct model)
        set_current_fold(fold)
        
        # Get test patient IDs for this fold
        patient_ids = get_test_patient_ids(fold)
        # use a small portion for preliminary evaluation, please remove the following later.
        if is_debug:
            patient_ids = patient_ids[:2]
        
        print(f"Test patients: {len(patient_ids)}")
        

        # Run evaluation
        results = evaluator.evaluate_all(
            variant=args.variant,
            patient_ids=patient_ids,
            show_progress=True
        )
        
        # Save results for this fold
        output_path = os.path.join(result_folder, f"results_fold_{fold}.csv")
        evaluator.save_results(results, output_path=output_path)
        print(f"Results saved to: {output_path}")
        
    # After all folds, calculate aggregated metrics
    cv_metrics = calculate_cv_metrics(evaluator, result_folder, n_folds=5)
    print_cv_summary(cv_metrics)

    # Save aggregated results as CSV
    if 'error' not in cv_metrics:
        # Convert the summary metrics to a DataFrame
        summary_data = {}
        for key, value in cv_metrics.items():
            if key != 'fold_results':  # Skip nested structure
                summary_data[key] = value
        
        summary_df = pd.DataFrame([summary_data])
        summary_csv_path = os.path.join(result_folder, "cv_results_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nAggregated summary saved to: {summary_csv_path}")
        
        # Save fold-wise results as CSV
        fold_results_df = pd.DataFrame(cv_metrics['fold_results'])
        fold_results_csv_path = os.path.join(result_folder, "cv_fold_results.csv")
        fold_results_df.to_csv(fold_results_csv_path, index=False)
        print(f"Fold-wise results saved to: {fold_results_csv_path}")
        
        # Also save as JSON for completeness
        json_path = os.path.join(result_folder, "cv_results_summary.json")
        with open(json_path, 'w') as f:
            json.dump(cv_metrics, f, indent=2)
        print(f"Complete results saved to: {json_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()