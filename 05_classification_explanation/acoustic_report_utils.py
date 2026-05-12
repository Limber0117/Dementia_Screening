"""
Acoustic Report Utilities

This module provides modular functions for:
1. Loading feature definitions from CSV
2. Looking up definitions for patient features
3. Building structured reports with human-readable descriptions
4. Converting structured reports to plain-text clinical summaries

Usage:
    from acoustic_report_utils import (
        load_feature_definitions,
        get_feature_definition,
        build_patient_report,
        convert_report_to_plain_text
    )
"""

import os
import csv
import logging
from typing import Optional, Union
from dataclasses import dataclass, field
from typing import TYPE_CHECKING,TypedDict,List

if TYPE_CHECKING:
    from llm_providers import LLMProvider

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# Default path to feature definitions CSV
feature_definition = os.getenv("FEATURE_DEFINITIONS_CSV", "datasets/output/acoustic_features/audio_features_definitions.csv")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureInfo:
    """Information about a single acoustic feature."""
    feature: str
    value: float
    definition: Optional[str] = None
    z_score: Optional[float] = None
    category: Optional[str] = None  # e.g., "normal", "elevated", "reduced"
    is_concerning: Optional[bool] = None

class FeatureDict(TypedDict):
    feature: str
    value: float
    z_score: float
    category: str
    is_concerning: bool

@dataclass
class PatientReport:
    """Structured report for a patient."""
    patient_id: str
    normal_features: list[FeatureInfo] = field(default_factory=list)
    abnormal_features: list[FeatureInfo] = field(default_factory=list)


@dataclass
class ConversionResult:
    """Result of converting a report to plain text."""
    patient_id: str
    plain_text: str
    success: bool
    error: Optional[str] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None


# =============================================================================
# FEATURE DEFINITIONS FUNCTIONS
# =============================================================================

def load_feature_definitions(
    csv_path: str = feature_definition
) -> dict[str, str]:
    """
    Load feature definitions from a CSV file.
    
    Args:
        csv_path: Path to the CSV file with 'feature_name' and 'definition' columns.
        
    Returns:
        Dictionary mapping feature names to their definitions.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If required columns are missing.
        
    Example:
        >>> definitions = load_feature_definitions()
        >>> print(definitions['sample_rate'])
        'Number of audio samples per second (Hz)'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Definitions CSV not found: {csv_path}")
    
    definitions = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate columns exist
        fieldnames = reader.fieldnames or []
        feature_col = None
        def_col = None
        
        for col in fieldnames:
            col_lower = col.lower().strip()
            if 'feature' in col_lower or col_lower == 'feature_name':
                feature_col = col
            if 'def' in col_lower or 'definition' in col_lower:
                def_col = col
        
        if not feature_col or not def_col:
            raise ValueError(
                f"CSV must have 'feature_name' and 'definition' columns. "
                f"Found columns: {fieldnames}"
            )
        
        for row in reader:
            feature_name = row[feature_col].strip()
            definition = row[def_col].strip()
            if feature_name:
                definitions[feature_name] = definition
    
    logger.info(f"\nLoaded {len(definitions)} feature definitions from {csv_path}")
    return definitions


def get_feature_definition(
    feature_name: str,
    definitions: dict[str, str],
    default: Optional[str] = None
) -> str:
    """
    Get the definition for a specific feature.
    
    Args:
        feature_name: Name of the feature to look up.
        definitions: Dictionary of feature definitions.
        default: Default value if feature not found. If None, returns the feature name.
        
    Returns:
        The feature definition or default/feature name if not found.
        
    Example:
        >>> definitions = load_feature_definitions()
        >>> desc = get_feature_definition('jitter', definitions)
        >>> print(desc)
        'Cycle-to-cycle variation in fundamental frequency (pitch perturbation)'
    """
    if feature_name in definitions:
        return definitions[feature_name]
    
    # Try case-insensitive lookup
    for key, value in definitions.items():
        if key.lower() == feature_name.lower():
            return value
    
    return default if default is not None else feature_name



# =============================================================================
# PATIENT REPORT BUILDING FUNCTIONS
# =============================================================================


def build_patient_report(
    patient_id: str,
    patient_data: List[FeatureDict],
    definitions: dict[str, str]
) -> PatientReport:
    """
    Build a structured PatientReport from patient data.
    
    Args:
        patient_id: Patient identifier.
        patient_data: List of FeatureDict objects.
        definitions: Dictionary of feature definitions.        
    Returns:
        PatientReport with features categorized as normal/abnormal.
    """
    report = PatientReport(patient_id=patient_id)
    
    for item in patient_data:
        feature_name = item['feature']
        value = item['value']
        z_score = item['z_score']
        category = item['category']
        is_concerning = item['is_concerning']

        feature = FeatureInfo(
            feature=feature_name,
            value=value,
            definition=get_feature_definition(feature_name, definitions),
            z_score=z_score,
            category=category,
            is_concerning=is_concerning
        )
                
        # Categorize as normal or abnormal
        if is_concerning:
            report.abnormal_features.append(feature)
        else:
            report.normal_features.append(feature)
    
    return report


def format_patient_report_structured(
    report: PatientReport,
    use_definitions: bool = True
) -> str:
    """
    Format a PatientReport as a structured text report.
    
    Args:
        report: The PatientReport object to format.
        use_definitions: If True, use feature definitions instead of names.
        
    Returns:
        Formatted structured report string.
        
    Example:
        >>> formatted = format_patient_report_structured(report)
        >>> print(formatted)
    """
    lines = []
    logger.info(f" - Patient {report.patient_id} - Acoustic Feature Analysis")
    lines.append("")
    
    # Normal features section
    if report.normal_features:
        lines.append("### Features Within Normal Limits:")
        for f in report.normal_features:
            display_name = f.definition if use_definitions and f.definition else f.name
            z_str = f"z={f.z_score:+.4f}" if f.z_score is not None else ""
            lines.append(f"- {display_name}: value is {f.value:.4f} | Z-Score is {z_str}")
        lines.append("")
    
    # Abnormal features section
    if report.abnormal_features:
        lines.append("### Features Requiring Attention:")
        for f in report.abnormal_features:
            display_name = f.definition if use_definitions and f.definition else f.name
            z_str = f"z={f.z_score:+.4f}" if f.z_score is not None else ""
            lines.append(f"* {display_name}: value is {f.value:.4f} | Z-Score is {z_str} | {f.category}")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# LLM CONVERSION FUNCTIONS
# =============================================================================

# System prompt for clinical summary generation
CLINICAL_SUMMARY_SYSTEM_PROMPT = """You are a clinical speech-language pathologist and voice specialist with expertise in acoustic analysis for cognitive impairment detection. Convert structured acoustic feature reports into readable, clinically meaningful plain-text summaries. This requires deep understanding of speech physiology, acoustic metrics, and clinical implications that are relevant to cognitive decline screening.



Guidelines:
1. Summarize all normal features concisely in a few sentences
2. For abnormal features, explain in detail:
   - Physiological significance
   - Possible causes/mechanisms
   - Relationships between abnormal features
   - Include Z-scores in parentheses for abnormal features
3. Provide clinical impression of the pattern using professional but accessible language    
4. Do not provide final diagnoses or recommendations; focus on feature interpretation only

Example: The patient presented with significant motor speech impairment. The most notable abnormalities were a significantly slowed speech rate (Z=-2.1) and extremely high frequency of pauses (Z=+1.8). Furthermore, the jitter/shimmer ratio was within the normal range, indicating that the vocal organs themselves were relatively functioning correctly, and the problem likely stemmed primarily from pauses at the cognitive planning level rather than from muscle control.   

Output plain text paragraphs (NOT JSON).
"""


def create_llm_provider(env_path: Optional[str] = None) -> "LLMProvider":

    from config import load_config, get_active_llm_config  # Lazy import
    from llm_providers import get_provider                  # Lazy import

    config = load_config(env_path)
    provider_name, llm_config = get_active_llm_config(config)
    
    return get_provider(
        provider_name=provider_name,
        llm_config=llm_config,
        processing_config=config.processing
    )


def convert_report_to_plain_text(
    structured_report: str,
    provider: Optional["LLMProvider"] = None,
    patient_id: str = None,
    env_path: Optional[str] = None
) -> ConversionResult:
    """
    Convert a structured report to plain-text clinical summary using LLM.
    
    Args:
        structured_report: The structured report string to convert.
        provider: Optional pre-configured LLM provider.
        patient_id: Patient identifier for tracking.
        env_path: Optional path to .env file (used if provider not given).
        
    Returns:
        Conversion result with plain text or error information.
        
    Example:
        >>> result = convert_report_to_plain_text(formatted_report)
        >>> if result.success:
        ...     print(result.plain_text)
    """
    # Create provider if not given
    if provider is None:
        provider = create_llm_provider(env_path)
    
    user_prompt = f"""Please convert the following structured acoustic feature report into a plain-text clinical summary.

    ORIGINAL STRUCTURED REPORT:
    {structured_report}

    Provide the summary with one paragraph for both normal features and abnormal features, following the clinical guidelines provided.
    """
    try:
        plain_text = provider._call_api(
            system_prompt=CLINICAL_SUMMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            patient_id=patient_id
        )
        
        return ConversionResult(
            patient_id=patient_id,
            plain_text=plain_text,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Conversion failed for {patient_id}: {e}")
        return ConversionResult(
            patient_id=patient_id,
            plain_text="",
            success=False,
            error=str(e)
        )


# =============================================================================
# HIGH-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def process_patient_to_plain_text(
    patient_id: str,
    #patient_data: list[dict[str, float,float,str,bool]],
    patient_data: list[FeatureDict],
    definitions_csv: str = feature_definition,
    provider: Optional["LLMProvider"] = None,
    use_definitions: bool = True
) -> ConversionResult:
    
    # Step 1: Load definitions
    try:
        definitions = load_feature_definitions(definitions_csv)
    except FileNotFoundError:
        logger.warning(f"Definitions file not found: {definitions_csv}. Using feature names.")
        definitions = {}
    logger.info(f" - Loaded {len(definitions)} feature definitions.")

    # Step 2: Build patient report, convert the feature names to detailed feature definitions
    report = build_patient_report(
        patient_id=patient_id,
        patient_data=patient_data,
        definitions=definitions
    )
    logger.info(f" - Built report for patient {patient_id} with {len(report.normal_features)} normal and {len(report.abnormal_features)} abnormal features.")

    # Step 3: Format as structured report, with key-value pairs and other details
    structured_report = format_patient_report_structured(
        report=report,
        use_definitions=use_definitions
    )
    logger.info(f" - Formatted structured report for patient {patient_id}.")
    
    # Step 4: Convert to plain text
    result = convert_report_to_plain_text(
        structured_report=structured_report,
        provider=provider,
        patient_id=patient_id
    )
    logger.info(f"\n - Converted report to plain text for patient {patient_id}. Success: {result.success}")
    
    return result




# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ACOUSTIC REPORT UTILITIES - MODULE TEST")
    print("=" * 70)
    
    print("\n1. To load feature definitions from CSV, run:")
    # 1. Load definitions
    definitions = load_feature_definitions()
    # 2. Generate report for a test patient
    report = build_patient_report(
        patient_id=None, # test patient ID
        patient_data=None, # load sample patient data for this test
        definitions=definitions # load definitions for this test
    )    
    # 3. To format structured report
    formatted = format_patient_report_structured(report, use_definitions=True)
    print(formatted)
    
    # 4. To convert to plain text using LLM

    #create LLM provider
    provider = create_llm_provider()
    result = process_patient_to_plain_text(
        patient_id=None, # test patient ID
        patient_data=None, # load sample patient data for this test
        definitions_csv=feature_definition,
        provider=provider,
        use_definitions=True
        )
    if result.success:
        print(result.plain_text)
