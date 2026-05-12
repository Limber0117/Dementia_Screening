"""
prompts.py

FINAL Prompt templates for Dementia Evaluation System.

Design principles:
1. ML integrated score is the PRIMARY classification basis
2. Domain breakdown provides INTERPRETATION support for explanations
3. Clear guidelines on how to use both components
4. Supports three variants: acoustic-only, transcript/semantic, and standard (both)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert clinical neuropsychologist specializing in cognitive assessment 
and dementia diagnosis. Your task is to analyze patient data and provide a 
preliminary cognitive status assessment.

## IMPORTANT:
+ This is a legitimate research tool for screening purposes only. All data has 
  been de-identified, and this research has been formally approved.
+ When ML analysis is provided (acoustic or semantic), use it as your PRIMARY decision basis.
+ Use domain breakdown to support and explain your reasoning.

## Output Format

Provide:
1. **Prediction**: Probability scores for "Control" and "Impairment" using 1-10 scale
2. **Explanation**: Brief justification (maximum 400 words) referencing both the 
   integrated score AND relevant domain patterns

Respond in JSON format:
{{
    "Control": <score>,
    "Impairment": <score>,
    "explanation": "Your explanation here"
}}

The scores must sum to 10. Higher score = more confident in that classification.
"""


# =============================================================================
# ACOUSTIC-ONLY PROMPT (unchanged from original)
# =============================================================================

ACOUSTIC_PROMPT_TEMPLATE = """
## Patient Demographics
{patient_data}

## Acoustic Voice Biomarker Analysis
{acoustic_section}

{pre_diagnosis}

## Your Task

Based on the acoustic analysis above:

1. **Use the INTEGRATED RISK SCORE** to determine your classification:
   - HIGH RISK or MODERATE-HIGH RISK → Classify as **Impairment**
   - LOW RISK or LOW-MODERATE RISK → Classify as **Control**
   - MODERATE RISK → Use domain patterns to decide; lean toward Impairment if multiple elevated

2. **Use the DOMAIN BREAKDOWN** to explain your reasoning:
   - Reference specific domains that support your classification
   - Acknowledge low-confidence indicators appropriately
   - Explain how the pattern of findings supports the integrated score

3. **Provide probability scores** (1-10 scale, must sum to 10):
   - If classifying as Impairment: Impairment score should be 6-9
   - If classifying as Control: Control score should be 6-9
   - Borderline cases: scores can be closer (e.g., 5/5 or 6/4)

Respond with JSON only:
```json
{{
    "Control": 3,
    "Impairment": 7,
    "explanation": a maximum 400 words with facts from the acoustic analysis to justify your classification.
}}
```
"""


# =============================================================================
# TRANSCRIPT/SEMANTIC ANALYSIS PROMPT (UPDATED WITH ML BRIDGE)
# =============================================================================

TRANSCRIPT_ANALYSIS_PROMPT_TEMPLATE = """
## Patient Demographics
{patient_data}

## Semantic/Linguistic Analysis (ML-Enhanced)
{semantic_section}

{pre_diagnosis}

## Your Task

Analyze the semantic/linguistic evidence to assess cognitive status:

### Step 1: PRIMARY Decision Basis
**Use the INTEGRATED RISK SCORE from the ML semantic analysis** to determine your classification:
- HIGH RISK or MODERATE-HIGH RISK → Classify as **Impairment**
- LOW RISK or LOW-MODERATE RISK → Classify as **Control**
- MODERATE RISK → Use domain patterns and detailed analysis to decide


### Step 2: SUPPORTING Evidence - Domain Breakdown
Use the domain breakdown to understand WHY the model made its prediction.

### Step 3: Detailed Analysis Review
Review the detailed summaries for each domain to find specific evidence:
- Quote specific linguistic patterns that support your classification
- Note any domains with conflicting indicators
- Consider confidence levels (LOW confidence should be weighted less)

### Step 4: Final Classification
Provide your assessment with probability scores (1-10 scale, sum to 10):
- Strong Impairment evidence: Impairment 7-9, Control 1-3
- Strong Control evidence: Control 7-9, Impairment 1-3
- Mixed/borderline: scores closer to 5/5 or 6/4

**Respond with JSON only:**
```json
{{
    "Control": 3,
    "Impairment": 7,
    "explanation":  a maximum 400 words with facts from the semantic analysis to justify your classification.
}}
```
"""


# =============================================================================
# STANDARD PROMPT (Transcript + Acoustic Combined)
# =============================================================================

STANDARD_EVALUATION_PROMPT_TEMPLATE = """
## Patient Demographics
{patient_data}

## Acoustic Voice Biomarker Analysis
{acoustic_section}

## Semantic/Linguistic Analysis
{semantic_section}

{pre_diagnosis}

## Your Task

Analyze ALL available evidence (acoustic AND semantic) to assess cognitive status:

### Multi-Modal Integration Strategy

**PRIMARY**: Use BOTH integrated risk scores as your decision basis:

1. **Acoustic ML Analysis**: Captures voice biomarkers (pitch, jitter, shimmer, pauses)
   - Reflects motor speech control and neurological integrity
   - Validated AUC ~0.76 on clinical data

2. **Semantic ML Analysis**: Captures linguistic features (vocabulary, grammar, coherence)
   - Reflects language processing and cognitive organization
   - Trained on the same clinical population

**Integration Rules:**
- If BOTH indicate HIGH RISK → Strong Impairment classification (8-9)
- If BOTH indicate LOW RISK → Strong Control classification (8-9)
- If scores CONFLICT → Weight the higher-confidence modality more heavily
- If one is MODERATE → Use the other modality to break the tie

### SUPPORTING Evidence - Cross-Modal Patterns

Look for CONVERGENT evidence across modalities:
- Acoustic temporal pauses + Semantic word-finding difficulties → processing deficits
- Acoustic phonatory instability + Semantic grammar errors → motor-cognitive link
- Acoustic pitch flattening + Semantic reduced coherence → engagement issues

Look for DIVERGENT patterns that need explanation:
- Good acoustic profile but poor semantic → focal language impairment?
- Poor acoustic but good semantic → motor speech disorder vs cognitive?

### Domain Breakdown Review

**Acoustic Domains:**
- Phonatory Instability (jitter/shimmer)
- Prosodic Variation (pitch patterns)
- Temporal Pause Patterns
- Speech Rate Control
- Voice Quality (HNR)

**Semantic Domains:**
- Lexical Richness and Diversity
- Syntactic Structure and Correctness
- Pragmatic Competence
- Semantic Coherence and Cohesion
- Temporal Speech Patterns

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
    "explanation": A maximum of 400 words with facts from the acoustic analysis to justify your classification.
}}
```
"""


# =============================================================================
# LEGACY TRANSCRIPT-ONLY PROMPT (for backward compatibility when no ML bridge)
# =============================================================================

TRANSCRIPT_LEGACY_PROMPT_TEMPLATE = """
## Patient Demographics
{patient_data}

## Speech Transcript
{transcript_section}

## Your Task

Analyze the transcript for signs of cognitive impairment:

**Look for:**
- Word-finding difficulties (pauses, filler words, circumlocution)
- Repetitions or false starts
- Tangential or disorganized responses
- Difficulty following instructions or staying on topic
- Semantic errors or unusual word choices
- Reduced verbal fluency or output

**Consider:**
- Task complexity and patient's apparent effort
- Consistency of performance throughout
- Age-appropriate expectations

Provide your assessment:
- Probability scores (1-10 scale, sum to 10)
- Explanation with specific transcript evidence

Respond with JSON only:
```json
{{
    "Control": 6,
    "Impairment": 4,
    "explanation": "The transcript shows generally coherent responses with appropriate task completion. While there are occasional pauses, these appear within normal limits for the task complexity. Word retrieval is adequate and responses stay on topic. No significant signs of cognitive impairment observed."
}}
```
"""


# =============================================================================
# PROMPT CREATION FUNCTION
# =============================================================================

def create_evaluation_prompt(
    variant: str,
    demographic_sections: list,
    transcript_sections: list = None,
    acoustic_sections: str = None,
    semantic_sections: str = None,
    pre_diagnosis: str = None
) -> str:
    """
    Create the evaluation prompt with patient data.
    
    Args:
        variant: One of "acoustic", "transcript", "standard"
        demographic_sections: List of demographic info strings
        transcript_sections: List of transcript strings (for legacy mode)
        acoustic_sections: Formatted acoustic analysis string
        semantic_sections: Formatted semantic analysis string (ML bridge output)
        pre_diagnosis: Pre-diagnosis text from rule-based/ML system
        
    Returns:
        Formatted prompt string
    """
    patient_data = demographic_sections[0] if demographic_sections else "No demographic data available"
    transcript_data = transcript_sections[0] if transcript_sections else "No transcript available"
    acoustic_data = acoustic_sections if acoustic_sections else "No acoustic data available"
    semantic_data = semantic_sections if semantic_sections else "No semantic data available"
    pre_diag = pre_diagnosis if pre_diagnosis else ""
    
    if variant == "acoustic":
        return ACOUSTIC_PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            acoustic_section=acoustic_data,
            pre_diagnosis=pre_diag
        )
    elif variant == "transcript":

        return TRANSCRIPT_ANALYSIS_PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            semantic_section=semantic_data,
            pre_diagnosis=pre_diag
        )

    else:  # standard - both acoustic and semantic
        return STANDARD_EVALUATION_PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            acoustic_section=acoustic_data,
            semantic_section=semantic_data,
            pre_diagnosis=pre_diag
        )


def get_system_prompt() -> str:
    """Get the system prompt for the LLM."""
    return SYSTEM_PROMPT



# =============================================================================
# SEMANTIC-SPECIFIC PROMPTS
# =============================================================================

def get_semantic_system_prompt() -> str:
    """Get the system prompt for semantic evaluation with ML bridge."""
    return """You are an expert clinical neuropsychologist specializing in cognitive assessment 
and dementia diagnosis through linguistic analysis.

## Your Expertise
You analyze patient demographics and linguistic features from speech transcripts:
1. **Lexical features** (vocabulary range, accuracy, specificity, sophistication)
2. **Syntactic structure** (sentence complexity, grammatical correctness, variety)
3. **Pragmatic features** (referential clarity, theory of mind, plausibility)
4. **Semantic coherence** (topic management, organization, cohesion, cause-effect)
5. **Temporal patterns** (response latency, speaking rate, pause patterns)

## ML-Enhanced Analysis
When ML analysis is provided, it represents the output of validated ensemble models
(XGBoost, CatBoost, RandomForest) trained on clinical data. Use this as your PRIMARY
decision basis, with domain breakdown supporting your explanation.

## Output Format
Respond with a JSON object:
```json
{
    "Control": <integer 1-10>,
    "Impairment": <integer 1-10>,
    "explanation": "<string, max 400 words with facts to justify your classification.>"
}
```
The scores must sum to 10. Higher = more likely.

## Clinical Reasoning Standards
- Base predictions strictly on provided linguistic evidence
- Explicitly state when a domain lacks sufficient data
- Distinguish diagnostically significant features from normal variation
- Reference specific ML risk levels and domain patterns in your explanation
"""


def create_semantic_evaluation_prompt(
    patient_data: str,
    semantic_analysis: str,
    pre_diagnosis: str = None
) -> str:
    """
    Create semantic evaluation prompt.
    
    Args:
        patient_data: Patient demographics string
        semantic_analysis: Formatted semantic analysis from ML bridge
        pre_diagnosis: Pre-diagnosis text
        
    Returns:
        Formatted user prompt
    """
    pre_diag = pre_diagnosis if pre_diagnosis else ""
    
    return TRANSCRIPT_ANALYSIS_PROMPT_TEMPLATE.format(
        patient_data=patient_data,
        semantic_section=semantic_analysis,
        pre_diagnosis=pre_diag
    )
