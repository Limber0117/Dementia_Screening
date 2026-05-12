"""
Prompts for Semantic Feature Evaluation using LLMs.

This module contains system and user prompts for evaluating
linguistic features in dementia screening transcripts.
"""

from typing import Optional

def get_semantic_evaluation_system_prompt() -> str:

    return """You are an expert clinical linguist specializing in speech pattern analysis for cognitive assessment. You conduct cognitive impairment screenings by evaluating linguistic data from patient assessments.
    
    ## Your Expertise

    You analyze patient demographics (age, gender) and four linguistic domains:  
    1. **Lexical features** (e.g., vocabulary range, lexical accurary, specificity, advanced vocabulary)
    2. **Syntactic structure** (e.g., sentence complexity, grammatical correctness, fragmentation and structural variety)
    3. **Pragmatic features** (e.g., referential clarity, state-of-mind language, implausible details)
    4. **Semantic coherence** (e.g., topic management, logical organisation, cohension, cause and effect, repetition, information prioritisation)

    ## Output Requirements

    For each patient, respond with a JSON object:
    ```json
    {
        "Control": <integer 1-10>,
        "Impairment": <integer 1-10>,
        "explanation": "<string, max 1000 words>"
    }
    ```
    The sum of two probability is equal to 10, while a higher value means more likely.

    ## Explanation Guidelines

    Your explanation will be reviewed by a physician for diagnostic decision-making. Structure it as follows:

    1. Open with a summary assessment
    2. Address each linguistic domain, citing specific findings from the input
    3. Reference age/gender if clinically relevant
    4. Acknowledge limitations, missing data, or conflicting indicators
    5. Conclude with key factors driving your probability assignments

    Write in clear clinical prose based on the provided evidence, without bullet points. Maximum 1000 words.

    ## Clinical Reasoning Standards

    - Base predictions strictly on provided linguistic evidence
    - Explicitly state when a domain lacks sufficient data
    - Distinguish diagnostically significant features from normal variation
    - Recognize that some features overlap across categories but differ in severity
"""

def get_semantic_evaluation_user_prompt(patient_data: str, patient_feature: str, pre_diagnosis: str) -> str:
    """
    Generate user prompt for dementia screening evaluation.
    
    Args:
        patient_data: Patient demographics (age, gender, etc.)
        patient_feature: Speech feature indicators with z scores and confidence values
    
    Returns:
        Formatted user prompt string
    """
    
    prompt = f"""## Patient Information

    ### Demographics
    {patient_data}

    ### Speech Feature Indicators
    {patient_feature}

    **Indicator schema:**
    - `z`: Deviation score (lower = more pathological)
    - `confidence`: Measurement reliability (0.0–1.0), only take it as secondary reference
    - `level`: Qualitative label 

    ---

    ## Instructions

    Analyze this patient's cognitive status for cognitive impairment screening.

    ### Reasoning Process

    1. **Examine evidence**: Review z scores and confidence across all domains. Identify domains with low z scores and clear evidence within the descriptions.

    2. **Assess pattern consistency**:
    - Control: z scores generally above or within normal range, or mildly reduced. May have a few mild abnormalities but more aspects are in good condition.
    - Impairment: Consistent abnormalities across multiple domains, e.g., consistently markedly decreased.

    3. **Assign probabilities**: Rate each category on a 1–10 scale based on evidence strength.

    ### Output Requirements

    Respond with **only** a JSON object in this exact format:
    ```json
    {{
        "Control": <int>,
        "Impairment": <int>,
        "explanation": "<string>"
    }}
    ```

    **Constraints:**
    - Each score: integer from 1 (very unlikely) to 10 (very likely)
    - Scores must sum to exactly 10
    - Explanation: maximum 1000 words with facts captured from input
    - Prioritise numeric z scores and qualitative labels in your reasoning
    - Take confidences as references, prioritise features with higher confidence
    - Reasoning based on available facts/evidence 

    Do not include any text outside the JSON object.
    """

    return prompt


def get_semantic_system_prompt() -> str:
    """
    Get the system prompt for semantic feature evaluation.
    
    Returns:
        System prompt string
    """
    return """You are an expert clinical linguist specializing in the analysis of speech patterns for cognitive assessment. Your task is to evaluate transcripts from patients describing a specifc topic, e.g., the "Cookie Theft" picture or an old stories. The topic is a standard clinical task used in cognitive impairment screening.

 You can detect the topic from the conversation transcript and find the benchmark example (for comparison) from your knowledge.

Your evaluation should be objective, precise, and clinically relevant. Focus on linguistic indicators that may suggest cognitive impairment, but also note preserved abilities.

For each feature you evaluate, provide:
1. A concise summary (max 300 words) describing your observations with specific examples
2. A score from 1 to 10 where:
    - 1 = Severely impaired / Minimal function
    - 2 = Markedly impaired / Very poor performance
    - 3 = Moderately-to-severely impaired / Poor performance
    - 4 = Moderately impaired / Below average
    - 5 = Mildly impaired / Slightly below average
    - 6 = Borderline / Low-normal
    - 7 = Normal / Average performance
    - 8 = Normal / Above average
    - 9 = Fully intact / Excellent performance
    - 10 = Superior / Exceptional performance
3. A confidence level from 0 to 10 indicating how certain you are of your rating

Be specific in your analysis - quote directly from the transcript when illustrating points.
Consider both what is present AND what is notably absent in the patient's description."""


def get_lexical_evaluation_prompt(transcript_text: str, pid: str) -> str:
    """
    Get prompt for lexical feature evaluation.
    
    Args:
        transcript_text: Patient's transcript text
        pid: Patient ID
        
    Returns:
        User prompt for lexical evaluation
    """
    return f"""Evaluate the LEXICAL features of the following transcript.

TRANSCRIPT:
{transcript_text}

Evaluate the following features:

1. **vocabulary_range**: Assess the variety and breadth of vocabulary used. Consider:
   - Number of different word types used
   - Whether vocabulary is limited or diverse
   - Use of synonyms and varied expressions

2. **lexical_accuracy**: Evaluate the correctness of word choice and its impact on clarity. Consider:
   - Appropriate word selection for intended meaning
   - Word substitution errors (semantic paraphasias)
   - Vague or imprecise word choices

3. **specificity**: Measure the precision in using specific terms over generic ones. Consider:
   - Use of specific nouns (e.g., "kitchen", "stool", "cookie jar") vs. vague terms ("thing", "that", "place")
   - Specific verbs vs. generic ones
   - Level of descriptive detail

4. **advanced_vocabulary**: Assess the use of complex and less common words. Consider:
   - Presence of sophisticated vocabulary
   - Use of technical or domain-specific terms when appropriate
   - Reliance on basic vs. advanced vocabulary

Respond in the following JSON format:
{{
    "vocabulary_range": {{
        "summary": "Your analysis with specific examples from the transcript (max 2300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "lexical_accuracy": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "specificity": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "advanced_vocabulary": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }}
}}"""


def get_syntactic_evaluation_prompt(transcript_text: str, pid: str) -> str:
    """
    Get prompt for syntactic feature evaluation.
    
    Args:
        transcript_text: Patient's transcript text
        pid: Patient ID
        
    Returns:
        User prompt for syntactic evaluation
    """
    return f"""Evaluate the SYNTACTIC features of the following transcript.

TRANSCRIPT:
{transcript_text}

Evaluate the following features:

1. **grammar_complexity**: Assess the use of complex sentence structures. Consider:
   - Simple vs. compound vs. complex sentences
   - Use of subordinate clauses
   - Sentence embedding and coordination
   - Variety in sentence length

2. **structure_variety**: Evaluate the diversity of grammatical forms used. Consider:
   - Variety of tenses (past, present, progressive)
   - Use of modal verbs (can, could, should, etc.)
   - Active vs. passive voice
   - Question forms, conditionals, etc.

3. **grammar_correctness**: Focus on the frequency and types of grammatical mistakes, and give a higher score if fewer errors occur. Consider:
   - Subject-verb agreement errors
   - Tense inconsistencies
   - Missing or incorrect articles
   - Incomplete sentences or abandoned utterances
   - Word order errors

Respond in the following JSON format:
{{
    "grammar_complexity": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "structure_variety": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "grammar_correctness": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words). Note: Higher score means FEWER errors.",
        "score": <1-10>,
        "confidence": <0-10>
    }}
}}"""


def get_pragmatic_evaluation_prompt(transcript_text: str, pid: str) -> str:
    """
    Get prompt for pragmatic feature evaluation.
    
    Args:
        transcript_text: Patient's transcript text
        pid: Patient ID
        
    Returns:
        User prompt for pragmatic evaluation
    """
    return f"""Evaluate the PRAGMATIC features of the following transcript.

TRANSCRIPT:
{transcript_text}

Evaluate the following features:

1. **referential_clarity**: Review accuracy in using pronouns or references. Consider:
   - Clear vs. ambiguous pronoun usage (who is "she", "he", "it"?)
   - Proper introduction of referents before using pronouns
   - Deictic terms ("this", "that", "here", "there") with clear referents
   - Listener's ability to follow who/what is being discussed

2. **state_of_mind_language**: Consider the use of language expressing thoughts, feelings, or psychological states. Consider:
   - Mental state verbs (think, know, want, believe, feel)
   - Attribution of intentions or goals to characters
   - Interpretation beyond literal description
   - Theory of mind markers

3. **implausible_details**: Examine whether the description contains unnecessary or unlikely details. Consider:
   - Confabulated or fabricated elements not in the picture
   - Tangential information irrelevant to the scene
   - Bizarre or inappropriate statements
   - Excessive focus on minor details vs. main elements

Respond in the following JSON format:
{{
    "referential_clarity": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "state_of_mind_language": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "implausible_details": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words). Note: Higher score means FEWER implausible details.",
        "score": <1-10>,
        "confidence": <0-10>
    }}
}}"""


def get_semantic_coherence_evaluation_prompt(transcript_text: str, pid: str) -> str:
    """
    Get prompt for semantic coherence feature evaluation.
    
    Args:
        transcript_text: Patient's transcript text
        pid: Patient ID
        
    Returns:
        User prompt for semantic coherence evaluation
    """
    return f"""Evaluate the SEMANTIC COHERENCE and COHESION features of the following transcript.

TRANSCRIPT:
{transcript_text}

Evaluate the following features:

1. **topic_management**: Evaluate the ability to stay on topic and maintain relevancy. Consider:
   - Focus on the picture description task
   - Digressions or tangential topics
   - Return to main topic after digressions
   - Overall task adherence

2. **logical_organization**: Assess the structuring of ideas and their smooth flow. Consider:
   - Logical sequence of description (spatial, thematic, etc.)
   - Narrative structure
   - Transitions between ideas
   - Overall coherence of the description

3. **cohesion**: Evaluate the use of cohesive devices. Consider:
   - Conjunctions (and, but, because, so)
   - Pronouns linking to previous mentions
   - Lexical cohesion (synonyms, repetition for effect)
   - Temporal and spatial markers

4. **cause_and_effect**: Evaluate ability to articulate causal relationships. Consider:
   - Explaining WHY events are happening (water overflowing because sink is running)
   - Connecting actions to consequences (boy might fall because stool is tipping)
   - Temporal and causal reasoning

5. **repetition**: Assess whether narrative avoids unnecessary repetition. Consider:
   - Perseverative repetition (saying the same thing multiple times)
   - Productive vs. unproductive repetition
   - Note: Higher score means LESS problematic repetition

6. **information_prioritization**: Examine ordering of information by importance. Consider:
   - Main action elements mentioned before background details
   - Key story elements (cookie theft, potential fall) given prominence
   - Balance between salient and peripheral information

Respond in the following JSON format:
{{
    "topic_management": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "logical_organization": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "cohesion": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "cause_and_effect": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "repetition": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }},
    "information_prioritization": {{
        "summary": "Your analysis with specific examples from the transcript (max 300 words)",
        "score": <1-10>,
        "confidence": <0-10>
    }}
}}"""


def get_combined_evaluation_prompt(transcript_text: str, pid: str) -> str:
    """
    Get a combined prompt for all LLM-rated features in a single call.
    This is more efficient but may be less reliable for complex transcripts.
    
    Args:
        transcript_text: Patient's transcript text
        pid: Patient ID
        
    Returns:
        Combined user prompt
    """
    return f"""You are evaluating the linguistic features of a transcript from patient describing the Cookie Theft picture for dementia screening.

TRANSCRIPT:
{transcript_text}

Evaluate ALL of the following linguistic features. For each feature, provide a summary (max 300 words), a score (1-5), and a confidence (0-10).

Scoring guide:
- 1 = Severely impaired
- 2 = Moderately impaired  
- 3 = Mildly impaired / Average
- 4 = Largely intact
- 5 = Fully intact / Excellent

**A. LEXICAL FEATURES:**
- vocabulary_range: Variety of words used
- lexical_accuracy: Correctness of word choice
- specificity: Use of specific vs. vague terms
- advanced_vocabulary: Use of sophisticated vocabulary

**B. SYNTACTIC FEATURES:**
- grammar_complexity: Complex sentence structures
- structure_variety: Diversity of grammatical forms
- grammar_errors: Frequency of grammatical mistakes (higher score = fewer errors)

**C. PRAGMATIC FEATURES:**
- referential_clarity: Clear pronoun/reference usage
- state_of_mind_language: Expression of mental states
- implausible_details: Presence of fabricated content (higher score = fewer implausible details)

**D. SEMANTIC COHERENCE:**
- topic_management: Staying on topic
- logical_organization: Logical flow of ideas
- cohesion: Use of cohesive devices
- cause_and_effect: Articulating causal relationships
- repetition: Avoiding unnecessary repetition (higher score = less problematic repetition)
- information_prioritization: Ordering by importance

Respond in this JSON format:
{{
    "vocabulary_range": {{"summary": "...", "score": X, "confidence": X.X}},
    "lexical_accuracy": {{"summary": "...", "score": X, "confidence": X.X}},
    "specificity": {{"summary": "...", "score": X, "confidence": X.X}},
    "advanced_vocabulary": {{"summary": "...", "score": X, "confidence": X.X}},
    "grammar_complexity": {{"summary": "...", "score": X, "confidence": X.X}},
    "structure_variety": {{"summary": "...", "score": X, "confidence": X.X}},
    "grammar_errors": {{"summary": "...", "score": X, "confidence": X.X}},
    "referential_clarity": {{"summary": "...", "score": X, "confidence": X.X}},
    "state_of_mind_language": {{"summary": "...", "score": X, "confidence": X.X}},
    "implausible_details": {{"summary": "...", "score": X, "confidence": X.X}},
    "topic_management": {{"summary": "...", "score": X, "confidence": X.X}},
    "logical_organization": {{"summary": "...", "score": X, "confidence": X.X}},
    "cohesion": {{"summary": "...", "score": X, "confidence": X.X}},
    "cause_and_effect": {{"summary": "...", "score": X, "confidence": X.X}},
    "repetition": {{"summary": "...", "score": X, "confidence": X.X}},
    "information_prioritization": {{"summary": "...", "score": X, "confidence": X.X}}
}}"""
