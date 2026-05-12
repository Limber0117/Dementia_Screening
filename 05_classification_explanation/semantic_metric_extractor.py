"""
Semantic Feature Evaluator for Dementia Screening.

This module orchestrates the evaluation of linguistic features
from patient transcripts using both local computation and LLM analysis.

This module is designed to work alongside the existing dementia evaluation
system, sharing config.py and llm_providers.py but using its own data models
defined in semantic_models.py.
"""

import os,sys,json,csv
import time
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from data_loader import DataLoader
import random

load_dotenv()


# Import semantic feature specific models (separate from main models.py)
from models import (
    FeatureScore,
    TemporalFeatures,
    LexicalFeatures, 
    SyntacticFeatures,
    PragmaticFeatures,
    SemanticFeatures,
    SemanticEvaluationResult,
    ParsedTranscript,
    PatientEvaluationInput,
    EvaluationResult
)

from transcript_parser import TranscriptParser, load_transcript
from temporal_calculator import TemporalFeatureCalculator
from lexical_calculator import LexicalCalculator
from semantic_prompts import (
    get_semantic_system_prompt,
    get_lexical_evaluation_prompt,
    get_syntactic_evaluation_prompt,
    get_pragmatic_evaluation_prompt,
    get_semantic_coherence_evaluation_prompt,
    get_combined_evaluation_prompt,
    get_semantic_evaluation_system_prompt,
    get_semantic_evaluation_user_prompt,
)

# Import from existing codebase (shared with main evaluation system)
from config import Config, load_config, get_active_llm_config
from llm_providers import get_provider, LLMProvider

logger = logging.getLogger(__name__)
from semantic_clinical_profile import PopulationStatsBuilder,SemanticClinicalProfile,format_semantic_for_llm, rule_based_semantic_diagnosis



result_foder = os.getenv("RESULTS_DIR", "datasets/results")
is_debug = os.getenv("DEBUG", "").lower() == "true"
is_reasoning = os.getenv("STOPREASOING", "false").lower() != "true"
is_discrete = os.getenv("DISCRETE", "false").lower() == "true"
normallisation_dataset_path = os.getenv("NORMALISATION_DATASET_PATH", "datasets/output/acoustic_features/merged_data.csv")
use_prediction = os.getenv("USE_PREPREDICTION", "False").lower() == "true"
semantic_feature_folder = os.getenv("SEMANTIC_FEATURE_FOLDER","datasets/output/semantic_features/")
groundtruth_csv = os.getenv("GROUNDTRUTH_CSV","datasets/groundtruth/groundtruth.csv")

class SemanticFeatureEvaluator:
    """
    Main evaluator class for semantic feature analysis.
    
    Combines local computation (temporal, lexical metrics) with
    LLM-based evaluation (syntactic, pragmatic, semantic coherence).
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        env_path: Optional[str] = None,
        use_combined_prompt: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            config: Optional pre-loaded configuration
            env_path: Optional path to .env file
            use_combined_prompt: If True, use single LLM call for all features.
                               If False, use separate calls per category.
        """
        self.config = config or load_config(env_path)
        self.provider: Optional[LLMProvider] = None
        self.results: List[SemanticEvaluationResult] = []
        
        self.data_loader = DataLoader(self.config.dataset)
        
        # Local calculators
        self.transcript_parser = TranscriptParser()
        self.temporal_calculator = TemporalFeatureCalculator()
        self.lexical_calculator = LexicalCalculator()
        
        # Settings
        self.use_combined_prompt = use_combined_prompt
        self.transcripts_dir = self.config.dataset.transcripts_dir
        self.results_dir = self.config.dataset.results_dir

    
    def setup_provider(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        """
        Set up the LLM provider.
        
        Args:
            provider_name: Override the configured provider
            model: Override the configured model
        """
        if provider_name:
            self.config.active_provider = provider_name
        if model:
            self.config.active_model = model
        
        provider_name, llm_config = get_active_llm_config(self.config)
        
        logger.info(f"Setting up provider: {provider_name} with model: {llm_config.model}")
        
        self.provider = get_provider(
            provider_name,
            llm_config,
            self.config.processing
        )
    
    def get_patient_ids(self) -> List[str]:
        """
        Get all patient IDs from transcripts directory.
        
        Returns:
            List of patient ID strings
        """
        patient_ids = []
        
        if os.path.isdir(self.transcripts_dir):
            for filename in sorted(os.listdir(self.transcripts_dir)):
                if filename.endswith('.txt'):
                    pid = filename[:-4]
                    patient_ids.append(pid)
        
        return patient_ids
    
    def load_transcript(self, pid: str) -> Optional[ParsedTranscript]:
        """
        Load and parse a transcript for a patient.
        
        Args:
            pid: Patient ID
            
        Returns:
            ParsedTranscript or None if not found
        """
        filepath = os.path.join(self.transcripts_dir, f"{pid}.txt")
        
        if not os.path.exists(filepath):
            logger.warning(f"Transcript not found: {filepath}")
            return None
        
        return load_transcript(filepath, pid)
    
    def clean_json_text(self, text: str) -> str:
        """Clean JSON text from LLM response."""
        # Remove ASCII control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # Remove markdown fences
        text = text.replace("```json", "").replace("```", "")
        return text.strip()
    
    def extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text."""
        stack = 0
        start = None
        
        for i, ch in enumerate(text):
            if ch == '{':
                if stack == 0:
                    start = i
                stack += 1
            elif ch == '}':
                stack -= 1
                if stack == 0 and start is not None:
                    return text[start:i+1]
        
        return None
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into feature scores.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Dictionary of feature scores
        """
        cleaned = self.clean_json_text(response_text)
        json_str = self.extract_json(cleaned)
        
        try:
            if json_str:
                data = json.loads(json_str)
            else:
                data = json.loads(cleaned)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return {}
    
    def create_feature_score(
        self,
        feature_name: str,
        data: Dict[str, Any]
    ) -> FeatureScore:
        """
        Create a FeatureScore from parsed data.
        
        Args:
            feature_name: Name of the feature
            data: Dictionary with summary, score, confidence
            
        Returns:
            FeatureScore object
        """
        return FeatureScore(
            feature_name=feature_name,
            score=float(data.get("score", 0)),
            confidence=float(data.get("confidence", 0)),
            summary=str(data.get("summary", ""))[:1000],  # Limit summary length
            is_computed=False
        )
    
    def extract_with_llm(
        self,
        transcript: ParsedTranscript
    ) -> Dict[str, FeatureScore]:
        """
        Evaluate features using LLM.
        
        Args:
            transcript: Parsed transcript
            
        Returns:
            Dictionary of feature scores
        """
        if self.provider is None:
            self.setup_provider()
        
        system_prompt = get_semantic_system_prompt()
        patient_text = transcript.patient_text
        pid = transcript.pid
        
        feature_scores = {}
        
        if self.use_combined_prompt:
            # Single LLM call for all features
            user_prompt = get_combined_evaluation_prompt(patient_text, pid)
            
            try:
                response = self.provider._call_api(system_prompt, user_prompt, pid)
                data = self.parse_llm_response(response)
                
                for feature_name, feature_data in data.items():
                    if isinstance(feature_data, dict):
                        feature_scores[feature_name] = self.create_feature_score(
                            feature_name, feature_data
                        )
            except Exception as e:
                logger.error(f"LLM evaluation failed for {pid}: {e}")
        
        else:
            # Separate LLM calls per category
            prompts = [
                ("lexical", get_lexical_evaluation_prompt(patient_text, pid)),
                ("syntactic", get_syntactic_evaluation_prompt(patient_text, pid)),
                ("pragmatic", get_pragmatic_evaluation_prompt(patient_text, pid)),
                ("semantic", get_semantic_coherence_evaluation_prompt(patient_text, pid)),
            ]
            
            for category, user_prompt in prompts:
                try:
                    response = self.provider._call_api(system_prompt, user_prompt, pid)
                    data = self.parse_llm_response(response)
                    
                    for feature_name, feature_data in data.items():
                        if isinstance(feature_data, dict):
                            feature_scores[feature_name] = self.create_feature_score(
                                feature_name, feature_data
                            )
                    
                    # Rate limiting
                    if is_reasoning:
                        time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"LLM {category} evaluation failed for {pid}: {e}")
        
        return feature_scores
    
    def extract_patient(self, pid: str) -> SemanticEvaluationResult:
        """
        Evaluate all features for a single patient.
        
        Args:
            pid: Patient ID
            
        Returns:
            SemanticEvaluationResult
        """
        start_time = time.time()
        result = SemanticEvaluationResult(pid=pid)
        
        # Load transcript
        transcript = self.load_transcript(pid)
        if transcript is None or not transcript.patient_utterances:
            result.error = "Transcript not found or empty"
            return result
        
        # Compute temporal features (local)
        result.temporal = self.temporal_calculator.calculate(transcript)
        
        # Compute lexical features (local)
        result.lexical = self.lexical_calculator.calculate(transcript)
        
        # Evaluate with LLM
        llm_scores = self.extract_with_llm(transcript)
        
        # Map LLM scores to feature categories
        # Lexical (LLM-rated)
        result.lexical.vocabulary_range = llm_scores.get("vocabulary_range")
        result.lexical.lexical_accuracy = llm_scores.get("lexical_accuracy")
        result.lexical.specificity = llm_scores.get("specificity")
        result.lexical.advanced_vocabulary = llm_scores.get("advanced_vocabulary")
        
        # Syntactic
        result.syntactic = SyntacticFeatures(
            grammar_complexity=llm_scores.get("grammar_complexity"),
            structure_variety=llm_scores.get("structure_variety"),
            grammar_correctness=llm_scores.get("grammar_correctness")
        )
        
        # Pragmatic
        result.pragmatic = PragmaticFeatures(
            referential_clarity=llm_scores.get("referential_clarity"),
            state_of_mind_language=llm_scores.get("state_of_mind_language"),
            implausible_details=llm_scores.get("implausible_details")
        )
        
        # Semantic
        result.semantic = SemanticFeatures(
            topic_management=llm_scores.get("topic_management"),
            logical_organization=llm_scores.get("logical_organization"),
            cohesion=llm_scores.get("cohesion"),
            cause_and_effect=llm_scores.get("cause_and_effect"),
            repetition=llm_scores.get("repetition"),
            information_prioritization=llm_scores.get("information_prioritization")
        )
        
        # Metadata
        result.processing_time_seconds = time.time() - start_time
        if self.provider:
            result.model_name = self.provider.config.model
            result.provider = self.provider.provider_name
        
        return result


     
    def extract_all(
        self,
        variant:str="transcript",
        patient_ids: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> List[SemanticEvaluationResult]:
        """
        Evaluate all patients.
        
        Args:
            patient_ids: Optional list of specific patient IDs
            show_progress: Whether to show progress bar
            
        Returns:
            List of evaluation results
        """
        if self.provider is None:
            self.setup_provider()
        
        if variant.lower()!="transcript" and variant.lower()!="standard":
            print(f"CAUTION: This module accept only transcript-based evaluation, but you used {variant}.")
            sys.exit() 

        if patient_ids is None:
            patient_ids = self.get_patient_ids()
        
        if show_progress:
            patient_ids = tqdm(patient_ids, desc="Evaluating semantic features")
        
        for pid in patient_ids:
            result = self.extract_patient(pid)
            self.results.append(result)
            
            # Rate limiting
            if is_reasoning:
                time.sleep(2)
        
        return self.results


    def merge_groundtruth_semantics(self, groundtruthCSV: str, results: List[SemanticEvaluationResult]) -> pd.DataFrame:
        """
        Merge groundtruth data with semantic evaluation results.
        
        Args:
            groundtruthCSV: Path to the groundtruth CSV/TSV file
            results: List of SemanticEvaluationResult objects from extract_all()
            
        Returns:
            pd.DataFrame: Merged dataframe containing both groundtruth and semantic features
        """
        # 1. Validate and read the groundtruth file
        
        output_path = os.path.join(semantic_feature_folder,"merged_semantic_data.csv")

        if os.path.exists(output_path):
            merged_df=pd.read_csv(Path(output_path), sep=',')
            print(f"Read existing merged feature file...")
        else:        
            print(f"Regenerate the file by merging the ground-truth and raw features...")
            groundtruthPath = Path(groundtruthCSV)
            if not groundtruthPath.exists():  
                print("\nError: The groundtruth file is missing.") 
                sys.exit(1)
            
            # Read groundtruth (handle both CSV and TSV formats)
            groundtruth = pd.read_csv(groundtruthPath, sep=',')
            print(f"Groundtruth shape: {groundtruth.shape}. First 10 columns: {list(groundtruth.columns[:10])}")
            
            # 2. Convert results list to DataFrame
            if not results:
                print("\nWarning: No results to merge.")
                return groundtruth
            
            # Use the to_flat_dict() method that already exists in SemanticEvaluationResult
            results_data = [result.to_flat_dict() for result in results]
            results_df = pd.DataFrame(results_data)
            print(f"Feature results shape: {results_df.shape}. The first ten columns: {list(results_df.columns[:10])}")
            
            # 3. Merge groundtruth and results by PID
            # Identify the PID column in groundtruth (common names: 'pid', 'PID', 'patient_id', 'ID')
            pid_column_groundtruth = None
            possible_pid_columns = ['pid', 'PID', 'patient_id', 'PatientID', 'ID', 'id']
            
            for col in possible_pid_columns:
                if col in groundtruth.columns:
                    pid_column_groundtruth = col
                    break
            
            if pid_column_groundtruth is None:
                print(f"\nError: Could not find PID column in groundtruth. Available columns: {list(groundtruth.columns)}")
                sys.exit(1)
            
            # Ensure PID columns are the same type (string) for proper merging
            groundtruth[pid_column_groundtruth] = groundtruth[pid_column_groundtruth].astype(str)
            results_df['pid'] = results_df['pid'].astype(str)
            
            # Perform the merge
            merged_df = pd.merge(
                groundtruth,
                results_df,
                left_on=pid_column_groundtruth,
                right_on='pid',
                how='inner'  # Only keep patients that exist in both datasets
            )
            
            # Remove duplicate PID column if they have different names
            if pid_column_groundtruth != 'pid' and 'pid' in merged_df.columns:
                merged_df = merged_df.drop(columns=['pid'])
            
            # Drop redundant participant_id column (keep PID)
            columns_to_drop = ['participant_id', 'sample_rate', 'filename', 'mediaName', 'MMSE', 'topic', 'doctor', 'duration']
            existing_columns_to_drop = [col for col in columns_to_drop if col in merged_df.columns]
            if existing_columns_to_drop:
                merged_df = merged_df.drop(columns=existing_columns_to_drop)        
            
            print(f"Merged shape: {merged_df.shape}")
            #print(f"Patients in groundtruth: {len(groundtruth)}")
            #print(f"Patients in results: {len(results_df)}")
            print(f"Patients after merge: {len(merged_df)}. The first ten columns: {list(merged_df.columns[:10])}")

            # Save to CSV
            # Create directory if it doesn't exist
            output_path_real = Path(output_path)
            output_path_real.parent.mkdir(parents=True, exist_ok=True)            
            merged_df.to_csv(output_path_real, index=False)
            print(f"\nSaved the merged semantic features to: {output_path_real}")

        return merged_df
    
    def get_record_as_dict(self,df, pid):
        """
        Retrieve a record by pid and convert values to a formatted dictionary.        
        Args:
            df: pandas DataFrame with 'pid' as the first column
            pid: the primary key value to look up
        
        Returns:
            dict: dictionary with column names as keys and formatted float values as strings
        """
        row = df[df['PID'] == pid]        
        if row.empty:
            return None
        
        summary = {}
        exclude_cols = {'PID', 'language', 'age', 'gender', 'diagnosis'}
        for col in df.columns:
            if col not in exclude_cols:
                value = row[col].values[0]
                summary[col] = value
        gender = row['gender'].values[0]
        age = row['age'].values[0]

        if is_debug:
            print(f" The current selected semantic features are: {summary} and gender is {gender}, and age is {age}.")
        
        return summary,gender,age
    

    def predict_all_merged(self,
        variant:str="transcript",
        mergedFeatureFile:str=None,
        show_progress: bool = True):
        """
        This function takes the LLM summary and temporal features as input and use LLM to predict all the patients' congnitive status.
        key steps:
        1. read merged raw 'results' from the given .csv file.
           |- 2. use the merged table to calculate the z scores
           |- 3. generate the prompt for each patient
           |- 4. send to LLM to make prediction
        5. extra the results and store to predictions[]
        return predictions['class', 'confidence', reasons_summary]

        """
        #initialisation
        if self.provider is None:
            self.setup_provider()
        
        self.results=[]

        # STEP 1: merge the groundtruth with current 'results' list.
        if not os.path.exists(mergedFeatureFile):
            print("Error! Cannot find the merged feature file ...")
            sys.exit(1)
            return None
        else:
            merged=pd.read_csv(mergedFeatureFile, sep=',')
            print(f"Read existing merged feature file...")
            PIDs = self.get_patient_ids_from_pd(merged)
            random.shuffle(PIDs)
            if is_debug:
                PIDs = PIDs[:5]

            if show_progress:
                PIDs = tqdm(PIDs, desc="Evaluating patients")

            for pid in PIDs:
                # STEP 2: retrieve semantic features to calculate the z scores
                semantic_feature, gender, age = self.get_record_as_dict(merged,pid)

                #evaluate one patient
                predict = self.predict_one_patient(variant="transcript",patient_features=semantic_feature,pid=pid, gender=gender, age=age)

                if is_reasoning:
                    self.results.append(predict)
                
                # Log progress
                if not show_progress:
                    logger.info(
                        f"Evaluated {pid}: "
                        f"{predict.prediction.value}"
                    )
                if is_reasoning:
                    time.sleep(5)  #no more than 60/3 requests per minute

            return self.results
    

    def predict_all(self,
        variant:str="transcript",
        rawFeature:List[EvaluationResult]=None,
        show_progress: bool = True):
        """
        This function takes the LLM summary and temporal features as input and use LLM to predict all the patients' congnitive status.
        key steps:
        1. merge the groundtruth with current 'results' list.
           |- 2. use the merged table to calculate the z scores
           |- 3. generate the prompt for each patient
           |- 4. send to LLM to make prediction
        5. extra the results and store to predictions[]
        return predictions['class', 'confidence', reasons_summary]

        """
        #initialisation
        if self.provider is None:
            self.setup_provider()
        
        self.results=[]

        # STEP 1: merge the groundtruth with current 'results' list.
        merged = self.merge_groundtruth_semantics(groundtruth_csv, rawFeature)
        PIDs = self.get_patient_ids_from_pd(merged)
        random.shuffle(PIDs)
        if is_debug:
            PIDs = PIDs[:5]

        if show_progress:
            PIDs = tqdm(PIDs, desc="Evaluating patients")

        for pid in PIDs:
            # STEP 2: retrieve semantic features to calculate the z scores
            semantic_feature, gender, age = self.get_record_as_dict(merged,pid)

            #evaluate one patient
            predict = self.predict_one_patient(variant="transcript",patient_features=semantic_feature,pid=pid, gender=gender, age=age)

            if is_reasoning:
                self.results.append(predict)
            
            # Log progress
            if not show_progress:
                logger.info(
                    f"Evaluated {pid}: "
                    f"{predict.prediction.value}"
                )
            if is_reasoning:
                time.sleep(5)  #no more than 60/3 requests per minute

        return self.results
    
    def save_to_txt(self, content, filename, folder):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, "w") as file:
            file.write(str(content))

    def predict_one_patient(
            self, 
            variant: str,
            patient_features: dict,
            pid:str,
            gender:str,
            age:int
        ) -> EvaluationResult:
            """
            Evaluate a single patient.
            
            Args:
                patient_data: Complete patient data
                
            Returns:
                EvaluationResult
            """
            if self.provider is None:
                self.setup_provider()            
                       
            #try:
            if True:
                # Get prompts
                system_prompt = get_semantic_evaluation_system_prompt()
                
                # Format patient data for the prompt
                patient_demo = f"This patient is {gender} and is {age} years old."

                # generate prompt and role-based prediction
                pre_dignosis, prompt = self.generate_new_semantic_prompt(patient_features=patient_features,pid=pid)

                #the function create_evaluation_prompt now takes variant and patient_data
                #accepable variants are "standard", "acoustic", "transcript". The default is "standard", which means both acoustic and transcript data are used for prediction.
                if is_discrete and (variant =="transcript" or variant =="standard" or variant==""):
                    if not use_prediction:
                        pre_dignosis = " "                                          
                    user_prompt = get_semantic_evaluation_user_prompt(patient_data = patient_demo, patient_feature=prompt,pre_diagnosis = pre_dignosis)
                else:
                    print("This module only takes discreatised semantic features as input.")
                    sys.exit(0)

                # Evaluate
                print(" + Evaluating patient PID:", pid)

                if is_debug:
                    print (user_prompt)

                if is_debug or (not is_reasoning):
                    all_prompt=f"System prompt: {system_prompt} \n\n Content{user_prompt}"                
                    txt_filename = pid + ".txt"
                    self.save_to_txt(all_prompt,txt_filename, result_foder)
                    print(f" - DEBUG: patient {pid} prompt saved. ")                

                if is_reasoning:
                    result = self.provider.evaluate(pid, system_prompt, user_prompt)
                else:
                    return None
                
                return result
                
            #except Exception as e:
            '''
                logger.error(f"Error evaluating patient {pid}: {e}")
                from models import DiagnosisResult
                return EvaluationResult(
                    pid=pid,
                    prediction=DiagnosisResult.UNKNOWN,
                    explanation="",
                    model_name=self.provider.config.model if self.provider else "",
                    provider=self.provider.provider_name if self.provider else "",
                    error=str(e),
                    variant=variant
                )
            '''

    def generate_new_semantic_prompt(self, patient_features:dict, gender:str="male", pid:str="")->Tuple[str, str]:
        
        # Get normative statistics
        healthy_stats, cohort_stats = self.feature_normalization()

        # Analyze patient to get clinical profile
        analyser = SemanticClinicalProfile(
            healthy_stats=healthy_stats,
            cohort_stats=cohort_stats
        )
        profile = analyser.analyse_patient(patient_features, pid)
        if is_debug:
            print(f" - Successfully analysed the single patient... profile size {len(profile)} ... feature size {len(patient_features)}")
        prompt = format_semantic_for_llm(profile, patient_features)
        if is_debug:
            print(" - Successfully generated patient's plain text features ...")
            print(f"\nThe generated z-score patient prompt is: \n {prompt}\n")
        pre_dignosis = rule_based_semantic_diagnosis(profile)
        if is_debug:
            print(" - Successfully generated a rule-based indicator ...")

        if  True or is_debug:
            #extract the discrete z score for further internal investigation
            records = []
            row = {
                "participant_id": profile["PID"]
            }
            # add all group z scores
            for group, z in profile["group_z"].items():
                row[group] = z
            records.append(row)
            fieldnames = ["participant_id"] + list(profile["group_z"].keys())
            output_csv = os.path.join(result_foder, "discrete_semantic_z_scores.csv")
            file_exists = os.path.isfile(output_csv)
            with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for record in records:
                    writer.writerow(record)

        return pre_dignosis, prompt
    
    def save_feature_results(
        self,
        results: Optional[List[SemanticEvaluationResult]] = None,
        output_path: Optional[str] = None,
        type:str="features"
    ) -> str:
        """
        Save evaluation results to CSV.
        
        Args:
            results: Results to save (uses self.results if not provided)
            output_path: Custom output path
            
        Returns:
            Path to saved file
        """
        results = results or self.results
        
        if not results:
            logger.warning("No results to save")
            return ""
        
        # Ensure output directory exists
        
        dir = semantic_feature_folder  
        os.makedirs(dir, exist_ok=True)
        
        # Generate filename
        if output_path is None:          
            filename = f"semantic_feature_result.csv"
            output_path = os.path.join(dir, filename)
        
        # Convert to DataFrame
        rows = [r.to_flat_dict() for r in results]
        df = pd.DataFrame(rows)
        
        # Save
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        return output_path
    
    def save_results(
        self,
        results: Optional[List[EvaluationResult]] = None,
        output_path: Optional[str] = None,
        type:str="features"
    ) -> str:
        """
        Save evaluation results to CSV.
        
        Args:
            results: Results to save (uses self.results if not provided)
            output_path: Custom output path
            
        Returns:
            Path to saved file
        """
        results = results or self.results
        
        if not results:
            logger.warning("No results to save")
            return ""
        
        # Ensure output directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Generate filename
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            provider = results[0].provider if results else "unknown"
            model = results[0].model_name if results else "unknown"
            model_clean = model.replace("/", "-").replace(":", "-")
            
            filename = f"semantic_{type}_{provider}_{model_clean}_{timestamp}.csv"
            output_path = os.path.join(self.results_dir, filename)
        
        # Convert to DataFrame
        rows = [r.to_flat_dict() for r in results]
        df = pd.DataFrame(rows)

        # Add ground truth if available
        ground_truths = []
        for result in results:
            gt = self.data_loader.get_groundtruth_diagnosis(result.pid)
            ground_truths.append(gt)
        df["GroundTruth"] = ground_truths
        
        # Save
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        return output_path
    
    def get_summary_statistics(
        self,
        results: Optional[List[SemanticEvaluationResult]] = None
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics across all results.
        
        Args:
            results: Results to summarize
            
        Returns:
            Dictionary of summary statistics
        """
        results = results or self.results
        
        if not results:
            return {}
        
        # Convert to DataFrame for easy statistics
        rows = [r.to_flat_dict() for r in results]
        df = pd.DataFrame(rows)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        summary = {
            "total_patients": len(results),
            "feature_means": {},
            "feature_stds": {}
        }
        
        for col in numeric_cols:
            if col not in ['pid', 'processing_time_seconds']:
                summary["feature_means"][col] = df[col].mean()
                summary["feature_stds"][col] = df[col].std()
        
        return summary


    def feature_normalization(self, dataset_path: str=None)->Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, float]]
        ]:
        # Load dataset for normative statistics
        dataset_path = os.path.join(semantic_feature_folder,"merged_semantic_data.csv")
        if os.path.exists(dataset_path):
            print(f"Merged feature file found, start to calculate the z values.")
            rows_all = []
            rows_healthy = []

            with open(dataset_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_all.append(row)
                    if row.get("diagnosis").lower() == "control":   # or however healthy controls are labelled
                        rows_healthy.append(row)

            healthy_stats = PopulationStatsBuilder.build(rows_healthy)
            cohort_stats = PopulationStatsBuilder.build(rows_all)
            
            return healthy_stats, cohort_stats
        else:
            print(f"ERROR: Merged feature file does not exist.")
        return None, None



    def calculate_metrics(
            self, 
            results: Optional[List[EvaluationResult]] = None
        ) -> dict:
        """
        Calculate evaluation metrics comparing predictions to ground truth.
        
        Metrics include:
        - Accuracy: Overall correct predictions / total
        - Precision: TP / (TP + FP)
        - Sensitivity/Recall: TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        - AUC: Area Under the ROC Curve
        
        Args:
            results: Results to evaluate (uses self.results if not provided)
            
        Returns:
            Dictionary with comprehensive metrics
        """
        results = results or self.results
        
        if not results:
            return {}
        
        from models import DiagnosisResult
        
        # Collect predictions and ground truths
        y_true = []
        y_pred = []
        y_scores = []  # For AUC calculation (if confidence scores available)
        
        for result in results:
            gt = self.data_loader.get_groundtruth_diagnosis(result.pid)
            if not gt:
                continue
            
            gt_normalized = DiagnosisResult.from_string(gt)
            
            if is_debug:
                print(f"Patient ID: {result.pid}, Ground Truth: {gt_normalized.value}, Prediction: {result.prediction.value}")
            
            if gt_normalized.value == "Unknown" or result.prediction.value == "Unknown":
                continue
            
            y_true.append(gt_normalized.value)
            y_pred.append(result.prediction.value)
            
            # Collect confidence scores if available (for proper AUC calculation)
            if hasattr(result, 'confidence') and result.confidence is not None:
                y_scores.append(result.confidence)
        
        if not y_true:
            return {}
        
        print(f"The total number of evaluated patients is: {len(y_true)}")
        
        # Convert to binary labels (1 = Impairment, 0 = Control)
        y_true_binary = [1 if label == "Impairment" else 0 for label in y_true]
        y_pred_binary = [1 if label == "Impairment" else 0 for label in y_pred]
        
        # Calculate confusion matrix values
        # Positive class = Impairment, Negative class = Control
        tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
        
        total = len(y_true)
        
        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Calculate AUC
        auc = None
        if y_scores and len(y_scores) == len(y_true_binary):
            # Use sklearn for proper AUC with probability scores
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true_binary, y_scores)
            except (ImportError, ValueError) as e:
                print(f"Could not calculate AUC with scores: {e}")
                auc = None
        
        if auc is None:
            # Fallback: estimate AUC from sensitivity and specificity
            # For hard predictions, AUC = (sensitivity + specificity) / 2
            auc = (sensitivity + specificity) / 2
        
        # Confusion matrix as nested dict for readability
        confusion_matrix = {
            "Control": {"Control": tn, "Impairment": fp},
            "Impairment": {"Control": fn, "Impairment": tp}
        }
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,  # Same as recall
            "recall": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "auc": auc,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "total": total,
            "support_impairment": tp + fn,
            "support_control": tn + fp,
            "confusion_matrix": confusion_matrix,
        }



def run_semantic_evaluation(
    env_path: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    patient_ids: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    use_combined_prompt: bool = True
) -> str:
    """
    Convenience function to run semantic feature evaluation.
    
    Args:
        env_path: Path to .env file
        provider: LLM provider to use
        model: Model to use
        patient_ids: Specific patients to evaluate
        output_path: Custom output path
        use_combined_prompt: Use single LLM call for all features
        
    Returns:
        Path to results file
    """
    evaluator = SemanticFeatureEvaluator(
        env_path=env_path,
        use_combined_prompt=use_combined_prompt
    )
    
    if provider or model:
        evaluator.setup_provider(provider, model)
    
    evaluator.extract_all("transcript",patient_ids)
    
    return evaluator.save_results(output_path=output_path)

if __name__ == "__main__":

    patient_ids = ['a-00090825-0','a-00090752-0']  # Evaluate all patients

    print(f"Total patient IDs to evaluate: {len(patient_ids)}")

    #patient_ids = ['a-00090323-0']  # Evaluate all patients
    output_path = "datasets/output/semantic_feature_results_new.csv"


    # The following is to execute the semantic evaluation with specified LLM models
    run_semantic_evaluation(
        env_path=".env",
        provider="google",
        model="gemini-2.5-pro",
        patient_ids=patient_ids,
        output_path=output_path,
        use_combined_prompt=False
    )