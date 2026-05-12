"""
LLM Provider implementations for Impairment Evaluation System.

This module contains classes for interacting with different LLM APIs.
"""

import json
import time
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

from config import LLMConfig, ProcessingConfig
from models import EvaluationResult, DiagnosisResult
import os
from dotenv import load_dotenv
from pathlib import Path


result_foder = os.getenv("RESULTS_DIR", "datasets/results")
is_debug = os.getenv("DEBUG", "").lower() == "true"

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self, 
        config: LLMConfig, 
        processing_config: ProcessingConfig
    ):
        self.config = config
        self.processing_config = processing_config
        self.provider_name = self.__class__.__name__.replace("Provider", "")
    
    @abstractmethod
    def _call_api(
        self, 
        system_prompt: str, 
        user_prompt: str,
        patient_id: str
    ) -> str:
        """Make the actual API call. Must be implemented by subclasses."""
        pass
    
    def save_to_txt(self, content, filename, folder):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding='utf-8') as file:
            file.write(str(content))

    def evaluate(
        self,
        pid: str,
        system_prompt: str,
        user_prompt: str
    ) -> EvaluationResult:
        """
        Evaluate a patient using the LLM.
        
        Args:
            pid: Patient ID
            system_prompt: System prompt for the LLM
            user_prompt: User prompt with patient data
            
        Returns:
            EvaluationResult with prediction and explanation
        """
        start_time = time.time()
        
        for attempt in range(self.processing_config.max_retries):
            try:
                response_text = self._call_api(system_prompt, user_prompt,pid)
                if is_debug:
                    print(f" - response_text is {response_text}")
                result = self._parse_response(pid, response_text)
                result.processing_time_seconds = time.time() - start_time
                result.model_name = self.config.model
                result.provider = self.provider_name
                return result
                
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for {pid}: {str(e)}"
                )
                if attempt < self.processing_config.max_retries - 1:
                    time.sleep(self.processing_config.retry_delay)
                else:
                    return EvaluationResult(
                        pid=pid,
                        prediction=DiagnosisResult.UNKNOWN,
                        explanation="",
                        model_name=self.config.model,
                        provider=self.provider_name,
                        processing_time_seconds=time.time() - start_time,
                        error=str(e)
                    )
    
    def clean_json_text(self, text: str) -> str:
        # Remove ASCII control characters except for standard whitespace
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Remove common markdown fences
        text = text.replace("```json", "").replace("```", "")
        
        return text.strip()

    def extract_json(self, text: str) -> str | None:
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

    def _parse_response(self, pid: str, response_text: str) -> EvaluationResult:
        """Parse the LLM response to extract prediction and explanation."""

        cleaned = self.clean_json_text(response_text)
        json_str = self.extract_json(cleaned)

        try:
            if json_str:
                data = json.loads(json_str)
            else:
                data = json.loads(cleaned)
            
            
            #prediction_str = data.get("prediction", "Unknown")
            prediction_str = max((k for k in data if k != "explanation"),
                                    key=lambda k: data[k]
                                )
            
            prediction = DiagnosisResult.from_string(prediction_str)
            print(f" - The current predicted status of PID {pid} is: {prediction}")
            explanation = data.get("explanation", "")
            #confidence 
            impairment_conf = data.get("Impairment",0)
            control_conf = data.get("Control",0)

            
            return EvaluationResult(
                pid=pid,
                prediction=prediction,
                explanation=explanation,
                impairment_conf = impairment_conf,
                control_conf  = control_conf
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract information from non-JSON response
            prediction = DiagnosisResult.UNKNOWN
            
            response_lower = response_text.lower()
            if "impairment" in response_lower or "alzheimer" in response_lower or "ad" in response_lower  or "mci" in response_lower or "dementia" in response_lower:
                prediction = DiagnosisResult.IMPAIRMENT
            elif "control" in response_lower or "normal" in response_lower or "healthy" in response_lower:
                prediction = DiagnosisResult.CONTROL
            
            return EvaluationResult(
                pid=pid,
                prediction=prediction,
                explanation=response_text[:500],
                error="JSON parsing failed, extracted from raw text"
            )


class OpenAIProvider(LLMProvider):
    """OpenAI (ChatGPT) provider implementation."""
    
    def __init__(
        self, 
        config: LLMConfig, 
        processing_config: ProcessingConfig
    ):
        super().__init__(config, processing_config)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key,
                timeout=processing_config.request_timeout
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def _call_api(self, system_prompt: str, user_prompt: str, patient_id:str) -> str:
        
        if "gpt-5" in self.config.model:
            response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            #max_tokens=self.config.max_tokens,
            max_completion_tokens = self.config.max_tokens,
            response_format={"type": "json_object"}
            #temperature=1 # e.g., GPT-5-mini model only supports 1. GPT-5.1 does not need it.
            )
        else:
            response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
             )


        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider implementation."""
    
    def __init__(
        self, 
        config: LLMConfig, 
        processing_config: ProcessingConfig
    ):
        super().__init__(config, processing_config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=config.api_key,
                timeout=processing_config.request_timeout
            )
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def _call_api(self, system_prompt: str, user_prompt: str, patient_id:str) -> str:
    
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text


class GoogleProvider(LLMProvider):
    """Google (Gemini) provider implementation."""
    
    def __init__(
        self, 
        config: LLMConfig, 
        processing_config: ProcessingConfig
    ):
        super().__init__(config, processing_config)
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            
            generation_config = genai.GenerationConfig(
                max_output_tokens=config.max_tokens,
                temperature=config.temperature
            )
            
            self.model = genai.GenerativeModel(
                model_name=config.model,
                generation_config=generation_config
            )
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
    
    def _call_api(self, system_prompt: str, user_prompt: str, patient_id:str) -> str:

        # Gemini combines system and user prompts
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        response = self.model.generate_content(full_prompt)
        return response.text

class QWenProvider(LLMProvider):
    """QWen provider implementation (uses OpenAI-compatible DashScope API)."""
    
    def __init__(
        self, 
        config: LLMConfig, 
        processing_config: ProcessingConfig
    ):
        super().__init__(config, processing_config)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=processing_config.request_timeout
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def _call_api(self, system_prompt: str, user_prompt: str, patient_id: str) -> str:

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content
    
class DeepSeekProvider(LLMProvider):
    """DeepSeek provider implementation (uses OpenAI-compatible API)."""
    
    def __init__(
        self, 
        config: LLMConfig, 
        processing_config: ProcessingConfig
    ):
        super().__init__(config, processing_config)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=processing_config.request_timeout,
                max_retries=0
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def _call_api(self, system_prompt: str, user_prompt: str, patient_id:str) -> str:

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Ollama (local LLaMA) provider implementation."""
    
    def __init__(
        self, 
        config: LLMConfig, 
        processing_config: ProcessingConfig
    ):
        super().__init__(config, processing_config)
        self.base_url = config.base_url or "http://localhost:11434"
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")
    
    def _call_api(self, system_prompt: str, user_prompt: str, patient_id:str) -> str:

        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        response = self.requests.post(
            url, 
            json=payload,
            timeout=self.processing_config.request_timeout
        )
        response.raise_for_status()
        
        return response.json()["message"]["content"]


def get_provider(
    provider_name: str,
    llm_config: LLMConfig,
    processing_config: ProcessingConfig
) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.
    
    Args:
        provider_name: Name of the provider (openai, anthropic, google, deepseek, ollama)
        llm_config: Configuration for the LLM
        processing_config: Processing configuration
        
    Returns:
        Instantiated LLM provider
        
    Raises:
        ValueError: If provider name is not recognized
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "deepseek": DeepSeekProvider,
        "ollama": OllamaProvider,
        "qwen": QWenProvider
    }
    
    provider_name = provider_name.lower()
    
    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Valid options: {list(providers.keys())}"
        )
    
    return providers[provider_name](llm_config, processing_config)
