"""
Configuration module for Dementia Evaluation System.

This module handles loading and validating configuration from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


@dataclass
class DatasetConfig:
    """Configuration for dataset paths."""
    groundtruth_csv: str = "datasets/groundtruth/groundtruth.csv"
    acoustic_features_csv: str = "datasets/output/acoustic_features/features.csv"
    transcripts_dir: str = "datasets/output/transcripts"
    results_dir: str = "datasets/results"
    merged_data_csv: str = "datasets/output/acoustic_features/merged_data.csv"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider."""
    api_key: Optional[str] = None
    model: str = ""
    max_tokens: int = 1000
    temperature: float = 0.1
    base_url: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Configuration for processing behavior."""
    max_retries: int = 3
    retry_delay: float = 10.0
    request_timeout: int = 150


@dataclass
class Config:
    """Main configuration container."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    active_provider: str = os.getenv("ACTIVE_LLM_PROVIDER")
    active_model: str = os.getenv("ACTIVE_LLM_MODEL")
    active_api_key: Optional[str] = None
    openai: LLMConfig = field(default_factory=LLMConfig)
    anthropic: LLMConfig = field(default_factory=LLMConfig)
    google: LLMConfig = field(default_factory=LLMConfig)
    deepseek: LLMConfig = field(default_factory=LLMConfig)
    qwen: LLMConfig = field(default_factory=LLMConfig)
    ollama: LLMConfig = field(default_factory=LLMConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    semantic_feature_folder: str=os.getenv("SEMANTIC_FEATURE_FOLDER")


def load_config(env_path: Optional[str] = None) -> Config:
    """
    Load configuration from environment variables.
    
    Args:
        env_path: Optional path to .env file. If None, looks in current directory.
        
    Returns:
        Config object with all settings loaded.
    """
    # Load .env file
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    config = Config()
    
    # Dataset configuration
    config.dataset.groundtruth_csv = os.getenv(
        "GROUNDTRUTH_CSV", config.dataset.groundtruth_csv
    )
    config.dataset.acoustic_features_csv = os.getenv(
        "ACOUSTIC_FEATURES_CSV", config.dataset.acoustic_features_csv
    )
    config.dataset.transcripts_dir = os.getenv(
        "TRANSCRIPTS_DIR", config.dataset.transcripts_dir
    )
    config.dataset.results_dir = os.getenv(
        "RESULTS_DIR", config.dataset.results_dir
    )
    
    # Active LLM configuration
    config.active_provider = os.getenv("ACTIVE_LLM_PROVIDER").lower()
    config.active_model = os.getenv("ACTIVE_LLM_MODEL")
    
    # OpenAI configuration
    config.openai.api_key = os.getenv("OPENAI_API_KEY")
    config.openai.model = os.getenv("OPENAI_MODEL", "gpt-4o")
    config.openai.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "8000"))
    config.openai.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # Anthropic configuration
    config.anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
    config.anthropic.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    config.anthropic.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8000"))
    config.anthropic.temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.1"))
    
    # Google configuration
    config.google.api_key = os.getenv("GOOGLE_API_KEY")
    config.google.model = os.getenv("GOOGLE_MODEL", "gemini-3-pro-preview")
    config.google.max_tokens = int(os.getenv("GOOGLE_MAX_TOKENS", "8000"))
    config.google.temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0.1"))
    
    # DeepSeek configuration
    config.deepseek.api_key = os.getenv("DEEPSEEK_API_KEY")
    config.deepseek.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    config.deepseek.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8000"))
    config.deepseek.temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1"))
    config.deepseek.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # QWen configuration
    config.qwen.api_key = os.getenv("QWEN_API_KEY")
    config.qwen.model = os.getenv("QWEN_MODEL", "qwen-plus")
    config.qwen.max_tokens = int(os.getenv("QWEN_MAX_TOKENS", "8000"))
    config.qwen.temperature = float(os.getenv("QWEN_TEMPERATURE", "0.1"))
    config.qwen.base_url = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    
    # Ollama configuration
    config.ollama.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    config.ollama.model = os.getenv("OLLAMA_MODEL", "llama3.2")
    config.ollama.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "8000"))
    config.ollama.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    
    # Processing configuration
    config.processing.max_retries = int(os.getenv("MAX_RETRIES", "1"))
    config.processing.retry_delay = float(os.getenv("RETRY_DELAY", "5"))
    config.processing.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))
    
    return config


def get_active_llm_config(config: Config) -> tuple[str, LLMConfig]:
    """
    Get the configuration for the currently active LLM.
    
    Args:
        config: Main configuration object.
        
    Returns:
        Tuple of (provider_name, LLMConfig).
        
    Raises:
        ValueError: If the active provider is not recognized.
    """
    provider = config.active_provider.lower()
    
    provider_map = {
        "openai": config.openai,
        "anthropic": config.anthropic,
        "google": config.google,
        "deepseek": config.deepseek,
        "ollama": config.ollama,
        "qwen": config.qwen
    }
    
    if provider not in provider_map:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Valid options: {list(provider_map.keys())}"
        )
    
    llm_config = provider_map[provider]
    
    # Override model with active model if specified
    if config.active_model:
        llm_config.model = config.active_model



    return provider, llm_config


def validate_config(config: Config) -> list[str]:
    """
    Validate the configuration and return any warnings or errors.
    
    Args:
        config: Configuration to validate.
        
    Returns:
        List of warning/error messages (empty if all valid).
    """
    warnings = []
    
    # Check dataset paths
    if not os.path.exists(config.dataset.groundtruth_csv):
        warnings.append(f"Groundtruth CSV not found: {config.dataset.groundtruth_csv}")
    
    if not os.path.exists(config.dataset.acoustic_features_csv):
        warnings.append(f"Acoustic features CSV not found: {config.dataset.acoustic_features_csv}")
    
    if not os.path.isdir(config.dataset.transcripts_dir):
        warnings.append(f"Transcripts directory not found: {config.dataset.transcripts_dir}")
    
    # Check active provider has API key (except Ollama which is local)
    provider, llm_config = get_active_llm_config(config)
    if provider != "ollama" and not llm_config.api_key:
        warnings.append(f"API key not set for active provider: {provider}")
    
    return warnings
