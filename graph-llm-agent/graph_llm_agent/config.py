import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict, EnvAlias
from dotenv import load_dotenv

# Load .env file from the project root if it exists
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    # Neo4j Connection Details
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Hugging Face Settings
    # Handled by transformers library environment variable HF_HOME,
    # but can be explicitly set if needed.
    HF_HOME: Optional[str] = None

    # LLM Configuration
    LLM_MODEL_NAME: str = "Qwen/Qwen1.5-1.8B-Chat"
    LLM_API_BASE_URL: Optional[str] = None # e.g., "http://localhost:8000/v1" for vLLM
    LLM_API_KEY: Optional[str] = None # API key if LLM is served behind an authenticated endpoint

    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = "intfloat/e5-large-v2" # Default embedding model
    EMBEDDING_DIMENSIONS: Optional[int] = 1024 # Default to 1024 for e5-large-v2. Change if model changes.

    # Application Settings
    LOG_LEVEL: str = "INFO"

    # Retrieval Settings
    CONTEXT_WINDOW_SIZE: int = 8192 # Max tokens in LLM context (example based on some Qwen models)
    COMPACTION_TRIGGER_TOKENS: int = 6000
    COMPACTION_OLDEST_TOKENS: int = 2000
    RETRIEVAL_K_VALUE: int = 8 # Number of memories to retrieve
    RETRIEVAL_SCORE_SIMILARITY_WEIGHT: float = 0.6
    RETRIEVAL_SCORE_RECENCY_WEIGHT: float = 0.3
    RETRIEVAL_SCORE_IMPORTANCE_WEIGHT: float = 0.1
    RETRIEVAL_RECENCY_DECAY_HOURS: float = 24.0 # Tau for recency decay: exp(-age_hours/TAU)

    # Metacognition Settings
    REFLECTION_INTERACTION_INTERVAL: int = 5

    # Tokenizer Override (optional)
    OVERRIDE_TOKENIZER: Optional[str] = None # e.g., "p50k_base" or "cl100k_base"

    # Test Mode for CI or specific testing scenarios
    TEST_MODE: bool = Field(default=False, validation_alias=EnvAlias('GRAPH_AGENT_TEST_MODE'))

    # Model config dictionary, compatible with pydantic BaseSettings
    model_config = SettingsConfigDict(
        env_file=str(dotenv_path), # Tell pydantic to load .env if DotEnvSettingsSource is used (though we load manually too)
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from .env rather than erroring
    )

# Instantiate settings
settings = Settings()

# Set HF_HOME environment variable if provided in settings
if settings.HF_HOME:
    os.environ['HF_HOME'] = settings.HF_HOME

if __name__ == "__main__":
    # For testing if the settings are loaded correctly
    print("Loaded settings:")
    print(f"  NEO4J_URI: {settings.NEO4J_URI}")
    print(f"  LLM_MODEL_NAME: {settings.LLM_MODEL_NAME}")
    print(f"  EMBEDDING_MODEL_NAME: {settings.EMBEDDING_MODEL_NAME}")
    print(f"  LOG_LEVEL: {settings.LOG_LEVEL}")
    print(f"  CONTEXT_WINDOW_SIZE: {settings.CONTEXT_WINDOW_SIZE}")
    # Print HF_HOME from environment to confirm if it was set
    print(f"  HF_HOME (from env): {os.getenv('HF_HOME')}")
    print(f"  EMBEDDING_DIMENSIONS: {settings.EMBEDDING_DIMENSIONS}")
