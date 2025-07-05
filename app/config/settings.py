"""
Application configuration settings using Pydantic Settings.
"""

from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application Settings
    app_name: str = Field(default="Colombia RAG Chatbot", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], 
        env="CORS_ORIGINS"
    )
    
    # Database Settings
    vector_db_path: str = Field(default="./data/vectorstore", env="VECTOR_DB_PATH")
    vector_db_collection_name: str = Field(
        default="colombia_documents", 
        env="VECTOR_DB_COLLECTION_NAME"
    )
    
    # RAG Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_context_length: int = Field(default=2000, env="MAX_CONTEXT_LENGTH")
    top_k_documents: int = Field(default=8, env="TOP_K_DOCUMENTS")
    similarity_threshold: float = Field(default=0.30, env="SIMILARITY_THRESHOLD")
    
    # Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL"
    )
    llm_model: str = Field(default="llama3.2:1b", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=500, env="MAX_TOKENS")
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    # Content validation limits
    max_document_length: int = Field(default=500000, env="MAX_DOCUMENT_LENGTH")  # 500k chars
    # Data Source
    wikipedia_url: str = Field(
        default="https://es.wikipedia.org/wiki/Colombia", 
        env="WIKIPEDIA_URL"
    )
    data_update_interval: int = Field(default=86400, env="DATA_UPDATE_INTERVAL")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production", 
        env="SECRET_KEY"
    )
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level. Must be one of: {valid_levels}')
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == 'development'
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == 'production'
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (for dependency injection)."""
    return settings