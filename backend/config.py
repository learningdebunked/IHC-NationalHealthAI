"""Configuration management for IHC Platform."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "IHC-Platform"
    environment: str = "development"
    debug: bool = True
    secret_key: str
    api_version: str = "v1"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Database
    database_url: str
    db_pool_size: int = 20
    db_max_overflow: int = 10
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    cache_ttl: int = 3600
    
    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # ML Models
    model_path: str = "./models/saved"
    eligibility_model_path: str = "./models/saved/eligibility_classifier.pth"
    spending_model_path: str = "./models/saved/spending_predictor.pth"
    assistant_model_path: str = "./models/saved/assistant_model.pth"
    
    # Model Configuration
    max_sequence_length: int = 512
    batch_size: int = 32
    confidence_threshold: float = 0.85
    
    # External APIs
    irs_api_key: Optional[str] = None
    meps_data_url: str = "https://meps.ahrq.gov/data"
    
    # File Upload
    max_upload_size: int = 10485760  # 10MB
    allowed_extensions: List[str] = ["jpg", "jpeg", "png", "pdf"]
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/ihc.log"
    
    # Monitoring
    prometheus_port: int = 9090
    metrics_enabled: bool = True
    
    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: str = "noreply@ihc-platform.com"
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # Security
    hipaa_encryption_key: Optional[str] = None
    data_encryption_enabled: bool = True
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def api_prefix(self) -> str:
        """Get API prefix."""
        return f"/api/{self.api_version}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()