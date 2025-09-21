"""
Configuration Settings
Application settings and configuration values.
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Application settings
    APP_NAME = "Resume Relevance Checker"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # File upload settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    
    # Scoring weights
    SCORING_WEIGHTS = {
        'skills_match': float(os.getenv("WEIGHT_SKILLS", "0.35")),
        'experience_match': float(os.getenv("WEIGHT_EXPERIENCE", "0.25")),
        'education_match': float(os.getenv("WEIGHT_EDUCATION", "0.15")),
        'projects_match': float(os.getenv("WEIGHT_PROJECTS", "0.15")),
        'certification_match': float(os.getenv("WEIGHT_CERTIFICATIONS", "0.10"))
    }
    
    # NLP settings
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    
    # Database settings (for future use)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///resume_checker.db")
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    
    # Analysis settings
    MIN_SCORE_HIGH = float(os.getenv("MIN_SCORE_HIGH", "80"))
    MIN_SCORE_MEDIUM = float(os.getenv("MIN_SCORE_MEDIUM", "50"))
    MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "5"))
    
    # Cache settings
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "False").lower() == "true"
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
    
    # Rate limiting (for future use)
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    
    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        settings = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                settings[attr] = getattr(cls, attr)
        return settings
    
    @classmethod
    def validate_settings(cls) -> bool:
        """Validate configuration settings"""
        # Check required directories exist
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        
        # Validate scoring weights sum to 1.0
        total_weight = sum(cls.SCORING_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Scoring weights sum to {total_weight}, expected 1.0")
        
        return True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    UPLOAD_DIR = "test_uploads"
    DATABASE_URL = "sqlite:///test.db"

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': Config
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    return config_map.get(config_name, Config)

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

# Load environment variables
load_env_file()