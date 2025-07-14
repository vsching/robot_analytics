"""Application configuration settings."""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv


# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


class Config:
    """Application configuration."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    TEMP_DIR = BASE_DIR / 'temp'
    EXPORT_DIR = BASE_DIR / 'exports'
    REPORTS_DIR = BASE_DIR / 'reports'
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_analyzer.db')
    DATABASE_POOL_SIZE = int(os.getenv('DATABASE_POOL_SIZE', '5'))
    DATABASE_POOL_TIMEOUT = int(os.getenv('DATABASE_POOL_TIMEOUT', '30'))
    
    # Application
    APP_ENV = os.getenv('APP_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # File Upload
    MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', '100'))
    ALLOWED_FILE_EXTENSIONS = os.getenv('ALLOWED_FILE_EXTENSIONS', 'csv,xlsx,xls').split(',')
    
    # Session
    SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))
    
    # Export
    EXPORT_TEMP_DIR = Path(os.getenv('EXPORT_TEMP_DIR', './temp/exports'))
    PDF_REPORT_TEMPLATE = os.getenv('PDF_REPORT_TEMPLATE', 'default')
    
    # Email (for report delivery)
    SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USER = os.getenv('SMTP_USER', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
    
    # Performance
    CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '3600'))
    ENABLE_PERFORMANCE_METRICS = os.getenv('ENABLE_PERFORMANCE_METRICS', 'True').lower() == 'true'
    
    # Streamlit specific
    STREAMLIT_SERVER_PORT = int(os.getenv('STREAMLIT_SERVER_PORT', '8501'))
    STREAMLIT_SERVER_HEADLESS = os.getenv('STREAMLIT_SERVER_HEADLESS', 'True').lower() == 'true'
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for directory in [cls.DATA_DIR, cls.TEMP_DIR, cls.EXPORT_DIR, cls.REPORTS_DIR, cls.EXPORT_TEMP_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_upload_path(cls, filename: str) -> Path:
        """Get the full path for an uploaded file."""
        cls.ensure_directories()
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_export_path(cls, filename: str) -> Path:
        """Get the full path for an export file."""
        cls.ensure_directories()
        return cls.EXPORT_DIR / filename
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed."""
        ext = filename.split('.')[-1].lower()
        return ext in cls.ALLOWED_FILE_EXTENSIONS
    
    @classmethod
    def validate_file_size(cls, file_size_bytes: int) -> bool:
        """Check if file size is within limits."""
        max_size_bytes = cls.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        return file_size_bytes <= max_size_bytes


# Create instance
config = Config()
config.ensure_directories()