"""Utility modules for the Trading Strategy Analyzer."""

from .csv_processor import CSVProcessor
from .csv_detector import CSVFormatDetector
from .csv_validator import CSVValidator, ValidationResult, ValidationIssue, ValidationSeverity
from .csv_transformer import CSVTransformer
from .csv_formats import Platform, CSVFormat, FORMATS, get_format
from .csv_streamer import CSVStreamer, StreamingCSVProcessor, ChunkResult
from .session_state import SessionStateManager, get_session_manager
from .auth_decorator import require_authentication
from .auth_check import check_authentication

__all__ = [
    'CSVProcessor',
    'CSVFormatDetector',
    'CSVValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'CSVTransformer',
    'Platform',
    'CSVFormat',
    'FORMATS',
    'get_format',
    'CSVStreamer',
    'StreamingCSVProcessor',
    'ChunkResult',
    'SessionStateManager',
    'get_session_manager',
    'require_authentication',
    'check_authentication'
]