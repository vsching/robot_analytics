"""Main CSV processor that orchestrates detection, validation, and transformation."""

import logging
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import io

from .csv_detector import CSVFormatDetector
from .csv_validator import CSVValidator, ValidationResult
from .csv_transformer import CSVTransformer
from .csv_formats import Platform, CSVFormat
from ..models import Trade
from config import config


logger = logging.getLogger(__name__)


class CSVProcessor:
    """Main processor for handling CSV uploads."""
    
    def __init__(self):
        self.detector = CSVFormatDetector()
        self.validator = CSVValidator()
        self.transformer = CSVTransformer()
        self.max_file_size = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    def process(self, file_content: bytes, filename: str, 
               strategy_id: int) -> Tuple[List[Trade], ValidationResult, Dict[str, Any]]:
        """
        Process a CSV file through the complete pipeline.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            strategy_id: Strategy ID for the trades
            
        Returns:
            Tuple of (trades, validation_result, metadata)
        """
        metadata = {
            'filename': filename,
            'file_size_bytes': len(file_content),
            'platform': None,
            'row_count': 0,
            'processed_count': 0
        }
        
        # Check file size
        if len(file_content) > self.max_file_size:
            validation_result = ValidationResult(is_valid=False)
            validation_result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"File size ({len(file_content) / 1024 / 1024:.1f}MB) exceeds maximum allowed ({config.MAX_UPLOAD_SIZE_MB}MB)"
            ))
            return [], validation_result, metadata
        
        try:
            # Step 1: Detect format
            logger.info(f"Processing file: {filename}")
            platform, format_spec = self.detector.detect_format(file_content, filename)
            metadata['platform'] = platform.value
            
            # Step 2: Read CSV with detected format
            df = self._read_csv(file_content, format_spec)
            metadata['row_count'] = len(df)
            
            # Step 3: Validate
            validation_result = self.validator.validate(df, platform, format_spec)
            
            if not validation_result.is_valid:
                logger.warning(f"Validation failed for {filename}: {validation_result.summary()}")
                return [], validation_result, metadata
            
            # Step 4: Transform
            transformed_df = self.transformer.transform(df, platform, format_spec)
            
            # Step 5: Convert to Trade objects
            trades = self.transformer.to_trades(transformed_df, strategy_id)
            metadata['processed_count'] = len(trades)
            
            logger.info(f"Successfully processed {len(trades)} trades from {filename}")
            
            return trades, validation_result, metadata
            
        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            validation_result = ValidationResult(is_valid=False)
            validation_result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Processing failed: {str(e)}"
            ))
            return [], validation_result, metadata
    
    def preview(self, file_content: bytes, filename: str, 
               rows: int = 10) -> Tuple[pd.DataFrame, Platform, ValidationResult]:
        """
        Preview a CSV file without full processing.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            rows: Number of rows to preview
            
        Returns:
            Tuple of (preview_df, platform, validation_result)
        """
        try:
            # Detect format
            platform, format_spec = self.detector.detect_format(file_content, filename)
            
            # Read limited rows
            df = self._read_csv(file_content, format_spec, nrows=rows)
            
            # Quick validation
            validation_result = self.validator.validate(df, platform, format_spec)
            
            # Transform for preview
            if validation_result.is_valid or len(validation_result.get_errors()) == 0:
                preview_df = self.transformer.transform(df, platform, format_spec)
            else:
                preview_df = df
            
            return preview_df, platform, validation_result
            
        except Exception as e:
            logger.error(f"Failed to preview CSV: {e}")
            return pd.DataFrame(), Platform.UNKNOWN, ValidationResult(is_valid=False)
    
    def _read_csv(self, file_content: bytes, format_spec: CSVFormat, 
                 nrows: Optional[int] = None) -> pd.DataFrame:
        """Read CSV content into DataFrame."""
        try:
            # Decode content
            encoding = format_spec.encoding
            try:
                text_content = file_content.decode(encoding)
            except UnicodeDecodeError:
                # Try with error handling
                text_content = file_content.decode(encoding, errors='ignore')
            
            # Read CSV
            df = pd.read_csv(
                io.StringIO(text_content),
                delimiter=format_spec.delimiter,
                nrows=nrows,
                on_bad_lines='skip',
                dtype=str,  # Read as strings initially for validation
                keep_default_na=True,
                na_values=['', 'N/A', 'NA', 'null', 'NULL', 'None', 'NONE']
            )
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Strip whitespace from string values
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise
    
    def analyze_format(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Analyze CSV format and return detailed information.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            
        Returns:
            Dictionary with format analysis
        """
        try:
            # Detect format
            platform, format_spec = self.detector.detect_format(file_content, filename)
            
            # Analyze sample data
            analysis = self.detector.analyze_sample_data(file_content, platform)
            
            # Add format information
            analysis['platform'] = platform.value
            analysis['format_name'] = format_spec.name
            analysis['required_columns'] = list(format_spec.required_columns)
            analysis['optional_columns'] = list(format_spec.optional_columns)
            analysis['delimiter'] = format_spec.delimiter
            analysis['date_formats'] = format_spec.date_formats
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze format: {e}")
            return {
                'error': str(e),
                'platform': Platform.UNKNOWN.value
            }
    
    def get_supported_formats(self) -> List[Dict[str, Any]]:
        """Get list of supported formats with their specifications."""
        formats = []
        
        for platform in Platform:
            if platform == Platform.UNKNOWN:
                continue
            
            format_spec = get_format(platform)
            formats.append({
                'platform': platform.value,
                'name': format_spec.name,
                'description': format_spec.description,
                'required_columns': list(format_spec.required_columns),
                'optional_columns': list(format_spec.optional_columns),
                'example_date_formats': format_spec.date_formats[:3]
            })
        
        return formats


# Import after class definition to avoid circular import
from .csv_validator import ValidationIssue, ValidationSeverity