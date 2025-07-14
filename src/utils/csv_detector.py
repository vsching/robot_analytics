"""CSV format detection logic for automatic platform identification."""

import csv
import io
import logging
from typing import Tuple, List, Dict, Optional, Set
from collections import Counter
import pandas as pd
from datetime import datetime

from .csv_formats import Platform, CSVFormat, FORMATS, get_format


logger = logging.getLogger(__name__)


class CSVFormatDetector:
    """Detects the format of trading CSV files from various platforms."""
    
    def __init__(self):
        self.formats = FORMATS
        self._common_delimiters = [',', ';', '\t', '|']
    
    def detect_format(self, file_content: bytes, filename: Optional[str] = None) -> Tuple[Platform, CSVFormat]:
        """
        Detect the format of a CSV file.
        
        Args:
            file_content: Raw file content as bytes
            filename: Optional filename for additional hints
            
        Returns:
            Tuple of (Platform, CSVFormat)
        """
        # Try to decode file content
        encoding = self._detect_encoding(file_content)
        try:
            text_content = file_content.decode(encoding)
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode with {encoding}, trying utf-8")
            text_content = file_content.decode('utf-8', errors='ignore')
        
        # Detect delimiter
        delimiter = self._detect_delimiter(text_content)
        
        # Parse headers
        headers = self._get_headers(text_content, delimiter)
        if not headers:
            logger.error("Could not parse CSV headers")
            return Platform.UNKNOWN, get_format(Platform.GENERIC)
        
        # Normalize headers for comparison
        normalized_headers = {h.strip().lower() for h in headers}
        
        # Score each format based on header matching
        scores = {}
        for platform, format_spec in self.formats.items():
            score = self._calculate_format_score(headers, normalized_headers, format_spec)
            scores[platform] = score
            logger.debug(f"Format {platform.value} score: {score}")
        
        # Get the best matching format
        best_platform = max(scores, key=scores.get)
        best_score = scores[best_platform]
        
        # If score is too low, use generic format
        if best_score < 0.5:
            logger.info(f"No strong format match (best score: {best_score}), using generic format")
            return Platform.GENERIC, get_format(Platform.GENERIC)
        
        logger.info(f"Detected format: {best_platform.value} (score: {best_score})")
        return best_platform, get_format(best_platform)
    
    def _detect_encoding(self, content: bytes) -> str:
        """Detect file encoding."""
        # Check for BOM
        if content.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        elif content.startswith(b'\xff\xfe'):
            return 'utf-16-le'
        elif content.startswith(b'\xfe\xff'):
            return 'utf-16-be'
        
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'utf-8'  # Default
    
    def _detect_delimiter(self, content: str) -> str:
        """Detect the delimiter used in the CSV."""
        # Take first few lines for analysis
        lines = content.strip().split('\n')[:10]
        if not lines:
            return ','
        
        # Count occurrences of each delimiter
        delimiter_counts = Counter()
        for line in lines:
            for delimiter in self._common_delimiters:
                delimiter_counts[delimiter] += line.count(delimiter)
        
        # Get the most common delimiter
        if delimiter_counts:
            delimiter = delimiter_counts.most_common(1)[0][0]
            logger.debug(f"Detected delimiter: '{delimiter}'")
            return delimiter
        
        return ','  # Default
    
    def _get_headers(self, content: str, delimiter: str) -> List[str]:
        """Extract headers from CSV content."""
        try:
            # Use csv.reader to handle quoted fields properly
            reader = csv.reader(io.StringIO(content), delimiter=delimiter)
            headers = next(reader, [])
            return [h.strip() for h in headers if h.strip()]
        except Exception as e:
            logger.error(f"Failed to parse headers: {e}")
            return []
    
    def _calculate_format_score(self, headers: List[str], normalized_headers: Set[str], 
                               format_spec: CSVFormat) -> float:
        """
        Calculate how well headers match a format specification.
        
        Returns:
            Score between 0 and 1, where 1 is perfect match
        """
        # Create normalized sets for comparison
        required_normalized = {col.lower() for col in format_spec.required_columns}
        optional_normalized = {col.lower() for col in format_spec.optional_columns}
        all_expected = required_normalized.union(optional_normalized)
        
        # Check exact matches first
        exact_required_matches = sum(1 for col in format_spec.required_columns if col in headers)
        exact_optional_matches = sum(1 for col in format_spec.optional_columns if col in headers)
        
        # Check normalized matches
        normalized_required_matches = len(required_normalized.intersection(normalized_headers))
        normalized_optional_matches = len(optional_normalized.intersection(normalized_headers))
        
        # Calculate base score
        if len(format_spec.required_columns) > 0:
            required_score = exact_required_matches / len(format_spec.required_columns)
        else:
            required_score = 1.0
        
        # Bonus for optional columns
        if len(format_spec.optional_columns) > 0:
            optional_score = exact_optional_matches / len(format_spec.optional_columns)
        else:
            optional_score = 0.0
        
        # Penalty for unknown columns
        unknown_columns = normalized_headers - all_expected
        if len(normalized_headers) > 0:
            unknown_penalty = len(unknown_columns) / len(normalized_headers)
        else:
            unknown_penalty = 0.0
        
        # Combine scores
        # Required columns are most important (70%)
        # Optional columns add bonus (20%)
        # Unknown columns reduce score (10%)
        final_score = (0.7 * required_score + 
                      0.2 * optional_score - 
                      0.1 * unknown_penalty)
        
        # Boost score for exact matches
        if exact_required_matches == len(format_spec.required_columns):
            final_score = min(1.0, final_score + 0.1)
        
        return max(0.0, min(1.0, final_score))
    
    def validate_date_format(self, date_string: str, date_formats: List[str]) -> Optional[datetime]:
        """
        Try to parse a date string with multiple formats.
        
        Args:
            date_string: Date string to parse
            date_formats: List of format strings to try
            
        Returns:
            Parsed datetime or None if parsing fails
        """
        date_string = date_string.strip()
        
        for date_format in date_formats:
            try:
                return datetime.strptime(date_string, date_format)
            except ValueError:
                continue
        
        # Try pandas parser as fallback
        try:
            return pd.to_datetime(date_string)
        except:
            return None
    
    def analyze_sample_data(self, file_content: bytes, platform: Platform, 
                          sample_size: int = 100) -> Dict[str, any]:
        """
        Analyze a sample of data to gather statistics.
        
        Args:
            file_content: Raw file content
            platform: Detected platform
            sample_size: Number of rows to analyze
            
        Returns:
            Dictionary with analysis results
        """
        format_spec = get_format(platform)
        
        try:
            # Read sample data
            df = pd.read_csv(
                io.BytesIO(file_content),
                delimiter=format_spec.delimiter,
                encoding=format_spec.encoding,
                nrows=sample_size,
                on_bad_lines='skip'
            )
            
            analysis = {
                'total_rows': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'unique_symbols': [],
                'date_range': None
            }
            
            # Try to identify symbol column
            symbol_columns = ['symbol', 'ticker', 'instrument', 'Symbol', 'Instrument']
            for col in symbol_columns:
                if col in df.columns:
                    analysis['unique_symbols'] = df[col].unique().tolist()[:10]
                    break
            
            # Try to identify date range
            for date_col in format_spec.date_columns:
                if date_col in df.columns:
                    try:
                        dates = pd.to_datetime(df[date_col])
                        analysis['date_range'] = {
                            'start': dates.min().isoformat(),
                            'end': dates.max().isoformat()
                        }
                        break
                    except:
                        continue
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze sample data: {e}")
            return {
                'error': str(e),
                'total_rows': 0,
                'columns': []
            }