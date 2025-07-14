"""CSV validation rules and error handling."""

import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

from .csv_formats import Platform, CSVFormat, get_format


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the CSV."""
    
    severity: ValidationSeverity
    message: str
    row: Optional[int] = None
    column: Optional[str] = None
    details: Optional[str] = None
    
    def __str__(self) -> str:
        """Format validation issue as string."""
        location = []
        if self.row is not None:
            location.append(f"Row {self.row}")
        if self.column is not None:
            location.append(f"Column '{self.column}'")
        
        location_str = f" [{', '.join(location)}]" if location else ""
        details_str = f" - {self.details}" if self.details else ""
        
        return f"{self.severity.value.upper()}{location_str}: {self.message}{details_str}"


@dataclass
class ValidationResult:
    """Result of CSV validation."""
    
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def summary(self) -> str:
        """Get validation summary."""
        errors = len(self.get_errors())
        warnings = len(self.get_warnings())
        
        if self.is_valid:
            return f"Validation passed with {warnings} warning(s)"
        else:
            return f"Validation failed with {errors} error(s) and {warnings} warning(s)"


class CSVValidator:
    """Validates trading CSV files according to format specifications."""
    
    def __init__(self):
        self.max_file_size_mb = 100
        self.min_rows = 1
        self.max_missing_percentage = 0.5  # 50% missing values threshold
    
    def validate(self, df: pd.DataFrame, platform: Platform, 
                format_spec: Optional[CSVFormat] = None) -> ValidationResult:
        """
        Validate a DataFrame according to platform format.
        
        Args:
            df: DataFrame to validate
            platform: Detected platform
            format_spec: Optional format specification override
            
        Returns:
            ValidationResult with issues and statistics
        """
        if format_spec is None:
            format_spec = get_format(platform)
        
        result = ValidationResult(is_valid=True)
        
        # Basic validation
        self._validate_basic(df, result)
        
        # Column validation
        self._validate_columns(df, format_spec, result)
        
        # Data type validation
        self._validate_data_types(df, format_spec, result)
        
        # Date validation
        self._validate_dates(df, format_spec, result)
        
        # Business rule validation
        self._validate_business_rules(df, format_spec, result)
        
        # Calculate statistics
        result.statistics = self._calculate_statistics(df, format_spec)
        
        return result
    
    def _validate_basic(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Perform basic DataFrame validation."""
        # Check if empty
        if df.empty:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="CSV file is empty"
            ))
            return
        
        # Check minimum rows
        if len(df) < self.min_rows:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"CSV has fewer than {self.min_rows} data rows"
            ))
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Duplicate column names found: {duplicate_cols}"
            ))
    
    def _validate_columns(self, df: pd.DataFrame, format_spec: CSVFormat, 
                         result: ValidationResult) -> None:
        """Validate required columns are present."""
        df_columns = set(df.columns)
        df_columns_lower = {col.lower() for col in df_columns}
        
        # Check required columns
        for required_col in format_spec.required_columns:
            if required_col not in df_columns and required_col.lower() not in df_columns_lower:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Required column '{required_col}' is missing",
                    details=f"Available columns: {list(df.columns)}"
                ))
        
        # Warn about extra columns
        expected_columns = format_spec.get_all_columns()
        expected_lower = {col.lower() for col in expected_columns}
        extra_columns = [col for col in df_columns if col.lower() not in expected_lower]
        
        if extra_columns and len(extra_columns) > 5:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Found {len(extra_columns)} unexpected columns",
                details=f"First few: {extra_columns[:5]}"
            ))
    
    def _validate_data_types(self, df: pd.DataFrame, format_spec: CSVFormat, 
                           result: ValidationResult) -> None:
        """Validate data types in columns."""
        # Check numeric columns
        numeric_columns = ['quantity', 'price', 'entry_price', 'exit_price', 'pnl', 
                          'commission', 'volume', 'qty', 'size']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if it should be numeric
            if any(num_col in col_lower for num_col in numeric_columns):
                try:
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    non_numeric_count = numeric_series.isna().sum() - df[col].isna().sum()
                    
                    if non_numeric_count > 0:
                        sample_values = df[col][numeric_series.isna() & df[col].notna()].head(3).tolist()
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Non-numeric values found in numeric column '{col}'",
                            details=f"{non_numeric_count} invalid values. Examples: {sample_values}"
                        ))
                except Exception as e:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        message=f"Failed to validate numeric column: {str(e)}"
                    ))
            
            # Check missing values
            missing_count = df[col].isna().sum()
            missing_percentage = missing_count / len(df)
            
            if missing_percentage > self.max_missing_percentage:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    column=col,
                    message=f"High percentage of missing values ({missing_percentage:.1%})",
                    details=f"{missing_count} out of {len(df)} rows"
                ))
    
    def _validate_dates(self, df: pd.DataFrame, format_spec: CSVFormat, 
                       result: ValidationResult) -> None:
        """Validate date columns."""
        for date_col in format_spec.date_columns:
            if date_col not in df.columns:
                continue
            
            # Try to parse dates
            try:
                # First try with specified formats
                parsed_dates = None
                for date_format in format_spec.date_formats:
                    try:
                        parsed_dates = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                        break
                    except:
                        continue
                
                # If no format worked, try pandas auto-detection
                if parsed_dates is None:
                    parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
                
                # Check for parsing failures
                failed_count = parsed_dates.isna().sum() - df[date_col].isna().sum()
                if failed_count > 0:
                    sample_values = df[date_col][parsed_dates.isna() & df[date_col].notna()].head(3).tolist()
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        column=date_col,
                        message=f"Invalid date format in column '{date_col}'",
                        details=f"{failed_count} unparseable dates. Examples: {sample_values}"
                    ))
                
                # Check date range
                if parsed_dates.notna().any():
                    min_date = parsed_dates.min()
                    max_date = parsed_dates.max()
                    
                    # Warn about future dates
                    if max_date > pd.Timestamp.now():
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            column=date_col,
                            message="Future dates detected",
                            details=f"Latest date: {max_date}"
                        ))
                    
                    # Warn about very old dates
                    if min_date < pd.Timestamp('2000-01-01'):
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            column=date_col,
                            message="Very old dates detected",
                            details=f"Earliest date: {min_date}"
                        ))
                
            except Exception as e:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    column=date_col,
                    message=f"Failed to parse dates: {str(e)}"
                ))
    
    def _validate_business_rules(self, df: pd.DataFrame, format_spec: CSVFormat, 
                               result: ValidationResult) -> None:
        """Validate business rules specific to trading data."""
        # Check for side/direction values
        side_columns = ['side', 'direction', 'type', 'market pos.', 'buy/sell']
        for col in df.columns:
            if col.lower() in side_columns:
                unique_values = df[col].dropna().unique()
                valid_sides = {'buy', 'sell', 'long', 'short', 'b', 's', 'l', 'sh'}
                
                # Normalize and check
                normalized_values = {str(v).lower().strip() for v in unique_values}
                invalid_values = normalized_values - valid_sides
                
                if invalid_values:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        column=col,
                        message=f"Unexpected values in side column '{col}'",
                        details=f"Found: {list(invalid_values)}"
                    ))
        
        # Check for negative quantities
        quantity_columns = ['quantity', 'qty', 'volume', 'size']
        for col in df.columns:
            if col.lower() in quantity_columns:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    negative_count = (numeric_values < 0).sum()
                    if negative_count > 0:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            column=col,
                            message=f"Negative values found in quantity column '{col}'",
                            details=f"{negative_count} negative values"
                        ))
                except:
                    pass
        
        # Check for zero prices
        price_columns = ['price', 'entry_price', 'exit_price', 'tradeprice']
        for col in df.columns:
            if col.lower() in price_columns:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    zero_count = (numeric_values == 0).sum()
                    if zero_count > 0:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            column=col,
                            message=f"Zero prices found in column '{col}'",
                            details=f"{zero_count} zero values"
                        ))
                except:
                    pass
    
    def _calculate_statistics(self, df: pd.DataFrame, format_spec: CSVFormat) -> Dict[str, Any]:
        """Calculate statistics about the data."""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': None,
            'unique_symbols': 0,
            'column_stats': {}
        }
        
        # Get date range
        for date_col in format_spec.date_columns:
            if date_col in df.columns:
                try:
                    dates = pd.to_datetime(df[date_col], errors='coerce')
                    if dates.notna().any():
                        stats['date_range'] = {
                            'start': dates.min().isoformat(),
                            'end': dates.max().isoformat(),
                            'days': (dates.max() - dates.min()).days
                        }
                    break
                except:
                    pass
        
        # Count unique symbols
        symbol_columns = ['symbol', 'ticker', 'instrument']
        for col in df.columns:
            if col.lower() in symbol_columns:
                stats['unique_symbols'] = df[col].nunique()
                break
        
        # Basic column statistics
        for col in df.columns:
            col_stats = {
                'missing_count': df[col].isna().sum(),
                'unique_count': df[col].nunique(),
                'dtype': str(df[col].dtype)
            }
            
            # Add numeric statistics if applicable
            try:
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                if numeric_values.notna().any():
                    col_stats.update({
                        'min': float(numeric_values.min()),
                        'max': float(numeric_values.max()),
                        'mean': float(numeric_values.mean()),
                        'std': float(numeric_values.std())
                    })
            except:
                pass
            
            stats['column_stats'][col] = col_stats
        
        return stats