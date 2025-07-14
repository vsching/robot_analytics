"""CSV streaming processor for handling large files efficiently."""

import logging
from typing import Iterator, Tuple, List, Optional, Dict, Any, Callable
import pandas as pd
import io
from dataclasses import dataclass

from .csv_formats import Platform, CSVFormat, get_format
from .csv_validator import CSVValidator, ValidationResult, ValidationIssue, ValidationSeverity
from .csv_transformer import CSVTransformer
from ..models import Trade


logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result from processing a chunk of data."""
    chunk_index: int
    trades: List[Trade]
    validation_issues: List[ValidationIssue]
    row_count: int
    processed_count: int
    error: Optional[str] = None


class CSVStreamer:
    """Streams and processes large CSV files in chunks."""
    
    def __init__(self, 
                 chunk_size: int = 10000,
                 max_chunks: Optional[int] = None):
        """
        Initialize CSV streamer.
        
        Args:
            chunk_size: Number of rows per chunk
            max_chunks: Maximum number of chunks to process (None for all)
        """
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.validator = CSVValidator()
        self.transformer = CSVTransformer()
    
    def stream_process(self, 
                      file_content: bytes,
                      platform: Platform,
                      strategy_id: int,
                      format_spec: Optional[CSVFormat] = None,
                      progress_callback: Optional[Callable[[float, str], None]] = None
                      ) -> Iterator[ChunkResult]:
        """
        Stream process a CSV file in chunks.
        
        Args:
            file_content: Raw file content
            platform: Detected platform
            strategy_id: Strategy ID for trades
            format_spec: Optional format specification
            progress_callback: Callback for progress updates (progress, message)
            
        Yields:
            ChunkResult for each processed chunk
        """
        if format_spec is None:
            format_spec = get_format(platform)
        
        # Decode content
        try:
            text_content = file_content.decode(format_spec.encoding)
        except UnicodeDecodeError:
            text_content = file_content.decode(format_spec.encoding, errors='ignore')
        
        # Create text stream
        text_stream = io.StringIO(text_content)
        
        # Process chunks
        chunk_index = 0
        total_rows_processed = 0
        
        try:
            # Read CSV in chunks
            for chunk_df in pd.read_csv(
                text_stream,
                delimiter=format_spec.delimiter,
                chunksize=self.chunk_size,
                dtype=str,
                keep_default_na=True,
                na_values=['', 'N/A', 'NA', 'null', 'NULL', 'None', 'NONE'],
                on_bad_lines='skip'
            ):
                # Check max chunks limit
                if self.max_chunks and chunk_index >= self.max_chunks:
                    break
                
                # Update progress
                if progress_callback:
                    progress = (chunk_index + 1) / (self.max_chunks or 10) * 100
                    progress_callback(
                        min(progress, 99),
                        f"Processing chunk {chunk_index + 1} ({len(chunk_df)} rows)"
                    )
                
                # Process chunk
                result = self._process_chunk(
                    chunk_df, 
                    chunk_index, 
                    platform, 
                    format_spec, 
                    strategy_id
                )
                
                total_rows_processed += result.row_count
                chunk_index += 1
                
                yield result
                
                # Log progress
                logger.info(f"Processed chunk {chunk_index}: {result.processed_count} trades from {result.row_count} rows")
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield ChunkResult(
                chunk_index=chunk_index,
                trades=[],
                validation_issues=[ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Streaming failed: {str(e)}"
                )],
                row_count=0,
                processed_count=0,
                error=str(e)
            )
        
        finally:
            if progress_callback:
                progress_callback(100, f"Completed processing {total_rows_processed} rows")
    
    def _process_chunk(self, 
                      chunk_df: pd.DataFrame,
                      chunk_index: int,
                      platform: Platform,
                      format_spec: CSVFormat,
                      strategy_id: int) -> ChunkResult:
        """Process a single chunk of data."""
        validation_issues = []
        trades = []
        
        try:
            # Strip whitespace from column names
            chunk_df.columns = chunk_df.columns.str.strip()
            
            # Quick validation (don't fail on warnings)
            validation_result = self.validator.validate(chunk_df, platform, format_spec)
            validation_issues.extend(validation_result.issues)
            
            # Only process if no errors
            if not validation_result.get_errors():
                # Transform data
                transformed_df = self.transformer.transform(chunk_df, platform, format_spec)
                
                # Convert to trades
                trades = self.transformer.to_trades(transformed_df, strategy_id)
            
            return ChunkResult(
                chunk_index=chunk_index,
                trades=trades,
                validation_issues=validation_issues,
                row_count=len(chunk_df),
                processed_count=len(trades)
            )
        
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_index}: {e}")
            return ChunkResult(
                chunk_index=chunk_index,
                trades=[],
                validation_issues=[ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Chunk processing failed: {str(e)}"
                )],
                row_count=len(chunk_df),
                processed_count=0,
                error=str(e)
            )
    
    def estimate_chunks(self, file_size_bytes: int) -> int:
        """
        Estimate number of chunks based on file size.
        
        Args:
            file_size_bytes: Size of file in bytes
            
        Returns:
            Estimated number of chunks
        """
        # Rough estimate: assume average 100 bytes per row
        estimated_rows = file_size_bytes / 100
        estimated_chunks = int(estimated_rows / self.chunk_size) + 1
        return estimated_chunks
    
    def validate_first_chunk(self, 
                           file_content: bytes,
                           platform: Platform,
                           format_spec: Optional[CSVFormat] = None) -> ValidationResult:
        """
        Validate only the first chunk of a file.
        
        Args:
            file_content: Raw file content
            platform: Detected platform
            format_spec: Optional format specification
            
        Returns:
            ValidationResult for the first chunk
        """
        if format_spec is None:
            format_spec = get_format(platform)
        
        try:
            # Decode content
            text_content = file_content.decode(format_spec.encoding, errors='ignore')
            
            # Read first chunk only
            df = pd.read_csv(
                io.StringIO(text_content),
                delimiter=format_spec.delimiter,
                nrows=self.chunk_size,
                dtype=str,
                keep_default_na=True,
                on_bad_lines='skip'
            )
            
            # Validate
            return self.validator.validate(df, platform, format_spec)
        
        except Exception as e:
            logger.error(f"Failed to validate first chunk: {e}")
            result = ValidationResult(is_valid=False)
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Failed to read file: {str(e)}"
            ))
            return result


class StreamingCSVProcessor:
    """Enhanced CSV processor with streaming support for large files."""
    
    def __init__(self, 
                 streaming_threshold_mb: int = 10,
                 chunk_size: int = 10000):
        """
        Initialize streaming processor.
        
        Args:
            streaming_threshold_mb: File size threshold to trigger streaming
            chunk_size: Rows per chunk for streaming
        """
        self.streaming_threshold_bytes = streaming_threshold_mb * 1024 * 1024
        self.chunk_size = chunk_size
        self.streamer = CSVStreamer(chunk_size=chunk_size)
        
        # Import here to avoid circular dependency
        from .csv_processor import CSVProcessor
        self.regular_processor = CSVProcessor()
    
    def should_stream(self, file_size_bytes: int) -> bool:
        """Determine if file should be streamed based on size."""
        return file_size_bytes > self.streaming_threshold_bytes
    
    def process_with_progress(self,
                            file_content: bytes,
                            filename: str,
                            platform: Platform,
                            strategy_id: int,
                            format_spec: Optional[CSVFormat] = None,
                            progress_callback: Optional[Callable[[float, str], None]] = None
                            ) -> Tuple[List[Trade], ValidationResult, Dict[str, Any]]:
        """
        Process file with progress updates, using streaming if needed.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            platform: Detected platform
            strategy_id: Strategy ID
            format_spec: Optional format spec
            progress_callback: Progress callback function
            
        Returns:
            Tuple of (trades, validation_result, metadata)
        """
        file_size = len(file_content)
        metadata = {
            'filename': filename,
            'file_size_bytes': file_size,
            'platform': platform.value,
            'streaming': self.should_stream(file_size),
            'chunks_processed': 0,
            'total_rows': 0,
            'total_trades': 0
        }
        
        if self.should_stream(file_size):
            logger.info(f"Using streaming for large file: {file_size / 1024 / 1024:.1f}MB")
            return self._process_streaming(
                file_content, platform, strategy_id, format_spec, 
                progress_callback, metadata
            )
        else:
            logger.info(f"Using regular processing for file: {file_size / 1024 / 1024:.1f}MB")
            # Use regular processor
            result = self.regular_processor.process(file_content, filename, strategy_id)
            if progress_callback:
                progress_callback(100, "Processing complete")
            return result
    
    def _process_streaming(self,
                         file_content: bytes,
                         platform: Platform,
                         strategy_id: int,
                         format_spec: Optional[CSVFormat],
                         progress_callback: Optional[Callable[[float, str], None]],
                         metadata: Dict[str, Any]
                         ) -> Tuple[List[Trade], ValidationResult, Dict[str, Any]]:
        """Process file using streaming."""
        all_trades = []
        validation_result = ValidationResult(is_valid=True)
        
        # First validate the first chunk
        first_chunk_validation = self.streamer.validate_first_chunk(
            file_content, platform, format_spec
        )
        
        if not first_chunk_validation.is_valid:
            return [], first_chunk_validation, metadata
        
        # Process chunks
        for chunk_result in self.streamer.stream_process(
            file_content, platform, strategy_id, format_spec, progress_callback
        ):
            # Collect trades
            all_trades.extend(chunk_result.trades)
            
            # Collect validation issues
            for issue in chunk_result.validation_issues:
                validation_result.add_issue(issue)
            
            # Update metadata
            metadata['chunks_processed'] += 1
            metadata['total_rows'] += chunk_result.row_count
            metadata['total_trades'] += chunk_result.processed_count
            
            # Stop if critical error
            if chunk_result.error and validation_result.get_errors():
                break
        
        metadata['processed_count'] = len(all_trades)
        
        return all_trades, validation_result, metadata