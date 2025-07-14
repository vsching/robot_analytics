"""Tests for CSV processing functionality."""

import pytest
import pandas as pd
import io
from datetime import date, datetime
from decimal import Decimal

from src.utils import (
    CSVProcessor, CSVFormatDetector, CSVValidator, CSVTransformer,
    Platform, ValidationSeverity, StreamingCSVProcessor
)
from src.models import Trade


class TestCSVFormatDetection:
    def test_detect_tradingview_format(self):
        """Test TradingView format detection."""
        csv_content = """Date/Time,Symbol,Side,Qty,Price,Commission
2024-01-15 10:30:00,BTC/USD,Buy,0.5,45000,10
2024-01-16 14:20:00,BTC/USD,Sell,0.5,46000,10"""
        
        detector = CSVFormatDetector()
        platform, format_spec = detector.detect_format(csv_content.encode(), "test.csv")
        
        assert platform == Platform.TRADINGVIEW
        assert "Date/Time" in format_spec.required_columns
    
    def test_detect_metatrader_format(self):
        """Test MetaTrader format detection."""
        csv_content = """Ticket,Open Time,Type,Size,Symbol,Price,S/L,T/P,Close Time,Price,Commission,Swap,Profit
12345,2024.01.15 10:30:00,buy,0.01,EURUSD,1.0850,0,0,2024.01.15 15:30:00,1.0870,0.5,0,20"""
        
        detector = CSVFormatDetector()
        platform, format_spec = detector.detect_format(csv_content.encode(), "test.csv")
        
        assert platform == Platform.METATRADER4
        assert "Ticket" in format_spec.required_columns
    
    def test_detect_generic_format(self):
        """Test generic format detection."""
        csv_content = """date,symbol,side,quantity,price,pnl
2024-01-15,AAPL,buy,100,150.50,500
2024-01-16,AAPL,sell,100,151.50,100"""
        
        detector = CSVFormatDetector()
        platform, format_spec = detector.detect_format(csv_content.encode(), "test.csv")
        
        assert platform == Platform.GENERIC


class TestCSVValidation:
    def test_validate_valid_data(self):
        """Test validation of valid CSV data."""
        df = pd.DataFrame({
            'Date/Time': ['2024-01-15 10:30:00', '2024-01-16 14:20:00'],
            'Symbol': ['BTC/USD', 'BTC/USD'],
            'Side': ['Buy', 'Sell'],
            'Qty': [0.5, 0.5],
            'Price': [45000, 46000],
            'Commission': [10, 10]
        })
        
        validator = CSVValidator()
        result = validator.validate(df, Platform.TRADINGVIEW)
        
        assert result.is_valid
        assert len(result.get_errors()) == 0
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'Symbol': ['BTC/USD', 'BTC/USD'],
            'Side': ['Buy', 'Sell']
        })
        
        validator = CSVValidator()
        result = validator.validate(df, Platform.TRADINGVIEW)
        
        assert not result.is_valid
        errors = result.get_errors()
        assert any("Required column" in str(e) for e in errors)
    
    def test_validate_invalid_data_types(self):
        """Test validation with invalid data types."""
        df = pd.DataFrame({
            'Date/Time': ['2024-01-15 10:30:00', '2024-01-16 14:20:00'],
            'Symbol': ['BTC/USD', 'BTC/USD'],
            'Side': ['Buy', 'Sell'],
            'Qty': ['invalid', 'quantity'],  # Invalid numeric values
            'Price': [45000, 46000]
        })
        
        validator = CSVValidator()
        result = validator.validate(df, Platform.TRADINGVIEW)
        
        errors = result.get_errors()
        assert any("Non-numeric values" in str(e) for e in errors)


class TestCSVTransformation:
    def test_transform_tradingview_data(self):
        """Test transformation of TradingView data."""
        df = pd.DataFrame({
            'Date/Time': ['2024-01-15 10:30:00', '2024-01-16 14:20:00'],
            'Symbol': ['BTC/USD', 'ETH/USD'],
            'Side': ['Buy', 'Sell'],
            'Qty': ['0.5', '1.0'],
            'Price': ['45000', '2500'],
            'Commission': ['10', '5']
        })
        
        transformer = CSVTransformer()
        transformed = transformer.transform(df, Platform.TRADINGVIEW)
        
        assert 'trade_date' in transformed.columns
        assert 'symbol' in transformed.columns
        assert 'quantity' in transformed.columns
        assert len(transformed) == 2
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(transformed['trade_date'])
        assert pd.api.types.is_numeric_dtype(transformed['quantity'])
    
    def test_transform_side_values(self):
        """Test side value standardization."""
        df = pd.DataFrame({
            'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'side': ['BUY', 'SELL', 'Long'],
            'quantity': [100, 100, 50],
            'price': [150, 155, 160]
        })
        
        transformer = CSVTransformer()
        transformed = transformer.transform(df, Platform.GENERIC)
        
        assert all(side in ['buy', 'sell', 'long', 'short'] for side in transformed['side'])
    
    def test_to_trades_conversion(self):
        """Test conversion to Trade objects."""
        df = pd.DataFrame({
            'trade_date': [pd.Timestamp('2024-01-15'), pd.Timestamp('2024-01-16')],
            'symbol': ['BTC/USD', 'ETH/USD'],
            'side': ['buy', 'sell'],
            'quantity': [0.5, 1.0],
            'entry_price': [45000.0, 2500.0],
            'exit_price': [46000.0, 2400.0],
            'pnl': [500.0, -100.0],
            'commission': [10.0, 5.0]
        })
        
        transformer = CSVTransformer()
        trades = transformer.to_trades(df, strategy_id=1)
        
        assert len(trades) == 2
        assert all(isinstance(t, Trade) for t in trades)
        assert trades[0].symbol == 'BTC/USD'
        assert trades[0].side == 'buy'
        assert trades[0].pnl == Decimal('500.0')


class TestCSVProcessor:
    def test_process_valid_csv(self):
        """Test processing a valid CSV file."""
        csv_content = """Date/Time,Symbol,Side,Qty,Price,Commission
2024-01-15 10:30:00,BTC/USD,Buy,0.5,45000,10
2024-01-16 14:20:00,BTC/USD,Sell,0.5,46000,10"""
        
        processor = CSVProcessor()
        trades, validation_result, metadata = processor.process(
            csv_content.encode(),
            "test.csv",
            strategy_id=1
        )
        
        assert validation_result.is_valid
        assert len(trades) == 2
        assert metadata['platform'] == 'TradingView'
        assert metadata['processed_count'] == 2
    
    def test_preview_csv(self):
        """Test CSV preview functionality."""
        csv_content = """date,symbol,side,quantity,price,pnl
2024-01-15,AAPL,buy,100,150.50,0
2024-01-16,AAPL,sell,100,151.50,100
2024-01-17,MSFT,buy,50,380.00,0
2024-01-18,MSFT,sell,50,385.00,250"""
        
        processor = CSVProcessor()
        preview_df, platform, validation_result = processor.preview(
            csv_content.encode(),
            "test.csv",
            rows=2
        )
        
        assert len(preview_df) <= 2
        assert platform == Platform.GENERIC
        assert validation_result.is_valid


class TestStreamingProcessor:
    def test_streaming_large_file(self):
        """Test streaming processing for large files."""
        # Create a large CSV content (simulate)
        rows = []
        for i in range(1000):
            rows.append(f"2024-01-{(i%30)+1:02d},AAPL,buy,100,{150+i*0.1:.2f},{i*10}")
        
        csv_content = "date,symbol,side,quantity,price,pnl\n" + "\n".join(rows)
        
        processor = StreamingCSVProcessor(streaming_threshold_mb=0.001)  # Force streaming
        
        # Process with streaming
        trades, validation_result, metadata = processor.process_with_progress(
            csv_content.encode(),
            "large_test.csv",
            Platform.GENERIC,
            strategy_id=1
        )
        
        assert metadata['streaming'] is True
        assert len(trades) == 1000
        assert metadata['chunks_processed'] > 0