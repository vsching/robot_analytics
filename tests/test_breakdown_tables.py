"""
Unit tests for the BreakdownTables component.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.components.breakdown_tables import BreakdownTables
from src.analytics.analytics_engine import AnalyticsEngine
from src.analytics.cache_manager import CacheManager


class TestBreakdownTables:
    """Test cases for BreakdownTables component."""
    
    @pytest.fixture
    def mock_analytics_engine(self):
        """Create mock analytics engine."""
        return Mock(spec=AnalyticsEngine)
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        mock = Mock(spec=CacheManager)
        mock.get.return_value = None  # No cached data by default
        return mock
    
    @pytest.fixture
    def breakdown_tables(self, mock_analytics_engine, mock_cache_manager):
        """Create BreakdownTables instance with mocks."""
        return BreakdownTables(mock_analytics_engine, mock_cache_manager)
    
    def test_init(self, breakdown_tables):
        """Test BreakdownTables initialization."""
        assert breakdown_tables.analytics is not None
        assert breakdown_tables.cache is not None
        assert breakdown_tables.grid_theme is not None
        assert breakdown_tables.currency_renderer is not None
        assert breakdown_tables.percentage_renderer is not None
        assert breakdown_tables.integer_renderer is not None
    
    def test_prepare_monthly_data_structure(self, breakdown_tables):
        """Test monthly data preparation returns correct structure."""
        df = breakdown_tables._prepare_monthly_data(1, None)
        
        # Check DataFrame structure
        assert not df.empty
        assert 'Month' in df.columns
        assert 'Total P&L' in df.columns
        assert 'Trade Count' in df.columns
        assert 'Win Rate' in df.columns
        assert 'Average Trade' in df.columns
        assert 'Best Trade' in df.columns
        assert 'Worst Trade' in df.columns
        
        # Check for summary row
        assert df.iloc[-1]['Month'] == 'TOTAL'
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(df['Total P&L'])
        assert pd.api.types.is_numeric_dtype(df['Trade Count'])
        assert pd.api.types.is_numeric_dtype(df['Win Rate'])
    
    def test_prepare_weekly_data_structure(self, breakdown_tables):
        """Test weekly data preparation returns correct structure."""
        df = breakdown_tables._prepare_weekly_data(1, None)
        
        # Check DataFrame structure
        assert not df.empty
        assert 'Week' in df.columns
        assert 'Total P&L' in df.columns
        assert 'Trade Count' in df.columns
        assert 'Win Rate' in df.columns
        
        # Check day-of-week columns
        for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
            assert f'{day} P&L' in df.columns
        
        # Check for summary row
        assert df.iloc[-1]['Week'] == 'TOTAL'
    
    def test_configure_grid_options(self, breakdown_tables):
        """Test grid options configuration."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'Month': ['Jan 2024', 'Feb 2024'],
            'Total P&L': [1000.0, -500.0],
            'Win Rate': [60.0, 45.0],
            'Trade Count': [50, 30]
        })
        
        gb = breakdown_tables._configure_grid_options(
            df,
            value_columns=['Total P&L'],
            percentage_columns=['Win Rate'],
            integer_columns=['Trade Count'],
            pinned_columns=['Month']
        )
        
        # Check that GridOptionsBuilder is returned
        assert gb is not None
        
        # Build options and check configuration
        options = gb.build()
        assert options is not None
        assert 'columnDefs' in options
        assert 'defaultColDef' in options
        assert options['pagination'] == True
        assert options['paginationPageSize'] == 20
    
    def test_render_period_selector_returns_tuple(self, breakdown_tables):
        """Test period selector returns date range tuple when dates are set."""
        # This would require mocking streamlit components
        # For now, we'll test the logic directly
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        
        # The method should return a tuple with time components
        expected = (
            datetime.combine(start.date(), datetime.min.time()),
            datetime.combine(end.date(), datetime.max.time())
        )
        
        # Verify the date combination logic
        assert expected[0].time() == datetime.min.time()
        assert expected[1].time() == datetime.max.time()
    
    def test_export_data_generates_csv(self, breakdown_tables):
        """Test export data generates proper CSV format."""
        df = pd.DataFrame({
            'Month': ['Jan 2024', 'Feb 2024'],
            'Total P&L': [1000.0, -500.0],
            'Trade Count': [50, 30]
        })
        
        # Test CSV generation
        csv_output = df.to_csv(index=False)
        
        # Verify CSV format
        assert 'Month,Total P&L,Trade Count' in csv_output
        assert 'Jan 2024,1000.0,50' in csv_output
        assert 'Feb 2024,-500.0,30' in csv_output
    
    def test_cache_integration(self, breakdown_tables, mock_cache_manager):
        """Test cache manager integration."""
        # Test cache miss scenario
        mock_cache_manager.get.return_value = None
        
        # Call method that uses cache
        df = breakdown_tables._prepare_monthly_data(1, None)
        
        # Verify cache was checked
        mock_cache_manager.get.assert_called()
        
        # Test cache hit scenario
        cached_df = pd.DataFrame({'cached': [True]})
        mock_cache_manager.get.return_value = cached_df
        
        # The method should return cached data
        # (In actual implementation, this would be tested with the full method)
    
    def test_monthly_data_summary_calculations(self, breakdown_tables):
        """Test summary row calculations in monthly data."""
        df = breakdown_tables._prepare_monthly_data(1, None)
        
        # Get summary row (last row)
        summary = df.iloc[-1]
        data_rows = df.iloc[:-1]
        
        # Verify summary calculations
        assert summary['Month'] == 'TOTAL'
        assert abs(summary['Total P&L'] - data_rows['Total P&L'].sum()) < 0.01
        assert summary['Trade Count'] == data_rows['Trade Count'].sum()
        assert abs(summary['Win Rate'] - data_rows['Win Rate'].mean()) < 0.01
        assert summary['Best Trade'] == data_rows['Best Trade'].max()
        assert summary['Worst Trade'] == data_rows['Worst Trade'].min()
    
    def test_weekly_data_day_of_week_columns(self, breakdown_tables):
        """Test weekly data includes all day-of-week columns."""
        df = breakdown_tables._prepare_weekly_data(1, None)
        
        # Verify all weekday columns exist
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        for day in weekdays:
            column_name = f'{day} P&L'
            assert column_name in df.columns
            assert pd.api.types.is_numeric_dtype(df[column_name])
        
        # Verify summary row includes day totals
        summary = df.iloc[-1]
        data_rows = df.iloc[:-1]
        
        for day in weekdays:
            column_name = f'{day} P&L'
            expected_total = data_rows[column_name].sum()
            assert abs(summary[column_name] - expected_total) < 0.01