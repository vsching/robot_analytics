"""Tests for export manager functionality."""

import pytest
import tempfile
import csv
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from src.exports.export_manager import ExportManager, ExportConfig, ReportTemplate
from src.models import Trade, Strategy
from src.analytics import PerformanceMetrics, PnLSummary, TradeStatistics


class TestExportManager:
    @pytest.fixture
    def temp_export_dir(self):
        """Create temporary export directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_strategy_manager(self):
        """Create mock strategy manager."""
        manager = Mock()
        
        # Mock strategies
        strategies = [
            Strategy(id=1, name="Test Strategy A", description="Test strategy A"),
            Strategy(id=2, name="Test Strategy B", description="Test strategy B")
        ]
        
        manager.get_all_strategies.return_value = strategies
        manager.get_strategy.side_effect = lambda id: next((s for s in strategies if s.id == id), None)
        
        return manager
    
    @pytest.fixture
    def mock_analytics_engine(self):
        """Create mock analytics engine."""
        engine = Mock()
        
        # Mock trades
        test_trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                quantity=Decimal("100"),
                pnl=Decimal("500.00"),
                commission=Decimal("10.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 2),
                symbol="MSFT",
                side="sell",
                entry_price=Decimal("380.00"),
                exit_price=Decimal("375.00"),
                quantity=Decimal("50"),
                pnl=Decimal("-250.00"),
                commission=Decimal("10.00")
            )
        ]
        
        engine.get_trades_for_strategy.return_value = test_trades
        
        # Mock metrics
        mock_pnl_summary = Mock(spec=PnLSummary)
        mock_pnl_summary.total_pnl = Decimal("250.00")
        mock_pnl_summary.average_pnl = Decimal("125.00")
        mock_pnl_summary.median_pnl = Decimal("125.00")
        mock_pnl_summary.max_pnl = Decimal("500.00")
        mock_pnl_summary.min_pnl = Decimal("-250.00")
        mock_pnl_summary.std_dev = Decimal("375.00")
        
        mock_trade_stats = Mock(spec=TradeStatistics)
        mock_trade_stats.total_trades = 2
        mock_trade_stats.winning_trades = 1
        mock_trade_stats.losing_trades = 1
        mock_trade_stats.win_rate = 50.0
        mock_trade_stats.average_win = Decimal("500.00")
        mock_trade_stats.average_loss = Decimal("-250.00")
        mock_trade_stats.profit_factor = 2.0
        mock_trade_stats.max_consecutive_wins = 1
        mock_trade_stats.max_consecutive_losses = 1
        
        mock_metrics = Mock(spec=PerformanceMetrics)
        mock_metrics.pnl_summary = mock_pnl_summary
        mock_metrics.trade_statistics = mock_trade_stats
        mock_metrics.advanced_statistics = None
        
        engine.calculate_metrics_for_strategy.return_value = mock_metrics
        engine.calculate_pnl_summary.return_value = mock_pnl_summary
        engine.calculate_trade_statistics.return_value = mock_trade_stats
        
        return engine
    
    @pytest.fixture
    def export_manager(self, mock_strategy_manager, mock_analytics_engine, temp_export_dir):
        """Create export manager instance."""
        return ExportManager(mock_strategy_manager, mock_analytics_engine, temp_export_dir)
    
    def test_initialization(self, export_manager, temp_export_dir):
        """Test export manager initialization."""
        assert export_manager.export_dir == Path(temp_export_dir)
        
        # Check subdirectories are created
        assert (Path(temp_export_dir) / "csv").exists()
        assert (Path(temp_export_dir) / "excel").exists()
        assert (Path(temp_export_dir) / "pdf").exists()
    
    def test_export_trades_to_csv_single_strategy(self, export_manager):
        """Test CSV export for single strategy."""
        config = ExportConfig(
            include_trades=True,
            delimiter=",",
            decimal_places=2,
            include_headers=True
        )
        
        filepath = export_manager.export_trades_to_csv(strategy_id=1, config=config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.csv')
        
        # Check CSV content
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            assert len(rows) == 2  # Two test trades
            assert 'strategy_name' in rows[0]
            assert 'trade_date' in rows[0]
            assert 'symbol' in rows[0]
            assert 'pnl' in rows[0]
            
            # Check data
            assert rows[0]['symbol'] == 'AAPL'
            assert rows[0]['pnl'] == '500.0'
            assert rows[1]['symbol'] == 'MSFT'
            assert rows[1]['pnl'] == '-250.0'
    
    def test_export_trades_to_csv_all_strategies(self, export_manager):
        """Test CSV export for all strategies."""
        config = ExportConfig()
        
        filepath = export_manager.export_trades_to_csv(strategy_id=None, config=config)
        
        assert Path(filepath).exists()
        assert 'all_trades' in Path(filepath).name
        
        # Verify content
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            # Should have trades from both strategies (2 each = 4 total)
            assert len(rows) >= 2
    
    def test_export_trades_to_csv_custom_columns(self, export_manager):
        """Test CSV export with custom columns."""
        config = ExportConfig(
            custom_columns=['trade_date', 'symbol', 'pnl'],
            include_headers=True
        )
        
        filepath = export_manager.export_trades_to_csv(strategy_id=1, config=config)
        
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            # Should only have the specified columns
            assert set(rows[0].keys()) == {'trade_date', 'symbol', 'pnl'}
    
    def test_export_trades_to_csv_custom_delimiter(self, export_manager):
        """Test CSV export with custom delimiter."""
        config = ExportConfig(delimiter=";")
        
        filepath = export_manager.export_trades_to_csv(strategy_id=1, config=config)
        
        with open(filepath, 'r') as file:
            content = file.read()
            # Should use semicolon delimiter
            assert ';' in content
    
    def test_export_trades_to_csv_no_trades(self, export_manager, mock_analytics_engine):
        """Test CSV export with no trades."""
        mock_analytics_engine.get_trades_for_strategy.return_value = []
        
        with pytest.raises(ValueError, match="No trades found to export"):
            export_manager.export_trades_to_csv(strategy_id=1)
    
    def test_export_metrics_to_csv_single_strategy(self, export_manager):
        """Test metrics CSV export for single strategy."""
        config = ExportConfig()
        
        filepath = export_manager.export_metrics_to_csv(strategy_id=1, config=config)
        
        assert Path(filepath).exists()
        
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            assert len(rows) == 1  # One strategy
            
            row = rows[0]
            assert 'strategy_name' in row
            assert 'total_pnl' in row
            assert 'win_rate' in row
            assert 'total_trades' in row
            
            # Check values
            assert row['strategy_name'] == 'Test Strategy A'
            assert row['total_pnl'] == '250.0'
            assert row['total_trades'] == '2'
            assert row['win_rate'] == '50.0'
    
    def test_export_metrics_to_csv_all_strategies(self, export_manager):
        """Test metrics CSV export for all strategies."""
        filepath = export_manager.export_metrics_to_csv(strategy_id=None)
        
        assert Path(filepath).exists()
        assert 'all_metrics' in Path(filepath).name
        
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            # Should have both strategies
            assert len(rows) == 2
    
    def test_export_metrics_to_csv_no_strategies(self, export_manager, mock_strategy_manager):
        """Test metrics export with no strategies."""
        mock_strategy_manager.get_all_strategies.return_value = []
        
        with pytest.raises(ValueError, match="No strategies found to export"):
            export_manager.export_metrics_to_csv()
    
    @patch('src.exports.export_manager.EXCEL_AVAILABLE', True)
    @patch('src.exports.export_manager.openpyxl')
    def test_export_to_excel_availability_check(self, mock_openpyxl, export_manager):
        """Test Excel export when openpyxl is available."""
        # Mock openpyxl components
        mock_wb = Mock()
        mock_ws = Mock()
        mock_wb.create_sheet.return_value = mock_ws
        mock_openpyxl.Workbook.return_value = mock_wb
        
        config = ExportConfig()
        
        # Should not raise ImportError
        try:
            filepath = export_manager.export_to_excel(strategy_id=1, config=config)
            # File path should be returned (even if mocked)
            assert filepath is not None
        except ImportError:
            pytest.fail("Should not raise ImportError when Excel is available")
    
    def test_export_to_excel_unavailable(self, export_manager):
        """Test Excel export when openpyxl is not available."""
        with patch('src.exports.export_manager.EXCEL_AVAILABLE', False):
            with pytest.raises(ImportError, match="openpyxl is required for Excel export"):
                export_manager.export_to_excel(strategy_id=1)
    
    def test_get_available_formats(self, export_manager):
        """Test getting available export formats."""
        formats = export_manager.get_available_formats()
        
        # CSV should always be available
        assert 'csv' in formats
        
        # Other formats depend on installed packages
        assert isinstance(formats, list)
        assert len(formats) >= 1
    
    def test_cleanup_old_exports(self, export_manager, temp_export_dir):
        """Test cleanup of old export files."""
        # Create some test files
        csv_dir = Path(temp_export_dir) / "csv"
        test_file = csv_dir / "test_export.csv"
        test_file.write_text("test,data\n1,2")
        
        # Modify file time to make it appear old
        old_time = datetime.now().timestamp() - (31 * 24 * 60 * 60)  # 31 days ago
        test_file.touch(times=(old_time, old_time))
        
        assert test_file.exists()
        
        # Cleanup files older than 30 days
        export_manager.cleanup_old_exports(days_old=30)
        
        # File should be deleted
        assert not test_file.exists()
    
    def test_get_export_history(self, export_manager, temp_export_dir):
        """Test getting export history."""
        # Create some test files
        csv_dir = Path(temp_export_dir) / "csv"
        test_file1 = csv_dir / "test1.csv"
        test_file2 = csv_dir / "test2.csv"
        
        test_file1.write_text("test1")
        test_file2.write_text("test2")
        
        history = export_manager.get_export_history()
        
        assert isinstance(history, list)
        assert len(history) == 2
        
        # Check history entry structure
        entry = history[0]
        assert 'filename' in entry
        assert 'format' in entry
        assert 'size_bytes' in entry
        assert 'created' in entry
        assert 'modified' in entry
        assert 'path' in entry
        
        # Check values
        assert entry['format'] == 'csv'
        assert entry['size_bytes'] > 0
        assert isinstance(entry['created'], datetime)
    
    def test_export_config_defaults(self):
        """Test ExportConfig default values."""
        config = ExportConfig()
        
        assert config.include_trades is True
        assert config.include_metrics is True
        assert config.include_monthly_breakdown is True
        assert config.include_confluence is False
        assert config.date_format == "%Y-%m-%d"
        assert config.decimal_places == 2
        assert config.delimiter == ","
        assert config.include_headers is True
        assert config.custom_columns is None
    
    def test_report_template_defaults(self):
        """Test ReportTemplate default values."""
        template = ReportTemplate()
        
        assert template.title == "Trading Strategy Analysis Report"
        assert template.subtitle == ""
        assert template.logo_path is None
        assert template.author == "Trading Strategy Analyzer"
        assert template.include_summary is True
        assert template.include_charts is True
        assert template.include_detailed_metrics is True
        assert template.include_confluence_analysis is False
        assert template.color_scheme == "default"
    
    def test_export_config_custom_values(self):
        """Test ExportConfig with custom values."""
        config = ExportConfig(
            include_trades=False,
            delimiter=";",
            decimal_places=4,
            custom_columns=['symbol', 'pnl']
        )
        
        assert config.include_trades is False
        assert config.delimiter == ";"
        assert config.decimal_places == 4
        assert config.custom_columns == ['symbol', 'pnl']
    
    def test_error_handling_invalid_strategy(self, export_manager, mock_strategy_manager):
        """Test error handling for invalid strategy ID."""
        mock_strategy_manager.get_strategy.return_value = None
        
        # Should handle gracefully and export empty result or raise appropriate error
        with pytest.raises(ValueError):
            export_manager.export_trades_to_csv(strategy_id=999)
    
    def test_date_formatting_in_csv(self, export_manager):
        """Test date formatting in CSV export."""
        config = ExportConfig(date_format="%m/%d/%Y")
        
        filepath = export_manager.export_trades_to_csv(strategy_id=1, config=config)
        
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            # Check date format
            assert rows[0]['trade_date'] == '01/01/2024'
    
    def test_decimal_precision_in_csv(self, export_manager):
        """Test decimal precision in CSV export."""
        config = ExportConfig(decimal_places=0)
        
        filepath = export_manager.export_trades_to_csv(strategy_id=1, config=config)
        
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            # Check decimal precision (should be integers)
            assert rows[0]['pnl'] == '500'
            assert '.' not in rows[0]['entry_price']