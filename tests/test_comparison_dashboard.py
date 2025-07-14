"""Tests for comparison dashboard components."""

import pytest
from datetime import date, datetime
from decimal import Decimal
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from src.components.comparison_dashboard import ComparisonDashboard
from src.models import Strategy, Trade


class TestComparisonDashboard:
    @pytest.fixture
    def mock_strategy_manager(self):
        """Create mock strategy manager."""
        manager = Mock()
        
        # Mock strategies
        strategies = [
            Strategy(id=1, name="Strategy A", description="Test strategy A"),
            Strategy(id=2, name="Strategy B", description="Test strategy B"),
            Strategy(id=3, name="Strategy C", description="Test strategy C")
        ]
        
        manager.get_all_strategies.return_value = strategies
        manager.get_strategy.side_effect = lambda id: next((s for s in strategies if s.id == id), None)
        
        return manager
    
    @pytest.fixture
    def mock_analytics_engine(self):
        """Create mock analytics engine."""
        engine = Mock()
        
        # Mock trades data
        trades_data = {
            1: [  # Strategy A trades
                Trade(strategy_id=1, trade_date=date(2024, 1, 1), symbol="AAPL", side="buy", pnl=Decimal("100")),
                Trade(strategy_id=1, trade_date=date(2024, 1, 2), symbol="MSFT", side="sell", pnl=Decimal("-50")),
                Trade(strategy_id=1, trade_date=date(2024, 1, 3), symbol="GOOGL", side="buy", pnl=Decimal("75"))
            ],
            2: [  # Strategy B trades
                Trade(strategy_id=2, trade_date=date(2024, 1, 1), symbol="TSLA", side="buy", pnl=Decimal("200")),
                Trade(strategy_id=2, trade_date=date(2024, 1, 2), symbol="NVDA", side="sell", pnl=Decimal("-100")),
                Trade(strategy_id=2, trade_date=date(2024, 1, 4), symbol="AMZN", side="buy", pnl=Decimal("150"))
            ],
            3: [  # Strategy C trades
                Trade(strategy_id=3, trade_date=date(2024, 1, 1), symbol="META", side="buy", pnl=Decimal("50")),
                Trade(strategy_id=3, trade_date=date(2024, 1, 3), symbol="NFLX", side="sell", pnl=Decimal("-25")),
                Trade(strategy_id=3, trade_date=date(2024, 1, 5), symbol="ORCL", side="buy", pnl=Decimal("100"))
            ]
        }
        
        engine.get_trades_for_strategy.side_effect = lambda id: trades_data.get(id, [])
        
        # Mock metrics
        mock_metrics = Mock()
        mock_metrics.pnl_summary.total_pnl = 125
        mock_metrics.trade_statistics.win_rate = 66.7
        mock_metrics.trade_statistics.total_trades = 3
        mock_metrics.trade_statistics.winning_trades = 2
        mock_metrics.trade_statistics.average_win = 87.5
        mock_metrics.trade_statistics.average_loss = -50
        mock_metrics.trade_statistics.profit_factor = 1.75
        mock_metrics.trade_statistics.max_consecutive_wins = 1
        mock_metrics.trade_statistics.max_consecutive_losses = 1
        mock_metrics.advanced_statistics.sharpe_ratio = 1.5
        mock_metrics.advanced_statistics.sortino_ratio = 2.0
        mock_metrics.advanced_statistics.max_drawdown = 0.1
        mock_metrics.advanced_statistics.calmar_ratio = 1.2
        
        engine.calculate_metrics_for_strategy.return_value = mock_metrics
        
        return engine
    
    @pytest.fixture
    def mock_viz(self):
        """Create mock visualization components."""
        viz = Mock()
        viz.theme = "dark"
        viz.bg_color = "#0E1117"
        viz.text_color = "#FAFAFA"
        viz.positive_color = "#00CC88"
        viz.negative_color = "#FF4444"
        viz.grid_color = "#333333"
        viz.layout_template = {"paper_bgcolor": "#0E1117", "plot_bgcolor": "#0E1117"}
        
        # Mock chart creation
        mock_fig = Mock()
        mock_fig.layout = Mock()
        mock_fig.data = [Mock()]
        viz._create_empty_chart.return_value = mock_fig
        
        return viz
    
    @pytest.fixture
    def comparison_dashboard(self, mock_strategy_manager, mock_analytics_engine, mock_viz):
        """Create comparison dashboard instance."""
        return ComparisonDashboard(
            strategy_manager=mock_strategy_manager,
            analytics_engine=mock_analytics_engine,
            viz_components=mock_viz
        )
    
    def test_initialization(self, comparison_dashboard, mock_strategy_manager, mock_analytics_engine, mock_viz):
        """Test dashboard initialization."""
        assert comparison_dashboard.strategy_manager == mock_strategy_manager
        assert comparison_dashboard.analytics_engine == mock_analytics_engine
        assert comparison_dashboard.viz == mock_viz
    
    def test_build_comparison_table(self, comparison_dashboard):
        """Test comparison table building."""
        strategy_ids = [1, 2, 3]
        df = comparison_dashboard.build_comparison_table(strategy_ids)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Three strategies
        
        # Check required columns
        expected_columns = [
            'Strategy', 'Total P&L', 'Win Rate', 'Total Trades',
            'Avg Win', 'Avg Loss', 'Profit Factor', 'Sharpe Ratio',
            'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio'
        ]
        
        for col in expected_columns:
            assert col in df.columns
        
        # Check ranking columns
        assert 'Total P&L_rank' in df.columns
        assert 'Win Rate_rank' in df.columns
        assert 'Max Drawdown_rank' in df.columns
    
    def test_build_comparison_table_empty(self, comparison_dashboard):
        """Test comparison table with empty strategy list."""
        df = comparison_dashboard.build_comparison_table([])
        assert df.empty
    
    def test_calculate_correlation_matrix(self, comparison_dashboard):
        """Test correlation matrix calculation."""
        strategy_ids = [1, 2, 3]
        correlation_matrix, returns_df = comparison_dashboard.calculate_correlation_matrix(strategy_ids)
        
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert isinstance(returns_df, pd.DataFrame)
        
        if not correlation_matrix.empty:
            # Check matrix properties
            assert correlation_matrix.shape[0] == correlation_matrix.shape[1]  # Square matrix
            assert len(correlation_matrix) <= len(strategy_ids)  # At most as many as strategies
            
            # Check diagonal is 1 (or close to it due to float precision)
            diag_values = np.diag(correlation_matrix.values)
            assert all(abs(val - 1.0) < 0.01 for val in diag_values)
    
    def test_calculate_correlation_matrix_insufficient_data(self, comparison_dashboard):
        """Test correlation matrix with insufficient data."""
        # Test with single strategy
        correlation_matrix, returns_df = comparison_dashboard.calculate_correlation_matrix([1])
        assert correlation_matrix.empty
        assert returns_df.empty
        
        # Test with empty list
        correlation_matrix, returns_df = comparison_dashboard.calculate_correlation_matrix([])
        assert correlation_matrix.empty
        assert returns_df.empty
    
    def test_render_correlation_heatmap(self, comparison_dashboard):
        """Test correlation heatmap rendering."""
        # Test with valid correlation matrix
        correlation_data = pd.DataFrame({
            'Strategy A': [1.0, 0.5, -0.2],
            'Strategy B': [0.5, 1.0, 0.3],
            'Strategy C': [-0.2, 0.3, 1.0]
        }, index=['Strategy A', 'Strategy B', 'Strategy C'])
        
        fig = comparison_dashboard.render_correlation_heatmap(correlation_data)
        assert fig is not None
        
        # Test with empty matrix
        empty_matrix = pd.DataFrame()
        fig_empty = comparison_dashboard.render_correlation_heatmap(empty_matrix)
        assert fig_empty is not None
    
    def test_render_relative_performance(self, comparison_dashboard):
        """Test relative performance chart rendering."""
        strategy_ids = [1, 2, 3]
        initial_capital = 10000
        
        fig = comparison_dashboard.render_relative_performance(strategy_ids, initial_capital)
        assert fig is not None
    
    def test_calculate_portfolio_optimization_success(self, comparison_dashboard):
        """Test successful portfolio optimization."""
        strategy_ids = [1, 2, 3]
        
        # Mock correlation matrix calculation to return valid data
        returns_data = pd.DataFrame({
            'Strategy A': [100, -50, 75, 0, 0],
            'Strategy B': [200, -100, 0, 150, 0],
            'Strategy C': [50, 0, -25, 0, 100]
        })
        
        comparison_dashboard.calculate_correlation_matrix = Mock(return_value=(
            returns_data.corr(),
            returns_data
        ))
        
        result = comparison_dashboard.calculate_portfolio_optimization(strategy_ids)
        
        assert result['status'] == 'success'
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'expected_volatility' in result
        assert 'sharpe_ratio' in result
        
        # Check weights sum to 1
        weights_sum = sum(result['weights'].values())
        assert abs(weights_sum - 1.0) < 0.01
    
    def test_calculate_portfolio_optimization_insufficient_data(self, comparison_dashboard):
        """Test portfolio optimization with insufficient data."""
        strategy_ids = [1]
        
        # Mock to return empty data
        comparison_dashboard.calculate_correlation_matrix = Mock(return_value=(
            pd.DataFrame(),
            pd.DataFrame()
        ))
        
        result = comparison_dashboard.calculate_portfolio_optimization(strategy_ids)
        assert result['status'] == 'error'
        assert 'message' in result
    
    def test_render_portfolio_weights_success(self, comparison_dashboard):
        """Test portfolio weights visualization with successful optimization."""
        optimization_results = {
            'status': 'success',
            'weights': {'Strategy A': 0.4, 'Strategy B': 0.35, 'Strategy C': 0.25},
            'sharpe_ratio': 1.5
        }
        
        fig = comparison_dashboard.render_portfolio_weights(optimization_results)
        assert fig is not None
    
    def test_render_portfolio_weights_failure(self, comparison_dashboard):
        """Test portfolio weights visualization with failed optimization."""
        optimization_results = {
            'status': 'error',
            'message': 'Optimization failed'
        }
        
        fig = comparison_dashboard.render_portfolio_weights(optimization_results)
        assert fig is not None
    
    def test_calculate_portfolio_metrics(self, comparison_dashboard):
        """Test portfolio-level metrics calculation."""
        strategy_ids = [1, 2]
        weights = {'Strategy A': 0.6, 'Strategy B': 0.4}
        
        # Mock additional methods needed
        comparison_dashboard.analytics_engine.calculate_pnl_summary = Mock(return_value=Mock(total_pnl=150))
        comparison_dashboard.analytics_engine.calculate_trade_statistics = Mock(return_value=Mock(win_rate=70))
        comparison_dashboard.analytics_engine.calculate_advanced_metrics = Mock(return_value={
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.08
        })
        
        result = comparison_dashboard.calculate_portfolio_metrics(strategy_ids, weights)
        
        assert 'total_pnl' in result
        assert 'win_rate' in result
        assert 'weights' in result
        assert result['weights']['Strategy A'] == 0.6
        assert result['weights']['Strategy B'] == 0.4
    
    def test_calculate_portfolio_metrics_invalid_weights(self, comparison_dashboard):
        """Test portfolio metrics with invalid weights."""
        strategy_ids = [1, 2]
        weights = {}  # Empty weights
        
        result = comparison_dashboard.calculate_portfolio_metrics(strategy_ids, weights)
        assert 'error' in result
    
    def test_render_performance_ranking(self, comparison_dashboard):
        """Test performance ranking visualization."""
        comparison_df = pd.DataFrame({
            'Strategy': ['Strategy A', 'Strategy B', 'Strategy C'],
            'Total P&L': [125, 250, 125],
            'Win Rate': [66.7, 60.0, 66.7],
            'Max Drawdown': [0.1, 0.15, 0.08]
        })
        
        # Test different sort metrics
        for sort_by in ['Total P&L', 'Win Rate', 'Max Drawdown']:
            fig = comparison_dashboard.render_performance_ranking(comparison_df, sort_by)
            assert fig is not None
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        fig_empty = comparison_dashboard.render_performance_ranking(empty_df)
        assert fig_empty is not None
    
    def test_error_handling(self, comparison_dashboard):
        """Test error handling in various methods."""
        # Test with non-existent strategy IDs
        invalid_ids = [999, 1000]
        
        df = comparison_dashboard.build_comparison_table(invalid_ids)
        assert df.empty
        
        correlation_matrix, returns_df = comparison_dashboard.calculate_correlation_matrix(invalid_ids)
        assert correlation_matrix.empty
        assert returns_df.empty
        
        fig = comparison_dashboard.render_relative_performance(invalid_ids)
        assert fig is not None