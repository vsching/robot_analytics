"""Tests for advanced performance metrics calculations."""

import pytest
import numpy as np
from decimal import Decimal
from datetime import date
import warnings

from src.analytics.advanced_metrics import AdvancedMetrics
from src.analytics import AnalyticsEngine
from src.models import Trade


class TestAdvancedMetrics:
    @pytest.fixture
    def advanced_metrics(self):
        """Create an AdvancedMetrics instance."""
        return AdvancedMetrics(annual_trading_days=252)
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample daily returns."""
        # Simulate 100 days of returns with some volatility
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% mean, 2% std dev
        return returns
    
    @pytest.fixture
    def positive_returns(self):
        """Create consistently positive returns."""
        return np.array([0.01, 0.02, 0.015, 0.03, 0.025, 0.01, 0.02, 0.015] * 10)
    
    @pytest.fixture
    def negative_returns(self):
        """Create consistently negative returns."""
        return np.array([-0.01, -0.02, -0.015, -0.03, -0.025, -0.01, -0.02, -0.015] * 10)
    
    @pytest.fixture
    def mixed_returns(self):
        """Create mixed positive and negative returns."""
        return np.array([0.02, -0.01, 0.03, -0.025, 0.015, -0.02, 0.04, -0.01] * 10)
    
    def test_sharpe_ratio_calculation(self, advanced_metrics, sample_returns):
        """Test Sharpe ratio calculation with various scenarios."""
        # Test with normal returns
        sharpe = advanced_metrics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        assert sharpe is not None
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range for Sharpe ratio
        
        # Test with all positive returns
        positive_returns = np.array([0.01] * 100)
        sharpe_positive = advanced_metrics.calculate_sharpe_ratio(positive_returns)
        assert sharpe_positive > 0
        
        # Test with zero volatility
        zero_vol_returns = np.array([0.001] * 100)
        sharpe_zero_vol = advanced_metrics.calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero_vol == float('inf')
        
        # Test with insufficient data
        short_returns = np.array([0.01])
        sharpe_short = advanced_metrics.calculate_sharpe_ratio(short_returns)
        assert sharpe_short is None
    
    def test_sharpe_ratio_frequencies(self, advanced_metrics):
        """Test Sharpe ratio with different frequencies."""
        daily_returns = np.random.normal(0.001, 0.02, 252)
        
        # Daily frequency
        sharpe_daily = advanced_metrics.calculate_sharpe_ratio(
            daily_returns, frequency='daily'
        )
        
        # Weekly frequency (aggregate to weekly)
        weekly_returns = daily_returns.reshape(-1, 5).mean(axis=1)
        sharpe_weekly = advanced_metrics.calculate_sharpe_ratio(
            weekly_returns, frequency='weekly'
        )
        
        # Monthly frequency (aggregate to monthly)
        monthly_returns = daily_returns.reshape(-1, 21).mean(axis=1)
        sharpe_monthly = advanced_metrics.calculate_sharpe_ratio(
            monthly_returns, frequency='monthly'
        )
        
        assert all(x is not None for x in [sharpe_daily, sharpe_weekly, sharpe_monthly])
    
    def test_sortino_ratio_calculation(self, advanced_metrics, mixed_returns):
        """Test Sortino ratio calculation."""
        sortino = advanced_metrics.calculate_sortino_ratio(mixed_returns)
        assert sortino is not None
        assert isinstance(sortino, float)
        
        # Test with all positive returns (no downside)
        positive_returns = np.array([0.01, 0.02, 0.03] * 20)
        sortino_positive = advanced_metrics.calculate_sortino_ratio(positive_returns)
        assert sortino_positive == float('inf')
        
        # Test with custom target return
        sortino_target = advanced_metrics.calculate_sortino_ratio(
            mixed_returns, target_return=0.1  # 10% annual target
        )
        assert sortino_target is not None
    
    def test_profit_factor_calculation(self, advanced_metrics):
        """Test profit factor calculation."""
        # Mixed P&L
        pnl_values = [
            Decimal('100'), Decimal('-50'), Decimal('200'), 
            Decimal('-75'), Decimal('150'), Decimal('-25')
        ]
        pf = advanced_metrics.calculate_profit_factor(pnl_values)
        assert pf == 450 / 150  # 3.0
        
        # All profits
        all_profits = [Decimal('100'), Decimal('200'), Decimal('150')]
        pf_all_profit = advanced_metrics.calculate_profit_factor(all_profits)
        assert pf_all_profit == float('inf')
        
        # All losses
        all_losses = [Decimal('-100'), Decimal('-200'), Decimal('-150')]
        pf_all_loss = advanced_metrics.calculate_profit_factor(all_losses)
        assert pf_all_loss == 0.0
        
        # Empty list
        pf_empty = advanced_metrics.calculate_profit_factor([])
        assert pf_empty is None
    
    def test_max_drawdown_calculation(self, advanced_metrics):
        """Test maximum drawdown calculation."""
        # Create equity curve with known drawdown
        equity_curve = np.array([
            10000, 10500, 11000, 10800, 10200,  # Drawdown from 11000 to 10200
            10400, 10600, 11200, 11500, 11300   # Recovery to new high
        ])
        
        max_dd, details = advanced_metrics.calculate_max_drawdown(equity_curve)
        
        # Expected drawdown: (11000 - 10200) / 11000 = 0.0727
        assert abs(max_dd - 0.0727) < 0.001
        assert details['peak_index'] == 2
        assert details['trough_index'] == 4
        assert details['peak_value'] == 11000
        assert details['trough_value'] == 10200
        assert details['drawdown_duration'] == 2
        assert details['recovery_index'] == 7  # When it exceeds 11000
        
        # Test with no drawdown
        no_dd_curve = np.array([10000, 10100, 10200, 10300, 10400])
        max_dd_none, _ = advanced_metrics.calculate_max_drawdown(no_dd_curve)
        assert max_dd_none == 0.0
    
    def test_calmar_ratio_calculation(self, advanced_metrics, sample_returns):
        """Test Calmar ratio calculation."""
        max_dd = 0.15  # 15% drawdown
        
        calmar = advanced_metrics.calculate_calmar_ratio(
            sample_returns, max_dd, frequency='daily'
        )
        assert calmar is not None
        assert isinstance(calmar, float)
        
        # Test with zero drawdown
        calmar_zero_dd = advanced_metrics.calculate_calmar_ratio(
            sample_returns, 0.0
        )
        assert calmar_zero_dd is None
    
    def test_value_at_risk_calculation(self, advanced_metrics, sample_returns):
        """Test VaR calculation."""
        # Historical VaR
        var_hist = advanced_metrics.calculate_value_at_risk(
            sample_returns, confidence_level=0.95, method='historical'
        )
        assert var_hist is not None
        assert var_hist > 0  # VaR is expressed as positive loss
        
        # Parametric VaR
        var_param = advanced_metrics.calculate_value_at_risk(
            sample_returns, confidence_level=0.95, method='parametric'
        )
        assert var_param is not None
        assert var_param > 0
        
        # Test with insufficient data
        short_returns = np.array([0.01] * 10)
        var_short = advanced_metrics.calculate_value_at_risk(short_returns)
        assert var_short is None
    
    def test_conditional_var_calculation(self, advanced_metrics, sample_returns):
        """Test CVaR (Expected Shortfall) calculation."""
        cvar = advanced_metrics.calculate_conditional_var(
            sample_returns, confidence_level=0.95
        )
        assert cvar is not None
        assert cvar > 0
        
        # CVaR should be greater than or equal to VaR
        var = advanced_metrics.calculate_value_at_risk(
            sample_returns, confidence_level=0.95
        )
        assert cvar >= var
    
    def test_information_ratio_calculation(self, advanced_metrics):
        """Test information ratio calculation."""
        strategy_returns = np.random.normal(0.002, 0.02, 100)
        benchmark_returns = np.random.normal(0.001, 0.015, 100)
        
        ir = advanced_metrics.calculate_information_ratio(
            strategy_returns, benchmark_returns
        )
        assert ir is not None
        assert isinstance(ir, float)
        
        # Test with identical returns (no active return)
        identical_returns = np.array([0.01] * 100)
        ir_zero = advanced_metrics.calculate_information_ratio(
            identical_returns, identical_returns
        )
        assert ir_zero == 0 or ir_zero == float('inf')
    
    def test_omega_ratio_calculation(self, advanced_metrics, mixed_returns):
        """Test Omega ratio calculation."""
        omega = advanced_metrics.calculate_omega_ratio(mixed_returns)
        assert omega is not None
        assert omega > 0
        
        # Test with all positive returns
        positive_returns = np.array([0.01, 0.02, 0.03] * 20)
        omega_positive = advanced_metrics.calculate_omega_ratio(positive_returns)
        assert omega_positive == float('inf')
        
        # Test with custom threshold
        omega_threshold = advanced_metrics.calculate_omega_ratio(
            mixed_returns, threshold=0.01
        )
        assert omega_threshold is not None
    
    def test_data_validation(self, advanced_metrics):
        """Test data validation functionality."""
        # Valid data
        valid_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        is_valid, error = advanced_metrics.validate_data(valid_returns, min_points=3)
        assert is_valid is True
        assert error is None
        
        # Insufficient data
        short_data = np.array([0.01])
        is_valid, error = advanced_metrics.validate_data(short_data, min_points=5)
        assert is_valid is False
        assert "Insufficient" in error
        
        # Contains NaN
        nan_data = np.array([0.01, np.nan, 0.02])
        is_valid, error = advanced_metrics.validate_data(nan_data, min_points=3)
        assert is_valid is False
        assert "Insufficient" in error  # After removing NaN
        
        # Contains infinity
        inf_data = np.array([0.01, np.inf, 0.02])
        is_valid, error = advanced_metrics.validate_data(inf_data)
        assert is_valid is False
        assert "infinite" in error
        
        # Invalid price data
        negative_prices = np.array([100, 95, -10, 105])
        is_valid, error = advanced_metrics.validate_data(
            negative_prices, data_type='prices'
        )
        assert is_valid is False
        assert "positive" in error


class TestAnalyticsEngineAdvanced:
    @pytest.fixture
    def analytics_engine(self):
        """Create an AnalyticsEngine instance."""
        return AnalyticsEngine()
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        trades = []
        dates = [date(2024, 1, i) for i in range(1, 21)]
        
        # Mix of winning and losing trades
        pnl_values = [100, -50, 200, -75, 150, -25, 300, -100, 250, -80,
                      180, -60, 220, -40, 160, -90, 280, -70, 190, -30]
        
        for i, (trade_date, pnl) in enumerate(zip(dates, pnl_values)):
            trade = Trade(
                strategy_id=1,
                trade_date=trade_date,
                symbol=f"STOCK{i}",
                side="buy" if i % 2 == 0 else "sell",
                entry_price=Decimal("100"),
                exit_price=Decimal(str(100 + pnl/10)),
                quantity=Decimal("10"),
                pnl=Decimal(str(pnl)),
                commission=Decimal("5")
            )
            trades.append(trade)
        
        return trades
    
    def test_calculate_returns_from_trades(self, analytics_engine, sample_trades):
        """Test return calculation from trades."""
        returns = analytics_engine.calculate_returns_from_trades(sample_trades)
        
        assert len(returns) > 0
        assert isinstance(returns, np.ndarray)
        assert all(isinstance(r, (int, float)) for r in returns)
        
        # Test with no trades
        empty_returns = analytics_engine.calculate_returns_from_trades([])
        assert len(empty_returns) == 0
    
    def test_calculate_equity_curve(self, analytics_engine, sample_trades):
        """Test equity curve calculation."""
        equity_curve = analytics_engine.calculate_equity_curve(sample_trades)
        
        assert len(equity_curve) > 0
        assert equity_curve[0] == 10000  # Initial capital
        assert all(isinstance(v, (int, float)) for v in equity_curve)
        
        # Verify cumulative nature
        total_pnl = sum(t.pnl for t in sample_trades if t.pnl is not None)
        expected_final = 10000 + float(total_pnl)
        assert abs(equity_curve[-1] - expected_final) < 0.01
    
    def test_calculate_advanced_metrics_integration(self, analytics_engine, sample_trades):
        """Test full advanced metrics calculation."""
        metrics = analytics_engine.calculate_advanced_metrics(sample_trades)
        
        # Check all metrics are present
        expected_keys = [
            'sharpe_ratio', 'sortino_ratio', 'profit_factor', 
            'max_drawdown', 'max_drawdown_details', 'calmar_ratio',
            'value_at_risk_95', 'conditional_var_95', 'omega_ratio',
            'total_trades', 'returns_count'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Verify metric values are reasonable
        assert metrics['total_trades'] == len(sample_trades)
        assert metrics['profit_factor'] > 0
        assert 0 <= metrics['max_drawdown'] <= 1
        
        # Check drawdown details
        dd_details = metrics['max_drawdown_details']
        assert 'peak_index' in dd_details
        assert 'trough_index' in dd_details
        assert 'drawdown_duration' in dd_details
    
    def test_advanced_metrics_edge_cases(self, analytics_engine):
        """Test advanced metrics with edge cases."""
        # All winning trades
        winning_trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, i),
                symbol="WIN",
                side="buy",
                pnl=Decimal("100")
            )
            for i in range(1, 11)
        ]
        
        win_metrics = analytics_engine.calculate_advanced_metrics(winning_trades)
        assert win_metrics['profit_factor'] == float('inf')
        assert win_metrics['max_drawdown'] == 0.0
        assert win_metrics['sortino_ratio'] == float('inf')
        
        # Single trade
        single_trade = [winning_trades[0]]
        single_metrics = analytics_engine.calculate_advanced_metrics(single_trade)
        assert single_metrics['total_trades'] == 1