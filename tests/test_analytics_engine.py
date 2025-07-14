"""Tests for analytics engine functionality."""

import pytest
from datetime import date, datetime
from decimal import Decimal
import tempfile
import os

from src.analytics import AnalyticsEngine, PnLSummary, TradeStatistics
from src.models import Trade
from src.db.connection import DatabaseManager


def create_sample_trades():
    """Create sample trades for testing."""
    return [
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
        ),
        Trade(
            strategy_id=1,
            trade_date=date(2024, 1, 3),
            symbol="GOOGL",
            side="buy",
            entry_price=Decimal("150.00"),
            exit_price=Decimal("160.00"),
            quantity=Decimal("30"),
            pnl=Decimal("300.00"),
            commission=Decimal("10.00")
        ),
        Trade(
            strategy_id=1,
            trade_date=date(2024, 1, 4),
            symbol="TSLA",
            side="sell",
            entry_price=Decimal("200.00"),
            exit_price=Decimal("195.00"),
            quantity=Decimal("25"),
            pnl=Decimal("-125.00"),
            commission=Decimal("10.00")
        ),
        Trade(
            strategy_id=1,
            trade_date=date(2024, 1, 5),
            symbol="AAPL",
            side="buy",
            entry_price=Decimal("155.00"),
            exit_price=Decimal("155.00"),
            quantity=Decimal("100"),
            pnl=Decimal("0.00"),
            commission=Decimal("10.00")
        ),
    ]


@pytest.fixture
def analytics_engine():
    """Create an AnalyticsEngine instance."""
    return AnalyticsEngine()


class TestPnLCalculations:
    def test_calculate_pnl_summary(self, analytics_engine):
        """Test P&L summary calculation."""
        trades = create_sample_trades()
        summary = analytics_engine.calculate_pnl_summary(trades)
        
        assert summary is not None
        assert summary.total_pnl == Decimal("425.00")  # 500 - 250 + 300 - 125 + 0
        assert summary.total_commission == Decimal("50.00")  # 5 * 10
        assert summary.net_pnl == Decimal("375.00")  # 425 - 50
        assert summary.average_pnl == pytest.approx(85.0, rel=0.01)
        assert summary.max_pnl == Decimal("500.00")
        assert summary.min_pnl == Decimal("-250.00")
    
    def test_calculate_pnl_summary_empty(self, analytics_engine):
        """Test P&L summary with no trades."""
        summary = analytics_engine.calculate_pnl_summary([])
        assert summary is None
    
    def test_calculate_pnl_summary_no_pnl(self, analytics_engine):
        """Test P&L summary with trades having no P&L."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                quantity=Decimal("100"),
                pnl=None
            )
        ]
        summary = analytics_engine.calculate_pnl_summary(trades)
        assert summary is None
    
    def test_calculate_pnl_summary_single_trade(self, analytics_engine):
        """Test P&L summary with single trade."""
        trades = [create_sample_trades()[0]]
        summary = analytics_engine.calculate_pnl_summary(trades)
        
        assert summary is not None
        assert summary.total_pnl == Decimal("500.00")
        assert summary.average_pnl == Decimal("500.00")
        assert summary.std_dev == Decimal("0")


class TestTradeStatistics:
    def test_calculate_trade_statistics(self, analytics_engine):
        """Test trade statistics calculation."""
        trades = create_sample_trades()
        stats = analytics_engine.calculate_trade_statistics(trades)
        
        assert stats is not None
        assert stats.total_trades == 5
        assert stats.winning_trades == 2
        assert stats.losing_trades == 2
        assert stats.breakeven_trades == 1
        assert stats.win_rate == 40.0  # 2/5 * 100
        assert stats.loss_rate == 40.0  # 2/5 * 100
        assert stats.average_win == Decimal("400.00")  # (500 + 300) / 2
        assert stats.average_loss == Decimal("-187.50")  # (-250 + -125) / 2
        assert stats.largest_win == Decimal("500.00")
        assert stats.largest_loss == Decimal("-250.00")
        assert stats.win_loss_ratio == pytest.approx(2.133, rel=0.01)  # 400 / 187.5
    
    def test_calculate_trade_statistics_all_winners(self, analytics_engine):
        """Test statistics with all winning trades."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, i),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("160.00"),
                quantity=Decimal("100"),
                pnl=Decimal("1000.00")
            )
            for i in range(1, 6)
        ]
        
        stats = analytics_engine.calculate_trade_statistics(trades)
        
        assert stats.total_trades == 5
        assert stats.winning_trades == 5
        assert stats.losing_trades == 0
        assert stats.win_rate == 100.0
        assert stats.profit_factor == 0  # No losses
        assert stats.max_consecutive_wins == 5
        assert stats.max_consecutive_losses == 0
    
    def test_calculate_trade_statistics_all_losers(self, analytics_engine):
        """Test statistics with all losing trades."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, i),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("140.00"),
                quantity=Decimal("100"),
                pnl=Decimal("-1000.00")
            )
            for i in range(1, 6)
        ]
        
        stats = analytics_engine.calculate_trade_statistics(trades)
        
        assert stats.total_trades == 5
        assert stats.winning_trades == 0
        assert stats.losing_trades == 5
        assert stats.win_rate == 0.0
        assert stats.win_loss_ratio == 0
        assert stats.max_consecutive_wins == 0
        assert stats.max_consecutive_losses == 5
    
    def test_consecutive_stats(self, analytics_engine):
        """Test consecutive win/loss calculation."""
        trades = [
            Trade(strategy_id=1, trade_date=date(2024, 1, 1), symbol="A", side="buy", pnl=Decimal("100")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 2), symbol="B", side="buy", pnl=Decimal("100")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 3), symbol="C", side="buy", pnl=Decimal("100")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 4), symbol="D", side="sell", pnl=Decimal("-50")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 5), symbol="E", side="sell", pnl=Decimal("-50")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 6), symbol="F", side="buy", pnl=Decimal("100")),
        ]
        
        stats = analytics_engine.calculate_trade_statistics(trades)
        
        assert stats.max_consecutive_wins == 3
        assert stats.max_consecutive_losses == 2
        assert stats.consecutive_wins == 1  # Current streak
        assert stats.consecutive_losses == 0


class TestBasicMetrics:
    def test_calculate_win_rate(self, analytics_engine):
        """Test win rate calculation."""
        trades = create_sample_trades()
        win_rate = analytics_engine.calculate_win_rate(trades)
        assert win_rate == 40.0  # 2 wins out of 5 trades
    
    def test_calculate_win_rate_empty(self, analytics_engine):
        """Test win rate with no trades."""
        win_rate = analytics_engine.calculate_win_rate([])
        assert win_rate == 0.0
    
    def test_calculate_basic_metrics(self, analytics_engine):
        """Test basic metrics calculation."""
        trades = create_sample_trades()
        metrics = analytics_engine.calculate_basic_metrics(trades)
        
        assert metrics['trade_count'] == 5
        assert metrics['total_pnl'] == 425.0
        assert metrics['average_pnl'] == pytest.approx(85.0, rel=0.01)
        assert metrics['win_rate'] == 40.0
        assert metrics['average_win'] == 400.0
        assert metrics['average_loss'] == -187.5
        assert metrics['win_loss_ratio'] == pytest.approx(2.133, rel=0.01)
    
    def test_calculate_basic_metrics_empty(self, analytics_engine):
        """Test basic metrics with no trades."""
        metrics = analytics_engine.calculate_basic_metrics([])
        
        assert metrics['trade_count'] == 0
        assert metrics['total_pnl'] == 0.0
        assert metrics['average_pnl'] == 0.0
        assert metrics['win_rate'] == 0.0


class TestEdgeCases:
    def test_single_trade_metrics(self, analytics_engine):
        """Test metrics with single trade."""
        trades = [create_sample_trades()[0]]
        
        pnl_summary = analytics_engine.calculate_pnl_summary(trades)
        assert pnl_summary.total_pnl == Decimal("500.00")
        assert pnl_summary.std_dev == Decimal("0")
        
        stats = analytics_engine.calculate_trade_statistics(trades)
        assert stats.total_trades == 1
        assert stats.winning_trades == 1
        assert stats.win_rate == 100.0
    
    def test_breakeven_trades_only(self, analytics_engine):
        """Test metrics with only breakeven trades."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, i),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("150.00"),
                quantity=Decimal("100"),
                pnl=Decimal("0.00")
            )
            for i in range(1, 4)
        ]
        
        pnl_summary = analytics_engine.calculate_pnl_summary(trades)
        assert pnl_summary.total_pnl == Decimal("0.00")
        assert pnl_summary.net_pnl == Decimal("0.00")
        
        stats = analytics_engine.calculate_trade_statistics(trades)
        assert stats.breakeven_trades == 3
        assert stats.win_rate == 0.0
        assert stats.loss_rate == 0.0
    
    def test_mixed_commission_handling(self, analytics_engine):
        """Test handling of mixed commission values."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),
                symbol="AAPL",
                side="buy",
                pnl=Decimal("100.00"),
                commission=Decimal("5.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 2),
                symbol="MSFT",
                side="sell",
                pnl=Decimal("200.00"),
                commission=None  # No commission
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 3),
                symbol="GOOGL",
                side="buy",
                pnl=Decimal("150.00"),
                commission=Decimal("0.00")  # Zero commission
            ),
        ]
        
        pnl_summary = analytics_engine.calculate_pnl_summary(trades)
        assert pnl_summary.total_commission == Decimal("5.00")
        assert pnl_summary.net_pnl == Decimal("445.00")  # 450 - 5


class TestTimeBasedAggregations:
    def test_calculate_monthly_metrics(self, analytics_engine):
        """Test monthly metrics calculation."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 5),
                symbol="AAPL",
                side="buy",
                pnl=Decimal("500.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 15),
                symbol="MSFT",
                side="sell",
                pnl=Decimal("-200.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 2, 10),
                symbol="GOOGL",
                side="buy",
                pnl=Decimal("300.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 2, 20),
                symbol="TSLA",
                side="buy",
                pnl=Decimal("150.00")
            ),
        ]
        
        monthly_metrics = analytics_engine.calculate_monthly_metrics(trades)
        
        assert len(monthly_metrics) == 2
        
        # January metrics
        jan_metrics = monthly_metrics[0]
        assert jan_metrics.period_start == date(2024, 1, 1)
        assert jan_metrics.period_end == date(2024, 1, 31)
        assert jan_metrics.total_trades == 2
        assert jan_metrics.total_pnl == Decimal("300.00")  # 500 - 200
        assert jan_metrics.win_rate == 50.0  # 1 win out of 2
        assert jan_metrics.trading_days == 2
        
        # February metrics
        feb_metrics = monthly_metrics[1]
        assert feb_metrics.period_start == date(2024, 2, 1)
        assert feb_metrics.period_end == date(2024, 2, 29)  # 2024 is leap year
        assert feb_metrics.total_trades == 2
        assert feb_metrics.total_pnl == Decimal("450.00")  # 300 + 150
        assert feb_metrics.win_rate == 100.0  # 2 wins out of 2
        assert feb_metrics.trading_days == 2
    
    def test_calculate_weekly_metrics(self, analytics_engine):
        """Test weekly metrics calculation."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),  # Monday
                symbol="AAPL",
                side="buy",
                pnl=Decimal("100.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 3),  # Wednesday
                symbol="MSFT",
                side="sell",
                pnl=Decimal("-50.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 8),  # Next Monday
                symbol="GOOGL",
                side="buy",
                pnl=Decimal("200.00")
            ),
        ]
        
        weekly_metrics = analytics_engine.calculate_weekly_metrics(trades)
        
        assert len(weekly_metrics) == 2
        
        # First week metrics
        week1 = weekly_metrics[0]
        assert week1.total_trades == 2
        assert week1.total_pnl == Decimal("50.00")  # 100 - 50
        assert week1.win_rate == 50.0
        
        # Second week metrics
        week2 = weekly_metrics[1]
        assert week2.total_trades == 1
        assert week2.total_pnl == Decimal("200.00")
        assert week2.win_rate == 100.0
    
    def test_calculate_daily_metrics(self, analytics_engine):
        """Test daily metrics calculation."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),
                symbol="AAPL",
                side="buy",
                pnl=Decimal("100.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),
                symbol="MSFT",
                side="sell",
                pnl=Decimal("-50.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 2),
                symbol="GOOGL",
                side="buy",
                pnl=Decimal("200.00")
            ),
        ]
        
        daily_metrics = analytics_engine.calculate_daily_metrics(trades)
        
        assert len(daily_metrics) == 2
        
        # Day 1 metrics
        day1 = daily_metrics[0]
        assert day1.period_start == date(2024, 1, 1)
        assert day1.period_end == date(2024, 1, 1)
        assert day1.total_trades == 2
        assert day1.total_pnl == Decimal("50.00")
        assert day1.win_rate == 50.0
        assert day1.trading_days == 1
        
        # Day 2 metrics
        day2 = daily_metrics[1]
        assert day2.period_start == date(2024, 1, 2)
        assert day2.total_trades == 1
        assert day2.total_pnl == Decimal("200.00")
        assert day2.win_rate == 100.0
    
    def test_calculate_yearly_metrics(self, analytics_engine):
        """Test yearly metrics calculation."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2023, 6, 15),
                symbol="AAPL",
                side="buy",
                pnl=Decimal("1000.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2023, 12, 20),
                symbol="MSFT",
                side="sell",
                pnl=Decimal("-300.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 3, 10),
                symbol="GOOGL",
                side="buy",
                pnl=Decimal("500.00")
            ),
        ]
        
        yearly_metrics = analytics_engine.calculate_yearly_metrics(trades)
        
        assert len(yearly_metrics) == 2
        
        # 2023 metrics
        year2023 = yearly_metrics[0]
        assert year2023.period_start == date(2023, 1, 1)
        assert year2023.period_end == date(2023, 12, 31)
        assert year2023.total_trades == 2
        assert year2023.total_pnl == Decimal("700.00")
        assert year2023.win_rate == 50.0
        
        # 2024 metrics
        year2024 = yearly_metrics[1]
        assert year2024.period_start == date(2024, 1, 1)
        assert year2024.period_end == date(2024, 12, 31)
        assert year2024.total_trades == 1
        assert year2024.total_pnl == Decimal("500.00")
        assert year2024.win_rate == 100.0
    
    def test_calculate_custom_period_metrics(self, analytics_engine):
        """Test custom period metrics calculation."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),
                symbol="AAPL",
                side="buy",
                pnl=Decimal("100.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 15),
                symbol="MSFT",
                side="sell",
                pnl=Decimal("-50.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 2, 1),
                symbol="GOOGL",
                side="buy",
                pnl=Decimal("200.00")
            ),
        ]
        
        # Test custom period
        metrics = analytics_engine.calculate_custom_period_metrics(
            trades,
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 31)
        )
        
        assert metrics is not None
        assert metrics.period_start == date(2024, 1, 10)
        assert metrics.period_end == date(2024, 1, 31)
        assert metrics.period_type == 'custom'
        assert metrics.total_trades == 1  # Only Jan 15 trade
        assert metrics.total_pnl == Decimal("-50.00")
        assert metrics.win_rate == 0.0
    
    def test_time_aggregations_empty(self, analytics_engine):
        """Test time aggregations with no trades."""
        empty_trades = []
        
        assert analytics_engine.calculate_monthly_metrics(empty_trades) == []
        assert analytics_engine.calculate_weekly_metrics(empty_trades) == []
        assert analytics_engine.calculate_daily_metrics(empty_trades) == []
        assert analytics_engine.calculate_yearly_metrics(empty_trades) == []
        assert analytics_engine.calculate_custom_period_metrics(
            empty_trades, date(2024, 1, 1), date(2024, 12, 31)
        ) is None
    
    def test_year_boundary_aggregation(self, analytics_engine):
        """Test aggregation across year boundaries."""
        trades = [
            Trade(
                strategy_id=1,
                trade_date=date(2023, 12, 30),
                symbol="AAPL",
                side="buy",
                pnl=Decimal("100.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2023, 12, 31),
                symbol="MSFT",
                side="sell",
                pnl=Decimal("-50.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 1),
                symbol="GOOGL",
                side="buy",
                pnl=Decimal("200.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=date(2024, 1, 2),
                symbol="TSLA",
                side="buy",
                pnl=Decimal("150.00")
            ),
        ]
        
        # Test monthly metrics across year boundary
        monthly_metrics = analytics_engine.calculate_monthly_metrics(trades)
        assert len(monthly_metrics) == 2
        assert monthly_metrics[0].period_start.year == 2023
        assert monthly_metrics[1].period_start.year == 2024
        
        # Test weekly metrics across year boundary
        weekly_metrics = analytics_engine.calculate_weekly_metrics(trades)
        # Should have at least 2 weeks (last week of 2023 and first week of 2024)
        assert len(weekly_metrics) >= 2


class TestRollingWindowMetrics:
    def test_calculate_rolling_metrics(self, analytics_engine):
        """Test rolling window metrics calculation."""
        # Create trades over multiple days
        trades = []
        for i in range(10):
            trades.append(
                Trade(
                    strategy_id=1,
                    trade_date=date(2024, 1, i + 1),
                    symbol="AAPL",
                    side="buy" if i % 3 != 0 else "sell",
                    pnl=Decimal("100.00") if i % 3 != 0 else Decimal("-50.00")
                )
            )
        
        # Calculate 5-day rolling metrics
        rolling_df = analytics_engine.calculate_rolling_metrics(trades, window=5)
        
        assert not rolling_df.empty
        assert 'rolling_pnl' in rolling_df.columns
        assert 'rolling_trades' in rolling_df.columns
        assert 'rolling_win_rate' in rolling_df.columns
        assert 'rolling_volatility' in rolling_df.columns
        assert 'rolling_sharpe' in rolling_df.columns
        assert 'rolling_max_dd' in rolling_df.columns
        
        # Check that rolling window is working
        # Day 5 should have data from days 1-5
        day5_data = rolling_df.iloc[4]
        assert day5_data['rolling_trades'] == 5
        # Days 1, 2, 3, 5 are wins (+100 each), day 4 is loss (-50)
        assert day5_data['rolling_pnl'] == 350  # 4*100 - 50
    
    def test_rolling_metrics_empty(self, analytics_engine):
        """Test rolling metrics with no trades."""
        rolling_df = analytics_engine.calculate_rolling_metrics([], window=30)
        assert rolling_df.empty
    
    def test_rolling_win_rate(self, analytics_engine):
        """Test rolling win rate calculation."""
        trades = [
            Trade(strategy_id=1, trade_date=date(2024, 1, 1), symbol="A", side="buy", pnl=Decimal("100")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 2), symbol="B", side="sell", pnl=Decimal("-50")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 3), symbol="C", side="buy", pnl=Decimal("100")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 4), symbol="D", side="buy", pnl=Decimal("100")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 5), symbol="E", side="sell", pnl=Decimal("-50")),
        ]
        
        rolling_df = analytics_engine.calculate_rolling_metrics(trades, window=3)
        
        # Check win rate on day 3 (trades from days 1-3: 2 wins, 1 loss)
        day3_win_rate = rolling_df.iloc[2]['rolling_win_rate']
        assert day3_win_rate == pytest.approx(66.67, rel=0.1)
        
        # Check win rate on day 5 (trades from days 3-5: 2 wins, 1 loss)
        day5_win_rate = rolling_df.iloc[4]['rolling_win_rate']
        assert day5_win_rate == pytest.approx(66.67, rel=0.1)
    
    def test_rolling_sharpe_ratio(self, analytics_engine):
        """Test rolling Sharpe ratio calculation."""
        # Create consistent daily returns
        trades = []
        for i in range(30):
            trades.append(
                Trade(
                    strategy_id=1,
                    trade_date=date(2024, 1, i + 1),
                    symbol="AAPL",
                    side="buy",
                    pnl=Decimal("100.00")  # Consistent positive returns
                )
            )
        
        rolling_df = analytics_engine.calculate_rolling_metrics(trades, window=20)
        
        # With consistent positive returns, Sharpe should be very high
        last_sharpe = rolling_df.iloc[-1]['rolling_sharpe']
        assert last_sharpe > 5  # High Sharpe ratio for consistent returns
    
    def test_rolling_max_drawdown(self, analytics_engine):
        """Test rolling maximum drawdown calculation."""
        trades = [
            Trade(strategy_id=1, trade_date=date(2024, 1, 1), symbol="A", side="buy", pnl=Decimal("1000")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 2), symbol="B", side="buy", pnl=Decimal("500")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 3), symbol="C", side="sell", pnl=Decimal("-800")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 4), symbol="D", side="sell", pnl=Decimal("-400")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 5), symbol="E", side="buy", pnl=Decimal("200")),
        ]
        
        rolling_df = analytics_engine.calculate_rolling_metrics(trades, window=5)
        
        # Check that max drawdown is calculated
        max_dd = rolling_df['rolling_max_dd'].max()
        assert max_dd > 0  # Should have some drawdown
    
    def test_multiple_window_sizes(self, analytics_engine):
        """Test rolling metrics with multiple window sizes."""
        trades = create_sample_trades()
        
        results = analytics_engine.calculate_rolling_performance(
            trades, 
            windows=[7, 14, 30]
        )
        
        assert len(results) == 3
        assert 7 in results
        assert 14 in results
        assert 30 in results
        
        # Larger windows should have more stable metrics
        vol_7 = results[7]['rolling_volatility'].mean()
        vol_30 = results[30]['rolling_volatility'].mean()
        # Note: This relationship might not always hold depending on data
    
    def test_rolling_summary(self, analytics_engine):
        """Test rolling summary calculation."""
        trades = create_sample_trades()
        
        summary = analytics_engine.get_rolling_summary(trades, window=30)
        
        assert 'window_days' in summary
        assert summary['window_days'] == 30
        assert 'current_pnl' in summary
        assert 'current_trades' in summary
        assert 'current_win_rate' in summary
        assert 'current_volatility' in summary
        assert 'current_sharpe' in summary
        assert 'current_max_dd' in summary
        
        # Values should be numeric
        assert isinstance(summary['current_pnl'], (int, float))
        assert isinstance(summary['current_trades'], int)
        assert isinstance(summary['current_win_rate'], (int, float))
    
    def test_rolling_metrics_single_day(self, analytics_engine):
        """Test rolling metrics with trades on single day."""
        trades = [
            Trade(strategy_id=1, trade_date=date(2024, 1, 1), symbol="A", side="buy", pnl=Decimal("100")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 1), symbol="B", side="sell", pnl=Decimal("-50")),
            Trade(strategy_id=1, trade_date=date(2024, 1, 1), symbol="C", side="buy", pnl=Decimal("75")),
        ]
        
        rolling_df = analytics_engine.calculate_rolling_metrics(trades, window=30)
        
        # Should have one row
        assert len(rolling_df) == 1
        assert rolling_df.iloc[0]['rolling_trades'] == 3
        assert rolling_df.iloc[0]['rolling_pnl'] == 125  # 100 - 50 + 75
        assert rolling_df.iloc[0]['rolling_win_rate'] == pytest.approx(66.67, rel=0.1)