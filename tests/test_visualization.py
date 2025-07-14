"""Tests for visualization components."""

import pytest
from datetime import date, datetime
from decimal import Decimal
import numpy as np
import pandas as pd

from src.components.visualization import VisualizationComponents
from src.models import Trade


class TestVisualizationComponents:
    @pytest.fixture
    def viz_dark(self):
        """Create visualization component with dark theme."""
        return VisualizationComponents(theme="dark")
    
    @pytest.fixture
    def viz_light(self):
        """Create visualization component with light theme."""
        return VisualizationComponents(theme="light")
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        trades = [
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
                trade_date=date(2024, 1, 5),
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
                trade_date=date(2024, 1, 8),
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
                trade_date=date(2024, 1, 10),
                symbol="NVDA",
                side="buy",
                entry_price=Decimal("500.00"),
                exit_price=Decimal("520.00"),
                quantity=Decimal("20"),
                pnl=Decimal("400.00"),
                commission=Decimal("10.00")
            ),
        ]
        return trades
    
    def test_theme_setup(self, viz_dark, viz_light):
        """Test theme configuration."""
        # Dark theme
        assert viz_dark.theme == "dark"
        assert viz_dark.bg_color == "#0E1117"
        assert viz_dark.text_color == "#FAFAFA"
        assert viz_dark.positive_color == "#00CC88"
        assert viz_dark.negative_color == "#FF4444"
        
        # Light theme
        assert viz_light.theme == "light"
        assert viz_light.bg_color == "#FFFFFF"
        assert viz_light.text_color == "#262730"
        assert viz_light.positive_color == "#22C55E"
        assert viz_light.negative_color == "#EF4444"
    
    def test_render_pnl_chart_bar(self, viz_dark, sample_trades):
        """Test P&L bar chart rendering."""
        fig = viz_dark.render_pnl_chart(sample_trades, period="daily", chart_type="bar")
        
        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert fig.layout.title.text == "Daily P&L Chart"
        
        # Check data
        bar_data = fig.data[0]
        assert len(bar_data.x) == len(sample_trades)
        assert sum(bar_data.y) == sum(float(t.pnl) for t in sample_trades)
    
    def test_render_pnl_chart_periods(self, viz_dark, sample_trades):
        """Test P&L chart with different periods."""
        # Daily
        daily_fig = viz_dark.render_pnl_chart(sample_trades, period="daily")
        assert len(daily_fig.data[0].x) == 5  # 5 unique days
        
        # Weekly
        weekly_fig = viz_dark.render_pnl_chart(sample_trades, period="weekly")
        assert len(weekly_fig.data[0].x) == 2  # 2 weeks
        
        # Monthly
        monthly_fig = viz_dark.render_pnl_chart(sample_trades, period="monthly")
        assert len(monthly_fig.data[0].x) == 1  # 1 month
    
    def test_render_pnl_chart_types(self, viz_dark, sample_trades):
        """Test different chart types."""
        # Bar chart
        bar_fig = viz_dark.render_pnl_chart(sample_trades, chart_type="bar")
        assert bar_fig.data[0].type == "bar"
        
        # Line chart
        line_fig = viz_dark.render_pnl_chart(sample_trades, chart_type="line")
        assert line_fig.data[0].type == "scatter"
        assert line_fig.data[0].mode == "lines+markers"
        
        # Area chart
        area_fig = viz_dark.render_pnl_chart(sample_trades, chart_type="area")
        assert area_fig.data[0].type == "scatter"
        assert area_fig.data[0].fill == "tozeroy"
    
    def test_render_cumulative_returns(self, viz_dark, sample_trades):
        """Test cumulative returns chart."""
        fig = viz_dark.render_cumulative_returns(sample_trades, initial_capital=10000)
        
        assert fig is not None
        assert len(fig.data) >= 1
        assert fig.data[0].name == "Strategy"
        assert fig.layout.title.text == "Cumulative Returns"
        
        # Check cumulative calculation
        total_pnl = sum(float(t.pnl) for t in sample_trades if t.pnl)
        expected_final_return = (total_pnl / 10000) * 100
        actual_final_return = fig.data[0].y[-1]
        assert abs(actual_final_return - expected_final_return) < 0.01
    
    def test_render_cumulative_returns_with_benchmark(self, viz_dark, sample_trades):
        """Test cumulative returns with benchmark."""
        fig = viz_dark.render_cumulative_returns(
            sample_trades, 
            initial_capital=10000,
            show_benchmark=True,
            benchmark_return=0.08
        )
        
        assert len(fig.data) == 2
        assert fig.data[0].name == "Strategy"
        assert fig.data[1].name == "Benchmark (8%)"
        assert fig.data[1].line.dash == "dash"
    
    def test_render_drawdown_chart(self, viz_dark, sample_trades):
        """Test drawdown chart rendering."""
        fig = viz_dark.render_drawdown_chart(sample_trades, initial_capital=10000)
        
        assert fig is not None
        assert len(fig.data) >= 1
        assert fig.data[0].name == "Drawdown"
        assert fig.layout.title.text == "Drawdown Analysis"
        
        # Drawdown should be negative or zero
        drawdown_values = fig.data[0].y
        assert all(dd <= 0 for dd in drawdown_values if not np.isnan(dd))
    
    def test_render_monthly_heatmap(self, viz_dark, sample_trades):
        """Test monthly heatmap rendering."""
        fig = viz_dark.render_monthly_heatmap(sample_trades)
        
        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"
        assert fig.layout.title.text == "Monthly Returns Heatmap"
        
        # Check heatmap dimensions
        heatmap_data = fig.data[0]
        assert len(heatmap_data.x) == 12  # 12 months
        assert heatmap_data.y[0] == 2024  # Year from sample trades
    
    def test_render_distribution_chart(self, viz_dark, sample_trades):
        """Test distribution chart rendering."""
        fig = viz_dark.render_distribution_chart(sample_trades, bins=20)
        
        assert fig is not None
        assert len(fig.data) >= 2  # Histogram + normal distribution
        assert fig.data[0].type == "histogram"
        assert fig.data[1].type == "scatter"
        assert fig.layout.title.text == "P&L Distribution"
    
    def test_render_performance_metrics_cards(self, viz_dark):
        """Test performance metrics cards."""
        metrics = {
            'total_pnl': 1500.50,
            'win_rate': 65.5,
            'sharpe_ratio': 1.85,
            'max_drawdown': 0.15
        }
        
        cards = viz_dark.render_performance_metrics_cards(metrics)
        
        assert len(cards) == 4
        assert all(hasattr(card, 'data') for card in cards)
        assert all(card.data[0].type == "indicator" for card in cards)
    
    def test_empty_trades_handling(self, viz_dark):
        """Test handling of empty trades list."""
        empty_trades = []
        
        # All chart types should handle empty trades gracefully
        pnl_fig = viz_dark.render_pnl_chart(empty_trades)
        assert "No trades to display" in str(pnl_fig.layout.annotations[0].text)
        
        returns_fig = viz_dark.render_cumulative_returns(empty_trades)
        assert "No trades to display" in str(returns_fig.layout.annotations[0].text)
        
        drawdown_fig = viz_dark.render_drawdown_chart(empty_trades)
        assert "No trades to display" in str(drawdown_fig.layout.annotations[0].text)
        
        heatmap_fig = viz_dark.render_monthly_heatmap(empty_trades)
        assert "No trades to display" in str(heatmap_fig.layout.annotations[0].text)
        
        dist_fig = viz_dark.render_distribution_chart(empty_trades)
        assert "No trades to display" in str(dist_fig.layout.annotations[0].text)
    
    def test_trades_with_no_pnl(self, viz_dark):
        """Test handling of trades with no P&L data."""
        trades_no_pnl = [
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
        
        dist_fig = viz_dark.render_distribution_chart(trades_no_pnl)
        assert "No P&L data to display" in str(dist_fig.layout.annotations[0].text)
    
    def test_large_dataset_performance(self, viz_dark):
        """Test performance with large dataset."""
        # Create 1000 trades
        large_trades = []
        for i in range(1000):
            trade = Trade(
                strategy_id=1,
                trade_date=date(2023, 1, 1) + pd.Timedelta(days=i % 365),
                symbol=f"STOCK{i % 10}",
                side="buy" if i % 2 == 0 else "sell",
                pnl=Decimal(str(np.random.normal(100, 500)))
            )
            large_trades.append(trade)
        
        # Should complete without error
        fig = viz_dark.render_pnl_chart(large_trades, period="monthly")
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_color_coding(self, viz_dark, sample_trades):
        """Test color coding for positive/negative values."""
        fig = viz_dark.render_pnl_chart(sample_trades, period="daily", chart_type="bar")
        
        colors = fig.data[0].marker.color
        pnl_values = [float(t.pnl) for t in sample_trades if t.pnl]
        
        # Check that positive values get positive color
        for i, pnl in enumerate(pnl_values):
            if pnl > 0:
                assert colors[i] == viz_dark.positive_color
            elif pnl < 0:
                assert colors[i] == viz_dark.negative_color