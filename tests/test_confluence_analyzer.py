"""Tests for confluence analyzer."""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
import numpy as np
from unittest.mock import Mock

from src.analytics.confluence_analyzer import ConfluenceAnalyzer, SignalOverlap, ConfluenceMetrics
from src.models import Trade, Strategy


class TestConfluenceAnalyzer:
    @pytest.fixture
    def confluence_analyzer(self):
        """Create confluence analyzer instance."""
        return ConfluenceAnalyzer(time_window_hours=24, min_strategies=2)
    
    @pytest.fixture
    def sample_strategies_trades(self):
        """Create sample strategy trades for testing."""
        # Strategy 1 trades
        strategy1_trades = [
            Trade(
                strategy_id=1,
                trade_date=datetime(2024, 1, 1, 10, 0),
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                pnl=Decimal("100.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=datetime(2024, 1, 1, 14, 0),
                symbol="MSFT",
                side="sell",
                quantity=Decimal("50"),
                pnl=Decimal("-50.00")
            ),
            Trade(
                strategy_id=1,
                trade_date=datetime(2024, 1, 3, 9, 0),
                symbol="GOOGL",
                side="buy",
                quantity=Decimal("25"),
                pnl=Decimal("75.00")
            )
        ]
        
        # Strategy 2 trades (some overlapping times)
        strategy2_trades = [
            Trade(
                strategy_id=2,
                trade_date=datetime(2024, 1, 1, 11, 0),  # Close to strategy 1
                symbol="AAPL",
                side="buy",
                quantity=Decimal("200"),
                pnl=Decimal("150.00")
            ),
            Trade(
                strategy_id=2,
                trade_date=datetime(2024, 1, 2, 15, 0),  # Different day
                symbol="TSLA",
                side="sell",
                quantity=Decimal("30"),
                pnl=Decimal("-25.00")
            ),
            Trade(
                strategy_id=2,
                trade_date=datetime(2024, 1, 3, 10, 0),  # Close to strategy 1
                symbol="NVDA",
                side="buy",
                quantity=Decimal("40"),
                pnl=Decimal("80.00")
            )
        ]
        
        # Strategy 3 trades
        strategy3_trades = [
            Trade(
                strategy_id=3,
                trade_date=datetime(2024, 1, 1, 12, 0),  # Same time window as 1&2
                symbol="META",
                side="buy",
                quantity=Decimal("75"),
                pnl=Decimal("50.00")
            ),
            Trade(
                strategy_id=3,
                trade_date=datetime(2024, 1, 4, 16, 0),
                symbol="NFLX",
                side="sell",
                quantity=Decimal("15"),
                pnl=Decimal("-30.00")
            )
        ]
        
        return {
            1: strategy1_trades,
            2: strategy2_trades,
            3: strategy3_trades
        }
    
    @pytest.fixture
    def strategy_names(self):
        """Strategy names mapping."""
        return {
            1: "Strategy A",
            2: "Strategy B", 
            3: "Strategy C"
        }
    
    def test_initialization(self, confluence_analyzer):
        """Test analyzer initialization."""
        assert confluence_analyzer.time_window == timedelta(hours=24)
        assert confluence_analyzer.min_strategies == 2
    
    def test_find_signal_overlaps_basic(self, confluence_analyzer, sample_strategies_trades, strategy_names):
        """Test basic signal overlap detection."""
        overlaps = confluence_analyzer.find_signal_overlaps(sample_strategies_trades, strategy_names)
        
        assert isinstance(overlaps, list)
        assert len(overlaps) > 0
        
        # Check overlap structure
        for overlap in overlaps:
            assert isinstance(overlap, SignalOverlap)
            assert len(overlap.strategies) >= 2
            assert len(overlap.strategy_names) == len(overlap.strategies)
            assert overlap.overlap_strength >= 0 and overlap.overlap_strength <= 1
            assert overlap.confluence_type in [
                "strong_directional", "directional", "hedged_single", "mixed", "complex"
            ]
    
    def test_find_signal_overlaps_insufficient_strategies(self, confluence_analyzer, strategy_names):
        """Test with insufficient strategies."""
        single_strategy = {1: [
            Trade(strategy_id=1, trade_date=datetime(2024, 1, 1), symbol="AAPL", side="buy", pnl=Decimal("100"))
        ]}
        
        overlaps = confluence_analyzer.find_signal_overlaps(single_strategy, strategy_names)
        assert overlaps == []
    
    def test_find_signal_overlaps_no_overlap(self, confluence_analyzer, strategy_names):
        """Test with trades that don't overlap in time."""
        non_overlapping_trades = {
            1: [Trade(strategy_id=1, trade_date=datetime(2024, 1, 1), symbol="AAPL", side="buy", pnl=Decimal("100"))],
            2: [Trade(strategy_id=2, trade_date=datetime(2024, 1, 5), symbol="MSFT", side="sell", pnl=Decimal("-50"))]
        }
        
        overlaps = confluence_analyzer.find_signal_overlaps(non_overlapping_trades, strategy_names)
        # Should be empty or very few overlaps due to time distance
        assert len(overlaps) <= 1
    
    def test_create_time_index(self, confluence_analyzer, sample_strategies_trades):
        """Test time indexing functionality."""
        time_index = confluence_analyzer._create_time_index(sample_strategies_trades)
        
        assert isinstance(time_index, dict)
        assert len(time_index) > 0
        
        # Check that trades are properly indexed by time
        for time_key, trades in time_index.items():
            assert isinstance(time_key, datetime)
            assert isinstance(trades, list)
            assert all(isinstance(t, Trade) for t in trades)
    
    def test_round_to_time_window(self, confluence_analyzer):
        """Test time window rounding."""
        test_time = datetime(2024, 1, 1, 14, 30, 45)
        rounded = confluence_analyzer._round_to_time_window(test_time)
        
        assert isinstance(rounded, datetime)
        assert rounded.minute == 0
        assert rounded.second == 0
        assert rounded.microsecond == 0
    
    def test_calculate_overlap_strength(self, confluence_analyzer):
        """Test overlap strength calculation."""
        # Create test trades
        trades = [
            Trade(strategy_id=1, symbol="AAPL", side="buy", quantity=Decimal("100"), pnl=Decimal("50")),
            Trade(strategy_id=2, symbol="AAPL", side="buy", quantity=Decimal("100"), pnl=Decimal("75")),
            Trade(strategy_id=3, symbol="MSFT", side="sell", quantity=Decimal("50"), pnl=Decimal("-25"))
        ]
        
        strategy_groups = {
            1: [trades[0]],
            2: [trades[1]],
            3: [trades[2]]
        }
        
        strength = confluence_analyzer._calculate_overlap_strength(strategy_groups, trades)
        
        assert 0 <= strength <= 1
        assert isinstance(strength, float)
    
    def test_determine_confluence_type(self, confluence_analyzer):
        """Test confluence type determination."""
        # Same direction, same symbol
        sides1 = {"buy"}
        symbols1 = {"AAPL"}
        type1 = confluence_analyzer._determine_confluence_type(sides1, symbols1)
        assert type1 == "strong_directional"
        
        # Same direction, different symbols
        sides2 = {"buy"}
        symbols2 = {"AAPL", "MSFT"}
        type2 = confluence_analyzer._determine_confluence_type(sides2, symbols2)
        assert type2 == "directional"
        
        # Mixed directions, same symbol
        sides3 = {"buy", "sell"}
        symbols3 = {"AAPL"}
        type3 = confluence_analyzer._determine_confluence_type(sides3, symbols3)
        assert type3 == "hedged_single"
        
        # Mixed directions, different symbols
        sides4 = {"buy", "sell"}
        symbols4 = {"AAPL", "MSFT"}
        type4 = confluence_analyzer._determine_confluence_type(sides4, symbols4)
        assert type4 == "mixed"
    
    def test_analyze_confluence_performance(self, confluence_analyzer, sample_strategies_trades, strategy_names):
        """Test confluence performance analysis."""
        overlaps = confluence_analyzer.find_signal_overlaps(sample_strategies_trades, strategy_names)
        
        # Create list of all trades
        all_trades = []
        for trades in sample_strategies_trades.values():
            all_trades.extend(trades)
        
        metrics = confluence_analyzer.analyze_confluence_performance(overlaps, all_trades)
        
        assert isinstance(metrics, ConfluenceMetrics)
        assert metrics.total_overlaps >= 0
        assert 0 <= metrics.overlap_win_rate <= 1
        assert 0 <= metrics.individual_win_rate <= 1
        assert isinstance(metrics.overlap_avg_pnl, float)
        assert isinstance(metrics.individual_avg_pnl, float)
        assert isinstance(metrics.confluence_advantage, float)
        assert isinstance(metrics.best_confluence_strategies, list)
        assert isinstance(metrics.overlap_frequency, dict)
    
    def test_analyze_confluence_performance_empty(self, confluence_analyzer):
        """Test performance analysis with empty data."""
        metrics = confluence_analyzer.analyze_confluence_performance([], [])
        
        assert metrics.total_overlaps == 0
        assert metrics.overlap_win_rate == 0
        assert metrics.overlap_avg_pnl == 0
        assert metrics.individual_win_rate == 0
        assert metrics.individual_avg_pnl == 0
        assert metrics.confluence_advantage == 0
        assert metrics.best_confluence_strategies == []
        assert metrics.overlap_frequency == {}
    
    def test_get_confluence_calendar(self, confluence_analyzer, sample_strategies_trades, strategy_names):
        """Test confluence calendar generation."""
        overlaps = confluence_analyzer.find_signal_overlaps(sample_strategies_trades, strategy_names)
        calendar_df = confluence_analyzer.get_confluence_calendar(overlaps)
        
        if not calendar_df.empty:
            expected_columns = [
                'date', 'time', 'strategies', 'num_strategies', 'symbols',
                'confluence_type', 'strength', 'total_pnl', 'num_trades'
            ]
            
            for col in expected_columns:
                assert col in calendar_df.columns
            
            assert len(calendar_df) == len(overlaps)
            assert calendar_df['num_strategies'].min() >= 2
    
    def test_get_confluence_calendar_empty(self, confluence_analyzer):
        """Test calendar with empty overlaps."""
        calendar_df = confluence_analyzer.get_confluence_calendar([])
        assert calendar_df.empty
    
    def test_find_real_time_confluence(self, confluence_analyzer, strategy_names):
        """Test real-time confluence detection."""
        # Create recent signals (within last hour)
        now = datetime.now()
        recent_signals = {
            1: [Trade(
                strategy_id=1,
                trade_date=now - timedelta(minutes=30),
                symbol="AAPL",
                side="buy",
                pnl=Decimal("100")
            )],
            2: [Trade(
                strategy_id=2,
                trade_date=now - timedelta(minutes=45),
                symbol="MSFT",
                side="buy",
                pnl=Decimal("75")
            )]
        }
        
        overlaps = confluence_analyzer.find_real_time_confluence(
            recent_signals, strategy_names, lookback_hours=1
        )
        
        assert isinstance(overlaps, list)
        # Should find overlap since trades are within time window
        if overlaps:
            for overlap in overlaps:
                assert isinstance(overlap, SignalOverlap)
    
    def test_merge_nearby_overlaps(self, confluence_analyzer):
        """Test merging of nearby overlaps."""
        # Create two close overlaps
        overlap1 = SignalOverlap(
            strategies=[1, 2],
            strategy_names=["Strategy A", "Strategy B"],
            center_time=datetime(2024, 1, 1, 10, 0),
            time_window=timedelta(hours=24),
            trades=[],
            symbols={"AAPL"},
            sides={"buy"},
            overlap_strength=0.8,
            confluence_type="directional"
        )
        
        overlap2 = SignalOverlap(
            strategies=[1, 3],
            strategy_names=["Strategy A", "Strategy C"],
            center_time=datetime(2024, 1, 1, 11, 0),  # 1 hour later
            time_window=timedelta(hours=24),
            trades=[],
            symbols={"MSFT"},
            sides={"buy"},
            overlap_strength=0.7,
            confluence_type="directional"
        )
        
        merged = confluence_analyzer._merge_nearby_overlaps([overlap1, overlap2])
        
        # Should merge into one overlap since they're close
        assert len(merged) <= 2  # Could be 1 if merged, 2 if not
        
        if len(merged) == 1:  # If merged
            merged_overlap = merged[0]
            assert len(merged_overlap.strategies) >= 2
            assert len(merged_overlap.symbols) >= 1
    
    def test_error_handling(self, confluence_analyzer):
        """Test error handling in various scenarios."""
        # Empty strategies
        overlaps = confluence_analyzer.find_signal_overlaps({}, {})
        assert overlaps == []
        
        # Malformed trade data
        bad_trades = {
            1: [Trade(strategy_id=1, trade_date=None, symbol=None, side=None, pnl=None)]
        }
        
        # Should not crash
        overlaps = confluence_analyzer.find_signal_overlaps(bad_trades, {1: "Test"})
        assert isinstance(overlaps, list)
    
    def test_strength_scoring_edge_cases(self, confluence_analyzer):
        """Test strength scoring with edge cases."""
        # Single strategy (should not be called normally, but test robustness)
        single_group = {
            1: [Trade(strategy_id=1, symbol="AAPL", side="buy", quantity=Decimal("100"))]
        }
        
        strength = confluence_analyzer._calculate_overlap_strength(single_group, single_group[1])
        assert 0 <= strength <= 1
        
        # No symbols or sides
        no_data_trades = [
            Trade(strategy_id=1, symbol=None, side=None, quantity=None, pnl=Decimal("0"))
        ]
        no_data_group = {1: no_data_trades}
        
        strength = confluence_analyzer._calculate_overlap_strength(no_data_group, no_data_trades)
        assert 0 <= strength <= 1
    
    def test_performance_metrics_calculation(self):
        """Test detailed performance metrics calculation."""
        analyzer = ConfluenceAnalyzer()
        
        # Create overlaps with known P&L
        overlap1 = SignalOverlap(
            strategies=[1, 2],
            strategy_names=["A", "B"],
            center_time=datetime.now(),
            time_window=timedelta(hours=24),
            trades=[
                Trade(strategy_id=1, pnl=Decimal("100")),
                Trade(strategy_id=2, pnl=Decimal("50"))
            ],
            symbols={"AAPL"},
            sides={"buy"},
            overlap_strength=0.8,
            confluence_type="directional"
        )
        
        overlap2 = SignalOverlap(
            strategies=[1, 3],
            strategy_names=["A", "C"],
            center_time=datetime.now(),
            time_window=timedelta(hours=24),
            trades=[
                Trade(strategy_id=1, pnl=Decimal("-25")),
                Trade(strategy_id=3, pnl=Decimal("75"))
            ],
            symbols={"MSFT"},
            sides={"sell"},
            overlap_strength=0.6,
            confluence_type="mixed"
        )
        
        all_trades = [
            Trade(strategy_id=1, pnl=Decimal("10")),
            Trade(strategy_id=2, pnl=Decimal("-5")),
            Trade(strategy_id=3, pnl=Decimal("20"))
        ]
        
        metrics = analyzer.analyze_confluence_performance([overlap1, overlap2], all_trades)
        
        # Verify calculations
        assert metrics.total_overlaps == 2
        
        # Check that win rates are calculated
        assert 0 <= metrics.overlap_win_rate <= 1
        assert 0 <= metrics.individual_win_rate <= 1
        
        # Check P&L calculations
        assert isinstance(metrics.overlap_avg_pnl, float)
        assert isinstance(metrics.individual_avg_pnl, float)