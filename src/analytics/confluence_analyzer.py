"""
Confluence Detection System

This module implements algorithms to identify signal overlaps between trading strategies,
calculate confluence strength scores, and analyze the performance impact of signal confluence.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from decimal import Decimal
import numpy as np
import pandas as pd
from collections import defaultdict

from ..models import Trade, Strategy


logger = logging.getLogger(__name__)


@dataclass
class SignalOverlap:
    """Represents a detected signal overlap between strategies."""
    strategies: List[int]  # Strategy IDs involved
    strategy_names: List[str]  # Strategy names for display
    center_time: datetime  # Central time of the overlap
    time_window: timedelta  # Time window of the overlap
    trades: List[Trade]  # Trades involved in the overlap
    symbols: Set[str]  # Symbols involved
    sides: Set[str]  # Trade sides (buy/sell)
    overlap_strength: float  # Calculated strength score (0-1)
    confluence_type: str  # Type of confluence (same_direction, mixed, etc.)


@dataclass
class ConfluenceMetrics:
    """Metrics for confluence performance analysis."""
    total_overlaps: int
    overlap_win_rate: float
    overlap_avg_pnl: float
    individual_win_rate: float
    individual_avg_pnl: float
    confluence_advantage: float  # Percentage improvement
    best_confluence_strategies: List[Tuple[List[str], float]]  # Top performing combinations
    overlap_frequency: Dict[str, int]  # Frequency by confluence type


class ConfluenceAnalyzer:
    """Analyzes signal confluence between trading strategies."""
    
    def __init__(self, time_window_hours: int = 24, min_strategies: int = 2):
        """
        Initialize confluence analyzer.
        
        Args:
            time_window_hours: Time window for detecting overlaps (hours)
            min_strategies: Minimum number of strategies for confluence
        """
        self.time_window = timedelta(hours=time_window_hours)
        self.min_strategies = min_strategies
        
    def find_signal_overlaps(self, 
                           strategies_trades: Dict[int, List[Trade]],
                           strategy_names: Dict[int, str]) -> List[SignalOverlap]:
        """
        Find signal overlaps between multiple strategies.
        
        Args:
            strategies_trades: Dict mapping strategy_id to list of trades
            strategy_names: Dict mapping strategy_id to strategy name
            
        Returns:
            List of detected signal overlaps
        """
        logger.info(f"Analyzing signal overlaps for {len(strategies_trades)} strategies")
        
        if len(strategies_trades) < self.min_strategies:
            logger.warning(f"Need at least {self.min_strategies} strategies for confluence analysis")
            return []
        
        overlaps = []
        strategy_ids = list(strategies_trades.keys())
        
        # Create time-indexed trade lookup for efficient overlap detection
        time_indexed_trades = self._create_time_index(strategies_trades)
        
        # Find overlaps for each time window
        for center_time, time_trades in time_indexed_trades.items():
            if len(time_trades) >= self.min_strategies:
                # Group trades by strategy
                strategy_trade_groups = defaultdict(list)
                for trade in time_trades:
                    strategy_trade_groups[trade.strategy_id].append(trade)
                
                # Only consider if multiple strategies have trades
                if len(strategy_trade_groups) >= self.min_strategies:
                    overlap = self._create_overlap(
                        strategy_trade_groups,
                        strategy_names,
                        center_time
                    )
                    if overlap:
                        overlaps.append(overlap)
        
        # Remove duplicates and merge nearby overlaps
        overlaps = self._merge_nearby_overlaps(overlaps)
        
        logger.info(f"Found {len(overlaps)} signal overlaps")
        return overlaps
    
    def _create_time_index(self, strategies_trades: Dict[int, List[Trade]]) -> Dict[datetime, List[Trade]]:
        """Create time-indexed lookup of all trades."""
        time_index = defaultdict(list)
        
        for strategy_id, trades in strategies_trades.items():
            for trade in trades:
                # Convert date to datetime for time window calculations
                if isinstance(trade.trade_date, datetime):
                    trade_time = trade.trade_date
                else:
                    trade_time = datetime.combine(trade.trade_date, datetime.min.time())
                
                # Group trades into time windows
                window_center = self._round_to_time_window(trade_time)
                time_index[window_center].append(trade)
        
        return time_index
    
    def _round_to_time_window(self, dt: datetime) -> datetime:
        """Round datetime to the nearest time window."""
        window_hours = self.time_window.total_seconds() / 3600
        rounded_hour = round(dt.hour / window_hours) * window_hours
        return dt.replace(hour=int(rounded_hour), minute=0, second=0, microsecond=0)
    
    def _create_overlap(self, 
                       strategy_trade_groups: Dict[int, List[Trade]],
                       strategy_names: Dict[int, str],
                       center_time: datetime) -> Optional[SignalOverlap]:
        """Create a SignalOverlap from grouped trades."""
        try:
            strategies = list(strategy_trade_groups.keys())
            names = [strategy_names.get(sid, f"Strategy {sid}") for sid in strategies]
            all_trades = []
            symbols = set()
            sides = set()
            
            for trades in strategy_trade_groups.values():
                all_trades.extend(trades)
                for trade in trades:
                    if trade.symbol:
                        symbols.add(trade.symbol)
                    if trade.side:
                        sides.add(trade.side.lower())
            
            # Calculate overlap strength
            strength = self._calculate_overlap_strength(strategy_trade_groups, all_trades)
            
            # Determine confluence type
            confluence_type = self._determine_confluence_type(sides, symbols)
            
            return SignalOverlap(
                strategies=strategies,
                strategy_names=names,
                center_time=center_time,
                time_window=self.time_window,
                trades=all_trades,
                symbols=symbols,
                sides=sides,
                overlap_strength=strength,
                confluence_type=confluence_type
            )
            
        except Exception as e:
            logger.error(f"Error creating overlap: {str(e)}")
            return None
    
    def _calculate_overlap_strength(self, 
                                  strategy_trade_groups: Dict[int, List[Trade]],
                                  all_trades: List[Trade]) -> float:
        """
        Calculate overlap strength score (0-1).
        
        Factors:
        - Number of strategies involved
        - Trade volume/size alignment
        - Symbol overlap
        - Time proximity
        """
        try:
            num_strategies = len(strategy_trade_groups)
            
            # Base strength from number of strategies
            strength = min(num_strategies / 5.0, 1.0)  # Cap at 5 strategies
            
            # Symbol overlap bonus
            all_symbols = set()
            common_symbols = None
            
            for trades in strategy_trade_groups.values():
                strategy_symbols = {t.symbol for t in trades if t.symbol}
                all_symbols.update(strategy_symbols)
                
                if common_symbols is None:
                    common_symbols = strategy_symbols.copy()
                else:
                    common_symbols &= strategy_symbols
            
            if all_symbols and common_symbols:
                symbol_overlap_ratio = len(common_symbols) / len(all_symbols)
                strength += symbol_overlap_ratio * 0.3
            
            # Trade direction alignment bonus
            all_sides = {t.side.lower() for t in all_trades if t.side}
            if len(all_sides) == 1:  # All same direction
                strength += 0.2
            
            # Trade size correlation bonus
            if len(all_trades) > 1:
                sizes = [float(t.quantity or 0) for t in all_trades if t.quantity]
                if len(sizes) > 1 and np.std(sizes) / (np.mean(sizes) + 1e-8) < 0.5:
                    strength += 0.1  # Low variance in trade sizes
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating overlap strength: {str(e)}")
            return 0.5  # Default moderate strength
    
    def _determine_confluence_type(self, sides: Set[str], symbols: Set[str]) -> str:
        """Determine the type of confluence based on trade characteristics."""
        if len(sides) == 1:
            if len(symbols) == 1:
                return "strong_directional"  # Same direction, same symbol
            else:
                return "directional"  # Same direction, different symbols
        elif len(sides) == 2:
            if len(symbols) == 1:
                return "hedged_single"  # Both directions, same symbol
            else:
                return "mixed"  # Mixed directions and symbols
        else:
            return "complex"  # Multiple directions
    
    def _merge_nearby_overlaps(self, overlaps: List[SignalOverlap]) -> List[SignalOverlap]:
        """Merge overlaps that are very close in time to avoid duplicates."""
        if not overlaps:
            return overlaps
        
        # Sort by time
        overlaps.sort(key=lambda x: x.center_time)
        
        merged = []
        current_overlap = overlaps[0]
        
        for next_overlap in overlaps[1:]:
            time_diff = abs((next_overlap.center_time - current_overlap.center_time).total_seconds())
            
            # If overlaps are within half a time window, merge them
            if time_diff < self.time_window.total_seconds() / 2:
                current_overlap = self._merge_two_overlaps(current_overlap, next_overlap)
            else:
                merged.append(current_overlap)
                current_overlap = next_overlap
        
        merged.append(current_overlap)
        return merged
    
    def _merge_two_overlaps(self, overlap1: SignalOverlap, overlap2: SignalOverlap) -> SignalOverlap:
        """Merge two overlapping signals."""
        # Combine all attributes
        combined_strategies = list(set(overlap1.strategies + overlap2.strategies))
        combined_names = list(set(overlap1.strategy_names + overlap2.strategy_names))
        combined_trades = overlap1.trades + overlap2.trades
        combined_symbols = overlap1.symbols | overlap2.symbols
        combined_sides = overlap1.sides | overlap2.sides
        
        # Use earlier time as center
        center_time = min(overlap1.center_time, overlap2.center_time)
        
        # Recalculate strength
        strength = max(overlap1.overlap_strength, overlap2.overlap_strength)
        
        # Redetermine type
        confluence_type = self._determine_confluence_type(combined_sides, combined_symbols)
        
        return SignalOverlap(
            strategies=combined_strategies,
            strategy_names=combined_names,
            center_time=center_time,
            time_window=self.time_window,
            trades=combined_trades,
            symbols=combined_symbols,
            sides=combined_sides,
            overlap_strength=strength,
            confluence_type=confluence_type
        )
    
    def analyze_confluence_performance(self, 
                                     overlaps: List[SignalOverlap],
                                     all_trades: List[Trade]) -> ConfluenceMetrics:
        """
        Analyze the performance impact of confluence periods.
        
        Args:
            overlaps: List of detected overlaps
            all_trades: All trades for comparison
            
        Returns:
            ConfluenceMetrics with performance analysis
        """
        logger.info(f"Analyzing performance for {len(overlaps)} confluence periods")
        
        if not overlaps:
            return ConfluenceMetrics(
                total_overlaps=0,
                overlap_win_rate=0,
                overlap_avg_pnl=0,
                individual_win_rate=0,
                individual_avg_pnl=0,
                confluence_advantage=0,
                best_confluence_strategies=[],
                overlap_frequency={}
            )
        
        # Extract overlap trades and calculate metrics
        overlap_trades = []
        for overlap in overlaps:
            overlap_trades.extend(overlap.trades)
        
        # Calculate overlap performance
        overlap_wins = sum(1 for t in overlap_trades if t.pnl and t.pnl > 0)
        overlap_win_rate = overlap_wins / len(overlap_trades) if overlap_trades else 0
        overlap_avg_pnl = np.mean([float(t.pnl) for t in overlap_trades if t.pnl]) if overlap_trades else 0
        
        # Calculate individual trade performance (non-overlap)
        overlap_trade_ids = {id(t) for t in overlap_trades}
        individual_trades = [t for t in all_trades if id(t) not in overlap_trade_ids]
        
        individual_wins = sum(1 for t in individual_trades if t.pnl and t.pnl > 0)
        individual_win_rate = individual_wins / len(individual_trades) if individual_trades else 0
        individual_avg_pnl = np.mean([float(t.pnl) for t in individual_trades if t.pnl]) if individual_trades else 0
        
        # Calculate confluence advantage
        if individual_avg_pnl != 0:
            confluence_advantage = ((overlap_avg_pnl - individual_avg_pnl) / abs(individual_avg_pnl)) * 100
        else:
            confluence_advantage = 0
        
        # Find best performing strategy combinations
        best_combinations = self._find_best_strategy_combinations(overlaps)
        
        # Count overlap frequency by type
        overlap_frequency = defaultdict(int)
        for overlap in overlaps:
            overlap_frequency[overlap.confluence_type] += 1
        
        return ConfluenceMetrics(
            total_overlaps=len(overlaps),
            overlap_win_rate=overlap_win_rate,
            overlap_avg_pnl=overlap_avg_pnl,
            individual_win_rate=individual_win_rate,
            individual_avg_pnl=individual_avg_pnl,
            confluence_advantage=confluence_advantage,
            best_confluence_strategies=best_combinations,
            overlap_frequency=dict(overlap_frequency)
        )
    
    def _find_best_strategy_combinations(self, overlaps: List[SignalOverlap]) -> List[Tuple[List[str], float]]:
        """Find the best performing strategy combinations."""
        combination_performance = defaultdict(list)
        
        for overlap in overlaps:
            # Create a key for the strategy combination
            combo_key = tuple(sorted(overlap.strategy_names))
            
            # Calculate total P&L for this overlap
            total_pnl = sum(float(t.pnl) for t in overlap.trades if t.pnl)
            combination_performance[combo_key].append(total_pnl)
        
        # Calculate average performance for each combination
        combo_averages = []
        for combo, pnls in combination_performance.items():
            avg_pnl = np.mean(pnls)
            combo_averages.append((list(combo), avg_pnl))
        
        # Sort by performance and return top 5
        combo_averages.sort(key=lambda x: x[1], reverse=True)
        return combo_averages[:5]
    
    def get_confluence_calendar(self, overlaps: List[SignalOverlap]) -> pd.DataFrame:
        """
        Create a calendar view of confluence events.
        
        Args:
            overlaps: List of signal overlaps
            
        Returns:
            DataFrame with confluence calendar data
        """
        if not overlaps:
            return pd.DataFrame()
        
        calendar_data = []
        for overlap in overlaps:
            total_pnl = sum(float(t.pnl) for t in overlap.trades if t.pnl)
            
            calendar_data.append({
                'date': overlap.center_time.date(),
                'time': overlap.center_time.time(),
                'strategies': ', '.join(overlap.strategy_names),
                'num_strategies': len(overlap.strategies),
                'symbols': ', '.join(overlap.symbols),
                'confluence_type': overlap.confluence_type,
                'strength': overlap.overlap_strength,
                'total_pnl': total_pnl,
                'num_trades': len(overlap.trades)
            })
        
        df = pd.DataFrame(calendar_data)
        return df.sort_values('date')
    
    def find_real_time_confluence(self, 
                                current_signals: Dict[int, List[Trade]],
                                strategy_names: Dict[int, str],
                                lookback_hours: int = 1) -> List[SignalOverlap]:
        """
        Find confluence in real-time or recent signals.
        
        Args:
            current_signals: Recent trades by strategy
            strategy_names: Strategy name mapping
            lookback_hours: How far back to look for signals
            
        Returns:
            List of current/recent confluence events
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        # Filter recent trades
        recent_trades = {}
        for strategy_id, trades in current_signals.items():
            recent = []
            for trade in trades:
                trade_time = trade.trade_date if isinstance(trade.trade_date, datetime) else datetime.combine(trade.trade_date, datetime.min.time())
                if trade_time >= cutoff_time:
                    recent.append(trade)
            
            if recent:
                recent_trades[strategy_id] = recent
        
        # Find overlaps in recent signals
        return self.find_signal_overlaps(recent_trades, strategy_names)