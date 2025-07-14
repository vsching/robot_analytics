"""Analytics engine for calculating trading performance metrics."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

from ..models import Trade
from ..db.connection import DatabaseManager
from .metrics import PnLSummary, TradeStatistics, TimeBasedMetrics, PerformanceMetrics, AdvancedStatistics
from .advanced_metrics import AdvancedMetrics


logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Engine for calculating trading performance metrics."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, annual_trading_days: int = 252):
        self.db_manager = db_manager or DatabaseManager()
        self.advanced_metrics = AdvancedMetrics(annual_trading_days)
    
    def calculate_pnl_summary(self, trades: List[Trade]) -> Optional[PnLSummary]:
        """
        Calculate P&L summary statistics.
        
        Args:
            trades: List of trades
            
        Returns:
            PnLSummary object or None if no trades
        """
        if not trades:
            return None
        
        pnl_values = [trade.pnl for trade in trades if trade.pnl is not None]
        commissions = [trade.commission or Decimal("0") for trade in trades]
        
        if not pnl_values:
            return None
        
        # Convert to numpy array for calculations
        pnl_array = np.array([float(p) for p in pnl_values])
        
        total_pnl = sum(pnl_values)
        total_commission = sum(commissions)
        
        return PnLSummary(
            total_pnl=total_pnl,
            average_pnl=Decimal(str(pnl_array.mean())),
            max_pnl=max(pnl_values),
            min_pnl=min(pnl_values),
            median_pnl=Decimal(str(np.median(pnl_array))),
            std_dev=Decimal(str(pnl_array.std())) if len(pnl_array) > 1 else Decimal("0"),
            total_commission=total_commission,
            net_pnl=total_pnl - total_commission
        )
    
    def calculate_trade_statistics(self, trades: List[Trade]) -> Optional[TradeStatistics]:
        """
        Calculate comprehensive trade statistics.
        
        Args:
            trades: List of trades
            
        Returns:
            TradeStatistics object or None if no trades
        """
        if not trades:
            return None
        
        # Filter trades with P&L
        trades_with_pnl = [t for t in trades if t.pnl is not None]
        if not trades_with_pnl:
            return None
        
        # Categorize trades
        winning_trades = [t for t in trades_with_pnl if t.pnl > 0]
        losing_trades = [t for t in trades_with_pnl if t.pnl < 0]
        breakeven_trades = [t for t in trades_with_pnl if t.pnl == 0]
        
        # Calculate basic statistics
        total_trades = len(trades_with_pnl)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        breakeven_count = len(breakeven_trades)
        
        win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0
        loss_rate = (losing_count / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate averages
        average_win = (sum(t.pnl for t in winning_trades) / winning_count) if winning_count > 0 else Decimal("0")
        average_loss = (sum(t.pnl for t in losing_trades) / losing_count) if losing_count > 0 else Decimal("0")
        
        # Find extremes
        largest_win = max((t.pnl for t in winning_trades), default=Decimal("0"))
        largest_loss = min((t.pnl for t in losing_trades), default=Decimal("0"))
        
        # Calculate ratios
        win_loss_ratio = float(abs(average_win / average_loss)) if average_loss != 0 else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else 0
        
        # Expectancy
        expectancy = (Decimal(win_rate)/Decimal(100) * average_win) + ((100-Decimal(win_rate))/Decimal(100) * average_loss)
        
        # Calculate consecutive wins/losses
        consecutive_stats = self._calculate_consecutive_stats(trades_with_pnl)
        
        return TradeStatistics(
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            breakeven_trades=breakeven_count,
            win_rate=win_rate,
            loss_rate=loss_rate,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            win_loss_ratio=win_loss_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            consecutive_wins=consecutive_stats['current_wins'],
            consecutive_losses=consecutive_stats['current_losses'],
            max_consecutive_wins=consecutive_stats['max_wins'],
            max_consecutive_losses=consecutive_stats['max_losses']
        )
    
    def _calculate_consecutive_stats(self, trades: List[Trade]) -> Dict[str, int]:
        """Calculate consecutive win/loss statistics."""
        if not trades:
            return {
                'current_wins': 0,
                'current_losses': 0,
                'max_wins': 0,
                'max_losses': 0
            }
        
        # Sort trades by date
        sorted_trades = sorted(trades, key=lambda t: t.trade_date)
        
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in sorted_trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                # Breakeven resets both
                current_wins = 0
                current_losses = 0
        
        return {
            'current_wins': current_wins,
            'current_losses': current_losses,
            'max_wins': max_wins,
            'max_losses': max_losses
        }
    
    def calculate_win_rate(self, trades: List[Trade]) -> float:
        """
        Calculate win rate percentage.
        
        Args:
            trades: List of trades
            
        Returns:
            Win rate as percentage (0-100)
        """
        if not trades:
            return 0.0
        
        trades_with_pnl = [t for t in trades if t.pnl is not None]
        if not trades_with_pnl:
            return 0.0
        
        winning_trades = sum(1 for t in trades_with_pnl if t.pnl > 0)
        return (winning_trades / len(trades_with_pnl)) * 100
    
    def calculate_basic_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """
        Calculate basic performance metrics.
        
        Args:
            trades: List of trades
            
        Returns:
            Dictionary of basic metrics
        """
        if not trades:
            return {
                'trade_count': 0,
                'total_pnl': 0.0,
                'average_pnl': 0.0,
                'win_rate': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'win_loss_ratio': 0.0
            }
        
        pnl_summary = self.calculate_pnl_summary(trades)
        trade_stats = self.calculate_trade_statistics(trades)
        
        if not pnl_summary or not trade_stats:
            return {
                'trade_count': len(trades),
                'total_pnl': 0.0,
                'average_pnl': 0.0,
                'win_rate': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'win_loss_ratio': 0.0
            }
        
        return {
            'trade_count': trade_stats.total_trades,
            'total_pnl': float(pnl_summary.total_pnl),
            'average_pnl': float(pnl_summary.average_pnl),
            'win_rate': trade_stats.win_rate,
            'average_win': float(trade_stats.average_win),
            'average_loss': float(trade_stats.average_loss),
            'win_loss_ratio': trade_stats.win_loss_ratio
        }
    
    def get_trades_for_strategy(self, strategy_id: int) -> List[Trade]:
        """
        Get all trades for a strategy from database.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            List of trades
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, strategy_id, trade_date, symbol, side,
                       entry_price, exit_price, quantity, pnl, commission
                FROM trades
                WHERE strategy_id = ?
                ORDER BY trade_date
                """,
                (strategy_id,)
            )
            
            rows = cursor.fetchall()
            trades = []
            
            for row in rows:
                trade = Trade(
                    id=row[0],
                    strategy_id=row[1],
                    trade_date=datetime.strptime(row[2], '%Y-%m-%d').date() if isinstance(row[2], str) else row[2],
                    symbol=row[3],
                    side=row[4],
                    entry_price=Decimal(str(row[5])) if row[5] is not None else None,
                    exit_price=Decimal(str(row[6])) if row[6] is not None else None,
                    quantity=Decimal(str(row[7])) if row[7] is not None else None,
                    pnl=Decimal(str(row[8])) if row[8] is not None else None,
                    commission=Decimal(str(row[9])) if row[9] is not None else Decimal("0")
                )
                trades.append(trade)
            
            return trades
    
    def get_trades_dataframe(self, strategy_id: int, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get trades for a strategy as a pandas DataFrame.
        
        Args:
            strategy_id: Strategy ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with trade data
        """
        with self.db_manager.get_connection() as conn:
            query = """
                SELECT id, strategy_id, trade_date, symbol, side,
                       entry_price, exit_price, quantity, pnl, commission,
                       entry_time, exit_time
                FROM trades
                WHERE strategy_id = ?
            """
            params = [strategy_id]
            
            if start_date:
                query += " AND trade_date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
            
            if end_date:
                query += " AND trade_date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))
            
            query += " ORDER BY trade_date, entry_time"
            
            # Read directly into DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            # Convert numeric columns
            numeric_columns = ['entry_price', 'exit_price', 'quantity', 'pnl', 'commission']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure trade_date is datetime
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            return df
    
    def calculate_metrics_for_strategy(self, strategy_id: int) -> Optional[PerformanceMetrics]:
        """
        Calculate all performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            PerformanceMetrics object or None if no trades
        """
        trades = self.get_trades_for_strategy(strategy_id)
        
        if not trades:
            return None
        
        pnl_summary = self.calculate_pnl_summary(trades)
        trade_stats = self.calculate_trade_statistics(trades)
        
        if not pnl_summary or not trade_stats:
            return None
        
        # Calculate time-based metrics
        monthly_metrics = self.calculate_monthly_metrics(trades)
        weekly_metrics = self.calculate_weekly_metrics(trades)
        
        # Calculate advanced metrics
        try:
            advanced_results = self.calculate_advanced_metrics(trades)
            advanced_stats = AdvancedStatistics(
                sharpe_ratio=advanced_results.get('sharpe_ratio'),
                sortino_ratio=advanced_results.get('sortino_ratio'),
                calmar_ratio=advanced_results.get('calmar_ratio'),
                omega_ratio=advanced_results.get('omega_ratio'),
                max_drawdown=advanced_results.get('max_drawdown', 0.0),
                max_drawdown_duration=advanced_results.get('max_drawdown_details', {}).get('drawdown_duration'),
                recovery_duration=advanced_results.get('max_drawdown_details', {}).get('recovery_duration'),
                value_at_risk_95=advanced_results.get('value_at_risk_95'),
                conditional_var_95=advanced_results.get('conditional_var_95')
            )
        except Exception as e:
            logger.warning(f"Failed to calculate advanced metrics: {str(e)}")
            advanced_stats = None
        
        return PerformanceMetrics(
            strategy_id=strategy_id,
            calculated_at=datetime.utcnow(),
            pnl_summary=pnl_summary,
            trade_statistics=trade_stats,
            monthly_metrics=monthly_metrics,
            weekly_metrics=weekly_metrics,
            advanced_statistics=advanced_stats
        )
    
    def calculate_monthly_metrics(self, trades: List[Trade]) -> List[TimeBasedMetrics]:
        """
        Calculate metrics aggregated by month.
        
        Args:
            trades: List of trades
            
        Returns:
            List of TimeBasedMetrics for each month
        """
        if not trades:
            return []
        
        # Create DataFrame for easier grouping
        df = pd.DataFrame([
            {
                'trade_date': t.trade_date,
                'pnl': float(t.pnl) if t.pnl is not None else 0,
                'symbol': t.symbol
            }
            for t in trades
        ])
        
        # Add month column
        df['month'] = pd.to_datetime(df['trade_date']).dt.to_period('M')
        
        monthly_metrics = []
        
        # Group by month
        for month, group in df.groupby('month'):
            # Get winning trades for the month
            wins = group[group['pnl'] > 0]
            total_trades = len(group)
            
            # Find best and worst days
            daily_pnl = group.groupby('trade_date')['pnl'].sum()
            best_day = daily_pnl.idxmax() if not daily_pnl.empty else None
            worst_day = daily_pnl.idxmin() if not daily_pnl.empty else None
            
            metric = TimeBasedMetrics(
                period_start=month.to_timestamp().date(),
                period_end=(month.to_timestamp() + pd.offsets.MonthEnd(0)).date(),
                period_type='monthly',
                total_trades=total_trades,
                total_pnl=Decimal(str(group['pnl'].sum())),
                win_rate=(len(wins) / total_trades * 100) if total_trades > 0 else 0,
                average_pnl=Decimal(str(group['pnl'].mean())),
                best_day=best_day,
                worst_day=worst_day,
                trading_days=group['trade_date'].nunique()
            )
            monthly_metrics.append(metric)
        
        return monthly_metrics
    
    def calculate_weekly_metrics(self, trades: List[Trade]) -> List[TimeBasedMetrics]:
        """
        Calculate metrics aggregated by week.
        
        Args:
            trades: List of trades
            
        Returns:
            List of TimeBasedMetrics for each week
        """
        if not trades:
            return []
        
        # Create DataFrame for easier grouping
        df = pd.DataFrame([
            {
                'trade_date': t.trade_date,
                'pnl': float(t.pnl) if t.pnl is not None else 0,
                'symbol': t.symbol
            }
            for t in trades
        ])
        
        # Add week column
        df['week'] = pd.to_datetime(df['trade_date']).dt.to_period('W')
        
        weekly_metrics = []
        
        # Group by week
        for week, group in df.groupby('week'):
            # Get winning trades for the week
            wins = group[group['pnl'] > 0]
            total_trades = len(group)
            
            # Find best and worst days
            daily_pnl = group.groupby('trade_date')['pnl'].sum()
            best_day = daily_pnl.idxmax() if not daily_pnl.empty else None
            worst_day = daily_pnl.idxmin() if not daily_pnl.empty else None
            
            # Week periods in pandas start on Monday
            week_start = week.to_timestamp().date()
            week_end = (week.to_timestamp() + timedelta(days=6)).date()
            
            metric = TimeBasedMetrics(
                period_start=week_start,
                period_end=week_end,
                period_type='weekly',
                total_trades=total_trades,
                total_pnl=Decimal(str(group['pnl'].sum())),
                win_rate=(len(wins) / total_trades * 100) if total_trades > 0 else 0,
                average_pnl=Decimal(str(group['pnl'].mean())),
                best_day=best_day,
                worst_day=worst_day,
                trading_days=group['trade_date'].nunique()
            )
            weekly_metrics.append(metric)
        
        return weekly_metrics
    
    def calculate_daily_metrics(self, trades: List[Trade]) -> List[TimeBasedMetrics]:
        """
        Calculate metrics aggregated by day.
        
        Args:
            trades: List of trades
            
        Returns:
            List of TimeBasedMetrics for each day
        """
        if not trades:
            return []
        
        # Create DataFrame for easier grouping
        df = pd.DataFrame([
            {
                'trade_date': t.trade_date,
                'pnl': float(t.pnl) if t.pnl is not None else 0,
                'symbol': t.symbol
            }
            for t in trades
        ])
        
        daily_metrics = []
        
        # Group by day
        for day, group in df.groupby('trade_date'):
            # Get winning trades for the day
            wins = group[group['pnl'] > 0]
            total_trades = len(group)
            
            metric = TimeBasedMetrics(
                period_start=day,
                period_end=day,
                period_type='daily',
                total_trades=total_trades,
                total_pnl=Decimal(str(group['pnl'].sum())),
                win_rate=(len(wins) / total_trades * 100) if total_trades > 0 else 0,
                average_pnl=Decimal(str(group['pnl'].mean())),
                best_day=day,
                worst_day=day,
                trading_days=1
            )
            daily_metrics.append(metric)
        
        return daily_metrics
    
    def calculate_yearly_metrics(self, trades: List[Trade]) -> List[TimeBasedMetrics]:
        """
        Calculate metrics aggregated by year.
        
        Args:
            trades: List of trades
            
        Returns:
            List of TimeBasedMetrics for each year
        """
        if not trades:
            return []
        
        # Create DataFrame for easier grouping
        df = pd.DataFrame([
            {
                'trade_date': t.trade_date,
                'pnl': float(t.pnl) if t.pnl is not None else 0,
                'symbol': t.symbol
            }
            for t in trades
        ])
        
        # Add year column
        df['year'] = pd.to_datetime(df['trade_date']).dt.year
        
        yearly_metrics = []
        
        # Group by year
        for year, group in df.groupby('year'):
            # Get winning trades for the year
            wins = group[group['pnl'] > 0]
            total_trades = len(group)
            
            # Find best and worst days
            daily_pnl = group.groupby('trade_date')['pnl'].sum()
            best_day = daily_pnl.idxmax() if not daily_pnl.empty else None
            worst_day = daily_pnl.idxmin() if not daily_pnl.empty else None
            
            metric = TimeBasedMetrics(
                period_start=date(year, 1, 1),
                period_end=date(year, 12, 31),
                period_type='yearly',
                total_trades=total_trades,
                total_pnl=Decimal(str(group['pnl'].sum())),
                win_rate=(len(wins) / total_trades * 100) if total_trades > 0 else 0,
                average_pnl=Decimal(str(group['pnl'].mean())),
                best_day=best_day,
                worst_day=worst_day,
                trading_days=group['trade_date'].nunique()
            )
            yearly_metrics.append(metric)
        
        return yearly_metrics
    
    def calculate_custom_period_metrics(self, trades: List[Trade], 
                                      start_date: date, 
                                      end_date: date) -> Optional[TimeBasedMetrics]:
        """
        Calculate metrics for a custom date range.
        
        Args:
            trades: List of trades
            start_date: Period start date
            end_date: Period end date
            
        Returns:
            TimeBasedMetrics for the period or None if no trades
        """
        # Filter trades within the period
        period_trades = [
            t for t in trades 
            if t.trade_date and start_date <= t.trade_date <= end_date
        ]
        
        if not period_trades:
            return None
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'trade_date': t.trade_date,
                'pnl': float(t.pnl) if t.pnl is not None else 0,
                'symbol': t.symbol
            }
            for t in period_trades
        ])
        
        # Get winning trades
        wins = df[df['pnl'] > 0]
        total_trades = len(df)
        
        # Find best and worst days
        daily_pnl = df.groupby('trade_date')['pnl'].sum()
        best_day = daily_pnl.idxmax() if not daily_pnl.empty else None
        worst_day = daily_pnl.idxmin() if not daily_pnl.empty else None
        
        return TimeBasedMetrics(
            period_start=start_date,
            period_end=end_date,
            period_type='custom',
            total_trades=total_trades,
            total_pnl=Decimal(str(df['pnl'].sum())),
            win_rate=(len(wins) / total_trades * 100) if total_trades > 0 else 0,
            average_pnl=Decimal(str(df['pnl'].mean())),
            best_day=best_day,
            worst_day=worst_day,
            trading_days=df['trade_date'].nunique()
        )
    
    def calculate_rolling_metrics(self, trades: List[Trade], window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling window metrics.
        
        Args:
            trades: List of trades
            window: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        if not trades:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'trade_date': t.trade_date,
                'pnl': float(t.pnl) if t.pnl is not None else 0,
                'symbol': t.symbol
            }
            for t in trades
        ])
        
        # Sort by date
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        
        # Create daily P&L series with all dates
        date_range = pd.date_range(
            start=df['trade_date'].min(),
            end=df['trade_date'].max(),
            freq='D'
        )
        
        # Aggregate P&L by day
        daily_pnl = df.groupby('trade_date')['pnl'].sum()
        daily_pnl = daily_pnl.reindex(date_range, fill_value=0)
        
        # Calculate rolling metrics
        rolling_df = pd.DataFrame(index=date_range)
        
        # Rolling sum of P&L
        rolling_df['rolling_pnl'] = daily_pnl.rolling(window=window, min_periods=1).sum()
        
        # Rolling trade count
        daily_trades = df.groupby('trade_date').size()
        daily_trades = daily_trades.reindex(date_range, fill_value=0)
        rolling_df['rolling_trades'] = daily_trades.rolling(window=window, min_periods=1).sum()
        
        # Rolling average P&L
        rolling_df['rolling_avg_pnl'] = rolling_df['rolling_pnl'] / rolling_df['rolling_trades'].replace(0, np.nan)
        
        # Rolling win rate
        wins = df[df['pnl'] > 0].groupby('trade_date').size()
        wins = wins.reindex(date_range, fill_value=0)
        rolling_wins = wins.rolling(window=window, min_periods=1).sum()
        rolling_df['rolling_win_rate'] = (rolling_wins / rolling_df['rolling_trades'].replace(0, np.nan)) * 100
        
        # Rolling volatility (standard deviation of daily returns)
        rolling_df['rolling_volatility'] = daily_pnl.rolling(window=window, min_periods=2).std()
        
        # Rolling Sharpe ratio (simplified - assuming 0 risk-free rate)
        rolling_mean = daily_pnl.rolling(window=window, min_periods=2).mean()
        rolling_std = daily_pnl.rolling(window=window, min_periods=2).std()
        rolling_df['rolling_sharpe'] = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)  # Annualized
        
        # Rolling maximum drawdown
        cumulative_pnl = daily_pnl.cumsum()
        rolling_df['cumulative_pnl'] = cumulative_pnl
        rolling_df['rolling_max_dd'] = self._calculate_rolling_drawdown(cumulative_pnl, window)
        
        # Fill NaN values
        rolling_df = rolling_df.fillna(0)
        
        return rolling_df
    
    def _calculate_rolling_drawdown(self, cumulative_pnl: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        rolling_dd = pd.Series(index=cumulative_pnl.index, dtype=float)
        
        for i in range(len(cumulative_pnl)):
            # Get window slice
            start_idx = max(0, i - window + 1)
            window_slice = cumulative_pnl.iloc[start_idx:i+1]
            
            if len(window_slice) > 0:
                # Calculate max drawdown in window
                running_max = window_slice.expanding(min_periods=1).max()
                drawdown = (window_slice - running_max) / running_max.replace(0, np.nan)
                max_dd = drawdown.min() if len(drawdown) > 0 else 0
                rolling_dd.iloc[i] = abs(max_dd) * 100  # Convert to percentage
            else:
                rolling_dd.iloc[i] = 0
        
        return rolling_dd
    
    def calculate_rolling_performance(self, 
                                    trades: List[Trade], 
                                    windows: List[int] = [7, 30, 90]) -> Dict[int, pd.DataFrame]:
        """
        Calculate rolling performance for multiple window sizes.
        
        Args:
            trades: List of trades
            windows: List of window sizes in days
            
        Returns:
            Dictionary mapping window size to rolling metrics DataFrame
        """
        results = {}
        
        for window in windows:
            results[window] = self.calculate_rolling_metrics(trades, window)
        
        return results
    
    def get_rolling_summary(self, trades: List[Trade], window: int = 30) -> Dict[str, Any]:
        """
        Get summary of current rolling window metrics.
        
        Args:
            trades: List of trades
            window: Rolling window size in days
            
        Returns:
            Dictionary with current rolling metrics
        """
        rolling_df = self.calculate_rolling_metrics(trades, window)
        
        if rolling_df.empty:
            return {
                'window_days': window,
                'current_pnl': 0,
                'current_trades': 0,
                'current_win_rate': 0,
                'current_volatility': 0,
                'current_sharpe': 0,
                'current_max_dd': 0
            }
        
        # Get latest values
        latest = rolling_df.iloc[-1]
        
        return {
            'window_days': window,
            'current_pnl': float(latest['rolling_pnl']),
            'current_trades': int(latest['rolling_trades']),
            'current_win_rate': float(latest['rolling_win_rate']),
            'current_volatility': float(latest['rolling_volatility']),
            'current_sharpe': float(latest['rolling_sharpe']),
            'current_max_dd': float(latest['rolling_max_dd'])
        }
    
    def calculate_returns_from_trades(self, trades: List[Trade]) -> np.ndarray:
        """
        Calculate returns from trades for use in advanced metrics.
        
        Args:
            trades: List of trades
            
        Returns:
            Array of returns (as decimals)
        """
        if not trades:
            return np.array([])
        
        # Sort trades by date
        sorted_trades = sorted(trades, key=lambda t: t.trade_date)
        
        # Group by date and calculate daily P&L
        daily_pnl = defaultdict(Decimal)
        for trade in sorted_trades:
            if trade.pnl is not None:
                daily_pnl[trade.trade_date] += trade.pnl
        
        if not daily_pnl:
            return np.array([])
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(list(daily_pnl.items()), columns=['date', 'pnl'])
        df = df.sort_values('date')
        
        # Calculate cumulative equity curve
        df['cumulative'] = df['pnl'].cumsum()
        
        # Add initial capital (assuming starting with 10000)
        initial_capital = Decimal('10000')
        df['equity'] = initial_capital + df['cumulative']
        
        # Calculate returns
        df['returns'] = df['equity'].pct_change()
        
        # Remove first NaN value
        returns = df['returns'].dropna().values
        
        return np.array([float(r) for r in returns])
    
    def calculate_equity_curve(self, trades: List[Trade], initial_capital: Decimal = Decimal('10000')) -> np.ndarray:
        """
        Calculate equity curve from trades.
        
        Args:
            trades: List of trades
            initial_capital: Starting capital
            
        Returns:
            Array of equity values
        """
        if not trades:
            return np.array([float(initial_capital)])
        
        # Sort trades by date
        sorted_trades = sorted(trades, key=lambda t: t.trade_date)
        
        # Group by date and calculate daily P&L
        daily_pnl = defaultdict(Decimal)
        for trade in sorted_trades:
            if trade.pnl is not None:
                daily_pnl[trade.trade_date] += trade.pnl
        
        # Create date range
        if daily_pnl:
            start_date = min(daily_pnl.keys())
            end_date = max(daily_pnl.keys())
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Build equity curve
            equity_curve = [float(initial_capital)]
            current_equity = initial_capital
            
            for date in date_range:
                date_key = date.date() if hasattr(date, 'date') else date
                daily_return = daily_pnl.get(date_key, Decimal('0'))
                current_equity += daily_return
                equity_curve.append(float(current_equity))
            
            return np.array(equity_curve)
        
        return np.array([float(initial_capital)])
    
    def calculate_advanced_metrics(self, trades: List[Trade], risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Calculate all advanced performance metrics.
        
        Args:
            trades: List of trades
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of advanced metrics
        """
        # Get returns and equity curve
        returns = self.calculate_returns_from_trades(trades)
        equity_curve = self.calculate_equity_curve(trades)
        
        # Get P&L values for profit factor
        pnl_values = [trade.pnl for trade in trades if trade.pnl is not None]
        
        # Calculate Sharpe ratio
        sharpe_ratio = self.advanced_metrics.calculate_sharpe_ratio(
            returns, risk_free_rate, frequency='daily'
        )
        
        # Calculate Sortino ratio
        sortino_ratio = self.advanced_metrics.calculate_sortino_ratio(
            returns, target_return=0.0, risk_free_rate=risk_free_rate, frequency='daily'
        )
        
        # Calculate profit factor
        profit_factor = self.advanced_metrics.calculate_profit_factor(pnl_values)
        
        # Calculate maximum drawdown
        max_drawdown, dd_details = self.advanced_metrics.calculate_max_drawdown(equity_curve)
        
        # Calculate Calmar ratio
        calmar_ratio = self.advanced_metrics.calculate_calmar_ratio(
            returns, max_drawdown, frequency='daily'
        ) if max_drawdown > 0 else None
        
        # Calculate VaR and CVaR
        var_95 = self.advanced_metrics.calculate_value_at_risk(
            returns, confidence_level=0.95, method='historical'
        )
        
        cvar_95 = self.advanced_metrics.calculate_conditional_var(
            returns, confidence_level=0.95
        )
        
        # Calculate Omega ratio
        omega_ratio = self.advanced_metrics.calculate_omega_ratio(returns, threshold=0.0)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_details': dd_details,
            'calmar_ratio': calmar_ratio,
            'value_at_risk_95': var_95,
            'conditional_var_95': cvar_95,
            'omega_ratio': omega_ratio,
            'total_trades': len(trades),
            'returns_count': len(returns)
        }
    
    def get_trades_dataframe(self, 
                            strategy_id: int, 
                            start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get trades as a pandas DataFrame with optional date filtering.
        
        Args:
            strategy_id: Strategy ID
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame with trade data
        """
        trades = self.get_trades_for_strategy(strategy_id)
        
        if not trades:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for trade in trades:
            data.append({
                'trade_date': trade.trade_date,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': float(trade.entry_price) if trade.entry_price else None,
                'exit_price': float(trade.exit_price) if trade.exit_price else None,
                'quantity': float(trade.quantity) if trade.quantity else None,
                'pnl': float(trade.pnl) if trade.pnl else 0,
                'commission': float(trade.commission) if trade.commission else 0
            })
        
        df = pd.DataFrame(data)
        
        # Apply date filtering
        if start_date:
            df = df[df['trade_date'] >= start_date]
        if end_date:
            df = df[df['trade_date'] <= end_date]
        
        return df