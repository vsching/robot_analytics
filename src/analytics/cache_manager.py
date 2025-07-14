"""Cache manager for performance metrics."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import hashlib
import logging

from ..db.connection import DatabaseManager
from .metrics import PerformanceMetrics, PnLSummary, TradeStatistics, TimeBasedMetrics


logger = logging.getLogger(__name__)


class MetricsCacheManager:
    """Manages caching of calculated performance metrics."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self._cache_ttl = timedelta(hours=1)  # Default cache TTL
        self._memory_cache = {}  # In-memory cache for general data
    
    def get_cached_metrics(self, strategy_id: int, metric_type: str = 'all-time') -> Optional[PerformanceMetrics]:
        """
        Get cached metrics for a strategy.
        
        Args:
            strategy_id: Strategy ID
            metric_type: Type of metrics (all-time, monthly, etc.)
            
        Returns:
            Cached PerformanceMetrics or None if not found/expired
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM performance_metrics
                WHERE strategy_id = ? AND period_type = ?
                ORDER BY calculated_at DESC
                LIMIT 1
                """,
                (strategy_id, metric_type)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Check if cache is still valid
            calculated_at = datetime.fromisoformat(row[15])  # calculated_at column
            if datetime.utcnow() - calculated_at > self._cache_ttl:
                logger.info(f"Cache expired for strategy {strategy_id}")
                return None
            
            # Reconstruct PerformanceMetrics from row
            logger.info(f"Cache hit for strategy {strategy_id}")
            
            # Create PnLSummary
            pnl_summary = PnLSummary(
                total_pnl=Decimal(str(row[4])),  # total_pnl
                average_pnl=Decimal(str(row[4] / row[5])) if row[5] > 0 else Decimal("0"),  # total_pnl / trade_count
                max_pnl=Decimal("0"),  # Not stored in cache table
                min_pnl=Decimal("0"),  # Not stored in cache table
                median_pnl=Decimal("0"),  # Not stored in cache table
                std_dev=Decimal("0"),  # Not stored in cache table
                total_commission=Decimal("0"),  # Not stored in cache table
                net_pnl=Decimal(str(row[4]))  # total_pnl (assuming commission already deducted)
            )
            
            # Create TradeStatistics
            trade_statistics = TradeStatistics(
                total_trades=row[5],  # trade_count
                winning_trades=row[6],  # win_count
                losing_trades=row[7],  # loss_count
                breakeven_trades=row[5] - row[6] - row[7],  # Calculate breakeven
                win_rate=row[8],  # win_rate
                loss_rate=(row[7] / row[5] * 100) if row[5] > 0 else 0,
                average_win=Decimal(str(row[9])) if row[9] is not None else Decimal("0"),  # avg_win
                average_loss=Decimal(str(row[10])) if row[10] is not None else Decimal("0"),  # avg_loss
                largest_win=Decimal("0"),  # Not stored in cache table
                largest_loss=Decimal("0"),  # Not stored in cache table
                win_loss_ratio=abs(row[9] / row[10]) if row[10] and row[10] != 0 else 0,
                profit_factor=row[11] if row[11] is not None else 0,  # profit_factor
                expectancy=Decimal("0"),  # Calculate from win rate and averages
                consecutive_wins=0,  # Not stored in cache table
                consecutive_losses=0,  # Not stored in cache table
                max_consecutive_wins=0,  # Not stored in cache table
                max_consecutive_losses=0  # Not stored in cache table
            )
            
            # For cached metrics, we'll return empty lists for time-based metrics
            # These would need to be calculated fresh if needed
            return PerformanceMetrics(
                strategy_id=strategy_id,
                calculated_at=calculated_at,
                pnl_summary=pnl_summary,
                trade_statistics=trade_statistics,
                monthly_metrics=[],
                weekly_metrics=[]
            )
    
    def cache_metrics(self, metrics: PerformanceMetrics) -> bool:
        """
        Cache calculated metrics.
        
        Args:
            metrics: PerformanceMetrics to cache
            
        Returns:
            True if successfully cached
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Cache main metrics
                cursor.execute(
                    """
                    INSERT INTO performance_metrics (
                        strategy_id, period_type, period_start, period_end,
                        total_pnl, trade_count, win_count, loss_count,
                        win_rate, avg_win, avg_loss, profit_factor,
                        sharpe_ratio, sortino_ratio, max_drawdown,
                        calculated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metrics.strategy_id,
                        'all-time',
                        None,  # period_start
                        None,  # period_end
                        float(metrics.pnl_summary.total_pnl),
                        metrics.trade_statistics.total_trades,
                        metrics.trade_statistics.winning_trades,
                        metrics.trade_statistics.losing_trades,
                        metrics.trade_statistics.win_rate,
                        float(metrics.trade_statistics.average_win),
                        float(metrics.trade_statistics.average_loss),
                        metrics.trade_statistics.profit_factor,
                        None,  # sharpe_ratio - will be calculated in advanced metrics
                        None,  # sortino_ratio - will be calculated in advanced metrics
                        None,  # max_drawdown - will be calculated in advanced metrics
                        metrics.calculated_at
                    )
                )
                
                conn.commit()
                logger.info(f"Cached metrics for strategy {metrics.strategy_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cache metrics: {str(e)}")
            return False
    
    def invalidate_cache(self, strategy_id: int) -> bool:
        """
        Invalidate cached metrics for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if successfully invalidated
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM performance_metrics WHERE strategy_id = ?",
                    (strategy_id,)
                )
                conn.commit()
                
                deleted_count = cursor.rowcount
                logger.info(f"Invalidated {deleted_count} cached entries for strategy {strategy_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {str(e)}")
            return False
    
    def invalidate_all_caches(self) -> bool:
        """
        Invalidate all cached metrics.
        
        Returns:
            True if successfully invalidated
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM performance_metrics")
                conn.commit()
                
                deleted_count = cursor.rowcount
                logger.info(f"Invalidated {deleted_count} cached entries")
                return True
                
        except Exception as e:
            logger.error(f"Failed to invalidate all caches: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total cache entries
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            total_entries = cursor.fetchone()[0]
            
            # Expired entries
            cutoff_time = datetime.utcnow() - self._cache_ttl
            cursor.execute(
                "SELECT COUNT(*) FROM performance_metrics WHERE calculated_at < ?",
                (cutoff_time,)
            )
            expired_entries = cursor.fetchone()[0]
            
            # Cache by strategy
            cursor.execute(
                """
                SELECT strategy_id, COUNT(*) 
                FROM performance_metrics 
                GROUP BY strategy_id
                """
            )
            by_strategy = dict(cursor.fetchall())
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'valid_entries': total_entries - expired_entries,
                'entries_by_strategy': by_strategy,
                'cache_ttl_hours': self._cache_ttl.total_seconds() / 3600
            }
    
    def warm_cache(self, strategy_ids: List[int], analytics_engine: Any) -> Dict[int, bool]:
        """
        Warm cache for specified strategies.
        
        Args:
            strategy_ids: List of strategy IDs to warm
            analytics_engine: AnalyticsEngine instance
            
        Returns:
            Dictionary mapping strategy_id to success status
        """
        results = {}
        
        for strategy_id in strategy_ids:
            try:
                # Check if already cached and valid
                existing = self.get_cached_metrics(strategy_id)
                if existing:
                    results[strategy_id] = True
                    continue
                
                # Calculate and cache metrics
                metrics = analytics_engine.calculate_metrics_for_strategy(strategy_id)
                if metrics:
                    success = self.cache_metrics(metrics)
                    results[strategy_id] = success
                else:
                    results[strategy_id] = False
                    
            except Exception as e:
                logger.error(f"Failed to warm cache for strategy {strategy_id}: {str(e)}")
                results[strategy_id] = False
        
        return results
    
    def cleanup_expired_cache(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        try:
            cutoff_time = datetime.utcnow() - self._cache_ttl
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM performance_metrics WHERE calculated_at < ?",
                    (cutoff_time,)
                )
                conn.commit()
                
                deleted_count = cursor.rowcount
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {str(e)}")
            return 0
    
    def set_cache_ttl(self, hours: float):
        """
        Set cache TTL.
        
        Args:
            hours: TTL in hours
        """
        self._cache_ttl = timedelta(hours=hours)
        logger.info(f"Set cache TTL to {hours} hours")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if datetime.utcnow() < entry['expires_at']:
                return entry['value']
            else:
                # Remove expired entry
                del self._memory_cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in memory cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 300)
        """
        if ttl is None:
            ttl = 300  # 5 minutes default
        
        self._memory_cache[key] = {
            'value': value,
            'expires_at': datetime.utcnow() + timedelta(seconds=ttl)
        }


class CachedAnalyticsEngine:
    """Analytics engine wrapper with caching support."""
    
    def __init__(self, analytics_engine: Any, cache_manager: Optional[MetricsCacheManager] = None):
        self.engine = analytics_engine
        self.cache = cache_manager or MetricsCacheManager()
    
    def calculate_metrics_for_strategy(self, strategy_id: int, force_refresh: bool = False) -> Optional[PerformanceMetrics]:
        """
        Calculate metrics with caching support.
        
        Args:
            strategy_id: Strategy ID
            force_refresh: Force recalculation even if cached
            
        Returns:
            PerformanceMetrics or None
        """
        # Check cache first
        if not force_refresh:
            cached = self.cache.get_cached_metrics(strategy_id)
            if cached:
                return cached
        
        # Calculate fresh metrics
        metrics = self.engine.calculate_metrics_for_strategy(strategy_id)
        
        # Cache the results
        if metrics:
            self.cache.cache_metrics(metrics)
        
        return metrics
    
    def invalidate_strategy_cache(self, strategy_id: int):
        """Invalidate cache when strategy data changes."""
        self.cache.invalidate_cache(strategy_id)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return self.cache.get_cache_stats()
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped engine."""
        return getattr(self.engine, name)