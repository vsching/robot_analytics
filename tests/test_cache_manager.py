"""Tests for cache manager functionality."""

import pytest
from datetime import datetime, timedelta, date
from decimal import Decimal
import tempfile
import os

from src.analytics.cache_manager import MetricsCacheManager, CachedAnalyticsEngine
from src.analytics import AnalyticsEngine, PerformanceMetrics, PnLSummary, TradeStatistics
from src.db.connection import DatabaseManager
from src.models import Trade


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db_manager = DatabaseManager(db_path)
    
    # Initialize schema
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        # Create performance_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                period_type TEXT NOT NULL,
                period_start DATE,
                period_end DATE,
                total_pnl REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                win_count INTEGER NOT NULL,
                loss_count INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                calculated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE
            )
        """)
        
        # Create strategies table for foreign key
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0
            )
        """)
        
        # Insert test strategy
        cursor.execute(
            "INSERT INTO strategies (id, name, description) VALUES (1, 'Test Strategy', 'Test')"
        )
        
        conn.commit()
    
    yield db_manager
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def cache_manager(temp_db):
    """Create a cache manager instance."""
    return MetricsCacheManager(temp_db)


@pytest.fixture
def sample_metrics():
    """Create sample performance metrics."""
    return PerformanceMetrics(
        strategy_id=1,
        calculated_at=datetime.utcnow(),
        pnl_summary=PnLSummary(
            total_pnl=Decimal("1000.00"),
            average_pnl=Decimal("100.00"),
            max_pnl=Decimal("500.00"),
            min_pnl=Decimal("-200.00"),
            median_pnl=Decimal("75.00"),
            std_dev=Decimal("150.00"),
            total_commission=Decimal("50.00"),
            net_pnl=Decimal("950.00")
        ),
        trade_statistics=TradeStatistics(
            total_trades=10,
            winning_trades=6,
            losing_trades=3,
            breakeven_trades=1,
            win_rate=60.0,
            loss_rate=30.0,
            average_win=Decimal("250.00"),
            average_loss=Decimal("-150.00"),
            largest_win=Decimal("500.00"),
            largest_loss=Decimal("-200.00"),
            win_loss_ratio=1.67,
            profit_factor=2.5,
            expectancy=Decimal("100.00"),
            consecutive_wins=2,
            consecutive_losses=0,
            max_consecutive_wins=4,
            max_consecutive_losses=2
        ),
        monthly_metrics=[],
        weekly_metrics=[]
    )


class TestMetricsCacheManager:
    def test_cache_metrics(self, cache_manager, sample_metrics):
        """Test caching metrics."""
        success = cache_manager.cache_metrics(sample_metrics)
        assert success is True
        
        # Verify data was stored
        with cache_manager.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            count = cursor.fetchone()[0]
            assert count == 1
    
    def test_get_cached_metrics(self, cache_manager, sample_metrics):
        """Test retrieving cached metrics."""
        # Cache metrics first
        cache_manager.cache_metrics(sample_metrics)
        
        # Retrieve cached metrics
        cached = cache_manager.get_cached_metrics(1, 'all-time')
        assert cached is not None
        assert cached.strategy_id == 1
        assert cached.pnl_summary.total_pnl == Decimal("1000.00")
        assert cached.trade_statistics.total_trades == 10
        assert cached.trade_statistics.win_rate == 60.0
    
    def test_cache_expiration(self, cache_manager, sample_metrics):
        """Test cache expiration."""
        # Set short TTL
        cache_manager.set_cache_ttl(0.001)  # 0.001 hours = 3.6 seconds
        
        # Cache metrics
        cache_manager.cache_metrics(sample_metrics)
        
        # Should be valid immediately
        cached = cache_manager.get_cached_metrics(1, 'all-time')
        assert cached is not None
        
        # Wait for expiration
        import time
        time.sleep(4)
        
        # Should be expired now
        cached = cache_manager.get_cached_metrics(1, 'all-time')
        assert cached is None
    
    def test_invalidate_cache(self, cache_manager, sample_metrics):
        """Test cache invalidation."""
        # Cache metrics
        cache_manager.cache_metrics(sample_metrics)
        
        # Verify cached
        cached = cache_manager.get_cached_metrics(1, 'all-time')
        assert cached is not None
        
        # Invalidate
        success = cache_manager.invalidate_cache(1)
        assert success is True
        
        # Should be gone
        cached = cache_manager.get_cached_metrics(1, 'all-time')
        assert cached is None
    
    def test_invalidate_all_caches(self, cache_manager, sample_metrics):
        """Test invalidating all caches."""
        # Cache metrics for multiple strategies
        cache_manager.cache_metrics(sample_metrics)
        
        # Create and cache metrics for strategy 2
        with cache_manager.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO strategies (id, name) VALUES (2, 'Strategy 2')"
            )
            conn.commit()
        
        metrics2 = sample_metrics
        metrics2.strategy_id = 2
        cache_manager.cache_metrics(metrics2)
        
        # Verify both cached
        assert cache_manager.get_cached_metrics(1, 'all-time') is not None
        assert cache_manager.get_cached_metrics(2, 'all-time') is not None
        
        # Invalidate all
        success = cache_manager.invalidate_all_caches()
        assert success is True
        
        # Both should be gone
        assert cache_manager.get_cached_metrics(1, 'all-time') is None
        assert cache_manager.get_cached_metrics(2, 'all-time') is None
    
    def test_get_cache_stats(self, cache_manager, sample_metrics):
        """Test cache statistics."""
        # Cache some metrics
        cache_manager.cache_metrics(sample_metrics)
        
        stats = cache_manager.get_cache_stats()
        
        assert stats['total_entries'] == 1
        assert stats['expired_entries'] == 0
        assert stats['valid_entries'] == 1
        assert 1 in stats['entries_by_strategy']
        assert stats['entries_by_strategy'][1] == 1
        assert stats['cache_ttl_hours'] == 1.0
    
    def test_cleanup_expired_cache(self, cache_manager, sample_metrics):
        """Test cleaning up expired cache entries."""
        # Set very short TTL
        cache_manager.set_cache_ttl(0.0001)  # Very short
        
        # Cache metrics
        cache_manager.cache_metrics(sample_metrics)
        
        # Wait for expiration
        import time
        time.sleep(1)
        
        # Cleanup expired
        deleted = cache_manager.cleanup_expired_cache()
        assert deleted == 1
        
        # Verify cleaned up
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] == 0


class TestCachedAnalyticsEngine:
    def test_calculate_with_cache(self, temp_db, sample_metrics):
        """Test analytics engine with caching."""
        # Create mock analytics engine
        class MockAnalyticsEngine:
            def __init__(self):
                self.call_count = 0
            
            def calculate_metrics_for_strategy(self, strategy_id):
                self.call_count += 1
                return sample_metrics
        
        mock_engine = MockAnalyticsEngine()
        cache_manager = MetricsCacheManager(temp_db)
        cached_engine = CachedAnalyticsEngine(mock_engine, cache_manager)
        
        # First call should calculate
        metrics1 = cached_engine.calculate_metrics_for_strategy(1)
        assert metrics1 is not None
        assert mock_engine.call_count == 1
        
        # Second call should use cache
        metrics2 = cached_engine.calculate_metrics_for_strategy(1)
        assert metrics2 is not None
        assert mock_engine.call_count == 1  # No additional call
        
        # Force refresh should calculate again
        metrics3 = cached_engine.calculate_metrics_for_strategy(1, force_refresh=True)
        assert metrics3 is not None
        assert mock_engine.call_count == 2
    
    def test_invalidate_strategy_cache(self, temp_db, sample_metrics):
        """Test invalidating strategy cache through cached engine."""
        mock_engine = type('MockEngine', (), {
            'calculate_metrics_for_strategy': lambda self, sid: sample_metrics
        })()
        
        cache_manager = MetricsCacheManager(temp_db)
        cached_engine = CachedAnalyticsEngine(mock_engine, cache_manager)
        
        # Cache metrics
        cached_engine.calculate_metrics_for_strategy(1)
        
        # Verify cached
        assert cache_manager.get_cached_metrics(1, 'all-time') is not None
        
        # Invalidate
        cached_engine.invalidate_strategy_cache(1)
        
        # Verify invalidated
        assert cache_manager.get_cached_metrics(1, 'all-time') is None
    
    def test_get_cache_info(self, temp_db, sample_metrics):
        """Test getting cache info through cached engine."""
        mock_engine = type('MockEngine', (), {})()
        cache_manager = MetricsCacheManager(temp_db)
        cached_engine = CachedAnalyticsEngine(mock_engine, cache_manager)
        
        info = cached_engine.get_cache_info()
        
        assert 'total_entries' in info
        assert 'expired_entries' in info
        assert 'valid_entries' in info
        assert 'entries_by_strategy' in info
        assert 'cache_ttl_hours' in info


class TestCacheWarming:
    def test_warm_cache(self, temp_db, sample_metrics):
        """Test cache warming functionality."""
        # Create analytics engine that returns metrics
        class MockAnalyticsEngine:
            def calculate_metrics_for_strategy(self, strategy_id):
                metrics = sample_metrics
                metrics.strategy_id = strategy_id
                return metrics
        
        mock_engine = MockAnalyticsEngine()
        cache_manager = MetricsCacheManager(temp_db)
        
        # Create additional strategies
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO strategies (id, name) VALUES (2, 'Strategy 2')")
            cursor.execute("INSERT INTO strategies (id, name) VALUES (3, 'Strategy 3')")
            conn.commit()
        
        # Warm cache for multiple strategies
        results = cache_manager.warm_cache([1, 2, 3], mock_engine)
        
        assert results[1] is True
        assert results[2] is True
        assert results[3] is True
        
        # Verify all are cached
        assert cache_manager.get_cached_metrics(1, 'all-time') is not None
        assert cache_manager.get_cached_metrics(2, 'all-time') is not None
        assert cache_manager.get_cached_metrics(3, 'all-time') is not None
    
    def test_warm_cache_skip_existing(self, temp_db, sample_metrics):
        """Test cache warming skips already cached metrics."""
        class MockAnalyticsEngine:
            def __init__(self):
                self.call_count = 0
            
            def calculate_metrics_for_strategy(self, strategy_id):
                self.call_count += 1
                metrics = sample_metrics
                metrics.strategy_id = strategy_id
                return metrics
        
        mock_engine = MockAnalyticsEngine()
        cache_manager = MetricsCacheManager(temp_db)
        
        # Pre-cache strategy 1
        cache_manager.cache_metrics(sample_metrics)
        
        # Warm cache - should only calculate for strategy 2
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO strategies (id, name) VALUES (2, 'Strategy 2')")
            conn.commit()
        
        results = cache_manager.warm_cache([1, 2], mock_engine)
        
        assert results[1] is True  # Already cached
        assert results[2] is True  # Newly cached
        assert mock_engine.call_count == 1  # Only called for strategy 2