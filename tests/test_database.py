"""Tests for database functionality."""

import pytest
import tempfile
import os
from datetime import date, datetime
from decimal import Decimal

from src.db import DatabaseManager, StrategyRepository, TradeRepository
from src.models import Strategy, Trade


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db_manager = DatabaseManager(db_path=db_path)
    yield db_manager
    
    # Cleanup
    db_manager.close()
    os.unlink(db_path)


@pytest.fixture
def strategy_repo(test_db):
    """Create a strategy repository with test database."""
    # Monkey patch to use test database
    import src.db.connection
    original_get_db = src.db.connection.get_db_manager
    src.db.connection.get_db_manager = lambda: test_db
    
    repo = StrategyRepository()
    yield repo
    
    # Restore original
    src.db.connection.get_db_manager = original_get_db


@pytest.fixture
def trade_repo(test_db):
    """Create a trade repository with test database."""
    # Monkey patch to use test database
    import src.db.connection
    original_get_db = src.db.connection.get_db_manager
    src.db.connection.get_db_manager = lambda: test_db
    
    repo = TradeRepository()
    yield repo
    
    # Restore original
    src.db.connection.get_db_manager = original_get_db


class TestDatabaseManager:
    def test_connection(self, test_db):
        """Test database connection."""
        with test_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_schema_creation(self, test_db):
        """Test that schema is created properly."""
        with test_db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if strategies table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='strategies'
            """)
            assert cursor.fetchone() is not None
            
            # Check if trades table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='trades'
            """)
            assert cursor.fetchone() is not None


class TestStrategyRepository:
    def test_create_strategy(self, strategy_repo):
        """Test creating a strategy."""
        strategy = Strategy(
            name="Test Strategy",
            description="A test trading strategy"
        )
        
        created = strategy_repo.create(strategy)
        assert created.id is not None
        assert created.name == "Test Strategy"
    
    def test_get_strategy_by_id(self, strategy_repo):
        """Test retrieving strategy by ID."""
        # Create a strategy
        strategy = Strategy(name="Test Strategy")
        created = strategy_repo.create(strategy)
        
        # Retrieve it
        retrieved = strategy_repo.get_by_id(created.id)
        assert retrieved is not None
        assert retrieved.name == "Test Strategy"
        assert retrieved.id == created.id
    
    def test_update_strategy(self, strategy_repo):
        """Test updating a strategy."""
        # Create a strategy
        strategy = Strategy(name="Test Strategy")
        created = strategy_repo.create(strategy)
        
        # Update it
        created.description = "Updated description"
        created.total_pnl = Decimal('1000.50')
        success = strategy_repo.update(created)
        
        assert success is True
        
        # Verify update
        retrieved = strategy_repo.get_by_id(created.id)
        assert retrieved.description == "Updated description"
        assert retrieved.total_pnl == Decimal('1000.50')
    
    def test_delete_strategy(self, strategy_repo):
        """Test deleting a strategy."""
        # Create a strategy
        strategy = Strategy(name="Test Strategy")
        created = strategy_repo.create(strategy)
        
        # Delete it
        success = strategy_repo.delete(created.id)
        assert success is True
        
        # Verify deletion
        retrieved = strategy_repo.get_by_id(created.id)
        assert retrieved is None


class TestTradeRepository:
    def test_create_trade(self, strategy_repo, trade_repo):
        """Test creating a trade."""
        # First create a strategy
        strategy = Strategy(name="Test Strategy")
        strategy = strategy_repo.create(strategy)
        
        # Create a trade
        trade = Trade(
            strategy_id=strategy.id,
            trade_date=date.today(),
            symbol="BTC/USD",
            side="buy",
            entry_price=Decimal('50000'),
            exit_price=Decimal('51000'),
            quantity=Decimal('0.1'),
            pnl=Decimal('100')
        )
        
        created = trade_repo.create(trade)
        assert created.id is not None
        assert created.symbol == "BTC/USD"
    
    def test_get_trades_by_strategy(self, strategy_repo, trade_repo):
        """Test retrieving trades by strategy."""
        # Create a strategy
        strategy = Strategy(name="Test Strategy")
        strategy = strategy_repo.create(strategy)
        
        # Create multiple trades
        trades = []
        for i in range(3):
            trade = Trade(
                strategy_id=strategy.id,
                trade_date=date.today(),
                symbol=f"SYMBOL{i}",
                side="buy",
                pnl=Decimal(str(100 * i))
            )
            trades.append(trade_repo.create(trade))
        
        # Retrieve trades
        retrieved = trade_repo.get_by_strategy(strategy.id)
        assert len(retrieved) == 3
        assert all(t.strategy_id == strategy.id for t in retrieved)
    
    def test_bulk_create_trades(self, strategy_repo, trade_repo):
        """Test bulk creating trades."""
        # Create a strategy
        strategy = Strategy(name="Test Strategy")
        strategy = strategy_repo.create(strategy)
        
        # Create multiple trades
        trades = []
        for i in range(5):
            trade = Trade(
                strategy_id=strategy.id,
                trade_date=date.today(),
                symbol=f"SYMBOL{i}",
                side="buy" if i % 2 == 0 else "sell",
                pnl=Decimal(str(50 * i))
            )
            trades.append(trade)
        
        # Bulk create
        count = trade_repo.bulk_create(trades)
        assert count == 5
        
        # Verify creation
        retrieved = trade_repo.get_by_strategy(strategy.id)
        assert len(retrieved) == 5