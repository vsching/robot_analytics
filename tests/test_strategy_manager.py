"""Tests for strategy management functionality."""

import pytest
from datetime import datetime
from decimal import Decimal
import tempfile
import os

from src.services import StrategyManager
from src.db.connection import DatabaseManager
from src.models import Strategy, Trade


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    db_manager = DatabaseManager(db_path)
    db_manager.initialize_database()
    
    yield db_manager
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def strategy_manager(test_db):
    """Create a StrategyManager with test database."""
    return StrategyManager(test_db)


class TestStrategyManager:
    def test_create_strategy_success(self, strategy_manager):
        """Test successful strategy creation."""
        strategy, error = strategy_manager.create_strategy(
            name="Test Strategy",
            description="A test trading strategy"
        )
        
        assert strategy is not None
        assert error is None
        assert strategy.id is not None
        assert strategy.name == "Test Strategy"
        assert strategy.description == "A test trading strategy"
        assert strategy.total_trades == 0
        assert strategy.total_pnl == Decimal("0")
    
    def test_create_strategy_duplicate_name(self, strategy_manager):
        """Test creating strategy with duplicate name."""
        # Create first strategy
        strategy1, _ = strategy_manager.create_strategy("Test Strategy")
        assert strategy1 is not None
        
        # Try to create duplicate
        strategy2, error = strategy_manager.create_strategy("Test Strategy")
        assert strategy2 is None
        assert error == "Strategy with name 'Test Strategy' already exists"
    
    def test_create_strategy_empty_name(self, strategy_manager):
        """Test creating strategy with empty name."""
        strategy, error = strategy_manager.create_strategy("")
        assert strategy is None
        assert error == "Strategy name cannot be empty"
        
        strategy, error = strategy_manager.create_strategy("   ")
        assert strategy is None
        assert error == "Strategy name cannot be empty"
    
    def test_get_strategy(self, strategy_manager):
        """Test retrieving strategy by ID."""
        # Create strategy
        created, _ = strategy_manager.create_strategy("Test Strategy")
        
        # Get by ID
        retrieved = strategy_manager.get_strategy(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name
        
        # Get non-existent
        assert strategy_manager.get_strategy(999) is None
    
    def test_get_strategies_pagination(self, strategy_manager):
        """Test retrieving strategies with pagination."""
        # Create multiple strategies
        for i in range(15):
            strategy_manager.create_strategy(f"Strategy {i+1}")
        
        # Get first page
        page1 = strategy_manager.get_strategies(limit=10, offset=0)
        assert len(page1) == 10
        
        # Get second page
        page2 = strategy_manager.get_strategies(limit=10, offset=10)
        assert len(page2) == 5
        
        # Verify no overlap
        page1_ids = {s.id for s in page1}
        page2_ids = {s.id for s in page2}
        assert len(page1_ids & page2_ids) == 0
    
    def test_search_strategies(self, strategy_manager):
        """Test searching strategies."""
        # Create strategies
        strategy_manager.create_strategy("Momentum Trading", "High frequency momentum")
        strategy_manager.create_strategy("Mean Reversion", "Statistical arbitrage")
        strategy_manager.create_strategy("Trend Following", "Long-term trends")
        
        # Search by name
        results = strategy_manager.get_strategies(search_query="Momentum")
        assert len(results) == 1
        assert results[0].name == "Momentum Trading"
        
        # Search by description
        results = strategy_manager.get_strategies(search_query="arbitrage")
        assert len(results) == 1
        assert results[0].name == "Mean Reversion"
        
        # Search with no matches
        results = strategy_manager.get_strategies(search_query="XYZ")
        assert len(results) == 0
    
    def test_update_strategy(self, strategy_manager):
        """Test updating strategy details."""
        # Create strategy
        strategy, _ = strategy_manager.create_strategy("Original Name", "Original desc")
        
        # Update name
        updated, error = strategy_manager.update_strategy(
            strategy.id,
            name="Updated Name"
        )
        assert error is None
        assert updated.name == "Updated Name"
        assert updated.description == "Original desc"
        
        # Update description
        updated, error = strategy_manager.update_strategy(
            strategy.id,
            description="Updated description"
        )
        assert error is None
        assert updated.name == "Updated Name"
        assert updated.description == "Updated description"
    
    def test_update_strategy_duplicate_name(self, strategy_manager):
        """Test updating strategy with duplicate name."""
        # Create two strategies
        strategy1, _ = strategy_manager.create_strategy("Strategy 1")
        strategy2, _ = strategy_manager.create_strategy("Strategy 2")
        
        # Try to rename strategy2 to strategy1's name
        updated, error = strategy_manager.update_strategy(
            strategy2.id,
            name="Strategy 1"
        )
        assert updated is None
        assert error == "Strategy with name 'Strategy 1' already exists"
    
    def test_delete_strategy_no_cascade(self, strategy_manager):
        """Test deleting strategy without cascade."""
        # Create strategy
        strategy, _ = strategy_manager.create_strategy("Test Strategy")
        
        # Delete without trades
        success, error = strategy_manager.delete_strategy(strategy.id, cascade=False)
        assert success is True
        assert error is None
        
        # Verify deleted
        assert strategy_manager.get_strategy(strategy.id) is None
    
    def test_delete_strategy_with_trades_no_cascade(self, strategy_manager):
        """Test deleting strategy with trades (no cascade)."""
        # Create strategy and add trades
        strategy, _ = strategy_manager.create_strategy("Test Strategy")
        
        trades = [
            Trade(
                strategy_id=strategy.id,
                trade_date=datetime.now().date(),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                quantity=Decimal("100"),
                pnl=Decimal("500.00")
            )
        ]
        
        strategy_manager.append_trades(strategy.id, trades)
        
        # Try to delete without cascade
        success, error = strategy_manager.delete_strategy(strategy.id, cascade=False)
        assert success is False
        assert "Cannot delete strategy with 1 trades" in error
    
    def test_delete_strategy_cascade(self, strategy_manager):
        """Test deleting strategy with cascade."""
        # Create strategy and add trades
        strategy, _ = strategy_manager.create_strategy("Test Strategy")
        
        trades = [
            Trade(
                strategy_id=strategy.id,
                trade_date=datetime.now().date(),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                quantity=Decimal("100"),
                pnl=Decimal("500.00")
            )
        ]
        
        strategy_manager.append_trades(strategy.id, trades)
        
        # Delete with cascade
        success, error = strategy_manager.delete_strategy(strategy.id, cascade=True)
        assert success is True
        assert error is None
        
        # Verify deleted
        assert strategy_manager.get_strategy(strategy.id) is None
    
    def test_append_trades(self, strategy_manager):
        """Test appending trades to strategy."""
        # Create strategy
        strategy, _ = strategy_manager.create_strategy("Test Strategy")
        
        # Create trades
        trades = [
            Trade(
                trade_date=datetime.now().date(),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                quantity=Decimal("100"),
                pnl=Decimal("500.00"),
                commission=Decimal("10.00")
            ),
            Trade(
                trade_date=datetime.now().date(),
                symbol="MSFT",
                side="sell",
                entry_price=Decimal("380.00"),
                exit_price=Decimal("375.00"),
                quantity=Decimal("50"),
                pnl=Decimal("-250.00"),
                commission=Decimal("10.00")
            )
        ]
        
        # Append trades
        count, error = strategy_manager.append_trades(strategy.id, trades)
        assert count == 2
        assert error is None
        
        # Verify statistics updated
        updated_strategy = strategy_manager.get_strategy(strategy.id)
        assert updated_strategy.total_trades == 2
        assert updated_strategy.total_pnl == Decimal("250.00")  # 500 - 250
    
    def test_replace_trades(self, strategy_manager):
        """Test replacing all trades for a strategy."""
        # Create strategy with initial trades
        strategy, _ = strategy_manager.create_strategy("Test Strategy")
        
        initial_trades = [
            Trade(
                trade_date=datetime.now().date(),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                quantity=Decimal("100"),
                pnl=Decimal("500.00")
            )
        ]
        
        strategy_manager.append_trades(strategy.id, initial_trades)
        
        # Replace with new trades
        new_trades = [
            Trade(
                trade_date=datetime.now().date(),
                symbol="MSFT",
                side="sell",
                entry_price=Decimal("380.00"),
                exit_price=Decimal("375.00"),
                quantity=Decimal("50"),
                pnl=Decimal("-250.00")
            ),
            Trade(
                trade_date=datetime.now().date(),
                symbol="GOOGL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("160.00"),
                quantity=Decimal("20"),
                pnl=Decimal("200.00")
            )
        ]
        
        count, error = strategy_manager.replace_trades(strategy.id, new_trades)
        assert count == 2
        assert error is None
        
        # Verify statistics updated
        updated_strategy = strategy_manager.get_strategy(strategy.id)
        assert updated_strategy.total_trades == 2
        assert updated_strategy.total_pnl == Decimal("-50.00")  # -250 + 200
    
    def test_get_strategy_statistics(self, strategy_manager):
        """Test getting detailed strategy statistics."""
        # Create strategy with trades
        strategy, _ = strategy_manager.create_strategy("Test Strategy")
        
        trades = [
            Trade(
                trade_date=datetime.now().date(),
                symbol="AAPL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                quantity=Decimal("100"),
                pnl=Decimal("500.00")
            ),
            Trade(
                trade_date=datetime.now().date(),
                symbol="MSFT",
                side="sell",
                entry_price=Decimal("380.00"),
                exit_price=Decimal("375.00"),
                quantity=Decimal("50"),
                pnl=Decimal("-250.00")
            ),
            Trade(
                trade_date=datetime.now().date(),
                symbol="GOOGL",
                side="buy",
                entry_price=Decimal("150.00"),
                exit_price=Decimal("150.00"),
                quantity=Decimal("20"),
                pnl=Decimal("0.00")
            )
        ]
        
        strategy_manager.append_trades(strategy.id, trades)
        
        # Get statistics
        stats = strategy_manager.get_strategy_statistics(strategy.id)
        
        assert stats['trade_count'] == 3
        assert stats['total_pnl'] == 250.0  # 500 - 250 + 0
        assert stats['avg_pnl'] == pytest.approx(83.33, rel=0.01)
        assert stats['max_pnl'] == 500.0
        assert stats['min_pnl'] == -250.0
        assert stats['winning_trades'] == 1
        assert stats['losing_trades'] == 1
        assert stats['breakeven_trades'] == 1
        assert stats['win_rate'] == pytest.approx(33.33, rel=0.01)
    
    def test_count_strategies(self, strategy_manager):
        """Test counting strategies."""
        # Initially empty
        assert strategy_manager.count_strategies() == 0
        
        # Add strategies
        for i in range(5):
            strategy_manager.create_strategy(f"Strategy {i+1}")
        
        assert strategy_manager.count_strategies() == 5
        
        # Count with search
        assert strategy_manager.count_strategies("Strategy 1") == 1
        assert strategy_manager.count_strategies("Strategy") == 5
        assert strategy_manager.count_strategies("XYZ") == 0