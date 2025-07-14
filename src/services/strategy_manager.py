"""Strategy management service for CRUD operations."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from decimal import Decimal
import logging

from ..db.connection import DatabaseManager
from ..models import Strategy, Trade
from ..db.base_repository import BaseRepository


logger = logging.getLogger(__name__)


class StrategyRepository(BaseRepository[Strategy]):
    """Repository for strategy-specific operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, 'strategies', Strategy)
    
    def get_by_name(self, name: str) -> Optional[Strategy]:
        """Get strategy by name."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {self.table_name} WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            return self._row_to_model(row) if row else None
    
    def search(self, query: str, limit: int = 50, offset: int = 0) -> List[Strategy]:
        """Search strategies by name or description."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (f'%{query}%', f'%{query}%', limit, offset)
            )
            rows = cursor.fetchall()
            return [self._row_to_model(row) for row in rows]
    
    def update_statistics(self, strategy_id: int) -> None:
        """Update strategy statistics (total_trades, total_pnl)."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate statistics from trades
            cursor.execute(
                """
                UPDATE strategies
                SET total_trades = (
                    SELECT COUNT(*) FROM trades WHERE strategy_id = ?
                ),
                total_pnl = (
                    SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE strategy_id = ?
                ),
                updated_at = ?
                WHERE id = ?
                """,
                (strategy_id, strategy_id, datetime.utcnow(), strategy_id)
            )
            conn.commit()


class StrategyManager:
    """Service for managing trading strategies."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.repository = StrategyRepository(self.db_manager)
    
    def create_strategy(self, name: str, description: str = "") -> Tuple[Optional[Strategy], Optional[str]]:
        """
        Create a new trading strategy.
        
        Args:
            name: Unique strategy name
            description: Strategy description
            
        Returns:
            Tuple of (created strategy, error message if any)
        """
        try:
            # Validate name
            if not name or not name.strip():
                return None, "Strategy name cannot be empty"
            
            name = name.strip()
            
            # Check for duplicate
            existing = self.repository.get_by_name(name)
            if existing:
                return None, f"Strategy with name '{name}' already exists"
            
            # Create strategy
            strategy = Strategy(
                name=name,
                description=description,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                total_trades=0,
                total_pnl=Decimal("0")
            )
            
            created = self.repository.create(strategy)
            logger.info(f"Created strategy: {created.name} (ID: {created.id})")
            
            return created, None
            
        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            return None, f"Failed to create strategy: {str(e)}"
    
    def get_strategy(self, strategy_id: int) -> Optional[Strategy]:
        """Get strategy by ID."""
        return self.repository.get_by_id(strategy_id)
    
    def get_strategies(self, 
                      limit: int = 50, 
                      offset: int = 0,
                      search_query: Optional[str] = None) -> List[Strategy]:
        """
        Get strategies with pagination and optional search.
        
        Args:
            limit: Maximum number of strategies to return
            offset: Number of strategies to skip
            search_query: Optional search query for name/description
            
        Returns:
            List of strategies
        """
        if search_query:
            return self.repository.search(search_query, limit, offset)
        else:
            return self.repository.get_all(limit, offset)
    
    def get_all_strategies(self, include_deleted: bool = False) -> List[Strategy]:
        """Get all strategies without pagination."""
        if include_deleted:
            return self.repository.get_all(limit=1000)
        else:
            # Only get active strategies
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM strategies WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1000"
                )
                rows = cursor.fetchall()
                return [self.repository._row_to_model(row) for row in rows]
    
    def update_strategy(self, 
                       strategy_id: int, 
                       name: Optional[str] = None,
                       description: Optional[str] = None) -> Tuple[Optional[Strategy], Optional[str]]:
        """
        Update strategy details.
        
        Args:
            strategy_id: Strategy ID to update
            name: New name (optional)
            description: New description (optional)
            
        Returns:
            Tuple of (updated strategy, error message if any)
        """
        try:
            # Get existing strategy
            strategy = self.repository.get_by_id(strategy_id)
            if not strategy:
                return None, f"Strategy with ID {strategy_id} not found"
            
            # Update fields
            if name is not None:
                name = name.strip()
                if not name:
                    return None, "Strategy name cannot be empty"
                
                # Check for duplicate name
                existing = self.repository.get_by_name(name)
                if existing and existing.id != strategy_id:
                    return None, f"Strategy with name '{name}' already exists"
                
                strategy.name = name
            
            if description is not None:
                strategy.description = description
            
            strategy.updated_at = datetime.utcnow()
            
            # Update in database
            updated = self.repository.update(strategy)
            logger.info(f"Updated strategy: {updated.name} (ID: {updated.id})")
            
            return updated, None
            
        except Exception as e:
            logger.error(f"Error updating strategy: {str(e)}")
            return None, f"Failed to update strategy: {str(e)}"
    
    def delete_strategy(self, strategy_id: int, cascade: bool = True, soft_delete: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Delete a strategy and optionally its related data.
        
        Args:
            strategy_id: Strategy ID to delete
            cascade: If True, delete all related trades
            soft_delete: If True, mark as inactive instead of deleting
            
        Returns:
            Tuple of (success, error message if any)
        """
        try:
            # Check if strategy exists
            strategy = self.repository.get_by_id(strategy_id)
            if not strategy:
                return False, f"Strategy with ID {strategy_id} not found"
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get dependent data counts for warning
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT t.id) as trade_count,
                        COUNT(DISTINCT pm.id) as metric_count,
                        COUNT(DISTINCT ca.id) as confluence_count,
                        COUNT(DISTINCT uh.id) as upload_count
                    FROM strategies s
                    LEFT JOIN trades t ON s.id = t.strategy_id
                    LEFT JOIN performance_metrics pm ON s.id = pm.strategy_id
                    LEFT JOIN confluence_analysis ca ON (s.id = ca.strategy1_id OR s.id = ca.strategy2_id)
                    LEFT JOIN upload_history uh ON s.id = uh.strategy_id
                    WHERE s.id = ?
                """, (strategy_id,))
                
                counts = cursor.fetchone()
                trade_count = counts[0] or 0
                metric_count = counts[1] or 0
                confluence_count = counts[2] or 0
                upload_count = counts[3] or 0
                
                if soft_delete:
                    # Soft delete - mark as inactive
                    cursor.execute(
                        "UPDATE strategies SET is_active = 0, updated_at = ? WHERE id = ?",
                        (datetime.utcnow(), strategy_id)
                    )
                    conn.commit()
                    logger.info(f"Soft deleted strategy: {strategy.name} (ID: {strategy_id})")
                    return True, None
                
                if cascade:
                    # Log what will be deleted
                    logger.info(f"Cascade deleting strategy {strategy_id}: "
                              f"{trade_count} trades, {metric_count} metrics, "
                              f"{confluence_count} confluence records, {upload_count} uploads")
                    
                    # Delete related data (cascade is handled by foreign keys)
                    cursor.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
                    conn.commit()
                    
                    logger.info(f"Deleted strategy: {strategy.name} (ID: {strategy_id}) and all related data")
                    return True, None
                else:
                    # Check if there are related records
                    if trade_count > 0 or metric_count > 0 or confluence_count > 0:
                        return False, (f"Cannot delete strategy with dependent data: "
                                     f"{trade_count} trades, {metric_count} metrics, "
                                     f"{confluence_count} confluence records. "
                                     f"Use cascade=True to delete all related data.")
                    
                    # Delete strategy (no dependent data)
                    cursor.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
                    conn.commit()
                    
                    logger.info(f"Deleted strategy: {strategy.name} (ID: {strategy_id})")
                    return True, None
                
        except Exception as e:
            logger.error(f"Error deleting strategy: {str(e)}")
            return False, f"Failed to delete strategy: {str(e)}"
    
    def restore_strategy(self, strategy_id: int) -> Tuple[Optional[Strategy], Optional[str]]:
        """
        Restore a soft-deleted strategy.
        
        Args:
            strategy_id: Strategy ID to restore
            
        Returns:
            Tuple of (restored strategy, error message if any)
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if strategy exists and is soft-deleted
                cursor.execute(
                    "SELECT * FROM strategies WHERE id = ? AND is_active = 0",
                    (strategy_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None, f"Strategy with ID {strategy_id} not found or not deleted"
                
                # Restore strategy
                cursor.execute(
                    "UPDATE strategies SET is_active = 1, updated_at = ? WHERE id = ?",
                    (datetime.utcnow(), strategy_id)
                )
                conn.commit()
                
                # Get restored strategy
                strategy = self.repository.get_by_id(strategy_id)
                logger.info(f"Restored strategy: {strategy.name} (ID: {strategy_id})")
                
                return strategy, None
                
        except Exception as e:
            logger.error(f"Error restoring strategy: {str(e)}")
            return None, f"Failed to restore strategy: {str(e)}"
    
    def get_deleted_strategies(self) -> List[Strategy]:
        """Get all soft-deleted strategies."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM strategies WHERE is_active = 0 ORDER BY updated_at DESC"
            )
            rows = cursor.fetchall()
            return [self.repository._row_to_model(row) for row in rows]
    
    def get_strategy_statistics(self, strategy_id: int) -> Dict[str, Any]:
        """
        Get detailed statistics for a strategy.
        
        Returns:
            Dictionary with statistics
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get basic statistics
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as trade_count,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(MAX(pnl), 0) as max_pnl,
                    COALESCE(MIN(pnl), 0) as min_pnl,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                    COUNT(CASE WHEN pnl = 0 THEN 1 END) as breakeven_trades
                FROM trades
                WHERE strategy_id = ?
                """,
                (strategy_id,)
            )
            
            row = cursor.fetchone()
            
            trade_count = row[0]
            win_rate = (row[5] / trade_count * 100) if trade_count > 0 else 0
            
            return {
                'trade_count': trade_count,
                'total_pnl': float(row[1]),
                'avg_pnl': float(row[2]),
                'max_pnl': float(row[3]),
                'min_pnl': float(row[4]),
                'winning_trades': row[5],
                'losing_trades': row[6],
                'breakeven_trades': row[7],
                'win_rate': win_rate
            }
    
    def append_trades(self, strategy_id: int, trades: List[Trade]) -> Tuple[int, Optional[str]]:
        """
        Append new trades to a strategy.
        
        Args:
            strategy_id: Strategy ID
            trades: List of trades to append
            
        Returns:
            Tuple of (number of trades added, error message if any)
        """
        conn = None
        try:
            # Verify strategy exists
            strategy = self.repository.get_by_id(strategy_id)
            if not strategy:
                return 0, f"Strategy with ID {strategy_id} not found"
            
            # Set strategy_id for all trades
            for trade in trades:
                trade.strategy_id = strategy_id
            
            # Insert trades in a transaction
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                for trade in trades:
                    cursor.execute(
                        """
                        INSERT INTO trades (
                            strategy_id, trade_date, symbol, side, 
                            entry_price, exit_price, quantity, pnl, commission
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trade.strategy_id,
                            trade.trade_date,
                            trade.symbol,
                            trade.side,
                            str(trade.entry_price),
                            str(trade.exit_price) if trade.exit_price else None,
                            str(trade.quantity),
                            str(trade.pnl) if trade.pnl else None,
                            str(trade.commission) if trade.commission else "0"
                        )
                    )
                
                # Commit transaction
                cursor.execute("COMMIT")
                
                # Update strategy statistics
                self.repository.update_statistics(strategy_id)
                
                logger.info(f"Appended {len(trades)} trades to strategy ID: {strategy_id}")
                return len(trades), None
                
            except Exception as e:
                # Rollback on error
                cursor.execute("ROLLBACK")
                raise e
            
        except Exception as e:
            logger.error(f"Error appending trades: {str(e)}")
            return 0, f"Failed to append trades: {str(e)}"
        finally:
            if conn:
                conn.close()
    
    def replace_trades(self, strategy_id: int, trades: List[Trade]) -> Tuple[int, Optional[str]]:
        """
        Replace all trades for a strategy.
        
        Args:
            strategy_id: Strategy ID
            trades: List of trades to replace with
            
        Returns:
            Tuple of (number of trades added, error message if any)
        """
        conn = None
        try:
            # Verify strategy exists
            strategy = self.repository.get_by_id(strategy_id)
            if not strategy:
                return 0, f"Strategy with ID {strategy_id} not found"
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                # Delete existing trades
                cursor.execute("DELETE FROM trades WHERE strategy_id = ?", (strategy_id,))
                deleted_count = cursor.rowcount
                
                # Insert new trades
                for trade in trades:
                    trade.strategy_id = strategy_id
                    cursor.execute(
                        """
                        INSERT INTO trades (
                            strategy_id, trade_date, symbol, side, 
                            entry_price, exit_price, quantity, pnl, commission
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trade.strategy_id,
                            trade.trade_date,
                            trade.symbol,
                            trade.side,
                            str(trade.entry_price),
                            str(trade.exit_price) if trade.exit_price else None,
                            str(trade.quantity),
                            str(trade.pnl) if trade.pnl else None,
                            str(trade.commission) if trade.commission else "0"
                        )
                    )
                
                # Commit transaction
                cursor.execute("COMMIT")
                
                # Update strategy statistics
                self.repository.update_statistics(strategy_id)
                
                logger.info(f"Replaced {deleted_count} trades for strategy ID: {strategy_id} with {len(trades)} new trades")
                return len(trades), None
                
            except Exception as e:
                # Rollback on error
                cursor.execute("ROLLBACK")
                raise e
            
        except Exception as e:
            logger.error(f"Error replacing trades: {str(e)}")
            return 0, f"Failed to replace trades: {str(e)}"
        finally:
            if conn:
                conn.close()
    
    def count_strategies(self, search_query: Optional[str] = None) -> int:
        """Count total number of strategies."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            if search_query:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM strategies
                    WHERE name LIKE ? OR description LIKE ?
                    """,
                    (f'%{search_query}%', f'%{search_query}%')
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM strategies")
            
            return cursor.fetchone()[0]