"""Repository for Trade CRUD operations."""

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from decimal import Decimal

from .base_repository import BaseRepository
from ..models.trade import Trade


class TradeRepository(BaseRepository[Trade]):
    """Repository for managing trades."""
    
    def __init__(self):
        super().__init__('trades', Trade)
    
    def _row_to_model(self, row: tuple, columns: List[str]) -> Trade:
        """Convert database row to Trade model."""
        data = dict(zip(columns, row))
        return Trade.from_dict(data)
    
    def _model_to_params(self, model: Trade) -> Dict[str, Any]:
        """Convert Trade model to database parameters."""
        return {
            'id': model.id,
            'strategy_id': model.strategy_id,
            'trade_date': model.trade_date.isoformat() if isinstance(model.trade_date, date) else model.trade_date,
            'symbol': model.symbol,
            'side': model.side,
            'entry_price': float(model.entry_price) if model.entry_price else None,
            'exit_price': float(model.exit_price) if model.exit_price else None,
            'quantity': float(model.quantity) if model.quantity else None,
            'pnl': float(model.pnl) if model.pnl else None,
            'commission': float(model.commission),
            'entry_time': model.entry_time.isoformat() if model.entry_time else None,
            'exit_time': model.exit_time.isoformat() if model.exit_time else None,
            'duration_hours': float(model.duration_hours) if model.duration_hours else None,
            'created_at': model.created_at.isoformat() if model.created_at else None
        }
    
    def get_by_strategy(self, strategy_id: int, 
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> List[Trade]:
        """
        Get trades for a specific strategy with optional date range.
        
        Args:
            strategy_id: Strategy ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List[Trade]: List of trades
        """
        query = f"SELECT * FROM {self.table_name} WHERE strategy_id = ?"
        params = [strategy_id]
        
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY trade_date, entry_time"
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [self._row_to_model(row, columns) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get trades for strategy {strategy_id}: {e}")
            raise
    
    def bulk_create(self, trades: List[Trade]) -> int:
        """
        Create multiple trades in a single transaction.
        
        Args:
            trades: List of trades to create
            
        Returns:
            int: Number of trades created
        """
        if not trades:
            return 0
        
        # Get column names from first trade
        first_params = self._model_to_params(trades[0])
        first_params.pop('id', None)  # Remove ID for insertion
        columns = list(first_params.keys())
        
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)
        
        query = f"""
            INSERT INTO {self.table_name} ({column_names})
            VALUES ({placeholders})
        """
        
        # Prepare parameter list
        params_list = []
        for trade in trades:
            params = self._model_to_params(trade)
            params.pop('id', None)
            params_list.append(tuple(params[col] for col in columns))
        
        try:
            cursor = self.db.executemany(query, params_list)
            count = cursor.rowcount
            logger.info(f"Created {count} trades in bulk")
            return count
        except Exception as e:
            logger.error(f"Failed to bulk create trades: {e}")
            raise
    
    def delete_by_strategy(self, strategy_id: int) -> int:
        """
        Delete all trades for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            int: Number of trades deleted
        """
        query = f"DELETE FROM {self.table_name} WHERE strategy_id = ?"
        
        try:
            cursor = self.db.execute(query, (strategy_id,))
            count = cursor.rowcount
            logger.info(f"Deleted {count} trades for strategy {strategy_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to delete trades for strategy {strategy_id}: {e}")
            raise
    
    def get_symbols_by_strategy(self, strategy_id: int) -> List[str]:
        """
        Get unique symbols traded by a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            List[str]: List of unique symbols
        """
        query = """
            SELECT DISTINCT symbol 
            FROM trades 
            WHERE strategy_id = ?
            ORDER BY symbol
        """
        
        try:
            rows = self.db.fetchall(query, (strategy_id,))
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get symbols for strategy {strategy_id}: {e}")
            raise
    
    def get_trade_summary(self, strategy_id: int) -> Dict[str, Any]:
        """
        Get trade summary statistics for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dict: Summary statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                COUNT(CASE WHEN pnl = 0 THEN 1 END) as breakeven_trades,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(AVG(pnl), 0) as avg_pnl,
                COALESCE(MAX(pnl), 0) as max_win,
                COALESCE(MIN(pnl), 0) as max_loss,
                COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) as gross_profit,
                COALESCE(SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), 0) as gross_loss
            FROM trades
            WHERE strategy_id = ?
        """
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (strategy_id,))
                
                columns = [desc[0] for desc in cursor.description]
                row = cursor.fetchone()
                
                if row:
                    return dict(zip(columns, row))
                return {}
        except Exception as e:
            logger.error(f"Failed to get trade summary for strategy {strategy_id}: {e}")
            raise


# Import logger after class definition to avoid circular import
import logging
logger = logging.getLogger(__name__)