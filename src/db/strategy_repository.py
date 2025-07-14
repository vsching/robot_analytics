"""Repository for Strategy CRUD operations."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from .base_repository import BaseRepository
from ..models.strategy import Strategy


class StrategyRepository(BaseRepository[Strategy]):
    """Repository for managing trading strategies."""
    
    def __init__(self):
        super().__init__('strategies', Strategy)
    
    def _row_to_model(self, row: tuple, columns: List[str]) -> Strategy:
        """Convert database row to Strategy model."""
        data = dict(zip(columns, row))
        return Strategy.from_dict(data)
    
    def _model_to_params(self, model: Strategy) -> Dict[str, Any]:
        """Convert Strategy model to database parameters."""
        return {
            'id': model.id,
            'name': model.name,
            'description': model.description,
            'total_trades': model.total_trades,
            'total_pnl': float(model.total_pnl),
            'is_active': 1 if model.is_active else 0,
            'created_at': model.created_at,
            'updated_at': model.updated_at
        }
    
    def get_by_name(self, name: str) -> Optional[Strategy]:
        """
        Get strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Optional[Strategy]: Strategy if found
        """
        strategies = self.find_by(name=name)
        return strategies[0] if strategies else None
    
    def get_active_strategies(self) -> List[Strategy]:
        """
        Get all active strategies.
        
        Returns:
            List[Strategy]: List of active strategies
        """
        return self.find_by(is_active=1)
    
    def update_stats(self, strategy_id: int) -> bool:
        """
        Update strategy statistics based on trades.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            bool: True if updated successfully
        """
        query = """
            UPDATE strategies
            SET total_trades = (
                    SELECT COUNT(*) FROM trades WHERE strategy_id = ?
                ),
                total_pnl = (
                    SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE strategy_id = ?
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        
        try:
            cursor = self.db.execute(query, (strategy_id, strategy_id, strategy_id))
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update stats for strategy {strategy_id}: {e}")
            raise
    
    def get_summary_stats(self) -> List[Dict[str, Any]]:
        """
        Get summary statistics for all strategies.
        
        Returns:
            List[Dict]: List of strategy summaries
        """
        query = """
            SELECT 
                s.id,
                s.name,
                s.total_trades,
                s.total_pnl,
                COUNT(DISTINCT t.symbol) as symbols_traded,
                MIN(t.trade_date) as first_trade_date,
                MAX(t.trade_date) as last_trade_date,
                CASE 
                    WHEN s.total_trades > 0 THEN ROUND(s.total_pnl / s.total_trades, 2)
                    ELSE 0
                END as avg_trade_pnl
            FROM strategies s
            LEFT JOIN trades t ON s.id = t.strategy_id
            WHERE s.is_active = 1
            GROUP BY s.id
            ORDER BY s.total_pnl DESC
        """
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get strategy summary stats: {e}")
            raise
    
    def create_or_update(self, strategy: Strategy) -> Strategy:
        """
        Create strategy if it doesn't exist, otherwise update it.
        
        Args:
            strategy: Strategy to create or update
            
        Returns:
            Strategy: Created or updated strategy
        """
        existing = self.get_by_name(strategy.name)
        
        if existing:
            strategy.id = existing.id
            strategy.created_at = existing.created_at
            self.update(strategy)
            return strategy
        else:
            return self.create(strategy)


# Import logger after class definition to avoid circular import
import logging
logger = logging.getLogger(__name__)