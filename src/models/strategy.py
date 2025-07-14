"""Strategy model for the Trading Strategy Analyzer."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from .base import TimestampedModel


@dataclass
class Strategy(TimestampedModel):
    """Represents a trading strategy."""
    
    name: str = ""
    description: Optional[str] = None
    total_trades: int = 0
    total_pnl: Decimal = Decimal('0.00')
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'total_trades': self.total_trades,
            'total_pnl': float(self.total_pnl),
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """Create strategy from dictionary."""
        # Handle datetime conversion
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=data.get('id'),
            name=data['name'],
            description=data.get('description'),
            total_trades=data.get('total_trades', 0),
            total_pnl=Decimal(str(data.get('total_pnl', 0))),
            is_active=data.get('is_active', True),
            created_at=created_at,
            updated_at=updated_at
        )
    
    def update_stats(self, trades: List['Trade']) -> None:
        """Update strategy statistics based on trades."""
        self.total_trades = len(trades)
        self.total_pnl = sum(Decimal(str(trade.pnl or 0)) for trade in trades)
        self.updated_at = datetime.now()