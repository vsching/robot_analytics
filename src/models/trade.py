"""Trade model for the Trading Strategy Analyzer."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, date, time
from decimal import Decimal

from .base import BaseModel


@dataclass
class Trade(BaseModel):
    """Represents a single trade."""
    
    strategy_id: int = 0
    trade_date: date = date.today()
    symbol: str = ""
    side: str = ""  # 'long', 'short', 'buy', 'sell'
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    pnl: Optional[Decimal] = None
    commission: Decimal = Decimal('0.00')
    entry_time: Optional[time] = None
    exit_time: Optional[time] = None
    duration_hours: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate trade data."""
        super().__post_init__()
        
        # Validate side
        valid_sides = ['long', 'short', 'buy', 'sell']
        if self.side.lower() not in valid_sides:
            raise ValueError(f"Invalid side: {self.side}. Must be one of {valid_sides}")
        
        # Normalize side
        self.side = self.side.lower()
        
        # Calculate duration if entry and exit times are provided
        if self.entry_time and self.exit_time and not self.duration_hours:
            # Simple calculation assuming same day
            entry_dt = datetime.combine(self.trade_date, self.entry_time)
            exit_dt = datetime.combine(self.trade_date, self.exit_time)
            duration = (exit_dt - entry_dt).total_seconds() / 3600
            self.duration_hours = Decimal(str(duration))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'id': self.id,
            'strategy_id': self.strategy_id,
            'trade_date': self.trade_date.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'quantity': float(self.quantity) if self.quantity else None,
            'pnl': float(self.pnl) if self.pnl else None,
            'commission': float(self.commission),
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'duration_hours': float(self.duration_hours) if self.duration_hours else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create trade from dictionary."""
        # Handle date/time conversions
        trade_date = data.get('trade_date')
        if trade_date and isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)
        
        entry_time = data.get('entry_time')
        if entry_time and isinstance(entry_time, str):
            entry_time = time.fromisoformat(entry_time)
        
        exit_time = data.get('exit_time')
        if exit_time and isinstance(exit_time, str):
            exit_time = time.fromisoformat(exit_time)
        
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return cls(
            id=data.get('id'),
            strategy_id=data['strategy_id'],
            trade_date=trade_date,
            symbol=data['symbol'],
            side=data['side'],
            entry_price=Decimal(str(data['entry_price'])) if data.get('entry_price') else None,
            exit_price=Decimal(str(data['exit_price'])) if data.get('exit_price') else None,
            quantity=Decimal(str(data['quantity'])) if data.get('quantity') else None,
            pnl=Decimal(str(data['pnl'])) if data.get('pnl') else None,
            commission=Decimal(str(data.get('commission', 0))),
            entry_time=entry_time,
            exit_time=exit_time,
            duration_hours=Decimal(str(data['duration_hours'])) if data.get('duration_hours') else None,
            created_at=created_at
        )