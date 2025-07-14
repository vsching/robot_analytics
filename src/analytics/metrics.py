"""Data classes for performance metrics."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime, date
from decimal import Decimal


@dataclass
class PnLSummary:
    """Summary of P&L metrics."""
    total_pnl: Decimal
    average_pnl: Decimal
    max_pnl: Decimal
    min_pnl: Decimal
    median_pnl: Decimal
    std_dev: Decimal
    total_commission: Decimal
    net_pnl: Decimal
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with float values."""
        return {
            'total_pnl': float(self.total_pnl),
            'average_pnl': float(self.average_pnl),
            'max_pnl': float(self.max_pnl),
            'min_pnl': float(self.min_pnl),
            'median_pnl': float(self.median_pnl),
            'std_dev': float(self.std_dev),
            'total_commission': float(self.total_commission),
            'net_pnl': float(self.net_pnl)
        }


@dataclass
class TradeStatistics:
    """Comprehensive trade statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    loss_rate: float
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    win_loss_ratio: float
    profit_factor: float
    expectancy: Decimal
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'breakeven_trades': self.breakeven_trades,
            'win_rate': self.win_rate,
            'loss_rate': self.loss_rate,
            'average_win': float(self.average_win),
            'average_loss': float(self.average_loss),
            'largest_win': float(self.largest_win),
            'largest_loss': float(self.largest_loss),
            'win_loss_ratio': self.win_loss_ratio,
            'profit_factor': self.profit_factor,
            'expectancy': float(self.expectancy),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses
        }


@dataclass
class TimeBasedMetrics:
    """Metrics for a specific time period."""
    period_start: date
    period_end: date
    period_type: str  # 'daily', 'weekly', 'monthly', 'yearly'
    total_trades: int
    total_pnl: Decimal
    win_rate: float
    average_pnl: Decimal
    best_day: Optional[date] = None
    worst_day: Optional[date] = None
    trading_days: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'period_type': self.period_type,
            'total_trades': self.total_trades,
            'total_pnl': float(self.total_pnl),
            'win_rate': self.win_rate,
            'average_pnl': float(self.average_pnl),
            'best_day': self.best_day.isoformat() if self.best_day else None,
            'worst_day': self.worst_day.isoformat() if self.worst_day else None,
            'trading_days': self.trading_days
        }


@dataclass
class AdvancedStatistics:
    """Advanced performance statistics."""
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    max_drawdown_duration: Optional[int] = None
    recovery_duration: Optional[int] = None
    value_at_risk_95: Optional[float] = None
    conditional_var_95: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'recovery_duration': self.recovery_duration,
            'value_at_risk_95': self.value_at_risk_95,
            'conditional_var_95': self.conditional_var_95
        }


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a strategy."""
    strategy_id: int
    calculated_at: datetime
    pnl_summary: PnLSummary
    trade_statistics: TradeStatistics
    monthly_metrics: List[TimeBasedMetrics]
    weekly_metrics: List[TimeBasedMetrics]
    advanced_statistics: Optional[AdvancedStatistics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'strategy_id': self.strategy_id,
            'calculated_at': self.calculated_at.isoformat(),
            'pnl_summary': self.pnl_summary.to_dict(),
            'trade_statistics': self.trade_statistics.to_dict(),
            'monthly_metrics': [m.to_dict() for m in self.monthly_metrics],
            'weekly_metrics': [m.to_dict() for m in self.weekly_metrics]
        }
        
        if self.advanced_statistics:
            result['advanced_statistics'] = self.advanced_statistics.to_dict()
        
        return result