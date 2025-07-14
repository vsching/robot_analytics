"""Advanced performance metrics calculations for trading analysis."""

from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, date, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy import stats
import logging
import warnings


logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Calculate advanced performance metrics for trading strategies."""
    
    def __init__(self, annual_trading_days: int = 252):
        """
        Initialize advanced metrics calculator.
        
        Args:
            annual_trading_days: Number of trading days per year (252 for stocks, 365 for crypto)
        """
        self.annual_trading_days = annual_trading_days
    
    def calculate_sharpe_ratio(self, 
                              returns: np.ndarray, 
                              risk_free_rate: float = 0.02,
                              frequency: str = 'daily') -> Optional[float]:
        """
        Calculate the Sharpe ratio for a series of returns.
        
        Args:
            returns: Array of returns (as decimals, not percentages)
            risk_free_rate: Annual risk-free rate (default 2%)
            frequency: Return frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            Annualized Sharpe ratio or None if insufficient data
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for Sharpe ratio calculation")
            return None
        
        # Convert to numpy array and remove NaN values
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return None
        
        # Calculate excess returns
        if frequency == 'daily':
            periods_per_year = self.annual_trading_days
            daily_rf_rate = risk_free_rate / periods_per_year
        elif frequency == 'weekly':
            periods_per_year = 52
            daily_rf_rate = risk_free_rate / periods_per_year
        elif frequency == 'monthly':
            periods_per_year = 12
            daily_rf_rate = risk_free_rate / periods_per_year
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        excess_returns = returns - daily_rf_rate
        
        # Calculate mean and standard deviation
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)  # Sample standard deviation
        
        # Handle zero volatility
        if std_excess == 0:
            logger.warning("Zero volatility detected in returns")
            return np.inf if mean_excess > 0 else -np.inf if mean_excess < 0 else 0
        
        # Calculate and annualize Sharpe ratio
        sharpe_ratio = mean_excess / std_excess * np.sqrt(periods_per_year)
        
        return float(sharpe_ratio)
    
    def calculate_sortino_ratio(self,
                               returns: np.ndarray,
                               target_return: float = 0.0,
                               risk_free_rate: float = 0.02,
                               frequency: str = 'daily') -> Optional[float]:
        """
        Calculate the Sortino ratio focusing on downside deviation.
        
        Args:
            returns: Array of returns
            target_return: Minimum acceptable return (MAR)
            risk_free_rate: Annual risk-free rate
            frequency: Return frequency
            
        Returns:
            Annualized Sortino ratio or None if insufficient data
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for Sortino ratio calculation")
            return None
        
        # Convert to numpy array and remove NaN values
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return None
        
        # Determine annualization factor
        if frequency == 'daily':
            periods_per_year = self.annual_trading_days
            period_rf_rate = risk_free_rate / periods_per_year
            period_target = target_return / periods_per_year
        elif frequency == 'weekly':
            periods_per_year = 52
            period_rf_rate = risk_free_rate / periods_per_year
            period_target = target_return / periods_per_year
        elif frequency == 'monthly':
            periods_per_year = 12
            period_rf_rate = risk_free_rate / periods_per_year
            period_target = target_return / periods_per_year
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        # Calculate excess returns over risk-free rate
        excess_returns = returns - period_rf_rate
        mean_excess = np.mean(excess_returns)
        
        # Calculate downside deviation
        downside_returns = returns[returns < period_target] - period_target
        
        if len(downside_returns) == 0:
            # No downside returns
            return np.inf if mean_excess > 0 else 0
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return np.inf if mean_excess > 0 else 0
        
        # Calculate and annualize Sortino ratio
        sortino_ratio = mean_excess / downside_deviation * np.sqrt(periods_per_year)
        
        return float(sortino_ratio)
    
    def calculate_profit_factor(self, pnl_values: List[Decimal]) -> Optional[float]:
        """
        Calculate profit factor as ratio of gross profits to gross losses.
        
        Args:
            pnl_values: List of P&L values
            
        Returns:
            Profit factor or None if no data
        """
        if not pnl_values:
            return None
        
        # Separate profits and losses
        gross_profits = sum(pnl for pnl in pnl_values if pnl > 0)
        gross_losses = abs(sum(pnl for pnl in pnl_values if pnl < 0))
        
        # Handle edge cases
        if gross_losses == 0:
            if gross_profits > 0:
                return float('inf')  # All profitable trades
            else:
                return 0.0  # No trades or all breakeven
        
        if gross_profits == 0:
            return 0.0  # All losing trades
        
        profit_factor = float(gross_profits / gross_losses)
        return profit_factor
    
    def calculate_max_drawdown(self, 
                              equity_curve: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate maximum drawdown and related statistics.
        
        Args:
            equity_curve: Array of cumulative equity values
            
        Returns:
            Tuple of (max_drawdown_percentage, drawdown_details)
        """
        if len(equity_curve) < 2:
            return 0.0, {
                'max_drawdown': 0.0,
                'peak_index': 0,
                'trough_index': 0,
                'recovery_index': None,
                'drawdown_duration': 0,
                'recovery_duration': None
            }
        
        # Calculate running maximum
        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_index = np.argmin(drawdown)
        max_drawdown = abs(drawdown[max_dd_index])
        
        # Find the peak before the maximum drawdown
        peak_index = np.where(running_max[:max_dd_index + 1] == running_max[max_dd_index])[0][0]
        
        # Find recovery point (if any)
        recovery_index = None
        recovery_duration = None
        
        if max_dd_index < len(equity_curve) - 1:
            # Look for recovery after the trough
            recovery_points = np.where(equity_curve[max_dd_index + 1:] >= running_max[peak_index])[0]
            if len(recovery_points) > 0:
                recovery_index = max_dd_index + 1 + recovery_points[0]
                recovery_duration = recovery_index - max_dd_index
        
        drawdown_duration = max_dd_index - peak_index
        
        details = {
            'max_drawdown': float(max_drawdown),
            'peak_index': int(peak_index),
            'trough_index': int(max_dd_index),
            'recovery_index': int(recovery_index) if recovery_index is not None else None,
            'drawdown_duration': int(drawdown_duration),
            'recovery_duration': int(recovery_duration) if recovery_duration is not None else None,
            'peak_value': float(equity_curve[peak_index]),
            'trough_value': float(equity_curve[max_dd_index])
        }
        
        return float(max_drawdown), details
    
    def calculate_calmar_ratio(self,
                              returns: np.ndarray,
                              max_drawdown: float,
                              frequency: str = 'daily') -> Optional[float]:
        """
        Calculate Calmar ratio as annualized return divided by maximum drawdown.
        
        Args:
            returns: Array of returns
            max_drawdown: Maximum drawdown (as positive decimal)
            frequency: Return frequency
            
        Returns:
            Calmar ratio or None if invalid data
        """
        if len(returns) == 0 or max_drawdown == 0:
            return None
        
        # Calculate annualized return
        if frequency == 'daily':
            periods_per_year = self.annual_trading_days
        elif frequency == 'weekly':
            periods_per_year = 52
        elif frequency == 'monthly':
            periods_per_year = 12
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        # Calculate compound annual return
        total_return = np.prod(1 + returns) - 1
        years = len(returns) / periods_per_year
        
        if years == 0:
            return None
        
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Calmar ratio
        calmar = annual_return / max_drawdown
        
        return float(calmar)
    
    def calculate_information_ratio(self,
                                   returns: np.ndarray,
                                   benchmark_returns: np.ndarray,
                                   frequency: str = 'daily') -> Optional[float]:
        """
        Calculate information ratio relative to a benchmark.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            frequency: Return frequency
            
        Returns:
            Information ratio or None if insufficient data
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return None
        
        # Calculate active returns
        active_returns = returns - benchmark_returns
        
        # Remove NaN values
        mask = ~(np.isnan(active_returns))
        active_returns = active_returns[mask]
        
        if len(active_returns) < 2:
            return None
        
        # Calculate tracking error (standard deviation of active returns)
        tracking_error = np.std(active_returns, ddof=1)
        
        if tracking_error == 0:
            return np.inf if np.mean(active_returns) > 0 else -np.inf
        
        # Annualize
        if frequency == 'daily':
            periods_per_year = self.annual_trading_days
        elif frequency == 'weekly':
            periods_per_year = 52
        elif frequency == 'monthly':
            periods_per_year = 12
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        # Information ratio
        ir = np.mean(active_returns) / tracking_error * np.sqrt(periods_per_year)
        
        return float(ir)
    
    def calculate_value_at_risk(self,
                               returns: np.ndarray,
                               confidence_level: float = 0.95,
                               method: str = 'historical') -> Optional[float]:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR (as positive number representing potential loss)
        """
        if len(returns) < 20:  # Need sufficient data for VaR
            return None
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        if method == 'historical':
            # Historical VaR
            var_percentile = (1 - confidence_level) * 100
            var = -np.percentile(returns, var_percentile)
        
        elif method == 'parametric':
            # Parametric VaR (assumes normal distribution)
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return float(var)
    
    def calculate_conditional_var(self,
                                 returns: np.ndarray,
                                 confidence_level: float = 0.95) -> Optional[float]:
        """
        Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR (as positive number)
        """
        if len(returns) < 20:
            return None
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        # Calculate VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # Calculate average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return None
        
        cvar = -np.mean(tail_returns)
        
        return float(cvar)
    
    def calculate_omega_ratio(self,
                             returns: np.ndarray,
                             threshold: float = 0.0) -> Optional[float]:
        """
        Calculate Omega ratio.
        
        Args:
            returns: Array of returns
            threshold: Threshold return (default 0)
            
        Returns:
            Omega ratio
        """
        if len(returns) < 2:
            return None
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        # Calculate probability-weighted gains and losses
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) == 0 or np.sum(losses) == 0:
            return float('inf') if len(gains) > 0 else 1.0
        
        omega = np.sum(gains) / np.sum(losses)
        
        return float(omega)
    
    def validate_data(self, 
                     data: Any, 
                     min_points: int = 2,
                     data_type: str = 'returns') -> Tuple[bool, Optional[str]]:
        """
        Validate input data for calculations.
        
        Args:
            data: Input data to validate
            min_points: Minimum required data points
            data_type: Type of data ('returns', 'prices', 'pnl')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if data exists
        if data is None:
            return False, f"No {data_type} data provided"
        
        # Convert to numpy array
        try:
            if isinstance(data, (list, pd.Series)):
                data_array = np.array(data, dtype=float)
            elif isinstance(data, np.ndarray):
                data_array = data.astype(float)
            else:
                return False, f"Invalid data type: {type(data)}"
        except (ValueError, TypeError) as e:
            return False, f"Cannot convert {data_type} to numeric array: {str(e)}"
        
        # Remove NaN values
        valid_data = data_array[~np.isnan(data_array)]
        
        # Check minimum data points
        if len(valid_data) < min_points:
            return False, f"Insufficient {data_type} data: {len(valid_data)} points (minimum {min_points} required)"
        
        # Check for infinite values
        if np.any(np.isinf(valid_data)):
            return False, f"{data_type} contains infinite values"
        
        # Data type specific validation
        if data_type == 'prices':
            if np.any(valid_data <= 0):
                return False, "Prices must be positive"
        
        return True, None