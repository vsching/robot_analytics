"""Multi-strategy comparison dashboard component."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple, Optional
from decimal import Decimal
import logging
from scipy import stats
from scipy.optimize import minimize

from ..models import Strategy, Trade
from ..analytics import AnalyticsEngine, PerformanceMetrics
from ..services.strategy_manager import StrategyManager
from .visualization import VisualizationComponents


logger = logging.getLogger(__name__)


class ComparisonDashboard:
    """Multi-strategy comparison and analysis dashboard."""
    
    def __init__(self, 
                 strategy_manager: StrategyManager,
                 analytics_engine: AnalyticsEngine,
                 viz_components: Optional[VisualizationComponents] = None):
        """
        Initialize comparison dashboard.
        
        Args:
            strategy_manager: Strategy manager instance
            analytics_engine: Analytics engine instance
            viz_components: Visualization components instance
        """
        self.strategy_manager = strategy_manager
        self.analytics_engine = analytics_engine
        self.viz = viz_components or VisualizationComponents(theme="dark")
        
    def render_strategy_selector(self, min_strategies: int = 2, max_strategies: int = 10) -> List[int]:
        """
        Render multi-strategy selection interface.
        
        Args:
            min_strategies: Minimum number of strategies to select
            max_strategies: Maximum number of strategies to select
            
        Returns:
            List of selected strategy IDs
        """
        st.subheader("Select Strategies for Comparison")
        
        # Get all active strategies
        strategies = self.strategy_manager.get_all_strategies()
        
        if len(strategies) < min_strategies:
            st.warning(f"You need at least {min_strategies} strategies to use the comparison dashboard. "
                      f"Currently you have {len(strategies)} strategies.")
            return []
        
        # Create strategy options
        strategy_options = {f"{s.name} (ID: {s.id})": s.id for s in strategies}
        
        # Multi-select widget
        selected_names = st.multiselect(
            "Choose strategies to compare:",
            options=list(strategy_options.keys()),
            default=list(strategy_options.keys())[:min_strategies],
            help=f"Select between {min_strategies} and {max_strategies} strategies"
        )
        
        selected_ids = [strategy_options[name] for name in selected_names]
        
        # Validation
        if len(selected_ids) < min_strategies:
            st.error(f"Please select at least {min_strategies} strategies")
            return []
        elif len(selected_ids) > max_strategies:
            st.error(f"Please select no more than {max_strategies} strategies")
            return []
        
        # Display selection summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Strategies", len(selected_ids))
        with col2:
            if st.button("Clear All"):
                st.experimental_rerun()
        with col3:
            select_all = st.button("Select All")
            if select_all and len(strategies) <= max_strategies:
                selected_ids = [s.id for s in strategies]
        
        return selected_ids
    
    def build_comparison_table(self, strategy_ids: List[int]) -> pd.DataFrame:
        """
        Build comparison table with key metrics for selected strategies.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for strategy_id in strategy_ids:
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if not strategy:
                continue
            
            # Calculate metrics
            metrics = self.analytics_engine.calculate_metrics_for_strategy(strategy_id)
            if not metrics:
                continue
            
            # Extract key metrics
            row_data = {
                'Strategy': strategy.name,
                'Total P&L': float(metrics.pnl_summary.total_pnl),
                'Win Rate': metrics.trade_statistics.win_rate,
                'Total Trades': metrics.trade_statistics.total_trades,
                'Avg Win': float(metrics.trade_statistics.average_win),
                'Avg Loss': float(metrics.trade_statistics.average_loss),
                'Profit Factor': metrics.trade_statistics.profit_factor,
                'Max Consecutive Wins': metrics.trade_statistics.max_consecutive_wins,
                'Max Consecutive Losses': metrics.trade_statistics.max_consecutive_losses
            }
            
            # Add advanced metrics if available
            if metrics.advanced_statistics:
                row_data.update({
                    'Sharpe Ratio': metrics.advanced_statistics.sharpe_ratio or 0,
                    'Sortino Ratio': metrics.advanced_statistics.sortino_ratio or 0,
                    'Max Drawdown': metrics.advanced_statistics.max_drawdown,
                    'Calmar Ratio': metrics.advanced_statistics.calmar_ratio or 0
                })
            else:
                row_data.update({
                    'Sharpe Ratio': 0,
                    'Sortino Ratio': 0,
                    'Max Drawdown': 0,
                    'Calmar Ratio': 0
                })
            
            comparison_data.append(row_data)
        
        df = pd.DataFrame(comparison_data)
        
        # Add ranking columns
        if not df.empty:
            # Higher is better
            for col in ['Total P&L', 'Win Rate', 'Profit Factor', 'Sharpe Ratio', 
                       'Sortino Ratio', 'Calmar Ratio']:
                if col in df.columns:
                    df[f'{col}_rank'] = df[col].rank(ascending=False)
            
            # Lower is better
            if 'Max Drawdown' in df.columns:
                df['Max Drawdown_rank'] = df['Max Drawdown'].rank(ascending=True)
        
        return df
    
    def calculate_correlation_matrix(self, strategy_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix between strategy returns.
        
        Args:
            strategy_ids: List of strategy IDs
            
        Returns:
            Tuple of (correlation matrix, returns DataFrame)
        """
        returns_data = {}
        
        for strategy_id in strategy_ids:
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if not strategy:
                continue
            
            # Get trades and calculate daily returns
            trades = self.analytics_engine.get_trades_for_strategy(strategy_id)
            if not trades:
                continue
            
            # Convert to daily P&L
            daily_pnl = {}
            for trade in trades:
                if trade.pnl is not None:
                    date_key = trade.trade_date
                    if date_key in daily_pnl:
                        daily_pnl[date_key] += float(trade.pnl)
                    else:
                        daily_pnl[date_key] = float(trade.pnl)
            
            # Create series
            if daily_pnl:
                returns_data[strategy.name] = pd.Series(daily_pnl)
        
        if len(returns_data) < 2:
            return pd.DataFrame(), pd.DataFrame()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.fillna(0)  # Fill missing dates with 0 P&L
        
        # Calculate correlation
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix, returns_df
    
    def render_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """
        Render correlation heatmap visualization.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            
        Returns:
            Plotly figure
        """
        if correlation_matrix.empty:
            return self.viz._create_empty_chart("Not enough data for correlation analysis")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="Strategy Correlation Matrix",
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'},
            width=800,
            height=600,
            **self.viz.layout_template
        )
        
        # Add diagonal line
        fig.update_xaxes(
            showgrid=False,
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            autorange='reversed'
        )
        
        return fig
    
    def render_relative_performance(self, strategy_ids: List[int], 
                                  initial_capital: float = 10000) -> go.Figure:
        """
        Render relative performance chart comparing multiple strategies.
        
        Args:
            strategy_ids: List of strategy IDs
            initial_capital: Initial capital for normalization
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Colors for different strategies
        colors = px.colors.qualitative.Set3[:len(strategy_ids)]
        
        for idx, strategy_id in enumerate(strategy_ids):
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if not strategy:
                continue
            
            # Get trades
            trades = self.analytics_engine.get_trades_for_strategy(strategy_id)
            if not trades:
                continue
            
            # Calculate cumulative returns
            sorted_trades = sorted(trades, key=lambda t: t.trade_date)
            
            dates = [sorted_trades[0].trade_date - timedelta(days=1)]
            cumulative_pnl = [0]
            equity = [initial_capital]
            current_pnl = 0
            
            for trade in sorted_trades:
                if trade.pnl:
                    current_pnl += float(trade.pnl)
                dates.append(trade.trade_date)
                cumulative_pnl.append(current_pnl)
                equity.append(initial_capital + current_pnl)
            
            # Calculate returns percentage
            returns = [(e / initial_capital - 1) * 100 for e in equity]
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=dates,
                y=returns,
                mode='lines',
                name=strategy.name,
                line=dict(color=colors[idx], width=2),
                hovertemplate=(
                    f"<b>{strategy.name}</b><br>" +
                    "Date: %{x|%Y-%m-%d}<br>" +
                    "Return: %{y:.2f}%<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout
        fig.update_layout(
            title="Relative Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            **self.viz.layout_template
        )
        
        # Add zero line
        fig.add_hline(y=0, line_color=self.viz.grid_color, line_dash="dash")
        
        return fig
    
    def calculate_portfolio_optimization(self, 
                                       strategy_ids: List[int],
                                       target_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal portfolio weights using modern portfolio theory.
        
        Args:
            strategy_ids: List of strategy IDs
            target_return: Target portfolio return (None for max Sharpe)
            
        Returns:
            Dictionary with optimization results
        """
        # Get returns data
        _, returns_df = self.calculate_correlation_matrix(strategy_ids)
        
        if returns_df.empty or len(returns_df.columns) < 2:
            return {
                'status': 'error',
                'message': 'Insufficient data for optimization'
            }
        
        # Calculate returns and covariance
        returns = returns_df.values
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        n_assets = len(returns_df.columns)
        
        # Define optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, mean_returns) - target_return
            })
        
        # Bounds (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Objective function (negative Sharpe ratio for maximization)
        def neg_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_std == 0:
                return 0
            return -portfolio_return / portfolio_std
        
        # Optimize
        try:
            result = minimize(
                neg_sharpe_ratio,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(optimal_weights, mean_returns)
                portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
                
                # Create weight mapping
                weight_dict = {
                    returns_df.columns[i]: float(optimal_weights[i]) 
                    for i in range(n_assets)
                }
                
                return {
                    'status': 'success',
                    'weights': weight_dict,
                    'expected_return': float(portfolio_return),
                    'expected_volatility': float(portfolio_std),
                    'sharpe_ratio': float(sharpe_ratio)
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Optimization failed: {result.message}'
                }
                
        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
            return {
                'status': 'error',
                'message': f'Optimization error: {str(e)}'
            }
    
    def render_portfolio_weights(self, optimization_results: Dict[str, Any]) -> go.Figure:
        """
        Render portfolio weight allocation chart.
        
        Args:
            optimization_results: Results from portfolio optimization
            
        Returns:
            Plotly figure
        """
        if optimization_results['status'] != 'success':
            return self.viz._create_empty_chart(
                optimization_results.get('message', 'Optimization failed')
            )
        
        weights = optimization_results['weights']
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.3,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(
                colors=px.colors.qualitative.Set3[:len(weights)],
                line=dict(color=self.viz.bg_color, width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<br><extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title="Optimal Portfolio Allocation",
            annotations=[
                dict(
                    text=f"Sharpe: {optimization_results['sharpe_ratio']:.2f}",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )
            ],
            **self.viz.layout_template
        )
        
        return fig
    
    def calculate_portfolio_metrics(self, 
                                  strategy_ids: List[int], 
                                  weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate portfolio-level metrics for given weights.
        
        Args:
            strategy_ids: List of strategy IDs
            weights: Dictionary mapping strategy names to weights
            
        Returns:
            Dictionary with portfolio metrics
        """
        portfolio_trades = []
        total_weight = sum(weights.values())
        
        # Normalize weights
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            return {'error': 'Invalid weights'}
        
        # Aggregate trades with weights
        for strategy_id in strategy_ids:
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if not strategy or strategy.name not in normalized_weights:
                continue
            
            weight = normalized_weights[strategy.name]
            if weight == 0:
                continue
            
            # Get strategy trades
            trades = self.analytics_engine.get_trades_for_strategy(strategy_id)
            
            # Apply weight to trades
            for trade in trades:
                weighted_trade = Trade(
                    strategy_id=strategy_id,
                    trade_date=trade.trade_date,
                    symbol=trade.symbol,
                    side=trade.side,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    quantity=trade.quantity * Decimal(str(weight)) if trade.quantity else None,
                    pnl=trade.pnl * Decimal(str(weight)) if trade.pnl else None,
                    commission=trade.commission * Decimal(str(weight)) if trade.commission else None
                )
                portfolio_trades.append(weighted_trade)
        
        if not portfolio_trades:
            return {'error': 'No trades in portfolio'}
        
        # Calculate portfolio metrics
        portfolio_metrics = self.analytics_engine.calculate_metrics_for_strategy(0)  # Dummy ID
        
        # Calculate basic metrics manually
        pnl_summary = self.analytics_engine.calculate_pnl_summary(portfolio_trades)
        trade_stats = self.analytics_engine.calculate_trade_statistics(portfolio_trades)
        advanced_metrics = self.analytics_engine.calculate_advanced_metrics(portfolio_trades)
        
        return {
            'total_pnl': float(pnl_summary.total_pnl) if pnl_summary else 0,
            'win_rate': trade_stats.win_rate if trade_stats else 0,
            'sharpe_ratio': advanced_metrics.get('sharpe_ratio', 0),
            'max_drawdown': advanced_metrics.get('max_drawdown', 0),
            'trade_count': len(portfolio_trades),
            'weights': normalized_weights
        }
    
    def render_performance_ranking(self, comparison_df: pd.DataFrame, 
                                 sort_by: str = 'Total P&L') -> go.Figure:
        """
        Render performance ranking visualization.
        
        Args:
            comparison_df: DataFrame with comparison metrics
            sort_by: Metric to sort by
            
        Returns:
            Plotly figure
        """
        if comparison_df.empty:
            return self.viz._create_empty_chart("No data for ranking")
        
        # Sort by selected metric
        is_ascending = sort_by == 'Max Drawdown'  # Lower is better for drawdown
        sorted_df = comparison_df.sort_values(by=sort_by, ascending=is_ascending)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Determine color based on positive/negative values
        colors = []
        for val in sorted_df[sort_by]:
            if sort_by == 'Max Drawdown':
                colors.append(self.viz.negative_color if val > 0.2 else self.viz.positive_color)
            else:
                colors.append(self.viz.positive_color if val > 0 else self.viz.negative_color)
        
        fig.add_trace(go.Bar(
            y=sorted_df['Strategy'],
            x=sorted_df[sort_by],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.2f}" if not pd.isna(v) else "N/A" for v in sorted_df[sort_by]],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' + sort_by + ': %{x:.2f}<br><extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Strategy Ranking by {sort_by}",
            xaxis_title=sort_by,
            yaxis_title="Strategy",
            height=max(400, len(sorted_df) * 50),
            **self.viz.layout_template
        )
        
        return fig