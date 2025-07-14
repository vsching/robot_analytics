"""Interactive visualization components for trading performance analysis."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
import streamlit as st
from decimal import Decimal

from ..models import Trade
from ..analytics import AnalyticsEngine


class VisualizationComponents:
    """Create interactive charts for trading performance visualization."""
    
    def __init__(self, theme: str = "dark"):
        """
        Initialize visualization components.
        
        Args:
            theme: Color theme ('dark' or 'light')
        """
        self.theme = theme
        self._setup_theme()
    
    def _setup_theme(self):
        """Set up color theme for charts."""
        if self.theme == "dark":
            self.bg_color = "#0E1117"
            self.paper_color = "#262730"
            self.text_color = "#FAFAFA"
            self.grid_color = "#31333F"
            self.positive_color = "#00CC88"
            self.negative_color = "#FF4444"
            self.neutral_color = "#FFB700"
            self.line_color = "#4E7FFF"
            self.area_color = "rgba(78, 127, 255, 0.2)"
        else:
            self.bg_color = "#FFFFFF"
            self.paper_color = "#F8F9FA"
            self.text_color = "#262730"
            self.grid_color = "#E1E4E8"
            self.positive_color = "#22C55E"
            self.negative_color = "#EF4444"
            self.neutral_color = "#F59E0B"
            self.line_color = "#3B82F6"
            self.area_color = "rgba(59, 130, 246, 0.2)"
        
        # Common layout settings
        self.layout_template = {
            "plot_bgcolor": self.paper_color,
            "paper_bgcolor": self.bg_color,
            "font": {"color": self.text_color, "family": "system-ui, sans-serif"},
            "xaxis": {
                "gridcolor": self.grid_color,
                "linecolor": self.grid_color,
                "tickcolor": self.text_color,
                "tickfont": {"color": self.text_color}
            },
            "yaxis": {
                "gridcolor": self.grid_color,
                "linecolor": self.grid_color,
                "tickcolor": self.text_color,
                "tickfont": {"color": self.text_color},
                "zeroline": True,
                "zerolinecolor": self.grid_color
            },
            "hoverlabel": {
                "bgcolor": self.paper_color,
                "bordercolor": self.grid_color,
                "font": {"color": self.text_color}
            }
        }
    
    def render_pnl_chart(self, 
                        trades: List[Trade], 
                        period: str = "daily",
                        chart_type: str = "bar") -> go.Figure:
        """
        Render P&L chart with period aggregation.
        
        Args:
            trades: List of trades
            period: Aggregation period ('daily', 'weekly', 'monthly')
            chart_type: Chart type ('bar', 'line', 'area')
            
        Returns:
            Plotly figure object
        """
        if not trades:
            return self._create_empty_chart("No trades to display")
        
        # Convert trades to DataFrame
        df = pd.DataFrame([{
            'date': t.trade_date,
            'pnl': float(t.pnl) if t.pnl else 0,
            'symbol': t.symbol
        } for t in trades])
        
        # Aggregate by period
        if period == "daily":
            df['period'] = df['date']
            period_format = "%Y-%m-%d"
        elif period == "weekly":
            df['period'] = pd.to_datetime(df['date']).dt.to_period('W').dt.start_time.dt.date
            period_format = "%Y-%m-%d"
        elif period == "monthly":
            df['period'] = pd.to_datetime(df['date']).dt.to_period('M').dt.start_time.dt.date
            period_format = "%Y-%m"
        else:
            raise ValueError(f"Invalid period: {period}")
        
        # Aggregate P&L
        agg_df = df.groupby('period').agg({
            'pnl': 'sum',
            'symbol': 'count'
        }).reset_index()
        agg_df.columns = ['period', 'total_pnl', 'trade_count']
        
        # Add color based on profit/loss
        agg_df['color'] = agg_df['total_pnl'].apply(
            lambda x: self.positive_color if x > 0 else self.negative_color if x < 0 else self.neutral_color
        )
        
        # Create figure
        fig = go.Figure()
        
        if chart_type == "bar":
            fig.add_trace(go.Bar(
                x=agg_df['period'],
                y=agg_df['total_pnl'],
                marker_color=agg_df['color'],
                text=[f"${x:,.2f}" for x in agg_df['total_pnl']],
                textposition='outside',
                hovertemplate=(
                    f"<b>{period.capitalize()} P&L</b><br>" +
                    "Date: %{x|" + period_format + "}<br>" +
                    "P&L: $%{y:,.2f}<br>" +
                    "Trades: %{customdata}<br>" +
                    "<extra></extra>"
                ),
                customdata=agg_df['trade_count']
            ))
        
        elif chart_type == "line":
            fig.add_trace(go.Scatter(
                x=agg_df['period'],
                y=agg_df['total_pnl'],
                mode='lines+markers',
                line=dict(color=self.line_color, width=2),
                marker=dict(size=8, color=agg_df['color']),
                text=[f"${x:,.2f}" for x in agg_df['total_pnl']],
                hovertemplate=(
                    f"<b>{period.capitalize()} P&L</b><br>" +
                    "Date: %{x|" + period_format + "}<br>" +
                    "P&L: $%{y:,.2f}<br>" +
                    "Trades: %{customdata}<br>" +
                    "<extra></extra>"
                ),
                customdata=agg_df['trade_count']
            ))
        
        elif chart_type == "area":
            fig.add_trace(go.Scatter(
                x=agg_df['period'],
                y=agg_df['total_pnl'],
                mode='lines',
                fill='tozeroy',
                line=dict(color=self.line_color, width=2),
                fillcolor=self.area_color,
                hovertemplate=(
                    f"<b>{period.capitalize()} P&L</b><br>" +
                    "Date: %{x|" + period_format + "}<br>" +
                    "P&L: $%{y:,.2f}<br>" +
                    "Trades: %{customdata}<br>" +
                    "<extra></extra>"
                ),
                customdata=agg_df['trade_count']
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{period.capitalize()} P&L Chart",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            showlegend=False,
            hovermode='x unified',
            **self.layout_template
        )
        
        # Add zero line
        fig.add_hline(y=0, line_color=self.grid_color, line_dash="dash")
        
        return fig
    
    def render_cumulative_returns(self, 
                                 trades: List[Trade],
                                 initial_capital: float = 10000,
                                 show_benchmark: bool = False,
                                 benchmark_return: float = 0.08) -> go.Figure:
        """
        Render cumulative returns chart.
        
        Args:
            trades: List of trades
            initial_capital: Starting capital
            show_benchmark: Whether to show benchmark line
            benchmark_return: Annual benchmark return
            
        Returns:
            Plotly figure object
        """
        if not trades:
            return self._create_empty_chart("No trades to display")
        
        # Sort trades by date
        sorted_trades = sorted(trades, key=lambda t: t.trade_date)
        
        # Calculate cumulative P&L
        dates = []
        cumulative_pnl = []
        equity = []
        current_pnl = 0
        
        # Add initial point
        dates.append(sorted_trades[0].trade_date - timedelta(days=1))
        cumulative_pnl.append(0)
        equity.append(initial_capital)
        
        # Process trades
        for trade in sorted_trades:
            if trade.pnl:
                current_pnl += float(trade.pnl)
            dates.append(trade.trade_date)
            cumulative_pnl.append(current_pnl)
            equity.append(initial_capital + current_pnl)
        
        # Calculate returns
        returns = [(e / initial_capital - 1) * 100 for e in equity]
        
        # Create figure
        fig = go.Figure()
        
        # Add strategy returns
        fig.add_trace(go.Scatter(
            x=dates,
            y=returns,
            mode='lines',
            name='Strategy',
            line=dict(color=self.line_color, width=3),
            fill='tozeroy',
            fillcolor=self.area_color,
            hovertemplate=(
                "<b>Strategy Performance</b><br>" +
                "Date: %{x|%Y-%m-%d}<br>" +
                "Return: %{y:.2f}%<br>" +
                "Equity: $%{customdata:,.2f}<br>" +
                "<extra></extra>"
            ),
            customdata=equity
        ))
        
        # Add benchmark if requested
        if show_benchmark:
            # Calculate benchmark returns
            start_date = dates[0]
            benchmark_dates = []
            benchmark_returns = []
            
            for i, date in enumerate(dates):
                days_elapsed = (date - start_date).days
                annual_return = (1 + benchmark_return) ** (days_elapsed / 365) - 1
                benchmark_dates.append(date)
                benchmark_returns.append(annual_return * 100)
            
            fig.add_trace(go.Scatter(
                x=benchmark_dates,
                y=benchmark_returns,
                mode='lines',
                name=f'Benchmark ({benchmark_return:.0%})',
                line=dict(color=self.neutral_color, width=2, dash='dash'),
                hovertemplate=(
                    "<b>Benchmark</b><br>" +
                    "Date: %{x|%Y-%m-%d}<br>" +
                    "Return: %{y:.2f}%<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout
        fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0)"
            ),
            **self.layout_template
        )
        
        # Add zero line
        fig.add_hline(y=0, line_color=self.grid_color, line_dash="dash")
        
        return fig
    
    def render_drawdown_chart(self, 
                             trades: List[Trade],
                             initial_capital: float = 10000) -> go.Figure:
        """
        Render drawdown chart showing underwater equity curve.
        
        Args:
            trades: List of trades
            initial_capital: Starting capital
            
        Returns:
            Plotly figure object
        """
        if not trades:
            return self._create_empty_chart("No trades to display")
        
        # Calculate equity curve
        analytics = AnalyticsEngine()
        equity_curve = analytics.calculate_equity_curve(trades, Decimal(str(initial_capital)))
        
        # Sort trades to get dates
        sorted_trades = sorted(trades, key=lambda t: t.trade_date)
        
        # Create date range
        if sorted_trades:
            start_date = sorted_trades[0].trade_date - timedelta(days=1)
            end_date = sorted_trades[-1].trade_date
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            date_range = [datetime.now().date()]
        
        # Ensure we have the right number of dates for equity curve
        if len(date_range) > len(equity_curve):
            date_range = date_range[:len(equity_curve)]
        elif len(date_range) < len(equity_curve):
            # Extend date range
            extra_days = len(equity_curve) - len(date_range)
            last_date = date_range[-1]
            for i in range(extra_days):
                date_range = date_range.append(pd.DatetimeIndex([last_date + timedelta(days=i+1)]))
        
        # Calculate running maximum and drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = []
        drawdown_ends = []
        
        i = 0
        while i < len(in_drawdown):
            if in_drawdown[i]:
                start = i
                while i < len(in_drawdown) and in_drawdown[i]:
                    i += 1
                end = i - 1
                drawdown_starts.append(start)
                drawdown_ends.append(end)
            else:
                i += 1
        
        # Create figure
        fig = go.Figure()
        
        # Add drawdown area
        fig.add_trace(go.Scatter(
            x=date_range[:len(drawdown)],
            y=drawdown,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.3)',
            line=dict(color=self.negative_color, width=2),
            name='Drawdown',
            hovertemplate=(
                "<b>Drawdown</b><br>" +
                "Date: %{x|%Y-%m-%d}<br>" +
                "Drawdown: %{y:.2f}%<br>" +
                "Equity: $%{customdata:,.2f}<br>" +
                "<extra></extra>"
            ),
            customdata=equity_curve[:len(drawdown)]
        ))
        
        # Highlight recovery periods
        for start, end in zip(drawdown_starts, drawdown_ends):
            if end < len(date_range) - 1:  # Has recovery
                # Find recovery point
                recovery = end + 1
                while recovery < len(drawdown) and drawdown[recovery] < 0:
                    recovery += 1
                
                if recovery < len(drawdown):
                    fig.add_vrect(
                        x0=date_range[end],
                        x1=date_range[recovery],
                        fillcolor=self.positive_color,
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                    )
        
        # Update layout
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            showlegend=False,
            hovermode='x unified',
            **self.layout_template
        )
        
        # Add annotations for maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        if max_dd_idx < len(date_range):
            fig.add_annotation(
                x=date_range[max_dd_idx],
                y=drawdown[max_dd_idx],
                text=f"Max Drawdown: {drawdown[max_dd_idx]:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.negative_color,
                font=dict(size=12, color=self.negative_color)
            )
        
        return fig
    
    def render_monthly_heatmap(self, trades: List[Trade]) -> go.Figure:
        """
        Render monthly returns heatmap.
        
        Args:
            trades: List of trades
            
        Returns:
            Plotly figure object
        """
        if not trades:
            return self._create_empty_chart("No trades to display")
        
        # Convert trades to DataFrame
        df = pd.DataFrame([{
            'date': t.trade_date,
            'pnl': float(t.pnl) if t.pnl else 0
        } for t in trades])
        
        # Extract year and month
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        
        # Aggregate by year and month
        monthly_pnl = df.groupby(['year', 'month'])['pnl'].sum().reset_index()
        
        # Create pivot table
        pivot = monthly_pnl.pivot(index='year', columns='month', values='pnl').fillna(0)
        
        # Ensure all months are present
        for month in range(1, 13):
            if month not in pivot.columns:
                pivot[month] = 0
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=month_names,
            y=pivot.index,
            colorscale=[
                [0, self.negative_color],
                [0.5, self.paper_color],
                [1, self.positive_color]
            ],
            zmid=0,
            text=[[f"${val:,.0f}" for val in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10, "color": self.text_color},
            hovertemplate=(
                "<b>%{x} %{y}</b><br>" +
                "P&L: $%{z:,.2f}<br>" +
                "<extra></extra>"
            ),
            colorbar=dict(
                title=dict(
                    text="P&L ($)",
                    font=dict(color=self.text_color)
                ),
                tickfont=dict(color=self.text_color)
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            **self.layout_template
        )
        
        # Reverse y-axis to show recent years at top
        fig.update_yaxes(autorange="reversed")
        
        return fig
    
    def render_distribution_chart(self, 
                                 trades: List[Trade],
                                 bins: int = 30) -> go.Figure:
        """
        Render return distribution histogram.
        
        Args:
            trades: List of trades
            bins: Number of histogram bins
            
        Returns:
            Plotly figure object
        """
        if not trades:
            return self._create_empty_chart("No trades to display")
        
        # Extract P&L values
        pnl_values = [float(t.pnl) for t in trades if t.pnl is not None]
        
        if not pnl_values:
            return self._create_empty_chart("No P&L data to display")
        
        # Calculate statistics
        mean_pnl = np.mean(pnl_values)
        median_pnl = np.median(pnl_values)
        std_pnl = np.std(pnl_values)
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=pnl_values,
            nbinsx=bins,
            name='P&L Distribution',
            marker_color=self.line_color,
            opacity=0.7,
            hovertemplate=(
                "<b>P&L Range</b><br>" +
                "Range: $%{x}<br>" +
                "Count: %{y}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add normal distribution overlay
        x_range = np.linspace(min(pnl_values), max(pnl_values), 100)
        normal_dist = ((1 / (std_pnl * np.sqrt(2 * np.pi))) * 
                      np.exp(-0.5 * ((x_range - mean_pnl) / std_pnl) ** 2))
        
        # Scale to match histogram
        hist, edges = np.histogram(pnl_values, bins=bins)
        bin_width = edges[1] - edges[0]
        normal_dist_scaled = normal_dist * len(pnl_values) * bin_width
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist_scaled,
            mode='lines',
            name='Normal Distribution',
            line=dict(color=self.negative_color, width=2),
            hovertemplate=(
                "<b>Normal Distribution</b><br>" +
                "P&L: $%{x:.2f}<br>" +
                "Probability: %{y:.4f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add vertical lines for statistics
        fig.add_vline(x=mean_pnl, line_color=self.positive_color, 
                     line_dash="dash", annotation_text=f"Mean: ${mean_pnl:.0f}")
        fig.add_vline(x=median_pnl, line_color=self.neutral_color, 
                     line_dash="dash", annotation_text=f"Median: ${median_pnl:.0f}")
        
        # Update layout
        fig.update_layout(
            title="P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0)"
            ),
            **self.layout_template
        )
        
        return fig
    
    def render_performance_metrics_cards(self, 
                                       metrics: Dict[str, Any]) -> List[go.Figure]:
        """
        Render performance metrics as card-style charts.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            List of Plotly figure objects
        """
        cards = []
        
        # Define metric cards
        card_configs = [
            {
                'title': 'Total P&L',
                'value': metrics.get('total_pnl', 0),
                'format': '${:,.2f}',
                'color': self.positive_color if metrics.get('total_pnl', 0) > 0 else self.negative_color
            },
            {
                'title': 'Win Rate',
                'value': metrics.get('win_rate', 0),
                'format': '{:.1f}%',
                'color': self.positive_color if metrics.get('win_rate', 0) > 50 else self.negative_color
            },
            {
                'title': 'Sharpe Ratio',
                'value': metrics.get('sharpe_ratio', 0),
                'format': '{:.2f}',
                'color': self.positive_color if metrics.get('sharpe_ratio', 0) > 1 else self.neutral_color
            },
            {
                'title': 'Max Drawdown',
                'value': metrics.get('max_drawdown', 0) * 100,
                'format': '{:.1f}%',
                'color': self.negative_color if metrics.get('max_drawdown', 0) > 0.2 else self.neutral_color
            }
        ]
        
        for config in card_configs:
            fig = go.Figure()
            
            # Add indicator
            fig.add_trace(go.Indicator(
                mode="number",
                value=config['value'],
                title={
                    'text': config['title'],
                    'font': {'size': 14, 'color': self.text_color}
                },
                number={
                    'font': {'size': 24, 'color': config['color']},
                    'valueformat': config['format'].replace('{', '').replace('}', '')
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            
            # Update layout
            fig.update_layout(
                height=120,
                margin=dict(l=20, r=20, t=40, b=20),
                **self.layout_template
            )
            
            cards.append(fig)
        
        return cards
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.text_color),
            opacity=0.5
        )
        
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            **self.layout_template
        )
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, format: str = "png"):
        """
        Save chart to file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format ('png', 'svg', 'pdf', 'html')
        """
        if format == "html":
            fig.write_html(filename)
        else:
            fig.write_image(filename, format=format, width=1200, height=600)
    
    def render_in_streamlit(self, fig: go.Figure, key: Optional[str] = None):
        """
        Render chart in Streamlit with proper configuration.
        
        Args:
            fig: Plotly figure
            key: Unique key for Streamlit component
        """
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'trading_chart',
                    'height': 600,
                    'width': 1200,
                    'scale': 2
                }
            },
            key=key
        )