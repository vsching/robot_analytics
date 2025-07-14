"""
Confluence Dashboard Component

Interactive dashboard for visualizing and analyzing signal confluence between trading strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import logging

from ..analytics.confluence_analyzer import ConfluenceAnalyzer, SignalOverlap, ConfluenceMetrics
from ..models import Trade, Strategy
from ..services.strategy_manager import StrategyManager
from ..analytics import AnalyticsEngine
from .visualization import VisualizationComponents


logger = logging.getLogger(__name__)


class ConfluenceDashboard:
    """Dashboard for confluence detection and analysis."""
    
    def __init__(self, 
                 strategy_manager: StrategyManager,
                 analytics_engine: AnalyticsEngine,
                 viz_components: Optional[VisualizationComponents] = None):
        """
        Initialize confluence dashboard.
        
        Args:
            strategy_manager: Strategy manager instance
            analytics_engine: Analytics engine instance
            viz_components: Visualization components instance
        """
        self.strategy_manager = strategy_manager
        self.analytics_engine = analytics_engine
        self.viz = viz_components or VisualizationComponents(theme="dark")
        self.confluence_analyzer = ConfluenceAnalyzer()
    
    def render_confluence_settings(self) -> Dict[str, Any]:
        """
        Render confluence analysis settings sidebar.
        
        Returns:
            Dictionary with analysis settings
        """
        st.sidebar.header("⚙️ Confluence Settings")
        
        # Time window for confluence detection
        time_window_hours = st.sidebar.slider(
            "Time Window (hours)",
            min_value=1,
            max_value=72,
            value=24,
            step=1,
            help="Maximum time difference between trades to be considered confluent"
        )
        
        # Minimum strategies for confluence
        min_strategies = st.sidebar.slider(
            "Minimum Strategies",
            min_value=2,
            max_value=6,
            value=2,
            step=1,
            help="Minimum number of strategies required for confluence"
        )
        
        # Minimum confluence strength
        min_strength = st.sidebar.slider(
            "Minimum Strength Score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum confluence strength score to display"
        )
        
        # Date range filter
        st.sidebar.subheader("📅 Date Range")
        
        # Get available date range
        all_strategies = self.strategy_manager.get_all_strategies()
        if all_strategies:
            all_trades = []
            for strategy in all_strategies:
                trades = self.analytics_engine.get_trades_for_strategy(strategy.id)
                all_trades.extend(trades)
            
            if all_trades:
                min_date = min(t.trade_date for t in all_trades)
                max_date = max(t.trade_date for t in all_trades)
                
                start_date = st.sidebar.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                end_date = st.sidebar.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                start_date = date.today() - timedelta(days=30)
                end_date = date.today()
        else:
            start_date = date.today() - timedelta(days=30)
            end_date = date.today()
        
        # Update analyzer settings
        self.confluence_analyzer.time_window = timedelta(hours=time_window_hours)
        self.confluence_analyzer.min_strategies = min_strategies
        
        return {
            'time_window_hours': time_window_hours,
            'min_strategies': min_strategies,
            'min_strength': min_strength,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def get_confluence_data(self, settings: Dict[str, Any]) -> Tuple[List[SignalOverlap], ConfluenceMetrics]:
        """
        Get confluence analysis data based on settings.
        
        Args:
            settings: Analysis settings from render_confluence_settings
            
        Returns:
            Tuple of (overlaps, metrics)
        """
        # Get all strategies and their trades
        strategies = self.strategy_manager.get_all_strategies()
        if len(strategies) < settings['min_strategies']:
            return [], None
        
        strategy_names = {s.id: s.name for s in strategies}
        strategies_trades = {}
        all_trades = []
        
        for strategy in strategies:
            trades = self.analytics_engine.get_trades_for_strategy(strategy.id)
            # Filter by date range
            filtered_trades = [
                t for t in trades 
                if settings['start_date'] <= t.trade_date <= settings['end_date']
            ]
            
            if filtered_trades:
                strategies_trades[strategy.id] = filtered_trades
                all_trades.extend(filtered_trades)
        
        if len(strategies_trades) < settings['min_strategies']:
            return [], None
        
        # Find signal overlaps
        overlaps = self.confluence_analyzer.find_signal_overlaps(strategies_trades, strategy_names)
        
        # Filter by minimum strength
        overlaps = [o for o in overlaps if o.overlap_strength >= settings['min_strength']]
        
        # Calculate performance metrics
        metrics = None
        if overlaps:
            metrics = self.confluence_analyzer.analyze_confluence_performance(overlaps, all_trades)
        
        return overlaps, metrics
    
    def render_confluence_overview(self, overlaps: List[SignalOverlap], metrics: ConfluenceMetrics):
        """Render confluence overview metrics."""
        if not overlaps or not metrics:
            st.warning("No confluence detected with current settings")
            return
        
        st.subheader("🎯 Confluence Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Confluences",
                metrics.total_overlaps,
                help="Number of detected signal overlaps"
            )
        
        with col2:
            advantage_delta = f"{metrics.confluence_advantage:+.1f}%" if metrics.confluence_advantage != 0 else None
            st.metric(
                "Confluence Win Rate",
                f"{metrics.overlap_win_rate:.1%}",
                delta=advantage_delta,
                help="Win rate during confluence periods"
            )
        
        with col3:
            pnl_delta = f"${metrics.overlap_avg_pnl - metrics.individual_avg_pnl:+.2f}" if metrics.individual_avg_pnl != 0 else None
            st.metric(
                "Avg Confluence P&L",
                f"${metrics.overlap_avg_pnl:.2f}",
                delta=pnl_delta,
                help="Average P&L during confluence periods"
            )
        
        with col4:
            st.metric(
                "Individual Win Rate",
                f"{metrics.individual_win_rate:.1%}",
                help="Win rate for non-confluence trades"
            )
        
        # Performance comparison
        if metrics.confluence_advantage != 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if metrics.confluence_advantage > 5:
                    st.success(f"✅ Confluence shows {metrics.confluence_advantage:.1f}% performance advantage")
                elif metrics.confluence_advantage > 0:
                    st.info(f"📊 Confluence shows {metrics.confluence_advantage:.1f}% slight advantage")
                else:
                    st.warning(f"⚠️ Confluence shows {abs(metrics.confluence_advantage):.1f}% performance disadvantage")
            
            with col2:
                # Best performing combinations
                if metrics.best_confluence_strategies:
                    st.markdown("**🏆 Top Strategy Combinations:**")
                    for i, (strategies, avg_pnl) in enumerate(metrics.best_confluence_strategies[:3], 1):
                        strategy_names = " + ".join(strategies)
                        st.text(f"{i}. {strategy_names}: ${avg_pnl:.2f}")
    
    def render_confluence_timeline(self, overlaps: List[SignalOverlap]) -> go.Figure:
        """
        Render timeline visualization of confluence events.
        
        Args:
            overlaps: List of signal overlaps
            
        Returns:
            Plotly figure
        """
        if not overlaps:
            return self.viz._create_empty_chart("No confluence events to display")
        
        # Prepare timeline data
        timeline_data = []
        colors = px.colors.qualitative.Set3
        
        for i, overlap in enumerate(overlaps):
            total_pnl = sum(float(t.pnl) for t in overlap.trades if t.pnl)
            
            timeline_data.append({
                'x': overlap.center_time,
                'y': len(overlap.strategies),
                'size': overlap.overlap_strength * 100,
                'color': total_pnl,
                'strategies': ', '.join(overlap.strategy_names),
                'pnl': total_pnl,
                'strength': overlap.overlap_strength,
                'type': overlap.confluence_type,
                'symbols': ', '.join(overlap.symbols),
                'num_trades': len(overlap.trades)
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add confluence events
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=df['size'],
                color=df['color'],
                colorscale='RdYlGn',
                colorbar=dict(title="P&L ($)"),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=[
                f"<b>{row['strategies']}</b><br>" +
                f"Time: {row['x'].strftime('%Y-%m-%d %H:%M')}<br>" +
                f"Type: {row['type']}<br>" +
                f"Strength: {row['strength']:.2f}<br>" +
                f"P&L: ${row['pnl']:.2f}<br>" +
                f"Symbols: {row['symbols']}<br>" +
                f"Trades: {row['num_trades']}"
                for _, row in df.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Confluence Events'
        ))
        
        # Update layout
        fig.update_layout(
            title="Confluence Timeline",
            xaxis_title="Date",
            yaxis_title="Number of Strategies",
            showlegend=False,
            **self.viz.layout_template
        )
        
        # Add strength reference lines
        fig.add_hline(y=2, line_dash="dash", line_color="gray", 
                     annotation_text="Min Strategies", annotation_position="bottom right")
        
        return fig
    
    def render_confluence_strength_distribution(self, overlaps: List[SignalOverlap]) -> go.Figure:
        """
        Render distribution of confluence strengths.
        
        Args:
            overlaps: List of signal overlaps
            
        Returns:
            Plotly figure
        """
        if not overlaps:
            return self.viz._create_empty_chart("No confluence data for distribution")
        
        strengths = [overlap.overlap_strength for overlap in overlaps]
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=strengths,
            nbinsx=20,
            name="Confluence Strength",
            marker_color=self.viz.positive_color,
            opacity=0.7
        ))
        
        # Add mean line
        mean_strength = np.mean(strengths)
        fig.add_vline(
            x=mean_strength,
            line_dash="dash",
            line_color=self.viz.text_color,
            annotation_text=f"Mean: {mean_strength:.2f}"
        )
        
        fig.update_layout(
            title="Confluence Strength Distribution",
            xaxis_title="Strength Score",
            yaxis_title="Frequency",
            **self.viz.layout_template
        )
        
        return fig
    
    def render_confluence_type_analysis(self, overlaps: List[SignalOverlap]) -> go.Figure:
        """
        Render analysis of confluence types.
        
        Args:
            overlaps: List of signal overlaps
            
        Returns:
            Plotly figure
        """
        if not overlaps:
            return self.viz._create_empty_chart("No confluence data for type analysis")
        
        # Count confluence types
        type_counts = {}
        type_performance = {}
        
        for overlap in overlaps:
            conf_type = overlap.confluence_type
            total_pnl = sum(float(t.pnl) for t in overlap.trades if t.pnl)
            
            if conf_type not in type_counts:
                type_counts[conf_type] = 0
                type_performance[conf_type] = []
            
            type_counts[conf_type] += 1
            type_performance[conf_type].append(total_pnl)
        
        # Calculate average performance by type
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        avg_performance = [np.mean(type_performance[t]) for t in types]
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Confluence Type Frequency", "Average Performance by Type"),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart for frequency
        fig.add_trace(
            go.Pie(
                labels=types,
                values=counts,
                name="Frequency",
                marker_colors=px.colors.qualitative.Set3[:len(types)]
            ),
            row=1, col=1
        )
        
        # Bar chart for performance
        colors = [self.viz.positive_color if p >= 0 else self.viz.negative_color for p in avg_performance]
        
        fig.add_trace(
            go.Bar(
                x=types,
                y=avg_performance,
                name="Avg P&L",
                marker_color=colors,
                text=[f"${p:.2f}" for p in avg_performance],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Confluence Type Analysis",
            **self.viz.layout_template
        )
        
        return fig
    
    def render_confluence_calendar(self, overlaps: List[SignalOverlap]) -> pd.DataFrame:
        """
        Render confluence calendar table.
        
        Args:
            overlaps: List of signal overlaps
            
        Returns:
            DataFrame for display
        """
        if not overlaps:
            return pd.DataFrame()
        
        calendar_df = self.confluence_analyzer.get_confluence_calendar(overlaps)
        
        if not calendar_df.empty:
            # Format for display
            display_df = calendar_df.copy()
            display_df['strength'] = display_df['strength'].apply(lambda x: f"{x:.2f}")
            display_df['total_pnl'] = display_df['total_pnl'].apply(lambda x: f"${x:.2f}")
            
            # Rename columns for better display
            display_df = display_df.rename(columns={
                'date': 'Date',
                'time': 'Time',
                'strategies': 'Strategies',
                'num_strategies': '# Strategies',
                'symbols': 'Symbols',
                'confluence_type': 'Type',
                'strength': 'Strength',
                'total_pnl': 'Total P&L',
                'num_trades': '# Trades'
            })
        
        return display_df
    
    def render_real_time_confluence(self, lookback_hours: int = 24) -> List[SignalOverlap]:
        """
        Render real-time confluence detection.
        
        Args:
            lookback_hours: Hours to look back for recent signals
            
        Returns:
            List of recent confluence events
        """
        st.subheader("🔴 Real-Time Confluence Detection")
        
        # Get recent trades
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        strategies = self.strategy_manager.get_all_strategies()
        strategy_names = {s.id: s.name for s in strategies}
        
        recent_signals = {}
        for strategy in strategies:
            trades = self.analytics_engine.get_trades_for_strategy(strategy.id)
            recent_trades = []
            
            for trade in trades:
                trade_time = trade.trade_date if isinstance(trade.trade_date, datetime) else datetime.combine(trade.trade_date, datetime.min.time())
                if trade_time >= cutoff_time:
                    recent_trades.append(trade)
            
            if recent_trades:
                recent_signals[strategy.id] = recent_trades
        
        # Find real-time confluence
        current_confluences = self.confluence_analyzer.find_real_time_confluence(
            recent_signals, strategy_names, lookback_hours
        )
        
        if current_confluences:
            st.success(f"🎯 {len(current_confluences)} active confluence event(s) detected!")
            
            for i, confluence in enumerate(current_confluences, 1):
                with st.expander(f"Confluence Event {i} - Strength: {confluence.overlap_strength:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Strategies:** {', '.join(confluence.strategy_names)}")
                        st.write(f"**Time:** {confluence.center_time.strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Type:** {confluence.confluence_type}")
                    
                    with col2:
                        st.write(f"**Symbols:** {', '.join(confluence.symbols)}")
                        st.write(f"**Sides:** {', '.join(confluence.sides)}")
                        st.write(f"**Trades:** {len(confluence.trades)}")
        else:
            st.info("No current confluence events detected")
        
        return current_confluences