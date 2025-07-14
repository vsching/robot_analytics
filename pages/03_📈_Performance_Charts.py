"""
Performance Charts Page

This page provides interactive visualizations for trading performance analysis
including P&L charts, cumulative returns, drawdown analysis, and distribution charts.
"""

import streamlit as st
from datetime import datetime, timedelta

from src.db.connection import get_db_manager
from src.services.strategy_manager import StrategyManager
from src.analytics import AnalyticsEngine, CachedAnalyticsEngine, MetricsCacheManager
from src.components import VisualizationComponents, StrategySelector


def main():
    st.set_page_config(
        page_title="Performance Charts - Trading Strategy Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Performance Charts")
    st.markdown("Interactive visualizations for comprehensive trading performance analysis")
    
    # Initialize components
    db_manager = get_db_manager()
    strategy_manager = StrategyManager(db_manager)
    cache_manager = MetricsCacheManager(db_manager)
    analytics_engine = AnalyticsEngine(db_manager)
    cached_analytics = CachedAnalyticsEngine(analytics_engine, cache_manager)
    viz = VisualizationComponents(theme="dark")
    
    # Strategy selection
    with st.sidebar:
        st.header("Chart Settings")
        
        # Strategy selector
        selector = StrategySelector(strategy_manager)
        selected_strategy_id = selector.render()
        
        if selected_strategy_id:
            strategy = strategy_manager.get_strategy(selected_strategy_id)
            st.info(f"Selected: {strategy.name}")
            
            # Date range filter
            st.subheader("Date Range")
            col1, col2 = st.columns(2)
            
            # Get trade date range
            trades = analytics_engine.get_trades_for_strategy(selected_strategy_id)
            if trades:
                min_date = min((t.trade_date.date() if hasattr(t.trade_date, 'date') else t.trade_date) for t in trades)
                max_date = max((t.trade_date.date() if hasattr(t.trade_date, 'date') else t.trade_date) for t in trades)
                
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                # Filter trades by date
                # Convert trade_date to date if it's a datetime
                trades = [t for t in trades 
                         if start_date <= (t.trade_date.date() if hasattr(t.trade_date, 'date') else t.trade_date) <= end_date]
    
    # Main content area
    if selected_strategy_id and trades:
        # Calculate metrics
        metrics = cached_analytics.calculate_metrics_for_strategy(selected_strategy_id)
        
        if metrics:
            # Performance metric cards
            st.subheader("Key Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            metric_data = {
                'total_pnl': metrics.pnl_summary.total_pnl,
                'win_rate': metrics.trade_statistics.win_rate,
                'sharpe_ratio': metrics.advanced_statistics.sharpe_ratio if metrics.advanced_statistics else None,
                'max_drawdown': metrics.advanced_statistics.max_drawdown if metrics.advanced_statistics else 0
            }
            
            cards = viz.render_performance_metrics_cards(metric_data)
            
            with col1:
                st.plotly_chart(cards[0], use_container_width=True, key="card1")
            with col2:
                st.plotly_chart(cards[1], use_container_width=True, key="card2")
            with col3:
                st.plotly_chart(cards[2], use_container_width=True, key="card3")
            with col4:
                st.plotly_chart(cards[3], use_container_width=True, key="card4")
        
        # Chart tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 P&L Analysis", 
            "📈 Returns", 
            "📉 Drawdown", 
            "🗓️ Monthly Heatmap", 
            "📊 Distribution"
        ])
        
        with tab1:
            st.subheader("Profit & Loss Analysis")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                period = st.selectbox(
                    "Aggregation Period",
                    ["daily", "weekly", "monthly"],
                    index=0
                )
            with col2:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["bar", "line", "area"],
                    index=0
                )
            
            # Render P&L chart
            pnl_fig = viz.render_pnl_chart(trades, period=period, chart_type=chart_type)
            viz.render_in_streamlit(pnl_fig, key="pnl_chart")
            
            # Summary statistics
            with st.expander("P&L Summary Statistics"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trades", len(trades))
                    st.metric("Winning Trades", metrics.trade_statistics.winning_trades)
                with col2:
                    st.metric("Average P&L", f"${metrics.pnl_summary.average_pnl:,.2f}")
                    st.metric("Median P&L", f"${metrics.pnl_summary.median_pnl:,.2f}")
                with col3:
                    st.metric("Best Trade", f"${metrics.pnl_summary.max_pnl:,.2f}")
                    st.metric("Worst Trade", f"${metrics.pnl_summary.min_pnl:,.2f}")
        
        with tab2:
            st.subheader("Cumulative Returns Analysis")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    max_value=1000000,
                    value=10000,
                    step=1000
                )
            with col2:
                show_benchmark = st.checkbox("Show Benchmark", value=True)
                if show_benchmark:
                    benchmark_return = st.slider(
                        "Annual Benchmark Return (%)",
                        min_value=0,
                        max_value=20,
                        value=8,
                        step=1
                    ) / 100
                else:
                    benchmark_return = 0.08
            
            # Render cumulative returns chart
            returns_fig = viz.render_cumulative_returns(
                trades, 
                initial_capital=initial_capital,
                show_benchmark=show_benchmark,
                benchmark_return=benchmark_return
            )
            viz.render_in_streamlit(returns_fig, key="returns_chart")
            
            # Return statistics
            if metrics.advanced_statistics:
                with st.expander("Return Statistics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sharpe Ratio", f"{metrics.advanced_statistics.sharpe_ratio:.2f}" 
                                 if metrics.advanced_statistics.sharpe_ratio else "N/A")
                        st.metric("Calmar Ratio", f"{metrics.advanced_statistics.calmar_ratio:.2f}"
                                 if metrics.advanced_statistics.calmar_ratio else "N/A")
                    with col2:
                        st.metric("Sortino Ratio", f"{metrics.advanced_statistics.sortino_ratio:.2f}"
                                 if metrics.advanced_statistics.sortino_ratio else "N/A")
                        st.metric("Omega Ratio", f"{metrics.advanced_statistics.omega_ratio:.2f}"
                                 if metrics.advanced_statistics.omega_ratio else "N/A")
        
        with tab3:
            st.subheader("Drawdown Analysis")
            
            # Render drawdown chart
            drawdown_fig = viz.render_drawdown_chart(trades, initial_capital=10000)
            viz.render_in_streamlit(drawdown_fig, key="drawdown_chart")
            
            # Drawdown statistics
            if metrics.advanced_statistics:
                with st.expander("Drawdown Statistics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Maximum Drawdown", 
                                 f"{metrics.advanced_statistics.max_drawdown:.1%}")
                    with col2:
                        st.metric("Drawdown Duration", 
                                 f"{metrics.advanced_statistics.max_drawdown_duration} days"
                                 if metrics.advanced_statistics.max_drawdown_duration else "N/A")
                    with col3:
                        st.metric("Recovery Duration", 
                                 f"{metrics.advanced_statistics.recovery_duration} days"
                                 if metrics.advanced_statistics.recovery_duration else "Not Recovered")
        
        with tab4:
            st.subheader("Monthly Returns Heatmap")
            
            # Render monthly heatmap
            heatmap_fig = viz.render_monthly_heatmap(trades)
            viz.render_in_streamlit(heatmap_fig, key="heatmap_chart")
            
            # Monthly statistics
            with st.expander("Monthly Performance Summary"):
                if metrics.monthly_metrics:
                    best_month = max(metrics.monthly_metrics, key=lambda x: x.total_pnl)
                    worst_month = min(metrics.monthly_metrics, key=lambda x: x.total_pnl)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best Month", 
                                 f"{best_month.period_start.strftime('%b %Y')}: ${best_month.total_pnl:,.2f}")
                    with col2:
                        st.metric("Worst Month", 
                                 f"{worst_month.period_start.strftime('%b %Y')}: ${worst_month.total_pnl:,.2f}")
        
        with tab5:
            st.subheader("P&L Distribution Analysis")
            
            bins = st.slider("Number of Bins", min_value=10, max_value=50, value=30, step=5)
            
            # Render distribution chart
            dist_fig = viz.render_distribution_chart(trades, bins=bins)
            viz.render_in_streamlit(dist_fig, key="distribution_chart")
            
            # Distribution statistics
            with st.expander("Distribution Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Standard Deviation", f"${metrics.pnl_summary.std_dev:,.2f}")
                    st.metric("Value at Risk (95%)", 
                             f"${metrics.advanced_statistics.value_at_risk_95 * 10000:.2f}"
                             if metrics.advanced_statistics and metrics.advanced_statistics.value_at_risk_95 else "N/A")
                with col2:
                    st.metric("Profit Factor", f"{metrics.trade_statistics.profit_factor:.2f}")
                    st.metric("Conditional VaR (95%)", 
                             f"${metrics.advanced_statistics.conditional_var_95 * 10000:.2f}"
                             if metrics.advanced_statistics and metrics.advanced_statistics.conditional_var_95 else "N/A")
        
        # Export options
        st.divider()
        st.subheader("Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download All Charts", type="primary"):
                st.info("Chart download functionality will be implemented in the export module")
        
        with col2:
            if st.button("📊 Export Raw Data"):
                st.info("Data export functionality will be implemented in the export module")
        
        with col3:
            if st.button("📄 Generate Report"):
                st.info("Report generation will be implemented in the export module")
    
    elif selected_strategy_id:
        st.warning("No trades found for the selected strategy in the specified date range.")
    else:
        st.info("👈 Please select a strategy from the sidebar to view performance charts.")
        
        # Show demo
        with st.expander("📚 Chart Types Overview"):
            st.markdown("""
            ### Available Visualizations:
            
            1. **P&L Analysis** - View profit/loss over time with daily, weekly, or monthly aggregation
            2. **Cumulative Returns** - Track portfolio growth with optional benchmark comparison
            3. **Drawdown Analysis** - Understand risk through underwater equity curves
            4. **Monthly Heatmap** - Identify seasonal patterns in trading performance
            5. **Distribution Analysis** - Analyze the statistical distribution of returns
            
            Each chart is fully interactive with zoom, pan, and hover capabilities. You can also download 
            charts in various formats for reports and presentations.
            """)


if __name__ == "__main__":
    main()