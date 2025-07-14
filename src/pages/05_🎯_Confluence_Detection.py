"""
Confluence Detection Page

This page provides tools for detecting and analyzing signal confluence between
multiple trading strategies, helping identify stronger trading opportunities.
"""

import streamlit as st
from datetime import datetime, timedelta

from src.db.connection import get_db_manager
from src.services.strategy_manager import StrategyManager
from src.analytics import AnalyticsEngine, CachedAnalyticsEngine, MetricsCacheManager
from src.components import VisualizationComponents
from src.components.confluence_dashboard import ConfluenceDashboard


def main():
    st.set_page_config(
        page_title="Confluence Detection - Trading Strategy Analyzer",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Signal Confluence Detection")
    st.markdown("""
    Identify when multiple trading strategies generate signals simultaneously, 
    which may indicate stronger trading opportunities with higher probability of success.
    """)
    
    # Initialize components
    db_manager = get_db_manager()
    strategy_manager = StrategyManager(db_manager)
    cache_manager = MetricsCacheManager(db_manager)
    analytics_engine = AnalyticsEngine(db_manager)
    cached_analytics = CachedAnalyticsEngine(analytics_engine, cache_manager)
    viz = VisualizationComponents(theme="dark")
    
    # Create confluence dashboard
    confluence_dashboard = ConfluenceDashboard(
        strategy_manager=strategy_manager,
        analytics_engine=cached_analytics,
        viz_components=viz
    )
    
    # Check if we have enough strategies
    strategies = strategy_manager.get_all_strategies()
    if len(strategies) < 2:
        st.error("⚠️ You need at least 2 strategies to perform confluence analysis.")
        st.info("💡 Please add more strategies using the Strategy Management page.")
        return
    
    # Render settings sidebar
    settings = confluence_dashboard.render_confluence_settings()
    
    # Add analysis trigger button
    with st.sidebar:
        st.divider()
        analyze_button = st.button("🔍 Analyze Confluence", type="primary")
        
        # Real-time monitoring toggle
        st.subheader("🔴 Real-Time Monitoring")
        enable_realtime = st.checkbox("Enable Real-Time Detection", value=False)
        
        if enable_realtime:
            realtime_lookback = st.slider(
                "Lookback Hours",
                min_value=1,
                max_value=72,
                value=24,
                help="How far back to look for recent signals"
            )
            
            if st.button("🔄 Refresh Real-Time"):
                st.experimental_rerun()
    
    # Main content area
    if analyze_button or 'confluence_data' in st.session_state:
        if analyze_button:
            with st.spinner("Analyzing signal confluence..."):
                overlaps, metrics = confluence_dashboard.get_confluence_data(settings)
                st.session_state['confluence_data'] = (overlaps, metrics)
        
        overlaps, metrics = st.session_state['confluence_data']
        
        if overlaps and metrics:
            # Overview metrics
            confluence_dashboard.render_confluence_overview(overlaps, metrics)
            
            # Main analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📅 Timeline", 
                "📊 Analysis", 
                "📋 Calendar", 
                "🎯 Live Detection",
                "📈 Performance"
            ])
            
            with tab1:
                st.subheader("Confluence Timeline")
                st.markdown("Interactive timeline showing when signal overlaps occurred")
                
                timeline_fig = confluence_dashboard.render_confluence_timeline(overlaps)
                st.plotly_chart(timeline_fig, use_container_width=True)
                
                # Timeline insights
                with st.expander("📊 Timeline Insights"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Most active period
                        overlap_hours = [o.center_time.hour for o in overlaps]
                        if overlap_hours:
                            most_active_hour = max(set(overlap_hours), key=overlap_hours.count)
                            st.metric(
                                "Most Active Hour",
                                f"{most_active_hour:02d}:00",
                                help="Hour with most confluence events"
                            )
                    
                    with col2:
                        # Average strength
                        avg_strength = sum(o.overlap_strength for o in overlaps) / len(overlaps)
                        st.metric(
                            "Average Strength",
                            f"{avg_strength:.2f}",
                            help="Average confluence strength score"
                        )
                    
                    with col3:
                        # Peak strategies
                        max_strategies = max(len(o.strategies) for o in overlaps)
                        st.metric(
                            "Max Strategies",
                            max_strategies,
                            help="Maximum strategies in single confluence"
                        )
            
            with tab2:
                st.subheader("Confluence Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Strength Distribution")
                    strength_fig = confluence_dashboard.render_confluence_strength_distribution(overlaps)
                    st.plotly_chart(strength_fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Type Analysis")
                    type_fig = confluence_dashboard.render_confluence_type_analysis(overlaps)
                    st.plotly_chart(type_fig, use_container_width=True)
                
                # Detailed statistics
                st.divider()
                st.markdown("#### 📊 Detailed Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Confluence Types:**")
                    type_counts = {}
                    for overlap in overlaps:
                        t = overlap.confluence_type
                        type_counts[t] = type_counts.get(t, 0) + 1
                    
                    for conf_type, count in sorted(type_counts.items()):
                        st.text(f"• {conf_type}: {count}")
                
                with col2:
                    st.markdown("**Strength Distribution:**")
                    strengths = [o.overlap_strength for o in overlaps]
                    st.text(f"• Min: {min(strengths):.2f}")
                    st.text(f"• Max: {max(strengths):.2f}")
                    st.text(f"• Mean: {sum(strengths)/len(strengths):.2f}")
                    st.text(f"• Median: {sorted(strengths)[len(strengths)//2]:.2f}")
                
                with col3:
                    st.markdown("**Strategy Participation:**")
                    strategy_participation = {}
                    for overlap in overlaps:
                        for strategy_name in overlap.strategy_names:
                            strategy_participation[strategy_name] = strategy_participation.get(strategy_name, 0) + 1
                    
                    # Show top 5 most active strategies
                    sorted_participation = sorted(strategy_participation.items(), key=lambda x: x[1], reverse=True)
                    for strategy, count in sorted_participation[:5]:
                        st.text(f"• {strategy}: {count}")
            
            with tab3:
                st.subheader("Confluence Calendar")
                st.markdown("Detailed view of all confluence events")
                
                calendar_df = confluence_dashboard.render_confluence_calendar(overlaps)
                
                if not calendar_df.empty:
                    # Filter options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        unique_types = calendar_df['Type'].unique()
                        selected_types = st.multiselect(
                            "Filter by Type",
                            options=unique_types,
                            default=unique_types
                        )
                    
                    with col2:
                        min_strength = st.slider(
                            "Minimum Strength",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.1
                        )
                    
                    with col3:
                        min_strategies = st.slider(
                            "Minimum # Strategies",
                            min_value=2,
                            max_value=calendar_df['# Strategies'].max(),
                            value=2
                        )
                    
                    # Apply filters
                    filtered_df = calendar_df[
                        (calendar_df['Type'].isin(selected_types)) &
                        (calendar_df['Strength'].astype(float) >= min_strength) &
                        (calendar_df['# Strategies'] >= min_strategies)
                    ]
                    
                    # Display table
                    st.dataframe(
                        filtered_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export option
                    if st.button("📥 Export Calendar Data"):
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="confluence_calendar.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("No confluence events found with current filters")
            
            with tab4:
                # Real-time confluence detection
                if enable_realtime:
                    recent_confluences = confluence_dashboard.render_real_time_confluence(realtime_lookback)
                    
                    if recent_confluences:
                        st.divider()
                        st.subheader("Recent Confluence Timeline")
                        
                        # Show mini timeline of recent events
                        recent_fig = confluence_dashboard.render_confluence_timeline(recent_confluences)
                        st.plotly_chart(recent_fig, use_container_width=True, key="recent_timeline")
                        
                        # Alert settings
                        st.subheader("🔔 Alert Settings")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            alert_threshold = st.slider(
                                "Alert Strength Threshold",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.7,
                                step=0.1,
                                help="Minimum strength to trigger alerts"
                            )
                        
                        with col2:
                            alert_strategies = st.slider(
                                "Alert Strategy Count",
                                min_value=2,
                                max_value=6,
                                value=3,
                                help="Minimum strategies for alert"
                            )
                        
                        # Check for high-strength alerts
                        high_strength_confluences = [
                            c for c in recent_confluences 
                            if c.overlap_strength >= alert_threshold and len(c.strategies) >= alert_strategies
                        ]
                        
                        if high_strength_confluences:
                            st.error(f"🚨 {len(high_strength_confluences)} High-Strength Confluence Alert(s)!")
                            for confluence in high_strength_confluences:
                                st.warning(
                                    f"🎯 **{', '.join(confluence.strategy_names)}** - "
                                    f"Strength: {confluence.overlap_strength:.2f} - "
                                    f"Time: {confluence.center_time.strftime('%H:%M')}"
                                )
                        else:
                            st.success("✅ No high-strength confluence alerts at this time")
                else:
                    st.info("Enable real-time monitoring in the sidebar to see live confluence detection")
                    
                    # Show example of what real-time monitoring provides
                    with st.expander("🔍 What is Real-Time Monitoring?"):
                        st.markdown("""
                        Real-time confluence monitoring provides:
                        
                        **🔴 Live Detection**
                        - Monitors recent trades across all strategies
                        - Identifies signal overlaps as they occur
                        - Configurable lookback period (1-72 hours)
                        
                        **🚨 Smart Alerts**
                        - Customizable strength thresholds
                        - Strategy count requirements
                        - Visual and text notifications
                        
                        **📊 Recent Timeline**
                        - Visual timeline of recent confluence events
                        - Immediate performance feedback
                        - Historical context for current signals
                        
                        **⚙️ Configuration**
                        - Adjustable sensitivity settings
                        - Custom alert parameters
                        - Real-time refresh capabilities
                        """)
            
            with tab5:
                st.subheader("Performance Impact Analysis")
                
                if metrics:
                    # Performance comparison chart
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Win rate comparison
                        win_rate_data = {
                            'Type': ['Confluence Trades', 'Individual Trades'],
                            'Win Rate': [metrics.overlap_win_rate * 100, metrics.individual_win_rate * 100],
                            'Color': ['Confluence', 'Individual']
                        }
                        
                        win_rate_fig = px.bar(
                            win_rate_data,
                            x='Type',
                            y='Win Rate',
                            color='Color',
                            title="Win Rate Comparison",
                            color_discrete_map={
                                'Confluence': viz.positive_color,
                                'Individual': viz.neutral_color
                            }
                        )
                        win_rate_fig.update_layout(**viz.layout_template)
                        st.plotly_chart(win_rate_fig, use_container_width=True)
                    
                    with col2:
                        # P&L comparison
                        pnl_data = {
                            'Type': ['Confluence Trades', 'Individual Trades'],
                            'Avg P&L': [metrics.overlap_avg_pnl, metrics.individual_avg_pnl],
                            'Color': ['Confluence', 'Individual']
                        }
                        
                        pnl_fig = px.bar(
                            pnl_data,
                            x='Type',
                            y='Avg P&L',
                            color='Color',
                            title="Average P&L Comparison",
                            color_discrete_map={
                                'Confluence': viz.positive_color,
                                'Individual': viz.neutral_color
                            }
                        )
                        pnl_fig.update_layout(**viz.layout_template)
                        st.plotly_chart(pnl_fig, use_container_width=True)
                    
                    # Best performing combinations
                    if metrics.best_confluence_strategies:
                        st.divider()
                        st.subheader("🏆 Top Performing Strategy Combinations")
                        
                        for i, (strategies, avg_pnl) in enumerate(metrics.best_confluence_strategies, 1):
                            with st.container():
                                col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                                
                                with col1:
                                    st.markdown(f"**#{i}**")
                                
                                with col2:
                                    strategy_names = " + ".join(strategies)
                                    st.markdown(f"**{strategy_names}**")
                                
                                with col3:
                                    color = "normal" if avg_pnl >= 0 else "inverse"
                                    st.metric("Avg P&L", f"${avg_pnl:.2f}", delta_color=color)
                    
                    # Performance insights
                    with st.expander("📊 Performance Insights"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Key Findings:**")
                            if metrics.confluence_advantage > 10:
                                st.success("✅ Strong confluence advantage detected")
                            elif metrics.confluence_advantage > 0:
                                st.info("📊 Moderate confluence benefit")
                            else:
                                st.warning("⚠️ Confluence may not provide significant advantage")
                            
                            st.text(f"• Confluence trades: {len([t for o in overlaps for t in o.trades])}")
                            st.text(f"• Average confluence strength: {sum(o.overlap_strength for o in overlaps)/len(overlaps):.2f}")
                            st.text(f"• Most common type: {max(metrics.overlap_frequency, key=metrics.overlap_frequency.get)}")
                        
                        with col2:
                            st.markdown("**Recommendations:**")
                            if metrics.confluence_advantage > 5:
                                st.markdown("🎯 **Focus on confluence signals** - They show clear performance advantage")
                            elif metrics.confluence_advantage > 0:
                                st.markdown("📊 **Monitor confluence patterns** - Some benefit is evident")
                            else:
                                st.markdown("🔍 **Refine detection settings** - Current confluence may not be optimal")
                            
                            if metrics.best_confluence_strategies:
                                best_combo = metrics.best_confluence_strategies[0][0]
                                st.markdown(f"💡 **Top combination**: {' + '.join(best_combo)}")
                else:
                    st.info("No performance metrics available")
        
        else:
            st.warning("No confluence detected with current settings. Try adjusting the parameters in the sidebar.")
            
            with st.expander("💡 Tips for Better Confluence Detection"):
                st.markdown("""
                ### Optimization Tips:
                
                **⏰ Time Window**
                - Shorter windows (1-6 hours): Detect very tight signal alignment
                - Medium windows (12-24 hours): Catch daily strategy coordination
                - Longer windows (24-72 hours): Identify weekly trend confluence
                
                **🎯 Strength Threshold**
                - Higher threshold (0.7-1.0): Only strongest confluences
                - Medium threshold (0.4-0.7): Balanced detection
                - Lower threshold (0.1-0.4): Catch weaker signals
                
                **📊 Strategy Count**
                - 2 strategies: Basic confluence detection
                - 3+ strategies: Stronger consensus signals
                - 4+ strategies: Rare but potentially powerful events
                
                **📅 Date Range**
                - Recent data: Current market behavior
                - Longer periods: More statistical significance
                - Bear/bull markets: Different confluence patterns
                """)
    
    else:
        # Initial state - show introduction
        st.info("👆 Configure your analysis settings in the sidebar and click 'Analyze Confluence' to begin")
        
        # Educational content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 What is Signal Confluence?")
            st.markdown("""
            Signal confluence occurs when multiple trading strategies generate 
            signals at similar times, potentially indicating:
            
            - **Stronger market opportunities**
            - **Higher probability setups**
            - **Reduced false signals**
            - **Better risk-adjusted returns**
            
            This analysis helps identify when your strategies "agree" 
            and whether these periods of agreement lead to better performance.
            """)
        
        with col2:
            st.subheader("📊 Analysis Features")
            st.markdown("""
            Our confluence detection provides:
            
            **🔍 Detection**
            - Time-window based signal overlap detection
            - Configurable sensitivity settings
            - Multiple confluence types identification
            
            **📈 Analysis**
            - Performance impact measurement
            - Strategy combination rankings
            - Strength scoring system
            
            **🔴 Monitoring**
            - Real-time confluence detection
            - Customizable alerts
            - Historical tracking
            """)
        
        # Sample visualization placeholder
        st.divider()
        st.subheader("📊 Sample Analysis Output")
        st.image("https://via.placeholder.com/800x400/1E1E1E/FFFFFF?text=Confluence+Timeline+Visualization", 
                caption="Example: Interactive timeline showing when multiple strategies generated signals simultaneously")


if __name__ == "__main__":
    main()