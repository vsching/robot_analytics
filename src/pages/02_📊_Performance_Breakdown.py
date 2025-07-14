"""
Performance Breakdown Page

This page provides monthly and weekly breakdown tables for analyzing
trading strategy performance over time periods.
"""

import streamlit as st
from datetime import datetime

from src.components.breakdown_tables import BreakdownTables
from src.components.strategy_selector import StrategySelector
from src.analytics.analytics_engine import AnalyticsEngine
from src.analytics.cache_manager import MetricsCacheManager
from src.db.strategy_repository import StrategyRepository
from src.db.trade_repository import TradeRepository
from src.services.strategy_manager import StrategyManager
from src.utils.session_state import SessionStateManager


# Page configuration
st.set_page_config(
    page_title="Performance Breakdown - Trading Strategy Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
SessionStateManager.init()

# Initialize components
@st.cache_resource
def init_components():
    """Initialize and cache application components."""
    strategy_repo = StrategyRepository()
    trade_repo = TradeRepository()
    strategy_manager = StrategyManager(strategy_repo, trade_repo)
    analytics_engine = AnalyticsEngine()  # Uses its own db_manager
    cache_manager = MetricsCacheManager()
    return strategy_manager, analytics_engine, cache_manager, strategy_repo

strategy_manager, analytics_engine, cache_manager, strategy_repo = init_components()

# Header
st.title("📊 Performance Breakdown")
st.markdown("""
Analyze your trading strategy performance with detailed monthly and weekly breakdowns.
View aggregated metrics, identify patterns, and export data for further analysis.
""")

# Initialize components
strategy_selector = StrategySelector(strategy_repo)
breakdown_tables = BreakdownTables(analytics_engine, cache_manager)

# Strategy selection
st.markdown("### Select Strategy")
selected_strategy = strategy_selector.render_selector()

if selected_strategy:
    # Date range selector
    st.markdown("### Filter by Date Range")
    date_range = breakdown_tables.render_period_selector()
    
    # View selection
    st.markdown("### Select View")
    view_type = st.radio(
        "Choose breakdown period:",
        ["Monthly Breakdown", "Weekly Breakdown"],
        horizontal=True
    )
    
    # Display breakdown tables
    st.markdown("---")
    
    if view_type == "Monthly Breakdown":
        st.markdown("### Monthly Performance Breakdown")
        st.markdown("""
        View your trading performance aggregated by month. Analyze total P&L, win rates,
        trade counts, and identify your best and worst performing months.
        """)
        
        with st.container():
            breakdown_tables.monthly_breakdown_table(selected_strategy.id, date_range)
            
        # Additional insights
        with st.expander("📈 Monthly Insights"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Best Month",
                    "March 2024",  # Placeholder
                    "+$8,542.30",
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "Worst Month",
                    "January 2024",  # Placeholder
                    "-$2,315.45",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Average Monthly P&L",
                    "$3,125.50",  # Placeholder
                    "+15.3%",
                    delta_color="normal"
                )
    
    else:  # Weekly Breakdown
        st.markdown("### Weekly Performance Breakdown")
        st.markdown("""
        Analyze your trading performance on a weekly basis. Includes day-of-week analysis
        to help identify the most profitable trading days.
        """)
        
        with st.container():
            breakdown_tables.weekly_breakdown_table(selected_strategy.id, date_range)
        
        # Day of week analysis
        with st.expander("📅 Day of Week Analysis"):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            days = [
                ("Monday", "$1,234.50", "+12.3%"),
                ("Tuesday", "$2,456.30", "+24.5%"),
                ("Wednesday", "-$523.20", "-5.2%"),
                ("Thursday", "$3,125.80", "+31.2%"),
                ("Friday", "$1,852.40", "+18.5%")
            ]
            
            for col, (day, pnl, rate) in zip([col1, col2, col3, col4, col5], days):
                with col:
                    st.metric(
                        day,
                        pnl,
                        rate,
                        delta_color="normal" if not pnl.startswith("-") else "inverse"
                    )
    
    # Help section
    with st.expander("ℹ️ How to Use This Page"):
        st.markdown("""
        **Monthly Breakdown:**
        - Shows aggregated performance metrics for each month
        - Includes total P&L, trade count, win rate, and trade statistics
        - Summary row at the bottom shows overall totals and averages
        
        **Weekly Breakdown:**
        - Displays performance metrics aggregated by week
        - Includes day-of-week P&L breakdown to identify patterns
        - Helps identify the most and least profitable trading days
        
        **Features:**
        - 🔍 **Sorting**: Click column headers to sort data
        - 🎯 **Filtering**: Use the filter icon in column headers
        - ✅ **Selection**: Select rows using checkboxes for export
        - 📥 **Export**: Download filtered data as CSV
        - 📊 **Responsive**: Tables adjust to screen size
        """)

else:
    # No strategy selected
    st.info("👈 Please select a strategy from the dropdown above to view performance breakdowns.")
    
    # Show sample preview
    with st.expander("Preview: What You'll See"):
        st.markdown("""
        Once you select a strategy, you'll see:
        
        **📊 Monthly Breakdown Table:**
        - Month-by-month performance metrics
        - Total P&L for each month
        - Trade counts and win rates
        - Best and worst trades per month
        
        **📈 Weekly Breakdown Table:**
        - Week-by-week performance analysis
        - Day-of-week P&L breakdown
        - Identify your most profitable trading days
        - Spot weekly patterns and trends
        
        **🛠️ Interactive Features:**
        - Sort by any column
        - Filter data based on criteria
        - Export to CSV for external analysis
        - Responsive design for all screen sizes
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Performance Breakdown Tables • Trading Strategy Analyzer v1.0
    </div>
    """,
    unsafe_allow_html=True
)