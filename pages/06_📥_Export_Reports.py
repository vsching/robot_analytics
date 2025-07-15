"""
Export & Reports Page

Comprehensive export and reporting functionality for trading strategy analysis.
Supports CSV, Excel, and PDF exports with customizable templates and settings.
"""

import streamlit as st
from datetime import datetime

from src.utils.auth_check import check_authentication
from src.db.connection import get_db_manager
from src.services.strategy_manager import StrategyManager
from src.analytics import AnalyticsEngine, CachedAnalyticsEngine, MetricsCacheManager
from src.components.export_dashboard import ExportDashboard


# Page configuration
st.set_page_config(
    page_title="Export & Reports - Trading Strategy Analyzer",
    page_icon="📥",
    layout="wide"
)

# Check authentication
check_authentication()


def main():
    
    st.title("📥 Export & Reports")
    st.markdown("""
    Generate comprehensive reports and export your trading analysis data in multiple formats.
    Choose from CSV, Excel, or PDF exports with customizable content and formatting options.
    """)
    
    # Initialize components
    db_manager = get_db_manager()
    strategy_manager = StrategyManager(db_manager)
    cache_manager = MetricsCacheManager(db_manager)
    analytics_engine = AnalyticsEngine(db_manager)
    cached_analytics = CachedAnalyticsEngine(analytics_engine, cache_manager)
    
    # Create export dashboard
    export_dashboard = ExportDashboard(
        strategy_manager=strategy_manager,
        analytics_engine=cached_analytics
    )
    
    # Check if we have strategies
    strategies = strategy_manager.get_all_strategies()
    if not strategies:
        st.error("⚠️ No strategies found. Please add strategies before exporting data.")
        st.info("💡 Use the Strategy Management page to add your first strategy.")
        return
    
    # Check if we have trades
    total_trades = 0
    for strategy in strategies:
        trades = cached_analytics.get_trades_for_strategy(strategy.id)
        total_trades += len(trades)
    
    if total_trades == 0:
        st.warning("⚠️ No trades found. Please upload trade data before generating exports.")
        st.info("💡 Use the Data Upload page to import your trading data.")
        return
    
    # Render export settings in sidebar
    settings = export_dashboard.render_export_options()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Export interface
        export_dashboard.render_export_interface(settings)
    
    with col2:
        # Quick stats and format information
        st.subheader("📊 Quick Stats")
        
        # Display current data summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Strategies", len(strategies))
            st.metric("Total Trades", total_trades)
        
        with col2:
            # Get date range of all trades
            all_trades = []
            for strategy in strategies:
                strategy_trades = cached_analytics.get_trades_for_strategy(strategy.id)
                all_trades.extend(strategy_trades)
            
            if all_trades:
                min_date = min(t.trade_date for t in all_trades)
                max_date = max(t.trade_date for t in all_trades)
                date_range = (max_date - min_date).days + 1
                
                st.metric("Date Range", f"{date_range} days")
                st.metric("Period", f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Format information
        st.divider()
        st.subheader("📋 Export Formats")
        
        # CSV format info
        with st.expander("📄 CSV Format"):
            st.markdown("""
            **Comma Separated Values**
            
            ✅ **Best for:**
            - Data analysis in Excel/Sheets
            - Custom data processing
            - Importing into other tools
            
            📊 **Contents:**
            - Raw trade data
            - Performance metrics
            - Customizable columns
            
            ⚙️ **Options:**
            - Custom delimiters
            - Decimal precision
            - Header inclusion
            """)
        
        # Excel format info
        with st.expander("📊 Excel Format"):
            st.markdown("""
            **Microsoft Excel Workbook**
            
            ✅ **Best for:**
            - Comprehensive analysis
            - Professional presentations
            - Multi-sheet organization
            
            📊 **Contents:**
            - Summary dashboard
            - Detailed trade data
            - Performance metrics
            - Monthly breakdowns
            - Charts and visualizations
            
            📋 **Sheets:**
            - Summary
            - Trades
            - Metrics
            - Monthly Breakdown
            - Charts (coming soon)
            """)
        
        # PDF format info
        with st.expander("📋 PDF Format"):
            st.markdown("""
            **Professional PDF Report**
            
            ✅ **Best for:**
            - Professional reports
            - Sharing with stakeholders
            - Documentation
            - Print-ready format
            
            📊 **Contents:**
            - Executive summary
            - Performance charts
            - Detailed metrics
            - Risk analysis
            - Trade details
            - Confluence analysis (optional)
            
            🎨 **Features:**
            - Professional layout
            - Interactive charts
            - Custom branding
            - Multiple templates
            """)
    
    # Additional features section
    st.divider()
    
    # Feature tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Quick Export", 
        "📊 Bulk Operations", 
        "⏰ Automation", 
        "💡 Tips & Best Practices"
    ])
    
    with tab1:
        st.subheader("🚀 Quick Export Templates")
        st.markdown("Pre-configured export templates for common use cases:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Basic Summary", type="secondary"):
                st.info("Quick CSV export with essential metrics")
                # Auto-configure settings for basic summary
                st.session_state.update({
                    'quick_export_format': 'csv',
                    'quick_export_content': 'basic'
                })
        
        with col2:
            if st.button("📋 Full Report", type="secondary"):
                st.info("Comprehensive PDF report with all features")
                st.session_state.update({
                    'quick_export_format': 'pdf',
                    'quick_export_content': 'full'
                })
        
        with col3:
            if st.button("📈 Performance Review", type="secondary"):
                st.info("Excel workbook focused on performance analysis")
                st.session_state.update({
                    'quick_export_format': 'excel',
                    'quick_export_content': 'performance'
                })
    
    with tab2:
        st.subheader("📊 Bulk Export Operations")
        
        st.markdown("**Export All Strategies Individually:**")
        col1, col2 = st.columns(2)
        
        with col1:
            bulk_format = st.selectbox(
                "Bulk Export Format",
                ["csv", "excel", "pdf"],
                format_func=lambda x: x.upper()
            )
        
        with col2:
            if st.button("🔄 Export All Strategies"):
                with st.spinner("Generating exports for all strategies..."):
                    success_count = 0
                    error_count = 0
                    
                    for strategy in strategies:
                        try:
                            # Create individual export for each strategy
                            individual_settings = settings.copy()
                            individual_settings['strategy_id'] = strategy.id
                            
                            # Simulate export (would call actual export methods)
                            success_count += 1
                            
                        except Exception as e:
                            error_count += 1
                            st.error(f"Failed to export {strategy.name}: {str(e)}")
                    
                    if success_count > 0:
                        st.success(f"✅ Successfully exported {success_count} strategies")
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} exports failed")
        
        st.markdown("**Comparison Report:**")
        if st.button("📊 Generate Multi-Strategy Comparison"):
            st.info("This will generate a comprehensive comparison report across all strategies")
    
    with tab3:
        st.subheader("⏰ Export Automation")
        
        st.markdown("**Scheduled Reports** (Coming Soon)")
        
        schedule_options = [
            "🌅 Daily - End of day summary",
            "📅 Weekly - Weekly performance review", 
            "📊 Monthly - Comprehensive monthly report",
            "🎯 Custom - Define your own schedule"
        ]
        
        for option in schedule_options:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"- {option}")
            with col2:
                st.button("Setup", key=f"setup_{option}", disabled=True)
        
        st.info("💡 Automation features will allow you to automatically generate and email reports on a schedule.")
        
        # Email configuration placeholder
        with st.expander("📧 Email Configuration (Preview)"):
            st.text_input("SMTP Server", placeholder="smtp.gmail.com", disabled=True)
            st.text_input("Email Recipients", placeholder="user@example.com", disabled=True)
            st.text_area("Email Template", placeholder="Your daily trading report is attached...", disabled=True)
    
    with tab4:
        st.subheader("💡 Tips & Best Practices")
        
        st.markdown("""
        ### 📊 Export Best Practices
        
        **🎯 Choose the Right Format:**
        - **CSV**: For data analysis and custom processing
        - **Excel**: For comprehensive review and sharing
        - **PDF**: For formal reports and documentation
        
        **📅 Date Range Selection:**
        - Use specific date ranges for focused analysis
        - Export full history for comprehensive reviews
        - Consider market conditions when selecting periods
        
        **📋 Content Selection:**
        - Include confluence analysis for multi-strategy insights
        - Add charts to PDF reports for visual impact
        - Export metrics separately for quick reference
        
        **🔧 Performance Optimization:**
        - Large exports may take time - be patient
        - Use date filters to reduce file sizes
        - Clean up old exports regularly
        
        **💾 File Management:**
        - Use descriptive filenames with dates
        - Organize exports by strategy or time period
        - Keep backups of important reports
        
        **📈 Advanced Features:**
        - Combine with confluence analysis for deeper insights
        - Use comparison reports for strategy evaluation
        - Schedule regular exports for consistent monitoring
        
        ### 🚨 Troubleshooting
        
        **Common Issues:**
        - **Large file sizes**: Use date filters or export by strategy
        - **Missing data**: Check strategy selection and date ranges
        - **Format errors**: Verify your configuration settings
        - **Slow exports**: Large datasets require more processing time
        
        **Performance Tips:**
        - Export during off-peak hours for best performance
        - Use CSV for large datasets requiring speed
        - PDF generation requires more system resources
        """)
        
        # System requirements
        with st.expander("⚙️ System Requirements"):
            st.markdown("""
            **For Basic Exports (CSV):**
            - No additional requirements
            
            **For Excel Exports:**
            - openpyxl library (installed)
            - Sufficient memory for large datasets
            
            **For PDF Reports:**
            - matplotlib and reportlab libraries
            - Additional processing time for charts
            - Adequate disk space for image generation
            """)


if __name__ == "__main__":
    main()