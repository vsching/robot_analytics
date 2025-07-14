"""
Export Dashboard Component

User interface for managing exports and generating reports.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from ..exports.export_manager import ExportManager, ExportConfig, ReportTemplate
from ..exports.pdf_generator import PDFReportGenerator
from ..services.strategy_manager import StrategyManager
from ..analytics import AnalyticsEngine, ConfluenceAnalyzer, SignalOverlap, ConfluenceMetrics
from .strategy_selector import StrategySelector


logger = logging.getLogger(__name__)


class ExportDashboard:
    """Dashboard for export and reporting functionality."""
    
    def __init__(self, 
                 strategy_manager: StrategyManager,
                 analytics_engine: AnalyticsEngine,
                 export_dir: str = "exports"):
        """
        Initialize export dashboard.
        
        Args:
            strategy_manager: Strategy manager instance
            analytics_engine: Analytics engine instance
            export_dir: Directory for exports
        """
        self.strategy_manager = strategy_manager
        self.analytics_engine = analytics_engine
        self.export_manager = ExportManager(strategy_manager, analytics_engine, export_dir)
        
        # Initialize PDF generator if available
        try:
            self.pdf_generator = PDFReportGenerator(strategy_manager, analytics_engine, f"{export_dir}/pdf")
            self.pdf_available = True
        except ImportError:
            self.pdf_generator = None
            self.pdf_available = False
        
        self.confluence_analyzer = ConfluenceAnalyzer()
    
    def render_export_options(self) -> Dict[str, Any]:
        """
        Render export options sidebar.
        
        Returns:
            Dictionary with export settings
        """
        st.sidebar.header("📥 Export Settings")
        
        # Export format selection
        available_formats = self.export_manager.get_available_formats()
        export_format = st.sidebar.selectbox(
            "Export Format",
            options=available_formats,
            format_func=lambda x: {
                'csv': '📄 CSV (Comma Separated)',
                'excel': '📊 Excel (Multi-sheet)',
                'pdf': '📋 PDF Report'
            }.get(x, x.upper())
        )
        
        # Strategy selection
        st.sidebar.subheader("🎯 Strategy Selection")
        
        strategies = self.strategy_manager.get_all_strategies()
        strategy_options = {"All Strategies": None}
        strategy_options.update({f"{s.name} (ID: {s.id})": s.id for s in strategies})
        
        selected_strategy_key = st.sidebar.selectbox(
            "Select Strategy",
            options=list(strategy_options.keys())
        )
        selected_strategy_id = strategy_options[selected_strategy_key]
        
        # Data selection
        st.sidebar.subheader("📋 Data Selection")
        
        include_trades = st.sidebar.checkbox("Include Trades", value=True)
        include_metrics = st.sidebar.checkbox("Include Performance Metrics", value=True)
        include_monthly = st.sidebar.checkbox("Include Monthly Breakdown", value=True)
        include_confluence = st.sidebar.checkbox("Include Confluence Analysis", value=False)
        
        # Date range filter
        st.sidebar.subheader("📅 Date Range")
        
        # Get available date range
        if selected_strategy_id:
            trades = self.analytics_engine.get_trades_for_strategy(selected_strategy_id)
        else:
            trades = []
            for strategy in strategies:
                strategy_trades = self.analytics_engine.get_trades_for_strategy(strategy.id)
                trades.extend(strategy_trades)
        
        if trades:
            min_date = min(t.trade_date for t in trades)
            max_date = max(t.trade_date for t in trades)
            
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
        
        # Format options
        st.sidebar.subheader("⚙️ Format Options")
        
        if export_format == 'csv':
            delimiter = st.sidebar.selectbox("Delimiter", [",", ";", "\t"], index=0)
            decimal_places = st.sidebar.slider("Decimal Places", 0, 6, 2)
            include_headers = st.sidebar.checkbox("Include Headers", value=True)
        else:
            delimiter = ","
            decimal_places = 2
            include_headers = True
        
        # PDF-specific options
        include_charts = False
        report_title = ""
        report_subtitle = ""
        
        if export_format == 'pdf' and self.pdf_available:
            st.sidebar.subheader("📋 PDF Options")
            include_charts = st.sidebar.checkbox("Include Charts", value=True)
            report_title = st.sidebar.text_input("Report Title", value="Trading Strategy Analysis Report")
            report_subtitle = st.sidebar.text_input("Report Subtitle", value="")
        
        return {
            'format': export_format,
            'strategy_id': selected_strategy_id,
            'include_trades': include_trades,
            'include_metrics': include_metrics,
            'include_monthly': include_monthly,
            'include_confluence': include_confluence,
            'start_date': start_date,
            'end_date': end_date,
            'delimiter': delimiter,
            'decimal_places': decimal_places,
            'include_headers': include_headers,
            'include_charts': include_charts,
            'report_title': report_title,
            'report_subtitle': report_subtitle
        }
    
    def render_export_interface(self, settings: Dict[str, Any]):
        """
        Render main export interface.
        
        Args:
            settings: Export settings from render_export_options
        """
        st.subheader("📥 Export & Reporting")
        
        # Export preview
        self._render_export_preview(settings)
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Generate Export", type="primary"):
                self._perform_export(settings)
        
        with col2:
            if st.button("👀 Preview Data"):
                self._show_data_preview(settings)
        
        with col3:
            if st.button("📋 Schedule Export"):
                self._show_schedule_interface()
        
        # Export history
        st.divider()
        self._render_export_history()
    
    def _render_export_preview(self, settings: Dict[str, Any]):
        """Render export preview information."""
        st.markdown("### 📋 Export Preview")
        
        # Get data counts
        if settings['strategy_id']:
            strategy = self.strategy_manager.get_strategy(settings['strategy_id'])
            strategy_name = strategy.name if strategy else f"Strategy {settings['strategy_id']}"
            trades = self.analytics_engine.get_trades_for_strategy(settings['strategy_id'])
        else:
            strategy_name = "All Strategies"
            trades = []
            strategies = self.strategy_manager.get_all_strategies()
            for strategy in strategies:
                strategy_trades = self.analytics_engine.get_trades_for_strategy(strategy.id)
                trades.extend(strategy_trades)
        
        # Filter by date range
        filtered_trades = [
            t for t in trades 
            if settings['start_date'] <= t.trade_date <= settings['end_date']
        ]
        
        # Preview information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Strategy", strategy_name)
        
        with col2:
            st.metric("Total Trades", len(filtered_trades))
        
        with col3:
            date_range = (settings['end_date'] - settings['start_date']).days + 1
            st.metric("Date Range", f"{date_range} days")
        
        with col4:
            st.metric("Format", settings['format'].upper())
        
        # Content preview
        content_items = []
        if settings['include_trades']:
            content_items.append("✅ Trade Details")
        if settings['include_metrics']:
            content_items.append("✅ Performance Metrics")
        if settings['include_monthly']:
            content_items.append("✅ Monthly Breakdown")
        if settings['include_confluence']:
            content_items.append("✅ Confluence Analysis")
        if settings.get('include_charts', False):
            content_items.append("✅ Performance Charts")
        
        if content_items:
            st.markdown("**Content Included:**")
            for item in content_items:
                st.markdown(f"- {item}")
        else:
            st.warning("No content selected for export")
    
    def _perform_export(self, settings: Dict[str, Any]):
        """Perform the actual export operation."""
        if not any([
            settings['include_trades'],
            settings['include_metrics'],
            settings['include_monthly'],
            settings['include_confluence']
        ]):
            st.error("Please select at least one type of data to export")
            return
        
        try:
            with st.spinner(f"Generating {settings['format'].upper()} export..."):
                
                # Create export config
                config = ExportConfig(
                    include_trades=settings['include_trades'],
                    include_metrics=settings['include_metrics'],
                    include_monthly_breakdown=settings['include_monthly'],
                    include_confluence=settings['include_confluence'],
                    delimiter=settings['delimiter'],
                    decimal_places=settings['decimal_places'],
                    include_headers=settings['include_headers']
                )
                
                if settings['format'] == 'csv':
                    # Export CSV
                    if settings['include_trades']:
                        trades_path = self.export_manager.export_trades_to_csv(
                            strategy_id=settings['strategy_id'],
                            config=config
                        )
                        st.success(f"✅ Trades exported to: {Path(trades_path).name}")
                        
                        # Download button
                        with open(trades_path, 'rb') as file:
                            st.download_button(
                                label="📥 Download Trades CSV",
                                data=file.read(),
                                file_name=Path(trades_path).name,
                                mime="text/csv"
                            )
                    
                    if settings['include_metrics']:
                        metrics_path = self.export_manager.export_metrics_to_csv(
                            strategy_id=settings['strategy_id'],
                            config=config
                        )
                        st.success(f"✅ Metrics exported to: {Path(metrics_path).name}")
                        
                        # Download button
                        with open(metrics_path, 'rb') as file:
                            st.download_button(
                                label="📥 Download Metrics CSV",
                                data=file.read(),
                                file_name=Path(metrics_path).name,
                                mime="text/csv"
                            )
                
                elif settings['format'] == 'excel':
                    # Export Excel
                    excel_path = self.export_manager.export_to_excel(
                        strategy_id=settings['strategy_id'],
                        config=config,
                        include_charts=settings.get('include_charts', False)
                    )
                    st.success(f"✅ Excel report exported to: {Path(excel_path).name}")
                    
                    # Download button
                    with open(excel_path, 'rb') as file:
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=file.read(),
                            file_name=Path(excel_path).name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                elif settings['format'] == 'pdf' and self.pdf_available:
                    # Export PDF
                    template = ReportTemplate(
                        title=settings['report_title'],
                        subtitle=settings['report_subtitle'],
                        include_charts=settings.get('include_charts', False),
                        include_confluence_analysis=settings['include_confluence']
                    )
                    
                    # Get confluence data if needed
                    confluence_data = None
                    if settings['include_confluence']:
                        confluence_data = self._get_confluence_data(settings['strategy_id'])
                    
                    if settings['strategy_id']:
                        # Single strategy report
                        pdf_path = self.pdf_generator.generate_strategy_report(
                            strategy_id=settings['strategy_id'],
                            template=template,
                            include_confluence=settings['include_confluence'],
                            confluence_data=confluence_data
                        )
                    else:
                        # Multi-strategy report
                        strategies = self.strategy_manager.get_all_strategies()
                        strategy_ids = [s.id for s in strategies]
                        pdf_path = self.pdf_generator.generate_multi_strategy_report(
                            strategy_ids=strategy_ids,
                            template=template
                        )
                    
                    st.success(f"✅ PDF report generated: {Path(pdf_path).name}")
                    
                    # Download button
                    with open(pdf_path, 'rb') as file:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=file.read(),
                            file_name=Path(pdf_path).name,
                            mime="application/pdf"
                        )
                
                else:
                    st.error(f"Export format '{settings['format']}' is not available")
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            logger.error(f"Export error: {str(e)}")
    
    def _show_data_preview(self, settings: Dict[str, Any]):
        """Show preview of data to be exported."""
        st.markdown("### 👀 Data Preview")
        
        # Get data
        if settings['strategy_id']:
            trades = self.analytics_engine.get_trades_for_strategy(settings['strategy_id'])
            strategy = self.strategy_manager.get_strategy(settings['strategy_id'])
            strategy_name = strategy.name if strategy else f"Strategy {settings['strategy_id']}"
        else:
            trades = []
            strategies = self.strategy_manager.get_all_strategies()
            for strategy in strategies:
                strategy_trades = self.analytics_engine.get_trades_for_strategy(strategy.id)
                trades.extend(strategy_trades)
            strategy_name = "All Strategies"
        
        # Filter by date range
        filtered_trades = [
            t for t in trades 
            if settings['start_date'] <= t.trade_date <= settings['end_date']
        ]
        
        if not filtered_trades:
            st.warning("No trades found for the selected criteria")
            return
        
        # Show previews based on selected content
        if settings['include_trades']:
            st.markdown("#### 📊 Trades Preview")
            
            # Create trades DataFrame
            trade_data = []
            for trade in filtered_trades[:10]:  # Show first 10
                trade_data.append({
                    'Date': trade.trade_date,
                    'Symbol': trade.symbol or '',
                    'Side': trade.side or '',
                    'Entry': float(trade.entry_price or 0),
                    'Exit': float(trade.exit_price or 0),
                    'Quantity': float(trade.quantity or 0),
                    'P&L': float(trade.pnl or 0)
                })
            
            if trade_data:
                df = pd.DataFrame(trade_data)
                st.dataframe(df, use_container_width=True)
                
                if len(filtered_trades) > 10:
                    st.info(f"Showing first 10 trades. Total trades to export: {len(filtered_trades)}")
        
        if settings['include_metrics']:
            st.markdown("#### 📈 Metrics Preview")
            
            if settings['strategy_id']:
                metrics = self.analytics_engine.calculate_metrics_for_strategy(settings['strategy_id'])
                if metrics:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total P&L", f"${float(metrics.pnl_summary.total_pnl):,.2f}")
                        st.metric("Win Rate", f"{metrics.trade_statistics.win_rate:.1f}%")
                    
                    with col2:
                        st.metric("Total Trades", metrics.trade_statistics.total_trades)
                        st.metric("Profit Factor", f"{metrics.trade_statistics.profit_factor:.2f}")
                    
                    with col3:
                        if metrics.advanced_statistics:
                            st.metric("Sharpe Ratio", f"{metrics.advanced_statistics.sharpe_ratio:.2f}" if metrics.advanced_statistics.sharpe_ratio else "N/A")
                            st.metric("Max Drawdown", f"{metrics.advanced_statistics.max_drawdown:.1%}")
            else:
                st.info("Metrics preview available for individual strategies only")
    
    def _show_schedule_interface(self):
        """Show scheduling interface (placeholder for future implementation)."""
        st.markdown("### 📅 Schedule Export")
        st.info("Scheduled export functionality will be available in a future update.")
        
        with st.expander("📋 Planned Features"):
            st.markdown("""
            **Upcoming scheduling features:**
            
            - **Daily Reports**: Automatically generate daily performance summaries
            - **Weekly Summaries**: Weekly strategy performance roundups
            - **Monthly Reports**: Comprehensive monthly analysis reports
            - **Email Delivery**: Send reports directly to email addresses
            - **Custom Schedules**: Flexible scheduling with cron-like expressions
            - **Report Templates**: Saved configurations for repeated exports
            """)
    
    def _render_export_history(self):
        """Render export history section."""
        st.markdown("### 📚 Export History")
        
        try:
            history = self.export_manager.get_export_history()
            
            if not history:
                st.info("No previous exports found")
                return
            
            # Create history DataFrame
            history_data = []
            for item in history[:20]:  # Show last 20
                history_data.append({
                    'Filename': item['filename'],
                    'Format': item['format'].upper(),
                    'Size': f"{item['size_bytes'] / 1024:.1f} KB",
                    'Created': item['created'].strftime('%Y-%m-%d %H:%M'),
                    'Path': item['path']
                })
            
            df = pd.DataFrame(history_data)
            
            # Display with download buttons
            for idx, row in df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 1])
                
                with col1:
                    st.text(row['Filename'])
                
                with col2:
                    st.text(row['Format'])
                
                with col3:
                    st.text(row['Size'])
                
                with col4:
                    st.text(row['Created'])
                
                with col5:
                    # Download button for each file
                    file_path = row['Path']
                    if Path(file_path).exists():
                        try:
                            with open(file_path, 'rb') as file:
                                mime_type = {
                                    'csv': 'text/csv',
                                    'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                    'pdf': 'application/pdf'
                                }.get(row['Format'].lower(), 'application/octet-stream')
                                
                                st.download_button(
                                    label="⬇️",
                                    data=file.read(),
                                    file_name=row['Filename'],
                                    mime=mime_type,
                                    key=f"download_{idx}"
                                )
                        except Exception as e:
                            st.text("❌")
                    else:
                        st.text("❌")
            
            # Cleanup options
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧹 Clean Old Exports"):
                    self.export_manager.cleanup_old_exports(days_old=30)
                    st.success("Cleaned exports older than 30 days")
                    st.experimental_rerun()
            
            with col2:
                st.info(f"Showing {len(history_data)} most recent exports")
        
        except Exception as e:
            st.error(f"Error loading export history: {str(e)}")
    
    def _get_confluence_data(self, strategy_id: Optional[int]) -> Optional[Tuple[List[SignalOverlap], ConfluenceMetrics]]:
        """Get confluence data for the report."""
        try:
            # Get strategies and trades
            if strategy_id:
                # For single strategy, get all strategies for confluence analysis
                strategies = self.strategy_manager.get_all_strategies()
                if len(strategies) < 2:
                    return None
            else:
                strategies = self.strategy_manager.get_all_strategies()
            
            strategy_names = {s.id: s.name for s in strategies}
            strategies_trades = {}
            all_trades = []
            
            for strategy in strategies:
                trades = self.analytics_engine.get_trades_for_strategy(strategy.id)
                if trades:
                    strategies_trades[strategy.id] = trades
                    all_trades.extend(trades)
            
            if len(strategies_trades) < 2:
                return None
            
            # Find overlaps
            overlaps = self.confluence_analyzer.find_signal_overlaps(strategies_trades, strategy_names)
            
            if not overlaps:
                return None
            
            # Calculate metrics
            metrics = self.confluence_analyzer.analyze_confluence_performance(overlaps, all_trades)
            
            return overlaps, metrics
        
        except Exception as e:
            logger.error(f"Error getting confluence data: {str(e)}")
            return None