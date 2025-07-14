"""Upload preview and feedback components."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

from ..utils import ValidationResult, ValidationSeverity, Platform


class UploadFeedback:
    """Component for displaying upload preview and validation feedback."""
    
    @staticmethod
    def show_validation_summary(validation_result: ValidationResult):
        """Display validation result summary."""
        if validation_result.is_valid:
            st.success("✅ **Validation Passed**")
        else:
            st.error("❌ **Validation Failed**")
        
        # Count issues by severity
        errors = validation_result.get_errors()
        warnings = validation_result.get_warnings()
        info_issues = [i for i in validation_result.issues if i.severity == ValidationSeverity.INFO]
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Errors", len(errors), delta=None, 
                     help="Critical issues that must be fixed")
        with col2:
            st.metric("Warnings", len(warnings), delta=None,
                     help="Issues that should be reviewed")
        with col3:
            st.metric("Info", len(info_issues), delta=None,
                     help="Informational messages")
        
        # Show detailed issues
        if errors or warnings:
            with st.expander("📋 Validation Details", expanded=len(errors) > 0):
                if errors:
                    st.subheader("🚨 Errors")
                    for i, issue in enumerate(errors[:10], 1):
                        st.error(f"{i}. {issue}")
                    if len(errors) > 10:
                        st.error(f"... and {len(errors) - 10} more errors")
                
                if warnings:
                    st.subheader("⚠️ Warnings")
                    for i, issue in enumerate(warnings[:5], 1):
                        st.warning(f"{i}. {issue}")
                    if len(warnings) > 5:
                        st.warning(f"... and {len(warnings) - 5} more warnings")
    
    @staticmethod
    def show_data_preview(df: pd.DataFrame, platform: Platform, 
                         show_charts: bool = True):
        """Display data preview with statistics."""
        st.subheader("📊 Data Preview")
        
        # Platform and basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Platform:** {platform.value}")
        with col2:
            st.info(f"**Rows:** {len(df):,}")
        with col3:
            st.info(f"**Columns:** {len(df.columns)}")
        
        # Data table
        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=400
        )
        
        if show_charts and len(df) > 0:
            # Show basic statistics
            tabs = st.tabs(["📈 Overview", "📊 Distributions", "📅 Timeline"])
            
            with tabs[0]:
                UploadFeedback._show_overview_stats(df)
            
            with tabs[1]:
                UploadFeedback._show_distributions(df)
            
            with tabs[2]:
                UploadFeedback._show_timeline(df)
    
    @staticmethod
    def _show_overview_stats(df: pd.DataFrame):
        """Show overview statistics."""
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            # Basic statistics
            st.subheader("Numeric Column Statistics")
            stats_df = df[numeric_cols].describe().T
            stats_df['missing'] = df[numeric_cols].isnull().sum()
            st.dataframe(stats_df, use_container_width=True)
        
        # Symbol distribution if available
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts().head(10)
            if not symbol_counts.empty:
                st.subheader("Top 10 Symbols")
                fig = px.bar(
                    x=symbol_counts.index,
                    y=symbol_counts.values,
                    labels={'x': 'Symbol', 'y': 'Count'},
                    title="Most Traded Symbols"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Side distribution if available
        if 'side' in df.columns:
            side_counts = df['side'].value_counts()
            if not side_counts.empty:
                st.subheader("Trade Direction Distribution")
                fig = px.pie(
                    values=side_counts.values,
                    names=side_counts.index,
                    title="Buy vs Sell Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _show_distributions(df: pd.DataFrame):
        """Show distribution charts."""
        # P&L distribution
        if 'pnl' in df.columns:
            pnl_data = df['pnl'].dropna()
            if len(pnl_data) > 0:
                st.subheader("P&L Distribution")
                
                # Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pnl_data,
                    nbinsx=50,
                    name="P&L Distribution",
                    marker_color='lightblue'
                ))
                
                # Add vertical line at zero
                fig.add_vline(x=0, line_dash="dash", line_color="red", 
                             annotation_text="Break Even")
                
                fig.update_layout(
                    title="Profit/Loss Distribution",
                    xaxis_title="P&L",
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Win/Loss metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    wins = (pnl_data > 0).sum()
                    st.metric("Winning Trades", f"{wins:,}")
                with col2:
                    losses = (pnl_data < 0).sum()
                    st.metric("Losing Trades", f"{losses:,}")
                with col3:
                    if len(pnl_data) > 0:
                        win_rate = (wins / len(pnl_data)) * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Quantity distribution
        if 'quantity' in df.columns:
            qty_data = df['quantity'].dropna()
            if len(qty_data) > 0:
                st.subheader("Position Size Distribution")
                fig = px.box(y=qty_data, title="Position Size Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _show_timeline(df: pd.DataFrame):
        """Show timeline analysis."""
        if 'trade_date' in df.columns:
            try:
                # Ensure datetime
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                
                # Date range
                date_range = df['trade_date'].max() - df['trade_date'].min()
                st.info(f"**Date Range:** {df['trade_date'].min().date()} to {df['trade_date'].max().date()} ({date_range.days} days)")
                
                # Trades over time
                daily_trades = df.groupby(df['trade_date'].dt.date).size()
                
                fig = px.line(
                    x=daily_trades.index,
                    y=daily_trades.values,
                    labels={'x': 'Date', 'y': 'Number of Trades'},
                    title="Trading Activity Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # P&L over time if available
                if 'pnl' in df.columns:
                    daily_pnl = df.groupby(df['trade_date'].dt.date)['pnl'].sum()
                    cumulative_pnl = daily_pnl.cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cumulative_pnl.index,
                        y=cumulative_pnl.values,
                        mode='lines',
                        name='Cumulative P&L',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    fig.update_layout(
                        title="Cumulative P&L Over Time",
                        xaxis_title="Date",
                        yaxis_title="Cumulative P&L",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.warning(f"Could not analyze timeline: {str(e)}")
    
    @staticmethod
    def show_upload_progress(current: int, total: int, message: str = ""):
        """Display upload progress."""
        progress = current / total if total > 0 else 0
        st.progress(progress)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(message or f"Processing... {current}/{total}")
        with col2:
            st.text(f"{progress*100:.1f}%")
    
    @staticmethod
    def show_processing_summary(metadata: Dict[str, Any], 
                              validation_result: ValidationResult,
                              success: bool = True):
        """Display processing summary after upload."""
        if success:
            st.success("✅ **Upload Successful!**")
        else:
            st.error("❌ **Upload Failed**")
        
        # Show metadata
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Platform", metadata.get('platform', 'Unknown'))
        
        with col2:
            st.metric("Rows Processed", f"{metadata.get('row_count', 0):,}")
        
        with col3:
            st.metric("Trades Created", f"{metadata.get('processed_count', 0):,}")
        
        with col4:
            file_size_mb = metadata.get('file_size_bytes', 0) / 1024 / 1024
            st.metric("File Size", f"{file_size_mb:.1f} MB")
        
        # Show validation summary if there were issues
        if validation_result.issues:
            st.markdown("---")
            UploadFeedback.show_validation_summary(validation_result)


def create_upload_preview(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Create a comprehensive preview of uploaded file.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Dictionary with preview data
    """
    from ..utils import CSVProcessor
    
    processor = CSVProcessor()
    
    try:
        # Analyze format
        analysis = processor.analyze_format(file_content, filename)
        
        # Get preview
        preview_df, platform, validation_result = processor.preview(file_content, filename)
        
        return {
            'success': True,
            'analysis': analysis,
            'preview_df': preview_df,
            'platform': platform,
            'validation_result': validation_result
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'analysis': {},
            'preview_df': pd.DataFrame(),
            'platform': Platform.UNKNOWN,
            'validation_result': ValidationResult(is_valid=False)
        }