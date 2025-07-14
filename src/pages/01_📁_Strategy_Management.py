"""Strategy Management page for uploading and managing trading strategies."""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
import io
import time
from datetime import datetime

from src.utils import CSVProcessor, ValidationSeverity
from src.db import StrategyRepository, TradeRepository, get_db_manager
from src.models import Strategy
from src.components.file_upload import FileUploadComponent
from src.components.strategy_selector import StrategySelector
from config import config


# Page configuration
st.set_page_config(
    page_title="Strategy Management",
    page_icon="📁",
    layout="wide"
)

# Initialize repositories
strategy_repo = StrategyRepository()
trade_repo = TradeRepository()
csv_processor = CSVProcessor()


def main():
    st.title("📁 Strategy Management")
    st.markdown("Upload CSV files, manage trading strategies, and organize your trading data.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload New Data", "📊 Manage Strategies", "ℹ️ Supported Formats"])
    
    with tab1:
        upload_tab()
    
    with tab2:
        manage_tab()
    
    with tab3:
        formats_tab()


def upload_tab():
    """Handle file upload and processing."""
    st.header("Upload Trading Data")
    
    # Strategy selection or creation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        existing_strategies = strategy_repo.get_active_strategies()
        strategy_names = ["Create New Strategy"] + [s.name for s in existing_strategies]
        
        selected_strategy = st.selectbox(
            "Select Strategy",
            strategy_names,
            help="Choose an existing strategy or create a new one"
        )
    
    with col2:
        if selected_strategy != "Create New Strategy":
            upload_action = st.radio(
                "Upload Action",
                ["Replace", "Append"],
                help="Replace: Delete existing trades and upload new ones\nAppend: Add new trades to existing data"
            )
        else:
            upload_action = "Replace"
    
    # New strategy creation
    if selected_strategy == "Create New Strategy":
        col1, col2 = st.columns([2, 1])
        with col1:
            new_strategy_name = st.text_input(
                "Strategy Name",
                placeholder="e.g., Momentum Strategy 2024",
                help="Choose a unique name for your strategy"
            )
        with col2:
            new_strategy_desc = st.text_area(
                "Description (Optional)",
                placeholder="Brief description of the strategy",
                height=68
            )
    
    # File upload component
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv', 'xlsx', 'xls'],
        help=f"Maximum file size: {config.MAX_UPLOAD_SIZE_MB}MB",
        key="csv_uploader"
    )
    
    if uploaded_file is not None:
        process_upload(
            uploaded_file,
            selected_strategy,
            upload_action,
            new_strategy_name if selected_strategy == "Create New Strategy" else None,
            new_strategy_desc if selected_strategy == "Create New Strategy" else None
        )


def process_upload(uploaded_file, selected_strategy: str, upload_action: str,
                  new_strategy_name: Optional[str] = None,
                  new_strategy_desc: Optional[str] = None):
    """Process the uploaded file."""
    
    # Validate inputs
    if selected_strategy == "Create New Strategy":
        if not new_strategy_name or not new_strategy_name.strip():
            st.error("Please enter a strategy name")
            return
        
        # Check if strategy name already exists
        if strategy_repo.get_by_name(new_strategy_name.strip()):
            st.error(f"Strategy '{new_strategy_name}' already exists. Please choose a different name.")
            return
    
    # Read file content
    file_content = uploaded_file.read()
    filename = uploaded_file.name
    
    # Show file info
    st.info(f"""
    **File:** {filename}  
    **Size:** {len(file_content) / 1024:.1f} KB  
    **Type:** {uploaded_file.type}
    """)
    
    # Preview section
    with st.expander("📋 Preview Data", expanded=True):
        preview_data(file_content, filename)
    
    # Process button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🚀 Process File", type="primary"):
            process_file(
                file_content,
                filename,
                selected_strategy,
                upload_action,
                new_strategy_name,
                new_strategy_desc
            )
    
    with col2:
        if st.button("❌ Cancel"):
            st.rerun()


def preview_data(file_content: bytes, filename: str):
    """Preview the uploaded CSV data."""
    try:
        # Get preview
        preview_df, platform, validation_result = csv_processor.preview(file_content, filename, rows=10)
        
        # Show detected format
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Format", platform.value)
        with col2:
            st.metric("Validation Status", 
                     "✅ Valid" if validation_result.is_valid else "❌ Invalid")
        
        # Show validation issues
        if validation_result.issues:
            with st.container():
                st.subheader("Validation Issues")
                
                # Group by severity
                errors = validation_result.get_errors()
                warnings = validation_result.get_warnings()
                
                if errors:
                    st.error(f"**{len(errors)} Error(s) Found:**")
                    for issue in errors[:5]:  # Show first 5
                        st.write(f"• {issue}")
                    if len(errors) > 5:
                        st.write(f"... and {len(errors) - 5} more errors")
                
                if warnings:
                    st.warning(f"**{len(warnings)} Warning(s) Found:**")
                    for issue in warnings[:3]:  # Show first 3
                        st.write(f"• {issue}")
                    if len(warnings) > 3:
                        st.write(f"... and {len(warnings) - 3} more warnings")
        
        # Show preview data
        if not preview_df.empty:
            st.subheader("Data Preview (First 10 rows)")
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.warning("No data could be previewed")
    
    except Exception as e:
        st.error(f"Failed to preview file: {str(e)}")


def process_file(file_content: bytes, filename: str, selected_strategy: str,
                upload_action: str, new_strategy_name: Optional[str],
                new_strategy_desc: Optional[str]):
    """Process the uploaded file and save to database."""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Create or get strategy
        status_text.text("Creating/retrieving strategy...")
        progress_bar.progress(20)
        
        if selected_strategy == "Create New Strategy":
            strategy = Strategy(
                name=new_strategy_name.strip(),
                description=new_strategy_desc.strip() if new_strategy_desc else None
            )
            strategy = strategy_repo.create(strategy)
            st.success(f"✅ Created new strategy: {strategy.name}")
        else:
            strategy = strategy_repo.get_by_name(selected_strategy)
            if not strategy:
                st.error(f"Strategy '{selected_strategy}' not found")
                return
        
        # Step 2: Process CSV
        status_text.text("Processing CSV file...")
        progress_bar.progress(40)
        
        trades, validation_result, metadata = csv_processor.process(
            file_content, filename, strategy.id
        )
        
        if not validation_result.is_valid:
            st.error("❌ File validation failed. Please fix the errors and try again.")
            return
        
        if not trades:
            st.warning("No valid trades found in the file")
            return
        
        # Step 3: Handle upload action
        status_text.text(f"Saving {len(trades)} trades...")
        progress_bar.progress(60)
        
        if upload_action == "Replace" and selected_strategy != "Create New Strategy":
            # Delete existing trades
            deleted_count = trade_repo.delete_by_strategy(strategy.id)
            if deleted_count > 0:
                st.info(f"Removed {deleted_count} existing trades")
        
        # Step 4: Save trades
        status_text.text("Saving trades to database...")
        progress_bar.progress(80)
        
        saved_count = trade_repo.bulk_create(trades)
        
        # Step 5: Update strategy statistics
        status_text.text("Updating strategy statistics...")
        progress_bar.progress(90)
        
        strategy_repo.update_stats(strategy.id)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Upload complete!")
        
        # Show success message
        st.success(f"""
        ✅ **Upload Successful!**
        
        - **Strategy:** {strategy.name}
        - **Platform:** {metadata['platform']}
        - **Trades Processed:** {saved_count}
        - **Action:** {upload_action}
        """)
        
        # Show summary statistics
        if saved_count > 0:
            summary = trade_repo.get_trade_summary(strategy.id)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", summary.get('total_trades', 0))
            with col2:
                st.metric("Total P&L", f"${summary.get('total_pnl', 0):,.2f}")
            with col3:
                win_rate = (summary.get('winning_trades', 0) / summary.get('total_trades', 1)) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col4:
                st.metric("Avg Trade", f"${summary.get('avg_pnl', 0):,.2f}")
        
        time.sleep(2)
        st.rerun()
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Processing failed: {str(e)}")
        st.exception(e)


def manage_tab():
    """Manage existing strategies."""
    st.header("Manage Strategies")
    
    # Get all strategies
    strategies = strategy_repo.get_active_strategies()
    
    if not strategies:
        st.info("No strategies found. Upload data to create your first strategy.")
        return
    
    # Strategy list
    for strategy in strategies:
        with st.expander(f"📊 {strategy.name}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Description:** {strategy.description or 'No description'}")
                st.write(f"**Created:** {strategy.created_at.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Last Updated:** {strategy.updated_at.strftime('%Y-%m-%d %H:%M')}")
            
            with col2:
                st.metric("Total Trades", f"{strategy.total_trades:,}")
                st.metric("Total P&L", f"${float(strategy.total_pnl):,.2f}")
            
            with col3:
                st.write("**Actions:**")
                
                if st.button("📝 Edit", key=f"edit_{strategy.id}"):
                    edit_strategy(strategy)
                
                if st.button("🗑️ Delete", key=f"delete_{strategy.id}"):
                    if st.confirm(f"Delete strategy '{strategy.name}' and all its trades?"):
                        strategy_repo.delete(strategy.id)
                        st.success(f"Deleted strategy: {strategy.name}")
                        st.rerun()
                
                if st.button("📥 Export Trades", key=f"export_{strategy.id}"):
                    export_trades(strategy)


def edit_strategy(strategy: Strategy):
    """Edit strategy details."""
    # Implementation for editing strategy
    st.info("Edit functionality coming soon!")


def export_trades(strategy: Strategy):
    """Export strategy trades."""
    # Implementation for exporting trades
    st.info("Export functionality coming soon!")


def formats_tab():
    """Show supported file formats."""
    st.header("Supported File Formats")
    
    formats = csv_processor.get_supported_formats()
    
    for fmt in formats:
        with st.expander(f"📄 {fmt['name']} ({fmt['platform']})"):
            st.write(f"**Description:** {fmt['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Required Columns:**")
                for col in fmt['required_columns']:
                    st.write(f"• {col}")
            
            with col2:
                st.write("**Optional Columns:**")
                for col in fmt['optional_columns'][:5]:
                    st.write(f"• {col}")
                if len(fmt['optional_columns']) > 5:
                    st.write(f"• ... and {len(fmt['optional_columns']) - 5} more")
            
            st.write("**Example Date Formats:**")
            for date_fmt in fmt['example_date_formats']:
                st.code(date_fmt)


if __name__ == "__main__":
    main()