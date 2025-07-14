"""Strategy Management page for uploading and managing trading strategies."""

import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
import io
import time
from datetime import datetime

from src.utils import CSVProcessor, ValidationSeverity, get_session_manager
from src.services import StrategyManager
from src.models import Strategy, Trade
from src.components import FileUploadComponent, StrategySelector, UploadFeedback, StrategyForms, StrategyFilters
from config import config


# Page configuration
st.set_page_config(
    page_title="Strategy Management",
    page_icon="📁",
    layout="wide"
)

# Initialize services
strategy_manager = StrategyManager()
csv_processor = CSVProcessor()
strategy_forms = StrategyForms(strategy_manager)
strategy_filters = StrategyFilters(strategy_manager)
session_manager = get_session_manager()


def main():
    st.title("📁 Strategy Management")
    st.markdown("Upload CSV files, manage trading strategies, and organize your trading data.")
    
    # Show active strategy if set
    if session_manager.active_strategy:
        st.info(f"🎯 Active Strategy: **{session_manager.active_strategy.name}**")
    
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
        existing_strategies = strategy_manager.get_all_strategies()
        strategy_names = ["Create New Strategy"] + [s.name for s in existing_strategies]
        
        # Default to active strategy if set
        default_index = 0
        if session_manager.active_strategy:
            try:
                default_index = strategy_names.index(session_manager.active_strategy.name)
            except ValueError:
                pass
        
        selected_strategy = st.selectbox(
            "Select Strategy",
            strategy_names,
            index=default_index,
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
        existing = strategy_manager.repository.get_by_name(new_strategy_name.strip())
        if existing:
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
            strategy, error = strategy_manager.create_strategy(
                name=new_strategy_name.strip(),
                description=new_strategy_desc.strip() if new_strategy_desc else ""
            )
            if error:
                st.error(f"Failed to create strategy: {error}")
                return
            st.success(f"✅ Created new strategy: {strategy.name}")
            # Set as active strategy
            session_manager.set_active_strategy(strategy.id)
        else:
            strategy = strategy_manager.repository.get_by_name(selected_strategy)
            if not strategy:
                st.error(f"Strategy '{selected_strategy}' not found")
                return
            # Set as active strategy
            session_manager.set_active_strategy(strategy.id)
        
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
            # Replace all trades
            saved_count, error = strategy_manager.replace_trades(strategy.id, trades)
            if error:
                st.error(f"Failed to save trades: {error}")
                return
        else:
            # Append trades
            saved_count, error = strategy_manager.append_trades(strategy.id, trades)
            if error:
                st.error(f"Failed to save trades: {error}")
                return
        
        # Statistics are automatically updated by the manager
        status_text.text("Updating strategy statistics...")
        progress_bar.progress(90)
        
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
            summary = strategy_manager.get_strategy_statistics(strategy.id)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", summary.get('trade_count', 0))
            with col2:
                st.metric("Total P&L", f"${summary.get('total_pnl', 0):,.2f}")
            with col3:
                st.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
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
    
    # Add new strategy button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("➕ New Strategy", type="primary"):
            st.session_state['show_create_form'] = True
    
    # Show create form if requested
    if st.session_state.get('show_create_form', False):
        created_strategy, submitted = strategy_forms.create_strategy_form()
        if submitted:
            st.session_state['show_create_form'] = False
            if created_strategy:
                # Set as active strategy
                session_manager.set_active_strategy(created_strategy.id)
                st.rerun()
    
    st.divider()
    
    # Render filters
    filters = strategy_filters.render_filters()
    
    # Get filter summary
    filter_summary = strategy_filters.get_filter_summary(filters)
    if filter_summary != "No filters applied":
        st.info(f"**Active Filters:** {filter_summary}")
    
    # Get all strategies with search
    search_query = st.text_input(
        "🔍 Search strategies",
        placeholder="Search by name or description...",
        key="strategy_search_main"
    )
    
    # Get strategies
    if search_query:
        strategies = strategy_manager.get_strategies(limit=1000, search_query=search_query)
    else:
        strategies = strategy_manager.get_all_strategies()
    
    # Apply filters
    if filters:
        strategies = strategy_filters.apply_filters(strategies, filters)
    
    if not strategies:
        if search_query or filters:
            st.info("No strategies found matching your criteria.")
        else:
            st.info("No strategies found. Click 'New Strategy' to create your first strategy.")
        return
    
    # Display count
    st.caption(f"Found {len(strategies)} strategies")
    
    # Display strategies with custom page size
    page_size = filters.get('page_size', 10)
    selected = strategy_forms.strategy_list_view(strategies, page_size=page_size, show_search=False)
    
    if selected:
        # Handle export if requested
        export_trades(selected)


def edit_strategy(strategy: Strategy):
    """Edit strategy details."""
    if strategy_forms.edit_strategy_dialog(strategy):
        st.rerun()


def export_trades(strategy: Strategy):
    """Export strategy trades."""
    try:
        # Get trades from database
        from src.db import get_db_manager
        db_manager = get_db_manager()
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM trades WHERE strategy_id = ? ORDER BY trade_date""",
                (strategy.id,)
            )
            trades = cursor.fetchall()
            
            if not trades:
                st.warning("No trades found for this strategy")
                return
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(trades, columns=[
                'id', 'strategy_id', 'trade_date', 'symbol', 'side',
                'entry_price', 'exit_price', 'quantity', 'pnl', 'commission'
            ])
            
            # Remove internal columns
            df = df.drop(columns=['id', 'strategy_id'])
            
            # Generate CSV
            csv = df.to_csv(index=False)
            
            # Download button
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{strategy.name.replace(' ', '_')}_trades.csv",
                mime="text/csv"
            )
            
            st.success(f"Ready to export {len(trades)} trades")
            
    except Exception as e:
        st.error(f"Failed to export trades: {str(e)}")


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