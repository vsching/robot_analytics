"""
Monthly and Weekly Breakdown Tables Component

This module provides interactive data tables for displaying trading performance
metrics aggregated by monthly and weekly periods with advanced sorting,
filtering, and export capabilities using streamlit-aggrid.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from st_aggrid.shared import ColumnsAutoSizeMode

from src.analytics.analytics_engine import AnalyticsEngine
from src.analytics.cache_manager import MetricsCacheManager
from src.models.strategy import Strategy
from src.models.trade import Trade


class BreakdownTables:
    """Interactive breakdown tables for trading performance analysis."""
    
    def __init__(self, analytics_engine: AnalyticsEngine, cache_manager: MetricsCacheManager):
        """
        Initialize the breakdown tables component.
        
        Args:
            analytics_engine: Analytics engine for calculations
            cache_manager: Cache manager for performance optimization
        """
        self.analytics = analytics_engine
        self.cache = cache_manager
        
        # Define grid themes
        self.grid_theme = {
            "header_background_color": "#262730",
            "header_text_color": "#FAFAFA",
            "odd_row_background_color": "#262730",
            "even_row_background_color": "#31333F",
            "grid_background_color": "#0E1117",
            "text_color": "#FAFAFA"
        }
        
        # Define cell renderers
        self.currency_renderer = JsCode("""
            function(params) {
                if (params.value == null) return '';
                const value = parseFloat(params.value);
                const formatted = new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }).format(Math.abs(value));
                
                if (value < 0) {
                    return '<span style="color: #FF4B4B;">' + '-' + formatted.slice(1) + '</span>';
                } else if (value > 0) {
                    return '<span style="color: #00CC88;">' + formatted + '</span>';
                } else {
                    return formatted;
                }
            }
        """)
        
        self.percentage_renderer = JsCode("""
            function(params) {
                if (params.value == null) return '';
                const value = parseFloat(params.value);
                const formatted = value.toFixed(2) + '%';
                
                if (value < 0) {
                    return '<span style="color: #FF4B4B;">' + formatted + '</span>';
                } else if (value > 0) {
                    return '<span style="color: #00CC88;">' + formatted + '</span>';
                } else {
                    return formatted;
                }
            }
        """)
        
        self.integer_renderer = JsCode("""
            function(params) {
                if (params.value == null) return '';
                return parseInt(params.value).toLocaleString();
            }
        """)

    def _configure_grid_options(self, 
                              df: pd.DataFrame, 
                              value_columns: List[str],
                              percentage_columns: List[str] = None,
                              integer_columns: List[str] = None,
                              pinned_columns: List[str] = None) -> GridOptionsBuilder:
        """
        Configure AgGrid options with custom formatting and themes.
        
        Args:
            df: DataFrame to display
            value_columns: Columns to format as currency
            percentage_columns: Columns to format as percentages
            integer_columns: Columns to format as integers
            pinned_columns: Columns to pin to the left
            
        Returns:
            Configured GridOptionsBuilder
        """
        gb = GridOptionsBuilder.from_dataframe(df)
        
        # Configure grid options
        gb.configure_default_column(
            resizable=True,
            filterable=True,
            sortable=True,
            editable=False,
            flex=1,
            min_width=100
        )
        
        # Configure specific column formatting
        for col in value_columns:
            if col in df.columns:
                gb.configure_column(
                    col,
                    cellRenderer=self.currency_renderer,
                    type=["numericColumn", "customCurrencyFormat"]
                )
        
        if percentage_columns:
            for col in percentage_columns:
                if col in df.columns:
                    gb.configure_column(
                        col,
                        cellRenderer=self.percentage_renderer,
                        type=["numericColumn", "customPercentageFormat"]
                    )
        
        if integer_columns:
            for col in integer_columns:
                if col in df.columns:
                    gb.configure_column(
                        col,
                        cellRenderer=self.integer_renderer,
                        type=["numericColumn"]
                    )
        
        # Pin columns
        if pinned_columns:
            for col in pinned_columns:
                if col in df.columns:
                    gb.configure_column(col, pinned="left", width=150)
        
        # Configure pagination
        gb.configure_pagination(
            enabled=True,
            paginationAutoPageSize=False,
            paginationPageSize=20
        )
        
        # Configure selection
        gb.configure_selection(
            selection_mode="multiple",
            use_checkbox=True,
            header_checkbox=True
        )
        
        # Configure grid options
        gb.configure_grid_options(
            domLayout='normal',
            enableRangeSelection=True,
            rowHeight=35,
            headerHeight=40,
            suppressMenuHide=True,
            animateRows=True,
            enableCellTextSelection=True
        )
        
        return gb

    def monthly_breakdown_table(self, strategy_id: int, date_range: Optional[Tuple[datetime, datetime]] = None) -> None:
        """
        Display monthly breakdown table for a strategy.
        
        Args:
            strategy_id: Strategy ID to analyze
            date_range: Optional date range filter
        """
        # Get cached data if available
        cache_key = f"monthly_breakdown_{strategy_id}_{date_range}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            df = cached_data
        else:
            # Fetch and aggregate data
            df = self._prepare_monthly_data(strategy_id, date_range)
            self.cache.set(cache_key, df, ttl=300)  # Cache for 5 minutes
        
        if df.empty:
            st.warning("No data available for the selected period.")
            return
        
        # Configure grid
        gb = self._configure_grid_options(
            df,
            value_columns=['Total P&L', 'Average Trade', 'Best Trade', 'Worst Trade'],
            percentage_columns=['Win Rate'],
            integer_columns=['Trade Count'],
            pinned_columns=['Month']
        )
        
        # Add summary row styling
        gb.configure_grid_options(
            getRowStyle=JsCode("""
                function(params) {
                    if (params.data && params.data.Month === 'TOTAL') {
                        return {
                            'font-weight': 'bold',
                            'background-color': '#1E1E2E',
                            'border-top': '2px solid #4A4A5E'
                        };
                    }
                }
            """)
        )
        
        # Display grid
        grid_response = AgGrid(
            df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=600,
            theme='streamlit',
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS
        )
        
        # Export functionality
        if st.button("Export Monthly Data", key="export_monthly"):
            self._export_data(df, f"monthly_breakdown_{strategy_id}")

    def weekly_breakdown_table(self, strategy_id: int, date_range: Optional[Tuple[datetime, datetime]] = None) -> None:
        """
        Display weekly breakdown table for a strategy.
        
        Args:
            strategy_id: Strategy ID to analyze
            date_range: Optional date range filter
        """
        # Get cached data if available
        cache_key = f"weekly_breakdown_{strategy_id}_{date_range}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            df = cached_data
        else:
            # Fetch and aggregate data
            df = self._prepare_weekly_data(strategy_id, date_range)
            self.cache.set(cache_key, df, ttl=300)  # Cache for 5 minutes
        
        if df.empty:
            st.warning("No data available for the selected period.")
            return
        
        # Configure grid
        gb = self._configure_grid_options(
            df,
            value_columns=['Total P&L', 'Average Trade', 'Best Trade', 'Worst Trade'] + 
                         [f'{day} P&L' for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']],
            percentage_columns=['Win Rate'],
            integer_columns=['Trade Count'],
            pinned_columns=['Week']
        )
        
        # Add summary row styling
        gb.configure_grid_options(
            getRowStyle=JsCode("""
                function(params) {
                    if (params.data && params.data.Week === 'TOTAL') {
                        return {
                            'font-weight': 'bold',
                            'background-color': '#1E1E2E',
                            'border-top': '2px solid #4A4A5E'
                        };
                    }
                }
            """)
        )
        
        # Display grid
        grid_response = AgGrid(
            df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=600,
            theme='streamlit',
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS
        )
        
        # Export functionality
        if st.button("Export Weekly Data", key="export_weekly"):
            self._export_data(df, f"weekly_breakdown_{strategy_id}")

    def _prepare_monthly_data(self, strategy_id: int, date_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
        """
        Prepare monthly aggregated data for display.
        
        Args:
            strategy_id: Strategy ID to analyze
            date_range: Optional date range filter
            
        Returns:
            DataFrame with monthly breakdown
        """
        # Fetch trades from analytics engine
        trades_df = self.analytics.get_trades_dataframe(
            strategy_id, 
            start_date=date_range[0] if date_range else None,
            end_date=date_range[1] if date_range else None
        )
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Ensure trade_date is datetime
        trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'])
        
        # Create month column
        trades_df['month'] = trades_df['trade_date'].dt.to_period('M')
        
        # Group by month and calculate metrics
        monthly_data = []
        for month, group in trades_df.groupby('month'):
            winning_trades = group[group['pnl'] > 0]
            total_trades = len(group)
            
            monthly_data.append({
                'Month': month.strftime('%B %Y'),
                'Total P&L': group['pnl'].sum(),
                'Trade Count': total_trades,
                'Win Rate': (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0,
                'Average Trade': group['pnl'].mean() if total_trades > 0 else 0,
                'Best Trade': group['pnl'].max() if total_trades > 0 else 0,
                'Worst Trade': group['pnl'].min() if total_trades > 0 else 0
            })
        
        df = pd.DataFrame(monthly_data)
        
        if df.empty:
            return df
        
        # Add summary row
        summary = {
            'Month': 'TOTAL',
            'Total P&L': df['Total P&L'].sum(),
            'Trade Count': df['Trade Count'].sum(),
            'Win Rate': df['Win Rate'].mean(),
            'Average Trade': df['Total P&L'].sum() / df['Trade Count'].sum() if df['Trade Count'].sum() > 0 else 0,
            'Best Trade': df['Best Trade'].max(),
            'Worst Trade': df['Worst Trade'].min()
        }
        
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        return df

    def _prepare_weekly_data(self, strategy_id: int, date_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
        """
        Prepare weekly aggregated data for display.
        
        Args:
            strategy_id: Strategy ID to analyze
            date_range: Optional date range filter
            
        Returns:
            DataFrame with weekly breakdown including day-of-week analysis
        """
        # Fetch trades from analytics engine
        trades_df = self.analytics.get_trades_dataframe(
            strategy_id, 
            start_date=date_range[0] if date_range else None,
            end_date=date_range[1] if date_range else None
        )
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Ensure trade_date is datetime
        trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'])
        
        # Create week and day of week columns
        trades_df['week'] = trades_df['trade_date'].dt.to_period('W')
        trades_df['day_of_week'] = trades_df['trade_date'].dt.day_name()
        
        # Group by week and calculate metrics
        weekly_data = []
        for week, week_group in trades_df.groupby('week'):
            winning_trades = week_group[week_group['pnl'] > 0]
            total_trades = len(week_group)
            
            week_data = {
                'Week': f"Week {week.start_time.isocalendar()[1]}, {week.start_time.year}",
                'Total P&L': week_group['pnl'].sum(),
                'Trade Count': total_trades,
                'Win Rate': (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0,
                'Average Trade': week_group['pnl'].mean() if total_trades > 0 else 0,
                'Best Trade': week_group['pnl'].max() if total_trades > 0 else 0,
                'Worst Trade': week_group['pnl'].min() if total_trades > 0 else 0
            }
            
            # Calculate day-of-week P&L
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                day_trades = week_group[week_group['day_of_week'] == day]
                day_abbr = day[:3]
                week_data[f'{day_abbr} P&L'] = day_trades['pnl'].sum() if not day_trades.empty else 0
            
            weekly_data.append(week_data)
        
        df = pd.DataFrame(weekly_data)
        
        if df.empty:
            return df
        
        # Add summary row
        summary = {
            'Week': 'TOTAL',
            'Total P&L': df['Total P&L'].sum(),
            'Trade Count': df['Trade Count'].sum(),
            'Win Rate': df['Win Rate'].mean(),
            'Average Trade': df['Total P&L'].sum() / df['Trade Count'].sum() if df['Trade Count'].sum() > 0 else 0,
            'Best Trade': df['Best Trade'].max(),
            'Worst Trade': df['Worst Trade'].min()
        }
        
        # Add day-of-week totals
        for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
            summary[f'{day} P&L'] = df[f'{day} P&L'].sum()
        
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        return df

    def _export_data(self, df: pd.DataFrame, filename_prefix: str) -> None:
        """
        Export DataFrame to CSV for download.
        
        Args:
            df: DataFrame to export
            filename_prefix: Prefix for the filename
        """
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )

    def render_period_selector(self) -> Optional[Tuple[datetime, datetime]]:
        """
        Render date range selector for filtering data.
        
        Returns:
            Selected date range or None
        """
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                key="breakdown_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="breakdown_end_date"
            )
        
        if start_date and end_date:
            return (datetime.combine(start_date, datetime.min.time()),
                   datetime.combine(end_date, datetime.max.time()))
        
        return None