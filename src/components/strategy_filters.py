"""Advanced filtering components for strategy management."""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta

from ..models import Strategy
from ..services import StrategyManager


class StrategyFilters:
    """Advanced filtering UI for strategies."""
    
    def __init__(self, strategy_manager: Optional[StrategyManager] = None):
        self.strategy_manager = strategy_manager or StrategyManager()
    
    def render_filters(self) -> Dict[str, Any]:
        """
        Render strategy filter UI and return filter criteria.
        
        Returns:
            Dictionary of filter criteria
        """
        filters = {}
        
        with st.expander("🔍 Advanced Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Performance filters
                st.subheader("Performance")
                
                # P&L filter
                pnl_filter = st.selectbox(
                    "P&L Status",
                    ["All", "Profitable", "Losing", "Breakeven"],
                    key="filter_pnl_status"
                )
                if pnl_filter != "All":
                    filters['pnl_status'] = pnl_filter
                
                # Min trades filter
                min_trades = st.number_input(
                    "Minimum Trades",
                    min_value=0,
                    value=0,
                    step=10,
                    key="filter_min_trades"
                )
                if min_trades > 0:
                    filters['min_trades'] = min_trades
                
                # Win rate range
                win_rate_range = st.slider(
                    "Win Rate Range (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    step=5,
                    key="filter_win_rate"
                )
                if win_rate_range != (0, 100):
                    filters['win_rate_min'] = win_rate_range[0]
                    filters['win_rate_max'] = win_rate_range[1]
            
            with col2:
                # Date filters
                st.subheader("Date Range")
                
                date_preset = st.selectbox(
                    "Quick Select",
                    ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", 
                     "This Year", "Last Year", "Custom"],
                    key="filter_date_preset"
                )
                
                if date_preset == "Custom":
                    date_from = st.date_input(
                        "From Date",
                        value=None,
                        key="filter_date_from"
                    )
                    date_to = st.date_input(
                        "To Date",
                        value=None,
                        key="filter_date_to"
                    )
                    if date_from:
                        filters['created_after'] = date_from
                    if date_to:
                        filters['created_before'] = date_to
                elif date_preset != "All Time":
                    today = date.today()
                    if date_preset == "Last 7 Days":
                        filters['created_after'] = today - timedelta(days=7)
                    elif date_preset == "Last 30 Days":
                        filters['created_after'] = today - timedelta(days=30)
                    elif date_preset == "Last 90 Days":
                        filters['created_after'] = today - timedelta(days=90)
                    elif date_preset == "This Year":
                        filters['created_after'] = date(today.year, 1, 1)
                    elif date_preset == "Last Year":
                        filters['created_after'] = date(today.year - 1, 1, 1)
                        filters['created_before'] = date(today.year - 1, 12, 31)
                
                # Activity filter
                activity = st.selectbox(
                    "Activity Status",
                    ["All", "Active (Has Trades)", "Empty (No Trades)", "Recently Updated"],
                    key="filter_activity"
                )
                if activity != "All":
                    filters['activity'] = activity
            
            with col3:
                # Sorting options
                st.subheader("Sort & Display")
                
                sort_by = st.selectbox(
                    "Sort By",
                    ["Created Date", "Updated Date", "Name", "Total P&L", 
                     "Trade Count", "Win Rate"],
                    key="filter_sort_by"
                )
                filters['sort_by'] = sort_by
                
                sort_order = st.radio(
                    "Sort Order",
                    ["Descending", "Ascending"],
                    key="filter_sort_order"
                )
                filters['sort_order'] = sort_order
                
                # Items per page
                page_size = st.selectbox(
                    "Items Per Page",
                    [5, 10, 20, 50],
                    index=1,
                    key="filter_page_size"
                )
                filters['page_size'] = page_size
            
            # Clear filters button
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Clear Filters", key="clear_filters"):
                    for key in list(st.session_state.keys()):
                        if key.startswith("filter_"):
                            del st.session_state[key]
                    st.rerun()
        
        return filters
    
    def apply_filters(self, strategies: List[Strategy], filters: Dict[str, Any]) -> List[Strategy]:
        """
        Apply filters to strategy list.
        
        Args:
            strategies: List of strategies to filter
            filters: Filter criteria dictionary
            
        Returns:
            Filtered list of strategies
        """
        filtered = strategies.copy()
        
        # Performance filters
        if 'pnl_status' in filters:
            if filters['pnl_status'] == "Profitable":
                filtered = [s for s in filtered if s.total_pnl > 0]
            elif filters['pnl_status'] == "Losing":
                filtered = [s for s in filtered if s.total_pnl < 0]
            elif filters['pnl_status'] == "Breakeven":
                filtered = [s for s in filtered if s.total_pnl == 0]
        
        if 'min_trades' in filters:
            filtered = [s for s in filtered if s.total_trades >= filters['min_trades']]
        
        # Win rate filter (requires fetching stats)
        if 'win_rate_min' in filters or 'win_rate_max' in filters:
            filtered_with_stats = []
            for strategy in filtered:
                stats = self.strategy_manager.get_strategy_statistics(strategy.id)
                win_rate = stats.get('win_rate', 0)
                if win_rate >= filters.get('win_rate_min', 0) and win_rate <= filters.get('win_rate_max', 100):
                    filtered_with_stats.append(strategy)
            filtered = filtered_with_stats
        
        # Date filters
        if 'created_after' in filters:
            filtered = [s for s in filtered if s.created_at.date() >= filters['created_after']]
        
        if 'created_before' in filters:
            filtered = [s for s in filtered if s.created_at.date() <= filters['created_before']]
        
        # Activity filter
        if 'activity' in filters:
            if filters['activity'] == "Active (Has Trades)":
                filtered = [s for s in filtered if s.total_trades > 0]
            elif filters['activity'] == "Empty (No Trades)":
                filtered = [s for s in filtered if s.total_trades == 0]
            elif filters['activity'] == "Recently Updated":
                cutoff = datetime.utcnow() - timedelta(days=7)
                filtered = [s for s in filtered if s.updated_at >= cutoff]
        
        # Sorting
        sort_by = filters.get('sort_by', 'Created Date')
        reverse = filters.get('sort_order', 'Descending') == 'Descending'
        
        if sort_by == "Created Date":
            filtered.sort(key=lambda s: s.created_at, reverse=reverse)
        elif sort_by == "Updated Date":
            filtered.sort(key=lambda s: s.updated_at, reverse=reverse)
        elif sort_by == "Name":
            filtered.sort(key=lambda s: s.name.lower(), reverse=reverse)
        elif sort_by == "Total P&L":
            filtered.sort(key=lambda s: float(s.total_pnl), reverse=reverse)
        elif sort_by == "Trade Count":
            filtered.sort(key=lambda s: s.total_trades, reverse=reverse)
        elif sort_by == "Win Rate":
            # Sort by win rate (requires fetching stats)
            strategies_with_stats = []
            for strategy in filtered:
                stats = self.strategy_manager.get_strategy_statistics(strategy.id)
                strategies_with_stats.append((strategy, stats.get('win_rate', 0)))
            strategies_with_stats.sort(key=lambda x: x[1], reverse=reverse)
            filtered = [s[0] for s in strategies_with_stats]
        
        return filtered
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of active filters.
        
        Args:
            filters: Filter criteria dictionary
            
        Returns:
            Summary string
        """
        if not filters:
            return "No filters applied"
        
        parts = []
        
        if 'pnl_status' in filters:
            parts.append(f"P&L: {filters['pnl_status']}")
        
        if 'min_trades' in filters:
            parts.append(f"Min trades: {filters['min_trades']}")
        
        if 'win_rate_min' in filters or 'win_rate_max' in filters:
            min_wr = filters.get('win_rate_min', 0)
            max_wr = filters.get('win_rate_max', 100)
            parts.append(f"Win rate: {min_wr}%-{max_wr}%")
        
        if 'created_after' in filters:
            parts.append(f"Created after: {filters['created_after']}")
        
        if 'activity' in filters:
            parts.append(f"Activity: {filters['activity']}")
        
        if 'sort_by' in filters:
            order = "↓" if filters.get('sort_order') == 'Descending' else "↑"
            parts.append(f"Sort: {filters['sort_by']} {order}")
        
        return " | ".join(parts)