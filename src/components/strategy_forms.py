"""Strategy management UI forms and dialogs."""

import streamlit as st
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

from ..services import StrategyManager
from ..models import Strategy


class StrategyForms:
    """UI forms for strategy management."""
    
    def __init__(self, strategy_manager: Optional[StrategyManager] = None):
        self.strategy_manager = strategy_manager or StrategyManager()
    
    def create_strategy_form(self) -> Tuple[Optional[Strategy], bool]:
        """
        Display strategy creation form.
        
        Returns:
            Tuple of (created strategy, form submitted)
        """
        with st.form("create_strategy_form"):
            st.subheader("Create New Strategy")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                name = st.text_input(
                    "Strategy Name *",
                    placeholder="e.g., Momentum Strategy 2024",
                    help="Choose a unique name for your strategy"
                )
            
            with col2:
                st.write("")  # Spacing
                st.write("")
                st.info("* Required field")
            
            description = st.text_area(
                "Description",
                placeholder="Describe your trading strategy...",
                height=100,
                help="Optional: Add details about the strategy's approach, timeframe, markets, etc."
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                submitted = st.form_submit_button("Create Strategy", type="primary")
            
            with col2:
                cancelled = st.form_submit_button("Cancel")
            
            if cancelled:
                return None, True
            
            if submitted:
                if not name or not name.strip():
                    st.error("Strategy name is required")
                    return None, False
                
                # Create strategy
                strategy, error = self.strategy_manager.create_strategy(
                    name=name.strip(),
                    description=description.strip() if description else ""
                )
                
                if error:
                    st.error(f"Failed to create strategy: {error}")
                    return None, False
                
                st.success(f"✅ Strategy '{strategy.name}' created successfully!")
                return strategy, True
        
        return None, False
    
    def edit_strategy_dialog(self, strategy: Strategy) -> bool:
        """
        Display strategy edit dialog.
        
        Args:
            strategy: Strategy to edit
            
        Returns:
            True if strategy was updated
        """
        dialog_key = f"edit_strategy_{strategy.id}"
        
        with st.expander(f"✏️ Edit Strategy: {strategy.name}", expanded=True):
            with st.form(f"edit_form_{strategy.id}"):
                st.write("**Current Details:**")
                st.caption(f"Created: {strategy.created_at.strftime('%Y-%m-%d %H:%M')}")
                st.caption(f"Last Updated: {strategy.updated_at.strftime('%Y-%m-%d %H:%M')}")
                
                st.divider()
                
                name = st.text_input(
                    "Strategy Name *",
                    value=strategy.name,
                    help="Choose a unique name for your strategy"
                )
                
                description = st.text_area(
                    "Description",
                    value=strategy.description or "",
                    height=100,
                    help="Optional: Add details about the strategy"
                )
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    save = st.form_submit_button("Save Changes", type="primary")
                
                with col2:
                    cancel = st.form_submit_button("Cancel")
                
                if cancel:
                    return False
                
                if save:
                    # Check if anything changed
                    if name == strategy.name and description == (strategy.description or ""):
                        st.info("No changes made")
                        return False
                    
                    # Update strategy
                    updated, error = self.strategy_manager.update_strategy(
                        strategy_id=strategy.id,
                        name=name.strip() if name != strategy.name else None,
                        description=description.strip() if description != strategy.description else None
                    )
                    
                    if error:
                        st.error(f"Failed to update strategy: {error}")
                        return False
                    
                    st.success("✅ Strategy updated successfully!")
                    return True
        
        return False
    
    def delete_confirmation_dialog(self, strategy: Strategy) -> Tuple[bool, bool]:
        """
        Display delete confirmation dialog.
        
        Args:
            strategy: Strategy to delete
            
        Returns:
            Tuple of (confirmed, soft_delete)
        """
        # Get dependent data counts
        stats = self.strategy_manager.get_strategy_statistics(strategy.id)
        
        st.warning(f"""
        ⚠️ **Delete Strategy: {strategy.name}?**
        
        This action will affect:
        - **{strategy.total_trades}** trade records
        - **Total P&L:** ${float(strategy.total_pnl):,.2f}
        - **Win Rate:** {stats.get('win_rate', 0):.1f}%
        - All associated performance metrics
        - All historical data
        """)
        
        # Delete options
        delete_type = st.radio(
            "Delete Type:",
            ["Soft Delete (Recoverable)", "Permanent Delete"],
            key=f"delete_type_{strategy.id}",
            help="Soft delete allows recovery later, permanent delete removes all data"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🗑️ Confirm Delete", type="primary", key=f"confirm_delete_{strategy.id}"):
                return True, delete_type == "Soft Delete (Recoverable)"
        
        with col2:
            if st.button("Cancel", key=f"cancel_delete_{strategy.id}"):
                return False, False
        
        return False, False
    
    def strategy_details_card(self, strategy: Strategy, show_actions: bool = True) -> Dict[str, Any]:
        """
        Display strategy details in a card format.
        
        Args:
            strategy: Strategy to display
            show_actions: Whether to show action buttons
            
        Returns:
            Dictionary with action results
        """
        actions = {
            'edit': False,
            'delete': False,
            'view': False,
            'export': False
        }
        
        with st.container():
            # Header
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"📊 {strategy.name}")
                if strategy.description:
                    st.write(strategy.description)
            
            with col2:
                status_color = "green" if strategy.total_trades > 0 else "gray"
                st.markdown(f"""
                <div style="text-align: right;">
                    <span style="color: {status_color}; font-size: 12px;">
                        ● {"Active" if strategy.total_trades > 0 else "No Data"}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", f"{strategy.total_trades:,}")
            
            with col2:
                pnl_color = "green" if strategy.total_pnl > 0 else "red"
                st.metric(
                    "Total P&L", 
                    f"${float(strategy.total_pnl):,.2f}",
                    delta=None
                )
            
            with col3:
                # Get win rate from statistics
                stats = self.strategy_manager.get_strategy_statistics(strategy.id)
                win_rate = stats.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col4:
                avg_pnl = stats.get('avg_pnl', 0)
                st.metric("Avg Trade", f"${avg_pnl:,.2f}")
            
            # Additional info
            with st.expander("More Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Trading Statistics:**")
                    st.write(f"- Winning Trades: {stats.get('winning_trades', 0)}")
                    st.write(f"- Losing Trades: {stats.get('losing_trades', 0)}")
                    st.write(f"- Breakeven Trades: {stats.get('breakeven_trades', 0)}")
                    st.write(f"- Max Win: ${stats.get('max_pnl', 0):,.2f}")
                    st.write(f"- Max Loss: ${stats.get('min_pnl', 0):,.2f}")
                
                with col2:
                    st.write("**Timeline:**")
                    st.write(f"- Created: {strategy.created_at.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"- Updated: {strategy.updated_at.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"- ID: {strategy.id}")
            
            # Action buttons
            if show_actions:
                st.divider()
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button("👁️ View", key=f"view_{strategy.id}"):
                        actions['view'] = True
                
                with col2:
                    if st.button("✏️ Edit", key=f"edit_{strategy.id}"):
                        actions['edit'] = True
                
                with col3:
                    if st.button("📥 Export", key=f"export_{strategy.id}"):
                        actions['export'] = True
                
                with col4:
                    if st.button("📊 Analyze", key=f"analyze_{strategy.id}"):
                        # Use session manager to set active strategy
                        from ..utils import get_session_manager
                        session_manager = get_session_manager()
                        session_manager.set_active_strategy(strategy.id)
                        st.switch_page("pages/02_📈_Performance_Dashboard.py")
                
                with col5:
                    if st.button("🗑️ Delete", key=f"delete_{strategy.id}", type="secondary"):
                        actions['delete'] = True
        
        return actions
    
    def strategy_list_view(self, 
                          strategies: list[Strategy], 
                          page_size: int = 10,
                          show_search: bool = True) -> Optional[Strategy]:
        """
        Display paginated list of strategies.
        
        Args:
            strategies: List of strategies to display
            page_size: Number of strategies per page
            show_search: Whether to show search box
            
        Returns:
            Selected strategy if any
        """
        selected_strategy = None
        
        # Search box
        if show_search and len(strategies) > 5:
            search_query = st.text_input(
                "🔍 Search strategies",
                placeholder="Search by name or description...",
                key="strategy_search"
            )
            
            if search_query:
                # Filter strategies
                strategies = [
                    s for s in strategies
                    if search_query.lower() in s.name.lower() or
                    (s.description and search_query.lower() in s.description.lower())
                ]
        
        if not strategies:
            st.info("No strategies found matching your search.")
            return None
        
        # Pagination
        total_pages = (len(strategies) - 1) // page_size + 1
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    key="strategy_page"
                )
        else:
            page = 1
        
        # Display strategies for current page
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(strategies))
        
        for i, strategy in enumerate(strategies[start_idx:end_idx]):
            with st.container():
                actions = self.strategy_details_card(strategy)
                
                # Handle actions
                if actions['edit']:
                    if self.edit_strategy_dialog(strategy):
                        st.rerun()
                
                if actions['delete']:
                    confirmed, soft_delete = self.delete_confirmation_dialog(strategy)
                    if confirmed:
                        success, error = self.strategy_manager.delete_strategy(
                            strategy.id, 
                            cascade=True,
                            soft_delete=soft_delete
                        )
                        if success:
                            if soft_delete:
                                st.success(f"Strategy '{strategy.name}' has been archived (soft deleted)")
                            else:
                                st.success(f"Strategy '{strategy.name}' deleted permanently")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete strategy: {error}")
                
                if actions['view'] or actions['export']:
                    selected_strategy = strategy
                
                if i < end_idx - start_idx - 1:
                    st.divider()
        
        # Page info
        if total_pages > 1:
            st.caption(f"Page {page} of {total_pages} ({len(strategies)} strategies total)")
        
        return selected_strategy