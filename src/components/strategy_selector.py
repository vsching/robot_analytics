"""Strategy selection component."""

import streamlit as st
from typing import List, Optional, Tuple
from src.models import Strategy
from src.db import StrategyRepository


class StrategySelector:
    """Component for selecting strategies."""
    
    def __init__(self, strategy_repo: Optional[StrategyRepository] = None):
        """
        Initialize strategy selector.
        
        Args:
            strategy_repo: Strategy repository instance
        """
        self.strategy_repo = strategy_repo or StrategyRepository()
    
    def render_single(self, 
                     key: str = "strategy_select",
                     include_create_option: bool = True,
                     label: str = "Select Strategy") -> Tuple[Optional[Strategy], bool]:
        """
        Render single strategy selector.
        
        Args:
            key: Unique key for the component
            include_create_option: Whether to include "Create New" option
            label: Label for the selectbox
            
        Returns:
            Tuple of (selected_strategy, is_new_strategy)
        """
        strategies = self.strategy_repo.get_active_strategies()
        
        if include_create_option:
            options = ["📝 Create New Strategy"] + [s.name for s in strategies]
        else:
            options = [s.name for s in strategies] if strategies else []
        
        if not options:
            st.warning("No strategies available. Please create a strategy first.")
            return None, False
        
        selected = st.selectbox(label, options, key=key)
        
        if selected == "📝 Create New Strategy":
            return None, True
        else:
            # Find the strategy object
            for strategy in strategies:
                if strategy.name == selected:
                    return strategy, False
        
        return None, False
    
    def render_multi(self,
                    key: str = "multi_strategy_select",
                    min_selection: int = 2,
                    max_selection: Optional[int] = None,
                    label: str = "Select Strategies") -> List[Strategy]:
        """
        Render multi-strategy selector.
        
        Args:
            key: Unique key for the component
            min_selection: Minimum number of strategies to select
            max_selection: Maximum number of strategies to select
            label: Label for the multiselect
            
        Returns:
            List of selected strategies
        """
        strategies = self.strategy_repo.get_active_strategies()
        
        if len(strategies) < min_selection:
            st.warning(f"At least {min_selection} strategies required. Currently have {len(strategies)}.")
            return []
        
        strategy_dict = {s.name: s for s in strategies}
        
        selected_names = st.multiselect(
            label,
            options=list(strategy_dict.keys()),
            key=key,
            help=f"Select between {min_selection} and {max_selection or 'any number of'} strategies"
        )
        
        # Validate selection count
        if len(selected_names) < min_selection:
            st.error(f"Please select at least {min_selection} strategies")
            return []
        
        if max_selection and len(selected_names) > max_selection:
            st.error(f"Please select at most {max_selection} strategies")
            return []
        
        return [strategy_dict[name] for name in selected_names]
    
    def render_comparison(self,
                         key: str = "compare_strategies") -> Tuple[Optional[Strategy], Optional[Strategy]]:
        """
        Render strategy comparison selector (exactly 2 strategies).
        
        Args:
            key: Unique key for the component
            
        Returns:
            Tuple of two selected strategies
        """
        strategies = self.strategy_repo.get_active_strategies()
        
        if len(strategies) < 2:
            st.warning("At least 2 strategies required for comparison.")
            return None, None
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy1 = st.selectbox(
                "First Strategy",
                options=[s.name for s in strategies],
                key=f"{key}_1"
            )
        
        with col2:
            # Filter out first selection from second dropdown
            remaining = [s.name for s in strategies if s.name != strategy1]
            strategy2 = st.selectbox(
                "Second Strategy",
                options=remaining,
                key=f"{key}_2"
            )
        
        # Find strategy objects
        strat1_obj = next((s for s in strategies if s.name == strategy1), None)
        strat2_obj = next((s for s in strategies if s.name == strategy2), None)
        
        return strat1_obj, strat2_obj
    
    def render_with_info(self,
                        key: str = "strategy_with_info",
                        show_stats: bool = True) -> Optional[Strategy]:
        """
        Render strategy selector with additional information.
        
        Args:
            key: Unique key for the component
            show_stats: Whether to show strategy statistics
            
        Returns:
            Selected strategy
        """
        strategy, is_new = self.render_single(key=key, include_create_option=False)
        
        if strategy and show_stats:
            # Show strategy info in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", f"{strategy.total_trades:,}")
            
            with col2:
                st.metric("Total P&L", f"${float(strategy.total_pnl):,.2f}")
            
            with col3:
                if strategy.total_trades > 0:
                    avg_pnl = float(strategy.total_pnl) / strategy.total_trades
                    st.metric("Avg Trade", f"${avg_pnl:,.2f}")
                else:
                    st.metric("Avg Trade", "$0.00")
            
            # Show description if available
            if strategy.description:
                st.info(f"📝 {strategy.description}")
        
        return strategy