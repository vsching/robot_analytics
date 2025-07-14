"""Session state management for Streamlit application."""

import streamlit as st
from typing import Optional, Any, Dict
from datetime import datetime

from ..services import StrategyManager
from ..models import Strategy


class SessionStateManager:
    """Manages application session state."""
    
    # Session state keys
    ACTIVE_STRATEGY_ID = 'active_strategy_id'
    ACTIVE_STRATEGY = 'active_strategy'
    LAST_STRATEGY_UPDATE = 'last_strategy_update'
    PAGE_STATE = 'page_state'
    FILTERS = 'filters'
    
    def __init__(self, strategy_manager: Optional[StrategyManager] = None):
        self.strategy_manager = strategy_manager or StrategyManager()
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize default session state values."""
        if self.ACTIVE_STRATEGY_ID not in st.session_state:
            st.session_state[self.ACTIVE_STRATEGY_ID] = None
        
        if self.ACTIVE_STRATEGY not in st.session_state:
            st.session_state[self.ACTIVE_STRATEGY] = None
        
        if self.LAST_STRATEGY_UPDATE not in st.session_state:
            st.session_state[self.LAST_STRATEGY_UPDATE] = None
        
        if self.PAGE_STATE not in st.session_state:
            st.session_state[self.PAGE_STATE] = {}
        
        if self.FILTERS not in st.session_state:
            st.session_state[self.FILTERS] = {}
    
    @property
    def active_strategy_id(self) -> Optional[int]:
        """Get the active strategy ID."""
        return st.session_state.get(self.ACTIVE_STRATEGY_ID)
    
    @property
    def active_strategy(self) -> Optional[Strategy]:
        """Get the active strategy object."""
        # Check if we need to reload the strategy
        strategy_id = self.active_strategy_id
        if strategy_id is None:
            return None
        
        current_strategy = st.session_state.get(self.ACTIVE_STRATEGY)
        last_update = st.session_state.get(self.LAST_STRATEGY_UPDATE)
        
        # Reload if strategy is None or if it's been more than 30 seconds
        should_reload = (
            current_strategy is None or 
            current_strategy.id != strategy_id or
            last_update is None or
            (datetime.utcnow() - last_update).total_seconds() > 30
        )
        
        if should_reload:
            self._reload_active_strategy()
        
        return st.session_state.get(self.ACTIVE_STRATEGY)
    
    def set_active_strategy(self, strategy_id: Optional[int]):
        """
        Set the active strategy by ID.
        
        Args:
            strategy_id: Strategy ID or None to clear
        """
        if strategy_id == self.active_strategy_id:
            return  # No change needed
        
        st.session_state[self.ACTIVE_STRATEGY_ID] = strategy_id
        
        if strategy_id is None:
            st.session_state[self.ACTIVE_STRATEGY] = None
            st.session_state[self.LAST_STRATEGY_UPDATE] = None
        else:
            self._reload_active_strategy()
    
    def _reload_active_strategy(self):
        """Reload the active strategy from database."""
        strategy_id = self.active_strategy_id
        if strategy_id is None:
            return
        
        try:
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if strategy:
                st.session_state[self.ACTIVE_STRATEGY] = strategy
                st.session_state[self.LAST_STRATEGY_UPDATE] = datetime.utcnow()
            else:
                # Strategy was deleted
                self.clear_active_strategy()
        except Exception as e:
            # Error loading strategy
            self.clear_active_strategy()
    
    def clear_active_strategy(self):
        """Clear the active strategy."""
        st.session_state[self.ACTIVE_STRATEGY_ID] = None
        st.session_state[self.ACTIVE_STRATEGY] = None
        st.session_state[self.LAST_STRATEGY_UPDATE] = None
    
    def refresh_active_strategy(self):
        """Force reload of the active strategy."""
        if self.active_strategy_id is not None:
            self._reload_active_strategy()
    
    def get_page_state(self, page: str, key: str, default: Any = None) -> Any:
        """
        Get state for a specific page and key.
        
        Args:
            page: Page identifier
            key: State key
            default: Default value if not found
            
        Returns:
            Stored value or default
        """
        page_states = st.session_state.get(self.PAGE_STATE, {})
        page_state = page_states.get(page, {})
        return page_state.get(key, default)
    
    def set_page_state(self, page: str, key: str, value: Any):
        """
        Set state for a specific page and key.
        
        Args:
            page: Page identifier
            key: State key
            value: Value to store
        """
        if self.PAGE_STATE not in st.session_state:
            st.session_state[self.PAGE_STATE] = {}
        
        if page not in st.session_state[self.PAGE_STATE]:
            st.session_state[self.PAGE_STATE][page] = {}
        
        st.session_state[self.PAGE_STATE][page][key] = value
    
    def clear_page_state(self, page: str):
        """Clear all state for a specific page."""
        if self.PAGE_STATE in st.session_state and page in st.session_state[self.PAGE_STATE]:
            st.session_state[self.PAGE_STATE][page] = {}
    
    def get_filter(self, filter_key: str, default: Any = None) -> Any:
        """Get a filter value."""
        filters = st.session_state.get(self.FILTERS, {})
        return filters.get(filter_key, default)
    
    def set_filter(self, filter_key: str, value: Any):
        """Set a filter value."""
        if self.FILTERS not in st.session_state:
            st.session_state[self.FILTERS] = {}
        st.session_state[self.FILTERS][filter_key] = value
    
    def clear_filters(self):
        """Clear all filters."""
        st.session_state[self.FILTERS] = {}
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current session state."""
        return {
            'active_strategy_id': self.active_strategy_id,
            'active_strategy_name': self.active_strategy.name if self.active_strategy else None,
            'has_filters': bool(st.session_state.get(self.FILTERS)),
            'page_states': list(st.session_state.get(self.PAGE_STATE, {}).keys())
        }


# Global instance
_session_manager = None

def get_session_manager() -> SessionStateManager:
    """Get the global session state manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionStateManager()
    return _session_manager