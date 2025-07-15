"""Streamlit page authentication wrapper."""

import streamlit as st
from typing import Dict, Any, Optional

from .auth_manager import AuthManager
from ..components.login import LoginComponent


def authenticated_page(
    page_title: str = "Trading Strategy Analyzer",
    page_icon: str = "📊",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded"
):
    """
    Set up page configuration and authentication.
    
    Args:
        page_title: Page title
        page_icon: Page icon
        layout: Page layout
        initial_sidebar_state: Sidebar state
        
    Returns:
        Tuple of (is_authenticated, auth_manager, login_component)
    """
    # Set page config first (must be called before any other st commands)
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state
    )
    
    # Initialize authentication
    auth_manager = AuthManager()
    login_component = LoginComponent(auth_manager)
    
    # Check if authenticated
    if not auth_manager.is_authenticated(st.session_state):
        # Show login page
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_component.render_login_form()
        return False, auth_manager, login_component
    
    # User is authenticated - render user menu
    login_component.render_user_menu()
    
    # Check if profile page should be shown
    if st.session_state.get('show_profile', False):
        login_component.render_profile_page()
        if st.button("← Back to App"):
            st.session_state['show_profile'] = False
            st.rerun()
        return False, auth_manager, login_component
    
    return True, auth_manager, login_component