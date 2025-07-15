"""Login component for Streamlit authentication."""

import streamlit as st
from typing import Optional, Callable
from src.auth.auth_manager import AuthManager


class LoginComponent:
    """Handles user login interface and authentication."""
    
    def __init__(self, auth_manager: AuthManager):
        """
        Initialize login component.
        
        Args:
            auth_manager: Authentication manager instance
        """
        self.auth_manager = auth_manager
    
    def render_login_form(self) -> bool:
        """
        Render login form.
        
        Returns:
            True if login successful
        """
        st.markdown("### 🔐 Login to Trading Strategy Analyzer")
        
        with st.form("login_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    submit = st.form_submit_button("Login", type="primary", use_container_width=True)
                with col1_2:
                    if st.form_submit_button("Forgot Password?", use_container_width=True):
                        st.info("Please contact your administrator to reset your password.")
            
            with col2:
                st.markdown("#### Default Credentials")
                st.markdown("""
                **Username:** admin  
                **Password:** admin123
                
                ⚠️ Please change the default password after first login.
                """)
        
        if submit:
            if not username:
                st.error("Please enter username")
                return False
            
            if not password:
                st.error("Please enter password")
                return False
            
            if self.auth_manager.login(username, password, st.session_state):
                st.success(f"Welcome back, {username}!")
                st.rerun()
                return True
            else:
                st.error("Invalid username or password")
                return False
        
        return False
    
    def render_user_menu(self):
        """Render user menu in sidebar."""
        username = self.auth_manager.get_current_user(st.session_state)
        
        if username:
            with st.sidebar:
                st.markdown("---")
                st.markdown(f"### 👤 {username}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Profile", use_container_width=True):
                        st.session_state['show_profile'] = True
                
                with col2:
                    if st.button("Logout", use_container_width=True):
                        self.auth_manager.logout(st.session_state)
                        st.rerun()
                
                # Session info
                login_time = st.session_state.get('login_time')
                if login_time:
                    from datetime import datetime
                    duration = datetime.now() - login_time
                    st.caption(f"Session: {int(duration.total_seconds() / 60)} minutes")
    
    def render_profile_page(self):
        """Render user profile page."""
        username = self.auth_manager.get_current_user(st.session_state)
        
        if not username:
            st.error("Not logged in")
            return
        
        st.title("👤 User Profile")
        st.markdown(f"### Welcome, {username}")
        
        tab1, tab2 = st.tabs(["Profile Info", "Change Password"])
        
        with tab1:
            user = self.auth_manager.users.get(username)
            if user:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Username", user.username)
                    st.metric("Status", "Active" if user.is_active else "Inactive")
                
                with col2:
                    if user.last_login:
                        st.metric("Last Login", user.last_login.strftime("%Y-%m-%d %H:%M"))
                    if user.created_at:
                        st.metric("Member Since", user.created_at.strftime("%Y-%m-%d"))
        
        with tab2:
            st.markdown("### Change Password")
            
            with st.form("change_password_form"):
                old_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("Change Password", type="primary"):
                    if not all([old_password, new_password, confirm_password]):
                        st.error("Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("New passwords do not match")
                    else:
                        success, message = self.auth_manager.change_password(
                            username, old_password, new_password
                        )
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
    
    def require_auth(self, render_func: Callable):
        """
        Decorator to require authentication for a page.
        
        Args:
            render_func: Function to render the page content
        """
        if not self.auth_manager.is_authenticated(st.session_state):
            # Center the login form
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                self.render_login_form()
        else:
            # Check if profile page should be shown
            if st.session_state.get('show_profile', False):
                self.render_profile_page()
                if st.button("← Back to App"):
                    st.session_state['show_profile'] = False
                    st.rerun()
            else:
                # Render user menu in sidebar
                self.render_user_menu()
                # Render the actual page content
                render_func()


def check_authentication() -> bool:
    """
    Quick check if user is authenticated.
    
    Returns:
        True if authenticated
    """
    return st.session_state.get('authenticated', False)