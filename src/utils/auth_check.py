"""Simple authentication check for Streamlit pages."""

import streamlit as st


def check_authentication():
    """
    Check if user is authenticated. If not, show login form and stop execution.
    
    This should be called at the beginning of each page.
    """
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Please login to access this page.")
        st.stop()
    
    # Check session timeout
    if 'login_time' in st.session_state:
        from datetime import datetime, timedelta
        login_time = st.session_state['login_time']
        if datetime.now() - login_time > timedelta(minutes=30):
            st.session_state.clear()
            st.warning("⏱️ Your session has expired. Please login again.")
            st.stop()