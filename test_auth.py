"""Test authentication system."""

import streamlit as st
from src.auth import AuthManager
from src.components import LoginComponent

# Page config
st.set_page_config(
    page_title="Auth Test",
    page_icon="🔐",
    layout="wide"
)

# Initialize auth
auth_manager = AuthManager()
login_component = LoginComponent(auth_manager)

def main_content():
    st.title("🎉 Authentication Successful!")
    st.write(f"Welcome, {auth_manager.get_current_user()}!")
    
    # Test creating a new user
    with st.expander("Admin: Create New User"):
        with st.form("create_user"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            if st.form_submit_button("Create User"):
                success, message = auth_manager.add_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

# Apply authentication
login_component.require_auth(main_content)