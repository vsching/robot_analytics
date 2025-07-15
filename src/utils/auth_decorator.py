"""Authentication decorator for Streamlit pages."""

import streamlit as st
from functools import wraps
from typing import Callable

from src.auth.auth_manager import AuthManager
from src.components.login import LoginComponent


def require_authentication(func: Callable) -> Callable:
    """
    Decorator to require authentication for Streamlit pages.
    
    Args:
        func: The main function of the page
        
    Returns:
        Wrapped function with authentication
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize authentication
        auth_manager = AuthManager()
        login_component = LoginComponent(auth_manager)
        
        # Wrap the page function with authentication
        login_component.require_auth(lambda: func(*args, **kwargs))
    
    return wrapper