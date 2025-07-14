"""Authentication module for the Trading Strategy Analyzer."""

from .auth_manager import AuthManager, User
from .page_auth import authenticated_page

__all__ = ['AuthManager', 'User', 'authenticated_page']