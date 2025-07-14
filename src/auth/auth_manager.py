"""Authentication manager for user login and session management."""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import streamlit as st
from dataclasses import dataclass
import json
import os


@dataclass
class User:
    """User model for authentication."""
    username: str
    password_hash: str
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AuthManager:
    """Manages user authentication and session state."""
    
    def __init__(self, users_file: str = "users.json", session_timeout_minutes: int = 30):
        """
        Initialize authentication manager.
        
        Args:
            users_file: Path to JSON file storing user credentials
            session_timeout_minutes: Session timeout in minutes
        """
        self.users_file = users_file
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._load_users()
        
    def _load_users(self):
        """Load users from JSON file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                    self.users = {
                        username: User(**user_data) 
                        for username, user_data in users_data.items()
                    }
            except Exception as e:
                st.error(f"Error loading users: {e}")
                self.users = {}
        else:
            # Create default admin user if no users file exists
            self.users = {
                'admin': User(
                    username='admin',
                    password_hash=self._hash_password('admin123')
                )
            }
            self._save_users()
    
    def _save_users(self):
        """Save users to JSON file."""
        users_data = {}
        for username, user in self.users.items():
            users_data[username] = {
                'username': user.username,
                'password_hash': user.password_hash,
                'is_active': user.is_active,
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'created_at': user.created_at.isoformat() if user.created_at else None
            }
        
        with open(self.users_file, 'w') as f:
            json.dump(users_data, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            True if authentication successful
        """
        if username not in self.users:
            return False
        
        user = self.users[username]
        if not user.is_active:
            return False
        
        if user.password_hash == self._hash_password(password):
            # Update last login
            user.last_login = datetime.now()
            self._save_users()
            return True
        
        return False
    
    def login(self, username: str, password: str) -> bool:
        """
        Log in user and create session.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            True if login successful
        """
        if self.authenticate(username, password):
            # Set session state
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.session_state['login_time'] = datetime.now()
            st.session_state['session_token'] = secrets.token_urlsafe(32)
            return True
        return False
    
    def logout(self):
        """Log out current user and clear session."""
        keys_to_remove = ['authenticated', 'username', 'login_time', 'session_token']
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
    
    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated and session is valid.
        
        Returns:
            True if authenticated and session valid
        """
        if not st.session_state.get('authenticated', False):
            return False
        
        # Check session timeout
        login_time = st.session_state.get('login_time')
        if login_time:
            if datetime.now() - login_time > self.session_timeout:
                self.logout()
                return False
        
        return True
    
    def get_current_user(self) -> Optional[str]:
        """Get current logged in username."""
        if self.is_authenticated():
            return st.session_state.get('username')
        return None
    
    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        if username not in self.users:
            return False, "User not found"
        
        user = self.users[username]
        if user.password_hash != self._hash_password(old_password):
            return False, "Invalid current password"
        
        # Validate new password
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters"
        
        user.password_hash = self._hash_password(new_password)
        self._save_users()
        return True, "Password changed successfully"
    
    def add_user(self, username: str, password: str, is_admin: bool = False) -> Tuple[bool, str]:
        """
        Add a new user.
        
        Args:
            username: Username
            password: Password
            is_admin: Whether user is admin (not used yet, for future expansion)
            
        Returns:
            Tuple of (success, message)
        """
        if username in self.users:
            return False, "User already exists"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        self.users[username] = User(
            username=username,
            password_hash=self._hash_password(password)
        )
        self._save_users()
        return True, f"User {username} created successfully"
    
    def deactivate_user(self, username: str) -> Tuple[bool, str]:
        """
        Deactivate a user.
        
        Args:
            username: Username to deactivate
            
        Returns:
            Tuple of (success, message)
        """
        if username not in self.users:
            return False, "User not found"
        
        self.users[username].is_active = False
        self._save_users()
        return True, f"User {username} deactivated"
    
    def reactivate_user(self, username: str) -> Tuple[bool, str]:
        """
        Reactivate a user.
        
        Args:
            username: Username to reactivate
            
        Returns:
            Tuple of (success, message)
        """
        if username not in self.users:
            return False, "User not found"
        
        self.users[username].is_active = True
        self._save_users()
        return True, f"User {username} reactivated"