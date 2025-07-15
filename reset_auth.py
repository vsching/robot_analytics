"""Reset authentication system by removing users.json file."""

import os

def reset_auth():
    """Remove users.json to reset authentication."""
    if os.path.exists('users.json'):
        os.remove('users.json')
        print("✅ Removed users.json file")
    else:
        print("ℹ️ No users.json file found")
    
    print("\nAuthentication has been reset.")
    print("Default credentials will be recreated on next login:")
    print("  Username: admin")
    print("  Password: admin123")

if __name__ == "__main__":
    reset_auth()