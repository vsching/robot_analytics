#!/bin/bash
# Fix authentication issues by removing users.json

echo "🔧 Fixing authentication system..."

if [ -f "users.json" ]; then
    rm users.json
    echo "✅ Removed existing users.json"
else
    echo "ℹ️  No users.json file found"
fi

echo ""
echo "✨ Authentication system reset!"
echo "Default credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Please restart your Streamlit app."