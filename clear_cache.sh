#!/bin/bash
# Clear all Streamlit caches

echo "🧹 Clearing Streamlit caches..."

# Clear local cache
rm -rf .streamlit/cache 2>/dev/null || true

# Clear home directory cache
rm -rf ~/.streamlit/cache 2>/dev/null || true

# Clear Python cache
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clear any session files
rm -f users.json 2>/dev/null || true

echo "✅ Caches cleared!"
echo ""
echo "Please restart your Streamlit app:"
echo "  streamlit run main.py"