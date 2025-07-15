import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.auth.auth_manager import AuthManager
from src.components.login import LoginComponent

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Confluence Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication
auth_manager = AuthManager()
login_component = LoginComponent(auth_manager)

def main_page():
    """Main page content."""
    st.title("📊 Trading Strategy Confluence Analyzer")
    st.markdown("""
    ### Welcome to the Trading Strategy Analysis System

    This application helps you:
    - 📁 **Upload** and manage multiple trading strategies
    - 📈 **Analyze** performance with comprehensive metrics
    - 🔄 **Detect** confluence patterns between strategies
    - 📊 **Visualize** results with interactive charts
    - 📄 **Export** detailed reports

    ---

    #### Getting Started
    1. Use the **Strategy Management** page to upload your CSV files
    2. View individual strategy performance in **Performance Analysis**
    3. Compare multiple strategies in **Confluence Analysis**
    4. Export your findings using the **Reports** section

    Select a page from the sidebar to begin.
    """)

    # Show system status
    with st.sidebar:
        st.header("System Status")
        st.success("✅ Application Running")
        st.info("📂 Database: SQLite (Local)")
        st.info("🔄 Version: 1.0.0")

# Apply authentication
if not auth_manager.is_authenticated(st.session_state):
    # Show login page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_component.render_login_form()
else:
    # Show main page and user menu
    login_component.render_user_menu()
    
    # Check if profile page should be shown
    if st.session_state.get('show_profile', False):
        login_component.render_profile_page()
        if st.button("← Back to App"):
            st.session_state['show_profile'] = False
            st.rerun()
    else:
        main_page()