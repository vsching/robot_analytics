"""Reusable file upload component with drag-and-drop support."""

import streamlit as st
from typing import Optional, List, Callable
import pandas as pd


class FileUploadComponent:
    """Enhanced file upload component with validation and preview."""
    
    def __init__(self, 
                 accepted_types: List[str] = ['csv', 'xlsx', 'xls'],
                 max_size_mb: int = 100,
                 preview_rows: int = 5):
        """
        Initialize file upload component.
        
        Args:
            accepted_types: List of accepted file extensions
            max_size_mb: Maximum file size in MB
            preview_rows: Number of rows to show in preview
        """
        self.accepted_types = accepted_types
        self.max_size_mb = max_size_mb
        self.preview_rows = preview_rows
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def render(self, 
               key: str = "file_uploader",
               on_file_selected: Optional[Callable] = None) -> Optional[any]:
        """
        Render the file upload component.
        
        Args:
            key: Unique key for the component
            on_file_selected: Callback function when file is selected
            
        Returns:
            Uploaded file object or None
        """
        # Custom CSS for drag-and-drop area
        st.markdown("""
        <style>
        .uploadedFile {
            display: none;
        }
        .stFileUploader {
            padding: 2rem;
        }
        .stFileUploader > div {
            padding: 3rem;
            border: 2px dashed #cccccc;
            border-radius: 10px;
            text-align: center;
            background-color: #f9f9f9;
        }
        .stFileUploader > div:hover {
            border-color: #4CAF50;
            background-color: #f1f8f4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Drop your CSV file here or click to browse",
            type=self.accepted_types,
            key=key,
            help=f"Accepted formats: {', '.join(self.accepted_types)}. Maximum size: {self.max_size_mb}MB"
        )
        
        if uploaded_file is not None:
            # Validate file
            if not self._validate_file(uploaded_file):
                return None
            
            # Show file info
            self._show_file_info(uploaded_file)
            
            # Preview if CSV
            if uploaded_file.name.endswith('.csv'):
                self._show_csv_preview(uploaded_file)
            
            # Call callback if provided
            if on_file_selected:
                on_file_selected(uploaded_file)
            
            return uploaded_file
        
        return None
    
    def _validate_file(self, uploaded_file) -> bool:
        """Validate uploaded file."""
        # Check file size
        file_size = uploaded_file.size
        if file_size > self.max_size_bytes:
            st.error(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed ({self.max_size_mb}MB)")
            return False
        
        # Check file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in self.accepted_types:
            st.error(f"File type '.{file_ext}' is not supported. Accepted types: {', '.join(self.accepted_types)}")
            return False
        
        return True
    
    def _show_file_info(self, uploaded_file):
        """Display file information."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        
        with col2:
            size_mb = uploaded_file.size / 1024 / 1024
            st.metric("File Size", f"{size_mb:.2f} MB")
        
        with col3:
            st.metric("File Type", uploaded_file.type)
    
    def _show_csv_preview(self, uploaded_file):
        """Show preview of CSV file."""
        try:
            # Read first few rows
            df = pd.read_csv(uploaded_file, nrows=self.preview_rows)
            uploaded_file.seek(0)  # Reset file pointer
            
            if not df.empty:
                st.subheader(f"Preview (First {self.preview_rows} rows)")
                st.dataframe(df, use_container_width=True)
                
                # Show basic stats
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Columns:** {len(df.columns)}")
                    st.write(f"**Preview Rows:** {len(df)}")
                with col2:
                    st.write("**Column Names:**")
                    st.write(", ".join(df.columns[:10]))
                    if len(df.columns) > 10:
                        st.write(f"... and {len(df.columns) - 10} more columns")
        
        except Exception as e:
            st.warning(f"Could not preview file: {str(e)}")


def create_drag_drop_area(key: str = "drag_drop") -> Optional[any]:
    """
    Create a simple drag-and-drop file upload area.
    
    Args:
        key: Unique key for the component
        
    Returns:
        Uploaded file or None
    """
    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] {
        padding: 2rem;
    }
    div[data-testid="stFileUploader"] > div {
        padding: 5rem 2rem;
        border: 3px dashed #1f77b4;
        border-radius: 15px;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] > div:hover {
        border-color: #ff6b6b;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 **Drag & Drop your CSV file here**\n\n*or click to browse*",
        type=['csv', 'xlsx', 'xls'],
        key=key
    )
    
    return uploaded_file