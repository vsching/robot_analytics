"""Export and reporting functionality for trading strategy analysis."""

from .export_manager import ExportManager, ExportConfig, ReportTemplate

# PDF generator only available if dependencies are installed
try:
    from .pdf_generator import PDFReportGenerator
    PDF_AVAILABLE = True
except ImportError:
    PDFReportGenerator = None
    PDF_AVAILABLE = False

__all__ = [
    'ExportManager',
    'ExportConfig', 
    'ReportTemplate',
    'PDFReportGenerator',
    'PDF_AVAILABLE'
]