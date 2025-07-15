#!/usr/bin/env python3
"""
Verification script for Trading Strategy Analyzer
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def check_database():
    """Check if database exists and has correct tables."""
    db_path = "data/trading_analyzer.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found at", db_path)
        return False
    
    print("✅ Database file exists")
    
    # Check tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = ['strategies', 'trades', 'metrics_cache']
    
    for table in expected_tables:
        if table in tables:
            print(f"✅ Table '{table}' exists")
        else:
            print(f"❌ Table '{table}' missing")
    
    conn.close()
    return True

def check_imports():
    """Check if all key modules can be imported."""
    modules_to_check = [
        ('src.services.strategy_manager', 'StrategyManager'),
        ('src.analytics.analytics_engine', 'AnalyticsEngine'),
        ('src.components.breakdown_tables', 'BreakdownTables'),
        ('src.utils.csv_processor', 'CSVProcessor'),
    ]
    
    all_good = True
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ Successfully imported {class_name} from {module_name}")
        except Exception as e:
            print(f"❌ Failed to import {class_name} from {module_name}: {e}")
            all_good = False
    
    return all_good

def check_config():
    """Check if configuration is properly set up."""
    try:
        from config import config
        print("✅ Configuration loaded successfully")
        print(f"   - Max upload size: {config.MAX_UPLOAD_SIZE_MB}MB")
        print(f"   - Database path: {config.DATABASE_PATH}")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def check_streamlit_pages():
    """Check if all Streamlit pages exist."""
    pages_dir = Path("pages")
    
    if not pages_dir.exists():
        print("❌ Pages directory not found")
        return False
    
    print("✅ Pages directory exists")
    
    expected_pages = [
        "01_📁_Strategy_Management.py",
        "02_📊_Performance_Breakdown.py",
        "03_📈_Performance_Charts.py",
        "04_🔄_Strategy_Comparison.py",
        "05_🎯_Confluence_Detection.py",
        "06_📥_Export_Reports.py"
    ]
    
    for page in expected_pages:
        page_path = pages_dir / page
        if page_path.exists():
            print(f"✅ Page exists: {page}")
        else:
            print(f"❌ Page missing: {page}")
    
    return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Trading Strategy Analyzer - Verification Report")
    print("=" * 60)
    print()
    
    print("1. Database Check:")
    print("-" * 40)
    check_database()
    print()
    
    print("2. Module Import Check:")
    print("-" * 40)
    check_imports()
    print()
    
    print("3. Configuration Check:")
    print("-" * 40)
    check_config()
    print()
    
    print("4. Streamlit Pages Check:")
    print("-" * 40)
    check_streamlit_pages()
    print()
    
    print("=" * 60)
    print("Verification complete!")
    print()
    print("To start the application:")
    print("1. Open your browser to: http://localhost:8501")
    print("2. Navigate to '📁 Strategy Management' to upload CSV files")
    print("3. Use other pages to analyze your trading strategies")
    print("=" * 60)

if __name__ == "__main__":
    main()