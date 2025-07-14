from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trading-strategy-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Trading Strategy Confluence Analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vsching/robot_analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.17.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "sqlalchemy>=2.0.0",
        "streamlit-aggrid>=0.3.4",
        "openpyxl>=3.1.0",
        "xlsxwriter>=3.1.0",
        "reportlab>=4.0.0",
        "python-dateutil>=2.8.0",
        "pytz>=2023.3",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "trading-analyzer=main:main",
        ],
    },
)