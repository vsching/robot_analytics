# Trading Strategy Confluence Analyzer

A web-based system for uploading, managing, and analyzing multiple trading strategies to identify confluence patterns and optimize multi-strategy execution.

## 🚀 Features

- **Strategy Management**: Upload and manage multiple trading strategies via CSV files
- **Performance Analysis**: Comprehensive metrics including P&L, win rate, Sharpe ratio, and drawdown analysis
- **Confluence Detection**: Identify when multiple strategies signal simultaneously
- **Interactive Visualizations**: Real-time charts and performance comparisons
- **Export & Reporting**: Generate detailed reports in CSV, Excel, and PDF formats

## 📋 Requirements

- Python 3.9+
- 2GB+ RAM
- 1GB+ free disk space

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/vsching/robot_analytics.git
cd robot_analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚦 Quick Start

1. Run the application:
```bash
streamlit run main.py
```

2. Open your browser to `http://localhost:8501`

3. Upload your trading strategy CSV files

4. Analyze performance and detect confluence patterns

## 📁 Project Structure

```
robot_analytics/
├── src/                    # Source code
│   ├── db/                # Database operations
│   ├── models/            # Data models
│   ├── utils/             # Utility functions
│   ├── components/        # UI components
│   └── pages/             # Streamlit pages
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── config/                # Configuration files
├── data/                  # CSV uploads (gitignored)
└── main.py               # Application entry point
```

## 📊 CSV Format

The system supports multiple CSV formats including:

### TradingView Format
- date, symbol, side, qty, price, commission

### MetaTrader Format
- ticket, open_time, type, size, symbol, price, sl, tp, close_time, close_price, commission, swap, profit

### Generic Format
- date, symbol, direction, quantity, entry_price, exit_price, pnl

## 🔧 Configuration

Database and application settings can be configured in `config/settings.py`

## 📝 License

This project is proprietary software.

## 🤝 Contributing

Please read our contributing guidelines before submitting PRs.

## 📧 Support

For support, please open an issue on GitHub.