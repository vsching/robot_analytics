# Trading Strategy Analysis System - Technical Specifications

## 1. System Architecture

### 1.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Upload        │   Analysis      │   Confluence            │
│   Management    │   Engine        │   Detection             │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                 Data Processing Layer                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│   CSV Processor │   Analytics     │   Visualization         │
│                 │   Calculator    │   Generator             │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Data Storage Layer                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│   SQLite        │   File System   │   Session State         │
│   Database      │   Cache         │   Management            │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### 1.2 Technology Stack
**Frontend Framework**: Streamlit 1.28+
**Backend Language**: Python 3.9+
**Database**: SQLite 3.0+ (primary), PostgreSQL 13+ (future)
**Data Processing**: Pandas 2.0+, NumPy 1.24+
**Visualization**: Plotly 5.0+, Streamlit-aggrid 0.3+
**Analytics**: SciPy 1.10+, Scikit-learn 1.3+

## 2. Database Design

### 2.1 Database Schema
```sql
-- Strategies table
CREATE TABLE strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(15,2) DEFAULT 0.0
);

-- Trades table
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT CHECK(side IN ('long', 'short', 'buy', 'sell')),
    entry_price DECIMAL(15,4),
    exit_price DECIMAL(15,4),
    quantity DECIMAL(15,4),
    pnl DECIMAL(15,2),
    commission DECIMAL(15,2) DEFAULT 0.0,
    entry_time TIME,
    exit_time TIME,
    duration_hours DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id) ON DELETE CASCADE
);

-- Performance metrics cache
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    period_type TEXT CHECK(period_type IN ('daily', 'weekly', 'monthly', 'yearly')),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_pnl DECIMAL(15,2),
    trade_count INTEGER,
    win_count INTEGER,
    win_rate DECIMAL(5,2),
    avg_win DECIMAL(15,2),
    avg_loss DECIMAL(15,2),
    profit_factor DECIMAL(10,2),
    max_drawdown DECIMAL(10,2),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id) ON DELETE CASCADE
);

-- Confluence analysis results
CREATE TABLE confluence_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy1_id INTEGER NOT NULL,
    strategy2_id INTEGER NOT NULL,
    analysis_date DATE NOT NULL,
    correlation_coefficient DECIMAL(10,4),
    overlap_count INTEGER,
    confluence_strength DECIMAL(10,4),
    combined_pnl DECIMAL(15,2),
    individual_pnl DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy1_id) REFERENCES strategies (id) ON DELETE CASCADE,
    FOREIGN KEY (strategy2_id) REFERENCES strategies (id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_trades_strategy_date ON trades(strategy_id, trade_date);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_performance_strategy_period ON performance_metrics(strategy_id, period_type, period_start);
```

### 2.2 Data Access Layer
```python
# Database connection and ORM
class DatabaseManager:
    def __init__(self, db_path: str = "trading_analyzer.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        
    def create_tables(self):
        """Initialize database schema"""
        
    def get_strategies(self) -> List[Strategy]:
        """Retrieve all strategies"""
        
    def create_strategy(self, name: str, description: str = None) -> int:
        """Create new strategy, return strategy_id"""
        
    def insert_trades(self, strategy_id: int, trades_df: pd.DataFrame):
        """Insert trades for a strategy"""
        
    def get_trades(self, strategy_id: int, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """Retrieve trades for analysis"""
```

## 3. Application Structure

### 3.1 Project Directory Structure
```
trading_analyzer/
├── main.py                          # Streamlit app entry point
├── config/
│   ├── __init__.py
│   ├── settings.py                  # Configuration settings
│   └── database_config.py           # Database configuration
├── pages/
│   ├── __init__.py
│   ├── 01_📁_Strategy_Management.py  # Upload and manage strategies
│   ├── 02_📊_Performance_Analysis.py # Individual strategy analysis
│   ├── 03_🔄_Confluence_Analysis.py # Multi-strategy comparison
│   └── 04_📈_Dashboard.py           # Overview dashboard
├── utils/
│   ├── __init__.py
│   ├── database.py                  # Database operations
│   ├── csv_processor.py             # CSV handling and validation
│   ├── analytics.py                 # Performance calculations
│   ├── confluence.py                # Confluence detection algorithms
│   └── visualizations.py            # Chart generation functions
├── models/
│   ├── __init__.py
│   ├── strategy.py                  # Strategy data model
│   ├── trade.py                     # Trade data model
│   └── performance.py               # Performance metrics model
├── components/
│   ├── __init__.py
│   ├── file_upload.py               # Reusable upload component
│   ├── strategy_selector.py         # Strategy selection widget
│   └── metric_cards.py              # Performance metric displays
├── tests/
│   ├── __init__.py
│   ├── test_csv_processor.py
│   ├── test_analytics.py
│   └── test_confluence.py
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # Project documentation
└── .streamlit/
    ├── config.toml                 # Streamlit configuration
    └── secrets.toml                # Secrets (gitignored)
```

### 3.2 Core Components

#### 3.2.1 CSV Processor
```python
class CSVProcessor:
    def __init__(self):
        self.required_columns = ['date', 'symbol', 'pnl']
        self.optional_columns = ['side', 'quantity', 'entry_price', 'exit_price']
    
    def validate_csv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate CSV format and return validation results"""
        
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types"""
        
    def detect_format(self, df: pd.DataFrame) -> str:
        """Auto-detect CSV format (TradingView, MetaTrader, etc.)"""
```

#### 3.2.2 Analytics Engine
```python
class PerformanceAnalytics:
    @staticmethod
    def calculate_monthly_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly performance breakdown"""
        
    @staticmethod
    def calculate_weekly_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weekly performance breakdown"""
        
    @staticmethod
    def calculate_key_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Sharpe ratio, profit factor, max drawdown, etc."""
        
    @staticmethod
    def calculate_rolling_metrics(trades_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
```

#### 3.2.3 Confluence Detector
```python
class ConfluenceDetector:
    def __init__(self, time_window: str = '1D'):
        self.time_window = time_window
    
    def find_signal_overlaps(self, strategy1_trades: pd.DataFrame, 
                           strategy2_trades: pd.DataFrame) -> pd.DataFrame:
        """Identify overlapping trading periods"""
        
    def calculate_correlation(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate correlation between strategy returns"""
        
    def analyze_confluence_performance(self, overlapping_trades: pd.DataFrame) -> Dict:
        """Analyze performance during confluence periods"""
```

## 4. User Interface Specifications

### 4.1 Streamlit Configuration
```toml
# .streamlit/config.toml
[global]
dataFrameSerialization = "legacy"
showWarningOnDirectExecution = false

[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### 4.2 Page Layout Specifications

#### 4.2.1 Strategy Management Page
- **Header**: Upload interface with drag-and-drop zone
- **Sidebar**: Strategy list with action buttons (view, edit, delete)
- **Main Area**: 
  - Strategy overview table
  - Upload status and validation results
  - Data preview for uploaded files

#### 4.2.2 Performance Analysis Page
- **Sidebar**: Strategy selector and time range filters
- **Main Area**: 
  - Key metrics cards (4-column layout)
  - Monthly/weekly breakdown tables
  - Performance charts (cumulative returns, drawdown)
  - Trade distribution analysis

#### 4.2.3 Confluence Analysis Page
- **Sidebar**: Multi-strategy selector (checkboxes)
- **Main Area**:
  - Correlation matrix heatmap
  - Confluence timeline chart
  - Performance comparison table
  - Risk-return scatter plot

## 5. API Design

### 5.1 Internal API Functions
```python
# Strategy Management API
def create_strategy(name: str, description: str = None) -> int
def get_strategy(strategy_id: int) -> Strategy
def update_strategy(strategy_id: int, **kwargs) -> bool
def delete_strategy(strategy_id: int) -> bool
def list_strategies() -> List[Strategy]

# Trade Data API
def upload_trades(strategy_id: int, csv_data: bytes, action: str = 'replace') -> bool
def get_trades(strategy_id: int, filters: Dict = None) -> pd.DataFrame
def get_performance_metrics(strategy_id: int, period: str = 'monthly') -> pd.DataFrame

# Analysis API
def calculate_confluence(strategy_ids: List[int], time_window: str = '1D') -> Dict
def generate_comparison_report(strategy_ids: List[int]) -> Dict
def export_analysis(analysis_type: str, strategy_ids: List[int]) -> bytes
```

## 6. Performance Specifications

### 6.1 Caching Strategy
```python
# Streamlit caching for expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_strategy_data(strategy_id: int) -> pd.DataFrame:
    """Cache strategy data loading"""

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def calculate_performance_metrics(trades_df: pd.DataFrame) -> Dict:
    """Cache performance calculations"""

@st.cache_resource
def get_database_connection() -> DatabaseManager:
    """Cache database connection"""
```

### 6.2 Memory Management
- Lazy loading of large datasets
- Pagination for trade data display (1000 rows per page)
- Cleanup of temporary files after processing
- Session state optimization for large dataframes

## 7. Security Specifications

### 7.1 Input Validation
```python
def validate_csv_upload(file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    # Check file size (max 100MB)
    # Validate file extension
    # Scan for malicious content
    # Validate CSV structure

def sanitize_strategy_name(name: str) -> str:
    """Sanitize user input for strategy names"""
    # Remove special characters
    # Limit length
    # Prevent SQL injection
```

### 7.2 Data Protection
- All data stored locally by default
- No external API calls with sensitive data
- Session-based access (no persistent authentication)
- Secure file handling and cleanup

## 8. Testing Specifications

### 8.1 Unit Tests
- CSV processing validation
- Performance calculation accuracy
- Confluence detection algorithms
- Database operations

### 8.2 Integration Tests
- End-to-end file upload workflow
- Multi-strategy analysis pipeline
- Data export functionality

### 8.3 Performance Tests
- Large dataset handling (100k+ trades)
- Memory usage monitoring
- Response time validation

## 9. Deployment Specifications

### 9.1 Local Development
```bash
# Setup commands
python -m venv trading_analyzer_env
source trading_analyzer_env/bin/activate  # or trading_analyzer_env\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run main.py
```

### 9.2 Production Deployment Options
- **Streamlit Cloud**: Direct GitHub integration
- **Docker**: Containerized deployment
- **Local Server**: Python + systemd service

### 9.3 Configuration Management
- Environment-specific configuration files
- Database connection settings
- Feature flags for experimental features
- Logging configuration for debugging