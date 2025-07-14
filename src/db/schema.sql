-- Trading Strategy Analyzer Database Schema
-- SQLite Database

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(15,2) DEFAULT 0.0,
    is_active BOOLEAN DEFAULT 1
);

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
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
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    period_type TEXT CHECK(period_type IN ('daily', 'weekly', 'monthly', 'yearly', 'all-time')),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_pnl DECIMAL(15,2),
    trade_count INTEGER,
    win_count INTEGER,
    loss_count INTEGER,
    win_rate DECIMAL(5,2),
    avg_win DECIMAL(15,2),
    avg_loss DECIMAL(15,2),
    profit_factor DECIMAL(10,2),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,2),
    max_drawdown_duration_days INTEGER,
    calmar_ratio DECIMAL(10,4),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id) ON DELETE CASCADE
);

-- Confluence analysis results
CREATE TABLE IF NOT EXISTS confluence_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy1_id INTEGER NOT NULL,
    strategy2_id INTEGER NOT NULL,
    analysis_date DATE NOT NULL,
    time_window_hours INTEGER DEFAULT 24,
    overlap_count INTEGER,
    total_opportunities INTEGER,
    correlation_coefficient DECIMAL(10,4),
    confluence_strength DECIMAL(10,4),
    combined_pnl DECIMAL(15,2),
    individual_pnl_strategy1 DECIMAL(15,2),
    individual_pnl_strategy2 DECIMAL(15,2),
    performance_improvement DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy1_id) REFERENCES strategies (id) ON DELETE CASCADE,
    FOREIGN KEY (strategy2_id) REFERENCES strategies (id) ON DELETE CASCADE,
    CHECK (strategy1_id < strategy2_id)  -- Ensure unique pairs
);

-- CSV upload history
CREATE TABLE IF NOT EXISTS upload_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER,
    row_count INTEGER,
    upload_type TEXT CHECK(upload_type IN ('replace', 'append')),
    upload_status TEXT CHECK(upload_status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies (id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_trades_strategy_date ON trades(strategy_id, trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(strategy_id, pnl);
CREATE INDEX IF NOT EXISTS idx_performance_strategy_period ON performance_metrics(strategy_id, period_type, period_start);
CREATE INDEX IF NOT EXISTS idx_confluence_strategies ON confluence_analysis(strategy1_id, strategy2_id, analysis_date);
CREATE INDEX IF NOT EXISTS idx_upload_history_strategy ON upload_history(strategy_id, uploaded_at);

-- Triggers to update timestamps
CREATE TRIGGER IF NOT EXISTS update_strategy_timestamp 
AFTER UPDATE ON strategies
BEGIN
    UPDATE strategies SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- View for strategy summary
CREATE VIEW IF NOT EXISTS strategy_summary AS
SELECT 
    s.id,
    s.name,
    s.description,
    s.total_trades,
    s.total_pnl,
    s.created_at,
    s.updated_at,
    COUNT(DISTINCT t.symbol) as symbols_traded,
    MIN(t.trade_date) as first_trade_date,
    MAX(t.trade_date) as last_trade_date,
    CASE 
        WHEN s.total_trades > 0 THEN ROUND(s.total_pnl / s.total_trades, 2)
        ELSE 0
    END as avg_trade_pnl
FROM strategies s
LEFT JOIN trades t ON s.id = t.strategy_id
WHERE s.is_active = 1
GROUP BY s.id;