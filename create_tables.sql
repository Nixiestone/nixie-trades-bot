-- Nix Trades Telegram Bot - Database Schema
-- PostgreSQL Schema for Supabase
-- NO EMOJIS - Professional SQL only
-- 
-- Tables:
-- 1. telegram_users - User accounts and settings
-- 2. signals - Generated automated setups
-- 3. trades - Executed trades and positions
-- 4. news_events - Cached economic calendar events
-- 5. model_metrics - ML model performance tracking

-- ==================== TABLE 1: TELEGRAM USERS ====================

CREATE TABLE IF NOT EXISTS telegram_users (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT UNIQUE NOT NULL,
    username TEXT,
    first_name TEXT,
    subscription_status TEXT DEFAULT 'inactive' CHECK (subscription_status IN ('active', 'inactive')),
    trial_started_at TIMESTAMPTZ,
    timezone TEXT DEFAULT 'UTC',
    last_8am_alert_sent TIMESTAMPTZ,
    risk_percent DECIMAL(3,1) DEFAULT 1.0 CHECK (risk_percent >= 0.1 AND risk_percent <= 5.0),
    mt5_login TEXT,
    mt5_password_encrypted TEXT,
    mt5_server TEXT,
    mt5_broker_name TEXT,
    mt5_connected BOOLEAN DEFAULT FALSE,
    symbol_mappings JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for telegram_users
CREATE INDEX IF NOT EXISTS idx_telegram_users_telegram_id ON telegram_users(telegram_id);
CREATE INDEX IF NOT EXISTS idx_telegram_users_subscription_status ON telegram_users(subscription_status);
CREATE INDEX IF NOT EXISTS idx_telegram_users_timezone ON telegram_users(timezone);
CREATE INDEX IF NOT EXISTS idx_telegram_users_mt5_connected ON telegram_users(mt5_connected);

-- Comments for telegram_users
COMMENT ON TABLE telegram_users IS 'Stores Telegram user accounts, subscription status, MT5 credentials, and preferences';
COMMENT ON COLUMN telegram_users.telegram_id IS 'Unique Telegram user ID from Telegram API';
COMMENT ON COLUMN telegram_users.mt5_password_encrypted IS 'Fernet-encrypted MT5 password for security';
COMMENT ON COLUMN telegram_users.symbol_mappings IS 'JSON mapping of standard symbols to broker-specific symbols';
COMMENT ON COLUMN telegram_users.risk_percent IS 'Risk percentage per trade (0.1-5.0)';

-- ==================== TABLE 2: SIGNALS ====================

CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    signal_number INT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    setup_type TEXT NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    stop_loss DECIMAL(10,5) NOT NULL,
    take_profit_1 DECIMAL(10,5) NOT NULL,
    take_profit_2 DECIMAL(10,5) NOT NULL,
    sl_pips DECIMAL(5,1),
    tp1_pips DECIMAL(5,1),
    tp2_pips DECIMAL(5,1),
    rr_tp1 DECIMAL(4,2),
    rr_tp2 DECIMAL(4,2),
    ml_score INT CHECK (ml_score >= 0 AND ml_score <= 100),
    lstm_score INT CHECK (lstm_score >= 0 AND lstm_score <= 100),
    xgboost_score INT CHECK (xgboost_score >= 0 AND xgboost_score <= 100),
    session TEXT,
    order_type TEXT CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for signals
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_signal_number ON signals(signal_number);
CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction);

-- Comments for signals
COMMENT ON TABLE signals IS 'Stores generated automated setups from SMC strategy';
COMMENT ON COLUMN signals.signal_number IS 'Auto-incrementing signal number for user reference';
COMMENT ON COLUMN signals.setup_type IS 'Setup classification: unicorn, standard_ob, breaker, bb_continuation';
COMMENT ON COLUMN signals.ml_score IS 'Combined ML confidence score (0-100)';
COMMENT ON COLUMN signals.lstm_score IS 'LSTM model score (internal tracking only, not shown to users)';
COMMENT ON COLUMN signals.xgboost_score IS 'XGBoost model score (internal tracking only, not shown to users)';
COMMENT ON COLUMN signals.order_type IS 'Determined order type: MARKET, LIMIT, or STOP';

-- ==================== TABLE 3: TRADES ====================

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES telegram_users(id) ON DELETE CASCADE,
    signal_id INT REFERENCES signals(id) ON DELETE SET NULL,
    mt5_order_id BIGINT,
    mt5_position_ticket BIGINT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    lot_size DECIMAL(6,2),
    entry_price DECIMAL(10,5),
    stop_loss DECIMAL(10,5),
    take_profit_1 DECIMAL(10,5),
    take_profit_2 DECIMAL(10,5),
    current_price DECIMAL(10,5),
    unrealized_pnl DECIMAL(10,2),
    realized_pnl DECIMAL(10,2),
    status TEXT DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'OPEN', 'TP1_HIT', 'CLOSED', 'EXPIRED', 'CANCELLED')),
    partial_close BOOLEAN DEFAULT FALSE,
    sl_moved_to_be BOOLEAN DEFAULT FALSE,
    breakeven_sl DECIMAL(10,5),
    partial_profit_usd DECIMAL(10,2),
    total_profit_usd DECIMAL(10,2),
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    expiry_time TIMESTAMPTZ,
    broker_account TEXT,
    expected_entry DECIMAL(10,5),
    risk_amount_usd DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for trades
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_user_status ON trades(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_mt5_position_ticket ON trades(mt5_position_ticket);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at DESC);

-- Comments for trades
COMMENT ON TABLE trades IS 'Stores executed trades and their lifecycle (pending -> open -> closed)';
COMMENT ON COLUMN trades.status IS 'Trade lifecycle: PENDING (limit/stop not filled), OPEN (active), TP1_HIT (partial close), CLOSED (complete), EXPIRED (order not filled), CANCELLED';
COMMENT ON COLUMN trades.partial_close IS 'TRUE if 50% closed at TP1';
COMMENT ON COLUMN trades.sl_moved_to_be IS 'TRUE if SL moved to breakeven after TP1';
COMMENT ON COLUMN trades.breakeven_sl IS 'Breakeven SL level (entry + buffer)';

-- ==================== TABLE 4: NEWS EVENTS ====================

CREATE TABLE IF NOT EXISTS news_events (
    id SERIAL PRIMARY KEY,
    event_time_utc TIMESTAMPTZ NOT NULL,
    currency TEXT NOT NULL,
    event_name TEXT NOT NULL,
    impact TEXT NOT NULL CHECK (impact IN ('HIGH', 'MEDIUM', 'LOW')),
    forecast TEXT,
    previous TEXT,
    actual TEXT,
    source TEXT DEFAULT 'ForexFactory',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for news_events
CREATE INDEX IF NOT EXISTS idx_news_events_event_time_utc ON news_events(event_time_utc);
CREATE INDEX IF NOT EXISTS idx_news_events_impact ON news_events(impact);
CREATE INDEX IF NOT EXISTS idx_news_events_currency ON news_events(currency);
CREATE INDEX IF NOT EXISTS idx_news_events_created_at ON news_events(created_at DESC);

-- Comments for news_events
COMMENT ON TABLE news_events IS 'Cached economic calendar events from ForexFactory/Investing.com';
COMMENT ON COLUMN news_events.impact IS 'Event impact level: HIGH (red folder), MEDIUM (orange), LOW (yellow)';
COMMENT ON COLUMN news_events.event_time_utc IS 'Event time in UTC (converted from source timezone)';
COMMENT ON COLUMN news_events.actual IS 'Actual released value (updated after event occurs)';

-- ==================== TABLE 5: MODEL METRICS ====================

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value DECIMAL(5,2),
    dataset_size INT,
    training_date TIMESTAMPTZ,
    validation_accuracy DECIMAL(5,2),
    test_accuracy DECIMAL(5,2),
    precision_score DECIMAL(5,2),
    recall_score DECIMAL(5,2),
    f1_score DECIMAL(5,2),
    notes TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for model_metrics
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_timestamp ON model_metrics(model_name, timestamp DESC);

-- Comments for model_metrics
COMMENT ON TABLE model_metrics IS 'ML model performance metrics for monitoring and retraining decisions';
COMMENT ON COLUMN model_metrics.model_name IS 'Model identifier: lstm_model, xgboost_model, ensemble';
COMMENT ON COLUMN model_metrics.metric_type IS 'Metric type: accuracy, precision, recall, f1, auc, etc.';

-- ==================== HELPER FUNCTIONS ====================

-- Function to update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at columns
DROP TRIGGER IF EXISTS update_telegram_users_updated_at ON telegram_users;
CREATE TRIGGER update_telegram_users_updated_at
    BEFORE UPDATE ON telegram_users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_trades_updated_at ON trades;
CREATE TRIGGER update_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_news_events_updated_at ON news_events;
CREATE TRIGGER update_news_events_updated_at
    BEFORE UPDATE ON news_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ==================== DATA VALIDATION ====================

-- Constraint: Ensure TP1 and TP2 are in correct direction relative to entry
ALTER TABLE signals ADD CONSTRAINT check_tp_direction_buy 
    CHECK (
        (direction = 'BUY' AND take_profit_1 > entry_price AND take_profit_2 > take_profit_1)
        OR direction = 'SELL'
    );

ALTER TABLE signals ADD CONSTRAINT check_tp_direction_sell 
    CHECK (
        (direction = 'SELL' AND take_profit_1 < entry_price AND take_profit_2 < take_profit_1)
        OR direction = 'BUY'
    );

-- Constraint: Ensure SL is in correct direction relative to entry
ALTER TABLE signals ADD CONSTRAINT check_sl_direction 
    CHECK (
        (direction = 'BUY' AND stop_loss < entry_price)
        OR (direction = 'SELL' AND stop_loss > entry_price)
    );

-- ==================== INITIAL DATA ====================

-- Insert default system user for demo/testing (optional)
-- INSERT INTO telegram_users (telegram_id, username, first_name, subscription_status)
-- VALUES (0, 'system', 'System', 'active')
-- ON CONFLICT (telegram_id) DO NOTHING;

-- ==================== PERMISSIONS (if needed) ====================

-- Grant appropriate permissions to Supabase service role
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO service_role;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO service_role;

-- ==================== CLEANUP POLICY ====================

-- Create cleanup function for old data (to be called by scheduler)
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete news events older than 7 days
    DELETE FROM news_events
    WHERE event_time_utc < NOW() - INTERVAL '7 days';
    
    -- Delete model metrics older than 90 days
    DELETE FROM model_metrics
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Archive closed trades older than 30 days (optional - commented out)
    -- UPDATE trades SET archived = TRUE
    -- WHERE status = 'CLOSED' AND exit_time < NOW() - INTERVAL '30 days';
    
    RAISE NOTICE 'Cleanup completed successfully';
END;
$$ LANGUAGE plpgsql;

-- ==================== SCHEMA VERSION ====================

-- Track schema version for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INT PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description)
VALUES (1, 'Initial schema with 5 core tables')
ON CONFLICT (version) DO NOTHING;

-- ==================== END OF SCHEMA ====================

-- To execute this schema in Supabase:
-- 1. Log into your Supabase dashboard
-- 2. Go to SQL Editor
-- 3. Paste this entire file
-- 4. Click "Run" to execute
--
-- Verify tables created:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';