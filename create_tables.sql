-- NIX TRADES - Supabase Database Schema
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==================== TELEGRAM USERS TABLE ====================

CREATE TABLE IF NOT EXISTS telegram_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    telegram_id BIGINT UNIQUE NOT NULL,
    username TEXT,
    first_name TEXT,
    timezone TEXT DEFAULT 'UTC',
    subscription_status TEXT DEFAULT 'inactive' CHECK (subscription_status IN ('active', 'inactive')),
    risk_percent DECIMAL(4,2) DEFAULT 1.00 CHECK (risk_percent >= 0.5 AND risk_percent <= 3.0),
    mt5_login BIGINT,
    mt5_password_encrypted TEXT,
    mt5_server TEXT,
    symbol_mappings JSONB DEFAULT '{}'::jsonb,
    peak_balance DECIMAL(15,2) DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on telegram_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_telegram_users_telegram_id ON telegram_users(telegram_id);
CREATE INDEX IF NOT EXISTS idx_telegram_users_subscription ON telegram_users(subscription_status);

-- ==================== SIGNALS TABLE ====================

CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_number SERIAL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    setup_type TEXT NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    stop_loss DECIMAL(10,5) NOT NULL,
    take_profit_1 DECIMAL(10,5) NOT NULL,
    take_profit_2 DECIMAL(10,5) NOT NULL,
    sl_pips DECIMAL(8,2) NOT NULL,
    tp1_pips DECIMAL(8,2) NOT NULL,
    tp2_pips DECIMAL(8,2) NOT NULL,
    rr_tp1 DECIMAL(6,2) NOT NULL,
    rr_tp2 DECIMAL(6,2) NOT NULL,
    ml_score INTEGER NOT NULL CHECK (ml_score >= 0 AND ml_score <= 100),
    lstm_score INTEGER CHECK (lstm_score >= 0 AND lstm_score <= 100),
    xgboost_score INTEGER CHECK (xgboost_score >= 0 AND xgboost_score <= 100),
    session TEXT,
    order_type TEXT CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP')),
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'expired', 'filled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

-- ==================== TRADES TABLE ====================

CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES signals(id),
    telegram_id BIGINT REFERENCES telegram_users(telegram_id),
    ticket BIGINT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    volume DECIMAL(10,2) NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    exit_price DECIMAL(10,5),
    stop_loss DECIMAL(10,5) NOT NULL,
    take_profit_1 DECIMAL(10,5) NOT NULL,
    take_profit_2 DECIMAL(10,5) NOT NULL,
    outcome TEXT CHECK (outcome IN ('TP1_HIT', 'TP2_HIT', 'STOPPED', 'MANUAL_CLOSE')),
    profit DECIMAL(15,2),
    profit_currency TEXT DEFAULT 'USD',
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    duration_minutes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_trades_telegram_id ON trades(telegram_id);
CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades(opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);

-- ==================== AUTO-UPDATE TRIGGERS ====================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for telegram_users
DROP TRIGGER IF EXISTS update_telegram_users_updated_at ON telegram_users;
CREATE TRIGGER update_telegram_users_updated_at
    BEFORE UPDATE ON telegram_users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ==================== HELPER FUNCTIONS ====================

-- Function to get user statistics
CREATE OR REPLACE FUNCTION get_user_statistics(user_telegram_id BIGINT)
RETURNS TABLE (
    total_setups INTEGER,
    successful_trades INTEGER,
    win_rate DECIMAL(5,2),
    total_profit DECIMAL(15,2),
    avg_rr DECIMAL(6,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER AS total_setups,
        COUNT(*) FILTER (WHERE outcome IN ('TP1_HIT', 'TP2_HIT'))::INTEGER AS successful_trades,
        CASE
            WHEN COUNT(*) > 0 THEN
                (COUNT(*) FILTER (WHERE outcome IN ('TP1_HIT', 'TP2_HIT'))::DECIMAL / COUNT(*)::DECIMAL * 100)
            ELSE 0
        END AS win_rate,
        COALESCE(SUM(profit), 0) AS total_profit,
        CASE
            WHEN COUNT(*) FILTER (WHERE outcome IN ('TP1_HIT', 'TP2_HIT')) > 0 THEN
                AVG((exit_price - entry_price) / (entry_price - stop_loss)) FILTER (WHERE outcome IN ('TP1_HIT', 'TP2_HIT'))
            ELSE 0
        END AS avg_rr
    FROM trades
    WHERE telegram_id = user_telegram_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest signal
CREATE OR REPLACE FUNCTION get_latest_signal()
RETURNS TABLE (
    signal_id UUID,
    signal_number INTEGER,
    symbol TEXT,
    direction TEXT,
    setup_type TEXT,
    entry_price DECIMAL(10,5),
    stop_loss DECIMAL(10,5),
    take_profit_1 DECIMAL(10,5),
    take_profit_2 DECIMAL(10,5),
    ml_score INTEGER,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        id,
        signal_number,
        s.symbol,
        s.direction,
        s.setup_type,
        s.entry_price,
        s.stop_loss,
        s.take_profit_1,
        s.take_profit_2,
        s.ml_score,
        s.created_at
    FROM signals s
    WHERE status = 'active'
    ORDER BY created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to get all subscribed users
CREATE OR REPLACE FUNCTION get_subscribed_users()
RETURNS TABLE (
    telegram_id BIGINT,
    username TEXT,
    first_name TEXT,
    timezone TEXT,
    risk_percent DECIMAL(4,2),
    mt5_login BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        u.telegram_id,
        u.username,
        u.first_name,
        u.timezone,
        u.risk_percent,
        u.mt5_login
    FROM telegram_users u
    WHERE subscription_status = 'active';
END;
$$ LANGUAGE plpgsql;

-- ==================== ROW LEVEL SECURITY (Optional) ====================

-- Enable RLS on tables
ALTER TABLE telegram_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own data
CREATE POLICY "Users can view own data" ON telegram_users
    FOR SELECT
    USING (telegram_id = current_setting('app.current_user_id', TRUE)::BIGINT);

CREATE POLICY "Users can update own data" ON telegram_users
    FOR UPDATE
    USING (telegram_id = current_setting('app.current_user_id', TRUE)::BIGINT);

-- Policy: All users can view signals (public)
CREATE POLICY "All users can view signals" ON signals
    FOR SELECT
    USING (TRUE);

-- Policy: Users can view own trades
CREATE POLICY "Users can view own trades" ON trades
    FOR SELECT
    USING (telegram_id = current_setting('app.current_user_id', TRUE)::BIGINT);

-- ==================== GRANTS ====================

-- Grant usage to authenticated users
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- ==================== COMMENTS ====================

COMMENT ON TABLE telegram_users IS 'Stores Telegram user information and MT5 connection details';
COMMENT ON TABLE signals IS 'Stores automated trading setups generated by the system';
COMMENT ON TABLE trades IS 'Stores executed trades and their outcomes';

COMMENT ON COLUMN telegram_users.mt5_password_encrypted IS 'Encrypted MT5 password using Fernet symmetric encryption';
COMMENT ON COLUMN telegram_users.symbol_mappings IS 'JSON object mapping standard symbols to broker-specific symbols';
COMMENT ON COLUMN telegram_users.peak_balance IS 'Used for drawdown calculation';

-- ==================== INITIAL DATA ====================

-- Insert test data (optional, remove for production)
-- INSERT INTO telegram_users (telegram_id, username, first_name, timezone)
-- VALUES (123456789, 'testuser', 'Test', 'UTC');