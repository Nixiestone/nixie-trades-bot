-- NIXIE TRADES - Complete Database Schema
-- Run this ENTIRE file in Supabase SQL Editor
-- It will drop old tables and create fresh ones
-- Role: Data Engineer + Infrastructure Engineer

-- ==================== DROP OLD TABLES ====================

DROP TABLE IF EXISTS model_metrics CASCADE;
DROP TABLE IF EXISTS news_events CASCADE;
DROP TABLE IF EXISTS trades CASCADE;
DROP TABLE IF EXISTS signals CASCADE;
DROP TABLE IF EXISTS telegram_users CASCADE;

-- ==================== EXTENSION ====================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==================== TELEGRAM USERS ====================

CREATE TABLE telegram_users (
    id                      BIGSERIAL PRIMARY KEY,
    telegram_id             BIGINT      NOT NULL UNIQUE,
    username                TEXT,
    first_name              TEXT,
    timezone                TEXT        NOT NULL DEFAULT 'UTC',
    subscription_status     TEXT        NOT NULL DEFAULT 'inactive'
                                        CHECK (subscription_status IN ('inactive', 'active', 'suspended')),
    risk_percent            NUMERIC(4,2) NOT NULL DEFAULT 1.0
                                        CHECK (risk_percent >= 0.1 AND risk_percent <= 5.0),

    -- MT5 Connection Fields
    mt5_login               BIGINT,
    mt5_password_encrypted  TEXT,
    mt5_server              TEXT,
    mt5_broker_name         TEXT,
    mt5_account_balance     NUMERIC(18,2),
    mt5_account_currency    TEXT,
    mt5_connected           BOOLEAN     NOT NULL DEFAULT FALSE,
    symbol_mappings         JSONB,
    
    -- Subscription tracking
    trial_started_at        TIMESTAMPTZ,
    
    -- Timestamps
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_telegram_users_telegram_id ON telegram_users (telegram_id);
CREATE INDEX idx_telegram_users_subscription_status ON telegram_users (subscription_status);
CREATE INDEX idx_telegram_users_mt5_connected ON telegram_users (mt5_connected);

COMMENT ON TABLE telegram_users IS 'Telegram bot user accounts with encrypted MT5 credentials.';
COMMENT ON COLUMN telegram_users.mt5_password_encrypted IS 'Fernet-encrypted MT5 password.';

-- ==================== SIGNALS ====================

CREATE TABLE signals (
    id              BIGSERIAL   PRIMARY KEY,
    signal_number   INT         NOT NULL UNIQUE,
    symbol          TEXT        NOT NULL,
    direction       TEXT        NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    setup_type      TEXT        NOT NULL CHECK (setup_type IN ('UNICORN', 'STANDARD')),
    entry_price     NUMERIC(18,5) NOT NULL,
    stop_loss       NUMERIC(18,5) NOT NULL,
    take_profit_1   NUMERIC(18,5) NOT NULL,
    take_profit_2   NUMERIC(18,5) NOT NULL,
    sl_pips         NUMERIC(10,2),
    tp1_pips        NUMERIC(10,2),
    tp2_pips        NUMERIC(10,2),
    rr_tp1          NUMERIC(6,2),
    rr_tp2          NUMERIC(6,2),
    ml_score        INT         CHECK (ml_score >= 0 AND ml_score <= 100),
    lstm_score      INT         CHECK (lstm_score >= 0 AND lstm_score <= 100),
    xgboost_score   INT         CHECK (xgboost_score >= 0 AND xgboost_score <= 100),
    session         TEXT,
    order_type      TEXT        CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_created_at ON signals (created_at DESC);
CREATE INDEX idx_signals_symbol ON signals (symbol);

COMMENT ON TABLE signals IS 'Automated SMC trading setup history. Shared across all users.';

-- ==================== TRADES ====================

CREATE TABLE trades (
    id              BIGSERIAL   PRIMARY KEY,
    telegram_id     BIGINT      NOT NULL REFERENCES telegram_users(telegram_id) ON DELETE CASCADE,
    signal_id       BIGINT      REFERENCES signals(id) ON DELETE SET NULL,
    mt5_ticket      BIGINT,
    symbol          TEXT        NOT NULL,
    direction       TEXT        NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    lot_size        NUMERIC(10,2) NOT NULL,
    entry_price     NUMERIC(18,5),
    stop_loss       NUMERIC(18,5),
    take_profit_1   NUMERIC(18,5),
    take_profit_2   NUMERIC(18,5),
    order_type      TEXT        CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP')),
    status          TEXT        NOT NULL DEFAULT 'OPEN'
                                CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED', 'EXPIRED')),
    tp1_hit         BOOLEAN     NOT NULL DEFAULT FALSE,
    breakeven_set   BOOLEAN     NOT NULL DEFAULT FALSE,
    realized_pnl    NUMERIC(18,2),
    rr_achieved     NUMERIC(6,2),
    close_price     NUMERIC(18,5),
    opened_at       TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_telegram_id ON trades (telegram_id);
CREATE INDEX idx_trades_status ON trades (status);
CREATE INDEX idx_trades_mt5_ticket ON trades (mt5_ticket);
CREATE INDEX idx_trades_telegram_status ON trades (telegram_id, status);

COMMENT ON TABLE trades IS 'Individual trade executions per user.';

-- ==================== NEWS EVENTS ====================

CREATE TABLE news_events (
    id              BIGSERIAL   PRIMARY KEY,
    event_time_utc  TIMESTAMPTZ NOT NULL,
    currency        TEXT        NOT NULL,
    event_name      TEXT        NOT NULL,
    impact          TEXT        NOT NULL CHECK (impact IN ('HIGH', 'MEDIUM', 'LOW')),
    forecast        TEXT,
    previous        TEXT,
    actual          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_news_events_time ON news_events (event_time_utc);
CREATE INDEX idx_news_events_currency ON news_events (currency);
CREATE INDEX idx_news_events_impact_time ON news_events (impact, event_time_utc);

COMMENT ON TABLE news_events IS 'Cached high-impact economic calendar events.';

-- ==================== MODEL METRICS ====================

CREATE TABLE model_metrics (
    id              BIGSERIAL   PRIMARY KEY,
    model_name      TEXT        NOT NULL,
    metric_type     TEXT        NOT NULL,
    metric_value    NUMERIC(10,6) NOT NULL,
    dataset_size    INT,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_metrics_model_timestamp ON model_metrics (model_name, timestamp DESC);

COMMENT ON TABLE model_metrics IS 'ML model performance metrics.';

-- ==================== ROW LEVEL SECURITY ====================

ALTER TABLE telegram_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

CREATE POLICY telegram_users_service_only ON telegram_users
    USING (auth.role() = 'service_role');

CREATE POLICY trades_service_only ON trades
    USING (auth.role() = 'service_role');

-- ==================== MESSAGE QUEUE FOR OFFLINE USERS ====================

CREATE TABLE message_queue (
    id              BIGSERIAL   PRIMARY KEY,
    telegram_id     BIGINT      NOT NULL REFERENCES telegram_users(telegram_id) ON DELETE CASCADE,
    message_text    TEXT        NOT NULL,
    message_type    TEXT        NOT NULL CHECK (message_type IN ('SETUP_ALERT', 'TRADE_NOTIFICATION', 'SYSTEM_MESSAGE')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent_at         TIMESTAMPTZ,
    status          TEXT        NOT NULL DEFAULT 'PENDING'
                                CHECK (status IN ('PENDING', 'SENT', 'FAILED'))
);

CREATE INDEX idx_message_queue_telegram_id ON message_queue (telegram_id);
CREATE INDEX idx_message_queue_status ON message_queue (status);
CREATE INDEX idx_message_queue_created_at ON message_queue (created_at);

COMMENT ON TABLE message_queue IS 'Queued messages for offline users with timestamps.';

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'All tables created successfully!';
END $$;