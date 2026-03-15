-- ============================================================
-- NIXIE TRADES — MASTER DATABASE SETUP
-- Version: 3.0 FINAL
-- Author: Blessing Omoregie (Nixie Trades)
-- Role: Data Engineer + Infrastructure Engineer + Security Engineer
--
-- SAFE TO RE-RUN: All statements use IF NOT EXISTS or DROP IF EXISTS,
-- so you can run this again without breaking anything.
-- ============================================================


-- ==================== STEP 1: EXTENSIONS ====================
-- uuid-ossp lets us generate unique IDs automatically.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ==================== STEP 2: DROP OLD TABLES ====================
-- This clears out any old version of the schema so we start clean.
-- CASCADE means any views or dependencies are dropped too.


-- ==================== STEP 3: TELEGRAM USERS ====================
-- This table stores every person who uses the bot.
-- Their broker password is stored encrypted (never plain text).

CREATE TABLE IF NOT EXISTS telegram_users (
    id                          BIGSERIAL       PRIMARY KEY,
    telegram_id                 BIGINT          NOT NULL UNIQUE,
    username                    TEXT,
    first_name                  TEXT,
    timezone                    TEXT            NOT NULL DEFAULT 'UTC',

    -- Subscription
    subscription_status         TEXT            NOT NULL DEFAULT 'inactive'
                                                CHECK (subscription_status IN ('inactive', 'active', 'suspended')),
    trial_started_at            TIMESTAMPTZ,
    disclaimer_accepted         BOOLEAN         NOT NULL DEFAULT FALSE,
    disclaimer_accepted_at      TIMESTAMPTZ,

    -- Risk management
    risk_percent                NUMERIC(4,2)    NOT NULL DEFAULT 1.0
                                                CHECK (risk_percent >= 0.1 AND risk_percent <= 5.0),

    -- MT5 broker connection
    mt5_login                   BIGINT,
    mt5_password_encrypted      TEXT,
    mt5_server                  TEXT,
    mt5_broker_name             TEXT,
    mt5_account_balance         NUMERIC(18,2),
    mt5_account_currency        TEXT,
    mt5_connected               BOOLEAN         NOT NULL DEFAULT FALSE,
    symbol_mappings             JSONB,

    -- Timestamps
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_telegram_users_telegram_id
    ON telegram_users (telegram_id);

CREATE INDEX idx_telegram_users_subscription_status
    ON telegram_users (subscription_status);

CREATE INDEX idx_telegram_users_mt5_connected
    ON telegram_users (mt5_connected);

COMMENT ON TABLE telegram_users IS
    'Telegram bot user accounts with encrypted MT5 credentials.';

COMMENT ON COLUMN telegram_users.mt5_password_encrypted IS
    'Fernet-encrypted MT5 broker password. Never stored in plain text.';


-- ==================== STEP 4: SIGNALS ====================
-- Every trade setup the bot detects is saved here.
-- All users share this table (one setup fired to everyone subscribed).

CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL       PRIMARY KEY,
    signal_number   INT             NOT NULL UNIQUE,
    symbol          TEXT            NOT NULL,
    direction       TEXT            NOT NULL    CHECK (direction IN ('BUY', 'SELL')),
    setup_type      TEXT            NOT NULL    CHECK (setup_type IN ('UNICORN', 'STANDARD')),
    timeframe       TEXT            NOT NULL    DEFAULT 'M15',
    expiry_hours    INTEGER         NOT NULL    DEFAULT 2,
    entry_price     NUMERIC(18,5)   NOT NULL,
    stop_loss       NUMERIC(18,5)   NOT NULL,
    take_profit_1   NUMERIC(18,5)   NOT NULL,
    take_profit_2   NUMERIC(18,5)   NOT NULL,
    sl_pips         NUMERIC(10,2),
    tp1_pips        NUMERIC(10,2),
    tp2_pips        NUMERIC(10,2),
    rr_tp1          NUMERIC(6,2),
    rr_tp2          NUMERIC(6,2),
    ml_score        INT             CHECK (ml_score     BETWEEN 0 AND 100),
    lstm_score      INT             CHECK (lstm_score   BETWEEN 0 AND 100),
    xgboost_score   INT             CHECK (xgboost_score BETWEEN 0 AND 100),
    session         TEXT,
    order_type      TEXT            CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP')),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_created_at
    ON signals (created_at DESC);

CREATE INDEX idx_signals_symbol
    ON signals (symbol);

CREATE INDEX idx_signals_symbol_direction_created
    ON signals (symbol, direction, created_at DESC);

COMMENT ON TABLE signals IS
    'Automated SMC trading setup history. One record per setup, shared across all users.';


-- ==================== STEP 5: TRADES ====================
-- When a user executes a setup (manually or via auto-trade),
-- one row is created here per user per trade.

CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL       PRIMARY KEY,
    telegram_id     BIGINT          NOT NULL
                                    REFERENCES telegram_users(telegram_id)
                                    ON DELETE CASCADE,
    signal_id       BIGINT          REFERENCES signals(id) ON DELETE SET NULL,
    mt5_ticket      BIGINT,
    symbol          TEXT            NOT NULL,
    direction       TEXT            NOT NULL    CHECK (direction IN ('BUY', 'SELL')),
    lot_size        NUMERIC(10,2)   NOT NULL,
    entry_price     NUMERIC(18,5),
    fill_price      NUMERIC(18,5),
    stop_loss       NUMERIC(18,5),
    take_profit_1   NUMERIC(18,5),
    take_profit_2   NUMERIC(18,5),
    close_price     NUMERIC(18,5),
    order_type      TEXT            CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP')),
    status          TEXT            NOT NULL DEFAULT 'OPEN'
                                    CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED', 'EXPIRED')),
    outcome         TEXT            CHECK (outcome IN ('WIN', 'LOSS', 'BREAKEVEN')),
    tp1_hit         BOOLEAN         NOT NULL DEFAULT FALSE,
    breakeven_set   BOOLEAN         NOT NULL DEFAULT FALSE,
    realized_pnl    NUMERIC(18,2),
    profit_pips     NUMERIC(10,2),
    rr_achieved     NUMERIC(6,2),
    opened_at       TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_telegram_id
    ON trades (telegram_id);

CREATE INDEX idx_trades_status
    ON trades (status);

CREATE INDEX idx_trades_mt5_ticket
    ON trades (mt5_ticket);

CREATE INDEX idx_trades_telegram_status
    ON trades (telegram_id, status);

COMMENT ON TABLE trades IS
    'Individual trade executions per user. One row per user per trade.';


-- ==================== STEP 6: NEWS EVENTS ====================
-- The bot caches upcoming economic news here so it can skip
-- trading during high-impact events.

CREATE TABLE IF NOT EXISTS news_events (
    id              BIGSERIAL       PRIMARY KEY,
    event_time_utc  TIMESTAMPTZ     NOT NULL,
    currency        TEXT            NOT NULL,
    event_name      TEXT            NOT NULL,
    impact          TEXT            NOT NULL    CHECK (impact IN ('HIGH', 'MEDIUM', 'LOW')),
    forecast        TEXT,
    previous        TEXT,
    actual          TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_news_events_time
    ON news_events (event_time_utc);

CREATE INDEX idx_news_events_currency
    ON news_events (currency);

CREATE INDEX idx_news_events_impact_time
    ON news_events (impact, event_time_utc);

COMMENT ON TABLE news_events IS
    'Cached high-impact economic calendar events used for news blackout filtering.';


-- ==================== STEP 7: MESSAGE QUEUE ====================
-- When the bot needs to send a Telegram message to a user but
-- they were offline or the send failed, the message is saved here
-- and retried on the next scan.

CREATE TABLE IF NOT EXISTS message_queue (
    id              BIGSERIAL       PRIMARY KEY,
    telegram_id     BIGINT          NOT NULL
                                    REFERENCES telegram_users(telegram_id)
                                    ON DELETE CASCADE,
    message_text    TEXT            NOT NULL,
    message_type    TEXT            NOT NULL
                                    CHECK (message_type IN (
                                        'SETUP_ALERT',
                                        'TRADE_NOTIFICATION',
                                        'SYSTEM_MESSAGE'
                                    )),
    status          TEXT            NOT NULL DEFAULT 'PENDING'
                                    CHECK (status IN ('PENDING', 'SENT', 'FAILED')),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    sent_at         TIMESTAMPTZ
);

CREATE INDEX idx_message_queue_telegram_id
    ON message_queue (telegram_id);

CREATE INDEX idx_message_queue_status
    ON message_queue (status);

CREATE INDEX idx_message_queue_created_at
    ON message_queue (created_at);

COMMENT ON TABLE message_queue IS
    'Queued Telegram messages for offline users. Retried on next scheduler cycle.';


-- ==================== STEP 8: ML TRAINING DATA ====================
-- Every completed trade feeds back into this table so the machine
-- learning models can retrain on real outcomes over time.

CREATE TABLE IF NOT EXISTS ml_training_data (
    id              BIGSERIAL       PRIMARY KEY,
    mt5_ticket      BIGINT,
    features_json   TEXT            NOT NULL,
    outcome         NUMERIC(3,1)    NOT NULL,   -- 1.0 = WIN,  0.0 = LOSS
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ml_training_created
    ON ml_training_data (created_at DESC);

COMMENT ON TABLE ml_training_data IS
    'Feature vectors and outcomes used to retrain ML models on live trade results.';


-- ==================== STEP 9: MODEL METRICS ====================
-- Every time the ML models are evaluated, their accuracy scores
-- are saved here so we can track improvement over time.

CREATE TABLE IF NOT EXISTS model_metrics (
    id              BIGSERIAL       PRIMARY KEY,
    model_name      TEXT            NOT NULL,
    metric_type     TEXT            NOT NULL,
    metric_value    NUMERIC(10,6)   NOT NULL,
    dataset_size    INT,
    timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_metrics_model_timestamp
    ON model_metrics (model_name, timestamp DESC);

COMMENT ON TABLE model_metrics IS
    'ML model performance metrics tracked over time.';


-- ==================== STEP 10: QUEUED_MESSAGES VIEW ====================
-- The bot code references "queued_messages" (with an s).
-- This view sits on top of the message_queue table and translates
-- the column names the code expects.
-- security_invoker = true means the view respects RLS — it does NOT
-- bypass row-level security the way a SECURITY DEFINER view would.

CREATE VIEW public.queued_messages
WITH (security_invoker = true)
AS
SELECT
    id,
    telegram_id,
    message_text,
    message_type,
    (status = 'SENT')   AS delivered,
    created_at,
    sent_at             AS delivered_at
FROM public.message_queue;

COMMENT ON VIEW public.queued_messages IS
    'Read-only view over message_queue. security_invoker=true enforces RLS of the calling role.';


-- ==================== STEP 11: ROW LEVEL SECURITY ====================
-- RLS means the database enforces access control at the row level.
-- We lock every table so ONLY the bot server (service_role) can
-- read or write. No direct public access is possible even if someone
-- gets hold of the anon key.

ALTER TABLE telegram_users      ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals             ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades              ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_events         ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_queue       ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_training_data    ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics       ENABLE ROW LEVEL SECURITY;

-- Drop old policies before recreating (safe to re-run)
DROP POLICY IF EXISTS telegram_users_service_only   ON telegram_users;
DROP POLICY IF EXISTS signals_service_only          ON signals;
DROP POLICY IF EXISTS trades_service_only           ON trades;
DROP POLICY IF EXISTS news_events_service_only      ON news_events;
DROP POLICY IF EXISTS message_queue_service_only    ON message_queue;
DROP POLICY IF EXISTS ml_training_data_service_only ON ml_training_data;
DROP POLICY IF EXISTS model_metrics_service_only    ON model_metrics;

CREATE POLICY telegram_users_service_only   ON telegram_users
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY signals_service_only          ON signals
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY trades_service_only           ON trades
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY news_events_service_only      ON news_events
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY message_queue_service_only    ON message_queue
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY ml_training_data_service_only ON ml_training_data
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY model_metrics_service_only    ON model_metrics
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');


-- ==================== STEP 12: FUNCTION SEARCH PATH HARDENING ====================
-- This patches any existing PostgreSQL functions so they cannot be
-- tricked into using the wrong schema via a search_path injection attack.
-- Skips gracefully if a function does not exist yet.

DO $$
DECLARE
    _fn TEXT;
BEGIN
    FOREACH _fn IN ARRAY ARRAY[
        'expire_old_signals',
        'cleanup_account_snapshots',
        'update_updated_at_column'
    ]
    LOOP
        IF EXISTS (
            SELECT 1
            FROM   pg_proc p
            JOIN   pg_namespace n ON n.oid = p.pronamespace
            WHERE  n.nspname = 'public'
            AND    p.proname = _fn
        ) THEN
            EXECUTE format(
                'ALTER FUNCTION public.%I() SET search_path = public, pg_catalog',
                _fn
            );
            RAISE NOTICE 'Patched search_path on function: %', _fn;
        ELSE
            RAISE NOTICE 'Function % not found as a Postgres function - skipped (normal).', _fn;
        END IF;
    END LOOP;
END
$$;


-- ==================== STEP 13: VERIFY ====================
-- This final block prints a confirmation that everything was created.

DO $$
DECLARE
    _table_count INT;
BEGIN
    SELECT COUNT(*) INTO _table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_type   = 'BASE TABLE'
      AND table_name IN (
          'telegram_users',
          'signals',
          'trades',
          'news_events',
          'message_queue',
          'ml_training_data',
          'model_metrics'
      );

    IF _table_count = 7 THEN
        RAISE NOTICE '====================================================';
        RAISE NOTICE 'All tables, views, and policies created successfully!';
        RAISE NOTICE 'Tables confirmed: %/7', _table_count;
        RAISE NOTICE '====================================================';
    ELSE
        RAISE WARNING 'Only %/7 tables found. Re-check for errors above.', _table_count;
    END IF;
END
$$;


-- NIXIE TRADES — SIGNAL NUMBER SEQUENCE FIX
-- Run this ONCE in the Supabase SQL Editor.
-- Safe to run even if you have zero signals so far.
-- This fixes the race condition where two simultaneous signals
-- could try to save with the same signal_number and one gets lost.
--
-- PLAIN ENGLISH:
-- Before this fix, the Python code calculated the next signal number
-- by counting existing rows and adding 1. If two scans ran at the exact
-- same moment, both would count the same number and try to insert the
-- same number, causing one to crash.
-- After this fix, the database itself hands out signal numbers one at
-- a time using a counter (called a sequence) that can never give the
-- same number twice, even to simultaneous requests.

-- Step 1: Create the sequence
CREATE SEQUENCE IF NOT EXISTS signals_signal_number_seq;

-- Step 2: Set it to start from wherever your current data is
-- (if you have 10 signals already, the next one will be 11)
SELECT setval(
    'signals_signal_number_seq',
    (SELECT COALESCE(MAX(signal_number), 0) FROM signals)
);

-- Step 3: Make the signal_number column use this sequence automatically
ALTER TABLE signals
    ALTER COLUMN signal_number
    SET DEFAULT nextval('signals_signal_number_seq');

-- Verify it worked
DO $$
DECLARE
    _default_val TEXT;
BEGIN
    SELECT column_default INTO _default_val
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name   = 'signals'
      AND column_name  = 'signal_number';

    IF _default_val LIKE '%nextval%' THEN
        RAISE NOTICE 'SUCCESS: signal_number sequence is active. Race condition fixed.';
    ELSE
        RAISE WARNING 'Something went wrong. column_default = %', _default_val;
    END IF;
END
$$;

ALTER TABLE telegram_users
    ADD COLUMN IF NOT EXISTS auto_position_management BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN telegram_users.auto_position_management IS
    'When TRUE, the bot automatically manages TP1 partial close and breakeven '
    'without asking the user for confirmation. User must accept the autonomous '
    'management disclaimer before this can be enabled.';