import os

# ==================== APPLICATION INFO ====================

APP_NAME         = "Nixie Trades"
BOT_USERNAME     = "@NixieTradesBot"
VERSION          = "1.0.0"
COMPANY_NAME     = "Nixie Trades Limited"
SUPPORT_EMAIL    = "support@nixietrades.com"
WEBSITE          = "nixietrades.com"
SUPPORT_CONTACT  = "@Nixiestone"
PRODUCT_NAME     = "Nixie Trades"
WATERMARK_TEXT   = "NIXIE TRADES"
TAGLINE          = "Smart Money, Automated Logic"
FOOTER           = "Nixie Trades | Educational Tool (Not Financial Advice)"

# ==================== TELEGRAM SETTINGS ====================

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
MAX_MESSAGE_LENGTH = 4096

# Admin Telegram user IDs. Only these users can download the ML setups CSV.
# Add your personal Telegram numeric ID here.
# To find your ID: message @userinfobot on Telegram.
ADMIN_USER_IDS: list = [
    int(x.strip()) for x in os.getenv('ADMIN_USER_IDS', '').split(',') if x.strip().isdigit()
]

# ==================== SUPABASE DATABASE ====================

SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')

# ==================== ENCRYPTION ====================

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', '')

# ==================== NEWS API ====================

NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# ==================== MT5 WORKER SERVICE ====================

MT5_WORKER_URL     = os.getenv('MT5_WORKER_URL', 'http://localhost:8000')
MT5_WORKER_API_KEY = os.getenv('MT5_WORKER_API_KEY', '')
MT5_TIMEOUT        = 30   # seconds for HTTP requests to MT5 Worker
MT5_RETRY_ATTEMPTS = 3
MT5_RETRY_DELAY    = 2    # seconds between retries

SYMBOL_SUFFIXES = ['.pro', '.raw', '.m', '.i', '.a', '.b', '.c', '_']

# ==================== TRADE EXECUTION ====================

MAGIC_NUMBER = 234567    # Unique identifier for all orders placed by this bot

# ==================== TRADING PARAMETERS ====================

DEFAULT_RISK_PERCENT  = 1.0
MIN_RISK_PERCENT      = 0.1
MAX_RISK_PERCENT      = 5.0
MAX_DAILY_LOSS_PERCENT = 5.0   # Stop auto-execution if daily loss exceeds this

# Risk-Reward ratios
MIN_RR_RATIO   = 1.5    # Minimum acceptable risk-reward ratio for any setup


# Fibonacci extension level for TP2 (1.618 = golden ratio)
FIB_EXTENSION_LEVEL = 1.618

# Breakeven settings
BREAKEVEN_BUFFER_PIPS = 5

# Order expiry
DEFAULT_ORDER_EXPIRY_HOURS   = 1
LIMIT_ORDER_EXPIRY_MINUTES   = 60
STOP_ORDER_EXPIRY_MINUTES    = 60

# ==================== MONITORED SYMBOLS ====================

MONITORED_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY',
    'XAUUSD', 'XAGUSD'
]

CURRENCY_PAIRS = MONITORED_SYMBOLS

PIP_SIZES = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001,
    'NZDUSD': 0.0001, 'USDCAD': 0.0001, 'EURGBP': 0.0001,
    'USDJPY': 0.01,   'EURJPY': 0.01,   'GBPJPY': 0.01,
    'USDCHF': 0.0001, 'XAUUSD': 0.10,   'XAGUSD': 0.01
}

SYMBOL_VARIATIONS = {
    'EURUSD': ['EURUSD', 'EURUSD.pro', 'EURUSD.raw', 'EURUSD-a', 'EURUSDm', 'EUR/USD'],
    'GBPUSD': ['GBPUSD', 'GBPUSD.pro', 'GBPUSD.raw', 'GBPUSD-a', 'GBPUSDm', 'GBP/USD'],
    'USDJPY': ['USDJPY', 'USDJPY.pro', 'USDJPY.raw', 'USDJPY-a', 'USDJPYm', 'USD/JPY'],
    'AUDUSD': ['AUDUSD', 'AUDUSD.pro', 'AUDUSD.raw', 'AUDUSD-a', 'AUDUSDm', 'AUD/USD'],
    'USDCAD': ['USDCAD', 'USDCAD.pro', 'USDCAD.raw', 'USDCAD-a', 'USDCADm', 'USD/CAD'],
    'NZDUSD': ['NZDUSD', 'NZDUSD.pro', 'NZDUSD.raw', 'NZDUSD-a', 'NZDUSDm', 'NZD/USD'],
    'USDCHF': ['USDCHF', 'USDCHF.pro', 'USDCHF.raw', 'USDCHF-a', 'USDCHFm', 'USD/CHF'],
    'EURGBP': ['EURGBP', 'EURGBP.pro', 'EURGBP.raw', 'EURGBP-a', 'EURGBPm', 'EUR/GBP'],
    'EURJPY': ['EURJPY', 'EURJPY.pro', 'EURJPY.raw', 'EURJPY-a', 'EURJPYm', 'EUR/JPY'],
    'GBPJPY': ['GBPJPY', 'GBPJPY.pro', 'GBPJPY.raw', 'GBPJPY-a', 'GBPJPYm', 'GBP/JPY'],
    'XAUUSD': ['XAUUSD', 'XAUUSDm', 'XAUUSD.pro', 'GOLD', 'GOLD.pro', 'GOLD-a', 'GOLDm'],
    'XAGUSD': ['XAGUSD', 'XAGUSDm', 'XAGUSD.pro', 'SILVER', 'SILVER.pro', 'SILVER-a']
}

# ==================== ML CONFIGURATION ====================

# Legacy threshold (kept for backward compatibility)
ML_THRESHOLD          = 60
ML_LSTM_WEIGHT        = 0.4   # Updated to match ensemble (XGB=60%, LSTM=40%)
ML_XGBOOST_WEIGHT     = 0.6
ML_SEQUENCE_LENGTH    = 100
ML_RETRAINING_INTERVAL = 100

# Tier thresholds: consensus_score decides which setups get sent
ML_TIER_PREMIUM        = 70   # 70%+ = Unicorn / Premium tier, can auto-execute
ML_TIER_STANDARD       = 60   # 60-69% = Standard tier, sent to subscribers
ML_TIER_DISCRETIONARY  = 55   # 55-59% = Discretionary (optional, lower confidence)

# Auto-execution threshold: only execute trades when ML agrees this strongly
ML_AUTO_EXECUTE_THRESHOLD = 70

# ==================== RISK MANAGEMENT ====================

MAX_RISK_PIPS        = 50
MIN_LOT_SIZE         = 0.01
MAX_LOT_SIZE         = 10.0
MAX_CURRENCY_EXPOSURE = 3

# ==================== SMC STRATEGY PARAMETERS ====================

VOLUME_THRESHOLD_OB      = 1.5
VOLUME_THRESHOLD_IMPULSE = 2.0
VOLUME_MULTIPLIER_OB      = VOLUME_THRESHOLD_OB       # Alias used by smc_strategy.py
VOLUME_MULTIPLIER_IMPULSE = VOLUME_THRESHOLD_IMPULSE  # Alias used by smc_strategy.py

# Inducement quality filters (used by smc_strategy.detect_inducement)
INDUCEMENT_WICK_RATIO          = 0.6    # Legacy alias
INDUCEMENT_MIN_PIPS            = 3      # Legacy alias
INDUCEMENT_MAX_PIPS            = 10     # Legacy alias
INDUCEMENT_WICK_MIN_PIPS       = 3      # Minimum pips the wick sweeps liquidity
INDUCEMENT_WICK_MAX_PIPS       = 10     # Maximum pips (larger = liquidity grab, not inducement)
INDUCEMENT_BODY_CLOSE_RATIO    = 0.6    # Candle body must be at least 60% of total range

# ATR filter parameters
ATR_PERIOD    = 14
ATR_MIN_RATIO = 0.7    # Setup rejected if ATR is below 70% of average
ATR_MAX_RATIO = 2.0    # Setup rejected if ATR is above 200% of average (too volatile)

# Confirmation candle requirements (for entry confirmation)
CONFIRMATION_BODY_RATIO = 0.6   # Body must be at least 60% of candle range

# ==================== NEWS FILTER ====================

NEWS_PROXIMITY_MINUTES = 30

# News blackout window (minutes)
NEWS_PROXIMITY_MINUTES       = 30
NEWS_BLACKOUT_BEFORE_MINUTES = 30    # Block trading 30 min before high-impact news
NEWS_BLACKOUT_AFTER_MINUTES  = 15    # Block trading 15 min after high-impact news

# Session filter switches
AVOID_ASIAN_SESSION       = True     # Only Unicorn setups allowed during Asian session
PREFER_LONDON_NY_OVERLAP  = True     # Prioritise 13:00-16:00 UTC overlap

# Correlation exposure limit
MAX_CORRELATED_POSITIONS  = 2        # Max open trades in the same currency direction

# ==================== ORDER TYPE DETECTION ====================

MARKET_ORDER_THRESHOLD_PIPS = 2
LIMIT_ORDER_THRESHOLD_PIPS  = 20

# ==================== DRAWDOWN-BASED RISK ADJUSTMENT ====================

DRAWDOWN_LEVEL_1 = 3.0
DRAWDOWN_LEVEL_2 = 5.0
DRAWDOWN_LEVEL_3 = 8.0

RISK_MULTIPLIER_LEVEL_1 = 1.0
RISK_MULTIPLIER_LEVEL_2 = 0.7
RISK_MULTIPLIER_LEVEL_3 = 0.5
RISK_MULTIPLIER_HALT    = 0.0

# ==================== FIBONACCI LEVELS ====================

FIBONACCI_LEVELS = {
    'strong_trend':   1.0,
    'moderate_trend': 0.618,
    'weak_trend':     0.5
}

USE_FIBONACCI_TP2 = True

# ==================== TRADING SESSIONS (UTC hours) ====================

TRADING_SESSIONS = {
    'asian':       {'start': 0,  'end': 7,  'unicorn_only': True},
    'london_open': {'start': 7,  'end': 8,  'trading_disabled': True},
    'london':      {'start': 8,  'end': 16, 'all_setups': True},
    'overlap':     {'start': 13, 'end': 16, 'all_setups': True},
    'newyork':     {'start': 13, 'end': 21, 'all_setups': True},
    'offhours':    {'start': 21, 'end': 24, 'trading_disabled': True}
}

# ==================== FORBIDDEN WORDS ====================

FORBIDDEN_WORDS = [
    'signal', 'signals',
    'prediction', 'predictions', 'predict',
    'forecast', 'forecasts', 'forecasting',
    'ai prediction', 'ai predictions',
    'guaranteed win', 'guaranteed wins', 'guarantee',
    'investment advice', 'financial advice',
    'we recommend you buy', 'we recommend you sell',
    'you should buy', 'you should sell',
    'alpha generation', 'generate alpha',
    'profit guarantee', 'guaranteed profit',
    'sure thing', 'sure win',
    "can't lose", 'cannot lose', 'cant lose',
    'risk-free', 'risk free', 'no risk',
    'win rate'
]

WORD_REPLACEMENTS = {
    'signal':             'setup',
    'signals':            'setups',
    'prediction':         'analysis',
    'predictions':        'analyses',
    'predict':            'analyse',
    'forecast':           'analyse',
    'forecasts':          'analyses',
    'forecasting':        'analysing',
    'ai prediction':      'model agreement score',
    'ai predictions':     'model agreement scores',
    'guaranteed win':     'historical pattern',
    'guaranteed wins':    'historical patterns',
    'guarantee':          'historical performance',
    'investment advice':  'technical analysis',
    'financial advice':   'technical analysis',
    'we recommend you buy':  'the setup indicates a long opportunity',
    'we recommend you sell': 'the setup indicates a short opportunity',
    'you should buy':     'the setup indicates a long opportunity',
    'you should sell':    'the setup indicates a short opportunity',
    'alpha generation':   'performance optimisation',
    'generate alpha':     'optimise performance',
    'profit guarantee':   'historical performance metric',
    'guaranteed profit':  'historical performance metric',
    'sure thing':         'high-probability setup',
    'sure win':           'high-probability setup',
    "can't lose":         'high-probability setup',
    'cannot lose':        'high-probability setup',
    'cant lose':          'high-probability setup',
    'risk-free':          'managed risk',
    'risk free':          'managed risk',
    'no risk':            'managed risk',
    'win rate':           'historical success rate',
}

# ==================== LOGGING CONFIGURATION ====================

LOG_LEVEL             = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT            = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
LOG_DATE_FORMAT       = '%Y-%m-%d %H:%M:%S'
LOG_FILE_MAX_BYTES    = 10 * 1024 * 1024   # 10 MB per main log file
LOG_FILE_BACKUP_COUNT = 5                   # 5 backups = 50 MB max for main log

# Dedicated trade history log (separate from main application log)
# Rotating: 5 files x 10 MB = 50 MB maximum total on disk
TRADE_HISTORY_LOG_FILE     = os.getenv('TRADE_HISTORY_LOG_FILE', 'trade_history.log')
TRADE_HISTORY_MAX_BYTES    = 10 * 1024 * 1024   # 10 MB per file
TRADE_HISTORY_BACKUP_COUNT = 4                   # 4 backups + 1 active = 5 files = 50 MB

# ==================== DATABASE CONFIGURATION ====================

DB_POOL_SIZE       = 10
DB_TIMEOUT_SECONDS = 30

# ==================== SCHEDULER CONFIGURATION ====================

POSITION_MONITOR_INTERVAL_SECONDS = 10    # How often position monitor checks MT5
POSITION_CHECK_INTERVAL_SECONDS   = 10    # Alias used by position_monitor.py
MARKET_SCAN_INTERVAL_MINUTES      = 15
NEWS_UPDATE_INTERVAL_MINUTES      = 15
ALERT_CHECK_INTERVAL_MINUTES      = 60

# ==================== HTTP REQUEST CONFIGURATION ====================

REQUEST_TIMEOUT_SECONDS = 10
REQUEST_MAX_RETRIES     = 3
REQUEST_BACKOFF_FACTOR  = 2

# ==================== ENCRYPTION ====================

ENCRYPTION_ALGORITHM = 'Fernet'

# ==================== LEGAL DISCLAIMER ====================

LEGAL_DISCLAIMER = (
    "RISK DISCLOSURE AND TERMS OF USE\n\n"
    "1. NOT FINANCIAL ADVICE\n"
    "All automated setups are educational demonstrations of technical analysis patterns. "
    "You are solely responsible for all trading decisions.\n\n"
    "2. EDUCATIONAL PURPOSE ONLY\n"
    "This tool is designed for learning about algorithmic trading, market structure, and "
    "order flow analysis. Historical success rates shown represent past performance of the "
    "underlying methodology and do NOT guarantee future results.\n\n"
    "3. RISK DISCLOSURE\n"
    "Trading forex, CFDs, and leveraged instruments carries a high level of risk and may "
    "not be suitable for all investors. You may lose some or all of your initial investment.\n\n"
    "4. USER CONTROLS ACCOUNT\n"
    "You maintain full control of your MetaTrader 5 account at all times. You can "
    "disconnect auto-execution, manually close trades, or modify parameters at any time.\n\n"
    "5. NO GUARANTEED RETURNS\n"
    "There are no guaranteed returns, no profit guarantees, and no risk-free trades. "
    "Every trade carries risk. Historical performance reflects backtested results only.\n\n"
    "6. ACKNOWLEDGMENT\n"
    "By clicking 'I Understand and Accept,' you confirm that:\n"
    "- You are 18 years or older\n"
    "- You understand this is an educational tool, not financial advice\n"
    "- You accept full responsibility for all trading decisions\n"
    "- You understand the risks of leveraged trading\n"
    f"- You will not hold {COMPANY_NAME} liable for any trading losses\n\n"
    f"For support: {SUPPORT_CONTACT}\n\n"
    f"{FOOTER}"
)

# ==================== BOT MESSAGES ====================

WELCOME_MESSAGE = (
    "Welcome to {product_name}\n\n"
    "Institutional-grade algorithmic trading using Smart Money Concepts (SMC) "
    "with precision refinements for high-probability forex entries.\n\n"
    "What you get:\n"
    "- Real-time automated setups via Telegram\n"
    "- Smart Money analysis (Order Blocks, Breaker Blocks, Market Structure)\n"
    "- Machine learning confidence scoring\n"
    "- Automatic execution on MetaTrader 5\n"
    "- Risk management with partial profit-taking\n\n"
    "This is an educational tool. NOT financial advice.\n\n"
    "Commands:\n"
    "/subscribe - Start receiving automated setups\n"
    "/help - Learn about features and SMC concepts\n"
    "/latest - View most recent automated setup\n"
    "/download - Download your trading history as CSV\n\n"
    "Support: {support_contact}\n\n"
    "{footer}"
)

ALREADY_SUBSCRIBED = (
    "You already have an active subscription.\n\n"
    "You are already receiving automated setup alerts.\n\n"
    "Use /status to see your account details.\n"
    "Use /settings to adjust your risk percentage.\n\n"
    f"{FOOTER}"
)

NO_RECENT_SETUPS = (
    "No setups have been detected in the last 24 hours.\n\n"
    "The system scans every 15 minutes. "
    "A setup will be sent automatically when conditions align.\n\n"
    f"{FOOTER}"
)

SUBSCRIBE_SUCCESS = (
    "Subscription activated successfully.\n\n"
    "You will receive automated setup alerts when market conditions align "
    "with Smart Money Concepts criteria.\n\n"
    "Daily market briefings will arrive at 8:00 AM UTC.\n"
    "Use /settings to set your timezone and risk.\n\n"
    "Want automatic trade execution?\n"
    "Use /connect_mt5 to link your MT5 broker account.\n\n"
    "For questions: {support_contact}\n\n"
    "{footer}"
)

HELP_MESSAGE = (
    "NIXIE TRADES - HELP GUIDE\n\n"
    "AVAILABLE COMMANDS:\n"
    "/start - Welcome message and overview\n"
    "/subscribe - Activate automated setup alerts\n"
    "/help - This help guide\n"
    "/connect_mt5 - Link your MT5 broker account\n"
    "/disconnect_mt5 - Unlink MT5 account\n"
    "/status - View subscription and trading statistics\n"
    "/latest - Get most recent automated setup\n"
    "/settings - Customize risk parameters and preferences\n"
    "/download - Download your trading history and ML results as CSV\n"
    "/unsubscribe - Stop receiving alerts\n\n"
    "SETUP QUALITY LEVELS:\n"
    "- Unicorn Setup: Breaker Block plus Fair Value Gap overlap (70%+ ML score)\n"
    "  Historical success rate: 72-78% (past performance only)\n"
    "- Standard Setup: Order Block or Breaker Block only (60-69% ML score)\n"
    "  Historical success rate: 58-62% (past performance only)\n\n"
    "SMC CONCEPTS EXPLAINED:\n"
    "- Order Block (OB): Last opposite candle before institutional impulse move\n"
    "- Breaker Block (BB): Failed supply/demand zone now acting as support/resistance\n"
    "- Fair Value Gap (FVG): Price imbalance where no trading occurred\n"
    "- Break of Structure (BOS): Price breaks swing high/low in trend direction\n"
    "- Market Structure Shift (MSS): Potential trend reversal breakout\n"
    "- Inducement: Liquidity sweep that triggers retail traders before reversal\n\n"
    "AUTO-EXECUTION FLOW:\n"
    "1. Setup generated when SMC criteria align\n"
    "2. Order placed on your MT5 account automatically\n"
    "3. Position monitored every 10 seconds\n"
    "4. TP1 hit: 50% closed, SL moved to breakeven\n"
    "5. TP2 hit: Remaining 50% closed\n\n"
    f"Need help? Contact {SUPPORT_CONTACT}\n\n"
    f"{FOOTER}"
)

UNSUBSCRIBE_CONFIRM = (
    "Are you sure you want to unsubscribe?\n\n"
    "You will no longer receive automated setup alerts.\n\n"
    "Your data remains for 30 days if you wish to reactivate."
)

UNSUBSCRIBE_SUCCESS = (
    "Unsubscribed successfully.\n\n"
    "You will no longer receive automated setup alerts.\n\n"
    "Your data remains for 30 days if you wish to reactivate.\n\n"
    "Thank you for using Nixie Trades. For feedback: {support_contact}\n\n"
    "{footer}"
)

ERROR_MESSAGES = {
    'not_subscribed': (
        "This feature requires an active subscription.\n\n"
        f"Use /subscribe to get started.\n\n{FOOTER}"
    ),
    'general_error': (
        "Something went wrong. Please try again in a moment.\n\n"
        f"If this continues, contact {SUPPORT_CONTACT}\n\n{FOOTER}"
    ),
    'mt5_not_connected': (
        "Your trading account is not connected.\n\n"
        f"Use /connect_mt5 to enable automatic execution.\n\n{FOOTER}"
    ),
    'invalid_format': (
        "The format you entered is not recognized.\n\n"
        f"Please follow the instructions and try again.\n\n{FOOTER}"
    ),
    'service_unavailable': (
        "The service is temporarily unavailable.\n\n"
        f"Please try again in a few minutes.\n\n{FOOTER}"
    )
}