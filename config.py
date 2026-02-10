"""
Configuration file for Nix Trades Telegram Bot
Contains constants, disclaimers, and configuration values
NO EMOJIS - Professional code only
"""

# Branding
PRODUCT_NAME = "Nix Trades"
BOT_USERNAME = "@NixTradesBot"
COMPANY_NAME = "Nix Trades Limited"
WATERMARK_TEXT = "NIX TRADES"
TAGLINE = "Smart Money, Automated Logic"
FOOTER = "Nix Trades | Smart Money, Automated Logic"
SUPPORT_CONTACT = "@Nixiestone"

# Legal Disclaimer - NO FORBIDDEN WORDS
LEGAL_DISCLAIMER = """
IMPORTANT LEGAL NOTICE - PLEASE READ CAREFULLY

This is an educational tool designed to demonstrate algorithmic trading concepts using Smart Money Concepts (SMC). By subscribing, you acknowledge and agree to the following:

1. NOT FINANCIAL ADVICE
This bot does NOT provide investment advice, recommendations, or financial guidance. All automated setups are educational demonstrations of technical analysis patterns. You are solely responsible for all trading decisions.

2. EDUCATIONAL PURPOSE ONLY
This tool is designed for learning about algorithmic trading, market structure, and order flow analysis. The historical success rates shown represent past performance of the underlying methodology and do NOT guarantee future results.

3. RISK DISCLOSURE
Trading foreign exchange (forex), contracts for difference (CFDs), and leveraged instruments carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade, you should carefully consider your investment objectives, level of experience, and risk appetite. You may lose some or all of your initial investment.

4. USER CONTROLS ACCOUNT
You maintain full control of your MetaTrader 5 account at all times. You can disconnect auto-execution, manually close trades, or modify parameters. This bot does not have custody of your funds.

5. NO GUARANTEED RETURNS
There are NO guaranteed returns, profit guarantees, or risk-free trades. Every trade carries risk. Historical performance data reflects backtested results and does not guarantee future outcomes.

6. MODEL AGREEMENT SCORES
The "Model Agreement Score" represents the level of confluence between multiple technical analysis algorithms. It is NOT a prediction, forecast, or guarantee of trade outcome. Scores are educational metrics only.

7. PAST PERFORMANCE
Past performance does not guarantee future results. Historical success rates are provided for educational reference only and do not indicate future trading outcomes.

8. THIRD-PARTY BROKER
This bot integrates with your self-selected MetaTrader 5 broker account. Nix Trades Limited is not affiliated with your broker and has no control over execution quality, spreads, slippage, or broker policies.

9. REGULATORY COMPLIANCE
You are responsible for ensuring your use of this tool complies with all applicable laws and regulations in your jurisdiction. Some jurisdictions restrict or prohibit algorithmic trading.

10. ACKNOWLEDGMENT
By clicking "I Understand and Accept," you confirm that:
- You are 18 years or older
- You understand this is an educational tool, not financial advice
- You accept full responsibility for all trading decisions
- You understand the risks involved in leveraged trading
- You will not hold Nix Trades Limited liable for any trading losses

For questions or support, contact: {support_contact}

Last updated: February 2026
"""

# Trading Sessions (UTC hours)
TRADING_SESSIONS = {
    'asian': {'start': 0, 'end': 7, 'unicorn_only': True},
    'london_open': {'start': 7, 'end': 8, 'trading_disabled': True},
    'london': {'start': 8, 'end': 16, 'all_setups': True},
    'overlap': {'start': 13, 'end': 16, 'all_setups': True},
    'newyork': {'start': 13, 'end': 21, 'all_setups': True},
    'offhours': {'start': 21, 'end': 24, 'trading_disabled': True}
}

# Supported Currency Pairs
CURRENCY_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY',
    'XAUUSD', 'XAGUSD'
]

# Pip Sizes by Symbol Type
PIP_SIZES = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001,
    'NZDUSD': 0.0001, 'USDCAD': 0.0001, 'EURGBP': 0.0001,
    'USDJPY': 0.01, 'EURJPY': 0.01, 'GBPJPY': 0.01,
    'USDCHF': 0.0001, 'XAUUSD': 0.10, 'XAGUSD': 0.01
}

# Symbol Variations for Normalization
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

# ML Configuration
ML_THRESHOLD = 60  # Minimum model agreement score (0-100)
ML_LSTM_WEIGHT = 0.7  # LSTM gets 70% weight in ensemble
ML_XGBOOST_WEIGHT = 0.3  # XGBoost gets 30% weight
ML_SEQUENCE_LENGTH = 100  # Number of candles for LSTM input
ML_RETRAINING_INTERVAL = 100  # Retrain every N setups

# Risk Management
DEFAULT_RISK_PERCENT = 1.0
MAX_RISK_PIPS = 50
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 10.0
MAX_CURRENCY_EXPOSURE = 3  # Max trades per currency

# SMC Strategy Parameters
VOLUME_THRESHOLD_OB = 1.5  # OB candle must have >= 1.5x avg volume
VOLUME_THRESHOLD_IMPULSE = 2.0  # Impulse must have >= 2.0x avg volume
INDUCEMENT_WICK_RATIO = 0.6  # Wick must be >= 60% of candle range
INDUCEMENT_MIN_PIPS = 3
INDUCEMENT_MAX_PIPS = 10
ATR_PERIOD = 14
ATR_MIN_RATIO = 0.7  # Minimum ATR ratio for valid volatility
ATR_MAX_RATIO = 2.0  # Maximum ATR ratio for valid volatility
BREAKEVEN_BUFFER_PIPS = 5  # Move SL to entry + 5 pips after TP1

# News Proximity Filter
NEWS_PROXIMITY_MINUTES = 30  # Don't trade within 30 min of high-impact news

# Order Expiration
LIMIT_ORDER_EXPIRY_MINUTES = 60
STOP_ORDER_EXPIRY_MINUTES = 60

# Order Type Detection Thresholds (in pips)
MARKET_ORDER_THRESHOLD_PIPS = 2  # <= 2 pips = market order
LIMIT_ORDER_THRESHOLD_PIPS = 20  # 3-20 pips = limit order
# > 20 pips = stop order

# Drawdown-Based Risk Adjustment
DRAWDOWN_LEVEL_1 = 3.0  # 0-3% drawdown = normal risk
DRAWDOWN_LEVEL_2 = 5.0  # 3-5% drawdown = 70% risk
DRAWDOWN_LEVEL_3 = 8.0  # 5-8% drawdown = 50% risk
# > 8% drawdown = HALT trading

RISK_MULTIPLIER_LEVEL_1 = 1.0
RISK_MULTIPLIER_LEVEL_2 = 0.7
RISK_MULTIPLIER_LEVEL_3 = 0.5
RISK_MULTIPLIER_HALT = 0.0

# Fibonacci Levels for Adaptive TP2
FIBONACCI_LEVELS = {
    'strong_trend': 1.0,    # 100% of range
    'moderate_trend': 0.618,  # 61.8% Fibonacci
    'weak_trend': 0.5       # 50% of range
}

# FORBIDDEN WORDS - These must be filtered from all user-facing messages
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
    'risk-free', 'risk free', 'no risk'
]

# Word Replacements - Map forbidden words to compliant alternatives
WORD_REPLACEMENTS = {
    'signal': 'automated setup',
    'signals': 'automated setups',
    'prediction': 'model agreement score',
    'predictions': 'model agreement scores',
    'predict': 'analyze',
    'forecast': 'historical confluence rating',
    'forecasts': 'historical confluence ratings',
    'forecasting': 'analyzing historical patterns',
    'ai prediction': 'model analysis',
    'ai predictions': 'model analyses',
    'guaranteed win': 'historical setup quality',
    'guaranteed wins': 'historical setup quality',
    'guarantee': 'historical data suggests',
    'investment advice': 'educational parameter suggestion',
    'financial advice': 'educational parameter suggestion',
    'we recommend you buy': 'setup parameters suggest long position (user has final decision)',
    'we recommend you sell': 'setup parameters suggest short position (user has final decision)',
    'you should buy': 'educational parameters indicate long (your decision)',
    'you should sell': 'educational parameters indicate short (your decision)',
    'alpha generation': 'historical edge identification',
    'generate alpha': 'identify historical edge',
    'profit guarantee': 'historical success rate (past performance does not guarantee future results)',
    'guaranteed profit': 'historical success rate (past performance does not guarantee future results)',
    'sure thing': 'high-confluence setup',
    'sure win': 'high-quality setup',
    "can't lose": 'favorable risk-reward',
    'cannot lose': 'favorable risk-reward',
    'cant lose': 'favorable risk-reward',
    'risk-free': 'risk-managed',
    'risk free': 'risk-managed',
    'no risk': 'controlled risk',
    'win rate': 'historical success rate (past performance does not guarantee future results)'
}

# Bot Messages - All professional, NO EMOJIS, NO FORBIDDEN WORDS
WELCOME_MESSAGE = """Welcome to {product_name}

Institutional-grade algorithmic trading using Smart Money Concepts (SMC) with precision refinements for high-probability forex entries.

What you get:
- Real-time automated setups via Telegram
- Smart Money analysis (Order Blocks, Breaker Blocks, Market Structure)
- Machine learning confidence scoring
- Automatic execution on MetaTrader 5
- Risk management with partial profit-taking

This is an educational tool. NOT financial advice.

Commands:
/subscribe - Start receiving automated setups
/help - Learn about features and SMC concepts
/latest - View most recent setup

Support: {support_contact}

{footer}
"""

SUBSCRIPTION_SUCCESS = """Subscription activated successfully.

You will receive automated setup alerts when market conditions align with Smart Money Concepts criteria.

Daily market briefings will arrive at 8:00 AM in your local timezone.

Want automatic trade execution?
Use /connect_mt5 to link your MetaTrader 5 broker account.

For questions: {support_contact}

{footer}
"""

HELP_MESSAGE = """NIX TRADES - HELP GUIDE

AVAILABLE COMMANDS:
/start - Welcome message and overview
/subscribe - Activate setup alerts
/help - This help guide
/connect_mt5 - Link your MT5 broker account
/disconnect_mt5 - Unlink MT5 account
/status - View subscription and trading statistics
/latest - Get most recent automated setup
/settings - Customize risk parameters and preferences
/unsubscribe - Stop receiving alerts

SETUP QUALITY LEVELS:
• Unicorn Setup: Breaker Block + Fair Value Gap overlap
  Historical success rate: 72-78% (past performance does not guarantee future results)
• Standard Setup: Order Block or Breaker Block only
  Historical success rate: 58-62% (past performance does not guarantee future results)

SMC CONCEPTS EXPLAINED:
• Order Block (OB): Last opposite candle before institutional impulse move
• Breaker Block (BB): Failed supply/demand zone now acting as support/resistance
• Fair Value Gap (FVG): Price imbalance where no trading occurred
• Break of Structure (BOS): Price breaks swing high/low in trend direction
• Market Structure Shift (MSS): Potential trend reversal breakout
• Inducement: Liquidity sweep that triggers retail traders before reversal

ORDER TYPES:
• Market Order: Entry within 2 pips of current price, executes immediately
• Limit Order: Entry 3-20 pips away, waits for pullback, expires in 1 hour
• Stop Order: Entry >20 pips away, breakout entry, expires in 1 hour

AUTO-EXECUTION FLOW:
1. Setup generated when SMC criteria align
2. Order placed on your MT5 account automatically
3. Position monitored every 10 seconds
4. TP1 hit: 50% closed, SL moved to breakeven
5. TP2 hit: Remaining 50% closed, final results

Need help? Contact {support_contact}

{footer}
"""

ALREADY_SUBSCRIBED = """You are already subscribed to Nix Trades automated setup alerts.

Use /status to view your subscription details and trading statistics.

{footer}
"""

MT5_CONNECTED_ALREADY = """You are already connected to MetaTrader 5.

Broker: {broker_name}
Account: {account_number}

Use /disconnect_mt5 if you need to change broker accounts.

{footer}
"""

MT5_CONNECTION_SUCCESS = """MetaTrader 5 connection successful.

Broker: {broker_name}
Account: {account_number}
Balance: {balance}

Automatic trade execution is now enabled.
You will receive notifications when trades are executed.

{footer}
"""

MT5_DISCONNECTION_CONFIRM = """Are you sure you want to disconnect MetaTrader 5?

This will disable automatic trade execution. You will still receive setup alerts, but trades will not execute automatically.

{footer}
"""

MT5_DISCONNECTION_SUCCESS = """MetaTrader 5 disconnected successfully.

Automatic trade execution is now disabled. You will continue to receive setup alerts.

Use /connect_mt5 to reconnect anytime.

{footer}
"""

NO_RECENT_SETUPS = """No automated setups generated in the last 24 hours.

Current market conditions have not aligned with Smart Money Concepts criteria. This is normal and part of disciplined trading.

Quality over quantity.

{footer}
"""

UNSUBSCRIBE_CONFIRM = """Are you sure you want to unsubscribe?

You will stop receiving automated setup alerts and daily market briefings.

Your account data will be retained for 30 days. Use /subscribe to reactivate anytime.

{footer}
"""

UNSUBSCRIBE_SUCCESS = """Unsubscribed successfully.

You will no longer receive automated setup alerts.

Your data remains for 30 days if you wish to reactivate.

Thank you for using Nix Trades. For feedback: {support_contact}

{footer}
"""

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 7  # Keep 7 days of logs

# Database Configuration
DB_POOL_SIZE = 10
DB_TIMEOUT_SECONDS = 30

# Scheduler Configuration
POSITION_MONITOR_INTERVAL_SECONDS = 10
MARKET_SCAN_INTERVAL_MINUTES = 15
NEWS_UPDATE_INTERVAL_MINUTES = 15
ALERT_CHECK_INTERVAL_MINUTES = 60  # Check for 8 AM alerts every hour

# HTTP Request Configuration
REQUEST_TIMEOUT_SECONDS = 10
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_FACTOR = 2  # 1s, 2s, 4s

# Encryption
ENCRYPTION_ALGORITHM = 'Fernet'  # Symmetric encryption for MT5 passwords