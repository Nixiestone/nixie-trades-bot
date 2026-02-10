# Nix Trades Telegram Bot

**Institutional-grade algorithmic trading platform using Smart Money Concepts (SMC) with automatic execution on MetaTrader 5.**

## Overview

Nix Trades is a production-ready Telegram bot that generates high-probability forex trading setups using Smart Money Concepts and delivers them via Telegram with automatic execution capability.

### Features

- Real-time automated setup generation using SMC methodology
- Machine learning confidence scoring (LSTM + XGBoost ensemble)
- Automatic trade execution on MetaTrader 5
- Daily market briefings at 8 AM user local time
- High-impact economic news integration (ForexFactory)
- Multi-timeframe analysis with 8 precision refinements
- Risk management with partial profit-taking (TP1/TP2)
- Symbol normalization across different brokers

### Key Technologies

- Python 3.10+
- python-telegram-bot 20.x
- MetaTrader 5 integration
- PostgreSQL via Supabase
- TensorFlow 2.15 (LSTM models)
- XGBoost 2.0 (ensemble learning)
- APScheduler (background tasks)

---

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.10 or higher** installed
2. **Git** for version control
3. **MetaTrader 5 terminal** (Windows required for MT5 integration)
4. **Supabase account** (free tier available)
5. **Telegram account** to create bot via @BotFather

### System Requirements

- **Operating System**: Windows 10/11 (for MT5 terminal integration)
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Stable connection for MT5 and Telegram API

---

## Installation Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/Nixiestone/nix-trades-bot.git
cd nix-trades-bot
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Set Up Supabase Database

1. Create free account at [supabase.com](https://supabase.com)
2. Create new project
3. Go to SQL Editor in Supabase dashboard
4. Copy entire contents of `create_tables.sql`
5. Paste and execute in SQL Editor
6. Verify tables created:

```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

You should see: `telegram_users`, `signals`, `trades`, `news_events`, `model_metrics`

### Step 4: Create Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` command
3. Follow prompts to name your bot
4. Save the bot token (looks like: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

### Step 5: Configure Environment Variables

1. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

2. Edit `.env` file and fill in your credentials:

```env
TELEGRAM_BOT_TOKEN=your_actual_bot_token_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key_here
ENCRYPTION_KEY=generate_this_below
```

3. Generate encryption key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy the output and paste it as `ENCRYPTION_KEY` in `.env`

### Step 6: Install MetaTrader 5

1. Download MT5 from your broker's website (e.g., XM, IC Markets, FTMO)
2. Install MT5 terminal on Windows
3. Open MT5 and create demo account for testing
4. Note your login credentials:
   - Login number
   - Password  
   - Server name (e.g., `XM-Global-Demo`)

Optional: Add MT5 demo credentials to `.env` for market data scanning:

```env
MT5_DEMO_LOGIN=12345678
MT5_DEMO_PASSWORD=your_demo_password
MT5_DEMO_SERVER=XM-Global-Demo
```

### Step 7: Initial ML Model Training (Optional)

For production use, train ML models on historical data:

```bash
python ml_training.py
```

This will:
- Fetch 5 years historical data from MT5
- Train LSTM and XGBoost models
- Save models to `models/` directory

Note: If you skip this step, the bot will use rule-based fallback scoring (still functional).

### Step 8: Test Locally

```bash
python bot.py
```

You should see:

```
====================================================
Nix Trades Telegram Bot - Starting Up
====================================================
Database connection established successfully
ML models loaded successfully
Scheduler started successfully
====================================================
ALL SYSTEMS OPERATIONAL - BOT READY
====================================================
```

### Step 9: Test Bot Commands

Open Telegram and find your bot. Test these commands:

```
/start - Welcome message
/help - Command list
/subscribe - Activate (shows legal disclaimer)
/status - Check subscription status
```

---

## Deployment to Contabo Windows VPS

For 24/7 operation, deploy to a Windows VPS (required for MT5 terminal).

See **[DEPLOYMENT_GUIDE_CONTABO_WINDOWS.md](DEPLOYMENT_GUIDE_CONTABO_WINDOWS.md)** for detailed deployment instructions.

Quick overview:

1. Purchase Contabo Windows VPS (€10-15/month)
2. Connect via Remote Desktop (RDP)
3. Install Python 3.10+
4. Install MT5 terminal on VPS
5. Clone repository to VPS
6. Configure `.env` file
7. Run initial ML training
8. Set up Windows Service or Task Scheduler for auto-start
9. Monitor logs in `bot.log`

---

## Configuration

### User Settings (via /settings command)

- **Risk per trade**: 0.5% - 5.0% (default: 1.0%)
- **Timezone**: IANA timezone (e.g., `America/New_York`)
- **Setup quality filter**: All setups or Unicorn only

### Advanced Configuration (config.py)

- `MAX_RISK_PIPS`: Maximum stop loss distance (default: 50 pips)
- `ML_THRESHOLD`: Minimum ML score for setup approval (default: 60%)
- `TRADING_SESSIONS`: Session-specific filters
- `FIBONACCI_LEVELS`: TP2 calculations for different trend strengths

---

## Architecture

### File Structure

```
nix-trades-bot/
├── bot.py                   # Main application entry point
├── config.py                # Configuration constants
├── database.py              # Supabase database operations
├── mt5_connector.py         # MT5 integration layer
├── smc_strategy.py          # Smart Money Concepts logic
├── ml_models.py             # LSTM + XGBoost ensemble
├── ml_training.py           # Model training & retraining
├── news_fetcher.py          # ForexFactory scraper
├── scheduler.py             # Background jobs (APScheduler)
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── create_tables.sql        # Database schema
├── .env.example             # Environment variables template
├── README.md                # This file
└── models/                  # ML model files (created after training)
    ├── lstm_model.h5
    └── xgboost_model.pkl
```

### Data Flow

1. **Market Scanning** (every 15 minutes)
   - Fetch OHLCV data from MT5
   - Analyze with SMC strategy
   - Score with ML ensemble
   - Validate with 12-point checklist

2. **Setup Alert** (when conditions align)
   - Determine order type (Market/Limit/Stop)
   - Calculate risk parameters
   - Send Telegram notification to all subscribers

3. **Auto-Execution** (if user has MT5 connected)
   - Place order on MT5
   - Monitor position every 10 seconds
   - Manage TP1 (50% close + breakeven SL)
   - Manage TP2 (close remaining 50%)

4. **Daily Alerts** (8 AM user local time)
   - High-impact news from ForexFactory
   - Market session overview
   - Yesterday's performance (if applicable)

---

## SMC Strategy Overview

### Multi-Timeframe Approach

1. **Higher Timeframe (Daily/4H)**: Determine trend bias
2. **Intermediate Timeframe (1H/15M)**: Identify setup type (MSS/BOS)
3. **Entry Timeframe (15M/5M)**: Precise entry via Order Blocks

### 8 Precision Refinements

1. **Volume-Weighted OB**: Require 1.5x volume confirmation
2. **Inducement Quality**: Wick-to-body ratio validation
3. **ATR-Adjusted SL**: Dynamic stop loss based on volatility
4. **Session Filtering**: Asian (Unicorn only), no London open trades
5. **Fibonacci TP2**: Adaptive based on trend strength
6. **Currency Exposure**: Max 3 trades per currency
7. **Volatility Regime**: Skip if ATR outside 0.7-2.0x range
8. **Drawdown Protection**: Reduce risk or halt at >8% drawdown

### 12-Point Validation Checklist

Every setup must pass all 12 checks:

1. HTF trend alignment
2. Double BOS (for continuations)
3. Volume confirmation
4. Inducement quality
5. Session filter
6. Volatility regime
7. Currency exposure
8. News proximity (>30 min)
9. Risk-reward ratio (>=1.5:1)
10. ML agreement (>=60%)
11. Drawdown protection (<8%)
12. Setup quality bonus (Unicorn gets +10 points)

---

## Legal & Compliance

### Important Disclaimers

- This is an **educational tool**, NOT financial advice
- Past performance does NOT guarantee future results
- All trading carries risk of loss
- Users maintain full control of their accounts
- Nix Trades Limited is NOT a financial advisor

### Forbidden Terminology

The bot automatically filters these words in all user-facing messages:

- ❌ "Signal" → ✅ "Automated setup"
- ❌ "Prediction" → ✅ "Model agreement score"
- ❌ "Forecast" → ✅ "Historical confluence rating"
- ❌ "Guaranteed win" → ✅ "Historical setup quality"
- ❌ "Win rate" → ✅ "Historical success rate (past performance does not guarantee future results)"

See `config.py` for complete list of replacements.

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'MetaTrader5'`
**Solution**: 
```bash
pip install MetaTrader5==5.0.45
```

**Issue**: `MT5 initialization failed`
**Solution**:
- Ensure MT5 terminal is running on same machine
- On Linux: MT5 requires Wine (complex setup, Windows VPS recommended)
- Check MT5 terminal is logged into demo/real account

**Issue**: `Supabase connection failed`
**Solution**:
- Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Check internet connection
- Verify Supabase project is active (not paused)

**Issue**: `Telegram bot not responding`
**Solution**:
- Verify `TELEGRAM_BOT_TOKEN` is correct
- Check bot.py is running without errors
- Ensure bot is not blocked by user
- Check logs in `bot.log` for errors

**Issue**: `Symbol not found on broker`
**Solution**:
- Bot will prompt user for symbol mapping
- Example: If broker uses `EURUSD.pro`, map `EURUSD` -> `EURUSD.pro`
- Saved in database for future use

### Logs

Check `bot.log` for detailed error messages:

```bash
tail -f bot.log
```

### Support

For issues not covered here:
- Check documentation in code comments
- Review JSON specification document
- Contact support: @Nixiestone (Telegram)

---

## Security Best Practices

1. **Never commit `.env` file** to version control
2. **Use different credentials** for development vs production
3. **Rotate encryption key** if compromised
4. **Use demo accounts** for testing
5. **Monitor logs** for suspicious activity
6. **Keep dependencies updated**:

```bash
pip list --outdated
pip install --upgrade package_name
```

7. **Backup Supabase data** regularly (Supabase dashboard -> Database -> Backups)

---

## Performance Monitoring

### Metrics to Track

- **Model accuracy**: Validation vs live performance
- **Historical success rate**: TP hits vs SL hits (per symbol, per session)
- **Sharpe ratio**: Risk-adjusted returns
- **Max drawdown**: Peak-to-trough loss
- **Setup distribution**: Unicorn vs Standard count
- **Order fill rate**: Market vs Limit/Stop fill rates

### Dashboard (Future Enhancement)

A web dashboard for advanced analytics is planned for future releases.

---

## Roadmap

### Current Version: 1.0.0 (MVP)

✅ SMC strategy with 8 refinements  
✅ ML ensemble (LSTM + XGBoost)  
✅ MT5 auto-execution  
✅ Telegram bot with full commands  
✅ Daily alerts at 8 AM local time  
✅ ForexFactory news integration  
✅ Symbol normalization  
✅ Order type detection (Market/Limit/Stop)  

### Planned Features (v1.1+)

- 📊 Real-time chart generation with annotations
- 📈 Web dashboard for performance analytics
- 🎯 Advanced risk management (trailing stops, scaling)
- 🔔 Customizable alert filters
- 💳 Stripe integration for paid subscriptions
- 📱 Mobile app (React Native)
- 🌐 Multi-language support

---

## Contributing

This is a proprietary project for Nix Trades Limited. External contributions are not currently accepted.

---

## License

Copyright © 2026 Nix Trades Limited. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

## Contact

- **Support**: @Nixiestone (Telegram)
- **Website**: nixtrades.com (coming soon)
- **Company**: Nix Trades Limited

---

## Acknowledgments

Built with:
- python-telegram-bot by @python-telegram-bot
- MetaTrader 5 by MetaQuotes
- Supabase for database infrastructure
- TensorFlow & XGBoost for machine learning

---

**Remember**: This is an educational tool. Trade responsibly. Never risk more than you can afford to lose.