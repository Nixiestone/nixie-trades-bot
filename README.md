# NIX TRADES - Automated SMC Trading Bot

Enterprise-grade Telegram bot for automated Smart Money Concepts (SMC) trading with MT5 integration.

## Features

### Trading Automation
- **Smart Money Concepts (SMC)**: Order Blocks, Breaker Blocks, Fair Value Gaps, Market Structure
- **8 Precision Refinements**: Volume confirmation, ATR filters, session filters, correlation limits
- **ML Ensemble**: LSTM + XGBoost with 75%+ accuracy on trained models
- **Multi-Currency Support**: Trade ANY account currency (USD, EUR, GBP, JPY, etc.)
- **Automatic Position Management**: TP1 (50% close), Breakeven, TP2 (100% close)
- **Real-Time Monitoring**: 10-second position checks with Telegram notifications

### Risk Management
- **Dynamic Lot Sizing**: Based on account balance, risk %, and stop loss distance
- **Configurable Risk**: 0.5% - 3.0% per trade
- **Breakeven Protection**: SL moved to entry + 5 pips after TP1
- **Adaptive Risk**: Reduces risk during drawdown periods
- **News Blackout**: No trading 30 min before / 15 min after high-impact news

### News Integration
- **Triple Redundancy**: NewsAPI.org → Forex Factory → Static Calendar
- **Red Folder Events**: NFP, FOMC, CPI, GDP, Central Bank decisions
- **Daily 8 AM Alert**: Market overview + high-impact news for the day
- **Currency-Specific**: Filters news by affected currency pairs

### User Features
- **9 Telegram Commands**: /start, /help, /subscribe, /connect_mt5, /status, /latest, /settings
- **Legal Compliance**: Disclaimers, forbidden word replacement, educational language
- **Multi-User Support**: Each user connects their own MT5 account
- **Timezone Support**: Dual time display (UTC + user's local time)

## System Requirements

### Hardware
- **VPS/Server**: 2GB RAM minimum, 4GB recommended
- **OS**: Ubuntu 20.04/22.04 or Windows Server 2019+
- **Storage**: 10GB minimum
- **Network**: Stable internet connection

### Software
- **Python**: 3.10 or 3.11 (not 3.12 due to MT5 compatibility)
- **MetaTrader 5**: Installed and accessible
- **Supabase Account**: Free tier available
- **Telegram Bot Token**: From @BotFather

## Installation Guide

### Step 1: Create Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow prompts to create your bot
4. Save the **Bot Token** (looks like: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

### Step 2: Setup Supabase Database

1. Go to [https://supabase.com](https://supabase.com)
2. Create free account
3. Create new project
4. Go to **SQL Editor**
5. Copy entire contents of `create_tables.sql`
6. Paste and click **Run**
7. Go to **Settings** → **API**
8. Save **Project URL** and **anon public key**

### Step 3: Setup Server (Contabo VPS)

```bash
# Login to your Contabo VPS via SSH
ssh root@your-vps-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip -y

# Install Wine (for MT5 on Linux)
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine64 wine32 -y

# Create project directory
mkdir -p /opt/nix-trades
cd /opt/nix-trades
```

### Step 4: Install MetaTrader 5

#### On Linux (via Wine):
```bash
cd /tmp
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
wine mt5setup.exe
# Follow installation prompts
# Default installation: ~/.wine/drive_c/Program Files/MetaTrader 5/
```

#### On Windows:
1. Download MT5 from your broker
2. Install normally
3. Login with demo or live account

### Step 5: Upload Bot Files

```bash
cd /opt/nix-trades

# Upload all Python files via SCP or SFTP:
# - bot.py
# - config.py
# - utils.py
# - database.py
# - mt5_connector.py
# - smc_strategy.py
# - ml_models.py
# - position_monitor.py
# - news_fetcher.py
# - scheduler.py
# - requirements.txt
# - .env.example

# Copy via SCP (from your local machine):
scp *.py root@your-vps-ip:/opt/nix-trades/
scp requirements.txt root@your-vps-ip:/opt/nix-trades/
scp .env.example root@your-vps-ip:/opt/nix-trades/
```

### Step 6: Configure Environment Variables

```bash
cd /opt/nix-trades

# Copy template
cp .env.example .env

# Edit with your credentials
nano .env
```

Add your values:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
ENCRYPTION_KEY=generate_with_command_below
NEWS_API_KEY=optional_get_from_newsapi_org
LOG_LEVEL=INFO
```

Generate encryption key:
```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Copy output to ENCRYPTION_KEY in .env
```

### Step 7: Install Dependencies

```bash
cd /opt/nix-trades

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 8: Train ML Models (Optional but Recommended)

```bash
# Activate virtual environment if not already
source venv/bin/activate

# Create training script
cat > train_models.py << 'EOF'
from mt5_connector import MT5Connector
from ml_models import MLEnsemble

# Connect to MT5
mt5 = MT5Connector()
mt5.connect(
    login=your_demo_login,  # Replace with your demo account
    password='your_password',
    server='YourBroker-Demo'
)

# Initialize ML ensemble
ml = MLEnsemble(mt5_connector=mt5)

# Train on historical data (2020-2026)
ml.train_on_historical_data(
    symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
    start_date='2020-01-01',
    end_date='2026-02-16',
    timeframes=['H4', 'H1', 'M15']
)

# Save trained models
ml.save_models('/opt/nix-trades/models')

print("Training complete!")
EOF

# Run training
python train_models.py
```

### Step 9: Run the Bot

```bash
# Activate virtual environment
source venv/bin/activate

# Run bot
python bot.py
```

You should see:
```
INFO - Supabase client initialized successfully
INFO - Nix Trades Bot initialized
INFO - Starting Nix Trades Bot...
```

### Step 10: Setup as System Service (Auto-Start)

```bash
# Create systemd service file
sudo nano /etc/systemd/system/nix-trades.service
```

Add this content:
```ini
[Unit]
Description=Nix Trades Telegram Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/nix-trades
Environment="PATH=/opt/nix-trades/venv/bin"
ExecStart=/opt/nix-trades/venv/bin/python /opt/nix-trades/bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nix-trades
sudo systemctl start nix-trades

# Check status
sudo systemctl status nix-trades

# View logs
sudo journalctl -u nix-trades -f
```

## Testing the Bot

1. Open Telegram
2. Search for your bot: `@YourBotName`
3. Send `/start`
4. You should receive welcome message
5. Send `/help` to see all commands
6. Send `/subscribe` to activate alerts
7. Send `/connect_mt5` to link your MT5 account

### Test MT5 Connection

Format:
```
LOGIN: 12345678
PASSWORD: YourPassword
SERVER: ICMarkets-Demo
```

You should see: "MT5 Connected Successfully"

### Test Setup Alert

Wait for market scan (runs every 15 minutes) or check `/latest`

## Troubleshooting

### Bot Not Starting

```bash
# Check logs
sudo journalctl -u nix-trades -f

# Common issues:
# 1. Missing .env file
# 2. Invalid Telegram token
# 3. Supabase connection error
# 4. Python version (must be 3.10 or 3.11)
```

### MT5 Connection Failed

```bash
# On Linux, ensure Wine is working:
wine --version

# Check MT5 terminal is accessible:
ls ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/

# Test MT5 Python library:
python3 -c "import MetaTrader5 as mt5; print(mt5.version())"
```

### Database Errors

```bash
# Verify Supabase credentials:
python3 -c "
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
print('Supabase connection OK')
"
```

### News Fetching Not Working

```bash
# Test news sources:
python3 -c "
from news_fetcher import NewsFetcher
news = NewsFetcher()
events = news.get_upcoming_news(hours_ahead=24)
print(f'Found {len(events)} news events')
"
```

## Monitoring

### View Bot Logs
```bash
sudo journalctl -u nix-trades -f
```

### Check Bot Status
```bash
sudo systemctl status nix-trades
```

### Restart Bot
```bash
sudo systemctl restart nix-trades
```

### Stop Bot
```bash
sudo systemctl stop nix-trades
```

## File Structure

```
/opt/nix-trades/
├── bot.py                  # Main bot application
├── config.py               # Configuration and constants
├── utils.py                # Utility functions
├── database.py             # Supabase operations
├── mt5_connector.py        # MT5 integration
├── smc_strategy.py         # SMC trading strategy
├── ml_models.py            # LSTM + XGBoost models
├── position_monitor.py     # Real-time position tracking
├── news_fetcher.py         # News integration
├── scheduler.py            # Daily alerts and scanning
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (SECRET)
├── .env.example            # Environment template
├── create_tables.sql       # Database schema
├── venv/                   # Virtual environment
├── models/                 # Trained ML models (optional)
└── nix_trades_bot.log      # Application logs
```

## Security Best Practices

1. **Never commit .env file** to version control
2. **Use strong encryption key** (32+ characters random)
3. **Rotate MT5 passwords** regularly
4. **Enable 2FA** on Telegram
5. **Keep VPS updated**: `sudo apt update && sudo apt upgrade -y`
6. **Use firewall**: Only allow SSH (22) and outbound connections
7. **Monitor logs** for suspicious activity

## Support

- **Documentation**: Check this README
- **Logs**: `sudo journalctl -u nix-trades -f`
- **Issues**: Check error messages in logs
- **Updates**: Pull latest code from repository

## Legal Disclaimer

This bot is for educational purposes only. Trading forex, gold, and cryptocurrencies carries substantial risk of loss. Only trade with capital you can afford to lose. Past performance does not guarantee future results. You are solely responsible for all trading decisions.

## License

Proprietary - Nix Trades Limited