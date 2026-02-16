# NIXIE LOGIC - Telegram Trading Bot
## Smart Money Concepts Automated Trading Platform

**Version:** 1.0.0  
**Status:** Production-Ready  
**Author:** NIXIE LOGIC Development Team  
**Contact:** support@nixielogic.com

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Quick Start Guide](#quick-start-guide)
5. [Deployment Options](#deployment-options)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [Support](#support)

---

## 🎯 Overview

NIXIE LOGIC is a production-ready Telegram bot that delivers institutional-grade forex trading setups using Smart Money Concepts (SMC) with 8 precision refinements. The bot features:

- **Automated Setup Generation** with ML confidence scoring
- **MT5 Auto-Execution** with intelligent order type detection
- **Daily 8 AM Alerts** in user's local timezone
- **Real-time Position Management** (TP1, TP2, breakeven protection)
- **High-Impact News Integration** from ForexFactory
- **Complete Risk Management** with adaptive drawdown protection

---

## ✨ Features

### Core Features
- ✅ Smart Money Concepts strategy with 8 refinements
- ✅ Multi-timeframe analysis (Daily/4H/1H/15M)
- ✅ LSTM + XGBoost machine learning models
- ✅ Volume-weighted Order Block detection
- ✅ ATR-adjusted stop loss calculation
- ✅ Session-aware filtering
- ✅ Currency exposure limits
- ✅ Volatility regime detection
- ✅ News proximity filtering

### Telegram Commands
- `/start` - Welcome message
- `/subscribe` - Activate alerts (with legal disclaimer)
- `/help` - Complete command reference
- `/status` - View statistics and account info
- `/latest` - Get most recent setup (last 24h)
- `/connect_mt5` - Link MT5 account for auto-trading
- `/disconnect_mt5` - Unlink MT5 account
- `/settings` - Configure preferences
- `/unsubscribe` - Stop receiving alerts

### Auto-Execution Features
- 🔧 Symbol normalization (EURUSD → EURUSD.pro, XAUUSD → XAUUSDm)
- 🎯 Intelligent order type selection (Market/Limit/Stop)
- 📊 Dynamic lot size calculation
- 🛡️ Triple-redundancy error handling
- ⚡ 10-second position monitoring
- 💰 Partial profit taking (50% at TP1)
- 🔒 Automatic breakeven protection

---

## 💻 System Requirements

### Minimum Requirements
- **OS:** Windows Server 2022 (for MT5) or Ubuntu 24.04 (with Wine)
- **RAM:** 4 GB
- **CPU:** 2 vCPU
- **Storage:** 20 GB
- **Python:** 3.10+

### Recommended VPS
- **Contabo VPS S Windows:** $6.99/month (RECOMMENDED)
- **RAM:** 4 GB
- **CPU:** 2 vCPU
- **Storage:** 50 GB SSD

---

## 🚀 Quick Start Guide

### Step 1: Rent Contabo Windows VPS

1. Go to [contabo.com](https://contabo.com)
2. Select **VPS S Windows** ($6.99/mo)
3. Choose **Windows Server 2022**
4. Complete purchase
5. Check email for login credentials

### Step 2: Connect to VPS

On Windows:
```bash
# Open Remote Desktop Connection
mstsc.exe

# Enter Contabo IP address
# Login with credentials from email
```

### Step 3: Install Python 3.10

On VPS (PowerShell):
```powershell
# Download Python installer
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe" -OutFile "python-installer.exe"

# Install (add to PATH)
.\python-installer.exe /quiet InstallAllUsers=1 PrependPath=1

# Verify
python --version  # Should show Python 3.10.11
```

### Step 4: Install MT5

1. Download MT5 from your broker (XM, IC Markets, etc.)
2. Install MT5 terminal
3. Login with your account credentials
4. **Enable Algo Trading:** Tools → Options → Expert Advisors → Allow Algo Trading ✓

### Step 5: Upload Bot Files

Use **WinSCP** or **Remote Desktop file transfer**:

```powershell
# Create project directory
mkdir C:\NixieLogic
cd C:\NixieLogic

# Upload all .py files to this directory
```

### Step 6: Install Dependencies

```powershell
cd C:\NixieLogic

# Install all required packages
pip install -r requirements.txt
```

### Step 7: Set Up Supabase Database

1. Go to [supabase.com](https://supabase.com) → Sign up (free)
2. Create new project
3. Go to **SQL Editor**
4. Copy contents of `create_tables.sql`
5. Paste and click **Run**
6. Verify tables created successfully

### Step 8: Create Telegram Bot

1. Open Telegram, search **@BotFather**
2. Send `/newbot`
3. Name: `Nixie Logic`
4. Username: `NixieLogicBot` (must end in 'bot')
5. Copy token (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 9: Generate Encryption Key

```powershell
# Run this command once
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Copy the output (looks like: gAAAAABh...)
```

### Step 10: Configure Environment Variables

Create `.env` file in `C:\NixieLogic\`:

```env
TELEGRAM_BOT_TOKEN=your_token_from_botfather
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
ENCRYPTION_KEY=your_fernet_key_from_step_9
LOG_LEVEL=INFO
```

### Step 11: Test Bot Locally

```powershell
cd C:\NixieLogic

# Run bot
python bot.py

# You should see:
# NIXIE LOGIC Bot Starting - Version 1.0.0
# Scheduler started successfully
```

Test in Telegram:
1. Search for your bot (@NixieLogicBot or your username)
2. Send `/start`
3. Should receive welcome message

**If working, proceed to Step 12. Otherwise, see Troubleshooting.**

### Step 12: Set Up as Windows Service (24/7 Operation)

Download NSSM (Non-Sucking Service Manager):

```powershell
# Download NSSM
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile "nssm.zip"
Expand-Archive nssm.zip

# Install bot as service
.\nssm\win64\nssm.exe install NixieLogicBot "C:\Python310\python.exe" "C:\NixieLogic\bot.py"

# Start service
.\nssm\win64\nssm.exe start NixieLogicBot

# Service will now auto-start on VPS reboot
```

Verify service:
```powershell
# Check service status
.\nssm\win64\nssm.exe status NixieLogicBot

# Should show: SERVICE_RUNNING
```

---

## 🔧 Configuration

### User Settings

Users can configure via `/settings` command:
- Risk per trade (0.5% - 2.0%)
- Setup quality filter (All / Unicorn Only)
- Notification preferences
- Timezone

### Advanced Configuration

Edit `config.py` for:
- Trading sessions
- ML model thresholds
- Pip sizes
- Risk management parameters
- News filtering

---

## 🧪 Testing

### Pre-Deployment Testing Checklist

- [ ] All bot commands respond (`/start`, `/help`, `/status`, etc.)
- [ ] Legal disclaimer shows on `/subscribe`
- [ ] NO EMOJIS in any bot messages
- [ ] NO FORBIDDEN WORDS ("signal", "prediction", etc.)
- [ ] Timezone detection works
- [ ] MT5 connects successfully (use demo account first)
- [ ] Symbol normalization works (try EURUSD, XAUUSD)
- [ ] Order placement works (Market, Limit, Stop)
- [ ] Position monitoring works (TP1, TP2, SL)
- [ ] Daily 8 AM alert arrives in user's timezone
- [ ] News fetcher retrieves ForexFactory events

### Test with Demo Account

**CRITICAL:** Test with MT5 demo account before using real money.

1. Create demo account with broker
2. Use `/connect_mt5` to link demo account
3. Wait for automated setup alert
4. Verify trade executed correctly
5. Monitor TP1/TP2/SL management
6. Run for 7 days minimum before going live

---

## 🐛 Troubleshooting

### Bot Won't Start

**Error:** `ModuleNotFoundError: No module named 'telegram'`
```powershell
# Solution: Install dependencies
pip install -r requirements.txt
```

**Error:** `Supabase credentials not configured`
```powershell
# Solution: Check .env file exists and has correct values
cat .env  # Should show SUPABASE_URL and SUPABASE_KEY
```

### MT5 Connection Failed

**Error:** `MT5 initialization failed`
- ✅ Verify MT5 terminal is running
- ✅ Check login credentials are correct
- ✅ Verify server name matches your broker
- ✅ Ensure "Allow Algo Trading" is enabled in MT5

**Common server names:**
- XM: `XM-Global-Real`, `XM-Demo`
- IC Markets: `ICMarkets-Live`, `ICMarkets-Demo`
- FTMO: `FTMO-Server`, `FTMO-Server2`

### Symbol Not Found

**Error:** `Symbol XAUUSD not available on broker`
```python
# Symbol normalization will try variations automatically:
# XAUUSD → XAUUSDm, GOLD, GOLD.pro, etc.

# If still not working, check your broker's symbol list in MT5
# Market Watch → Right-click → Show All
```

### No Daily Alerts

**Check:**
1. Subscription is active (`/status` should show "ACTIVE")
2. Timezone is correct (`/status` shows your timezone)
3. Scheduler is running (check bot logs)
4. Check `last_8am_alert_sent` in database

### Position Not Closing at TP1

**Check:**
1. MT5 connection is active
2. Position monitoring scheduler is running
3. Check bot logs for errors
4. Verify position still exists in MT5

---

## 📞 Support

### Documentation
- Complete PRD: See project files
- SMC Strategy Guide: `SMC_STRATEGY_ANALYSIS_AND_REFINEMENTS.md`
- API Reference: Contact support

### Contact
- **Email:** support@nixielogic.com
- **Telegram:** (Coming soon)
- **Discord:** (Coming soon)

### Reporting Issues

When reporting issues, include:
1. Bot version (`config.VERSION`)
2. Error message (full stack trace)
3. Steps to reproduce
4. Reference ID (if shown in error)

---

## 📄 License

Proprietary Software - NIXIE LOGIC Limited  
Copyright © 2026 NIXIE LOGIC Limited  
All Rights Reserved

---

## ⚠️ Disclaimer

This software is an EDUCATIONAL TOOL, not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. You are solely responsible for all trading decisions.

---

**NIXIE LOGIC | Smart Money, Automated Logic**