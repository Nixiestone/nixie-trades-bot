# Nixie Trades Bot

**Private and Confidential**

This repository contains internal trading automation software owned by Nixie Trades. It is for authorised use only. Do not distribute this codebase, documentation, credentials, screenshots, logs, or derived materials outside the approved team.

## Overview

Nixie Trades Bot is a private Telegram-based trading operations system. At a high level, it:

- monitors configured instruments
- generates chart-based trade setups
- sends Telegram alerts
- supports MT5 account connectivity and assisted execution
- manages trade lifecycle events and user notifications
- stores operational state in the project database
- trains and loads internal ML models used for setup scoring

This README is intentionally brief. It does not document proprietary strategy logic, internal decision rules, infrastructure topology, or deployment architecture in detail.

## Repository Structure

Main files:

- `bot.py` - Telegram application entrypoint
- `scheduler.py` - recurring scans, alerts, and scheduled jobs
- `smc_strategy.py` - internal market structure and setup logic
- `ml_models.py` - ML training, loading, and scoring
- `position_monitor.py` - live trade and position management
- `mt5_connector.py` - MT5 / worker / execution connectivity layer
- `mt5_worker.py` - local MT5 worker service
- `database.py` - encrypted credential storage and database access
- `chart_generator.py` - chart rendering for alerts and sample images
- `train_models.py` - historical model training entrypoint
- `config.py` - project configuration
- `create_tables.sql` - database schema bootstrap

## Requirements

Minimum local requirements:

- Python 3.11+
- a valid `.env` file
- database access
- Telegram bot token
- MT5 worker access for MT5-backed flows

Depending on how the environment is configured, some deployments may also use an external execution provider. That setup is intentionally not described here.

Deployment-specific requirements files:

- `requirements.oracle.txt` for the Oracle/Linux bot host
- `requirements.mt5-worker.txt` for the separate Windows MT5 worker fallback

## Environment

Create a `.env` file with the required project secrets and connection settings.

Common variables used by this project:

- `TELEGRAM_BOT_TOKEN`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `ENCRYPTION_KEY`
- `ADMIN_USER_IDS`
- `MT5_WORKER_URL`
- `MT5_WORKER_API_KEY`
- `METAAPI_TOKEN`
- `NEWS_API_KEY`
- `LOG_LEVEL`

Notes:

- keep all secrets out of source control
- keep the encryption key backed up securely
- do not paste live credentials into issues, chats, or screenshots
- MT5 credentials must remain encrypted at rest

## Initial Setup

1. Create and activate a virtual environment.
2. Install dependencies with the requirements file for your target environment.
3. Configure the `.env` file.
4. Initialise the database schema with `create_tables.sql` if required.
5. Start the MT5 worker if your environment uses worker-based MT5 access.
6. Run model training if you need to refresh local models.

## Run Commands

Start the MT5 worker:

```powershell
pip install -r requirements.mt5-worker.txt
python mt5_worker.py
```

Train models:

```powershell
python train_models.py
```

Start the bot:

```powershell
pip install -r requirements.oracle.txt
python bot.py
```

## Operational Notes

- The bot expects the required services and secrets to be available before startup.
- If MT5-backed features are in use, confirm the MT5 worker is healthy before running scans or training.
- The bot and trainer read encrypted MT5 credentials from the database when needed.
- Model files are stored locally under `models/`.
- Logs are rotated automatically by the project logging configuration.

## Security

Treat the following as sensitive:

- `.env`
- database credentials
- encryption keys
- MT5 login details
- API tokens
- trade history exports
- internal charts and signal screenshots

Do not:

- commit secrets
- share internal strategy rules publicly
- publish full operational architecture
- expose private logs without review

## Maintenance

Routine tasks:

- restart `mt5_worker.py` after worker-side code changes
- restart `bot.py` after application code changes
- rerun `train_models.py` when refreshing models
- check logs when troubleshooting startup, database, MT5, or Telegram issues

## Support

For internal support, use the approved team channel and private maintainer workflow. Do not open public issues or publish operational details externally.

## License / Access

All rights reserved. Internal use only.
