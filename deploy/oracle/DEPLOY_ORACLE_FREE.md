# Oracle Free Tier Deployment

This deployment is for the bot host only.

## Put This On Oracle

- `bot.py`
- `scheduler.py`
- `smc_strategy.py`
- `ml_models.py`
- `position_monitor.py`
- `mt5_connector.py`
- `chart_generator.py`
- `news_fetcher.py`
- `database.py`
- `utils.py`
- `config.py`
- `logging_config.py`
- `models/`
- `requirements.oracle.txt`
- files under `deploy/oracle/`

## Do Not Put This On Oracle

- `mt5_worker.py`
- `requirements.mt5-worker.txt`
- `NixieTradesEA.mq5`
- Windows MT5 terminal dependencies
- `MetaTrader5`, `flask`, `flask-limiter`, `pywin32`

Keep the MT5 worker on a separate Windows VPS only, as a fallback when MetaApi is unavailable or when you choose to use it.

## Recommended Oracle Layout

- App code: `/opt/nixie-trades-bot/app`
- Virtualenv: `/opt/nixie-trades-bot/venv`
- Env file: `/etc/nixie-trades-bot/bot.env`
- Service user: `nixiebot`

## Install Steps

1. Provision one Oracle Ampere A1 instance.
2. Install Python 3.11, `git`, and build tools.
3. Create user `nixiebot`.
4. Copy the repo to `/opt/nixie-trades-bot/app`.
5. Create the venv at `/opt/nixie-trades-bot/venv`.
6. Install with:

```bash
pip install -r requirements.oracle.txt
```

7. Create `/etc/nixie-trades-bot/bot.env` with:
- Telegram bot token
- Supabase URL and key
- encryption key
- `METAAPI_TOKEN` as primary
- optional `MT5_WORKER_URL` and `MT5_WORKER_API_KEY` as fallback
- Paystack keys
- optional crypto checkout backend keys

8. Copy:
- `deploy/oracle/nixie-bot.service` to `/etc/systemd/system/`
- `deploy/oracle/nixie-bot-healthcheck.service` to `/etc/systemd/system/`
- `deploy/oracle/nixie-bot-healthcheck.timer` to `/etc/systemd/system/`

9. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nixie-bot.service
sudo systemctl enable --now nixie-bot-healthcheck.timer
```

## Security Notes

- Use `METAAPI_TOKEN` on Oracle. Do not install a local MT5 terminal there.
- Keep `/etc/nixie-trades-bot/bot.env` owned by `root:root` with `chmod 600`.
- Restrict SSH to your IP only, or use OCI Bastion.
- Do not expose the bot itself on a public HTTP port.

## Monitoring Notes

- Systemd auto-restarts the bot if the process exits.
- The health-check timer detects stale logs.
- Add OCI alarms for CPU, memory, disk, and missing custom logs.
- Ship `nixie_trades_bot.log` into OCI Logging if possible.
- Send alarms to:
  - email subscriptions via OCI Notifications
  - Slack through an HTTPS relay or webhook-compatible integration
- Recommended alarm targets:
  - instance not reachable
  - bot service inactive
  - stale bot log
  - high CPU
  - high memory
  - low disk

## Payment Note

This repo currently generates payment links but does not yet ship a dedicated webhook HTTP service for automatic payment confirmation. Keep `PAYMENT_CALLBACK_URL` pointed at the payment webhook service you deploy for Paystack.
