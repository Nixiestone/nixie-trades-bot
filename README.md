# Nixie Trades

Nixie Trades is an algorithmic trading operations platform that turns institutional-style market analysis into disciplined, automated execution and clear, plain-language alerts. It monitors instruments in real time, scores high-probability setups with a Smart Money Concepts (SMC) engine and machine-learning models, manages trades end to end, and delivers everything to users over Telegram.

> **Note:** A distributed, Kubernetes-based infrastructure layer (k3s + Kafka event streaming with in-cluster observability) is currently being built on top of this core. See the [Roadmap](#roadmap).

---

## What it does

- Real-time monitoring of configured instruments
- Setup detection via an SMC market-structure engine
- Machine-learning setup scoring (training and inference)
- Automated chart generation for alerts
- MetaTrader 5 connectivity and assisted execution
- Trade lifecycle management and user notifications
- Subscription and payment handling

## Architecture (high level)

| Component | Responsibility |
|---|---|
| `bot.py` | Telegram application entrypoint |
| `scheduler.py` | Recurring scans, alerts, and scheduled jobs |
| `smc_strategy.py` | Market-structure and setup logic |
| `ml_models.py` / `train_models.py` | Model training, loading, and scoring |
| `position_monitor.py` | Live trade and position management |
| `mt5_connector.py` / `mt5_worker.py` | MT5 execution layer |
| `database.py` | Encrypted credential storage and data access |
| `payment_handler.py` | Subscription and payment flows |

## Tech stack

Python 3.11+ · MetaTrader 5 · scikit-learn (ML) · PostgreSQL (Supabase) · Telegram Bot API · Oracle Cloud (deployment)

## Roadmap

- [ ] **Distributed infrastructure** — a four-node k3s Kubernetes cluster with Kafka (Strimzi) event streaming and in-cluster observability (Prometheus / Grafana)
- [ ] **Infrastructure as Code** — full provisioning with OpenTofu / Terraform
- [ ] **CI/CD** — automated build, test, and deploy pipeline

## Getting started

1. Create and activate a virtual environment.
2. Install dependencies — `requirements.oracle.txt` for the bot host, `requirements.mt5-worker.txt` for the MT5 worker.
3. Copy `.env.example` to `.env` and fill in your configuration.
4. Initialise the database schema (`create_tables.sql`).
5. Start the MT5 worker (if used), then run `python bot.py`.

All required configuration variables are listed in `.env.example`. **Secrets are never committed** — credentials are encrypted at rest and kept out of version control.

## Status & license

Active development. © 2026 Nixie Trades. All rights reserved. Source published for portfolio and demonstration purposes.
