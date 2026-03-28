import os
import sys
import logging
import signal
import time
from datetime import datetime, timezone, timedelta

# Make sure we can import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import logging_config
logging_config.setup_logging(log_level='INFO')

logger = logging.getLogger('train_models')

import config
import database as db
from mt5_connector import MT5Connector
from ml_models import MLEnsemble

_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    print("\n\nShutdown requested. Finishing current symbol and saving progress...")
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

def main():
    end_date   = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    # Calculate the training window dynamically from today backwards.
    # Two years of M15 history produces approximately 70,000 bars per symbol,
    # which is within the reliable delivery limit of all major retail MT5
    # brokers. A fixed date such as 2020 requests 200,000+ bars and causes
    # the broker to return empty responses, silently discarding the symbol.
    start_date = (datetime.now(timezone.utc) - timedelta(days=730)).strftime('%Y-%m-%d')

    logger.info("=" * 60)
    logger.info("NIXIE TRADES - ML MODEL TRAINING")
    logger.info("Training on MT5 historical data from %s to %s", start_date, end_date)
    logger.info("=" * 60)

    # ---- Step 0: Initialise database so encrypted MT5 credentials can be reused ----
    logger.info("Step 0: Initialising database for encrypted MT5 credential access ...")
    try:
        db.init_supabase()
    except Exception as e:
        logger.error(
            "Could not initialise database. train_models.py now reuses the same "
            "encrypted MT5 credentials used for auto-execution, so Supabase must be available: %s",
            e,
        )
        sys.exit(1)

    # ---- Step 1: Connect to MT5 worker ----
    logger.info("Step 1: Connecting to MT5 worker at %s ...", config.MT5_WORKER_URL)
    mt5 = MT5Connector()

    if not mt5.is_service_reachable_sync():
        if mt5._use_metaapi:
            logger.error(
                "MetaApi is not reachable. "
                "Check your METAAPI_TOKEN in .env and your internet connection.")
        else:
            logger.error(
                "MT5 worker is not reachable at %s. "
                "You must start mt5_worker.py before running train_models.py. "
                "Open a second Command Prompt window and run: python mt5_worker.py",
                mt5._worker_url)
        sys.exit(1)

    logger.info("MT5 worker is reachable.")

    # ---- Step 2: Initialise ML ensemble ----
    logger.info("Step 2: Initialising ML ensemble ...")
    ml = MLEnsemble(mt5_connector=mt5)

    if ml.models_trained:
        logger.info(
            "Trained models already exist on disk (trained at: %s, samples: %s).",
            ml.training_metadata.get('trained_at', 'unknown'),
            ml.training_metadata.get('samples', 'unknown'))

        answer = input(
            "\nModels already exist. Do you want to retrain from scratch? "
            "This will overwrite the existing models. [y/N]: ").strip().lower()

        if answer != 'y':
            logger.info("Training cancelled. Existing models are unchanged.")
            sys.exit(0)

        logger.info("Retraining from scratch as requested.")

    # ---- Step 3: Run training ----
    logger.info("Step 3: Downloading historical data and training models ...")
    logger.info(
        "Symbols: %s", ', '.join(getattr(config, 'MONITORED_SYMBOLS',
        ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'AUDUSD'])))
    logger.info("Period: %s to %s", start_date, end_date)
    logger.info("This may take 10-30 minutes. Please wait ...")

    start_time = time.time()

    success = ml.train_on_historical_data(
        symbols=getattr(config, 'MONITORED_SYMBOLS',
                        ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'AUDUSD']),
        start_date=start_date,
        end_date=end_date,
    )

    elapsed = time.time() - start_time

    # ---- Step 4: Result ----
    if success:
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE in %.1f minutes.", elapsed / 60)
        logger.info("XGBoost accuracy: %s",
                    ml.training_metadata.get('xgboost_accuracy', 'N/A'))
        logger.info("XGBoost AUC:      %s",
                    ml.training_metadata.get('xgboost_auc', 'N/A'))
        logger.info("LSTM accuracy:    %s",
                    ml.training_metadata.get('lstm_accuracy', 'N/A'))
        logger.info("LSTM AUC:         %s",
                    ml.training_metadata.get('lstm_auc', 'N/A'))
        logger.info("Training samples: %s",
                    ml.training_metadata.get('samples', 'N/A'))
        logger.info("Models saved to:  models/")
        logger.info("=" * 60)
        logger.info("You can now start the bot: python bot.py")
        logger.info("The bot will load these trained models automatically.")
    else:
        logger.error("=" * 60)
        logger.error("TRAINING FAILED.")
        logger.error(
            "Common causes:\n"
            "  1. MT5 worker not running (start mt5_worker.py first)\n"
            "  2. MT5 not logged in (use /connect_mt5 in the bot)\n"
            "  3. Broker does not provide enough historical data\n"
            "  4. Network connectivity issues\n\n"
            "Check the logs above for the specific error.")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
