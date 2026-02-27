import logging
import logging.handlers
import os
import re
from typing import Optional

import config

# ==================== CREDENTIAL SCRUBBER ====================

_SENSITIVE_PATTERNS = [
    re.compile(r'("mt5_password(?:_encrypted)?"\s*:\s*")[^"]*(")', re.IGNORECASE),
    re.compile(r'("password"\s*:\s*")[^"]*(")',                     re.IGNORECASE),
    re.compile(r"('mt5_password(?:_encrypted)?'\s*:\s*')[^']*(')", re.IGNORECASE),
    re.compile(r"('password'\s*:\s*')[^']*(')",                    re.IGNORECASE),
    re.compile(r'(password\s*=\s*)\S+',                            re.IGNORECASE),
    re.compile(r'(gAAAAA[A-Za-z0-9_\-=]{20,})',                    re.IGNORECASE),
]


class _CredentialScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            for pattern in _SENSITIVE_PATTERNS:
                msg = pattern.sub(r'\1[REDACTED]\2', msg)
            record.msg  = msg
            record.args = ()
        except Exception:
            pass
        return True


# ==================== FORMAT ====================

_FILE_FORMAT    = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
_CONSOLE_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
_DATE_FORMAT    = '%Y-%m-%d %H:%M:%S'
_TRADE_FORMAT   = '%(asctime)s | %(message)s'

_COLOR_MAP = {
    'DEBUG':    'cyan',
    'INFO':     'green',
    'WARNING':  'yellow',
    'ERROR':    'red',
    'CRITICAL': 'red,bg_white',
}

_trade_logger: Optional[logging.Logger] = None


# ==================== MAIN SETUP ====================

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Configure production-grade logging for all Nixie Trades modules.

    Sets up:
      1. Main rotating log file (10 MB per file, 5 backups = 50 MB max)
      2. Colour-coded console output
      3. Credential scrubbing on both handlers
      4. Dedicated trade history rotating log (10 MB x 5 files = 50 MB max)
      5. Silenced third-party library noise
    """
    global _trade_logger

    if log_file is None:
        log_file = os.getenv('LOG_FILE', 'nixie_trades_bot.log')

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    scrubber      = _CredentialScrubFilter()

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8',
    )
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    file_handler.addFilter(scrubber)

    try:
        import colorlog
        console_fmt = colorlog.ColoredFormatter(
            '%(log_color)s' + _CONSOLE_FORMAT + '%(reset)s',
            datefmt=_DATE_FORMAT, log_colors=_COLOR_MAP)
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(console_fmt)
    except ImportError:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    console_handler.addFilter(scrubber)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _trade_logger = _setup_trade_history_logger(scrubber)

    for noisy in ('httpx', 'httpcore', 'urllib3', 'requests', 'aiohttp'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for tg in ('telegram', 'telegram.ext', 'telegram.request'):
        logging.getLogger(tg).setLevel(logging.ERROR)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    own_modules = [
        '__main__', 'bot', 'database', 'mt5_connector', 'mt5_worker',
        'smc_strategy', 'ml_models', 'news_fetcher', 'scheduler',
        'position_monitor', 'logging_config', 'utils', 'train_models',
    ]
    for mod in own_modules:
        logging.getLogger(mod).setLevel(numeric_level)

    logging.info(
        "Logging ready. Level: %s | Main log: %s (%d MB x %d files) | "
        "Trade log: %s (%d MB x %d files = 50 MB max) | Scrubbing: active",
        log_level.upper(), log_file,
        config.LOG_FILE_MAX_BYTES // (1024 * 1024),
        config.LOG_FILE_BACKUP_COUNT + 1,
        config.TRADE_HISTORY_LOG_FILE,
        config.TRADE_HISTORY_MAX_BYTES // (1024 * 1024),
        config.TRADE_HISTORY_BACKUP_COUNT + 1,
    )


def _setup_trade_history_logger(scrubber: _CredentialScrubFilter) -> logging.Logger:
    """
    Build the dedicated trade history logger.

    Properties:
      - Writes ONLY to trade_history.log, never to console
      - Format: "2026-02-25 14:30:00 | EVENT_TYPE | field=value | ..."
      - Rotates at 10 MB, keeps 4 backups = 5 total files = 50 MB max on disk
      - propagate=False so records never appear in the main application log

    Event format written by position_monitor:
      OPENED   | ticket=12345 | symbol=EURUSD | direction=BUY | ...
      TP1_HIT  | ticket=12345 | symbol=EURUSD | pips=+15.3 | ...
      BE_SET   | ticket=12345 | symbol=EURUSD | be_price=1.08500 | ...
      TP2_HIT  | ticket=12345 | symbol=EURUSD | pips=+30.2 | outcome=WIN | ...
      STOPPED  | ticket=12345 | symbol=EURUSD | pips=-18.0 | outcome=LOSS | ...
    """
    trade_log = logging.getLogger('trade_history')
    trade_log.setLevel(logging.INFO)
    trade_log.propagate = False
    trade_log.handlers.clear()

    handler = logging.handlers.RotatingFileHandler(
        filename=config.TRADE_HISTORY_LOG_FILE,
        maxBytes=config.TRADE_HISTORY_MAX_BYTES,
        backupCount=config.TRADE_HISTORY_BACKUP_COUNT,
        encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter(_TRADE_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(scrubber)
    trade_log.addHandler(handler)
    return trade_log


# ==================== PUBLIC API ====================

def get_trade_logger() -> logging.Logger:
    """
    Return the dedicated trade history logger.

    Call from position_monitor.py to log every trade lifecycle event.
    If setup_logging() has not been called yet, initialises the logger safely.

    Example usage in position_monitor.py:
        from logging_config import get_trade_logger
        _trade_log = get_trade_logger()

        _trade_log.info(
            "OPENED | ticket=%d | symbol=%s | direction=%s | "
            "entry=%.5f | sl=%.5f | tp1=%.5f | tp2=%.5f | lots=%.2f",
            ticket, symbol, direction, entry, sl, tp1, tp2, volume
        )

    The /download Telegram command reads this file and converts it to CSV.
    """
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = _setup_trade_history_logger(_CredentialScrubFilter())
    return _trade_logger


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper: return a named logger."""
    return logging.getLogger(name)