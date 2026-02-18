"""
NIXIE TRADES - Logging Configuration
Production-ready logging with automatic rotation and size limits.
Role: DevOps Engineer + Infrastructure Engineer

This file prevents logs from filling up disk space and crashing the bot.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import colorlog


def setup_logging(log_level: str = 'INFO'):
    """
    Configure production-grade logging with:
    - Automatic log rotation (prevents disk fill-up)
    - Size limits per log file
    - Filtered out noisy libraries (httpx, httpcore)
    - Color-coded console output for debugging
    
    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    
    # ==================== LOG FILE ROTATION ====================
    # Prevents a single log file from growing infinitely
    
    log_file = 'nixie_trades_bot.log'
    
    # Rotate when file reaches 10MB
    # Keep 5 backup files (total: 50MB max)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,               # Keep 5 old files
        encoding='utf-8'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # ==================== CONSOLE OUTPUT (COLOR-CODED) ====================
    
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    # ==================== ROOT LOGGER SETUP ====================
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # ==================== SILENCE NOISY LIBRARIES ====================
    # These libraries log EVERY HTTP request (hundreds per hour)
    # We don't need this noise in production
    
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Telegram library is also verbose - only show errors
    logging.getLogger('telegram').setLevel(logging.ERROR)
    logging.getLogger('telegram.ext').setLevel(logging.ERROR)
    
    # APScheduler logs every job execution - reduce to warnings only
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    
    # ==================== KEEP IMPORTANT LOGS ====================
    # Our own modules should log at the configured level
    
    logging.getLogger('__main__').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('database').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('mt5_connector').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('mt5_worker').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('smc_strategy').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('ml_models').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('news_fetcher').setLevel(getattr(logging, log_level.upper()))
    
    logging.info("Logging configured: Level=%s, File=%s (max 10MB, 5 backups)", log_level, log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# ==================== LOG ROTATION EXPLAINED ====================
"""
How log rotation works:

1. Bot writes to nixie_trades_bot.log
2. When file reaches 10MB, it's renamed to nixie_trades_bot.log.1
3. New logs go to fresh nixie_trades_bot.log
4. When it reaches 10MB again:
   - nixie_trades_bot.log.1 → nixie_trades_bot.log.2
   - nixie_trades_bot.log → nixie_trades_bot.log.1
   - New logs to fresh file
5. This continues up to .log.5
6. When .log.5 is created, .log.6 is deleted (oldest discarded)

RESULT: Maximum disk space used = 50MB (5 files x 10MB)

This prevents the bot from crashing due to:
- Disk full errors
- Log file growing to gigabytes
- Running out of inodes
- Slow file I/O from huge files
"""


# ==================== PRODUCTION BEST PRACTICES ====================
"""
For production deployment:

1. ✅ Use systemd or supervisor to run the bot
2. ✅ Logs rotate automatically (this file)
3. ✅ Old logs deleted automatically
4. ✅ Console output color-coded for debugging
5. ✅ Noisy libraries silenced
6. ✅ Important events still logged

Optional improvements:
- Send ERROR logs to Telegram/email
- Ship logs to centralized service (Datadog, CloudWatch)
- Add structured logging (JSON format)
- Compress old log files (.gz)
"""