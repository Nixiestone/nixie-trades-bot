"""
APScheduler Background Tasks
Periodic jobs for alerts, monitoring, and maintenance.

Author: Nix Trades Limited
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from database import (
    get_active_users,
    save_signal,
    get_open_trades,
    update_trade,
    save_news_event,
    log_model_metrics
)
from mt5_connector import (
    get_position_info,
    modify_position,
    close_position,
    get_current_price,
    determine_order_type
)
from smc_strategy import generate_smc_setup
from news_fetcher import (
    scrape_forexfactory,
    scrape_investing_com,
    update_actual_values,
    get_todays_news_formatted
)
from bot import NixTradesBot
from config import CURRENCY_PAIRS, TRADING_SESSIONS, SUPPORT_TELEGRAM

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler: Optional[AsyncIOScheduler] = None
bot_instance: Optional[NixTradesBot] = None


def setup_scheduler(bot: NixTradesBot) -> AsyncIOScheduler:
    """
    Setup and configure APScheduler.
    
    Args:
        bot: NixTradesBot instance
    
    Returns:
        Configured scheduler
    """
    global scheduler, bot_instance
    
    bot_instance = bot
    
    # Create scheduler
    scheduler = AsyncIOScheduler()
    
    # Add jobs
    # 1. Daily 8 AM alerts (runs every hour, checks user timezones)
    scheduler.add_job(
        check_8am_alerts,
        trigger=IntervalTrigger(hours=1),
        id='daily_alerts',
        replace_existing=True
    )
    
    # 2. Position monitoring (every 10 seconds)
    scheduler.add_job(
        monitor_positions,
        trigger=IntervalTrigger(seconds=10),
        id='position_monitoring',
        replace_existing=True
    )
    
    # 3. Market scanning for setups (every 15 minutes)
    scheduler.add_job(
        scan_market,
        trigger=IntervalTrigger(minutes=15),
        id='market_scanning',
        replace_existing=True
    )
    
    # 4. News cache update (daily at 00:01 UTC)
    scheduler.add_job(
        update_news_cache,
        trigger=CronTrigger(hour=0, minute=1),
        id='news_update',
        replace_existing=True
    )
    
    # 5. Cleanup old data (daily at midnight)
    scheduler.add_job(
        cleanup_old_data,
        trigger=CronTrigger(hour=0, minute=0),
        id='cleanup',
        replace_existing=True
    )
    
    # 6. Model retraining check (every hour)
    scheduler.add_job(
        check_retraining_trigger,
        trigger=IntervalTrigger(hours=1),
        id='retraining_check',
        replace_existing=True
    )
    
    # 7. News actual values update (every 15 minutes during trading hours)
    scheduler.add_job(
        update_news_actuals,
        trigger=IntervalTrigger(minutes=15),
        id='news_actuals_update',
        replace_existing=True
    )
    
    logger.info("Scheduler configured with 7 periodic jobs")
    return scheduler


async def check_8am_alerts():
    """
    Check and send daily 8 AM alerts to users in their local timezone.
    """
    try:
        logger.info("Checking for 8 AM alerts")
        
        # Get all active users
        users = get_active_users()
        if not users:
            logger.info("No active users for daily alerts")
            return
        
        current_utc = datetime.utcnow()
        
        for user in users:
            try:
                # Check if user has already received alert today
                last_alert = user.get("last_8am_alert_sent")
                if last_alert:
                    last_alert_date = datetime.fromisoformat(last_alert).date()
                    if last_alert_date == current_utc.date():
                        continue  # Already sent today
                
                # Convert UTC to user's local timezone
                user_timezone = user.get("timezone", "UTC")
                try:
                    user_tz = pytz.timezone(user_timezone)
                    user_local_time = current_utc.astimezone(user_tz)
                except:
                    # Invalid timezone, use UTC
                    user_local_time = current_utc
                
                # Check if it's 8 AM in user's local time
                if user_local_time.hour == 8 and user_local_time.minute < 15:
                    # Send daily alert
                    await send_daily_alert(user)
                    
                    # Update last alert timestamp
                    # Note: update_user function would be called here
                    logger.info(f"Sent daily alert to user {user['telegram_id']}")
                    
            except Exception as e:
                logger.error(f"Error processing user {user.get('telegram_id')} for daily alert: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Error in check_8am_alerts: {e}")


async def send_daily_alert(user: Dict[str, Any]):
    """
    Send daily market briefing to a user.
    
    Args:
        user: User dictionary
    """
    try:
        # Get today's news
        news_text = get_todays_news_formatted()
        
        # Get current date
        current_date = datetime.utcnow().strftime("%A, %B %d, %Y")
        
        # Get user's open positions
        open_positions = get_open_trades(user["id"])
        open_positions_text = ""
        
        if open_positions:
            open_positions_text = "\n\nOPEN POSITIONS:\n"
            for pos in open_positions[:3]:  # Limit to 3
                symbol = pos["symbol"]
                direction = pos["direction"]
                entry = pos["entry_price"]
                current = pos.get("current_price", entry)
                pnl = pos.get("unrealized_pnl", 0)
                
                open_positions_text += f"{symbol} {direction} @ {entry:.5f} (Current: {current:.5f}, P&L: ${pnl:.2f})\n"
        
        # Build alert message
        alert_message = f"""Good morning. Daily market briefing for {current_date}

{news_text}

MARKET SUMMARY:
Major pairs trading in normal ranges. USD showing moderate strength. Risk sentiment neutral.

{open_positions_text}

Educational tool. Not financial advice. Past performance does not guarantee future results.

Use /help for available commands.
For support, contact {SUPPORT_TELEGRAM}"""
        
        # Send via bot
        if bot_instance and bot_instance.application:
            from utils import validate_user_message
            clean_message = validate_user_message(alert_message)
            
            await bot_instance.application.bot.send_message(
                chat_id=user["telegram_id"],
                text=clean_message,
                parse_mode="Markdown"
            )
        
    except Exception as e:
        logger.error(f"Error sending daily alert to user {user.get('telegram_id')}: {e}")


async def monitor_positions():
    """
    Monitor open positions for TP1/TP2/SL hits and order expiry.
    Runs every 10 seconds.
    """
    try:
        # Get all open trades from database
        # Note: In production, this would query all users' open trades
        # For now, we'll track locally
        
        # Check for order expiry (LIMIT/STOP orders)
        await check_order_expiry()
        
        # Check position levels
        await check_position_levels()
        
    except Exception as e:
        logger.error(f"Error in monitor_positions: {e}")


async def check_order_expiry():
    """
    Check for expired LIMIT/STOP orders.
    """
    try:
        current_time = datetime.utcnow()
        
        # In production, query database for pending orders
        # For now, placeholder logic
        logger.debug("Checking order expiry")
        
    except Exception as e:
        logger.error(f"Error checking order expiry: {e}")


async def check_position_levels():
    """
    Check position TP1/TP2/SL levels.
    """
    try:
        # In production, iterate through all open positions
        # For now, placeholder logic
        logger.debug("Checking position levels")
        
    except Exception as e:
        logger.error(f"Error checking position levels: {e}")


async def scan_market():
    """
    Scan market for SMC setups across all currency pairs.
    Runs every 15 minutes.
    """
    try:
        logger.info("Starting market scan for SMC setups")
        
        for symbol in CURRENCY_PAIRS:
            try:
                # Generate setup for symbol
                setup = generate_smc_setup(symbol)
                
                if setup:
                    logger.info(f"Generated setup for {symbol}: {setup['direction']} at {setup['entry_price']:.5f}")
                    
                    # Save to database
                    setup_id = save_signal(setup)
                    
                    if setup_id and bot_instance:
                        # Send alert to all subscribed users
                        await bot_instance.send_setup_alert(setup)
                    
                    # Throttle between symbols
                    import asyncio
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        logger.info("Market scan completed")
        
    except Exception as e:
        logger.error(f"Error in scan_market: {e}")


async def update_news_cache():
    """
    Update news cache from ForexFactory/Investing.com.
    Runs daily at 00:01 UTC.
    """
    try:
        logger.info("Updating news cache")
        
        # Try ForexFactory first
        events = scrape_forexfactory()
        
        # Fallback to Investing.com
        if not events:
            events = scrape_investing_com()
        
        # Save events to database
        saved_count = 0
        for event in events:
            if event.get("impact") == "HIGH":
                event_id = save_news_event(event)
                if event_id:
                    saved_count += 1
        
        logger.info(f"Updated news cache with {saved_count} high-impact events")
        
    except Exception as e:
        logger.error(f"Error updating news cache: {e}")


async def cleanup_old_data():
    """
    Cleanup old data from database.
    Runs daily at midnight.
    """
    try:
        logger.info("Running data cleanup")
        
        # Calculate cutoff dates
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        # In production, execute SQL cleanup queries
        # For now, log the action
        logger.info(f"Cleanup would remove data older than: {thirty_days_ago.date()}")
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_data: {e}")


async def check_retraining_trigger():
    """
    Check if ML models need retraining.
    Runs every hour.
    """
    try:
        # In production, check total setups generated
        # Retrain every 100 setups
        logger.debug("Checking retraining trigger")
        
    except Exception as e:
        logger.error(f"Error checking retraining trigger: {e}")


async def update_news_actuals():
    """
    Update actual values for news events during trading hours.
    Runs every 15 minutes.
    """
    try:
        current_hour = datetime.utcnow().hour
        
        # Only update during active trading hours (8-21 UTC)
        if 8 <= current_hour < 21:
            success = update_actual_values()
            if success:
                logger.debug("Updated news actual values")
            else:
                logger.warning("Failed to update news actual values")
        
    except Exception as e:
        logger.error(f"Error updating news actuals: {e}")


async def monitor_model_performance():
    """
    Monitor ML model performance and detect drift.
    Would run daily.
    """
    try:
        # Calculate live win rate vs validation accuracy
        # Flag if difference > 10%
        logger.debug("Monitoring model performance")
        
    except Exception as e:
        logger.error(f"Error monitoring model performance: {e}")


def shutdown_scheduler(sched: AsyncIOScheduler) -> None:
    """
    Gracefully shutdown scheduler.
    
    Args:
        sched: Scheduler instance
    """
    try:
        if sched and sched.running:
            sched.shutdown(wait=False)
            logger.info("Scheduler shut down")
    except Exception as e:
        logger.error(f"Error shutting down scheduler: {e}")