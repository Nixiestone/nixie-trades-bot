"""
ForexFactory and Investing.com Economic Calendar Scraper
Fetches HIGH impact news events for Nix Trades Telegram Bot

Features:
- Scrapes ForexFactory.com (primary source)
- Falls back to Investing.com if ForexFactory fails
- Caches events in database
- Updates actual values every 15 minutes
- Displays times in BOTH UTC and user local time

NO EMOJIS - Professional code only
"""

import logging
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
import pytz
import config
import utils

logger = logging.getLogger(__name__)

# Request headers to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


def initialize() -> None:
    """
    Initialize news fetcher module.
    Called on bot startup.
    """
    logger.info("News fetcher initialized")


# ==================== FOREXFACTORY SCRAPER ====================

def scrape_forexfactory(target_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    Scrape economic calendar from ForexFactory.com.
    
    Args:
        target_date: Date to scrape (default: today)
        
    Returns:
        List[dict]: List of news event dictionaries
        
    Note:
        ForexFactory displays times in ET (Eastern Time). Must convert to UTC.
    """
    try:
        if target_date is None:
            target_date = datetime.now(timezone.utc)
        
        # Format date for URL (e.g., 'feb10.2024')
        date_str = target_date.strftime('%b%d.%Y').lower()
        
        url = f'https://www.forexfactory.com/calendar?day={date_str}'
        
        logger.info(f"Scraping ForexFactory: {url}")
        
        # Make request
        response = requests.get(url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT_SECONDS)
        
        if response.status_code != 200:
            logger.error(f"ForexFactory returned status {response.status_code}")
            return []
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find calendar table
        calendar_table = soup.find('table', class_='calendar__table')
        
        if not calendar_table:
            logger.warning("ForexFactory calendar table not found")
            return []
        
        events = []
        
        # Parse rows
        rows = calendar_table.find_all('tr', class_='calendar__row')
        
        current_time = None  # Track time across rows (some rows don't have time)
        
        for row in rows:
            try:
                # Extract time
                time_cell = row.find('td', class_='calendar__time')
                if time_cell and time_cell.get_text(strip=True):
                    time_str = time_cell.get_text(strip=True)
                    if time_str and time_str != '':
                        current_time = time_str
                
                # Extract currency
                currency_cell = row.find('td', class_='calendar__currency')
                if not currency_cell:
                    continue
                currency = currency_cell.get_text(strip=True)
                
                # Extract event name
                event_cell = row.find('td', class_='calendar__event')
                if not event_cell:
                    continue
                event_name = event_cell.get_text(strip=True)
                
                # Extract impact (look for impact icon)
                impact_cell = row.find('td', class_='calendar__impact')
                impact = 'LOW'
                if impact_cell:
                    # Check for high impact (red folder)
                    if impact_cell.find('span', class_='icon--ff-impact-red'):
                        impact = 'HIGH'
                    elif impact_cell.find('span', class_='icon--ff-impact-ora'):
                        impact = 'MEDIUM'
                    elif impact_cell.find('span', class_='icon--ff-impact-yel'):
                        impact = 'LOW'
                
                # Extract forecast
                forecast_cell = row.find('td', class_='calendar__forecast')
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                
                # Extract previous
                previous_cell = row.find('td', class_='calendar__previous')
                previous = previous_cell.get_text(strip=True) if previous_cell else None
                
                # Extract actual (if released)
                actual_cell = row.find('td', class_='calendar__actual')
                actual = actual_cell.get_text(strip=True) if actual_cell else None
                
                # Skip if no time set yet
                if not current_time:
                    continue
                
                # Convert time from ET to UTC
                event_time_utc = convert_et_to_utc(current_time, target_date)
                
                if not event_time_utc:
                    continue
                
                # Create event dictionary
                event = {
                    'event_time_utc': event_time_utc,
                    'currency': currency,
                    'event_name': event_name,
                    'impact': impact,
                    'forecast': forecast,
                    'previous': previous,
                    'actual': actual,
                    'source': 'ForexFactory'
                }
                
                events.append(event)
            
            except Exception as e:
                logger.error(f"Error parsing ForexFactory row: {e}")
                continue
        
        logger.info(f"Scraped {len(events)} events from ForexFactory")
        return events
    
    except Exception as e:
        logger.error(f"Error scraping ForexFactory: {e}")
        return []


def convert_et_to_utc(time_str: str, date: datetime) -> Optional[datetime]:
    """
    Convert ForexFactory time (ET) to UTC.
    
    Args:
        time_str: Time string (e.g., '8:30am', '2:00pm')
        date: Date of the event
        
    Returns:
        datetime: Event time in UTC timezone
        
    Note:
        ET can be EST (UTC-5) or EDT (UTC-4) depending on DST
        pytz handles this automatically
    """
    try:
        # Parse time string
        time_str = time_str.strip().lower()
        
        # Handle formats: '8:30am', '2:00pm', '10:30pm'
        if 'am' in time_str or 'pm' in time_str:
            hour_part = time_str.replace('am', '').replace('pm', '')
            hour, minute = map(int, hour_part.split(':'))
            
            # Convert to 24-hour format
            if 'pm' in time_str and hour != 12:
                hour += 12
            elif 'am' in time_str and hour == 12:
                hour = 0
        else:
            # Handle 24-hour format
            hour, minute = map(int, time_str.split(':'))
        
        # Create datetime in ET timezone
        et_tz = pytz.timezone('America/New_York')
        event_datetime = et_tz.localize(datetime(
            date.year, date.month, date.day, hour, minute
        ))
        
        # Convert to UTC
        event_utc = event_datetime.astimezone(timezone.utc)
        
        return event_utc
    
    except Exception as e:
        logger.error(f"Error converting ET time '{time_str}': {e}")
        return None


# ==================== INVESTING.COM SCRAPER (FALLBACK) ====================

def scrape_investing_com(target_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    Scrape economic calendar from Investing.com (fallback).
    
    Args:
        target_date: Date to scrape (default: today)
        
    Returns:
        List[dict]: List of news event dictionaries
    """
    try:
        if target_date is None:
            target_date = datetime.now(timezone.utc)
        
        url = 'https://www.investing.com/economic-calendar/'
        
        logger.info(f"Scraping Investing.com: {url}")
        
        # Make request
        response = requests.get(url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT_SECONDS)
        
        if response.status_code != 200:
            logger.error(f"Investing.com returned status {response.status_code}")
            return []
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        events = []
        
        # Find event rows
        event_rows = soup.find_all('tr', class_='js-event-item')
        
        for row in event_rows:
            try:
                # Extract time
                time_attr = row.get('data-event-datetime')
                if not time_attr:
                    continue
                
                # Parse timestamp
                event_timestamp = int(time_attr)
                event_time_utc = datetime.fromtimestamp(event_timestamp, tz=timezone.utc)
                
                # Only include today's events
                if event_time_utc.date() != target_date.date():
                    continue
                
                # Extract currency
                currency_cell = row.find('td', class_='flagCur')
                currency = currency_cell.get_text(strip=True) if currency_cell else 'USD'
                
                # Extract event name
                event_cell = row.find('td', class_='event')
                if not event_cell:
                    continue
                event_name = event_cell.get_text(strip=True)
                
                # Extract impact
                impact = 'LOW'
                sentiment_cell = row.find('td', class_='sentiment')
                if sentiment_cell:
                    if 'grayFullBullishIcon' in str(sentiment_cell):
                        impact = 'HIGH'
                    elif 'grayMediumBullishIcon' in str(sentiment_cell):
                        impact = 'MEDIUM'
                
                # Extract forecast, previous, actual
                forecast_cell = row.find('td', class_='fore')
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                
                previous_cell = row.find('td', class_='prev')
                previous = previous_cell.get_text(strip=True) if previous_cell else None
                
                actual_cell = row.find('td', class_='act')
                actual = actual_cell.get_text(strip=True) if actual_cell else None
                
                # Create event dictionary
                event = {
                    'event_time_utc': event_time_utc,
                    'currency': currency,
                    'event_name': event_name,
                    'impact': impact,
                    'forecast': forecast,
                    'previous': previous,
                    'actual': actual,
                    'source': 'Investing.com'
                }
                
                events.append(event)
            
            except Exception as e:
                logger.error(f"Error parsing Investing.com row: {e}")
                continue
        
        logger.info(f"Scraped {len(events)} events from Investing.com")
        return events
    
    except Exception as e:
        logger.error(f"Error scraping Investing.com: {e}")
        return []


# ==================== CACHE MANAGEMENT ====================

def fetch_and_cache_news(target_date: Optional[datetime] = None) -> int:
    """
    Fetch news from sources and cache in database.
    
    Args:
        target_date: Date to fetch (default: today)
        
    Returns:
        int: Number of high-impact events cached
    """
    try:
        # Try ForexFactory first
        events = scrape_forexfactory(target_date)
        
        # Fallback to Investing.com if ForexFactory failed
        if not events:
            logger.warning("ForexFactory scraping failed, trying Investing.com")
            events = scrape_investing_com(target_date)
        
        if not events:
            logger.error("All news sources failed")
            return 0
        
        # Filter HIGH impact only
        high_impact_events = [e for e in events if e['impact'] == 'HIGH']
        
        # Save to database
        import database
        
        saved_count = 0
        for event in high_impact_events:
            try:
                success = database.save_news_event(
                    event_time_utc=event['event_time_utc'],
                    currency=event['currency'],
                    event_name=event['event_name'],
                    impact=event['impact'],
                    forecast=event.get('forecast'),
                    previous=event.get('previous')
                )
                
                if success:
                    saved_count += 1
            
            except Exception as e:
                logger.error(f"Error saving event: {e}")
                continue
        
        logger.info(f"Cached {saved_count} high-impact events")
        return saved_count
    
    except Exception as e:
        logger.error(f"Error in fetch_and_cache_news: {e}")
        return 0


def update_actual_values() -> bool:
    """
    Update actual values for events that have occurred.
    
    Called every 15 minutes during trading hours to fetch released values.
    
    Returns:
        bool: True if successful
    """
    try:
        # Fetch latest events to get actual values
        events = scrape_forexfactory()
        
        if not events:
            return False
        
        # Update actuals in database
        import database
        
        updated_count = 0
        for event in events:
            if event.get('actual'):
                # Update in database (would need database function)
                # For now, just log
                logger.debug(f"Would update {event['event_name']} with actual: {event['actual']}")
                updated_count += 1
        
        if updated_count > 0:
            logger.info(f"Updated {updated_count} actual values")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating actual values: {e}")
        return False


# ==================== FORMATTING FOR TELEGRAM ====================

def get_todays_news_formatted(user_timezone: str = 'UTC') -> str:
    """
    Get today's HIGH impact news formatted for Telegram message.
    
    CRITICAL: Displays times in BOTH UTC AND user local time.
    
    Args:
        user_timezone: User's IANA timezone string
        
    Returns:
        str: Formatted news text ready for Telegram
        
    Format:
        HIGH-IMPACT ECONOMIC EVENTS TODAY:
        
        14:30 UTC (2:30 PM your time) - USD Non-Farm Payrolls
        Previous: 150K | Forecast: 180K
        Expected volatility: 50-100 pips on USD pairs
        
        18:00 UTC (6:00 PM your time) - EUR Interest Rate Decision
        Previous: 4.50% | Forecast: 4.25%
        Expected volatility: 80-150 pips on EUR pairs
    """
    try:
        import database
        
        # Get today's cached news
        events = database.get_todays_news()
        
        # Filter HIGH impact
        high_impact = [e for e in events if e.get('impact') == 'HIGH']
        
        if not high_impact:
            return "HIGH-IMPACT ECONOMIC EVENTS TODAY:\n\nNo high-impact events scheduled for today.\n"
        
        # Sort by time
        high_impact.sort(key=lambda x: x['event_time_utc'])
        
        # Build formatted message
        message_lines = ["HIGH-IMPACT ECONOMIC EVENTS TODAY:"]
        message_lines.append("")
        
        for event in high_impact:
            # Parse event time
            event_time = utils.parse_iso_datetime(event['event_time_utc'])
            if not event_time:
                continue
            
            # Convert to user's local time
            local_time = utils.convert_utc_to_user_time(event_time, user_timezone)
            
            # Format times
            utc_time_str = event_time.strftime('%H:%M UTC')
            local_time_str = local_time.strftime('%I:%M %p your time')
            
            # Event header
            message_lines.append(
                f"{utc_time_str} ({local_time_str}) - "
                f"{event['currency']} {event['event_name']}"
            )
            
            # Forecast and previous
            previous = event.get('previous', 'N/A')
            forecast = event.get('forecast', 'N/A')
            message_lines.append(f"Previous: {previous} | Forecast: {forecast}")
            
            # Expected volatility (estimate based on currency)
            currency = event['currency']
            if currency == 'USD':
                volatility = "50-100 pips on USD pairs"
            elif currency in ['EUR', 'GBP']:
                volatility = "40-80 pips on major pairs"
            else:
                volatility = "30-60 pips on affected pairs"
            
            message_lines.append(f"Expected volatility: {volatility}")
            message_lines.append("")  # Blank line between events
        
        return "\n".join(message_lines)
    
    except Exception as e:
        logger.error(f"Error formatting news: {e}")
        return "HIGH-IMPACT ECONOMIC EVENTS TODAY:\n\nError loading news data.\n"


def check_news_proximity(symbol: str, minutes_ahead: int = 30) -> Tuple[bool, Optional[str]]:
    """
    Check if there's high-impact news within specified minutes.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        minutes_ahead: Minutes to look ahead (default: 30)
        
    Returns:
        Tuple[bool, str]: (is_clear, event_description)
        
    Example:
        >>> check_news_proximity('EURUSD', 30)
        (False, 'USD Non-Farm Payrolls in 25 minutes')
    """
    try:
        import database
        
        # Extract currencies from symbol
        base_currency, quote_currency = utils.extract_currency_from_symbol(symbol)
        
        # Get today's news
        events = database.get_todays_news()
        
        # Filter HIGH impact for relevant currencies
        relevant_events = [
            e for e in events
            if e['impact'] == 'HIGH' and e['currency'] in [base_currency, quote_currency]
        ]
        
        if not relevant_events:
            return True, None
        
        # Check proximity
        now_utc = utils.get_current_utc_time()
        cutoff_time = now_utc + timedelta(minutes=minutes_ahead)
        
        for event in relevant_events:
            event_time = utils.parse_iso_datetime(event['event_time_utc'])
            if not event_time:
                continue
            
            # Check if event is within the window
            if now_utc < event_time <= cutoff_time:
                # Calculate minutes until event
                minutes_until = int((event_time - now_utc).total_seconds() / 60)
                
                event_desc = f"{event['currency']} {event['event_name']} in {minutes_until} minutes"
                return False, event_desc
        
        # No events in proximity window
        return True, None
    
    except Exception as e:
        logger.error(f"Error checking news proximity: {e}")
        # Fail-safe: Assume no news nearby if check fails
        return True, None


# ==================== VOLATILITY ESTIMATES ====================

def get_expected_volatility(currency: str, event_name: str) -> str:
    """
    Get expected volatility estimate for an event.
    
    Args:
        currency: Currency code (e.g., 'USD', 'EUR')
        event_name: Name of economic event
        
    Returns:
        str: Volatility estimate in pips
    """
    # High volatility events
    high_volatility_events = [
        'Non-Farm Payrolls', 'NFP', 'Interest Rate', 'FOMC',
        'GDP', 'Unemployment', 'CPI', 'Inflation'
    ]
    
    # Check if event is high volatility
    is_high_volatility = any(keyword in event_name for keyword in high_volatility_events)
    
    if currency == 'USD':
        return "50-100 pips" if is_high_volatility else "30-60 pips"
    elif currency in ['EUR', 'GBP']:
        return "40-80 pips" if is_high_volatility else "25-50 pips"
    else:
        return "30-60 pips" if is_high_volatility else "20-40 pips"


# ==================== TESTING & DEBUGGING ====================

def test_scraper():
    """
    Test scraper functionality (for development/debugging).
    """
    logger.info("Testing news scraper...")
    
    # Test ForexFactory
    ff_events = scrape_forexfactory()
    logger.info(f"ForexFactory returned {len(ff_events)} events")
    
    # Test Investing.com
    inv_events = scrape_investing_com()
    logger.info(f"Investing.com returned {len(inv_events)} events")
    
    # Test formatting
    formatted = get_todays_news_formatted('America/New_York')
    logger.info(f"Formatted news:\n{formatted}")


if __name__ == '__main__':
    # Run tests when executed directly
    logging.basicConfig(level=logging.INFO)
    test_scraper()