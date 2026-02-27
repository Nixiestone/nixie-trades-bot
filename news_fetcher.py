import logging
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import config

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """Data class for news events."""
    title: str
    currency: str
    impact: str  # 'HIGH', 'MEDIUM', 'LOW'
    timestamp: datetime
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    source: str = 'Unknown'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'currency': self.currency,
            'impact': self.impact,
            'timestamp': self.timestamp.isoformat(),
            'forecast': self.forecast,
            'previous': self.previous,
            'actual': self.actual,
            'source': self.source
        }
    
    def affects_symbol(self, symbol: str) -> bool:
        """
        Check if this news event affects a trading symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            bool: True if affects symbol
        """
        return self.currency in symbol


class NewsAPIRateLimiter:
    """Rate limiter for news API calls."""
    
    def __init__(self, calls_per_hour: int = 100):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_hour: Maximum API calls per hour
        """
        self.calls_per_hour = calls_per_hour
        self.call_timestamps: List[float] = []
        self.logger = logging.getLogger(f"{__name__}.NewsAPIRateLimiter")
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        current_time = time.time()
        
        # Clean calls older than 1 hour
        self.call_timestamps = [
            t for t in self.call_timestamps
            if current_time - t < 3600
        ]
        
        # Check if limit reached
        if len(self.call_timestamps) >= self.calls_per_hour:
            oldest_call = self.call_timestamps[0]
            wait_time = 3600 - (current_time - oldest_call)
            
            if wait_time > 0:
                self.logger.warning(f"News API rate limit reached, sleeping {wait_time:.0f}s")
                time.sleep(wait_time)
                current_time = time.time()
                self.call_timestamps = []
        
        # Record this call
        self.call_timestamps.append(current_time)


class NewsFetcher:
    """
    Fetches high-impact forex news events.
    Uses free NewsAPI.org with fallback to manual calendar.
    """
    
    def __init__(self):
        """Initialize News Fetcher."""
        self.logger = logging.getLogger(f"{__name__}.NewsFetcher")
        self.api_key = config.NEWS_API_KEY
        self.rate_limiter = NewsAPIRateLimiter(calls_per_hour=100)
        self.cache: Dict[str, List[NewsEvent]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=30)
        
        # Red folder keywords (HIGH impact events)
        self.red_folder_keywords = [
            'NFP', 'Non-Farm', 'Payroll', 'Employment',
            'FOMC', 'Federal Reserve', 'Fed Meeting', 'Interest Rate',
            'CPI', 'Inflation', 'Consumer Price',
            'GDP', 'Gross Domestic',
            'Retail Sales',
            'PMI', 'Manufacturing',
            'Central Bank', 'ECB', 'BOE', 'BOJ',
            'Trade Balance',
            'Unemployment'
        ]
        
        self.logger.info("News Fetcher initialized")
    
    # ==================== NEWS FETCHING ====================
    
    def get_upcoming_news(
        self,
        hours_ahead: int = 24,
        impact_filter: str = 'HIGH'
    ) -> List[NewsEvent]:
        """
        Get upcoming high-impact news events with caching and rate limiting.
        
        Args:
            hours_ahead: Look ahead this many hours
            impact_filter: Filter by impact ('HIGH', 'MEDIUM', 'LOW', 'ALL')
            
        Returns:
            list: List of NewsEvent objects
        """
        try:
            # Check cache first
            cache_key = f"{hours_ahead}_{impact_filter}"
            
            if cache_key in self.cache:
                if datetime.now() < self.cache_expiry[cache_key]:
                    self.logger.debug(f"Returning cached news ({len(self.cache[cache_key])} events)")
                    return self.cache[cache_key]
            
            # Fetch new data
            events = []
            
            # Priority 1: Forex Factory - most accurate economic calendar
            events = self._fetch_from_forex_factory(hours_ahead)

            # Priority 2: Investing.com - fallback when FF is blocked (403)
            if not events:
                events = self._fetch_from_investing_com(hours_ahead)

            # Priority 3: NewsAPI - last resort, general financial news only
            if not events and self.api_key:
                self.rate_limiter.wait_if_needed()
                events = self._fetch_from_newsapi(hours_ahead)

            # Priority 4: Static calendar - always available, fixed recurring events
            if not events:
                events = self._fetch_from_static_calendar(hours_ahead)
            # Filter by impact
            if impact_filter != 'ALL':
                events = [e for e in events if e.impact == impact_filter]
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            
            # Update cache
            self.cache[cache_key] = events
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            self.logger.info(f"Fetched {len(events)} news events (next {hours_ahead}h)")
            return events
        
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
    
    def _fetch_from_newsapi(self, hours_ahead: int) -> List[NewsEvent]:
        """
        Fetch news from NewsAPI.org.
        
        Args:
            hours_ahead: Look ahead hours
            
        Returns:
            list: NewsEvent objects
        """
        try:
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            
            # Search for forex-related news
            params = {
                'apiKey': self.api_key,
                'q': 'forex OR "central bank" OR "interest rate" OR inflation OR GDP',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': datetime.now().isoformat(),
                'to': (datetime.now() + timedelta(hours=hours_ahead)).isoformat()
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"NewsAPI returned status {response.status_code}")
                return []
            
            data = response.json()
            
            if data.get('status') != 'ok':
                self.logger.error(f"NewsAPI error: {data.get('message')}")
                return []
            
            articles = data.get('articles', [])
            events = []
            
            for article in articles:
                # Extract news event from article
                title = article.get('title', '')
                published_at = article.get('publishedAt', '')
                
                # Classify impact based on keywords
                impact = self._classify_impact(title)
                
                # Determine affected currency
                currency = self._extract_currency(title)
                
                if impact == 'HIGH' and currency:
                    try:
                        timestamp = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        
                        event = NewsEvent(
                            title=title,
                            currency=currency,
                            impact=impact,
                            timestamp=timestamp,
                            source='NewsAPI'
                        )
                        
                        events.append(event)
                    
                    except Exception as e:
                        self.logger.debug(f"Error parsing article: {e}")
                        continue
            
            return events
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"NewsAPI request error: {e}")
            return []
        
        except Exception as e:
            self.logger.error(f"Error parsing NewsAPI response: {e}")
            return []
    
    def _fetch_from_forex_factory(self, hours_ahead: int) -> List[NewsEvent]:
        """
        Fetch news from Forex Factory economic calendar.
        Fallback when NewsAPI unavailable.
        
        Args:
            hours_ahead: Look ahead hours
            
        Returns:
            list: NewsEvent objects
        """
        try:
            from bs4 import BeautifulSoup
            
            self.logger.info("Fetching news from Forex Factory...")
            
            # Forex Factory calendar URL
            url = "https://www.forexfactory.com/calendar"
            
            # Headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Rate limiting for web scraping (be respectful)
            time.sleep(2)  # 2 second delay between requests
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                self.logger.error(f"Forex Factory returned status {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find calendar table
            calendar_table = soup.find('table', class_='calendar__table')
            
            if not calendar_table:
                self.logger.error("Could not find calendar table on Forex Factory")
                return []
            
            events = []
            current_date = None
            now = datetime.now(timezone.utc)
            future_time = now + timedelta(hours=hours_ahead)
            
            # Parse calendar rows
            rows = calendar_table.find_all('tr', class_='calendar__row')
            
            for row in rows:
                try:
                    # Check if this is a date row
                    date_cell = row.find('td', class_='calendar__cell calendar__date')
                    if date_cell and date_cell.get_text(strip=True):
                        date_str = date_cell.get_text(strip=True)
                        # Parse date (format: "Mon Jan 15")
                        try:
                            current_date = datetime.strptime(f"{date_str} {now.year}", "%a %b %d %Y").date()
                        except:
                            continue
                    
                    if not current_date:
                        continue
                    
                    # Get time
                    time_cell = row.find('td', class_='calendar__cell calendar__time')
                    if not time_cell:
                        continue
                    
                    time_str = time_cell.get_text(strip=True)
                    if not time_str or time_str == 'All Day' or time_str == 'Tentative':
                        continue
                    
                    # Get currency
                    currency_cell = row.find('td', class_='calendar__cell calendar__currency')
                    if not currency_cell:
                        continue
                    
                    currency = currency_cell.get_text(strip=True)
                    if not currency or len(currency) != 3:
                        continue
                    
                    # Get impact (shown as colored icons)
                    impact_cell = row.find('td', class_='calendar__cell calendar__impact')
                    impact = 'LOW'
                    if impact_cell:
                        impact_span = impact_cell.find('span')
                        if impact_span:
                            impact_class = impact_span.get('class', [])
                            if 'icon--ff-impact-red' in impact_class:
                                impact = 'HIGH'
                            elif 'icon--ff-impact-ora' in impact_class or 'icon--ff-impact-orange' in impact_class:
                                impact = 'MEDIUM'
                    
                    # Only process HIGH impact events
                    if impact != 'HIGH':
                        continue
                    
                    # Get event title
                    event_cell = row.find('td', class_='calendar__cell calendar__event')
                    if not event_cell:
                        continue
                    
                    event_title = event_cell.get_text(strip=True)
                    if not event_title:
                        continue
                    
                    # Get forecast, previous, actual (if available)
                    forecast = None
                    previous = None
                    actual = None
                    
                    forecast_cell = row.find('td', class_='calendar__cell calendar__forecast')
                    if forecast_cell:
                        forecast = forecast_cell.get_text(strip=True)
                    
                    previous_cell = row.find('td', class_='calendar__cell calendar__previous')
                    if previous_cell:
                        previous = previous_cell.get_text(strip=True)
                    
                    actual_cell = row.find('td', class_='calendar__cell calendar__actual')
                    if actual_cell:
                        actual = actual_cell.get_text(strip=True)
                    
                    # Parse time and create datetime
                    try:
                        # Time format: "1:30pm" or "13:30"
                        if 'am' in time_str.lower() or 'pm' in time_str.lower():
                            event_time = datetime.strptime(time_str, "%I:%M%p").time()
                        else:
                            event_time = datetime.strptime(time_str, "%H:%M").time()
                        
                        event_datetime = datetime.combine(current_date, event_time)
                        
                        # Check if event is in our time window
                        if now <= event_datetime <= future_time:
                            event = NewsEvent(
                                title=event_title,
                                currency=currency,
                                impact=impact,
                                timestamp=event_datetime,
                                forecast=forecast,
                                previous=previous,
                                actual=actual,
                                source='Forex Factory'
                            )
                            
                            events.append(event)
                    
                    except Exception as e:
                        self.logger.debug(f"Error parsing time '{time_str}': {e}")
                        continue
                
                except Exception as e:
                    self.logger.debug(f"Error parsing row: {e}")
                    continue
            
            self.logger.info(f"Fetched {len(events)} events from Forex Factory")
            return events
        
        except ImportError:
            self.logger.error("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return []
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Forex Factory request error: {e}")
            return []
        
        except Exception as e:
            self.logger.error(f"Error scraping Forex Factory: {e}")
            return []
    
    def _fetch_from_investing_com(self, hours_ahead: int) -> list:
        """
        Fetch economic calendar from Investing.com.
        Used as fallback when Forex Factory returns 403.
        """
        try:
            url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
            headers = {
                'User-Agent':       'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer':          'https://www.investing.com/economic-calendar/',
            }
            payload = {
                'country[]':    ['5', '4', '17', '39', '7', '14', '10'],
                'importance[]': ['3'],
                'timeZone':     '55',
                'timeFilter':   'timeRemain',
                'currentTab':   'custom',
                'submitFilters': '1',
                'limit_from':   '0',
            }
            import time as _time
            _time.sleep(1)
            resp = requests.post(url, headers=headers, data=payload, timeout=15)
            if resp.status_code != 200:
                self.logger.error("Investing.com returned status %d", resp.status_code)
                return []

            data = resp.json()
            html_content = data.get('data', '')
            if not html_content:
                return []

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            rows = soup.find_all('tr', class_='js-event-item')

            events = []
            now    = datetime.now(timezone.utc)
            future = now + timedelta(hours=hours_ahead)

            for row in rows:
                try:
                    time_cell     = row.find('td', class_='time')
                    currency_cell = row.find('td', class_='flagCur')
                    event_cell    = row.find('td', class_='event')
                    bull_icons    = row.find_all('i', class_='grayFullBullishIcon')

                    if not time_cell or not currency_cell or not event_cell:
                        continue
                    if len(bull_icons) < 3:
                        continue

                    time_str = time_cell.get_text(strip=True)
                    currency = currency_cell.get_text(strip=True).replace('\xa0', '').strip()[-3:]
                    title    = event_cell.get_text(strip=True)

                    if not time_str or not currency or len(currency) != 3:
                        continue

                    try:
                        evt_time = datetime.strptime(time_str, '%H:%M').replace(
                            year=now.year, month=now.month, day=now.day,
                            tzinfo=timezone.utc
                        )
                        if evt_time < now:
                            evt_time = evt_time + timedelta(days=1)
                    except Exception:
                        continue

                    if now <= evt_time <= future:
                        events.append(NewsEvent(
                            title=title,
                            currency=currency,
                            impact='HIGH',
                            timestamp=evt_time,
                            source='Investing.com',
                        ))
                except Exception as row_err:
                    self.logger.debug("Error parsing Investing.com row: %s", row_err)
                    continue

            self.logger.info("Fetched %d events from Investing.com", len(events))
            return events

        except ImportError:
            self.logger.warning(
                "beautifulsoup4 not installed. Run: pip install beautifulsoup4")
            return []
        except Exception as e:
            self.logger.error("Investing.com fetch error: %s", e)
            return []
    
    def _fetch_from_static_calendar(self, hours_ahead: int) -> List[NewsEvent]:
        """
        Fetch news from static economic calendar.
        Fallback when API unavailable.
        
        Args:
            hours_ahead: Look ahead hours
            
        Returns:
            list: NewsEvent objects
        """
        try:
            # Static calendar of recurring high-impact events
            # Times are approximate (UTC)
            static_events = [
                # US Events (usually 13:30 UTC or 15:00 UTC)
                {'title': 'US Non-Farm Payrolls', 'currency': 'USD', 'hour': 13, 'minute': 30, 'day_of_month': 1, 'weekday': 4},  # First Friday
                {'title': 'US CPI Report', 'currency': 'USD', 'hour': 13, 'minute': 30, 'day_of_month': 15},
                {'title': 'US Retail Sales', 'currency': 'USD', 'hour': 13, 'minute': 30, 'day_of_month': 15},
                {'title': 'FOMC Interest Rate Decision', 'currency': 'USD', 'hour': 19, 'minute': 0, 'day_of_month': 1},
                
                # EUR Events (usually 09:00 UTC)
                {'title': 'ECB Interest Rate Decision', 'currency': 'EUR', 'hour': 12, 'minute': 45, 'day_of_month': 1},
                {'title': 'Eurozone CPI', 'currency': 'EUR', 'hour': 10, 'minute': 0, 'day_of_month': 1},
                {'title': 'German PMI', 'currency': 'EUR', 'hour': 8, 'minute': 30, 'day_of_month': 1},
                
                # GBP Events (usually 07:00 UTC)
                {'title': 'BOE Interest Rate Decision', 'currency': 'GBP', 'hour': 12, 'minute': 0, 'day_of_month': 1},
                {'title': 'UK CPI', 'currency': 'GBP', 'hour': 7, 'minute': 0, 'day_of_month': 15},
                {'title': 'UK GDP', 'currency': 'GBP', 'hour': 7, 'minute': 0, 'day_of_month': 10},
            ]
            
            now = datetime.now(timezone.utc)
            future_time = now + timedelta(hours=hours_ahead)
            events = []
            
            # Generate events for current and next month
            for month_offset in [0, 1]:
                check_date = now.replace(day=1) + timedelta(days=32 * month_offset)
                check_date = check_date.replace(day=1)
                
                for event_template in static_events:
                    # Calculate event datetime
                    try:
                        event_date = check_date.replace(
                            day=event_template['day_of_month'],
                            hour=event_template['hour'],
                            minute=event_template['minute'],
                            second=0,
                            microsecond=0
                        )
                        
                        # Check if event is in our time window
                        if now <= event_date <= future_time:
                            event = NewsEvent(
                                title=event_template['title'],
                                currency=event_template['currency'],
                                impact='HIGH',
                                timestamp=event_date,
                                source='Static Calendar'
                            )
                            
                            events.append(event)
                    
                    except ValueError:
                        # Invalid day for month (e.g., Feb 30)
                        continue
            
            return events
        
        except Exception as e:
            self.logger.error(f"Error generating static calendar: {e}")
            return []
    
    # ==================== CLASSIFICATION & FILTERING ====================
    
    def _classify_impact(self, text: str) -> str:
        """
        Classify news impact based on keywords.
        
        Args:
            text: News title or description
            
        Returns:
            str: 'HIGH', 'MEDIUM', or 'LOW'
        """
        text_upper = text.upper()
        
        # Check red folder keywords
        for keyword in self.red_folder_keywords:
            if keyword.upper() in text_upper:
                return 'HIGH'
        
        # Medium impact indicators
        medium_keywords = ['JOBLESS', 'CLAIMS', 'HOUSING', 'FACTORY', 'ORDERS', 'SENTIMENT']
        for keyword in medium_keywords:
            if keyword in text_upper:
                return 'MEDIUM'
        
        return 'LOW'
    
    def _extract_currency(self, text: str) -> Optional[str]:
        """
        Extract currency from news text.
        
        Args:
            text: News title or description
            
        Returns:
            str: Currency code or None
        """
        text_upper = text.upper()
        
        # Currency mappings
        currency_keywords = {
            'USD': ['US ', 'U.S.', 'UNITED STATES', 'AMERICA', 'DOLLAR', 'FED', 'FOMC'],
            'EUR': ['EURO', 'EUROZONE', 'ECB', 'EUROPE'],
            'GBP': ['UK ', 'BRITAIN', 'BRITISH', 'POUND', 'BOE', 'STERLING'],
            'JPY': ['JAPAN', 'YEN', 'BOJ', 'NIKKEI'],
            'AUD': ['AUSTRALIA', 'AUSSIE', 'RBA'],
            'CAD': ['CANADA', 'CANADIAN', 'LOONIE'],
            'CHF': ['SWISS', 'SWITZERLAND', 'SNB', 'FRANC'],
            'NZD': ['NEW ZEALAND', 'KIWI', 'RBNZ']
        }
        
        for currency, keywords in currency_keywords.items():
            for keyword in keywords:
                if keyword in text_upper:
                    return currency
        
        return None
    
    def get_red_folder_events(self, hours_ahead: int = 24) -> List[NewsEvent]:
        """
        Get only red folder (HIGH impact) events.
        
        Args:
            hours_ahead: Look ahead hours
            
        Returns:
            list: HIGH impact NewsEvent objects
        """
        return self.get_upcoming_news(hours_ahead, impact_filter='HIGH')
    
    def check_news_blackout(self, symbol: str) -> Tuple[bool, Optional[NewsEvent]]:
        """
        Check if in news blackout period for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            tuple: (is_blackout, nearest_event)
        """
        try:
            # Get upcoming news
            events = self.get_red_folder_events(hours_ahead=2)
            
            now = datetime.now(timezone.utc)
            
            for event in events:
                # Check if event affects this symbol
                if not event.affects_symbol(symbol):
                    continue
                
                # Calculate time until event
                event_ts = event.timestamp
                if event_ts.tzinfo is None:
                    event_ts = event_ts.replace(tzinfo=timezone.utc)
                time_until = (event_ts - now).total_seconds() / 60
                
                # Check if within blackout window
                if -config.NEWS_BLACKOUT_AFTER_MINUTES <= time_until <= config.NEWS_BLACKOUT_BEFORE_MINUTES:
                    return True, event
            
            return False, None
        
        except Exception as e:
            self.logger.error(f"Error checking news blackout: {e}")
            return False, None
    
    def get_next_high_impact(self, symbol: str) -> Optional[NewsEvent]:
        """
        Get next high-impact event affecting a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            NewsEvent or None
        """
        try:
            events = self.get_red_folder_events(hours_ahead=48)
            
            for event in events:
                if event.affects_symbol(symbol):
                    return event
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting next high impact: {e}")
            return None
    
    def format_news_summary(self, hours_ahead: int = 24) -> str:
        """
        Format news summary for daily 8 AM alert.
        
        Args:
            hours_ahead: Look ahead hours
            
        Returns:
            str: Formatted news summary
        """
        try:
            events = self.get_red_folder_events(hours_ahead)
            
            if not events:
                return "No high-impact news events scheduled for today."
            
            summary = f"HIGH-IMPACT NEWS TODAY ({len(events)} events):\n\n"
            
            for i, event in enumerate(events, 1):
                time_str = event.timestamp.strftime('%H:%M UTC')
                summary += f"{i}. {time_str} - {event.currency} - {event.title}\n"
            
            summary += "\nAvoid trading 30 minutes before and 15 minutes after these events."
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error formatting news summary: {e}")
            return "Error fetching news events."
    
    # ==================== CACHE MANAGEMENT ====================
    
    def clear_cache(self):
        """Clear news cache."""
        self.cache = {}
        self.cache_expiry = {}
        self.logger.info("News cache cleared")