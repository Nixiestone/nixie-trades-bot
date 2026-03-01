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
    Uses live sources only (Forex Factory, Investing.com, NewsAPI).
    """
    
    def __init__(self):
        """Initialize News Fetcher."""
        self.logger = logging.getLogger(f"{__name__}.NewsFetcher")
        self.api_key = config.NEWS_API_KEY
        self.rate_limiter = NewsAPIRateLimiter(calls_per_hour=100)
        self.cache: Dict[str, List[NewsEvent]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(
            minutes=max(15, int(getattr(config, 'NEWS_UPDATE_INTERVAL_MINUTES', 120)))
        )
        
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
            now_utc = datetime.now(timezone.utc)

            if cache_key in self.cache and cache_key in self.cache_expiry:
                if now_utc < self.cache_expiry[cache_key]:
                    self.logger.debug(
                        "Returning cached news (%d events)",
                        len(self.cache[cache_key]),
                    )
                    return self.cache[cache_key]

            stale_cache = self.cache.get(cache_key, [])
            
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
            if not events:
                self.logger.warning(
                    "No live news events available from configured sources "
                    "(hours_ahead=%d, impact_filter=%s).",
                    hours_ahead,
                    impact_filter,
                )
                if stale_cache:
                    stale_filtered = []
                    stale_cutoff = now_utc + timedelta(hours=hours_ahead)
                    for event in stale_cache:
                        ts = event.timestamp
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        else:
                            ts = ts.astimezone(timezone.utc)
                        if now_utc <= ts <= stale_cutoff:
                            event.timestamp = ts
                            stale_filtered.append(event)
                    self.logger.warning(
                        "Using stale cached news (%d events) for hours_ahead=%d.",
                        len(stale_filtered),
                        hours_ahead,
                    )
                    stale_filtered.sort(key=lambda x: x.timestamp)
                    self.cache[cache_key] = stale_filtered
                    self.cache_expiry[cache_key] = now_utc + self.cache_duration
                    return stale_filtered
            # Filter by impact
            if impact_filter != 'ALL':
                events = [e for e in events if e.impact == impact_filter]

            # De-duplicate (source overlap can return same event multiple times)
            deduped = {}
            for event in events:
                ts = event.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                key = (
                    event.currency.upper().strip(),
                    event.title.strip().lower(),
                    ts.replace(second=0, microsecond=0).isoformat(),
                )
                if key not in deduped:
                    event.timestamp = ts
                    deduped[key] = event
            events = list(deduped.values())
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            
            # Update cache
            self.cache[cache_key] = events
            self.cache_expiry[cache_key] = now_utc + self.cache_duration
            
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
            now_utc = datetime.now(timezone.utc)
            
            # Search for forex-related news
            params = {
                'apiKey': self.api_key,
                'q': 'forex OR "central bank" OR "interest rate" OR inflation OR GDP',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': now_utc.isoformat(),
                'to': (now_utc + timedelta(hours=hours_ahead)).isoformat()
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
        Fetch news from Forex Factory's public JSON calendar API.
        This endpoint is stable and does not block automated requests.
        URLs:
          This week: https://nfs.faireconomy.media/ff_calendar_thisweek.json
          Next week: https://nfs.faireconomy.media/ff_calendar_nextweek.json
        """
        try:
            now        = datetime.now(timezone.utc)
            future     = now + timedelta(hours=hours_ahead)
            events     = []

            urls = [
                'https://nfs.faireconomy.media/ff_calendar_thisweek.json',
                'https://nfs.faireconomy.media/ff_calendar_nextweek.json',
            ]

            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
                'Accept': 'application/json',
            }

            for url in urls:
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code != 200:
                        if resp.status_code in (403, 404, 429):
                            self.logger.debug(
                                "FF JSON endpoint unavailable (%d): %s",
                                resp.status_code,
                                url,
                            )
                        else:
                            self.logger.warning(
                                "FF JSON API returned %d for %s", resp.status_code, url)
                        continue

                    data = resp.json()
                    if not isinstance(data, list):
                        continue

                    for item in data:
                        try:
                            impact = item.get('impact', '').strip().upper()
                            if impact != 'HIGH':
                                continue

                            currency = (item.get('country') or '').strip().upper()
                            if not currency or len(currency) != 3:
                                continue

                            title    = (item.get('title') or '').strip()
                            date_str = (item.get('date') or '').strip()
                            time_str = (item.get('time') or '').strip()

                            if not date_str or not title:
                                continue

                            # Parse datetime.
                            # New FF format: "2026-02-27T08:30:00-05:00"
                            # Old FF format: "01-27-2026" + "8:30am"
                            try:
                                if 'T' in date_str:
                                    event_dt = datetime.fromisoformat(
                                        date_str.replace('Z', '+00:00')
                                    )
                                elif time_str:
                                    dt_str = f"{date_str} {time_str}".replace('\u202f', ' ')
                                    try:
                                        event_dt = datetime.strptime(
                                            dt_str, '%m-%d-%Y %I:%M%p'
                                        )
                                    except ValueError:
                                        event_dt = datetime.strptime(
                                            dt_str, '%m-%d-%Y %H:%M'
                                        )
                                else:
                                    event_dt = datetime.strptime(
                                        date_str, '%m-%d-%Y'
                                    )

                                if event_dt.tzinfo is None:
                                    # Older FF entries are US Eastern local time.
                                    import pytz
                                    eastern  = pytz.timezone('America/New_York')
                                    event_dt = eastern.localize(event_dt)

                                event_dt = event_dt.astimezone(timezone.utc)

                            except Exception as parse_err:
                                self.logger.debug(
                                    "FF date parse error for '%s %s': %s",
                                    date_str, time_str, parse_err)
                                continue

                            if now <= event_dt <= future:
                                events.append(NewsEvent(
                                    title=title,
                                    currency=currency,
                                    impact='HIGH',
                                    timestamp=event_dt,
                                    forecast=item.get('forecast', ''),
                                    previous=item.get('previous', ''),
                                    actual=item.get('actual', ''),
                                    source='Forex Factory',
                                ))

                        except Exception as item_err:
                            self.logger.debug(
                                "FF item parse error: %s", item_err)
                            continue

                except requests.exceptions.RequestException as req_err:
                    self.logger.warning(
                        "FF JSON request failed for %s: %s", url, req_err)
                    continue

            self.logger.info(
                "Fetched %d HIGH impact events from Forex Factory JSON API.",
                len(events))
            return events

        except Exception as e:
            self.logger.error("Error in _fetch_from_forex_factory: %s", e)
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
            now = datetime.now(timezone.utc)
            future = now + timedelta(hours=hours_ahead)
            payload = {
                'country[]':    ['5', '4', '17', '39', '7', '14', '10'],
                'importance[]': ['3'],
                'timeZone':     '55',
                'timeFilter':   'timeRemain',
                'currentTab':   'custom',
                'submitFilters': '1',
                'limit_from':   '0',
                'dateFrom':     now.strftime('%Y-%m-%d'),
                'dateTo':       future.strftime('%Y-%m-%d'),
            }
            import time as _time
            _time.sleep(1)
            resp = requests.post(url, headers=headers, data=payload, timeout=15)
            if resp.status_code != 200:
                self.logger.warning("Investing.com returned status %d", resp.status_code)
                return []

            data = resp.json() if resp.content else {}
            html_content = data.get('data', '') if isinstance(data, dict) else ''
            if not html_content:
                return []

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            rows = soup.find_all('tr', class_='js-event-item')

            events = []

            for row in rows:
                try:
                    currency_cell = row.find('td', class_='flagCur')
                    event_cell    = row.find('td', class_='event')
                    bull_icons    = row.find_all('i', class_='grayFullBullishIcon')

                    if not currency_cell or not event_cell:
                        continue
                    if len(bull_icons) < 3:
                        continue

                    currency = currency_cell.get_text(strip=True).replace('\xa0', '').strip()[-3:]
                    title    = event_cell.get_text(strip=True)

                    if not title or not currency or len(currency) != 3:
                        continue

                    evt_time = None
                    ts_attr = row.get('data-event-datetime')
                    if ts_attr:
                        ts_raw = str(ts_attr).strip()
                        # Investing can return either Unix epoch or formatted datetime.
                        for parser in (
                            lambda v: datetime.fromtimestamp(int(float(v)), tz=timezone.utc),
                            lambda v: datetime.strptime(v, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc),
                            lambda v: datetime.strptime(v, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc),
                            lambda v: datetime.strptime(v, '%Y/%m/%d %H:%M').replace(tzinfo=timezone.utc),
                        ):
                            try:
                                evt_time = parser(ts_raw)
                                break
                            except Exception:
                                continue

                    if evt_time is None:
                        time_cell = row.find('td', class_='time')
                        if not time_cell:
                            continue
                        time_str = time_cell.get_text(strip=True)
                        try:
                            evt_time = datetime.strptime(time_str, '%H:%M').replace(
                                year=now.year,
                                month=now.month,
                                day=now.day,
                                tzinfo=timezone.utc,
                            )
                            if evt_time < now - timedelta(hours=12):
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
