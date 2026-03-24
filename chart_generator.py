"""
Nixie Trades - Chart Generator
Role: Quant Software Engineer + Lead Architect

Generates annotated M15 candlestick chart images for setup alerts.
Live OHLCV data is passed in by the scheduler (fetched from MetaApi or MT5 worker).

Markups drawn (all at exact price coordinates):
  - Order Blocks        : amber (bullish) / rose-red (bearish)
  - Breaker Blocks      : emerald (bullish) / violet (bearish)
  - Fair Value Gaps     : semi-transparent horizontal bands
  - Break of Structure  : dashed purple horizontal lines
  - Entry / SL / TP1 / TP2 price level lines

Watermark: "NIXIE TRADES" centred behind candles, TradingView-style.
Output:    optimised PNG, target <= 250 KB.
"""

import io
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Non-interactive backend must be set BEFORE importing pyplot on headless servers.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection

try:
    from PIL import Image as _PIL_Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logging.getLogger(__name__).info(
        "Pillow not installed. PNG optimisation will be skipped. "
        "Run: pip install Pillow --break-system-packages"
    )

import config
import utils

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Renders annotated dark-theme candlestick charts for Nixie Trades setup alerts.

    Design principle: every markup is drawn at its raw float price so that
    the visual matches the signal numbers exactly — no rounding artefacts.

    Usage (called from scheduler after fetching live data):
        cg = ChartGenerator()
        png_bytes = cg.generate_setup_chart(live_df, setup_data, poi, ...)
    """

    # ── Figure dimensions ─────────────────────────────────────────────────────
    # 11 x 6.5 inches at 80 DPI = 880 x 520 px — ideal for Telegram's 800 px wide display.
    FIG_W  = 11.0
    FIG_H  = 6.5
    DPI    = 80

    # ── TradingView-style dark colour palette ─────────────────────────────────
    C_BG        = '#131722'
    C_GRID      = '#1f2533'
    C_BORDER    = '#2a2e39'
    C_TEXT      = '#d1d4dc'
    C_TEXT_DIM  = '#6b7280'

    # Candles
    C_UP    = '#26a69a'   # teal  (bullish)
    C_DOWN  = '#ef5350'   # red   (bearish)

    # SMC zones
    C_OB_BULL  = '#f59e0b'   # amber
    C_OB_BEAR  = '#f43f5e'   # rose-red
    C_BB_BULL  = '#10b981'   # emerald green
    C_BB_BEAR  = '#8b5cf6'   # violet
    C_FVG_BULL = '#22c55e'   # green
    C_FVG_BEAR = '#ef4444'   # red
    C_BOS      = '#7c3aed'   # deep purple

    # Price level lines
    C_ENTRY = '#60a5fa'   # sky blue
    C_SL    = '#ef4444'   # red
    C_TP1   = '#4ade80'   # light green
    C_TP2   = '#16a34a'   # dark green

    # ── Zone opacity ─────────────────────────────────────────────────────────
    A_OB_FILL  = 0.14
    A_BB_FILL  = 0.16
    A_FVG_FILL = 0.09
    A_ZONE_EDGE = 0.65

    # ── Candle geometry ───────────────────────────────────────────────────────
    BODY_W = 0.50   # fraction of one bar slot
    WICK_W = 0.80   # linewidth in points for wicks

    # ── Right-side label margin (in bar units) ────────────────────────────────
    # Allows entry/SL/TP labels to be printed without clipping.
    X_RIGHT_MARGIN = 8

    # ── How many bars to display on the chart ────────────────────────────────
    DISPLAY_BARS = 75

    def generate_setup_chart(
        self,
        data: pd.DataFrame,
        setup_data: Dict,
        poi: Dict,
        additional_pois: Optional[List[Dict]] = None,
        fvgs: Optional[List[Dict]] = None,
        bos_events: Optional[List[Dict]] = None,
    ) -> Optional[bytes]:
        """
        Render the complete annotated chart and return optimised PNG bytes.

        Args:
            data:            Live M15 OHLCV DataFrame with UTC DatetimeIndex.
                             Passed in fresh from MetaApi or MT5 worker by scheduler.
            setup_data:      Setup alert dict containing entry, stop_loss,
                             take_profit_1, take_profit_2, direction, symbol, etc.
            poi:             Primary Point of Interest (OB or BB) dict.
            additional_pois: Up to 4 extra unmitigated zone dicts.
            fvgs:            Fair Value Gap dicts from detect_fair_value_gaps().
            bos_events:      BOS event dicts from detect_break_of_structure().

        Returns:
            PNG image as bytes if successful, None if rendering fails.
        """
        if data is None or len(data) < 10:
            logger.warning(
                "Chart skipped: insufficient bars (%d).",
                len(data) if data is not None else 0)
            return None

        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Chart skipped: data has no DatetimeIndex.")
            return None

        try:
            plot_df = data.tail(self.DISPLAY_BARS).copy().reset_index(drop=False)
            # Keep 'time' column for label formatting; reset index to integer positions.
            if 'time' not in plot_df.columns and isinstance(data.index, pd.DatetimeIndex):
                plot_df.insert(0, 'time', data.tail(self.DISPLAY_BARS).index)

            n = len(plot_df)

            fig, (ax, ax_vol) = plt.subplots(
                2, 1,
                figsize=(self.FIG_W, self.FIG_H),
                dpi=self.DPI,
                gridspec_kw={'height_ratios': [5, 1], 'hspace': 0},
                facecolor=self.C_BG,
            )

            # ─ Style axes ──────────────────────────────────────────────────
            self._style_ax(ax)
            self._style_ax(ax_vol)

            # ─ Compute y-range from candles + level lines ─────────────────
            y_min, y_max = self._compute_y_range(plot_df, setup_data)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(-0.5, n - 0.5 + self.X_RIGHT_MARGIN)

            ax_vol.set_xlim(-0.5, n - 0.5 + self.X_RIGHT_MARGIN)

            # ─ Grid ────────────────────────────────────────────────────────
            self._draw_grid(ax, n, y_min, y_max)

            # ─ Watermark (zorder 2 — behind zones and candles) ───────────
            self._draw_watermark(ax, setup_data.get('symbol', ''))

            # ─ SMC overlays (zorder 3–4) ──────────────────────────────────
            if fvgs:
                self._draw_fvgs(ax, plot_df, fvgs, n, y_min, y_max)

            if bos_events:
                self._draw_bos(ax, bos_events, n)

            if additional_pois:
                for extra in (additional_pois or [])[:4]:
                    self._draw_zone(ax, plot_df, extra, n, primary=False)

            if poi:
                self._draw_zone(ax, plot_df, poi, n, primary=True)

            # ─ Price level lines (zorder 6) ────────────────────────────────
            self._draw_levels(ax, setup_data, n)

            # ─ Candles (zorder 5) ─────────────────────────────────────────
            self._draw_candles(ax, plot_df)

            # ─ Volume sub-panel ───────────────────────────────────────────
            self._draw_volume(ax_vol, plot_df, n)

            # ─ Time labels on x-axis ──────────────────────────────────────
            self._draw_time_labels(ax, plot_df, n)

            # ─ Header ─────────────────────────────────────────────────────
            self._draw_header(fig, setup_data)

            # ─ Render to bytes ────────────────────────────────────────────
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format='png',
                dpi=self.DPI,
                facecolor=self.C_BG,
                edgecolor='none',
                bbox_inches='tight',
                pad_inches=0.05,
            )
            plt.close(fig)

            return self._optimise_png(buf)

        except Exception as exc:
            logger.error("Chart generation error: %s", exc, exc_info=True)
            try:
                plt.close('all')
            except Exception:
                pass
            return None

    # =========================================================================
    # CANDLES
    # =========================================================================

    def _draw_candles(self, ax: plt.Axes, data: pd.DataFrame):
        """
        Draw candlesticks as filled rectangles + wicks.
        x positions are integer bar indices (0..n-1).
        y positions are raw float prices — no rounding.
        """
        for i, row in data.iterrows():
            try:
                o = float(row['open'])
                h = float(row['high'])
                l = float(row['low'])
                c = float(row['close'])
            except (KeyError, TypeError, ValueError):
                continue

            bull   = c >= o
            color  = self.C_UP if bull else self.C_DOWN
            b_bot  = min(o, c)
            b_top  = max(o, c)
            b_h    = max(b_top - b_bot, 1e-12)

            # Wick
            ax.plot(
                [i, i], [l, h],
                color=color, linewidth=self.WICK_W,
                solid_capstyle='round', zorder=5,
            )

            # Body
            rect = mpatches.Rectangle(
                (i - self.BODY_W / 2, b_bot),
                self.BODY_W, b_h,
                facecolor=color, edgecolor=color,
                linewidth=0.2, alpha=0.95, zorder=5,
            )
            ax.add_patch(rect)

    # =========================================================================
    # SMC ZONE DRAWING
    # =========================================================================

    def _draw_zone(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        poi: Dict,
        n: int,
        primary: bool = True,
    ):
        """
        Draw an Order Block or Breaker Block zone at exact price coordinates.

        The zone starts at the bar where the POI was formed (determined by
        matching the POI timestamp against the live data index).  If no match
        is found the zone covers the right-half of the chart — still at the
        correct price levels.
        """
        high = float(poi.get('high', 0))
        low  = float(poi.get('low',  0))
        if high <= low or high <= 0:
            return

        p_type    = str(poi.get('type', 'OB')).upper()
        direction = str(poi.get('direction', 'BULLISH')).upper()
        is_bull   = direction in ('BULLISH', 'BUY')

        if p_type in ('BB', 'BREAKER'):
            color = self.C_BB_BULL if is_bull else self.C_BB_BEAR
            label = 'BB'
            fill_a = self.A_BB_FILL * (1.0 if primary else 0.55)
        else:
            color = self.C_OB_BULL if is_bull else self.C_OB_BEAR
            label = 'OB'
            fill_a = self.A_OB_FILL * (1.0 if primary else 0.55)

        edge_a = self.A_ZONE_EDGE * (1.0 if primary else 0.50)

        # Map formation timestamp to bar index
        start_bar = self._ts_to_bar(data, poi.get('timestamp'))
        if start_bar is None:
            start_bar = max(0, n - 40)

        x0     = float(start_bar) - 0.5
        width  = float(n) - float(start_bar) + float(self.X_RIGHT_MARGIN)

        # Fill
        ax.add_patch(mpatches.Rectangle(
            (x0, low), width, high - low,
            facecolor=color, edgecolor='none',
            alpha=fill_a, zorder=3,
        ))
        # Edge outline
        ax.add_patch(mpatches.Rectangle(
            (x0, low), width, high - low,
            facecolor='none', edgecolor=color,
            linewidth=0.7 if primary else 0.4,
            alpha=edge_a, zorder=3,
        ))

        if primary:
            mid = (high + low) / 2
            ax.text(
                n + 0.3, mid,
                f' {label}',
                color=color, fontsize=7, fontweight='bold',
                va='center', ha='left', alpha=0.9, zorder=7,
            )

    # =========================================================================
    # FAIR VALUE GAPS
    # =========================================================================

    def _draw_fvgs(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        fvgs: List[Dict],
        n: int,
        y_min: float,
        y_max: float,
    ):
        """Draw FVG bands at exact gap price levels."""
        for fvg in fvgs:
            high = float(fvg.get('high', 0))
            low  = float(fvg.get('low',  0))
            if high <= low or high <= 0:
                continue
            # Only draw if the gap is visible in the current y-range
            if high < y_min or low > y_max:
                continue

            direction = str(fvg.get('direction', 'BULLISH')).upper()
            color     = self.C_FVG_BULL if direction in ('BULLISH', 'BUY') else self.C_FVG_BEAR

            start_bar = self._ts_to_bar(data, fvg.get('timestamp'))
            if start_bar is None:
                continue
            x0    = float(start_bar) - 0.5
            width = float(n) - float(start_bar) + float(self.X_RIGHT_MARGIN)

            ax.add_patch(mpatches.Rectangle(
                (x0, low), width, high - low,
                facecolor=color, edgecolor=color,
                alpha=self.A_FVG_FILL, linewidth=0.3, zorder=3,
            ))

    # =========================================================================
    # BREAK OF STRUCTURE LINES
    # =========================================================================

    def _draw_bos(self, ax: plt.Axes, bos_events: List[Dict], n: int):
        """Draw BOS as dashed horizontal lines at exact price levels."""
        seen = set()
        for bos in bos_events[:3]:
            level = float(bos.get('level', 0))
            if level <= 0 or level in seen:
                continue
            seen.add(level)

            ax.axhline(
                y=level,
                color=self.C_BOS,
                linewidth=0.65,
                linestyle=(0, (5, 5)),
                alpha=0.55, zorder=4,
            )
            ax.text(
                0.5, level,
                ' BOS ',
                transform=ax.get_yaxis_transform(),
                color=self.C_BOS, fontsize=6.5,
                va='bottom', ha='left', alpha=0.75, zorder=7,
            )

    # =========================================================================
    # PRICE LEVEL LINES (Entry / SL / TP)
    # =========================================================================

    def _draw_levels(self, ax: plt.Axes, setup_data: Dict, n: int):
        """Draw Entry, SL, TP1, TP2 as labelled horizontal lines."""
        symbol = setup_data.get('symbol', '')

        levels = [
            # (dict_key,       colour,      label,   linewidth, linestyle)
            ('entry_price',   self.C_ENTRY, 'ENTRY', 1.1, 'solid'),
            ('stop_loss',     self.C_SL,    'SL',    0.85, (0, (6, 3))),
            ('take_profit_1', self.C_TP1,   'TP1',   0.85, (0, (3, 2))),
            ('take_profit_2', self.C_TP2,   'TP2',   0.85, (0, (3, 2))),
        ]

        for key, color, tag, lw, ls in levels:
            price = setup_data.get(key)
            if price is None or float(price) <= 0:
                continue
            price = float(price)

            ax.axhline(
                y=price, color=color, linewidth=lw,
                linestyle=ls, alpha=0.88, zorder=6,
            )

            price_str = utils.format_price(symbol, price)
            ax.text(
                n + self.X_RIGHT_MARGIN - 0.3, price,
                f'{tag}  {price_str}',
                color=color, fontsize=7.5,
                va='center', ha='right',
                fontweight='bold' if tag == 'ENTRY' else 'normal',
                alpha=0.95, zorder=7,
            )

    # =========================================================================
    # VOLUME
    # =========================================================================

    def _draw_volume(self, ax_vol: plt.Axes, data: pd.DataFrame, n: int):
        """Draw volume bars in the lower sub-panel."""
        ax_vol.set_facecolor(self.C_BG)
        for sp in ax_vol.spines.values():
            sp.set_color(self.C_BORDER)
            sp.set_linewidth(0.4)

        if 'volume' not in data.columns:
            ax_vol.set_visible(False)
            return

        vols   = data['volume'].fillna(0).values
        closes = data['close'].values
        opens  = data['open'].values
        if vols.max() <= 0:
            ax_vol.set_visible(False)
            return

        colors = [self.C_UP if c >= o else self.C_DOWN
                  for c, o in zip(closes, opens)]

        for i, (v, c) in enumerate(zip(vols, colors)):
            ax_vol.bar(i, v, color=c, alpha=0.45, width=0.55, zorder=2)

        ax_vol.set_xlim(-0.5, n - 0.5 + self.X_RIGHT_MARGIN)
        ax_vol.set_ylim(0, vols.max() * 1.3)
        ax_vol.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False, labelright=False,
        )
        ax_vol.set_ylabel('Vol', color=self.C_TEXT_DIM,
                          fontsize=6.5, labelpad=2)

    # =========================================================================
    # WATERMARK (TradingView style — behind candles)
    # =========================================================================

    def _draw_watermark(self, ax: plt.Axes, symbol: str = ''):
        """
        Render 'NIXIE TRADES' centred behind all price data, zorder=2.
        Uses axes-fraction coordinates so it never shifts with price scale.
        """
        lines = ['NIXIE TRADES']
        if symbol:
            lines.append(f'M15  {symbol}')

        ax.text(
            0.5, 0.5,
            '\n'.join(lines),
            transform=ax.transAxes,
            color=self.C_TEXT,
            fontsize=26,
            fontweight='bold',
            ha='center', va='center',
            alpha=0.04,        # very faint — identical to TradingView watermark
            zorder=2,
            linespacing=1.6,
        )

    # =========================================================================
    # TIME LABELS
    # =========================================================================

    def _draw_time_labels(self, ax: plt.Axes, data: pd.DataFrame, n: int):
        """Print compact M15 time labels beneath the candles."""
        interval  = max(1, n // 7)
        y_label   = ax.get_ylim()[0]

        for i in range(0, n, interval):
            try:
                ts  = data.iloc[i]['time'] if 'time' in data.columns else None
                if ts is None:
                    continue
                lbl = pd.Timestamp(ts).strftime('%d %b\n%H:%M')
            except Exception:
                continue
            ax.text(
                i, y_label,
                lbl,
                color=self.C_TEXT_DIM, fontsize=6,
                ha='center', va='top', alpha=0.7, zorder=7,
            )

    # =========================================================================
    # HEADER BANNER
    # =========================================================================

    def _draw_header(self, fig: plt.Figure, setup_data: Dict):
        """Draw a one-line branding + setup summary above the chart area."""
        symbol    = setup_data.get('symbol', 'N/A')
        direction = 'LONG' if setup_data.get('direction', 'BUY') == 'BUY' else 'SHORT'
        tier_lbl  = ('UNICORN SETUP'
                     if 'UNICORN' in str(setup_data.get('setup_type', '')).upper()
                     else 'STANDARD SETUP')
        sig_num   = setup_data.get('signal_number', 0)
        ml_score  = setup_data.get('ml_score', 0)
        session   = setup_data.get('session', 'N/A')
        dir_color = self.C_UP if direction == 'LONG' else self.C_DOWN

        fig.text(
            0.012, 0.985,
            f'NIXIE TRADES  |  SETUP #{sig_num}  —  {tier_lbl}',
            color=self.C_TEXT, fontsize=9, fontweight='bold',
            ha='left', va='top', transform=fig.transFigure,
        )
        fig.text(
            0.012, 0.970,
            f'{symbol}   M15   {direction}   '
            f'AI Score: {ml_score}%   Session: {session}',
            color=dir_color, fontsize=8,
            ha='left', va='top', transform=fig.transFigure,
        )

    # =========================================================================
    # AXES STYLE
    # =========================================================================

    def _style_ax(self, ax: plt.Axes):
        """Apply dark theme to an axes object."""
        ax.set_facecolor(self.C_BG)
        for sp in ax.spines.values():
            sp.set_color(self.C_BORDER)
            sp.set_linewidth(0.4)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(
            axis='y', which='both',
            right=True, left=False,
            labelright=True, labelleft=False,
            colors=self.C_TEXT_DIM, labelsize=7,
        )
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    def _draw_grid(
        self,
        ax: plt.Axes,
        n: int,
        y_min: float,
        y_max: float,
    ):
        """Draw subtle horizontal grid lines."""
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(
            axis='y', which='major',
            color=self.C_GRID, linewidth=0.35, alpha=0.7, zorder=0,
        )

    # =========================================================================
    # Y-RANGE HELPER
    # =========================================================================

    def _compute_y_range(
        self, data: pd.DataFrame, setup_data: Dict
    ) -> tuple:
        """
        Compute y-axis min/max that includes all candle prices and level lines.
        Adds 8% padding so labels are not clipped.
        """
        prices: List[float] = []

        for col in ('high', 'low', 'close', 'open'):
            if col in data.columns:
                prices.extend(data[col].dropna().tolist())

        for key in ('entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2'):
            v = setup_data.get(key)
            if v:
                try:
                    prices.append(float(v))
                except (TypeError, ValueError):
                    pass

        if not prices:
            return 1.0, 2.0

        p_min = min(prices)
        p_max = max(prices)
        pad   = (p_max - p_min) * 0.08
        return p_min - pad, p_max + pad

    # =========================================================================
    # TIMESTAMP-TO-BAR MAPPING
    # =========================================================================

    def _ts_to_bar(
        self, data: pd.DataFrame, ts
    ) -> Optional[int]:
        """
        Map a timestamp to the nearest integer bar index in `data`.
        Uses the 'time' column (reset index copy) if present.

        Returns None if the timestamp is absent, unparseable, or falls
        outside the chart's time range.
        """
        if ts is None:
            return None

        time_col = 'time' if 'time' in data.columns else None
        if time_col is None:
            return None

        try:
            ts_pd = pd.Timestamp(ts)
            if ts_pd.tzinfo is None:
                ts_pd = ts_pd.tz_localize('UTC')
            else:
                ts_pd = ts_pd.tz_convert('UTC')

            times = pd.to_datetime(data[time_col], utc=True, errors='coerce')
            diffs = (times - ts_pd).abs()
            idx   = int(diffs.idxmin())
            return idx
        except Exception:
            return None

    # =========================================================================
    # PNG OPTIMISATION
    # =========================================================================

    def _optimise_png(self, raw_buf: io.BytesIO) -> bytes:
        """
        Post-process the matplotlib output with PIL to reduce file size.
        Converts to RGB (drops alpha), then saves with compress_level=7.
        Falls back to raw bytes if PIL is unavailable.

        Typical output: 80–200 KB for an 880 x 520 px chart.
        """
        raw_buf.seek(0)

        if not _PIL_AVAILABLE:
            return raw_buf.read()

        try:
            img     = _PIL_Image.open(raw_buf).convert('RGB')
            out_buf = io.BytesIO()
            img.save(out_buf, 'PNG', optimize=True, compress_level=7)
            out_buf.seek(0)
            result = out_buf.read()
            logger.debug("PNG optimised: %d KB", len(result) // 1024)
            return result
        except Exception as exc:
            logger.warning("PIL optimisation failed (%s). Using raw PNG.", exc)
            raw_buf.seek(0)
            return raw_buf.read()
        
    @staticmethod
    def generate_sample_chart() -> Optional[bytes]:
        """
        Generate a pre-rendered sample M15 EURUSD bullish OB setup chart
        for use in /start and /help command responses.
        Uses purely synthetic data — no live feed required.
        This is a one-time generation cached at module level after first call.
        """
        try:
            import numpy as _np
            import pandas as _pd

            _np.random.seed(42)
            n    = 75
            base = 1.08200
            prices = [base]

            for i in range(1, n):
                if i < 20:
                    # Asian session: tight ranging
                    move = _np.random.normal(0.0000, 0.00015)
                elif i < 32:
                    # London open: bullish impulse creating the OB
                    move = _np.random.normal(0.00038, 0.00014)
                elif i < 48:
                    # Pullback to OB zone (inducement sweep)
                    move = _np.random.normal(-0.00020, 0.00011)
                elif i < 58:
                    # Consolidation above demand zone
                    move = _np.random.normal(0.00006, 0.00009)
                else:
                    # New bullish leg beginning
                    move = _np.random.normal(0.00028, 0.00013)
                prices.append(max(prices[-1] + move, base * 0.994))

            rows  = []
            times = _pd.date_range(
                '2026-03-27 00:00', periods=n, freq='15min', tz='UTC')

            for i in range(n):
                c = prices[i]
                o = prices[i - 1] if i > 0 else c
                spread_h = abs(_np.random.normal(0.00014, 0.00007))
                spread_l = abs(_np.random.normal(0.00014, 0.00007))
                h = max(o, c) + spread_h
                l = min(o, c) - spread_l
                base_vol = 1800 if 20 <= i < 32 else 700
                v = max(100, int(abs(_np.random.normal(base_vol, base_vol * 0.35))))
                rows.append({
                    'open': round(o, 5), 'high': round(h, 5),
                    'low':  round(l, 5), 'close': round(c, 5),
                    'volume': v,
                })

            df = _pd.DataFrame(rows, index=times)
            df.index.name = 'time'

            # Build OB at the last down-candle before the London impulse (bar 19)
            ob_high = round(prices[19] + 0.00012, 5)
            ob_low  = round(prices[19] - 0.00018, 5)

            sample_poi = {
                'type':         'OB',
                'direction':    'BULLISH',
                'high':         ob_high,
                'low':          ob_low,
                'timestamp':    times[19],
                'index':        19,
                'volume_ratio': 2.3,
                'impulse_pips': 38.0,
                'confidence':   78,
            }

            # Entry at top of OB (limit order at zone boundary)
            entry = ob_high
            sl    = round(ob_low - 0.00035, 5)
            risk  = entry - sl
            tp1   = round(entry + risk * 2.5, 5)
            tp2   = round(entry + risk * 5.0, 5)

            sample_setup = {
                'symbol':        'EURUSD',
                'direction':     'BUY',
                'setup_type':    'UNICORN SETUP',
                'signal_number': 7,
                'session':       'London',
                'ml_score':      79,
                'entry_price':   entry,
                'stop_loss':     sl,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
            }

            bos_level    = round(max(prices[10:20]) + 0.00006, 5)
            sample_bos   = [{'level': bos_level, 'direction': 'BULLISH'}]

            cg = ChartGenerator()
            return cg.generate_setup_chart(
                data=df,
                setup_data=sample_setup,
                poi=sample_poi,
                bos_events=sample_bos,
            )

        except Exception as exc:
            logger.error(
                "Sample chart generation failed: %s", exc, exc_info=True)
            return None