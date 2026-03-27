"""
Nixie Trades - Chart Generator v2
Role: Quant Software Engineer

TradingView-dark-style annotated M15 candlestick chart for setup alerts.
Live OHLCV data is passed in by the scheduler (MetaApi or MT5 worker).

Changes from v1:
  - 150 DPI (was 80) — eliminates blur on all screen sizes
  - Full price axis on the right with all levels visible like TradingView
  - Visible shaded SMC zone boxes with thick coloured borders
  - Correct zone geometry: zones extend from formation bar to right edge
  - Entry / SL / TP lines with price labels that never overlap
  - NIXIE TRADES watermark behind candles at low opacity
  - PIL post-processing at compress_level=7, RGB output, target <= 300 KB
"""

import io
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

try:
    from PIL import Image as _PIL_Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

import config
import utils

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Renders annotated dark-theme candlestick charts for Nixie Trades alerts.
    Every markup is drawn at its raw float price so that the visual
    matches the signal numbers exactly.
    """

    # Figure dimensions: 12 x 7 inches at 150 DPI = 1800 x 1050 px
    FIG_W = 12.0
    FIG_H = 7.0
    DPI   = 150

    # TradingView dark colour palette
    C_BG       = '#131722'
    C_GRID     = '#1e2235'
    C_BORDER   = '#2a2e39'
    C_TEXT     = '#d1d4dc'
    C_TEXT_DIM = '#787b86'

    C_UP   = '#26a69a'
    C_DOWN = '#ef5350'

    C_OB_BULL  = '#f59e0b'
    C_OB_BEAR  = '#f43f5e'
    C_BB_BULL  = '#10b981'
    C_BB_BEAR  = '#a855f7'
    C_FVG_BULL = '#22c55e'
    C_FVG_BEAR = '#ef4444'
    C_BOS_LINE = '#7c3aed'

    C_ENTRY = '#60a5fa'
    C_SL    = '#ef4444'
    C_TP1   = '#4ade80'
    C_TP2   = '#16a34a'

    ZONE_FILL_ALPHA   = 0.22
    ZONE_BORDER_ALPHA = 0.92
    ZONE_BORDER_LW    = 1.8
    FVG_FILL_ALPHA    = 0.12

    BODY_W  = 0.52
    WICK_LW = 1.0

    DISPLAY_BARS   = 80
    LABEL_OFFSET   = 7
    Y_PAD_FRACTION = 0.06

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
            setup_data:      Setup alert dict (entry, stop_loss, tp1, tp2).
            poi:             Primary Point of Interest dict.
            additional_pois: Up to 4 extra unmitigated zone dicts.
            fvgs:            FVG dicts from detect_fair_value_gaps().
            bos_events:      BOS dicts from detect_break_of_structure().

        Returns:
            PNG bytes or None on failure.
        """
        if data is None or len(data) < 10:
            logger.warning(
                "Chart skipped: insufficient bars (%d).",
                len(data) if data is not None else 0)
            return None

        try:
            tail       = data.tail(self.DISPLAY_BARS).copy()
            time_index = list(tail.index)
            tail       = tail.reset_index(drop=True)
            n          = len(tail)
            x_right    = n + self.LABEL_OFFSET
            symbol     = setup_data.get('symbol', 'FOREX')
            decimals   = self._price_decimals(symbol)

            fig, (ax, ax_vol) = plt.subplots(
                2, 1,
                figsize=(self.FIG_W, self.FIG_H),
                dpi=self.DPI,
                gridspec_kw={'height_ratios': [5.5, 1], 'hspace': 0},
                facecolor=self.C_BG,
            )
            fig.subplots_adjust(
                left=0.005, right=0.84, top=0.915, bottom=0.04)

            y_min, y_max = self._compute_y_range(tail, setup_data)

            self._style_main_ax(ax, y_min, y_max, n, x_right, decimals)
            self._style_vol_ax(ax_vol, n, x_right)

            self._draw_watermark(ax, symbol)

            if fvgs:
                for fvg in fvgs:
                    self._draw_fvg(ax, fvg, time_index, n, x_right,
                                   y_min, y_max)

            if bos_events:
                self._draw_bos_lines(ax, bos_events)

            for extra in (additional_pois or [])[:4]:
                self._draw_zone(ax, extra, time_index, n, x_right,
                                primary=False)

            if poi:
                self._draw_zone(ax, poi, time_index, n, x_right,
                                primary=True)

            self._draw_price_levels(ax, setup_data, n, x_right,
                                    decimals, y_min, y_max)
            self._draw_candles(ax, tail)
            self._draw_volume(ax_vol, tail, n)
            self._draw_time_labels(ax, tail, time_index, n, y_min)
            self._draw_header(fig, setup_data)

            buf = io.BytesIO()
            fig.savefig(
                buf, format='png', dpi=self.DPI,
                facecolor=self.C_BG, edgecolor='none',
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
    # AXES SETUP
    # =========================================================================

    def _style_main_ax(
        self,
        ax: plt.Axes,
        y_min: float,
        y_max: float,
        n: int,
        x_right: int,
        decimals: int,
    ):
        ax.set_facecolor(self.C_BG)
        ax.set_xlim(-0.5, x_right)
        ax.set_ylim(y_min, y_max)

        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter(f'%.{decimals}f')
        )

        price_range = y_max - y_min
        raw_step    = price_range / 9.0
        if raw_step > 0:
            magnitude = 10 ** int(np.floor(np.log10(raw_step)))
            nice_step = round(raw_step / magnitude) * magnitude
            if nice_step <= 0:
                nice_step = raw_step
            ax.yaxis.set_major_locator(mticker.MultipleLocator(nice_step))

        ax.tick_params(
            axis='y',
            right=True, left=False,
            labelright=True, labelleft=False,
            colors=self.C_TEXT, labelsize=7.5,
            length=3, width=0.5, pad=4,
        )
        ax.tick_params(
            axis='x', which='both',
            bottom=False, top=False, labelbottom=False,
        )
        ax.yaxis.grid(
            True, which='major',
            color=self.C_GRID, linewidth=0.45, alpha=0.85, zorder=0,
        )
        ax.xaxis.grid(False)

    def _style_vol_ax(self, ax_vol: plt.Axes, n: int, x_right: int):
        ax_vol.set_facecolor(self.C_BG)
        ax_vol.set_xlim(-0.5, x_right)
        for sp in ax_vol.spines.values():
            sp.set_visible(False)
        ax_vol.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False, labelright=False,
        )

    # =========================================================================
    # CANDLES
    # =========================================================================

    def _draw_candles(self, ax: plt.Axes, data: pd.DataFrame):
        for i in range(len(data)):
            row = data.iloc[i]
            try:
                o  = float(row['open'])
                h  = float(row['high'])
                lo = float(row['low'])
                c  = float(row['close'])
            except (KeyError, TypeError, ValueError):
                continue

            bull  = c >= o
            color = self.C_UP if bull else self.C_DOWN
            b_bot = min(o, c)
            b_top = max(o, c)
            b_h   = max(b_top - b_bot, 1e-10)

            ax.plot(
                [i, i], [lo, h],
                color=color, linewidth=self.WICK_LW,
                solid_capstyle='butt', zorder=7,
            )
            ax.add_patch(mpatches.Rectangle(
                (i - self.BODY_W / 2, b_bot),
                self.BODY_W, b_h,
                facecolor=color, edgecolor='none',
                alpha=0.93, zorder=7,
            ))

    # =========================================================================
    # SMC ZONE BOXES
    # =========================================================================

    def _draw_zone(
        self,
        ax: plt.Axes,
        poi: Dict,
        time_index: list,
        n: int,
        x_right: int,
        primary: bool = True,
    ):
        """
        Draw a fully shaded zone box from formation bar to the right edge.
        Top and bottom border lines are drawn separately for maximum visibility.
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
        elif p_type == 'UNICORN':
            color = self.C_BB_BULL if is_bull else self.C_BB_BEAR
            label = 'UNICORN'
        else:
            color = self.C_OB_BULL if is_bull else self.C_OB_BEAR
            label = 'OB'

        fill_a   = self.ZONE_FILL_ALPHA   * (1.0 if primary else 0.50)
        border_a = self.ZONE_BORDER_ALPHA * (1.0 if primary else 0.55)
        border_w = self.ZONE_BORDER_LW    * (1.0 if primary else 0.55)

        start_x = self._ts_to_bar_index(poi.get('timestamp'), time_index)
        if start_x is None:
            start_x = max(0, n - 38)

        box_left  = float(start_x) - 0.5
        box_width = float(x_right) - box_left

        # Filled rectangle
        ax.add_patch(mpatches.Rectangle(
            (box_left, low), box_width, high - low,
            facecolor=color, edgecolor='none',
            alpha=fill_a, zorder=5,
        ))
        # Top border
        ax.plot(
            [box_left, x_right], [high, high],
            color=color, linewidth=border_w,
            alpha=border_a, zorder=6,
        )
        # Bottom border
        ax.plot(
            [box_left, x_right], [low, low],
            color=color, linewidth=border_w,
            alpha=border_a, zorder=6,
        )

        if primary:
            ax.text(
                x_right - 0.5, (high + low) / 2.0,
                f' {label}',
                color=color, fontsize=8.0,
                fontweight='bold',
                va='center', ha='right',
                alpha=0.95, zorder=8,
            )

    # =========================================================================
    # FAIR VALUE GAPS
    # =========================================================================

    def _draw_fvg(
        self,
        ax: plt.Axes,
        fvg: Dict,
        time_index: list,
        n: int,
        x_right: int,
        y_min: float,
        y_max: float,
    ):
        high = float(fvg.get('high', 0))
        low  = float(fvg.get('low',  0))
        if high <= low or high < y_min or low > y_max:
            return

        direction = str(fvg.get('direction', 'BULLISH')).upper()
        color     = (self.C_FVG_BULL
                     if direction in ('BULLISH', 'BUY')
                     else self.C_FVG_BEAR)

        start_x = self._ts_to_bar_index(fvg.get('timestamp'), time_index)
        if start_x is None:
            return

        width = float(x_right) - float(start_x) + 0.5
        ax.add_patch(mpatches.Rectangle(
            (float(start_x) - 0.5, low), width, high - low,
            facecolor=color, edgecolor=color,
            alpha=self.FVG_FILL_ALPHA, linewidth=0.35, zorder=3,
        ))

    # =========================================================================
    # BOS LINES
    # =========================================================================

    def _draw_bos_lines(self, ax: plt.Axes, bos_events: List[Dict]):
        seen: set = set()
        for bos in bos_events[:3]:
            level = float(bos.get('level', 0))
            if level <= 0 or level in seen:
                continue
            seen.add(level)
            ax.axhline(
                y=level,
                color=self.C_BOS_LINE,
                linewidth=0.75,
                linestyle=(0, (6, 4)),
                alpha=0.65, zorder=4,
            )
            ax.text(
                0.5, level,
                ' BOS',
                color=self.C_BOS_LINE, fontsize=6.5,
                va='bottom', ha='left',
                alpha=0.80, zorder=8,
            )

    # =========================================================================
    # PRICE LEVEL LINES
    # =========================================================================

    def _draw_price_levels(
        self,
        ax: plt.Axes,
        setup_data: Dict,
        n: int,
        x_right: int,
        decimals: int,
        y_min: float,
        y_max: float,
    ):
        """
        Draw Entry, SL, TP1, TP2 as full-width horizontal lines.
        Price labels are placed outside the chart on the right.
        Vertical positions are staggered to prevent label overlap.
        """
        y_range = max(y_max - y_min, 1e-10)
        min_sep = y_range * 0.024

        levels = [
            ('entry_price',   self.C_ENTRY, 'ENTRY', 1.5, 'solid'),
            ('stop_loss',     self.C_SL,    'SL',    1.0, (0, (5, 3))),
            ('take_profit_1', self.C_TP1,   'TP1',   1.0, (0, (3, 2))),
            ('take_profit_2', self.C_TP2,   'TP2',   1.0, (0, (3, 2))),
        ]

        rendered: List[float] = []

        for key, color, tag, lw, ls in levels:
            price = setup_data.get(key)
            if price is None:
                continue
            price = float(price)
            if price <= 0:
                continue

            ax.plot(
                [-0.5, x_right],
                [price, price],
                color=color, linewidth=lw,
                linestyle=ls, alpha=0.90, zorder=6,
            )
            ax.plot(
                x_right - 0.2, price,
                marker='<', color=color,
                markersize=5, alpha=0.90, zorder=6,
            )

            label_y = price
            for prev_y in rendered:
                if abs(label_y - prev_y) < min_sep:
                    label_y = prev_y + min_sep * (
                        1 if label_y >= prev_y else -1)
            rendered.append(label_y)

            price_fmt = f'%.{decimals}f' % price
            ax.text(
                x_right + 0.25, label_y,
                f'{tag}  {price_fmt}',
                color=color,
                fontsize=7.5,
                fontweight='bold' if tag == 'ENTRY' else 'normal',
                va='center', ha='left',
                alpha=0.96, zorder=8,
            )

    # =========================================================================
    # VOLUME
    # =========================================================================

    def _draw_volume(self, ax_vol: plt.Axes, data: pd.DataFrame, n: int):
        if 'volume' not in data.columns:
            ax_vol.set_visible(False)
            return
        vols   = data['volume'].fillna(0).values.astype(float)
        closes = data['close'].values
        opens  = data['open'].values
        if vols.max() <= 0:
            ax_vol.set_visible(False)
            return
        colors = [self.C_UP if c >= o else self.C_DOWN
                  for c, o in zip(closes, opens)]
        for i, (v, c) in enumerate(zip(vols, colors)):
            ax_vol.bar(i, v, color=c, alpha=0.40, width=0.52, zorder=2)
        ax_vol.set_ylim(0, vols.max() * 1.6)
        ax_vol.text(
            0.5, 0.85, 'Volume',
            transform=ax_vol.transAxes,
            color=self.C_TEXT_DIM, fontsize=6.5,
            ha='left', va='top', alpha=0.55,
        )

    # =========================================================================
    # WATERMARK
    # =========================================================================

    def _draw_watermark(self, ax: plt.Axes, symbol: str = ''):
        text = f'NIXIE TRADES\nM15  {symbol}' if symbol else 'NIXIE TRADES'
        ax.text(
            0.5, 0.50, text,
            transform=ax.transAxes,
            color=self.C_TEXT, fontsize=28, fontweight='bold',
            ha='center', va='center',
            alpha=0.045, linespacing=1.55, zorder=2,
        )

    # =========================================================================
    # TIME LABELS
    # =========================================================================

    def _draw_time_labels(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        time_index: list,
        n: int,
        y_min: float,
    ):
        interval = max(1, n // 8)
        for i in range(0, n, interval):
            try:
                ts  = time_index[i]
                lbl = pd.Timestamp(ts).strftime('%d %b\n%H:%M')
            except Exception:
                continue
            ax.text(
                i, y_min,
                lbl,
                color=self.C_TEXT_DIM, fontsize=6.0,
                ha='center', va='top',
                alpha=0.75, zorder=8,
            )

    # =========================================================================
    # HEADER
    # =========================================================================

    def _draw_header(self, fig: plt.Figure, setup_data: Dict):
        symbol    = setup_data.get('symbol', 'N/A')
        direction = ('LONG'
                     if setup_data.get('direction', 'BUY') == 'BUY'
                     else 'SHORT')
        tier_lbl  = (
            'UNICORN SETUP'
            if 'UNICORN' in str(setup_data.get('setup_type', '')).upper()
            else 'STANDARD SETUP'
        )
        sig_num   = setup_data.get('signal_number', 0)
        ml_score  = setup_data.get('ml_score', 0)
        session   = setup_data.get('session', 'N/A')
        dir_color = self.C_UP if direction == 'LONG' else self.C_DOWN

        fig.text(
            0.010, 0.984,
            f'NIXIE TRADES  |  SETUP #{sig_num}  -  {tier_lbl}',
            color=self.C_TEXT, fontsize=9.5, fontweight='bold',
            ha='left', va='top', transform=fig.transFigure,
        )
        fig.text(
            0.010, 0.966,
            f'{symbol}   M15   {direction}   '
            f'AI Score: {ml_score}%   Session: {session}',
            color=dir_color, fontsize=8.5,
            ha='left', va='top', transform=fig.transFigure,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _compute_y_range(
        self, data: pd.DataFrame, setup_data: Dict
    ) -> tuple:
        prices: list = []
        for col in ('high', 'low', 'close', 'open'):
            if col in data.columns:
                prices.extend([
                    float(v) for v in data[col].dropna()
                    if float(v) > 0
                ])
        for key in ('entry_price', 'stop_loss',
                    'take_profit_1', 'take_profit_2'):
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
        pad   = (p_max - p_min) * self.Y_PAD_FRACTION
        return p_min - pad, p_max + pad

    def _ts_to_bar_index(
        self, ts, time_index: list
    ) -> Optional[int]:
        if ts is None or not time_index:
            return None
        try:
            ts_pd = pd.Timestamp(ts)
            if ts_pd.tzinfo is None:
                import pytz
                ts_pd = pytz.utc.localize(ts_pd)
            else:
                ts_pd = ts_pd.tz_convert('UTC')

            best_i    = None
            best_diff = None
            for i, t in enumerate(time_index):
                try:
                    t_pd = pd.Timestamp(t)
                    if t_pd.tzinfo is None:
                        import pytz
                        t_pd = pytz.utc.localize(t_pd)
                    else:
                        t_pd = t_pd.tz_convert('UTC')
                    diff = abs((t_pd - ts_pd).total_seconds())
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_i    = i
                except Exception:
                    continue
            return best_i
        except Exception:
            return None

    def _price_decimals(self, symbol: str) -> int:
        s = symbol.upper()
        if 'XAU' in s or 'XAG' in s:
            return 2
        if 'JPY' in s:
            return 3
        if 'BTC' in s:
            return 1
        return 5

    def _optimise_png(self, raw_buf: io.BytesIO) -> bytes:
        raw_buf.seek(0)
        if not _PIL_AVAILABLE:
            return raw_buf.read()
        try:
            img     = _PIL_Image.open(raw_buf).convert('RGB')
            out_buf = io.BytesIO()
            img.save(out_buf, 'PNG', optimize=True, compress_level=7)
            out_buf.seek(0)
            result = out_buf.read()
            logger.debug("Chart PNG: %d KB", len(result) // 1024)
            return result
        except Exception as exc:
            logger.warning("PIL optimisation failed: %s. Using raw PNG.", exc)
            raw_buf.seek(0)
            return raw_buf.read()

    # =========================================================================
    # SAMPLE CHART
    # =========================================================================

    @staticmethod
    def generate_sample_chart() -> Optional[bytes]:
        """
        Generate a synthetic XAUUSD M15 Breaker Block LONG sample chart.
        No live data required. Called once and cached in bot.py.
        """
        try:
            rng  = np.random.default_rng(42)
            n    = 80
            base = 2485.0

            prices = [base]
            for i in range(1, n):
                if i < 20:
                    move = rng.normal(0.0,  0.25)
                elif i < 35:
                    move = rng.normal(-1.5, 0.60)
                elif i < 42:
                    move = rng.normal(-0.5, 0.35)
                elif i < 52:
                    move = rng.normal(3.8,  0.80)
                elif i < 62:
                    move = rng.normal(-1.2, 0.45)
                else:
                    move = rng.normal(1.8,  0.55)
                prices.append(max(prices[-1] + move, base * 0.985))

            times = pd.date_range(
                '2026-03-27 07:00', periods=n, freq='15min', tz='UTC')
            rows = []
            for i in range(n):
                c  = prices[i]
                o  = prices[i - 1] if i > 0 else c
                h  = max(o, c) + abs(rng.normal(0, 0.45))
                lo = min(o, c) - abs(rng.normal(0, 0.45))
                mul = 2200 if 42 <= i < 52 else 900
                v   = max(100, int(abs(rng.normal(mul, mul * 0.35))))
                rows.append({
                    'open': round(o, 2), 'high': round(h, 2),
                    'low':  round(lo, 2), 'close': round(c, 2),
                    'volume': v,
                })

            df = pd.DataFrame(rows, index=times)
            df.index.name = 'time'

            bb_high = round(prices[40] + 1.8, 2)
            bb_low  = round(prices[40] - 2.5, 2)
            bos_lvl = round(max(prices[20:42]) + 0.8, 2)
            entry   = bb_high
            sl      = round(bb_low - 3.5, 2)
            risk    = entry - sl
            tp1     = round(entry + risk * 2.5, 2)
            tp2     = round(entry + risk * 5.0, 2)

            sample_poi = {
                'type':         'BB',
                'direction':    'BULLISH',
                'high':         bb_high,
                'low':          bb_low,
                'timestamp':    times[41],
                'index':        41,
                'volume_ratio': 2.6,
                'impulse_pips': 52.0,
                'confidence':   82,
            }
            sample_setup = {
                'symbol':        'XAUUSD',
                'direction':     'BUY',
                'setup_type':    'STANDARD SETUP',
                'signal_number': 11,
                'session':       'London',
                'ml_score':      62,
                'entry_price':   entry,
                'stop_loss':     sl,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
            }
            sample_bos = [{'level': bos_lvl, 'direction': 'BULLISH'}]

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