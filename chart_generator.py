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

    # Wide-format chart canvas so Telegram users effectively see a full-screen
    # chart with minimal empty margins while preserving room for right-side labels.
    FIG_W = 15.5
    FIG_H = 8.7
    DPI   = 180

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
    LABEL_OFFSET   = 10
    Y_PAD_FRACTION = 0.06
    POSITION_MIN_WIDTH = 18

    def generate_setup_chart(
        self,
        data: pd.DataFrame,
        setup_data: Dict,
        poi: Dict,
        refined_pois: Optional[List[Dict]] = None,
        additional_pois: Optional[List[Dict]] = None,
        fvgs: Optional[List[Dict]] = None,
        bos_events: Optional[List[Dict]] = None,
        swing_levels: Optional[List[Dict]] = None,
        inducement_data: Optional[Dict] = None,
        htf_swing_high: float = 0.0,
        htf_swing_low: float = 0.0,
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
                left=0.004, right=0.90, top=0.95, bottom=0.04)

            y_min, y_max = self._compute_y_range(tail, setup_data)

            self._style_main_ax(ax, y_min, y_max, n, x_right, decimals)
            self._style_vol_ax(ax_vol, n, x_right)

            _chart_tf_label = str(setup_data.get('chart_timeframe', 'M15')).upper()
            self._draw_watermark(ax, symbol, _chart_tf_label)

            # Draw premium/discount zone background shading first so all
            # price structure sits on top of it.
            if htf_swing_high > 0 and htf_swing_low > 0 and htf_swing_high > htf_swing_low:
                _is_buy_dir = str(setup_data.get('direction', 'BUY')).upper() == 'BUY'
                self._draw_premium_discount_zone(
                    ax, htf_swing_high, htf_swing_low, _is_buy_dir, y_min, y_max)

            if fvgs:
                for fvg in fvgs:
                    self._draw_fvg(ax, fvg, time_index, n, x_right,
                                   y_min, y_max)

            if bos_events:
                self._draw_bos_lines(ax, bos_events, time_index, n, x_right)

            # Draw secondary M15 zones (other OBs and BBs detected on the
            # same timeframe as the chart data — bar positions are accurate).
            for extra in (additional_pois or [])[:6]:
                self._draw_zone(ax, extra, time_index, n, x_right,
                                primary=False)

            # Draw the primary entry zone last so it renders on top.
            if poi:
                self._draw_zone(ax, poi, time_index, n, x_right,
                                primary=True)

            # Draw swing structure markers so users can see the HH/HL/LH/LL
            # sequence that informed the BOS and CHOCH detection.
            if swing_levels:
                self._draw_swing_levels(ax, swing_levels, time_index, n,
                                        y_min, y_max)

            # Draw IDM (inducement sweep) marker on the exact sweep candle.
            if inducement_data:
                self._draw_inducement(ax, inducement_data, time_index, n,
                                      x_right, y_min, y_max)

            _entry_px  = float(setup_data.get('entry_price', 0))
            _entry_dir = str(setup_data.get('direction', 'BUY'))
            position_anchor = self._resolve_position_anchor(
                poi, refined_pois or [], time_index, n,
                data=tail,
                entry_price=_entry_px,
                direction=_entry_dir,
                order_type=str(setup_data.get('order_type', 'LIMIT')),
            )
            self._draw_price_levels(ax, setup_data, n, x_right,
                                    decimals, y_min, y_max,
                                    anchor_x=position_anchor,
                                    last_close=float(tail.iloc[-1]['close']),
                                    last_open=float(tail.iloc[-1]['open']))
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
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(nice_step / 2.0))

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
        ax.tick_params(
            axis='y', which='minor',
            right=True, left=False,
            colors=self.C_TEXT_DIM,
            length=2, width=0.3,
        )
        ax.yaxis.grid(
            True, which='major',
            color=self.C_GRID, linewidth=0.45, alpha=0.85, zorder=0,
        )
        ax.yaxis.grid(
            True, which='minor',
            color=self.C_GRID, linewidth=0.22, alpha=0.32, zorder=0,
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
        timeframe = str(poi.get('timeframe', '')).upper()
        role      = str(poi.get('role', '')).upper()
        is_bull   = direction in ('BULLISH', 'BUY')
        is_refined = role == 'REFINEMENT'

        if p_type in ('BB', 'BREAKER'):
            color = self.C_BB_BULL if is_bull else self.C_BB_BEAR
            label = 'BB'
        elif p_type == 'UNICORN':
            color = self.C_BB_BULL if is_bull else self.C_BB_BEAR
            label = 'UNICORN'
        else:
            color = self.C_OB_BULL if is_bull else self.C_OB_BEAR
            label = 'OB'

        tf_scale = {
            'H1': 1.00,
            'M15': 0.78,
            'M5': 0.62,
        }.get(timeframe, 0.72 if is_refined else 1.0)
        fill_a   = self.ZONE_FILL_ALPHA   * (1.0 if primary else (0.78 if is_refined else 0.50))
        border_a = self.ZONE_BORDER_ALPHA * (1.0 if primary else (0.90 if is_refined else 0.55))
        border_w = self.ZONE_BORDER_LW    * tf_scale * (1.0 if primary else (0.95 if is_refined else 0.55))

        start_x = self._ts_to_bar_index(poi.get('timestamp'), time_index)
        if start_x is None:
            # Timestamp is outside the chart window — zone formed before
            # visible bars. Draw from the left edge to show it as an
            # established zone, not a recently formed one at the right edge.
            start_x = 0
        start_x = max(0, min(start_x, n - 1))

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

        show_label = primary or is_refined
        if show_label:
            label_prefix = f'{timeframe} ' if timeframe else ''
            ax.text(
                x_right - 0.5, (high + low) / 2.0,
                f' {label_prefix}{label}',
                color=color, fontsize=8.0,
                fontweight='bold' if primary else 'normal',
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
            # FVG timestamp has no matching bar in the chart window.
            # This occurs when the FVG formed before the visible candles
            # or during a weekend gap. Skip to avoid phantom zone boxes.
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

    def _draw_bos_lines(
        self,
        ax: plt.Axes,
        bos_events: List[Dict],
        time_index: list,
        n: int,
        x_right: int,
    ):
        """
        Draw a short horizontal line precisely at the bar where the structure
        broke. The line spans 10 bars before to 12 bars after the break candle
        so the user can clearly see which candle caused the BOS and what level
        was taken out. A faint dotted extension tracks the level to the right
        edge without dominating the chart visually.
        """
        seen:      set   = set()
        ylim             = ax.get_ylim()
        _y_range         = max(ylim[1] - ylim[0], 1e-10)
        _tick_size       = _y_range * 0.012
        _label_offset    = _y_range * 0.016

        for bos in bos_events[:4]:
            level = float(bos.get('level', 0))
            if level <= 0 or level in seen:
                continue
            seen.add(level)

            bos_bar = self._ts_to_bar_index(bos.get('timestamp'), time_index)
            if bos_bar is None:
                # No matching candle in the chart window for this BOS timestamp.
                # This happens when the BOS occurred during a weekend gap or
                # before the visible chart window. Skip it entirely rather than
                # placing the line at an arbitrary bar that has no structural meaning.
                continue
            bos_bar = max(0, min(bos_bar, n - 1))

            direction = str(bos.get('direction', 'BULLISH')).upper()

            # Short precise horizontal line centered on the break candle.
            # 10 bars before shows the level that was holding as resistance/support.
            # 12 bars after shows the confirmed break and close beyond it.
            line_start = max(0, bos_bar - 10)
            line_end   = min(n - 1, bos_bar + 12)

            ax.plot(
                [float(line_start), float(line_end)],
                [level, level],
                color=self.C_BOS_LINE,
                linewidth=1.4,
                linestyle='solid',
                alpha=0.85,
                zorder=5,
            )

            # Faint dotted extension to the right edge so the level can still
            # be referenced against current price without being distracting.
            if line_end < x_right:
                ax.plot(
                    [float(line_end), float(x_right)],
                    [level, level],
                    color=self.C_BOS_LINE,
                    linewidth=0.5,
                    linestyle=(0, (2, 5)),
                    alpha=0.30,
                    zorder=3,
                )

            # Vertical break marker — a thicker line at the exact break candle
            # so the user's eye is drawn to the specific bar that confirmed BOS.
            ax.plot(
                [float(bos_bar), float(bos_bar)],
                [level - _tick_size * 2.0, level + _tick_size * 2.0],
                color=self.C_BOS_LINE,
                linewidth=2.2,
                alpha=0.92,
                zorder=6,
                solid_capstyle='round',
            )

            # Small filled circle at the exact break bar for visual precision.
            ax.plot(
                float(bos_bar), level,
                marker='o',
                markersize=4.5,
                color=self.C_BOS_LINE,
                alpha=0.90,
                zorder=7,
                linestyle='none',
            )

            # Direction arrow + label placed ABOVE for bullish BOS, BELOW for bearish.
            is_bull_bos  = direction == 'BULLISH'
            arrow_symbol = 'BOS ▲' if is_bull_bos else 'BOS ▼'
            label_y      = (
                level + _label_offset if is_bull_bos
                else level - _label_offset
            )
            va_anchor = 'bottom' if is_bull_bos else 'top'

            ax.text(
                float(bos_bar),
                label_y,
                arrow_symbol,
                color=self.C_BOS_LINE,
                fontsize=6.8,
                fontweight='bold',
                va=va_anchor,
                ha='center',
                alpha=0.92,
                zorder=8,
            )

    # =========================================================================
    # SWING STRUCTURE LEVELS
    # =========================================================================
    

    def _draw_inducement(
        self,
        ax: plt.Axes,
        inducement_data: Dict,
        time_index: list,
        n: int,
        x_right: int,
        y_min: float,
        y_max: float,
    ):
        """
        Draw the Inducement (IDM) sweep level and highlight the sweep candle.
        The IDM is the M15 internal pullback that was swept before the entry.
        A clear sweep label tells the user the stop hunt is complete.
        """
        sweep_level = float(inducement_data.get('sweep_level', 0))
        if sweep_level <= 0:
            return

        # The sweep candle data is nested inside inducement_data.
        sweep_candle = inducement_data.get('sweep_candle', {}) or {}
        direction    = str(inducement_data.get('direction', 'BULLISH')).upper()
        sweep_pips   = float(inducement_data.get('sweep_pips', 0))
        quality      = str(inducement_data.get('quality', 'MODERATE')).upper()
        is_buy       = direction in ('BULLISH', 'BUY')

        # Colour: teal for buy IDM, orange for sell IDM.
        idm_color = '#06b6d4' if is_buy else '#f97316'

        # Draw the IDM sweep level as a horizontal dotted line.
        ax.axhline(
            y=sweep_level,
            color=idm_color,
            linewidth=0.80,
            linestyle=(0, (2, 3)),
            alpha=0.70,
            zorder=4,
        )

        # Locate the sweep candle bar by its timestamp.
        sweep_ts  = sweep_candle.get('timestamp')
        sweep_bar = self._ts_to_bar_index(sweep_ts, time_index)

        if sweep_bar is not None and 0 <= sweep_bar < n:
            # Highlight the sweep candle with a coloured background rectangle.
            ax.add_patch(mpatches.Rectangle(
                (sweep_bar - 0.5, y_min),
                1.0,
                y_max - y_min,
                facecolor=idm_color,
                edgecolor='none',
                alpha=0.08,
                zorder=3,
            ))
            # Arrow pointing to the sweep wick.
            arrow_y = (
                float(sweep_candle.get('sweep_low', sweep_level)) - (y_max - y_min) * 0.015
                if is_buy
                else float(sweep_candle.get('sweep_high', sweep_level)) + (y_max - y_min) * 0.015
            )
            ax.annotate(
                '',
                xy=(sweep_bar, sweep_level),
                xytext=(sweep_bar, arrow_y),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=idm_color,
                    lw=1.2,
                ),
                zorder=9,
            )
            # Label above/below the sweep candle.
            label_y = (
                arrow_y - (y_max - y_min) * 0.018
                if is_buy
                else arrow_y + (y_max - y_min) * 0.018
            )
            quality_tag = 'STRONG' if quality == 'STRONG' else 'MOD'
            ax.text(
                sweep_bar,
                label_y,
                'IDM SWEPT\n%.1f pips [%s]' % (sweep_pips, quality_tag),
                color=idm_color,
                fontsize=5.8,
                fontweight='bold',
                ha='center',
                va='top' if is_buy else 'bottom',
                alpha=0.92,
                zorder=9,
            )

        # Right-edge label for the sweep level.
        ax.text(
            x_right + 0.3,
            sweep_level,
            ' IDM %.5f ' % sweep_level,
            color=idm_color,
            fontsize=6.5,
            fontweight='bold',
            va='center',
            ha='left',
            alpha=0.88,
            zorder=9,
            clip_on=False,
            bbox={
                'boxstyle':  'round,pad=0.18',
                'facecolor': idm_color,
                'edgecolor': 'none',
                'alpha':     0.22,
            },
        )

    def _draw_premium_discount_zone(
        self,
        ax: plt.Axes,
        htf_swing_high: float,
        htf_swing_low: float,
        is_buy: bool,
        y_min: float,
        y_max: float,
    ):
        """
        Shade the chart background to show the 50 percent equilibrium split.
        Premium zone (top 50 percent) is shaded red — valid only for SELL.
        Discount zone (bottom 50 percent) is shaded green — valid only for BUY.
        The equilibrium line at 50 percent is drawn in white at low opacity.
        """
        equilibrium = (htf_swing_high + htf_swing_low) / 2.0
        chart_min   = max(y_min, htf_swing_low)
        chart_max   = min(y_max, htf_swing_high)

        if chart_max <= chart_min:
            # HTF swing is outside the visible chart range — draw equilibrium only.
            if y_min <= equilibrium <= y_max:
                ax.axhline(
                    y=equilibrium,
                    color='#ffffff',
                    linewidth=0.55,
                    linestyle=(0, (8, 6)),
                    alpha=0.20,
                    zorder=2,
                )
            return

        eq_visible = max(chart_min, min(equilibrium, chart_max))

        # Discount zone (below equilibrium): light green background.
        discount_top = eq_visible
        discount_bot = chart_min
        if discount_top > discount_bot:
            ax.axhspan(
                discount_bot,
                discount_top,
                facecolor='#22c55e',
                alpha=0.04,
                zorder=1,
            )

        # Premium zone (above equilibrium): light red background.
        premium_top = chart_max
        premium_bot = eq_visible
        if premium_top > premium_bot:
            ax.axhspan(
                premium_bot,
                premium_top,
                facecolor='#ef4444',
                alpha=0.04,
                zorder=1,
            )

        # Equilibrium line.
        ax.axhline(
            y=equilibrium,
            color='#ffffff',
            linewidth=0.65,
            linestyle=(0, (8, 5)),
            alpha=0.28,
            zorder=2,
        )

        # Label the equilibrium line at the horizontal midpoint of the chart.
        # n is not in scope here — derive the midpoint from the axes x-limits.
        zone_label  = 'EQ (DISCOUNT)' if is_buy else 'EQ (PREMIUM)'
        _xlim       = ax.get_xlim()
        _label_x    = (_xlim[0] + _xlim[1]) / 2.0
        ax.text(
            _label_x,
            equilibrium,
            ' 50% EQ',
            color='#ffffff',
            fontsize=5.5,
            va='bottom',
            ha='center',
            alpha=0.38,
            zorder=8,
        )

    def _draw_swing_levels(
        self,
        ax: plt.Axes,
        swing_levels: List[Dict],
        time_index: list,
        n: int,
        y_min: float,
        y_max: float,
    ):
        """
        Draw swing high and swing low markers as small triangles with faint
        dotted extension lines to the right edge.
        """
        price_range = max(y_max - y_min, 1e-10)
        for swing in swing_levels[-20:]:
            direction = str(swing.get('direction', '')).upper()
            price     = float(swing.get('price', 0))
            if price <= 0 or price < y_min * 0.98 or price > y_max * 1.02:
                continue

            pos = swing.get('index', None)
            if pos is None:
                continue
            pos = int(pos)
            # Swing indices from _identify_swings are relative to the slice
            # passed to the function (tail(40) in the scanner). They must
            # fall within [0, n-1] of the chart's reset index.
            if pos < 0 or pos >= n:
                continue
            # Additional guard: skip if the swing price is not within the
            # visible price range to avoid markers floating outside the axes.
            if price < y_min or price > y_max:
                continue

            if direction == 'HIGH':
                color  = self.C_DOWN
                marker = 'v'
                y_pos  = price + price_range * 0.007
            else:
                color  = self.C_UP
                marker = '^'
                y_pos  = price - price_range * 0.007

            ax.plot(
                pos, y_pos,
                marker=marker,
                color=color,
                markersize=4.0,
                alpha=0.65,
                zorder=8,
                linestyle='none',
            )
            ax.plot(
                [pos, n - 1],
                [price, price],
                color=color,
                linewidth=0.45,
                linestyle=(0, (2, 5)),
                alpha=0.25,
                zorder=3,
            )
            sw_type = str(swing.get('type', '')).upper()
            if sw_type in ('HH', 'LL', 'HL', 'LH'):
                ax.text(
                    pos + 0.5, y_pos,
                    sw_type,
                    color=color,
                    fontsize=5.2,
                    va='center',
                    ha='left',
                    alpha=0.60,
                    zorder=8,
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
        anchor_x: int,
        last_close: float,
        last_open: float,
    ):
        """
        Draw a TradingView-style position tool plus right-side price markers.
        """
        y_range = max(y_max - y_min, 1e-10)
        min_sep = y_range * 0.024

        try:
            entry = float(setup_data.get('entry_price', 0))
            stop  = float(setup_data.get('stop_loss', 0))
            tp1   = float(setup_data.get('take_profit_1', 0))
            tp2   = float(setup_data.get('take_profit_2', 0))
        except (TypeError, ValueError):
            return

        if min(entry, stop, tp1, tp2) <= 0:
            return

        is_long   = str(setup_data.get('direction', 'BUY')).upper() == 'BUY'
        max_left  = max(0.5, float(n) - self.POSITION_MIN_WIDTH - 0.7)
        pos_left  = max(0.5, min(float(anchor_x), max_left))
        pos_right = min(pos_left + self.POSITION_MIN_WIDTH, float(n) - 0.5)

        reward_low = min(entry, tp2)
        reward_high = max(entry, tp2)
        risk_low = min(entry, stop)
        risk_high = max(entry, stop)

        ax.add_patch(mpatches.Rectangle(
            (pos_left, reward_low),
            pos_right - pos_left,
            reward_high - reward_low,
            facecolor=self.C_TP2,
            edgecolor=self.C_TP2,
            linewidth=1.2,
            alpha=0.16,
            zorder=5.35,
        ))
        ax.add_patch(mpatches.Rectangle(
            (pos_left, risk_low),
            pos_right - pos_left,
            risk_high - risk_low,
            facecolor=self.C_SL,
            edgecolor=self.C_SL,
            linewidth=1.2,
            alpha=0.14,
            zorder=5.34,
        ))

        # Entry, SL, TP1, TP2 span the full chart width so users can see
        # exactly which candles are above or below each level. The position
        # rectangle drawn above provides the coloured background context.
        # Drawing lines only inside pos_left..pos_right was hiding levels
        # from the candle history and making entries unverifiable visually.
        for y_val, color, lw, ls in (
            (entry, self.C_ENTRY, 1.6, 'solid'),
            (stop, self.C_SL, 1.1, (0, (5, 3))),
            (tp2, self.C_TP2, 1.1, (0, (5, 3))),
            (tp1, self.C_TP1, 1.0, (0, (1.2, 2.2))),
        ):
            ax.plot(
                [-0.5, float(x_right)],
                [y_val, y_val],
                color=color,
                linewidth=lw,
                linestyle=ls,
                alpha=0.90,
                zorder=8.2,
            )

        # Bright segment inside the position rectangle to make the entry
        # line stand out against the coloured background.
        ax.plot(
            [pos_left, pos_right],
            [entry, entry],
            color=self.C_ENTRY,
            linewidth=2.2,
            linestyle='solid',
            alpha=1.0,
            zorder=8.6,
        )

        live_color = self.C_UP if last_close >= last_open else self.C_DOWN
        ax.plot(
            [-0.5, x_right],
            [last_close, last_close],
            color=live_color,
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            alpha=0.75,
            zorder=8.0,
        )

        rr = abs(tp2 - entry) / max(abs(entry - stop), 1e-10)
        pos_label = 'LONG POSITION' if is_long else 'SHORT POSITION'
        reward_center = reward_high - (reward_high - reward_low) * 0.18
        risk_center = risk_low + (risk_high - risk_low) * 0.18

        ax.text(
            pos_left + 0.7,
            reward_center,
            f'{pos_label}   {rr:.2f}R',
            color=self.C_TEXT,
            fontsize=7.8,
            fontweight='bold',
            ha='left',
            va='center',
            alpha=0.96,
            zorder=8.5,
        )
        ax.text(
            pos_left + 0.7,
            risk_center,
            'RISK',
            color=self.C_TEXT,
            fontsize=7.2,
            ha='left',
            va='center',
            alpha=0.90,
            zorder=8.5,
        )

        levels = [
            {'price': stop, 'color': self.C_SL, 'tag': 'SL'},
            {'price': entry, 'color': self.C_ENTRY, 'tag': 'ENTRY'},
            {'price': tp1, 'color': self.C_TP1, 'tag': 'TP1'},
            {'price': tp2, 'color': self.C_TP2, 'tag': 'TP2'},
            {'price': last_close, 'color': live_color, 'tag': 'LIVE'},
        ]
        self._draw_right_price_markers(
            ax, levels, x_right, decimals, y_min, y_max, min_sep)

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

    def _draw_watermark(self, ax: plt.Axes, symbol: str = '', timeframe: str = 'M15'):
        text = f'NIXIE TRADES\n{timeframe}  {symbol}' if symbol else 'NIXIE TRADES'
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
        setup_text = ' '.join([
            str(setup_data.get('setup_type', '')),
            str(setup_data.get('setup_label', '')),
            str(setup_data.get('entry_type', '')),
        ]).upper()
        if 'SNIPER' in setup_text:
            tier_lbl = 'SNIPER SETUP'
        elif 'UNICORN' in setup_text:
            tier_lbl = 'UNICORN SETUP'
        else:
            tier_lbl = 'STANDARD SETUP'
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
        chart_tf = str(setup_data.get('chart_timeframe', 'M15')).upper()
        fig.text(
            0.010, 0.966,
            f'{symbol}   {chart_tf}   {direction}   '
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

    def _resolve_position_anchor(
        self,
        poi: Optional[Dict],
        refined_pois: List[Dict],
        time_index: list,
        n: int,
        data: Optional[pd.DataFrame] = None,
        entry_price: float = 0.0,
        direction: str = 'BUY',
        order_type: str = 'LIMIT',
    ) -> int:
        """
        Return the bar index where the left edge of the position tool starts.

        For a filled MARKET order: the tool starts at the bar where price
        last touched the entry level within the final 12 bars of the chart.
        Only the rightmost 12 bars are searched — anything earlier is a
        historical touch unrelated to this setup and would incorrectly
        place the tool in the middle of the chart.

        For a pending LIMIT/STOP order: the tool is
        anchored to the right edge of the chart so it sits flush against
        the most recent candles, accurately showing where the order is
        waiting relative to current price.

        The position tool right edge is always capped at n - 0.5 by
        _draw_price_levels so it never overflows past the last bar.
        """
        order_type_u = str(order_type or 'LIMIT').upper()
        if order_type_u != 'MARKET':
            return max(1, n - self.POSITION_MIN_WIDTH)

        if data is not None and entry_price > 0 and len(data) > 0:
            # Only search the last 12 bars for MARKET orders. Searching
            # further back was finding
            # irrelevant historical price levels and placing the tool at bar
            # 30-40 of an 80-bar chart, appearing in the middle of the screen.
            search_start = max(0, len(data) - 12)
            for i in range(len(data) - 1, search_start - 1, -1):
                try:
                    lo = float(data.iloc[i]['low'])
                    hi = float(data.iloc[i]['high'])
                    if lo <= entry_price <= hi:
                        return max(1, min(i, n - self.POSITION_MIN_WIDTH))
                except Exception:
                    continue

        # No recent price touch found: the order is pending.
        # Anchor the left edge so the right edge lands on the last bar.
        # n - POSITION_MIN_WIDTH puts pos_right at exactly n,
        # which _draw_price_levels caps at n - 0.5 (the last candle).
        return max(1, n - self.POSITION_MIN_WIDTH)

    def _draw_right_price_markers(
        self,
        ax: plt.Axes,
        levels: List[Dict],
        x_right: int,
        decimals: int,
        y_min: float,
        y_max: float,
        min_sep: float,
    ):
        valid_levels = []
        for level in levels:
            try:
                price = float(level.get('price', 0))
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            valid_levels.append({
                'price': price,
                'color': level.get('color', self.C_TEXT),
                'tag': str(level.get('tag', 'PRICE')).upper(),
            })

        if not valid_levels:
            return

        valid_levels.sort(key=lambda item: item['price'])
        label_positions: List[float] = []
        for item in valid_levels:
            label_y = item['price']
            if label_positions and (label_y - label_positions[-1]) < min_sep:
                label_y = label_positions[-1] + min_sep
            label_positions.append(label_y)

        overflow = label_positions[-1] - (y_max - min_sep * 0.35)
        if overflow > 0:
            label_positions = [y - overflow for y in label_positions]

        underflow = (y_min + min_sep * 0.35) - label_positions[0]
        if underflow > 0:
            label_positions = [y + underflow for y in label_positions]

        for item, label_y in zip(valid_levels, label_positions):
            price_fmt = f'%.{decimals}f' % item['price']
            ax.plot(
                [x_right - 0.65, x_right + 0.15],
                [item['price'], label_y],
                color=item['color'],
                linewidth=0.9,
                alpha=0.88,
                zorder=8.8,
                clip_on=False,
            )
            ax.text(
                x_right + 0.30,
                label_y,
                f' {item["tag"]} {price_fmt} ',
                color='#ffffff',
                fontsize=7.3,
                fontweight='bold',
                va='center',
                ha='left',
                alpha=0.98,
                zorder=9.2,
                clip_on=False,
                bbox={
                    'boxstyle': 'round,pad=0.22,rounding_size=0.16',
                    'facecolor': item['color'],
                    'edgecolor': 'none',
                    'alpha': 0.96,
                },
            )

    def _ts_to_bar_index(
        self,
        ts,
        time_index: list,
        max_tolerance_seconds: float = 7200.0,
    ) -> Optional[int]:
        """
        Find the chart bar index closest to a given timestamp.
        max_tolerance_seconds prevents matching completely unrelated bars
        when the POI timestamp falls outside the chart window.
        Default 7200s (2 hours) handles H1 zones on M15 charts:
        every H1 open is always within 15 minutes of an M15 bar.
        """
        if ts is None or not time_index:
            return None
        try:
            import pytz
            ts_pd = pd.Timestamp(ts)
            if ts_pd.tzinfo is None:
                ts_pd = pytz.utc.localize(ts_pd)
            else:
                ts_pd = ts_pd.tz_convert('UTC')

            best_i    = None
            best_diff = None
            for i, t in enumerate(time_index):
                try:
                    t_pd = pd.Timestamp(t)
                    if t_pd.tzinfo is None:
                        t_pd = pytz.utc.localize(t_pd)
                    else:
                        t_pd = t_pd.tz_convert('UTC')
                    diff = abs((t_pd - ts_pd).total_seconds())
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_i    = i
                except Exception:
                    continue

            if best_i is not None and best_diff is not None and best_diff <= max_tolerance_seconds:
                return best_i
            return None
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
            img     = _PIL_Image.open(raw_buf)
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
            tp1     = round(entry + risk * 1.5, 2)
            tp2     = round(entry + risk * 2.0, 2)

            sample_poi = {
                'type':         'BB',
                'direction':    'BULLISH',
                'timeframe':    'H1',
                'role':         'PRIMARY',
                'high':         bb_high,
                'low':          bb_low,
                'timestamp':    times[41],
                'index':        41,
                'volume_ratio': 2.6,
                'impulse_pips': 52.0,
                'confidence':   82,
            }
            sample_refined = [
                {
                    'type': 'BB',
                    'direction': 'BULLISH',
                    'timeframe': 'M15',
                    'role': 'REFINEMENT',
                    'high': round(entry, 2),
                    'low': round(entry - 2.9, 2),
                    'timestamp': times[46],
                    'index': 46,
                    'volume_ratio': 2.1,
                    'impulse_pips': 24.0,
                    'confidence': 79,
                },
                {
                    'type': 'OB',
                    'direction': 'BULLISH',
                    'timeframe': 'M5',
                    'role': 'REFINEMENT',
                    'high': round(entry, 2),
                    'low': round(entry - 1.4, 2),
                    'timestamp': times[48],
                    'index': 48,
                    'volume_ratio': 1.8,
                    'impulse_pips': 13.0,
                    'confidence': 74,
                },
            ]
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
                refined_pois=sample_refined,
                bos_events=sample_bos,
            )

        except Exception as exc:
            logger.error(
                "Sample chart generation failed: %s", exc, exc_info=True)
            return None
