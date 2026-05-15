"""
Plotting for ENE306-analyseverktøyet.

Denne modulen tegner og lagrer figurer. Den skal ikke lese CSV-filer direkte.
Figurene bygger på formateringen fra main_1.py: konsekvente titler, lagring som
PDF, ryddige akser, og legends som ikke skjuler målepunkter.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, MultipleLocator, NullFormatter
from numpy.typing import NDArray

from config import (
    THRESHOLDS_CRITICAL,
    THRESHOLDS_OPTIMAL_HUMIDITY,
    THRESHOLDS_TEMPERATURE,
    THRESHOLDS_WARN,
)
from data_processing import filter_weather


PLOT_DPI = 180
SAVE_DPI = 1200
TITLE_FONT_SIZE = 17
SUBPLOT_TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11
LEGEND_FONT_SIZE = 11
LEGEND_TITLE_FONT_SIZE = 12
SERIES_LINE_WIDTH = 1.6
PRIMARY_SERIES_LINE_WIDTH = 2.2
WEATHER_LINE_WIDTH = 1.8
THRESHOLD_LINE_WIDTH = 2.5
GRID_LINE_WIDTH = 0.75
MINOR_GRID_LINE_WIDTH = 0.55

plt.rcParams.update(
    {
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": SAVE_DPI,
        "pdf.fonttype": 42,
        "font.size": 12,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.1,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
        "grid.color": "0.70",
        "grid.alpha": 0.75,
    }
)

FIGURE_DIR = Path(__file__).parent / "figurer"
ThresholdLine = Tuple[float, str, Optional[str]]
FloatArray = NDArray[np.float64]

REPORT_LABEL_OVERRIDES = {
    "Luftfuktighet (%)": "Relativ luftfuktighet (%)",
    "CO2 (ppm)": "CO₂ (ppm)",
}

NORWEGIAN_MONTH_ABBREVIATIONS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "Mai",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sept",
    10: "Okt",
    11: "Nov",
    12: "Des",
}


def report_label(variable: str, *, indoor: bool = False, outdoor: bool = False) -> str:
    """Returner et pent variabelnavn for figurtekst."""
    if variable == "Temperatur (°C)":
        if indoor:
            return "Innetemperatur (°C)"
        if outdoor:
            return "Utetemperatur (°C)"
        return "Temperatur (°C)"
    if variable == "Luftfuktighet (%)":
        if outdoor:
            return "Relativ uteluftfuktighet (%)"
        return "Relativ luftfuktighet (%)"
    return REPORT_LABEL_OVERRIDES.get(variable, variable)


def label_without_unit(variable: str) -> str:
    """Fjern enhet i parentes fra variabelnavn til korte subplot-titler."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", report_label(variable))


def unit_axis_label(variable: str, fallback: str = "Konsentrasjon") -> str:
    """Hent enhet fra variabelnavn og legg den på verdiaksen."""
    match = re.search(r"\(([^)]*)\)\s*$", report_label(variable))
    if match:
        return f"{fallback} ({match.group(1)})"
    return fallback


def report_thresholds(variable: str, context: str = "full") -> List[ThresholdLine]:
    """Definer terskler og farger slik at alle figurer bruker samme uttrykk."""
    if variable == "Temperatur (°C)":
        lower = THRESHOLDS_TEMPERATURE["night"]["min"]
        split = THRESHOLDS_TEMPERATURE["day"]["min"]
        upper = THRESHOLDS_TEMPERATURE["day"]["max"]
        if context == "above":
            return [(upper, "firebrick", f"Øvre grense: {upper:g} °C")]
        if context == "below":
            return [(lower, "firebrick", f"Nedre grense: {lower:g} °C")]
        return [
            (upper, "firebrick", f"Øvre grense: {upper:g} °C"),
            (split, "darkorange", f"Dag/natt-grense: {split:g} °C"),
            (lower, "firebrick", f"Nedre grense: {lower:g} °C"),
        ]

    if variable == "Luftfuktighet (%)":
        limits = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        lower = limits["critical_min"]
        upper = limits["critical_max"]
        optimal_min = limits.get("optimal_min")
        optimal_max = limits.get("optimal_max")

        if context == "above":
            return [(upper, "firebrick", f"Øvre grense: {upper:g} %")]
        if context == "below":
            return [(lower, "firebrick", f"Nedre grense: {lower:g} %")]

        lines: List[ThresholdLine] = [(upper, "firebrick", f"Øvre kritisk grense: {upper:g} %")]
        if optimal_max is not None:
            lines.append((optimal_max, "darkorange", f"Øvre anbefalt grense: {optimal_max:g} %"))
        if optimal_min is not None:
            lines.append((optimal_min, "darkorange", f"Nedre anbefalt grense: {optimal_min:g} %"))
        lines.append((lower, "firebrick", f"Nedre kritisk grense: {lower:g} %"))
        return lines

    upper = THRESHOLDS_CRITICAL.get(variable)
    if upper is None or context == "below":
        return []

    warning = THRESHOLDS_WARN.get(variable)
    lines = []
    if warning is not None:
        lines.append((warning, "darkorange", f"Varselgrense: {warning:g}"))
    lines.append((upper, "firebrick", f"Kritisk grense: {upper:g}"))
    return lines


def format_scope_label(scope: Any) -> str:
    """Lag periodetekst basert på Scope-objektet."""
    if scope.mode == "all":
        return "Hele måleperioden"
    if scope.mode == "year":
        return f"Året {scope.year}"
    if scope.mode == "month":
        return f"Måned {scope.month}/{scope.year}"
    if scope.mode == "week":
        return f"Uke {scope.week} i {scope.year}"
    if scope.mode == "fall":
        return f"Høst-semesteret {scope.year}"
    if scope.mode == "spring":
        return f"Vår-semesteret {scope.year}"
    return "Valgt periode"


def report_title(
    kind: str,
    scope: Any,
    variable: Optional[str] = None,
    *,
    with_weather: bool = False,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> str:
    """Lag rapporttittel med periode og faktisk tidsintervall."""
    period = format_scope_label(scope)

    if kind == "availability":
        title = f"Datadekning - {period}"
        if start is not None and end is not None:
            title += f"\n{start:%d.%m.%Y} til {end:%d.%m.%Y}"
        return title

    if kind == "building_distribution" and variable is not None:
        return f"Fordeling av {report_label(variable)} - {period}"

    if kind == "time_series" and variable is not None:
        suffix = " inne og ute" if with_weather else ""
        return f"Utvikling i {report_label(variable)}{suffix} - {period}"

    return period


def report_time_series_limits(variable: str) -> Tuple[Optional[float], Optional[float]]:
    """Gi faste y-grenser der det gjør figurene mer sammenlignbare."""
    if variable == "Temperatur (°C)":
        return 10.0, 35.0
    if variable == "CO2 (ppm)":
        return 400.0, 2000.0

    critical_value = THRESHOLDS_CRITICAL.get(variable)
    if critical_value is not None:
        return 0.0, float(2 * critical_value)
    return None, None


def style_legend(legend: Any) -> None:
    """Gi legends lik stil i alle figurer."""
    if legend is None:
        return
    legend.get_frame().set_alpha(0.96)
    legend.get_frame().set_edgecolor("0.45")
    legend.get_frame().set_linewidth(0.9)
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(LEGEND_TITLE_FONT_SIZE)
        legend.get_title().set_fontweight("bold")
    for text_item in legend.get_texts():
        text_item.set_fontsize(LEGEND_FONT_SIZE)


def require_timestamp(value: object, label: str = "timestamp") -> pd.Timestamp:
    """Konverter verdi til Timestamp og stopp tidlig hvis den er ugyldig."""
    timestamp = pd.Timestamp(cast(Any, value))
    if pd.isna(timestamp):
        raise ValueError(f"Ugyldig tidspunkt for {label}.")
    return timestamp


def as_text(value: object) -> str:
    """Konverter verdier fra pandas til vanlig tekst."""
    return str(cast(Any, value))


def unique_text_values(series: pd.Series, *, reverse: bool = False) -> List[str]:
    """Returner unike tekstverdier i stabil sortert rekkefølge."""
    values = [as_text(value) for value in series.dropna().to_list()]
    return sorted(set(values), reverse=reverse)


def format_axis_date_label(timestamp: pd.Timestamp) -> str:
    """Formater datoetikett på x-aksen."""
    return f"{timestamp.day}. {NORWEGIAN_MONTH_ABBREVIATIONS[timestamp.month]}"


def compute_frame_time_limits(df_list: Sequence[pd.DataFrame]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Finn faktisk start og slutt i datagrunnlaget."""
    valid_frames = [df for df in df_list if not df.empty]
    if not valid_frames:
        return None, None
    return min(df.index.min() for df in valid_frames), max(df.index.max() for df in valid_frames)


def shift_timestamp(base: pd.Timestamp, *, days: int = 0, hours: int = 0) -> pd.Timestamp:
    """Flytt et tidspunkt på en eksplisitt og testbar måte."""
    shifted = base.to_pydatetime() + timedelta(days=days, hours=hours)
    return require_timestamp(shifted, "shifted timestamp")


def compute_scope_limits(
    scope: Any,
    df_list: Optional[Sequence[pd.DataFrame]] = None,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Finn x-aksegrenser for aktiv periode."""
    if scope.mode == "week" and scope.year is not None and scope.week is not None:
        start = require_timestamp(datetime.fromisocalendar(scope.year, scope.week, 1), "week start")
        return start, shift_timestamp(start, days=7)
    if scope.mode == "month" and scope.year is not None and scope.month is not None:
        start = require_timestamp(pd.Timestamp(year=scope.year, month=scope.month, day=1), "month start")
        end = require_timestamp(start + pd.offsets.MonthEnd(1), "month end")
        return start, shift_timestamp(end, days=1)
    if scope.mode == "year" and scope.year is not None:
        start = require_timestamp(pd.Timestamp(year=scope.year, month=1, day=1), "year start")
        end = require_timestamp(pd.Timestamp(year=scope.year, month=12, day=31), "year end")
        return start, shift_timestamp(end, days=1)
    if scope.mode == "fall" and scope.year is not None:
        start = require_timestamp(pd.Timestamp(year=scope.year, month=8, day=10), "fall start")
        end = require_timestamp(pd.Timestamp(year=scope.year, month=12, day=10), "fall end")
        return start, shift_timestamp(end, days=1)
    if scope.mode == "spring" and scope.year is not None:
        start = require_timestamp(pd.Timestamp(year=scope.year, month=1, day=6), "spring start")
        end = require_timestamp(pd.Timestamp(year=scope.year, month=6, day=6), "spring end")
        return start, shift_timestamp(end, days=1)
    if df_list is not None:
        return compute_frame_time_limits(df_list)
    return None, None


def configure_time_axis(
    ax: Axes,
    scope: Any,
    start_vis: Optional[pd.Timestamp],
    end_vis: Optional[pd.Timestamp],
) -> None:
    """Formater tidsaksen. Nærmeste zoom i menyen er uke."""
    if start_vis is not None and end_vis is not None:
        ax.set_xlim(start_vis, end_vis)

    if scope.mode in {"fall", "spring"} and start_vis is not None and end_vis is not None:
        display_end = shift_timestamp(end_vis, days=-1)
        major_ticks = [start_vis]
        month_ticks = pd.date_range(start=start_vis.replace(day=1), end=display_end, freq="MS")

        for tick in month_ticks:
            tick_timestamp = require_timestamp(tick, "month tick")
            if start_vis < tick_timestamp < display_end:
                major_ticks.append(tick_timestamp)

        if display_end > start_vis:
            major_ticks.append(display_end)

        labels = []
        for tick in major_ticks:
            label = format_axis_date_label(tick)
            if tick == start_vis or tick == display_end:
                label = f"\n{label}"
            labels.append(label)

        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(tick) for tick in major_ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis="x", which="major", linestyle=":", linewidth=GRID_LINE_WIDTH, alpha=0.8)
    elif scope.mode in {"year", "all"}:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, _position: NORWEGIAN_MONTH_ABBREVIATIONS[mdates.num2date(value).month]))
        ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=MINOR_GRID_LINE_WIDTH, alpha=0.65)
    elif scope.mode == "month":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%d."))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    elif scope.mode == "week" and start_vis is not None and end_vis is not None:
        start_day = start_vis.normalize()
        day_ticks = [shift_timestamp(start_day, days=index) for index in range(7)] + [end_vis]
        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(day) for day in day_ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter(["man", "tir", "ons", "tor", "fre", "lør", "søn", ""]))

    ax.tick_params(axis="x", labelrotation=0, labelsize=TICK_LABEL_FONT_SIZE)


def scope_folder_name(scope: Any) -> str:
    """Lag stabilt mappenavn for figurer."""
    if scope.mode == "all":
        return "hele_perioden"
    if scope.mode == "fall" and scope.year is not None:
        return f"host_{scope.year}"
    if scope.mode == "spring" and scope.year is not None:
        return f"var_{scope.year}"
    if scope.mode == "year" and scope.year is not None:
        return f"ar_{scope.year}"
    if scope.mode == "month" and scope.year is not None and scope.month is not None:
        return f"maned_{scope.year}_{scope.month:02d}"
    if scope.mode == "week" and scope.year is not None and scope.week is not None:
        return f"uke_{scope.year}_{scope.week:02d}"
    return "ukjent_periode"


def sanitize_filename(name: str) -> str:
    """Gjør filnavn trygge på tvers av operativsystem."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.strip("._") or "figur"


def save_figure(fig: plt.Figure, figure_dir: Path | str, stem: str, scope: Any) -> Path:
    """Lagre figur som PDF i en periodebasert mappe."""
    output_dir = Path(figure_dir) / scope_folder_name(scope)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sanitize_filename(stem)}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=SAVE_DPI)
    return output_path


def finalize_figure(fig: plt.Figure, figure_dir: Path | str, stem: str, scope: Any, show: bool = True) -> Path:
    """Lagre figur og vis den dersom show=True."""
    output_path = save_figure(fig, figure_dir, stem, scope)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def series_to_float_array(series: pd.Series) -> FloatArray:
    """Konverter pandas-serie til numpy-array med flyttall."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    return np.asarray(numeric_series, dtype=float)


def insert_nan_for_gaps(df: pd.DataFrame, column: str) -> pd.Series:
    """Sett inn NaN ved tidshull slik at linjen brytes visuelt."""
    sorted_df = df.sort_index().copy()
    time_diff_seconds = sorted_df.index.to_series().diff().dt.total_seconds()
    sorted_df.loc[time_diff_seconds > 3600, column] = np.nan

    start = sorted_df.index.min()
    end = sorted_df.index.max()
    full_index = pd.date_range(start=start, end=end, freq="1h")
    return sorted_df[column].reindex(full_index)


def plot_capped_series(
    ax: Axes,
    x_values: pd.Index,
    y_values: FloatArray,
    *,
    color: Any,
    label: str,
    linewidth: float,
    y_min: Optional[float],
    y_max: Optional[float],
    scatter_points: bool = False,
) -> Line2D:
    """Tegn serie og marker verdier som ligger utenfor fast y-område."""
    plot_values = y_values.copy()
    above = np.zeros_like(plot_values, dtype=bool)
    below = np.zeros_like(plot_values, dtype=bool)

    if y_max is not None:
        above = plot_values > y_max
        plot_values[above] = y_max
    if y_min is not None:
        below = plot_values < y_min
        plot_values[below] = y_min

    line, = ax.plot(x_values, plot_values, color=color, label=label, linewidth=linewidth)

    if scatter_points:
        ax.scatter(x_values, plot_values, color=color, s=10, zorder=5)
    y_range = None if y_min is None or y_max is None else float(y_max - y_min)
    # Markøren skal ligge helt oppe ved rammen, slik at den tydelig betyr "over plottegrensen".
    marker_offset = 0.004 * y_range if y_range and y_range > 0 else 0.0

    if above.any():
        marker_values = plot_values[above] - marker_offset
        ax.scatter(
            np.asarray(x_values)[above],
            marker_values,
            marker="o",
            facecolors="none",
            edgecolors="firebrick",
            linewidths=1.4,
            s=46,
            zorder=7,
            clip_on=True,
        )
    if below.any():
        marker_values = plot_values[below] + marker_offset
        ax.scatter(np.asarray(x_values)[below], marker_values, marker="v", color=color, s=22, zorder=6, clip_on=True)

    return line


def _plot_room_series_with_weather(
    df_list: List[pd.DataFrame],
    scope: Any,
    *,
    value_column: str,
    indoor_ylabel: str,
    thresholds: Sequence[ThresholdLine],
    weather_specs: Sequence[Tuple[str, str, str]],
    weather_ylabel: str,
    title_metric: str,
    stem: str,
    empty_message: str,
    df_weather: Optional[pd.DataFrame] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Felles motor for temperatur- og fuktighetsplot."""
    if not df_list:
        raise ValueError(empty_message)

    y_min, y_max = report_time_series_limits(title_metric)
    weather_filtered = pd.DataFrame()

    if df_weather is not None:
        weather_filtered = filter_weather(
            df_weather,
            mode=scope.mode,
            year=scope.year,
            month=scope.month,
            week=scope.week,
            day=scope.day,
        )

    show_weather = df_weather is not None and not weather_filtered.empty
    if show_weather:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(13.5, 8.8),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15},
        )
        ax, ax_weather = cast(Tuple[Axes, Axes], tuple(axes))
    else:
        fig, ax = plt.subplots(figsize=(13.5, 7.0))
        ax_weather = None

    color_map = colormaps["tab10"]

    for index, room_df in enumerate(df_list, start=1):
        if value_column not in room_df.columns or room_df.empty:
            continue

        series = insert_nan_for_gaps(room_df, value_column)
        plot_capped_series(
            ax,
            series.index,
            series_to_float_array(series),
            color=color_map((index - 1) % 10),
            label="_nolegend_",
            linewidth=PRIMARY_SERIES_LINE_WIDTH,
            y_min=y_min,
            y_max=y_max,
        )

    threshold_handles: List[Line2D] = []
    threshold_labels: List[str] = []
    used_labels: set[str] = set()

    for level, color, label in thresholds:
        handle = ax.axhline(y=level, color=color, linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, label=label or "_nolegend_")
        if label is not None and label not in used_labels:
            threshold_handles.append(handle)
            threshold_labels.append(label)
            used_labels.add(label)

    if y_min is not None:
        ax.set_ylim(bottom=float(y_min))
    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    ax.set_ylabel(indoor_ylabel)
    ax.grid(True, linestyle=":", linewidth=GRID_LINE_WIDTH, alpha=0.85)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)


    if threshold_handles:
        threshold_legend = ax.legend(
            threshold_handles,
            threshold_labels,
            frameon=True,
            title="Terskler",
        )
        style_legend(threshold_legend)

    if ax_weather is not None:
        for column, label, color in weather_specs:
            if column not in weather_filtered.columns:
                continue
            series = weather_filtered[column].dropna()
            if series.empty:
                continue
            ax_weather.plot(series.index, series_to_float_array(series), label=label, color=color, linewidth=WEATHER_LINE_WIDTH)

        ax_weather.set_ylabel(weather_ylabel)
        ax_weather.set_xlabel("Dato")
        ax_weather.grid(True, linestyle=":", linewidth=GRID_LINE_WIDTH, alpha=0.85)
        ax_weather.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        weather_legend = ax_weather.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), title="Uteklima kilder", frameon=True)
        style_legend(weather_legend)
    else:
        ax.set_xlabel("Dato")

    start_vis, end_vis = compute_scope_limits(scope, df_list)
    configure_time_axis(ax, scope, start_vis, end_vis)
    if ax_weather is not None:
        configure_time_axis(ax_weather, scope, start_vis, end_vis)

    ax.set_title(report_title("time_series", scope, title_metric, with_weather=ax_weather is not None, start=start_vis, end=end_vis), fontsize=TITLE_FONT_SIZE)

    if ax_weather is not None:
        fig.subplots_adjust(left=0.07, right=0.78, top=0.88, bottom=0.10, hspace=0.15)
    else:
        fig.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.10)

    return finalize_figure(fig, figure_dir, f"{stem}_{scope_folder_name(scope)}", scope, show)


def plot_temperature(
    df_list: List[pd.DataFrame],
    scope: Any,
    df_weather: Optional[pd.DataFrame] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Plot temperaturserier."""
    return _plot_room_series_with_weather(
        df_list,
        scope,
        value_column="Temperatur (°C)",
        indoor_ylabel=report_label("Temperatur (°C)", indoor=True),
        thresholds=report_thresholds("Temperatur (°C)", context="full"),
        weather_specs=[("utetemp_seklima", "Seklima", "blue"), ("utetemp_kunak", "Kunak", "orange")],
        weather_ylabel=report_label("Temperatur (°C)", outdoor=True),
        title_metric="Temperatur (°C)",
        stem="tidsserie_temperatur",
        empty_message="Ingen innendørs temperaturdata i valgt utvalg.",
        df_weather=df_weather,
        figure_dir=figure_dir,
        show=show,
    )


def plot_humidity(
    df_list: List[pd.DataFrame],
    scope: Any,
    df_weather: Optional[pd.DataFrame] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Plot luftfuktighetsserier."""
    return _plot_room_series_with_weather(
        df_list,
        scope,
        value_column="Luftfuktighet (%)",
        indoor_ylabel=report_label("Luftfuktighet (%)", indoor=True),
        thresholds=report_thresholds("Luftfuktighet (%)", context="full"),
        weather_specs=[("ute_rh_seklima", "Seklima", "blue"), ("ute_rh_kunak", "Kunak", "orange")],
        weather_ylabel=report_label("Luftfuktighet (%)", outdoor=True),
        title_metric="Luftfuktighet (%)",
        stem="tidsserie_luftfuktighet",
        empty_message="Ingen innendørs luftfuktighetsdata i valgt utvalg.",
        df_weather=df_weather,
        figure_dir=figure_dir,
        show=show,
    )


def plot_all_rooms_variable(
    df_list: List[pd.DataFrame],
    variable: str,
    scope: Any,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Plot øvrige inneklimavariabler uten romlegend."""
    if not df_list:
        raise ValueError(f"Ingen data funnet for {variable}.")

    y_min, y_max = report_time_series_limits(variable)
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    color_map = colormaps["tab20"]

    for index, room_df in enumerate(df_list, start=1):
        column = "Value" if "Value" in room_df.columns else variable
        if column not in room_df.columns or room_df.empty:
            continue

        series = insert_nan_for_gaps(room_df, column)
        plot_capped_series(
            ax,
            series.index,
            series_to_float_array(series),
            color=color_map((index - 1) % 20),
            label="_nolegend_",
            linewidth=SERIES_LINE_WIDTH,
            y_min=y_min,
            y_max=y_max,
        )

    for level, color, label in report_thresholds(variable):
        ax.axhline(y=level, color=color, linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, label=label)

    if y_min is not None:
        ax.set_ylim(bottom=float(y_min))
    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    start_vis, end_vis = compute_scope_limits(scope, df_list)
    configure_time_axis(ax, scope, start_vis, end_vis)

    ax.set_title(report_title("time_series", scope, variable, start=start_vis, end=end_vis), fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel(report_label(variable))
    ax.set_xlabel("Dato")
    ax.grid(True, axis="y", linestyle=":", linewidth=GRID_LINE_WIDTH, alpha=0.85)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

    threshold_legend = ax.legend(title="Terskler", frameon=True)
    style_legend(threshold_legend)

    fig.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.10)
    stem = f"tidsserie_{sanitize_filename(variable)}_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def plot_pm_time_series(
    pm_plot_data: Sequence[Tuple[str, Sequence[pd.DataFrame]]],
    scope: Any,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Plot alle luftpartikkelvariabler som 2x2 tidsserie-subplots."""
    if not pm_plot_data or not any(frames for _, frames in pm_plot_data):
        raise ValueError("Ingen luftpartikkeldata å plotte.")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2), sharex=True)
    flat_axes = cast(Sequence[Axes], axes.flatten())
    color_map = colormaps["tab20"]
    all_frames = [frame for _, frames in pm_plot_data for frame in frames]
    start_vis, end_vis = compute_scope_limits(scope, all_frames)

    for ax, (variable, room_frames) in zip(flat_axes, pm_plot_data):
        if not room_frames:
            ax.axis("off")
            ax.set_title(f"{label_without_unit(variable)} - ingen data", fontsize=SUBPLOT_TITLE_FONT_SIZE)
            continue

        y_min, y_max = report_time_series_limits(variable)

        for index, room_df in enumerate(room_frames, start=1):
            column = "Value" if "Value" in room_df.columns else variable
            if column not in room_df.columns or room_df.empty:
                continue

            series = insert_nan_for_gaps(room_df, column)
            plot_capped_series(
                ax,
                series.index,
                series_to_float_array(series),
                color=color_map((index - 1) % 20),
                label="_nolegend_",
                linewidth=SERIES_LINE_WIDTH,
                y_min=y_min,
                y_max=y_max,
            )

        for level, color, label in report_thresholds(variable):
            ax.axhline(y=level, color=color, linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, label=label)

        if y_min is not None:
            ax.set_ylim(bottom=float(y_min))
        if y_max is not None:
            ax.set_ylim(top=float(y_max))

        configure_time_axis(ax, scope, start_vis, end_vis)
        ax.set_title(label_without_unit(variable), fontsize=SUBPLOT_TITLE_FONT_SIZE + 1)
        ax.set_ylabel(unit_axis_label(variable))
        ax.grid(True, axis="y", linestyle=":", linewidth=GRID_LINE_WIDTH, alpha=0.85)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(handles, labels, title="Terskler", frameon=True, fontsize=LEGEND_FONT_SIZE)
            style_legend(legend)

    for ax in flat_axes[-2:]:
        ax.set_xlabel("Dato")

    fig.suptitle(
        report_title("time_series", scope, "Luftpartikler", start=start_vis, end=end_vis),
        fontsize=TITLE_FONT_SIZE + 2,
        fontweight="bold",
        x=0.5,
        y=0.965,
        ha="center",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=1.2, w_pad=1.8)

    stem = f"tidsserie_pm_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def draw_thresholds(ax: Axes, variable: str) -> None:
    """Tegn vertikale terskellinjer i boxplot."""
    for value, color, label in report_thresholds(variable, context="full"):
        ax.axvline(value, color=color, linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, label=label)


def recommended_boxplot_left_limit(variable: str, current_left: float) -> float:
    """Unngå unødvendig tomrom til venstre i boxplot."""
    if variable == "CO2 (ppm)":
        return 400.0
    if variable in THRESHOLDS_CRITICAL or variable in {"Luftfuktighet (%)"}:
        return max(0.0, current_left)
    return current_left


def draw_horizontal_boxplot_panel(
    ax: Axes,
    variable: str,
    building_data: Sequence[Sequence[float]],
    building_labels: Sequence[str],
    *,
    xlabel: str,
    show_ylabel: bool = True,
) -> None:
    """Tegn ett horisontalt boxplotpanel.

    Terskellegend bruker Matplotlib sin standardplassering slik at plasseringen
    velges ut fra hvor det er mest ledig plass i hvert panel.
    """
    ax.boxplot(
        list(building_data)[::-1],
        tick_labels=list(building_labels)[::-1],
        vert=False,
        showfliers=True,
        boxprops={"linewidth": 1.4, "color": "0.10"},
        whiskerprops={"linewidth": 1.3, "color": "0.10"},
        capprops={"linewidth": 1.3, "color": "0.10"},
        medianprops={"linewidth": 1.8, "color": "firebrick"},
    )
    draw_thresholds(ax, variable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bygg" if show_ylabel else "")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

    if variable in {"Temperatur (°C)", "Luftfuktighet (%)"}:
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis="x", which="minor", length=4)
        ax.grid(axis="x", which="major", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=0.8)
        ax.grid(axis="x", which="minor", linestyle=":", linewidth=MINOR_GRID_LINE_WIDTH, alpha=0.65)
    else:
        ax.grid(axis="x", which="major", linestyle=":", linewidth=GRID_LINE_WIDTH, alpha=0.75)

    current_left, _ = ax.get_xlim()
    ax.set_xlim(left=recommended_boxplot_left_limit(variable, float(current_left)))

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    legend = ax.legend(handles, labels, title="Terskler", frameon=True, fontsize=LEGEND_FONT_SIZE)
    style_legend(legend)


def plot_building_boxplot(
    variable: str,
    building_data: Sequence[Sequence[float]],
    building_labels: Sequence[str],
    scope: Any,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Plot boxplot for én variabel fordelt på bygg."""
    if not building_data:
        raise ValueError(f"Ingen data funnet for {variable} i valgt utvalg.")

    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    draw_horizontal_boxplot_panel(ax, variable, building_data, building_labels, xlabel=report_label(variable))
    ax.set_title(report_title("building_distribution", scope, variable), fontsize=TITLE_FONT_SIZE)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)

    stem = f"boxplot_{sanitize_filename(variable)}_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def plot_pm_boxplots(
    pm_plot_data: Sequence[Tuple[str, Sequence[Sequence[float]], Sequence[str]]],
    scope: Any,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Plot alle luftpartikkelvariabler som 2x2 boxplot-subplots."""
    if not pm_plot_data or not any(building_data for _, building_data, _ in pm_plot_data):
        raise ValueError("Ingen luftpartikkeldata å plotte.")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2))
    flat_axes = cast(Sequence[Axes], axes.flatten())

    for ax, (variable, building_data, building_labels) in zip(flat_axes, pm_plot_data):
        if not building_data:
            ax.axis("off")
            ax.set_title(f"{label_without_unit(variable)} - ingen data", fontsize=SUBPLOT_TITLE_FONT_SIZE)
            continue

        draw_horizontal_boxplot_panel(
            ax,
            variable,
            building_data,
            building_labels,
            xlabel=unit_axis_label(variable),
            show_ylabel=False,
        )
        ax.set_title(label_without_unit(variable), fontsize=SUBPLOT_TITLE_FONT_SIZE + 1)

    fig.suptitle(
        f"Fordeling av luftpartikler - {format_scope_label(scope)}",
        fontsize=TITLE_FONT_SIZE + 2,
        fontweight="bold",
        x=0.5,
        y=0.965,
        ha="center",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.4, w_pad=2.0)

    stem = f"boxplot_pm_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def plot_room_data_availability(
    availability_df: pd.DataFrame,
    scope: Any,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    """Plot datadekning per rom."""
    if availability_df.empty:
        raise ValueError("Ingen data å vise for valgt utvalg.")

    plot_df = availability_df.copy()
    plot_df["Etikett"] = "B" + plot_df["Bygg"].astype(str) + "-R" + plot_df["Rom"].astype(str)
    plot_df.sort_values(by=["Etikett", "Start"], inplace=True)
    plot_df.reset_index(drop=True, inplace=True)

    unique_buildings = unique_text_values(plot_df["Bygg"])
    color_map = colormaps["tab10"].resampled(max(1, len(unique_buildings)))
    building_colors = {building: color_map(index) for index, building in enumerate(unique_buildings)}

    unique_labels = unique_text_values(plot_df["Etikett"], reverse=True)
    fig, ax = plt.subplots(figsize=(13.5, max(6.0, len(unique_labels) * 0.58)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    min_date = require_timestamp(plot_df["Start"].min(), "availability minimum")
    max_date = require_timestamp(plot_df["Slutt"].max(), "availability maximum")

    for row in plot_df.itertuples(index=False):
        row_label = as_text(row.Etikett)
        row_building = as_text(row.Bygg)
        row_start = require_timestamp(row.Start, "availability start")
        row_end = require_timestamp(row.Slutt, "availability end")
        y_value = label_to_index[row_label]

        ax.plot(
            [mdates.date2num(row_start.to_pydatetime()), mdates.date2num(row_end.to_pydatetime())],
            [y_value, y_value],
            linewidth=4,
            color=building_colors[row_building],
            zorder=1,
        )

    ax.set_yticks(list(label_to_index.values()))
    ax.set_yticklabels(unique_labels, fontsize=TICK_LABEL_FONT_SIZE + 2)
    configure_time_axis(ax, scope, min_date, max_date)

    for building, color in building_colors.items():
        ax.plot(np.array([], dtype=float), np.array([], dtype=float), label=f"Bygg {int(building)}", color=color, linewidth=7)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), title="Bygg", frameon=True, fontsize=LEGEND_FONT_SIZE + 1)
    style_legend(legend)

    ax.set_xlabel("Dato", fontsize=AXIS_LABEL_FONT_SIZE + 2)
    ax.set_ylabel("Romkode", fontsize=AXIS_LABEL_FONT_SIZE + 2)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE + 2)
    ax.set_title(report_title("availability", scope, start=min_date, end=max_date), fontsize=TITLE_FONT_SIZE + 1)
    fig.subplots_adjust(left=0.14, right=0.80, top=0.88, bottom=0.12)

    stem = f"datadekning_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)
