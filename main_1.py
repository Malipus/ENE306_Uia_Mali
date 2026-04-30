from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator, NullFormatter
from numpy.typing import NDArray

from config import (
    INNEKLIMA_DIR,
    NORWEGIAN_MONTHS,
    PM_VARIABLER,
    THRESHOLDS_CRITICAL,
    THRESHOLDS_OPTIMAL_HUMIDITY,
    THRESHOLDS_TEMPERATURE,
    THRESHOLDS_WARN,
    TILGJENGELIGE_BYGG,
    VARIABLE_CHOICES,
)
from data_processing import fetch_csv, fetch_weather, filter_data, filter_weather, set_datetime_index

plt.rcParams["svg.fonttype"] = "none"

FIGURE_DIR = Path(__file__).parent / "figurer"
ThresholdDirection = Literal["above", "below"]
ThresholdLine = Tuple[float, str, Optional[str]]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SemesterThresholdRule:
    lower: Optional[float]
    upper: Optional[float]
    lower_label: Optional[str] = None
    upper_label: Optional[str] = None
    lower_pass_label: Optional[str] = None
    upper_pass_label: Optional[str] = None
    two_sided: bool = False


@dataclass
class Scope:
    buildings: List[str] = field(default_factory=lambda: list(TILGJENGELIGE_BYGG.keys()))
    rooms_by_building: Dict[str, Optional[List[str]]] = field(default_factory=dict)
    mode: str = "all"
    year: Optional[int] = None
    month: Optional[int] = None
    week: Optional[int] = None
    day: Optional[pd.Timestamp] = None

REPORT_LABEL_OVERRIDES = {
    "Luftfuktighet (%)": "Relativ fuktighet (%)",
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
    if variable == "Temperatur (°C)":
        if indoor:
            return "Innetemperatur (°C)"
        if outdoor:
            return "Utetemperatur (°C)"
        return "Temperatur (°C)"
    if variable == "Luftfuktighet (%)":
        if outdoor:
            return "Uteluftfuktighet (%)"
        return "Relativ fuktighet (%)"
    return REPORT_LABEL_OVERRIDES.get(variable, variable)


def resolve_variable_choice(variable: str) -> str:
    variable_key = variable.strip()
    if variable_key in VARIABLE_CHOICES:
        return VARIABLE_CHOICES[variable_key]

    normalized = variable_key.lower()
    for value in VARIABLE_CHOICES.values():
        if normalized in {value.lower(), report_label(value).lower()}:
            return value

    valid_choices = ", ".join(sorted(VARIABLE_CHOICES))
    raise ValueError(f"Ukjent variabelvalg '{variable}'. Bruk et navn eller et tall mellom {valid_choices}.")


def report_thresholds(variable: str, context: str = "full") -> list[ThresholdLine]:
    if variable == "Temperatur (°C)":
        lower = THRESHOLDS_TEMPERATURE["night"]["min"]
        split = THRESHOLDS_TEMPERATURE["day"]["min"]
        upper = THRESHOLDS_TEMPERATURE["day"]["max"]
        if context == "above":
            return [(upper, "orange", f"Øvre grense: {upper:g} °C")]
        if context == "below":
            return [(lower, "purple", f"Nedre grense: {lower:g} °C")]
        return [
            (upper, "orange", f"Øvre daggrense: {upper:g} °C"),
            (split, "black", f"Dag/natt-grense: {split:g} °C"),
            (lower, "purple", f"Nedre nattgrense: {lower:g} °C"),
        ]

    if variable == "Luftfuktighet (%)":
        lower = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]["critical_min"]
        upper = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]["critical_max"]
        if context == "above":
            return [(upper, "red", f"Øvre grense: {upper:g} %")]
        if context == "below":
            return [(lower, "blue", f"Nedre grense: {lower:g} %")]
        return [
            (upper, "red", f"Øvre grense: {upper:g} %"),
            (lower, "blue", f"Nedre grense: {lower:g} %"),
        ]

    upper = THRESHOLDS_CRITICAL.get(variable)
    if upper is None:
        return []
    if context == "below":
        return []
    warn = THRESHOLDS_WARN.get(variable)
    if context == "warning":
        if warn is None:
            return []
        label = report_label(variable)
        unit = ""
        if "(" in label and ")" in label:
            unit = label[label.find("(") + 1:label.find(")")]
        suffix = f" {unit}" if unit else ""
        return [(warn, "orange", f"Advarselsgrense: {warn:g}{suffix}")]

    label = report_label(variable)
    unit = ""
    if "(" in label and ")" in label:
        unit = label[label.find("(") + 1:label.find(")")]
    suffix = f" {unit}" if unit else ""
    upper_line: ThresholdLine = (upper, "red", f"Øvre grense: {upper:g}{suffix}")
    if context == "above":
        return [upper_line]
    if warn is None:
        return [upper_line]
    return [
        (warn, "orange", f"Advarselsgrense: {warn:g}{suffix}"),
        upper_line,
    ]


def report_title(kind: str, scope: Scope, variable: str | None = None, with_weather: bool = False) -> str:
    if scope.mode == "fall" and scope.year is not None:
        period = f"høsten {scope.year}"
    elif scope.mode == "spring" and scope.year is not None:
        period = f"våren {scope.year}"
    elif scope.mode == "year" and scope.year is not None:
        period = f"{scope.year}"
    else:
        period = format_scope_label(scope).lower()

    if kind == "availability":
        return f"Datadekning per rom {period}"

    if variable is None:
        raise ValueError("variable må oppgis for denne tittelen")

    label = report_label(variable)

    if kind == "building_distribution":
        return f"Fordeling av {label} per bygg {period}"
    if kind == "histogram":
        return f"Fordeling av {label} for alle målinger {period}"
    if kind == "above":
        return f"Målinger over øvre grense for {label} {period}"
    if kind == "below":
        return f"Målinger under nedre grense for {label} {period}"
    if kind == "warning":
        return f"Målinger over advarselsgrense for {label} {period}"
    if kind == "time_series":
        if with_weather:
            return f"Utvikling i {label} inne og ute {period}"
        return f"Utvikling i {label} {period}"
    return f"{label} {period}"

def report_time_series_limits(variable: str) -> Tuple[Optional[float], Optional[float]]:
    if variable == "Temperatur (°C)":
        return 10.0, 35.0
    if variable == "CO2 (ppm)":
        critical_value = THRESHOLDS_CRITICAL.get(variable)
        if critical_value is not None:
            return 400.0, float(2 * critical_value)
    critical_value = THRESHOLDS_CRITICAL.get(variable)
    if critical_value is not None:
        return 0.0, float(2 * critical_value)
    return None, None


def histogram_setup(variable: str) -> Tuple[float, float, FloatArray]:
    if variable == "Temperatur (°C)":
        x_min, x_max = 10.0, 35.0
        bins = np.arange(x_min, x_max + 1, 1)
    elif variable == "Luftfuktighet (%)":
        x_min, x_max = 15.0, 75.0
        bins = np.arange(x_min, x_max + 2, 2)
    elif variable == "CO2 (ppm)":
        x_min, x_max = 400.0, 1200.0
        bins = np.arange(400, 1200 + 30, 30)
    elif "Formaldehyd" in variable:
        x_min, x_max = 0.0, 125.0
        bins = np.arange(x_min, x_max + 20, 20)
    elif "TVOC" in variable:
        x_min, x_max = 0.0, 1500.0
        bins = np.arange(x_min, x_max + 30, 30)
    elif "PM 1.0" in variable:
        x_min, x_max = 0.0, 40.0
        bins = np.arange(x_min, x_max + 5, 5)
    elif "PM 2.5" in variable:
        x_min, x_max = 0.0, 40.0
        bins = np.arange(x_min, x_max + 5, 5)
    elif "PM 4" in variable:
        x_min, x_max = 0.0, 40.0
        bins = np.arange(x_min, x_max + 25, 25)
    elif "PM 10" in variable:
        x_min, x_max = 0.0, 110.0
        bins = np.arange(x_min, x_max + 5, 5)
    else:
        raise ValueError(f"Ingen forhåndsdefinert oppsett for '{variable}'.")

    return x_min, x_max, np.asarray(bins, dtype=float)


def make_scope(
    buildings: Optional[Sequence[str]] = None,
    rooms_by_building: Optional[Dict[str, Optional[Sequence[str]]]] = None,
    semester: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[datetime | pd.Timestamp] = None,
) -> Scope:
    scope = Scope(
        buildings=[str(building) for building in (buildings or TILGJENGELIGE_BYGG.keys())],
        rooms_by_building={
            str(building): None if rooms is None else [str(room) for room in rooms]
            for building, rooms in (rooms_by_building or {}).items()
        },
    )

    if semester is not None:
        semester_normalized = semester.strip().lower()
        if semester_normalized not in {"fall", "spring"}:
            raise ValueError("semester må være 'fall' eller 'spring'")
        if year is None:
            raise ValueError("year må oppgis når semester brukes")
        scope.mode = semester_normalized
        scope.year = year
        return scope

    if day is not None:
        day_timestamp = require_timestamp(day, "day")
        scope.day = day_timestamp
        scope.mode = "day"
        scope.year = day_timestamp.year
        scope.month = day_timestamp.month
        scope.week = int(day_timestamp.strftime("%W"))
        return scope

    if month is not None:
        if year is None:
            raise ValueError("year må oppgis når month brukes")
        scope.mode = "month"
        scope.year = year
        scope.month = month
        return scope

    if week is not None:
        if year is None:
            raise ValueError("year må oppgis når week brukes")
        scope.mode = "week"
        scope.year = year
        scope.week = week
        return scope

    if year is not None:
        scope.mode = "year"
        scope.year = year

    return scope


def _resolve_scope(
    scope: Optional[Scope] = None,
    semester: Optional[str] = None,
    year: Optional[int] = None,
) -> Scope:
    if scope is not None:
        return scope
    return make_scope(semester=semester, year=year)


def require_timestamp(value: object, label: str = "timestamp") -> pd.Timestamp:
    timestamp = pd.Timestamp(cast(Any, value))
    if pd.isna(timestamp):
        raise ValueError("Ugyldig %s: %s" % (label, value))
    return timestamp


def as_text(value: object) -> str:
    return str(cast(Any, value))


def unique_text_values(series: pd.Series, *, reverse: bool = False) -> List[str]:
    values = [as_text(value) for value in series.dropna().to_list()]
    return sorted(set(values), reverse=reverse)


def style_legend(legend: Any) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_alpha(1.0)
    frame.set_facecolor("white")
    frame.set_edgecolor("0.75")


def format_axis_date_label(timestamp: pd.Timestamp) -> str:
    return f"{timestamp.day}. {NORWEGIAN_MONTH_ABBREVIATIONS[timestamp.month]}"


def compute_frame_time_limits(df_list: Sequence[pd.DataFrame]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    valid_frames = [df for df in df_list if not df.empty]
    if not valid_frames:
        return None, None

    start_vis = min(require_timestamp(df.index.min(), "data start") for df in valid_frames)
    latest = max(require_timestamp(df.index.max(), "data end") for df in valid_frames)
    slutt_vis = shift_timestamp(require_timestamp(latest.normalize(), "data end day"), days=1)
    return start_vis, slutt_vis


def compute_datetime_series_limits(values: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    cleaned = values.dropna()
    if cleaned.empty:
        return None, None

    start_vis = require_timestamp(cleaned.min(), "series start")
    latest = require_timestamp(cleaned.max(), "series end")
    slutt_vis = shift_timestamp(require_timestamp(latest.normalize(), "series end day"), days=1)
    return start_vis, slutt_vis


def recommended_boxplot_left_limit(variable: str, current_left: float) -> float:
    if variable == "CO2 (ppm)":
        return 400.0
    if variable == "Temperatur (°C)":
        return max(0.0, 5.0 * float(np.floor(current_left / 5.0)))
    return 0.0


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
    clipped_values = np.asarray(y_values, dtype=float).copy()
    finite_mask = np.isfinite(clipped_values)
    over_mask = np.zeros(clipped_values.shape, dtype=bool)
    under_mask = np.zeros(clipped_values.shape, dtype=bool)

    if y_max is not None:
        over_mask = finite_mask & (clipped_values > y_max)
        clipped_values[over_mask] = y_max
    if y_min is not None:
        under_mask = finite_mask & (clipped_values < y_min)
        clipped_values[under_mask] = y_min

    line = ax.plot(
        x_values,
        clipped_values,
        label=label,
        color=color,
        linewidth=linewidth,
    )[0]

    if scatter_points:
        ax.scatter(x_values, clipped_values, color=color, s=10, zorder=5)

    if y_max is not None and np.any(over_mask):
        ax.scatter(
            x_values[over_mask],
            np.full(int(over_mask.sum()), y_max),
            marker="^",
            color=color,
            s=28,
            clip_on=False,
            zorder=6,
        )

    if y_min is not None and np.any(under_mask):
        ax.scatter(
            x_values[under_mask],
            np.full(int(under_mask.sum()), y_min),
            marker="v",
            color=color,
            s=28,
            clip_on=False,
            zorder=6,
        )

    return line


def set_datetime_xlim(ax: Axes, start_vis: pd.Timestamp, slutt_vis: pd.Timestamp) -> None:
    ax.set_xlim(    left=mdates.date2num(start_vis.to_pydatetime()),
                    right=mdates.date2num(slutt_vis.to_pydatetime()),)


def iter_room_variable_data(
    variable: str,
    scope: Scope,
) -> Iterable[Tuple[str, str, str, pd.Series]]:
    for building, room, filtered_df in iter_filtered_room_data(scope):
        if variable not in filtered_df.columns:
            continue

        series = pd.Series(pd.to_numeric(filtered_df[variable], errors="coerce"), index=filtered_df.index).dropna()
        if series.empty:
            continue

        label = f"R{room}" if len(scope.buildings) == 1 else f"B{building}-R{room}"
        yield str(building), str(room), label, series


def shift_timestamp(base: pd.Timestamp, *, days: int = 0, hours: int = 0) -> pd.Timestamp:
    shifted = base.to_pydatetime() + timedelta(days=days, hours=hours)
    return require_timestamp(shifted, "shifted timestamp")


def format_scope_label(scope: Scope) -> str:
    if scope.mode == "all":
        return "Hele perioden"
    if scope.mode == "year":
        return f"År {scope.year}"
    if scope.mode == "month":
        return f"Måned {scope.month}/{scope.year}"
    if scope.mode == "week":
        return f"Uke {scope.week} i {scope.year}"
    day_value = scope.day
    if scope.mode == "day" and day_value is not None:
        return day_value.strftime("%Y-%m-%d")
    if scope.mode == "fall":
        return f"Høst {scope.year}"
    if scope.mode == "spring":
        return f"Vår {scope.year}"
    return "Ukjent periode"


def build_title(subject: str, metric: str, scope: Scope) -> str:
    if scope.mode == "fall" and scope.year is not None:
        return f"{subject} – {metric} høsten {scope.year}"
    if scope.mode == "spring" and scope.year is not None:
        return f"{subject} – {metric} våren {scope.year}"
    if scope.mode == "year" and scope.year is not None:
        return f"{subject} – {metric} {scope.year}"
    if scope.mode == "month" and scope.year is not None and scope.month is not None:
        return f"{subject} – {metric} {NORWEGIAN_MONTHS[scope.month]} {scope.year}"
    if scope.mode == "week" and scope.year is not None and scope.week is not None:
        return f"{subject} – {metric} uke {scope.week} i {scope.year}"
    if scope.mode == "day" and scope.day is not None:
        return f"{subject} – {metric} {scope.day.strftime('%d.%m.%Y')}"
    if scope.mode == "all":
        return f"{subject} – {metric} hele måleperioden"
    return f"{subject} – {metric}"



def scope_folder_name(scope: Scope) -> str:
    if scope.mode == "all":
        return "alle_maledata"
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
    day_value = scope.day
    if scope.mode == "day" and day_value is not None:
        return day_value.strftime("dag_%Y_%m_%d")
    return "ukjent_scope"


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.strip("._") or "figur"


def save_figure(fig: plt.Figure, figure_dir: Path | str, stem: str, scope: Scope) -> Path:
    output_dir = Path(figure_dir) / scope_folder_name(scope)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sanitize_filename(stem)}.svg"
    fig.savefig(output_path, bbox_inches="tight")
    print(f"✅ Lagret figur: {output_path}")
    return output_path


def finalize_figure(
    fig: plt.Figure,
    figure_dir: Path | str,
    stem: str,
    scope: Scope,
    show: bool = True,
) -> Path:
    output_path = save_figure(fig, figure_dir, stem, scope)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def scope_to_filter_kwargs(scope: Scope) -> Dict[str, object]:
    return {
        "mode": scope.mode,
        "year": scope.year,
        "month": scope.month,
        "week": scope.week,
        "day": scope.day,
    }


def room_is_selected(scope: Scope, building: str, room: str) -> bool:
    selected_rooms = scope.rooms_by_building.get(str(building))
    if selected_rooms is None:
        return True
    return str(room) in {str(selected_room) for selected_room in selected_rooms}


def compute_scope_limits(
    scope: Scope,
    df_list: Optional[Sequence[pd.DataFrame]] = None,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if scope.mode == "day" and scope.day is not None:
        start_vis = require_timestamp(scope.day.replace(hour=0, minute=0, second=0), "scope start")
        slutt_vis = shift_timestamp(start_vis, days=1)
        return start_vis, slutt_vis

    if scope.mode == "week" and scope.year is not None and scope.week is not None:
        start_vis = require_timestamp(datetime.fromisocalendar(scope.year, scope.week, 1), "scope start")
        slutt_vis = shift_timestamp(start_vis, days=7)
        return start_vis, slutt_vis

    if scope.mode == "month" and scope.year is not None and scope.month is not None:
        start_vis = require_timestamp(pd.Timestamp(year=scope.year, month=scope.month, day=1), "scope start")
        month_end = require_timestamp(start_vis + pd.offsets.MonthEnd(1), "month end")
        slutt_vis = shift_timestamp(month_end, days=1)
        return start_vis, slutt_vis

    if scope.mode == "year" and scope.year is not None:
        start_vis = require_timestamp(pd.Timestamp(year=scope.year, month=1, day=1), "scope start")
        year_end = require_timestamp(pd.Timestamp(year=scope.year, month=12, day=31), "year end")
        slutt_vis = shift_timestamp(year_end, days=1)
        return start_vis, slutt_vis

    if scope.mode == "fall" and scope.year is not None:
        start_vis = require_timestamp(pd.Timestamp(year=scope.year, month=8, day=10), "scope start")
        fall_end = require_timestamp(pd.Timestamp(year=scope.year, month=12, day=10), "fall end")
        slutt_vis = shift_timestamp(fall_end, days=1)
        return start_vis, slutt_vis

    if scope.mode == "spring" and scope.year is not None:
        start_vis = require_timestamp(pd.Timestamp(year=scope.year, month=1, day=6), "scope start")
        spring_end = require_timestamp(pd.Timestamp(year=scope.year, month=6, day=6), "spring end")
        slutt_vis = shift_timestamp(spring_end, days=1)
        return start_vis, slutt_vis

    if df_list is not None:
        valid_frames = [df for df in df_list if not df.empty]
        if valid_frames:
            start_vis = min(require_timestamp(df.index.min(), "data start") for df in valid_frames)
            latest = max(require_timestamp(df.index.max(), "data end") for df in valid_frames)
            slutt_vis = shift_timestamp(latest, hours=1)
            return start_vis, slutt_vis

    return None, None


def configure_time_axis(
    ax: Axes,
    scope: Scope,
    start_vis: Optional[pd.Timestamp],
    slutt_vis: Optional[pd.Timestamp],
) -> None:
    if start_vis is not None and slutt_vis is not None:
        set_datetime_xlim(ax, start_vis, slutt_vis)

    if scope.mode in {"fall", "spring"} and start_vis is not None and slutt_vis is not None:
        display_end = shift_timestamp(slutt_vis, days=-1)
        major_ticks = [start_vis]

        month_ticks = pd.date_range(
            start=require_timestamp(start_vis.replace(day=1), "month tick start"),
            end=display_end,
            freq="MS",
        )
        for tick in month_ticks:
            tick_ts = require_timestamp(tick, "month tick")
            if start_vis < tick_ts < display_end:
                major_ticks.append(tick_ts)

        if display_end > start_vis:
            major_ticks.append(display_end)

        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(tick) for tick in major_ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter([format_axis_date_label(tick) for tick in major_ticks]))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="major", labelsize=9, rotation=0, pad=8)
        ax.tick_params(axis="x", which="minor", length=4)
        ax.grid(True, axis="x", which="major", linestyle=":", linewidth=0.7, alpha=0.7)
    elif scope.mode in {"year", "all"}:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(DateFormatter("%b"))
        ax.tick_params(axis="x", which="minor", labelsize=8, rotation=0, pad=10)
        for label in ax.get_xticklabels(minor=True):
            text = label.get_text()
            label.set_text(text[0] if text else "")
        ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.4)
    elif scope.mode == "month":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%d."))
    elif scope.mode == "week" and start_vis is not None and slutt_vis is not None:
        start_day = start_vis.normalize()
        day_ticks = [shift_timestamp(start_day, days=index) for index in range(7)] + [slutt_vis]
        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(day) for day in day_ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter(["man", "tir", "ons", "tor", "fre", "lor", "son", ""]))
    elif scope.mode == "day":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    ax.tick_params(axis="x", labelrotation=0, labelsize=9)


def iter_filtered_room_data(scope: Scope) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    filter_kwargs = scope_to_filter_kwargs(scope)

    for building in scope.buildings:
        dfs, room_names, _ = fetch_csv(directory=INNEKLIMA_DIR, building_number=str(building))
        for raw_df, room in zip(dfs, room_names):
            if not room_is_selected(scope, str(building), str(room)):
                continue

            room_df = set_datetime_index(raw_df)
            filtered_list = filter_data([room_df], **filter_kwargs)
            if not filtered_list:
                continue

            filtered_df = filtered_list[0]
            if filtered_df.empty:
                continue

            yield str(building), str(room), filtered_df


def collect_room_series(
    variable: str,
    scope: Scope,
    rename_to_value: bool = True,
) -> Tuple[List[pd.DataFrame], List[str]]:
    room_frames: List[pd.DataFrame] = []
    room_labels: List[str] = []
    column_name = "Value" if rename_to_value else variable

    for _, _, label, series in iter_room_variable_data(variable, scope):
        room_frames.append(series.to_frame(name=column_name))
        room_labels.append(label)

    return room_frames, room_labels



def collect_all_values_for_variable(variable: str, scope: Scope) -> pd.Series:
    series_collection = [series for _, _, _, series in iter_room_variable_data(variable, scope)]
    if not series_collection:
        return pd.Series(dtype="float64")
    return pd.concat(series_collection, axis=0)


def series_to_float_array(series: pd.Series) -> FloatArray:
    numeric_series = pd.to_numeric(series, errors="coerce")
    return np.asarray(numeric_series.to_numpy(dtype=float, na_value=np.nan), dtype=float)


def insert_nan_for_gaps(df: pd.DataFrame, column: str) -> pd.Series:
    df_sorted = df.sort_index().copy()
    time_diff_seconds = df_sorted.index.to_series().diff().dt.total_seconds()
    gap_mask = time_diff_seconds.gt(3600).fillna(False)
    df_sorted["tdiff"] = time_diff_seconds
    df_sorted.loc[gap_mask, column] = pd.NA
    df_sorted.drop(columns=["tdiff"], inplace=True)

    start_index = require_timestamp(df_sorted.index.min(), "series start")
    end_index = require_timestamp(df_sorted.index.max(), "series end")
    full_index = pd.date_range(start=start_index, end=end_index, freq="1h")
    return df_sorted[column].reindex(full_index)

def _plot_room_series_with_weather(
    df_list: List[pd.DataFrame],
    scope: Scope,
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
    room_names: Optional[List[str]] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    if not df_list:
        raise ValueError(empty_message)

    resolved_room_names: Sequence[str] = room_names if room_names is not None else []
    y_min, y_max = report_time_series_limits(title_metric)

    show_weather = False
    weather_df_filtered = pd.DataFrame()
    if df_weather is not None:
        weather_df_filtered = filter_weather(df_weather, **scope_to_filter_kwargs(scope))
        show_weather = not weather_df_filtered.empty

    if show_weather:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15},
        )
        ax, ax_weather = cast(Tuple[Axes, Axes], tuple(axes))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax_weather = None

    color_map = colormaps["tab10"]
    room_handles: List[Line2D] = []
    room_labels: List[str] = []

    for index, room_df in enumerate(df_list, start=1):
        if value_column not in room_df.columns or room_df.empty:
            continue

        series = insert_nan_for_gaps(room_df, value_column)
        series_values = series_to_float_array(series)
        label = resolved_room_names[index - 1] if index - 1 < len(resolved_room_names) else f"Rom {index}"
        line = plot_capped_series(
            ax,
            series.index,
            series_values,
            color=color_map((index - 1) % 10),
            label=label,
            linewidth=2,
            y_min=y_min,
            y_max=y_max,
            scatter_points=scope.mode == "day",
        )
        room_handles.append(line)
        room_labels.append(label)

    threshold_handles: List[Line2D] = []
    threshold_labels: List[str] = []
    used_labels: set[str] = set()

    for level, color, label in thresholds:
        plot_label = label if label is not None else "_nolegend_"
        handle = ax.axhline(y=level, color=color, linestyle="--", linewidth=2, label=plot_label)
        if label is None or label in used_labels:
            continue
        threshold_handles.append(handle)
        threshold_labels.append(label)
        used_labels.add(label)

    if y_min is not None:
        ax.set_ylim(bottom=float(y_min))
    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    ax.set_ylabel(indoor_ylabel)
    if ax_weather is None:
        ax.set_xlabel("Dato")

    ax.grid(True)

    if room_handles:
        room_legend = ax.legend(
            room_handles,
            room_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            frameon=True,
            fontsize=9,
            labelspacing=0.3,
            borderaxespad=0.0,
        )
        style_legend(room_legend)
        ax.add_artist(room_legend)

    if threshold_handles:
        threshold_legend = ax.legend(
            threshold_handles,
            threshold_labels,
            loc="upper right",
            frameon=True,
        )
        style_legend(threshold_legend)

    if ax_weather is not None:
        for column, label, color in weather_specs:
            if column not in weather_df_filtered.columns:
                continue
            series = weather_df_filtered[column].dropna()
            if series.empty:
                continue
            ax_weather.plot(series.index, series_to_float_array(series), label=label, color=color, linewidth=1.5)

        ax_weather.set_ylabel(weather_ylabel)
        ax_weather.set_xlabel("Dato")
        ax_weather.grid(True)
        weather_legend = ax_weather.legend(loc="upper right", frameon=True)
        style_legend(weather_legend)

    if scope.mode in {"fall", "spring"}:
        start_vis, slutt_vis = compute_frame_time_limits(df_list)
    else:
        start_vis, slutt_vis = compute_scope_limits(scope, df_list)
    configure_time_axis(ax, scope, start_vis, slutt_vis)
    if ax_weather is not None:
        configure_time_axis(ax_weather, scope, start_vis, slutt_vis)

    ax.set_title(report_title("time_series", scope, title_metric, with_weather=ax_weather is not None))

    if ax_weather is not None:
        fig.subplots_adjust(left=0.07, right=0.82, top=0.92, bottom=0.10, hspace=0.15)
    else:
        fig.subplots_adjust(left=0.07, right=0.82, top=0.92, bottom=0.10)

    return finalize_figure(fig, figure_dir, f"{stem}_{scope_folder_name(scope)}", scope, show)



def plot_temperature(
    df_list: List[pd.DataFrame],
    scope: Scope,
    df_weather: Optional[pd.DataFrame] = None,
    room_names: Optional[List[str]] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    return _plot_room_series_with_weather(
        df_list,
        scope,
        value_column="Temperatur (°C)",
        indoor_ylabel=report_label("Temperatur (°C)", indoor=True),
        thresholds=report_thresholds("Temperatur (°C)", context="full"),
        weather_specs=[
            ("utetemp_seklima", "Seklima", "blue"),
            ("utetemp_kunak", "Kunak", "orange"),
        ],
        weather_ylabel=report_label("Temperatur (°C)", outdoor=True),
        title_metric="Temperatur (°C)",
        stem="tidsserie_temperatur",
        empty_message="Ingen innendørs temperaturdata i valgt datasett.",
        df_weather=df_weather,
        room_names=room_names,
        figure_dir=figure_dir,
        show=show,
    )




def plot_humidity(
    df_list: List[pd.DataFrame],
    scope: Scope,
    df_weather: Optional[pd.DataFrame] = None,
    room_names: Optional[List[str]] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    return _plot_room_series_with_weather(
        df_list,
        scope,
        value_column="Luftfuktighet (%)",
        indoor_ylabel=report_label("Luftfuktighet (%)", indoor=True),
        thresholds=report_thresholds("Luftfuktighet (%)", context="full"),
        weather_specs=[
            ("ute_rh_seklima", "Seklima", "blue"),
            ("ute_rh_kunak", "Kunak", "orange"),
        ],
        weather_ylabel=report_label("Luftfuktighet (%)", outdoor=True),
        title_metric="Luftfuktighet (%)",
        stem="tidsserie_fuktighet",
        empty_message="Ingen innendørs fuktighetsdata i valgt datasett.",
        df_weather=df_weather,
        room_names=room_names,
        figure_dir=figure_dir,
        show=show,
    )




def plot_all_rooms_variable(
    df_list: List[pd.DataFrame],
    variable: str,
    scope: Scope,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    if not df_list:
        raise ValueError(f"Ingen data funnet for variabelen '{variable}'.")

    y_label = report_label(variable)
    y_min, y_max = report_time_series_limits(variable)
    thresholds = report_thresholds(variable, context="full")

    fig, ax = plt.subplots(figsize=(12, 6))
    color_map = colormaps["tab20"]

    for index, room_df in enumerate(df_list, start=1):
        if room_df.empty:
            continue

        column_name = "Value" if "Value" in room_df.columns else variable
        if column_name not in room_df.columns:
            continue

        room_frame = room_df.sort_index().copy()
        if not isinstance(room_frame.index, pd.DatetimeIndex):
            room_frame = set_datetime_index(room_frame)

        series = insert_nan_for_gaps(room_frame, column_name)
        plot_capped_series(
            ax,
            series.index,
            series_to_float_array(series),
            color=color_map((index - 1) % 20),
            label="_nolegend_",
            linewidth=1,
            y_min=y_min,
            y_max=y_max,
        )

    for value, color, label in sorted(thresholds, key=lambda item: item[0], reverse=True):
        if label is None:
            ax.axhline(y=value, color=color, linestyle="--", linewidth=1.5)
        else:
            ax.axhline(y=value, color=color, linestyle="--", linewidth=1.5, label=label)

    if y_min is not None:
        ax.set_ylim(bottom=float(y_min))
    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    if scope.mode in {"fall", "spring"}:
        start_vis, slutt_vis = compute_frame_time_limits(df_list)
    else:
        start_vis, slutt_vis = compute_scope_limits(scope, df_list)
    configure_time_axis(ax, scope, start_vis, slutt_vis)

    ax.set_title(report_title("time_series", scope, variable))
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Dato")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    threshold_legend = ax.legend(loc="upper right", frameon=True)
    style_legend(threshold_legend)
    plt.tight_layout()

    stem = f"tidsserie_{sanitize_filename(variable)}_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def draw_thresholds(ax: Axes, variable: str) -> None:
    for value, color, label in report_thresholds(variable, context="full"):
        if label is None:
            continue
        ax.axvline(value, color=color, linestyle="--", linewidth=2, label=label)


def collect_boxplot_data_by_building(variable: str, scope: Scope) -> Tuple[List[List[float]], List[str]]:
    values_by_building: Dict[str, List[float]] = {str(building): [] for building in scope.buildings}

    for building, _, _, series in iter_room_variable_data(variable, scope):
        values_by_building[str(building)].extend(
            float(value) for value in series_to_float_array(series) if not np.isnan(value)
        )

    building_data: List[List[float]] = []
    building_labels: List[str] = []

    for building in scope.buildings:
        values = values_by_building[str(building)]
        if values:
            building_data.append(values)
            building_labels.append(f"Bygg {int(building)}")

    return building_data, building_labels


def build_boxplot_summary_df(building_data: Sequence[Sequence[float]], building_labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for label, values in zip(building_labels, building_data):
        series = pd.Series(values, dtype="float64").dropna()
        if series.empty:
            continue

        rows.append(
            {
                "Bygg": label,
                "Antall": int(series.count()),
                "Minimum": round(series.min(), 2),
                "Q1": round(series.quantile(0.25), 2),
                "Median": round(series.median(), 2),
                "Gjennomsnitt": round(series.mean(), 2),
                "Q3": round(series.quantile(0.75), 2),
                "Maksimum": round(series.max(), 2),
            }
        )

    return pd.DataFrame(rows)


def print_boxplot_summary(variable: str, building_data: Sequence[Sequence[float]], building_labels: Sequence[str], scope: Scope) -> None:
    summary_df = build_boxplot_summary_df(building_data, building_labels)
    print(f"\nBoxplot-verdier for {variable}")
    print(f"Periode: {format_scope_label(scope)}")

    if summary_df.empty:
        print("Ingen data tilgjengelig.\n")
        return

    print(summary_df.to_string(index=False))
    print()


def print_pm_boxplot_summary(pm_plot_data: Sequence[Tuple[str, Sequence[Sequence[float]], Sequence[str]]], scope: Scope) -> None:
    print("\nBoxplot-verdier for partikler")
    print(f"Periode: {format_scope_label(scope)}\n")

    for variable, building_data, building_labels in pm_plot_data:
        summary_df = build_boxplot_summary_df(building_data, building_labels)
        print(variable)
        if summary_df.empty:
            print("Ingen data tilgjengelig.\n")
        else:
            print(summary_df.to_string(index=False))
            print()

def draw_horizontal_boxplot_panel(
    ax: Axes,
    variable: str,
    building_data: Sequence[Sequence[float]],
    building_labels: Sequence[str],
    *,
    xlabel: str,
) -> None:
    ax.boxplot(list(building_data)[::-1], tick_labels=list(building_labels)[::-1], vert=False)
    draw_thresholds(ax, variable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bygg")

    if variable in {"Temperatur (°C)", "Luftfuktighet (%)"}:
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis="x", which="minor", length=4)
        ax.grid(axis="x", which="major", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

    current_left, _ = ax.get_xlim()
    ax.set_xlim(left=recommended_boxplot_left_limit(variable, float(current_left)))

    legend = ax.legend(loc="upper right", frameon=True)
    style_legend(legend)



def plot_building_boxplot(
    variable: str,
    building_data: Sequence[Sequence[float]],
    building_labels: Sequence[str],
    scope: Scope,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    if not building_data:
        raise ValueError(f"Ingen data funnet for {variable} i valgt datasett.")

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_horizontal_boxplot_panel(ax, variable, building_data, building_labels, xlabel=report_label(variable))
    ax.set_title(report_title("building_distribution", scope, variable))
    plt.tight_layout()

    stem = f"boxplot_{sanitize_filename(variable)}_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)



def plot_pm_boxplots(
    pm_plot_data: Sequence[Tuple[str, Sequence[Sequence[float]], Sequence[str]]],
    scope: Scope,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    flat_axes = cast(Sequence[Axes], axes.flatten())

    for ax, (variable, building_data, building_labels) in zip(flat_axes, pm_plot_data):
        if not building_data:
            ax.axis("off")
            continue

        draw_horizontal_boxplot_panel(ax, variable, building_data, building_labels, xlabel=report_label(variable))
        ax.set_title(report_title("building_distribution", scope, variable))

    plt.suptitle(f"Fordeling av partikler per bygg {format_scope_label(scope).lower()}")
    plt.tight_layout()

    stem = f"boxplot_pm_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)



def build_threshold_scatter_specs(variable: str) -> List[Tuple[float, ThresholdDirection, str]]:
    specs: List[Tuple[float, ThresholdDirection, str]] = []

    for value, _, _ in report_thresholds(variable, context="below"):
        specs.append((value, "below", "below"))
    for value, _, _ in report_thresholds(variable, context="above"):
        specs.append((value, "above", "above"))

    return specs


def collect_threshold_scatter_data(
    variable: str,
    scope: Scope,
    threshold: float,
    direction: ThresholdDirection,
) -> pd.DataFrame:
    rows = []

    for building, room, filtered_df in iter_filtered_room_data(scope):
        if variable not in filtered_df.columns:
            continue

        series = pd.Series(pd.to_numeric(filtered_df[variable], errors="coerce"), index=filtered_df.index).dropna()
        if series.empty:
            continue

        if direction == "above":
            breaches = series[series > threshold]
        elif direction == "below":
            breaches = series[series < threshold]
        else:
            raise ValueError(f"Ukjent retning: {direction}")

        for timestamp, value in breaches.items():
            rows.append(
                {
                    "Tid": timestamp,
                    "Verdi": float(value),
                    "Byggkode": building,
                    "Bygg": f"Bygg {int(building)}",
                    "Rom": room,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Tid", "Verdi", "Byggkode", "Bygg", "Rom"])

    df = pd.DataFrame(rows)
    df.sort_values(by=["Tid", "Bygg", "Rom"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def plot_threshold_scatter(
    scatter_df: pd.DataFrame,
    variable: str,
    threshold: float,
    direction: ThresholdDirection,
    scope: Scope,
    title: str,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Optional[Path]:
    if scatter_df.empty:
        direction_text = "over" if direction == "above" else "under"
        print(f"❌ Ingen målinger {direction_text} {threshold:g} for {variable} i valgt datasett.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = scatter_df.copy()

    if plot_df["Bygg"].nunique() == 1:
        group_col = "Rom"
        plot_df[group_col] = plot_df[group_col].fillna("Ukjent rom")
    else:
        group_col = "Bygg"

    group_series = plot_df[group_col].map(as_text)
    groups = unique_text_values(plot_df[group_col])
    color_map = colormaps["tab10"].resampled(max(1, len(groups)))

    for index, group in enumerate(groups):
        subset = plot_df[group_series == group]
        ax.scatter(subset["Tid"], subset["Verdi"], label=group, color=color_map(index), s=18, alpha=0.75)

    threshold_lines = report_thresholds(variable, context=title)
    if threshold_lines:
        _, threshold_color, threshold_text = threshold_lines[0]
        ax.axhline(y=threshold, color=threshold_color, linestyle="--", linewidth=2, label=threshold_text)

    if scope.mode in {"fall", "spring"}:
        start_vis, slutt_vis = compute_datetime_series_limits(plot_df["Tid"])
    else:
        start_vis, slutt_vis = compute_scope_limits(scope)
    if start_vis is not None and slutt_vis is not None:
        configure_time_axis(ax, scope, start_vis, slutt_vis)
    else:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(DateFormatter("%d.%m.%Y"))

    ax.set_ylabel(report_label(variable))
    ax.set_xlabel("Dato")
    ax.set_title(report_title(title, scope, variable))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    threshold_pairs = [(handle, label) for handle, label in zip(handles, labels) if "grense" in label.lower()]
    group_pairs = [(handle, label) for handle, label in zip(handles, labels) if "grense" not in label.lower()]
    legend = ax.legend(
        [handle for handle, _ in threshold_pairs + group_pairs],
        [label for _, label in threshold_pairs + group_pairs],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=True,
    )
    style_legend(legend)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.90, bottom=0.12)

    stem = f"scatter_{sanitize_filename(variable)}_{title}_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def plot_room_data_availability(
    scope: Scope,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
    directory: Path = INNEKLIMA_DIR,
) -> Path:
    overview_rows: List[Dict[str, object]] = []

    for building in scope.buildings:
        dfs, room_names, _ = fetch_csv(directory=directory, building_number=str(building))
        room_frames = [set_datetime_index(df) for df in dfs]

        for room, room_df in zip(room_names, room_frames):
            if not room_is_selected(scope, str(building), str(room)):
                continue

            filtered_list = filter_data([room_df], **scope_to_filter_kwargs(scope))
            if not filtered_list:
                continue

            filtered_df = filtered_list[0].copy()
            if filtered_df.empty or filtered_df.index.inferred_type != "datetime64":
                continue

            unique_days = sorted(pd.DatetimeIndex(filtered_df.index).normalize().unique())
            for day_value in unique_days:
                start = require_timestamp(day_value, "availability start")
                end = shift_timestamp(start, days=1)
                overview_rows.append(
                    {
                        "Bygg": str(building),
                        "Rom": str(room),
                        "Start": start,
                        "Slutt": end,
                    }
                )

    availability_df = pd.DataFrame(overview_rows)
    if availability_df.empty:
        raise ValueError("Ingen data å vise for valgt datasett.")

    availability_df["Etikett"] = "B" + availability_df["Bygg"] + "-R" + availability_df["Rom"]
    availability_df.sort_values(by=["Etikett", "Start"], inplace=True)
    availability_df.reset_index(drop=True, inplace=True)

    unique_buildings = unique_text_values(availability_df["Bygg"])
    color_map = colormaps["tab10"].resampled(len(unique_buildings))
    building_colors = {
        building: color_map(index)
        for index, building in enumerate(unique_buildings)
    }

    unique_labels = unique_text_values(availability_df["Etikett"], reverse=True)
    fig, ax = plt.subplots(figsize=(12, max(5.0, len(unique_labels) * 0.5)))

    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    min_date = require_timestamp(availability_df["Start"].min(), "availability minimum")
    max_date = require_timestamp(availability_df["Slutt"].max(), "availability maximum")

    for row in availability_df.itertuples(index=False):
        row_label = as_text(row.Etikett)
        row_building = as_text(row.Bygg)
        row_start = require_timestamp(row.Start, "availability row start")
        row_end = require_timestamp(row.Slutt, "availability row end")
        y_value = label_to_index[row_label]
        ax.plot(
            [mdates.date2num(row_start.to_pydatetime()), mdates.date2num(row_end.to_pydatetime())],
            [y_value, y_value],
            linewidth=3,
            color=building_colors[row_building],
            zorder=1,
        )

    ax.set_yticks(list(label_to_index.values()))
    ax.set_yticklabels(unique_labels)
    configure_time_axis(ax, scope, min_date, max_date)

    for building, color in building_colors.items():
        ax.plot(np.array([], dtype=float), np.array([], dtype=float), label=building, color=color, linewidth=6)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), frameon=True)
    style_legend(legend)
    ax.set_xlabel("Dato")
    ax.set_ylabel("Romkode")
    ax.set_title(report_title("availability", scope))
    plt.tight_layout()

    stem = f"datadekning_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def get_semester_threshold_rule(variable: str) -> Optional[SemesterThresholdRule]:
    if variable == "Temperatur (°C)":
        return SemesterThresholdRule(
            lower=THRESHOLDS_TEMPERATURE["night"]["min"],
            upper=THRESHOLDS_TEMPERATURE["day"]["max"],
            lower_label="For kaldt (%)",
            upper_label="For varmt (%)",
            lower_pass_label="Oppfyller kald-kravet",
            upper_pass_label="Oppfyller varm-kravet",
            two_sided=True,
        )

    if variable == "Luftfuktighet (%)":
        limits = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        return SemesterThresholdRule(
            lower=limits["critical_min"],
            upper=limits["critical_max"],
            lower_label="For tørt (%)",
            upper_label="For fuktig (%)",
            lower_pass_label="Oppfyller tørr-kravet",
            upper_pass_label="Oppfyller fuktig-kravet",
            two_sided=True,
        )

    upper = THRESHOLDS_CRITICAL.get(variable)
    if upper is not None:
        return SemesterThresholdRule(
            lower=None,
            upper=upper,
            upper_label="Over grense (%)",
            upper_pass_label="Oppfyller 5 %-kravet",
            two_sided=False,
        )

    return None

def get_exceedance_series(series: pd.Series, threshold: float, direction: str) -> pd.Series:
    if direction == "below":
        return (threshold - series[series < threshold]).dropna()
    if direction == "above":
        return (series[series > threshold] - threshold).dropna()
    raise ValueError(f"Ukjent retning: {direction}")


def median_exceedance(series: pd.Series, threshold: float, direction: str) -> float:
    exceedance = get_exceedance_series(series, threshold, direction)
    if exceedance.empty:
        return 0.0
    return round(float(exceedance.median()), 2)



def longest_breach_duration_hours(mask: pd.Series) -> int:
    if mask.empty or not mask.any():
        return 0

    change_groups = (mask.astype(bool) != mask.astype(bool).shift()).cumsum()
    run_lengths = mask.astype(bool).groupby(change_groups).sum()
    if run_lengths.empty:
        return 0
    return int(run_lengths.max())


def max_exceedance(series: pd.Series, threshold: float, direction: str) -> float:
    exceedance = get_exceedance_series(series, threshold, direction)
    if exceedance.empty:
        return 0.0
    return round(float(exceedance.max()), 2)


def collect_semester_summary_by_building(variable: str, scope: Scope, max_outside_pct: float = 5.0) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rule = get_semester_threshold_rule(variable)
    if rule is None:
        return pd.DataFrame()

    lower_limit = rule.lower
    upper_limit = rule.upper
    if lower_limit is None and upper_limit is None:
        return pd.DataFrame()

    for building in scope.buildings:
        building_series: List[pd.Series] = []
        room_stats: List[Tuple[str, float]] = []

        for room_building, room, filtered_df in iter_filtered_room_data(scope):
            if room_building != str(building):
                continue
            if variable not in filtered_df.columns:
                continue

            series = pd.Series(pd.to_numeric(filtered_df[variable], errors="coerce"), index=filtered_df.index).dropna()
            if series.empty:
                continue

            building_series.append(series)

            room_below_mask = pd.Series(False, index=series.index)
            room_above_mask = pd.Series(False, index=series.index)
            if lower_limit is not None:
                room_below_mask = series < lower_limit
            if upper_limit is not None:
                room_above_mask = series > upper_limit

            room_outside_pct = 100 * (room_below_mask | room_above_mask).mean()
            room_stats.append((room, float(room_outside_pct)))

        if not building_series:
            continue

        combined_series = pd.Series(pd.concat(building_series, axis=0)).dropna()
        if combined_series.empty:
            continue

        below_mask = pd.Series(False, index=combined_series.index)
        above_mask = pd.Series(False, index=combined_series.index)
        if lower_limit is not None:
            below_mask = combined_series < lower_limit
        if upper_limit is not None:
            above_mask = combined_series > upper_limit

        below_pct = 100 * below_mask.mean()
        above_pct = 100 * above_mask.mean()
        below_median_exceedance = 0.0
        above_median_exceedance = 0.0
        below_longest_hours = 0
        above_longest_hours = 0
        above_max_exceedance = 0.0

        if lower_limit is not None:
            below_median_exceedance = median_exceedance(combined_series, lower_limit, "below")
            below_longest_hours = longest_breach_duration_hours(below_mask)
        if upper_limit is not None:
            above_median_exceedance = median_exceedance(combined_series, upper_limit, "above")
            above_longest_hours = longest_breach_duration_hours(above_mask)
            above_max_exceedance = max_exceedance(combined_series, upper_limit, "above")

        worst_room = max(room_stats, key=lambda item: item[1])[0] if room_stats else "-"
        row: Dict[str, object] = {"Bygg": f"Bygg {int(building)}"}

        if rule.two_sided:
            assert rule.lower_label is not None
            assert rule.upper_label is not None
            assert rule.lower_pass_label is not None
            assert rule.upper_pass_label is not None

            row[rule.lower_label] = round(below_pct, 2)
            row[rule.upper_label] = round(above_pct, 2)

            if variable == "Temperatur (°C)":
                row["Median kald-avvik"] = below_median_exceedance
                row["Median varm-avvik"] = above_median_exceedance
                row["Lengste kald-brudd (t)"] = below_longest_hours
                row["Lengste varm-brudd (t)"] = above_longest_hours
            else:
                row["Median tørr-avvik"] = below_median_exceedance
                row["Median fuktig-avvik"] = above_median_exceedance
                row["Lengste tørre brudd (t)"] = below_longest_hours
                row["Lengste fuktige brudd (t)"] = above_longest_hours

            row[rule.lower_pass_label] = "Ja" if below_pct < max_outside_pct else "Nei"
            row[rule.upper_pass_label] = "Ja" if above_pct < max_outside_pct else "Nei"
        else:
            assert rule.upper_label is not None
            assert rule.upper_pass_label is not None

            row[rule.upper_label] = round(above_pct, 2)
            row["Median overskridelse"] = above_median_exceedance
            row["Maks overskridelse"] = above_max_exceedance
            row["Lengste brudd (t)"] = above_longest_hours
            row[rule.upper_pass_label] = "Ja" if above_pct < max_outside_pct else "Nei"

        row["Verste rom"] = f"Rom {worst_room}" if worst_room != "-" else "-"
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    summary_df["Bygg_sort"] = summary_df["Bygg"].str.extract(r"(\d+)").astype(int)
    summary_df.sort_values(by=["Bygg_sort"], inplace=True)
    summary_df.drop(columns=["Bygg_sort"], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df


def plot_distribution(
    series: pd.Series,
    variable: str,
    scope: Scope,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    threshold_values = report_thresholds(variable, context="full")

    data: FloatArray = np.asarray(pd.to_numeric(series, errors="coerce").dropna().to_list(), dtype=float)
    if data.size == 0:
        raise ValueError(f"Ingen gyldige målinger for '{variable}'.")

    x_min, x_max, bins = histogram_setup(variable)

    fig, ax = plt.subplots(figsize=(9, 6))

    weights: FloatArray = np.ones(data.shape, dtype=float) * (100.0 / float(len(data)))
    ax.hist(data, bins=bins, weights=weights, color="skyblue", edgecolor="black")
    ax.set_ylabel("Relativ frekvens (%)")
    ax.set_xlabel(report_label(variable))
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=float(x_min), right=float(x_max))

    existing_ticks: List[float] = [float(tick) for tick in ax.get_xticks()]
    for level, _, _ in threshold_values:
        if level < x_min:
            existing_ticks.append(float(x_min))
        elif level > x_max:
            existing_ticks.append(float(x_max))
        else:
            existing_ticks.append(float(level))
    existing_ticks = sorted(set(existing_ticks))
    ax.set_xticks(existing_ticks)
    ax.set_xticklabels([f"{int(tick)}" for tick in existing_ticks])

    used_colors = set()
    for level, color, label in threshold_values:
        visible_level = float(np.clip(level, x_min, x_max))
        visible_label = label if color not in used_colors else ""
        ax.axvline(x=visible_level, color=color, linestyle="--", linewidth=2, label=visible_label)
        used_colors.add(color)

    ax.set_title(report_title("histogram", scope, variable))
    legend = ax.legend(loc="upper right", fontsize=9, frameon=True)
    style_legend(legend)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    stem = f"fordeling_{sanitize_filename(variable)}_{scope_folder_name(scope)}"
    return finalize_figure(fig, figure_dir, stem, scope, show)


def run_time_series(
    variable: str,
    scope: Optional[Scope] = None,
    compare_weather: bool = False,
    semester: Optional[str] = None,
    year: Optional[int] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    resolved_variable = resolve_variable_choice(variable)
    resolved_scope = _resolve_scope(scope, semester=semester, year=year)
    rename_to_value = not compare_weather
    dfs, room_labels = collect_room_series(resolved_variable, resolved_scope, rename_to_value=rename_to_value)
    if not dfs:
        raise ValueError(f"Ingen data funnet for variabelen '{resolved_variable}' i valgt datasett.")

    if compare_weather:
        if resolved_variable not in {"Temperatur (°C)", "Luftfuktighet (%)"}:
            raise ValueError("Uteklima-sammenligning støttes bare for temperatur og luftfuktighet.")

        weather_df = fetch_weather()
        if resolved_variable == "Temperatur (°C)":
            return plot_temperature(
                dfs,
                scope=resolved_scope,
                df_weather=weather_df,
                room_names=room_labels,
                figure_dir=figure_dir,
                show=show,
            )

        return plot_humidity(
            dfs,
            scope=resolved_scope,
            df_weather=weather_df,
            room_names=room_labels,
            figure_dir=figure_dir,
            show=show,
        )

    return plot_all_rooms_variable(
        dfs,
        variable=resolved_variable,
        scope=resolved_scope,
        figure_dir=figure_dir,
        show=show,
    )


def run_distribution(
    variable: str,
    scope: Optional[Scope] = None,
    semester: Optional[str] = None,
    year: Optional[int] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    resolved_variable = resolve_variable_choice(variable)
    resolved_scope = _resolve_scope(scope, semester=semester, year=year)
    series = collect_all_values_for_variable(resolved_variable, resolved_scope)
    if series.empty:
        raise ValueError(f"Ingen data funnet for variabelen '{resolved_variable}' i valgt datasett.")
    return plot_distribution(series, resolved_variable, resolved_scope, figure_dir=figure_dir, show=show)


def run_boxplot(
    variable: str,
    scope: Optional[Scope] = None,
    semester: Optional[str] = None,
    year: Optional[int] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    resolved_variable = resolve_variable_choice(variable)
    resolved_scope = _resolve_scope(scope, semester=semester, year=year)

    if resolved_variable.strip().lower() in {"pm", "partikler"}:
        pm_plot_data = []
        for pm_variable in PM_VARIABLER:
            building_data, building_labels = collect_boxplot_data_by_building(pm_variable, resolved_scope)
            pm_plot_data.append((pm_variable, building_data, building_labels))

        print_pm_boxplot_summary(pm_plot_data, resolved_scope)
        return plot_pm_boxplots(pm_plot_data, resolved_scope, figure_dir=figure_dir, show=show)

    building_data, building_labels = collect_boxplot_data_by_building(resolved_variable, resolved_scope)
    if not building_data:
        raise ValueError(f"Ingen data funnet for {resolved_variable} i valgt datasett.")

    print_boxplot_summary(resolved_variable, building_data, building_labels, resolved_scope)
    return plot_building_boxplot(resolved_variable, building_data, building_labels, resolved_scope, figure_dir=figure_dir, show=show)


def run_threshold_scatter(
    variable: str,
    scope: Optional[Scope] = None,
    semester: Optional[str] = None,
    year: Optional[int] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> List[Path]:
    resolved_variable = resolve_variable_choice(variable)
    resolved_scope = _resolve_scope(scope, semester=semester, year=year)
    specs: List[Tuple[float, ThresholdDirection, str]] = build_threshold_scatter_specs(resolved_variable)
    if not specs:
        raise ValueError(f"Ingen grenseverdier definert for '{resolved_variable}'.")

    saved_paths: List[Path] = []
    for threshold, direction_value, title in specs:
        direction = cast(ThresholdDirection, direction_value)
        scatter_df = collect_threshold_scatter_data(resolved_variable, resolved_scope, threshold, direction)
        saved_path = plot_threshold_scatter(
            scatter_df,
            variable=resolved_variable,
            threshold=threshold,
            direction=direction,
            scope=resolved_scope,
            title=title,
            figure_dir=figure_dir,
            show=show,
        )
        if saved_path is not None:
            saved_paths.append(saved_path)

    return saved_paths


def run_data_availability(
    scope: Optional[Scope] = None,
    semester: Optional[str] = None,
    year: Optional[int] = None,
    figure_dir: Path | str = FIGURE_DIR,
    show: bool = True,
) -> Path:
    resolved_scope = _resolve_scope(scope, semester=semester, year=year)
    return plot_room_data_availability(resolved_scope, figure_dir=figure_dir, show=show)


def run_semester_analysis(
    scope: Optional[Scope] = None,
    semester: Optional[str] = None,
    year: Optional[int] = None,
    variables: Optional[Sequence[str]] = None,
    max_outside_pct: float = 5.0,
) -> Dict[str, pd.DataFrame]:
    resolved_scope = _resolve_scope(scope, semester=semester, year=year)
    if resolved_scope.mode not in {"fall", "spring"}:
        raise ValueError("run_semester_analysis krever et semester-scope, for eksempel make_scope(semester='fall', year=2023).")

    selected_variables = [resolve_variable_choice(variable) for variable in (variables or VARIABLE_CHOICES.values())]
    print("\nSEMESTERANALYSE PER BYGG")
    print(f"Kriterium: mindre enn {max_outside_pct:g} % av målingene utenfor grenseverdiene")
    print(f"Periode: {format_scope_label(resolved_scope)}")

    summaries: Dict[str, pd.DataFrame] = {}
    for variable in selected_variables:
        summary_df = collect_semester_summary_by_building(variable, resolved_scope, max_outside_pct=max_outside_pct)
        summaries[variable] = summary_df

        print(f"\n{variable}")
        print("-" * len(variable))
        if summary_df.empty:
            print("Ingen data funnet for denne variabelen.")
            continue

        with pd.option_context("display.max_columns", None, "display.width", 1400):
            print(summary_df.to_string(index=False))
            print()

    return summaries


if __name__ == "__main__":
    h_23 = make_scope(semester="fall", year=2023)
    sem = h_23

    variabel = "2"
    valgt_variabel = resolve_variable_choice(variabel)
    sammenlign_ute = valgt_variabel in {"Temperatur (°C)", "Luftfuktighet (%)"}

    # Funksjoner:
    run_time_series(valgt_variabel, scope=sem, compare_weather=sammenlign_ute)
    run_distribution(valgt_variabel, scope=sem)
    run_boxplot(valgt_variabel, scope=sem)
    run_threshold_scatter(valgt_variabel, scope=sem)
    # run_data_availability(scope=sem)
    # run_semester_analysis(scope=sem)
