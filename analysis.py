"""
Analyse- og datagrunnlag for ENE306.

Denne modulen binder sammen config.py, data_processing.py og plotting.py.
Den skal ikke inneholde menytekst, men den skal hente, filtrere og strukturere
dataene som figurene trenger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import plotting
from config import (
    INNEKLIMA_DIR,
    PM_VARIABLER,
    THRESHOLDS_CRITICAL,
    THRESHOLDS_OPTIMAL_HUMIDITY,
    THRESHOLDS_TEMPERATURE,
    TILGJENGELIGE_BYGG,
    VARIABLE_CHOICES,
)
from data_processing import fetch_csv, fetch_weather, filter_data, list_room_numbers, set_datetime_index


TABLE_DIR = Path(__file__).parent / "tabeller"
BREACH_TARGET_PERCENT = 5.0
SENSOR_RESOLUTION_DECIMALS = {
    VARIABLE_CHOICES["1"]: 1,
    VARIABLE_CHOICES["2"]: 0,
    VARIABLE_CHOICES["3"]: 0,
    VARIABLE_CHOICES["4"]: 0,
    VARIABLE_CHOICES["5"]: 0,
    VARIABLE_CHOICES["6"]: 0,
    VARIABLE_CHOICES["7"]: 0,
    VARIABLE_CHOICES["8"]: 0,
    VARIABLE_CHOICES["9"]: 0,
}
_FILTERED_ROOM_DATA_CACHE: Dict[tuple[object, ...], List[Tuple[str, str, pd.DataFrame]]] = {}


@dataclass
class Scope:
    """Beskriver aktivt utvalg for analysene."""

    buildings: List[str] = field(default_factory=lambda: list(TILGJENGELIGE_BYGG.keys()))
    rooms_by_building: Dict[str, Optional[List[str]]] = field(default_factory=dict)
    mode: str = "all"
    year: Optional[int] = None
    month: Optional[int] = None
    week: Optional[int] = None
    day: Optional[pd.Timestamp] = None


def make_default_scope() -> Scope:
    """Returner standardutvalg: alle bygg, alle rom og hele måleperioden."""
    return Scope()


def format_scope_label(scope: Scope) -> str:
    """Lag en norsk tekstbeskrivelse av perioden."""
    if scope.mode == "all":
        return "Hele perioden"
    if scope.mode == "year":
        return f"År {scope.year}"
    if scope.mode == "month":
        return f"Måned {scope.month}/{scope.year}"
    if scope.mode == "week":
        return f"Uke {scope.week} i {scope.year}"
    if scope.mode == "fall":
        return f"Høst {scope.year}"
    if scope.mode == "spring":
        return f"Vår {scope.year}"
    if scope.mode == "day" and scope.day is not None:
        return scope.day.strftime("%d.%m.%Y")
    return "Ukjent periode"


def describe_scope(scope: Scope) -> str:
    """Beskriv aktivt utvalg for visning i menyen."""
    building_text = ", ".join(f"Bygg {int(building)}" for building in scope.buildings)
    room_parts = []

    for building in scope.buildings:
        selected_rooms = scope.rooms_by_building.get(str(building))
        if selected_rooms:
            room_parts.append(f"Bygg {int(building)}: rom {', '.join(selected_rooms)}")

    room_text = "; ".join(room_parts) if room_parts else "Alle rom i valgte bygg"
    return f"Periode: {format_scope_label(scope)}\nBygg: {building_text}\nRom: {room_text}"


def scope_to_filter_kwargs(scope: Scope) -> dict[str, object]:
    """Oversett Scope til argumentene data_processing.filter_data bruker."""
    return {
        "mode": scope.mode,
        "year": scope.year,
        "month": scope.month,
        "week": scope.week,
        "day": scope.day,
    }


def get_available_rooms(building: str) -> List[str]:
    """Hent romnumre for ett bygg basert på filnavn."""
    return list_room_numbers(INNEKLIMA_DIR, str(building))


def room_sort_key(room: object) -> tuple[int, object]:
    """Sorter romnummer stabilt, også hvis et romnavn ikke er rent tall."""
    room_text = str(room)
    if room_text.isdigit():
        return 0, int(room_text)
    return 1, room_text


def scope_cache_key(scope: Scope) -> tuple[object, ...]:
    """Lag nøkkel for cache basert på aktivt bygg-, rom- og periodeutvalg."""
    room_parts = []
    for building in scope.buildings:
        selected_rooms = scope.rooms_by_building.get(str(building))
        if selected_rooms is None:
            room_parts.append((str(building), None))
        else:
            room_parts.append((str(building), tuple(sorted((str(room) for room in selected_rooms), key=room_sort_key))))

    day_key = None if scope.day is None else pd.Timestamp(scope.day).isoformat()
    return (
        tuple(str(building) for building in scope.buildings),
        tuple(room_parts),
        scope.mode,
        scope.year,
        scope.month,
        scope.week,
        day_key,
    )


def get_filtered_room_data(scope: Scope) -> List[Tuple[str, str, pd.DataFrame]]:
    """Les og filtrer romdata én gang per aktivt utvalg.

    Semesterpakken lager mange figurer fra samme periode. Uten cache måtte samme
    CSV-filer leses og resamples på nytt for hver variabel. Denne funksjonen
    mellomlagrer ferdig filtrerte romdata så videre beregninger bare plukker ut
    ønsket kolonne.
    """
    cache_key = scope_cache_key(scope)
    if cache_key in _FILTERED_ROOM_DATA_CACHE:
        return _FILTERED_ROOM_DATA_CACHE[cache_key]

    filter_kwargs = scope_to_filter_kwargs(scope)
    room_data: List[Tuple[str, str, pd.DataFrame]] = []

    for building in scope.buildings:
        data_frames, room_names, _ = fetch_csv(directory=INNEKLIMA_DIR, building_number=str(building))

        for raw_df, room in zip(data_frames, room_names):
            if not room_is_selected(scope, str(building), str(room)):
                continue

            room_df = set_datetime_index(raw_df)
            filtered_list = filter_data([room_df], **filter_kwargs)
            if not filtered_list:
                continue

            filtered_df = filtered_list[0]
            if not filtered_df.empty:
                room_data.append((str(building), str(room), filtered_df))

    _FILTERED_ROOM_DATA_CACHE[cache_key] = room_data
    return room_data


def room_is_selected(scope: Scope, building: str, room: str) -> bool:
    """Sjekk om et rom skal være med i aktivt utvalg."""
    selected_rooms = scope.rooms_by_building.get(str(building))
    if selected_rooms is None:
        return True
    return str(room) in selected_rooms


def iter_filtered_room_data(scope: Scope) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    """Iterer gjennom alle romdata som inngår i aktivt utvalg."""
    yield from get_filtered_room_data(scope)


def iter_room_variable_data(scope: Scope, variable: str) -> Iterable[Tuple[str, str, str, pd.Series]]:
    """Iterer gjennom numeriske måleserier for én variabel."""
    for building, room, filtered_df in iter_filtered_room_data(scope):
        if variable not in filtered_df.columns:
            continue

        numeric_series = pd.Series(pd.to_numeric(filtered_df[variable], errors="coerce"), index=filtered_df.index)
        numeric_series = numeric_series.dropna()
        if numeric_series.empty:
            continue

        label = f"R{room}" if len(scope.buildings) == 1 else f"B{building}-R{room}"
        yield building, room, label, numeric_series


def collect_room_series(
    scope: Scope,
    variable: str,
    rename_to_value: bool = True,
) -> List[pd.DataFrame]:
    """Samle én DataFrame per rom til tidsserieplot."""
    room_frames: List[pd.DataFrame] = []
    column_name = "Value" if rename_to_value else variable

    for _, _, _, series in iter_room_variable_data(scope, variable):
        room_frames.append(series.to_frame(name=column_name))

    return room_frames


def collect_boxplot_data_by_building(scope: Scope, variable: str) -> Tuple[List[List[float]], List[str]]:
    """Samle alle rom i hvert bygg til ett datagrunnlag per bygg."""
    values_by_building: Dict[str, List[float]] = {str(building): [] for building in scope.buildings}

    for building, _, _, series in iter_room_variable_data(scope, variable):
        values_by_building[str(building)].extend(float(value) for value in series.dropna().to_list())

    building_data: List[List[float]] = []
    building_labels: List[str] = []

    for building in scope.buildings:
        values = values_by_building[str(building)]
        if values:
            building_data.append(values)
            building_labels.append(f"Bygg {int(building)}")

    return building_data, building_labels


def calculate_outlier_counts(series: pd.Series) -> Tuple[int, int]:
    """Tell outliers etter IQR-regelen som brukes i boxplot."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return int((series < lower_limit).sum()), int((series > upper_limit).sum())


def threshold_limits_for_variable(variable: str) -> Tuple[Optional[float], Optional[float]]:
    """Returner nedre og øvre grense som brukes i grensebrudd-tabellen."""
    if variable == "Temperatur (°C)":
        return float(THRESHOLDS_TEMPERATURE["night"]["min"]), float(THRESHOLDS_TEMPERATURE["day"]["max"])

    if variable == "Luftfuktighet (%)":
        humidity_limits = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        return float(humidity_limits["critical_min"]), float(humidity_limits["critical_max"])

    upper_limit = THRESHOLDS_CRITICAL.get(variable)
    if upper_limit is None:
        return None, None
    return None, float(upper_limit)


def calculate_breach_episode_metrics(series: pd.Series, variable: str) -> dict[str, object]:
    """Tell sammenhengende grensebrudd og lengste varighet.

    Et brudd starter når en verdi går utenfor terskel og avsluttes først når
    neste gyldige verdi er tilbake innenfor terskel. Siden datagrunnlaget
    resamples til timesmiddel i data_processing.py, tolkes ett målepunkt her
    som omtrent én time. Tidshull større enn 1,5 time bryter episoden, slik at
    manglende data ikke blir regnet som et sammenhengende brudd.
    """
    lower_limit, upper_limit = threshold_limits_for_variable(variable)
    if lower_limit is None and upper_limit is None:
        return {"Antall brudd": 0, "Lengste brudd (timer)": 0}

    numeric_series = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index).dropna()
    if numeric_series.empty:
        return {"Antall brudd": 0, "Lengste brudd (timer)": 0}

    if isinstance(numeric_series.index, pd.DatetimeIndex):
        numeric_series = numeric_series.sort_index()

    outside_threshold = pd.Series(False, index=numeric_series.index)
    if lower_limit is not None:
        outside_threshold = outside_threshold | (numeric_series < lower_limit)
    if upper_limit is not None:
        outside_threshold = outside_threshold | (numeric_series > upper_limit)

    breach_count = 0
    current_length = 0
    longest_length = 0
    previous_time: Optional[pd.Timestamp] = None

    for timestamp, is_outside in outside_threshold.items():
        current_time = pd.Timestamp(timestamp) if isinstance(numeric_series.index, pd.DatetimeIndex) else None
        has_large_gap = (
            previous_time is not None
            and current_time is not None
            and current_time - previous_time > pd.Timedelta(hours=1.5)
        )

        if has_large_gap and current_length > 0:
            longest_length = max(longest_length, current_length)
            current_length = 0

        if bool(is_outside):
            if current_length == 0:
                breach_count += 1
            current_length += 1
        else:
            longest_length = max(longest_length, current_length)
            current_length = 0

        previous_time = current_time

    longest_length = max(longest_length, current_length)
    return {"Antall brudd": int(breach_count), "Lengste brudd (timer)": int(longest_length)}



def sensor_resolution_decimals(variable: Optional[str]) -> int:
    """Returner antall desimaler tabellverdier bør vises med.

    Oppløsningen følger sensorspesifikasjonen: temperatur vises med 0,1,
    mens relativ fuktighet, CO2, TVOC, formaldehyd og luftpartikler vises i
    hele enheter.
    """
    if variable is None:
        return 2
    return SENSOR_RESOLUTION_DECIMALS.get(variable, 2)


def round_sensor_value(value: object, variable: Optional[str]) -> object:
    """Avrund en tabellverdi etter måleoppløsningen for valgt variabel."""
    if value is None or pd.isna(value):
        return None

    decimals = sensor_resolution_decimals(variable)
    quantum = Decimal("1") if decimals == 0 else Decimal("1").scaleb(-decimals)
    rounded_value = Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)

    if decimals == 0:
        return int(rounded_value)
    return float(rounded_value)

def calculate_threshold_breach_metrics(series: pd.Series, variable: str) -> dict[str, object]:
    """Beregn tid over/under terskel, bruddepisoder og om 5 %-målet er oppfylt."""
    lower_limit, upper_limit = threshold_limits_for_variable(variable)
    numeric_series = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index).dropna()
    count = int(numeric_series.count())

    if count == 0:
        return {
            "Nedre terskel": round_sensor_value(lower_limit, variable),
            "Øvre terskel": round_sensor_value(upper_limit, variable),
            "Tid under terskel (% tid)": 0.0,
            "Tid over terskel (% tid)": 0.0,
            "Antall brudd": 0,
            "Lengste brudd (timer)": 0,
            "Grensebrudd < 5,00 (% tid)": "Ingen data",
        }

    below_percent = 0.0
    above_percent = 0.0

    if lower_limit is not None:
        below_percent = float((numeric_series < lower_limit).sum()) / count * 100.0
    if upper_limit is not None:
        above_percent = float((numeric_series > upper_limit).sum()) / count * 100.0

    meets_target = below_percent <= BREACH_TARGET_PERCENT and above_percent <= BREACH_TARGET_PERCENT
    return {
        "Nedre terskel": round_sensor_value(lower_limit, variable),
        "Øvre terskel": round_sensor_value(upper_limit, variable),
        "Tid under terskel (% tid)": round(below_percent, 2),
        "Tid over terskel (% tid)": round(above_percent, 2),
        **calculate_breach_episode_metrics(numeric_series, variable),
        "Grensebrudd < 5,00 (% tid)": "Ja" if meets_target else "Nei",
    }


def build_boxplot_summary_df(
    building_data: Sequence[Sequence[float]],
    building_labels: Sequence[str],
    variable: Optional[str] = None,
) -> pd.DataFrame:
    """Lag tabellverdier som samsvarer med boxplotgrunnlaget."""
    rows = []

    for label, values in zip(building_labels, building_data):
        series = pd.Series(values, dtype="float64").dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_outliers, upper_outliers = calculate_outlier_counts(series)

        row: dict[str, object] = {
            "Bygg": label,
            "Antall": int(series.count()),
            "Q1": round_sensor_value(q1, variable),
            "Q3": round_sensor_value(q3, variable),
            "IQR": round_sensor_value(iqr, variable),
            "Nedre outliers": lower_outliers,
            "Øvre outliers": upper_outliers,
        }
        if variable is not None:
            row.update(calculate_threshold_breach_metrics(series, variable))

        rows.append(row)

    return pd.DataFrame(rows)


def print_boxplot_summary(variable: str, building_data: Sequence[Sequence[float]], building_labels: Sequence[str], scope: Scope) -> None:
    """Skriv tabellverdier for boxplot i terminalen."""
    summary_df = build_boxplot_summary_df(building_data, building_labels, variable)
    print(f"\nBoxplot-verdier for {variable}")
    print(f"Periode: {format_scope_label(scope)}")

    if summary_df.empty:
        print("Ingen data tilgjengelig.")
        return

    print(summary_df.to_string(index=False))


def print_pm_boxplot_summary(pm_plot_data: Sequence[Tuple[str, Sequence[Sequence[float]], Sequence[str]]], scope: Scope) -> None:
    """Skriv tabellverdier for alle luftpartikkelvariabler."""
    print("\nBoxplot-verdier for luftpartikler")
    print(f"Periode: {format_scope_label(scope)}")

    for variable, building_data, building_labels in pm_plot_data:
        print(f"\n{variable}")
        summary_df = build_boxplot_summary_df(building_data, building_labels, variable)
        if summary_df.empty:
            print("Ingen data tilgjengelig.")
        else:
            print(summary_df.to_string(index=False))


def expand_boxplot_variables(variables: Sequence[str]) -> List[str]:
    """Utvid PM-gruppe til enkeltvariabler og behold stabil rekkefølge."""
    expanded_variables: List[str] = []
    seen_variables: set[str] = set()

    for variable in variables:
        resolved_variable = "pm" if variable.strip().lower() in {"pm", "partikler", "luftpartikler"} else resolve_variable_choice(variable)
        selected_variables = PM_VARIABLER if resolved_variable == "pm" or resolved_variable in PM_VARIABLER else [resolved_variable]

        for selected_variable in selected_variables:
            if selected_variable not in seen_variables:
                expanded_variables.append(selected_variable)
                seen_variables.add(selected_variable)

    return expanded_variables


def build_building_level_summary_table(scope: Scope, variable: str) -> pd.DataFrame:
    """Lag byggvis tabell og tell brudd per rom før byggverdiene samles.

    Fordelingsverdier som Q1 og Q3 beregnes på alle målepunkter i bygget.
    Antall brudd summeres derimot rom for rom, fordi et brudd er en tidsperiode
    i ett rom. Dette unngår at to ulike rom tolkes som én sammenhengende episode.
    """
    rows: List[dict[str, object]] = []
    series_by_building: Dict[str, List[pd.Series]] = {str(building): [] for building in scope.buildings}

    for building, _, _, series in iter_room_variable_data(scope, variable):
        series_by_building[str(building)].append(series)

    for building in scope.buildings:
        room_series_list = [
            pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index).dropna()
            for series in series_by_building.get(str(building), [])
        ]
        room_series_list = [series for series in room_series_list if not series.empty]
        if not room_series_list:
            continue

        combined_series = pd.concat(room_series_list, axis=0).dropna()
        if combined_series.empty:
            continue

        q1 = combined_series.quantile(0.25)
        q3 = combined_series.quantile(0.75)
        iqr = q3 - q1
        lower_outliers, upper_outliers = calculate_outlier_counts(combined_series)
        threshold_metrics = calculate_threshold_breach_metrics(combined_series, variable)
        room_breach_metrics = [calculate_breach_episode_metrics(series, variable) for series in room_series_list]

        threshold_metrics["Antall brudd"] = sum(int(metrics["Antall brudd"]) for metrics in room_breach_metrics)
        threshold_metrics["Lengste brudd (timer)"] = max(
            (int(metrics["Lengste brudd (timer)"]) for metrics in room_breach_metrics),
            default=0,
        )

        row: dict[str, object] = {
            "Periode": format_scope_label(scope),
            "Variabel": variable,
            "Bygg": f"Bygg {int(building)}",
            "Antall rom": int(len(room_series_list)),
            "Antall": int(combined_series.count()),
            "Q1": round_sensor_value(q1, variable),
            "Q3": round_sensor_value(q3, variable),
            "IQR": round_sensor_value(iqr, variable),
            "Nedre outliers": lower_outliers,
            "Øvre outliers": upper_outliers,
        }
        row.update(threshold_metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def build_combined_boxplot_summary_table(scope: Scope, variables: Sequence[str]) -> pd.DataFrame:
    """Lag én samlet tabell med boxplot-, terskel- og bruddverdier."""
    columns = [
        "Periode",
        "Variabel",
        "Bygg",
        "Antall rom",
        "Antall",
        "Q1",
        "Q3",
        "IQR",
        "Nedre outliers",
        "Øvre outliers",
        "Nedre terskel",
        "Øvre terskel",
        "Tid under terskel (% tid)",
        "Tid over terskel (% tid)",
        "Antall brudd",
        "Lengste brudd (timer)",
        "Grensebrudd < 5,00 (% tid)",
    ]
    rows: List[dict[str, object]] = []
    for variable in expand_boxplot_variables(variables):
        table = build_building_level_summary_table(scope, variable)
        if table.empty:
            continue

        for record in table.to_dict("records"):
            rows.append({column: record.get(column) for column in columns})

    return pd.DataFrame(rows, columns=columns)


def save_table_values(summary_df: pd.DataFrame, scope: Scope, stem: str = "boxplot_tabellverdier") -> str:
    """Lagre samlet tabell som semikolonseparert CSV for norsk Excel/rapportbruk."""
    output_dir = TABLE_DIR / plotting.scope_folder_name(scope)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_{plotting.scope_folder_name(scope)}.csv"
    summary_df.to_csv(output_path, sep=";", decimal=",", encoding="utf-8-sig", index=False)
    return str(output_path)


def build_room_level_summary_table(scope: Scope, variables: Sequence[str]) -> pd.DataFrame:
    """Lag egen tabell med boxplot- og terskelverdier per rom.

    Denne brukes for Formaldehyd og TVOC, fordi romvise avvik er mer nyttig enn
    bare bygggjennomsnitt når enkeltrom kan ha lokale kilder eller ventilasjonsavvik.
    """
    columns = [
        "Periode",
        "Variabel",
        "Bygg",
        "Rom",
        "Antall",
        "Q1",
        "Q3",
        "Q4",
        "IQR",
        "Nedre outliers",
        "Øvre outliers",
        "Nedre terskel",
        "Øvre terskel",
        "Tid under terskel (% tid)",
        "Tid over terskel (% tid)",
        "Antall brudd",
        "Lengste brudd (timer)",
        "Grensebrudd < 5,00 (% tid)",
    ]
    rows: List[dict[str, object]] = []

    for variable in variables:
        resolved_variable = resolve_variable_choice(variable)
        for building, room, _, series in iter_room_variable_data(scope, resolved_variable):
            numeric_series = pd.Series(series, dtype="float64").dropna()
            if numeric_series.empty:
                continue

            q1 = numeric_series.quantile(0.25)
            q3 = numeric_series.quantile(0.75)
            iqr = q3 - q1
            q4 = q3 + 1.5 * iqr
            lower_outliers, upper_outliers = calculate_outlier_counts(numeric_series)

            row: dict[str, object] = {
                "Periode": format_scope_label(scope),
                "Variabel": resolved_variable,
                "Bygg": f"Bygg {int(building)}",
                "Rom": str(room),
                "Antall": int(numeric_series.count()),
                "Q1": round_sensor_value(q1, resolved_variable),
                "Q3": round_sensor_value(q3, resolved_variable),
                "Q4": round_sensor_value(q4, resolved_variable),
                "IQR": round_sensor_value(iqr, resolved_variable),
                "Nedre outliers": lower_outliers,
                "Øvre outliers": upper_outliers,
            }
            row.update(calculate_threshold_breach_metrics(numeric_series, resolved_variable))
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def build_room_availability_table(scope: Scope) -> pd.DataFrame:
    """Lag datagrunnlag for datadekningsfigur."""
    rows: List[dict[str, object]] = []

    for building, room, filtered_df in iter_filtered_room_data(scope):
        unique_days = sorted(pd.DatetimeIndex(filtered_df.index).normalize().unique())

        for day_value in unique_days:
            start = pd.Timestamp(day_value)
            rows.append(
                {
                    "Bygg": str(building),
                    "Rom": str(room),
                    "Start": start,
                    "Slutt": start + pd.Timedelta(days=1),
                }
            )

    return pd.DataFrame(rows)


def build_room_measurement_period_table(scope: Scope) -> pd.DataFrame:
    """Lag oversikt over første og siste gyldige måling per rom.

    Tabellen brukes som dokumentasjon av datagrunnlaget. Første og siste måling
    beregnes etter samme filtrering og timesresampling som figurene bruker, slik
    at tabellen beskriver nøyaktig perioden som faktisk analyseres.
    """
    columns = [
        "Periode",
        "Sensor",
        "Bygg",
        "Rom",
        "Første måling",
        "Siste måling",
        "Antall timer med data",
        "Måleperiode (dager)",
    ]
    rows: List[dict[str, object]] = []
    for building, room, filtered_df in iter_filtered_room_data(scope):
        # Tabellen skal beskrive faktiske målepunkter. Derfor konverteres alle
        # kolonner til tall, og rader uten noen numeriske måleverdier ignoreres.
        numeric_measurements = filtered_df.apply(pd.to_numeric, errors="coerce")
        valid_measurements = numeric_measurements.dropna(how="all")

        if valid_measurements.empty:
            continue

        measurement_index = pd.DatetimeIndex(valid_measurements.index).dropna().unique().sort_values()
        if len(measurement_index) == 0:
            continue

        first_measurement = pd.Timestamp(measurement_index.min())
        last_measurement = pd.Timestamp(measurement_index.max())
        measurement_days = (last_measurement - first_measurement).total_seconds() / (24 * 60 * 60)

        rows.append(
            {
                "Periode": format_scope_label(scope),
                "Sensor": f"Bygg {int(building)} - rom {room}",
                "Bygg": f"Bygg {int(building)}",
                "Rom": str(room),
                "Første måling": first_measurement.strftime("%d.%m.%Y %H:%M"),
                "Siste måling": last_measurement.strftime("%d.%m.%Y %H:%M"),
                "Antall timer med data": int(len(measurement_index)),
                "Måleperiode (dager)": round(measurement_days, 2),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def resolve_variable_choice(choice: str) -> str:
    """Tillat både tallvalg og direkte variabelnavn."""
    cleaned = choice.strip()
    if cleaned in VARIABLE_CHOICES:
        return VARIABLE_CHOICES[cleaned]

    normalized = cleaned.lower()
    for variable in VARIABLE_CHOICES.values():
        if normalized == variable.lower():
            return variable

    valid_choices = ", ".join(VARIABLE_CHOICES)
    raise ValueError(f"Ukjent variabelvalg '{choice}'. Gyldige tallvalg er {valid_choices}.")



def copy_scope(scope: Scope) -> Scope:
    """Lag en uavhengig kopi av aktivt utvalg."""
    return Scope(
        buildings=list(scope.buildings),
        rooms_by_building=deepcopy(scope.rooms_by_building),
        mode=scope.mode,
        year=scope.year,
        month=scope.month,
        week=scope.week,
        day=scope.day,
    )


def make_semester_scope(base_scope: Scope, semester: str, year: int) -> Scope:
    """Lag semesterutvalg basert på valgte bygg og rom."""
    semester_scope = copy_scope(base_scope)
    semester_scope.mode = semester
    semester_scope.year = year
    semester_scope.month = None
    semester_scope.week = None
    semester_scope.day = None
    return semester_scope


def get_data_processing_notes(scope: Scope) -> List[str]:
    """Returner en kort og redelig beskrivelse av databehandlingen."""
    return [
        "CSV-filer leses fra mappen som er definert i config.py.",
        "Filene sorteres før lesing slik at bygg og rom behandles i stabil rekkefølge.",
        "Romnummer hentes fra filnavnet, for eksempel data_RES08015_...csv -> rom 5.",
        "Date og Time kombineres til en datetime-indeks.",
        "Kolonnenavn oversettes til norske rapportnavn som Temperatur (°C) og Luftfuktighet (%).",
        "Måleverdier konverteres til tall; komma og punktum håndteres som desimalskilletegn.",
        "Inneklimadata resamples til timesmiddel, og rader uten måleverdier fjernes.",
        f"Data filtreres til aktiv periode: {format_scope_label(scope)}.",
        "Tidsserieplot bryter linjen ved tidshull, slik at manglende data ikke tegnes som kontinuerlig måling.",
        "Boxplot viser outlier-punkter og teller outliers i terminaltabell eller lagret tabellfil.",
        "Figurer lagres som PDF i figurer/<periode>/ med stabile filnavn.",
    ]


def print_data_processing_report(scope: Scope, variables: Sequence[str] | None = None) -> None:
    """Skriv databehandlingen synlig i terminalen før rapportkjøring."""
    print("\n" + "-" * 72)
    print("Databehandling og aktivt utvalg")
    print("-" * 72)
    print(describe_scope(scope))

    if variables:
        print("\nVariabler som kjøres:")
        for variable in variables:
            if variable == "pm":
                print(f"- Luftpartikler samlet ({', '.join(PM_VARIABLER)})")
            else:
                print(f"- {variable}")

    print("\nData behandles slik:")
    for note in get_data_processing_notes(scope):
        print(f"- {note}")
    print()


def run_semester_package(base_scope: Scope, semester: str, year: int, show: bool = False) -> List[str]:
    """Kjør alle rapportfigurene for valgt semester.

    Dette er ment som en reproduserbar rapportkjøring. Den bruker valgte bygg og
    rom fra aktivt utvalg, men setter perioden til valgt vår- eller høstsemester.
    Tabellverdier for boxplot og terskelbrudd lagres samlet i én CSV-fil.
    """
    if semester not in {"spring", "fall"}:
        raise ValueError("Semester må være 'spring' eller 'fall'.")

    semester_scope = make_semester_scope(base_scope, semester, year)
    time_series_variables = [
        VARIABLE_CHOICES["1"],
        VARIABLE_CHOICES["2"],
        VARIABLE_CHOICES["3"],
        VARIABLE_CHOICES["4"],
        VARIABLE_CHOICES["5"],
        "pm",
    ]
    boxplot_variables = [
        VARIABLE_CHOICES["1"],
        VARIABLE_CHOICES["2"],
        VARIABLE_CHOICES["3"],
        VARIABLE_CHOICES["4"],
        VARIABLE_CHOICES["5"],
        "pm",
    ]
    saved_paths: List[str] = []

    print_data_processing_report(semester_scope, time_series_variables)
    print("Starter semesterpakke. Dette kan ta litt tid.\n")

    tasks: List[Tuple[str, Callable[[], str]]] = [
        ("Datadekning per rom", lambda: run_data_availability(semester_scope, show=show)),
    ]

    for time_series_variable in time_series_variables:
        include_weather = time_series_variable in {"Temperatur (°C)", "Luftfuktighet (%)"}
        tasks.append(
            (
                f"Tidsserie - {time_series_variable}",
                lambda selected_variable=time_series_variable, selected_weather=include_weather: run_time_series(
                    semester_scope,
                    selected_variable,
                    compare_weather=selected_weather,
                    show=show,
                ),
            )
        )

    for boxplot_variable in boxplot_variables:
        label = "Luftpartikler samlet" if boxplot_variable == "pm" else boxplot_variable
        tasks.append(
            (
                f"Boxplot - {label}",
                lambda selected_variable=boxplot_variable: run_boxplot(
                    semester_scope,
                    selected_variable,
                    show=show,
                    print_table=False,
                ),
            )
        )

    for title, task_function in tasks:
        print(f"Kjører: {title}")
        try:
            saved_path = task_function()
            saved_paths.append(saved_path)
            print(f"  Lagret: {saved_path}")
        except ValueError as error:
            print(f"  Hoppet over: {error}")
        except RuntimeError as error:
            print(f"  Datafeil: {error}")

    summary_table = build_combined_boxplot_summary_table(semester_scope, boxplot_variables)
    if summary_table.empty:
        print("\nIngen tabellverdier å lagre for valgt semester.")
    else:
        table_path = save_table_values(summary_table, semester_scope)
        saved_paths.append(table_path)
        print(f"\nTabellverdier lagret: {table_path}")

    room_table_variables = [VARIABLE_CHOICES["4"], VARIABLE_CHOICES["5"]]
    room_summary_table = build_room_level_summary_table(semester_scope, room_table_variables)
    if room_summary_table.empty:
        print("Ingen romvise tabellverdier å lagre for Formaldehyd og TVOC.")
    else:
        room_table_path = save_table_values(
            room_summary_table,
            semester_scope,
            stem="rom_tabellverdier_formaldehyd_tvoc",
        )
        saved_paths.append(room_table_path)
        print(f"Romvise tabellverdier for Formaldehyd og TVOC lagret: {room_table_path}")

    measurement_period_table = build_room_measurement_period_table(semester_scope)
    if measurement_period_table.empty:
        print("Ingen oversikt over første og siste måling å lagre.")
    else:
        measurement_period_path = save_table_values(
            measurement_period_table,
            semester_scope,
            stem="sensor_maleperiode",
        )
        saved_paths.append(measurement_period_path)
        print(f"Oversikt over første og siste måling lagret: {measurement_period_path}")

    print("\nSemesterpakken er ferdig.")
    print(f"Antall filer lagret: {len(saved_paths)}")
    return saved_paths

def run_time_series(
    scope: Scope,
    variable: str,
    compare_weather: bool = False,
    show: bool = True,
) -> str:
    """Lag tidsseriefigur og returner lagret filsti."""
    resolved_variable = "pm" if variable.strip().lower() in {"pm", "partikler", "luftpartikler"} else resolve_variable_choice(variable)
    if resolved_variable in PM_VARIABLER or resolved_variable == "pm":
        pm_plot_data = []
        for pm_variable in PM_VARIABLER:
            room_frames = collect_room_series(scope, pm_variable, rename_to_value=True)
            pm_plot_data.append((pm_variable, room_frames))

        return str(plotting.plot_pm_time_series(pm_plot_data, scope, show=show))

    rename_to_value = not compare_weather
    room_frames = collect_room_series(scope, resolved_variable, rename_to_value=rename_to_value)

    if not room_frames:
        raise ValueError(f"Ingen data funnet for {resolved_variable} i valgt utvalg.")

    if compare_weather and resolved_variable in {"Temperatur (°C)", "Luftfuktighet (%)"}:
        weather_df = fetch_weather()
        if resolved_variable == "Temperatur (°C)":
            path = plotting.plot_temperature(room_frames, scope, df_weather=weather_df, show=show)
        else:
            path = plotting.plot_humidity(room_frames, scope, df_weather=weather_df, show=show)
    else:
        path = plotting.plot_all_rooms_variable(room_frames, resolved_variable, scope, show=show)

    return str(path)


def run_boxplot(scope: Scope, variable: str, show: bool = True, print_table: bool = True) -> str:
    """Lag boxplot og eventuelt skriv tabellgrunnlag i terminalen."""
    resolved_variable = "pm" if variable.strip().lower() in {"pm", "partikler", "luftpartikler"} else resolve_variable_choice(variable)
    if resolved_variable in PM_VARIABLER or resolved_variable == "pm":
        pm_plot_data = []

        for pm_variable in PM_VARIABLER:
            building_data, building_labels = collect_boxplot_data_by_building(scope, pm_variable)
            pm_plot_data.append((pm_variable, building_data, building_labels))

        if print_table:
            print_pm_boxplot_summary(pm_plot_data, scope)
        return str(plotting.plot_pm_boxplots(pm_plot_data, scope, show=show))

    building_data, building_labels = collect_boxplot_data_by_building(scope, resolved_variable)

    if not building_data:
        raise ValueError(f"Ingen data funnet for {resolved_variable} i valgt utvalg.")

    if print_table:
        print_boxplot_summary(resolved_variable, building_data, building_labels, scope)
    return str(plotting.plot_building_boxplot(resolved_variable, building_data, building_labels, scope, show=show))

def run_data_availability(scope: Scope, show: bool = True) -> str:
    """Lag datadekningsfigur og returner lagret filsti."""
    availability_table = build_room_availability_table(scope)
    if availability_table.empty:
        raise ValueError("Ingen datadekning å vise for valgt utvalg.")

    return str(plotting.plot_room_data_availability(availability_table, scope, show=show))
