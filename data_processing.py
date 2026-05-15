"""
Databehandling for ENE306-analyseverktøyet.

Denne modulen har ett ansvar: lese rå CSV-filer og gjøre dem klare for analyse.
All brukerinteraksjon ligger i main.py, og all plotting ligger i plotting.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from config import INNEKLIMA_DIR, INNEKLIMA_PREFIX_TEMPLATE, KUNAK_FILE, SEKLIMA_FILE


COLUMN_TRANSLATIONS = {
    "Temperature (°C)": "Temperatur (°C)",
    "Humidity (%)": "Luftfuktighet (%)",
    "Formaldehyde (µg/m³)": "Formaldehyd (µg/m³)",
    "TVOC (ppb)": "TVOC (ppb)",
    "PM 1.0 (µg/m³)": "PM 1.0 (µg/m³)",
    "PM 2.5 (µg/m³)": "PM 2.5 (µg/m³)",
    "PM 4.0 (µg/m³)": "PM 4.0 (µg/m³)",
    "PM 10 (µg/m³)": "PM 10 (µg/m³)",
    "CO2 (ppm)": "CO2 (ppm)",
}


def extract_room_number(filename: str) -> str:
    """Hent romnummer fra filnavn på formen data_RES08015_....csv."""
    try:
        room_code = filename.split("RES")[1].split("_")[0]
        return str(int(room_code[-1]))
    except (IndexError, ValueError) as error:
        raise ValueError(f"Kan ikke hente romnummer fra filnavnet '{filename}'.") from error


def list_room_numbers(directory: Path = INNEKLIMA_DIR, building_number: str = "7") -> List[str]:
    """List rom som finnes for ett bygg uten å lese hele CSV-innholdet."""
    prefix = INNEKLIMA_PREFIX_TEMPLATE.format(bygg=building_number)
    rooms = []

    for filename in sorted(os.listdir(directory)):
        if filename.startswith(prefix) and filename.lower().endswith(".csv"):
            rooms.append(extract_room_number(filename))

    return sorted(set(rooms), key=int)


def fetch_csv(
    directory: Path = INNEKLIMA_DIR,
    building_number: str = "7",
    filenames: Optional[List[str]] = None,
) -> Tuple[List[pd.DataFrame], List[str], int]:
    """Les inneklima-CSV-filer for ett bygg.

    Filene sorteres alfabetisk før lesing. Det gjør rekkefølgen stabil fra kjøring
    til kjøring, og er viktig for reproduserbare figurer.
    """
    data_frames: List[pd.DataFrame] = []
    room_names: List[str] = []
    registered_rooms: set[str] = set()

    prefix = INNEKLIMA_PREFIX_TEMPLATE.format(bygg=building_number)
    if filenames is None:
        filenames = [
            filename
            for filename in os.listdir(directory)
            if filename.startswith(prefix) and filename.lower().endswith(".csv")
        ]

    for filename in sorted(filenames):
        file_path = directory / filename
        if not file_path.exists():
            print(f"Advarsel: Fant ikke {file_path}. Filen hoppes over.")
            continue

        try:
            room_number = extract_room_number(filename)
            if room_number in registered_rooms:
                print(f"Info: Duplikat for rom {room_number}. {filename} hoppes over.")
                continue

            data_frames.append(pd.read_csv(file_path, sep=";"))
            room_names.append(room_number)
            registered_rooms.add(room_number)
        except Exception as error:
            print(f"Advarsel: Kunne ikke lese {filename}: {error}")

    return data_frames, room_names, len(data_frames)


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Konverter målekolonner til tall og håndter både punktum og komma."""
    converted = df.copy()
    for column in converted.columns:
        converted[column] = pd.to_numeric(
            converted[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return converted


def set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Standardiser inneklimadata før analyse.

    Behandling:
    - Date og Time kombineres til datetime-indeks.
    - Ubrukte kolonner fjernes.
    - Kolonnenavn oversettes til norske rapportnavn.
    - Måleverdier konverteres til numeriske verdier.
    - Data resamples til timesmiddel.
    """
    processed = df.copy()

    if not isinstance(processed.index, pd.DatetimeIndex):
        timestamp = pd.to_datetime(
            processed.iloc[:, 0].astype(str) + " " + processed.iloc[:, 1].astype(str),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        processed.index = timestamp

    processed = processed[~processed.index.isna()].copy()

    for column in ["Date", "Time", "Virus Index"]:
        if column in processed.columns:
            processed.drop(columns=column, inplace=True)

    processed.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
    processed = convert_numeric_columns(processed)
    processed.sort_index(inplace=True)

    # Timesmiddel gjør at rom og bygg kan sammenlignes på samme tidsoppløsning.
    return processed.resample("1h").mean().dropna(how="all")


def fetch_weather(seklima_path: Path = SEKLIMA_FILE, kunak_path: Path = KUNAK_FILE) -> pd.DataFrame:
    """Les og kombiner uteklima fra Seklima og Kunak."""
    try:
        seklima = pd.read_csv(seklima_path, sep=";")
    except FileNotFoundError as error:
        raise RuntimeError(f"Kan ikke finne Seklima-filen: {seklima_path}") from error

    seklima["tidspunkt"] = pd.to_datetime(
        seklima["Tid(norsk normaltid)"],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )
    seklima.set_index("tidspunkt", inplace=True)
    seklima["utetemp_seklima"] = pd.to_numeric(
        seklima["Middeltemperatur (1 t)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    seklima["ute_rh_seklima"] = pd.to_numeric(
        seklima["Midlere relativ luftfuktighet (1 t)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    seklima = seklima[["utetemp_seklima", "ute_rh_seklima"]]

    try:
        kunak = pd.read_csv(kunak_path, sep=";")
    except FileNotFoundError as error:
        raise RuntimeError(f"Kan ikke finne Kunak-filen: {kunak_path}") from error

    kunak["Datetime"] = pd.to_datetime(kunak["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    kunak.set_index("Datetime", inplace=True)
    kunak["utetemp_kunak"] = pd.to_numeric(
        kunak["Temp ext (C)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    kunak["ute_rh_kunak"] = pd.to_numeric(
        kunak["Humidity ext (%)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    kunak = kunak[["utetemp_kunak", "ute_rh_kunak"]]

    weather = pd.merge(seklima, kunak, how="outer", left_index=True, right_index=True)
    weather.sort_index(inplace=True)
    return weather


def filter_data(
    df_list: List[pd.DataFrame],
    mode: str = "all",
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[pd.Timestamp] = None,
) -> List[pd.DataFrame]:
    """Filtrer DataFrames til valgt periode.

    Brukergrensesnittet tilbyr ikke kortere zoom enn uke, men day støttes fortsatt
    for bakoverkompatibilitet og eventuell senere kontroll.
    """
    if not df_list:
        return []

    if mode == "all":
        return [df.copy() for df in df_list if not df.empty]

    if mode in {"year", "month", "week", "fall", "spring"} and year is None:
        raise ValueError(f"Må oppgi år for periodevalget '{mode}'.")

    filtered_list: List[pd.DataFrame] = []

    for df in df_list:
        filtered = df.copy()

        if mode == "year":
            filtered = filtered[filtered.index.year == year]
        elif mode == "month":
            filtered = filtered[(filtered.index.year == year) & (filtered.index.month == month)]
        elif mode == "week":
            iso_calendar = filtered.index.isocalendar()
            filtered = filtered[(iso_calendar.year == year) & (iso_calendar.week == week)]
        elif mode == "day" and day is not None:
            filtered = filtered[filtered.index.normalize() == pd.Timestamp(day).normalize()]
        elif mode == "spring":
            start = pd.Timestamp(year=year, month=1, day=6)
            end = pd.Timestamp(year=year, month=6, day=6) + pd.Timedelta(days=1)
            filtered = filtered[(filtered.index >= start) & (filtered.index < end)]
        elif mode == "fall":
            start = pd.Timestamp(year=year, month=8, day=10)
            end = pd.Timestamp(year=year, month=12, day=10) + pd.Timedelta(days=1)
            filtered = filtered[(filtered.index >= start) & (filtered.index < end)]
        else:
            raise ValueError(f"Ukjent periodevalg: {mode}")

        if not filtered.empty:
            filtered_list.append(filtered)

    return filtered_list


def filter_weather(
    df_weather: pd.DataFrame,
    mode: str = "all",
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Filtrer uteklima med samme perioderegel som inneklima."""
    filtered = filter_data([df_weather], mode=mode, year=year, month=month, week=week, day=day)
    return filtered[0] if filtered else pd.DataFrame()
