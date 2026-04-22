import os
import pandas as pd

from pathlib import Path
from typing import List, Tuple, Optional
from config import (SEKLIMA_FILE, KUNAK_FILE, INNEKLIMA_PREFIX_TEMPLATE, INNEKLIMA_DIR)

def fetch_csv(directory: Path = INNEKLIMA_DIR, building_number: str = "7", filenames: Optional[List[str]] = None)\
        -> Tuple[List[pd.DataFrame], List[str], int]:
    """
    Leser alle inneklima‐CSV‐filer for ett bygg, og returnerer:
      - dfs: List[pd.DataFrame], én DataFrame per rom
      - romnavn: List[str], romnummer (som string)
      - antall innleste DataFrames: int

    Args:
      directory      : Path til mappa med inneklima‐filer
      building_number: Byggkode, f.eks. "7"
      filenames      : Hvis man vil spesifisere en separat liste over filnavn

    Filnavn‐mønster: data_RES{bygg}01... .csv
      - Ex: "data_RES08015_09_56_04.csv"
      - Vi trekker ut siste siffer fra 'RES'-delen som romnummer.
    """
    dfs: List[pd.DataFrame] = []
    romnavn: List[str] = []
    registrerte_rom = set()

    prefix = INNEKLIMA_PREFIX_TEMPLATE.format(bygg=building_number)

    # Hent alle CSV‐filer som starter med prefix, eller bruk en forhåndsdefinert liste
    if filenames is None:
        alle_filer = [
            f for f in os.listdir(directory)
            if f.startswith(prefix) and f.lower().endswith(".csv")
        ]
    else:
        alle_filer = filenames.copy()

    alle_filer.sort()  # Sørg for deterministisk rekkefølge

    for f_name in alle_filer:
        file_path = directory / f_name
        if not file_path.exists():
            print(f"⚠️ Filen {file_path} finnes ikke – hopper over.")
            continue

        try:
            # Filnavn: data_RES08015_09_56_04.csv
            # Del opp for å hente romkode (f.eks. "08015"), og romnummer = siste siffer
            delen_etter_res = f_name.split("RES")[1]        # "08015_09_56_04.csv"
            romkode = delen_etter_res.split("_")[0]         # "08015"
            romnummer = str(int(romkode[-1]))               # “5” → rom “5”

            if romnummer in registrerte_rom:
                print(f"ℹ️ Duplikat funnet for rom {romnummer} – hopper over {f_name}")
                continue

            df = pd.read_csv(file_path, sep=';')
            dfs.append(df)
            romnavn.append(romnummer)
            registrerte_rom.add(romnummer)

        except Exception as e:
            print(f"⚠️ Kunne ikke lese fil {f_name}: {e}")
            continue

    return dfs, romnavn, len(dfs)


def fetch_weather(seklima_path: Path = SEKLIMA_FILE, kunak_path: Path   = KUNAK_FILE) -> pd.DataFrame:
    """
    Leser og kombinerer Seklima‐ og Kunak‐CSV til én DataFrame med kolonnene:
      ['utetemp_seklima', 'ute_rh_seklima', 'utetemp_kunak', 'ute_rh_kunak']

    Returnerer:
      - En DataFrame med index = pd.DatetimeIndex
      - Hvis ingen data finnes, kan den være tom, men definitivt ha en datetime‐index.
    """
    # --- Seklima ---
    try:
        df_seklima = pd.read_csv(seklima_path, sep=';')
    except FileNotFoundError:
        raise RuntimeError(f"Kan ikke finne Seklima‐filen: {seklima_path}")
    except Exception as e:
        raise RuntimeError(f"Feilet ved lesing av Seklima: {e}")

    # Parse dato og tid til datetime:
    df_seklima['Middeltemperatur (1 t)'] = (
        df_seklima['Middeltemperatur (1 t)'].astype(str).str.replace(",", ".")
    )
    df_seklima['tidspunkt'] = pd.to_datetime(
        df_seklima['Tid(norsk normaltid)'],
        format="%d.%m.%Y %H:%M",
        errors='coerce'
    )
    df_seklima.set_index('tidspunkt', inplace=True)

    df_seklima['utetemp_seklima'] = pd.to_numeric(
        df_seklima['Middeltemperatur (1 t)'], errors='coerce'
    )
    df_seklima['ute_rh_seklima'] = pd.to_numeric(
        df_seklima['Midlere relativ luftfuktighet (1 t)'], errors='coerce'
    )
    df_seklima = df_seklima[['utetemp_seklima', 'ute_rh_seklima']]

    # --- Kunak ---
    try:
        df_kunak = pd.read_csv(kunak_path, sep=';')
    except FileNotFoundError:
        raise RuntimeError(f"Kan ikke finne Kunak‐filen: {kunak_path}")
    except Exception as e:
        raise RuntimeError(f"Feilet ved lesing av Kunak: {e}")

    df_kunak['Datetime'] = pd.to_datetime(
        df_kunak['Datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce'
    )
    df_kunak.set_index('Datetime', inplace=True)
    df_kunak['utetemp_kunak'] = pd.to_numeric(
        df_kunak['Temp ext (C)'].astype(str).str.replace(",", "."), errors='coerce'
    )
    df_kunak['ute_rh_kunak'] = pd.to_numeric(
        df_kunak['Humidity ext (%)'].astype(str).str.replace(",", "."), errors='coerce'
    )
    df_kunak = df_kunak[['utetemp_kunak', 'ute_rh_kunak']]

    # --- Slå sammen Seklima + Kunak på tidsindeks (outer join) ---
    df_combined = pd.merge(
        df_seklima, df_kunak,
        how='outer', left_index=True, right_index=True
    )
    df_combined.sort_index(inplace=True)
    return df_combined


def filter_weather(df_weather: pd.DataFrame, mode: str = "year", year: Optional[int] = None, month: Optional[int] = None, week: Optional[int] = None,
                        day: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Filtrerer værdata til samme periode‐logikk som innendørs‐data.
    Returnerer en ny DataFrame (eller tom DataFrame hvis ingen treff).
    """
    # Hent filter‐funksjonen fra oss selv (unngå sirkulær import til plot):
    filtered_list = filter_data([df_weather], mode, year, month, week, day)
    if filtered_list:
        return filtered_list[0]
    return pd.DataFrame()


def set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tar en inneklima‐DataFrame (slik den ser ut rett etter pd.read_csv), og gjør:
      1) Kombinerer kolonne0='Date' + kolonne1='Time' til én DatetimeIndex
      2) Fjerner kolonnene ['Date', 'Time', 'Virus Index']
      3) Oversetter engelske kolonnenavn til norske (Temperatur, Luftfuktighet, …)
      4) Resampler til times‐gjennomsnitt (dropper kun rader som er 100% NaN)
      5) Returnerer den nye DataFrame med datetime‐indeks og norske kolonnenavn

    Eksempel‐input‐rader:
      Date;Time;Ventilation Indicator;IAQ Indicator;Virus Index;thermalIndicator;Temperature (°C);Humidity (%);CO2 (ppm);…
      2023-07-06;09:50:54;3.00;77.00;3.00;;18.80;62.00;;10.00;0.00;1.00;1.00;1.00;1.00
    """
    # 1) Parse dato + tid til datetime
    try:
        df.index = pd.to_datetime(
            df.iloc[:, 0] + ' ' + df.iloc[:, 1],
            format="%Y-%m-%d %H:%M:%S",
            errors='coerce'
        )
        n_invalid = df.index.isna().sum()
        if n_invalid > 0:
            print(f"⚠️  {n_invalid} rader hadde ugyldig dato/tid og ble satt til NaT")
    except Exception as e:
        print(f"❌ Feil ved parsing av datetime: {e}")

    # 2) Fjern kolonner som vi ikke bruker videre:
    for col in ['Date', 'Time', 'Virus Index']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # 3) Oversett kolonnenavn fra engelsk til norsk
    kolonneoversettelser = {
        "Temperature (°C)":        "Temperatur (°C)",
        "Humidity (%)":            "Luftfuktighet (%)",
        "Formaldehyde (µg/m³)":    "Formaldehyd (µg/m³)",
        "TVOC (ppb)":              "TVOC (ppb)",
        "PM 1.0 (µg/m³)":          "PM 1.0 (µg/m³)",
        "PM 2.5 (µg/m³)":          "PM 2.5 (µg/m³)",
        "PM 4.0 (µg/m³)":          "PM 4.0 (µg/m³)",
        "PM 10 (µg/m³)":           "PM 10 (µg/m³)",
        "CO2 (ppm)":               "CO2 (ppm)"
    }
    df.rename(columns=kolonneoversettelser, inplace=True)
    df.sort_index(inplace=True)

    # 4) Times‐resample og dropp rader som ikke har *noen* data (dropna(how="all"))
    df = df.resample("1h").mean().dropna(how="all")
    return df


def filter_data(df_list: List[pd.DataFrame], mode: str = "year", year: Optional[int] = None, month: Optional[int] = None,week: Optional[int] = None,
                    day: Optional[pd.Timestamp] = None) -> List[pd.DataFrame]:
    """
    Filtrerer en liste av DataFrames basert på 'mode':
      - 'all'    → returneres akkurat som df_list
      - 'year'   → beholder rader der df.index.year == year
      - 'month'  → beholder (år == year) & (måned == month)
      - 'week'   → beholder isocalendar.week == week  (årskryssende uke håndtert av pandas)
      - 'day'    → beholder eksakt dato (Year-Month-Day)
      - "' → okt–des i year OR jan–mars i year+1
      - 'spring' → april–september i year
    Hver filtrert DataFrame blir lagt i resultatlista, men tomme DataFrames kastes.

    Returnerer:
      - List med filtrerte DataFrames (kun de som ikke er tomme etter filteret)
    """
    if not df_list:
        return []

    if mode == "all":
        return df_list

    if mode in ["year", "fall", "spring"] and year is None:
        raise ValueError(f"Må oppgi år for modus '{mode}'")

    filtered_list: List[pd.DataFrame] = []
    for df in df_list:
        df_copy = df.copy()
        if mode == "year":
            df_copy = df_copy[df_copy.index.year == year]
        elif mode == "month":
            df_copy = df_copy[(df_copy.index.year == year) & (df_copy.index.month == month)]
        elif mode == "week":
            df_copy = df_copy[
                (df_copy.index.isocalendar().year == year) &
                (df_copy.index.isocalendar().week == week)
            ]
        elif mode == "day":
            df_copy = df_copy[
                (df_copy.index.year == day.year) &
                (df_copy.index.month == day.month) &
                (df_copy.index.day == day.day)
            ]
        elif mode == "spring":
            start = pd.Timestamp(year=year, month=1, day=6)
            end = pd.Timestamp(year=year, month=6, day=6) + pd.Timedelta(days=1)
            df_copy = df_copy[(df_copy.index >= start) & (df_copy.index < end)]

        elif mode == "fall":
            start = pd.Timestamp(year=year, month=8, day=10)
            end = pd.Timestamp(year=year, month=12, day=10) + pd.Timedelta(days=1)
            df_copy = df_copy[(df_copy.index >= start) & (df_copy.index < end)]
        else:
            raise ValueError(f"Ukjent mode: '{mode}'")

        if not df_copy.empty:
            filtered_list.append(df_copy)

    return filtered_list

