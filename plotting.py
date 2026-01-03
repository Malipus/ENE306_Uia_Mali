# plotting.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import cm
from matplotlib.legend import Legend
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter

from typing import List, Optional, Union
from pathlib import Path

from data_processing import (
    INNEKLIMA_DIR,
    fetch_csv,
    fetch_weather,
    set_datetime_index,
    filter_data,
    filter_weather,
)

from config import (
    THRESHOLDS_TEMPERATURE,
    THRESHOLDS_OPTIMAL_HUMIDITY,
    THRESHOLDS_WARN,
    THRESHOLDS_CRITICAL,
    LUFTKVALITETS_VARIABLER_I_REKKE,
)

# ──────────────────────────────────────────────────────────────────────────────
#  1) plot_temperature
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import cm
from matplotlib.legend import Legend
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter

from typing import List, Optional, Dict
from pathlib import Path

from data_processing import (
    INNEKLIMA_DIR,
    fetch_csv,
    set_datetime_index,
    filter_data,
    fetch_weather  # <-- sørg for at dette er importert
)
from config import (
    THRESHOLDS_TEMPERATURE,
    THRESHOLDS_OPTIMAL_HUMIDITY,
    THRESHOLDS_WARN,
    THRESHOLDS_CRITICAL,
    LUFTKVALITETS_VARIABLER_I_REKKE,
)

def plot_temperature(
    df_list: List[pd.DataFrame],
    mode: str = "year",
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[pd.Timestamp] = None,
    df_weather: Optional[pd.DataFrame] = None,
    byggkode: str = "",
    romnavn: List[str] = []
) -> None:
    """
    Plotter temperatur for én eller flere rom i df_list, inkludert utendørsdata
    (hvis df_weather er gitt). Rom‐legend bygges ved å samle linjeobjekter
    for hvert rom direkte i room_lines. Tittel inkluderer kun byggkode.
    """

    # 1) Filtrer innendørsdata
    df_filtered_list = filter_data(df_list, mode, year, month, week, day)
    if not df_filtered_list:
        print("❌ Ingen innendørs temperaturdata i den valgte perioden.")
        return

    # 2) Finn terskler for temperatur
    temp_col = "Temperatur (°C)"
    thresholds = [
        (THRESHOLDS_TEMPERATURE["day"]["min"],   "black",  "Dag/natt‐grense"),
        (THRESHOLDS_TEMPERATURE["day"]["max"],   "orange", "Maks dagtemperatur"),
        (THRESHOLDS_TEMPERATURE["night"]["min"], "purple", "Min nattetemperatur")
    ]

    # 3) Filtrer utendørsdata dersom tilgjengelig
    show_weather = False
    df_weather_filt = pd.DataFrame()
    if df_weather is not None:
        df_weather_filt = filter_weather(df_weather, mode, year, month, week, day)
        if not df_weather_filt.empty:
            show_weather = True

    # 4) Lag figur/aks(er)
    if show_weather:
        fig, (ax, ax_weather) = plt.subplots(
            nrows=2, ncols=1, sharex=True,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15}
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax_weather = None

    romfarger = cm.get_cmap("tab10")
    room_lines = []
    room_labels = []

    # 5) Plot innendørs temperatur for hvert rom
    for idx, df2 in enumerate(df_filtered_list, start=1):
        if temp_col not in df2.columns or df2.empty:
            continue

        # a) Hull over 1 time → NaN
        gap_threshold = pd.Timedelta(hours=1)
        df_sort = df2.sort_index().copy()
        df_sort["Time Diff"] = df_sort.index.to_series().diff()
        df_sort.loc[df_sort["Time Diff"] > gap_threshold, temp_col] = pd.NA
        df_sort.drop(columns=["Time Diff"], inplace=True)

        # b) Reindeks til timeserie
        start = df_sort.index.min()
        slutt = df_sort.index.max()
        full_idx = pd.date_range(start=start, end=slutt, freq="1h")
        series = df_sort[temp_col].reindex(full_idx)

        # c) Sett label til romnavn eller fallback “Rom {idx}”
        navn = romnavn[idx-1] if idx-1 < len(romnavn) else f"Rom {idx}"
        line, = ax.plot(
            series.index,
            series.values,
            label=navn,
            color=romfarger((idx - 1) % 10),
            linewidth=2
        )
        room_lines.append(line)
        room_labels.append(navn)

        if mode == "day":
            ax.scatter(
                series.index,
                series.values,
                color=romfarger((idx - 1) % 10),
                s=10,
                zorder=5
            )

    # 6) Tegn terskellinjer
    used_threshold_labels = []
    threshold_handles = []
    threshold_labels = []
    for nivå, farge, etikett in thresholds:
        nivå_vis = nivå
        if etikett not in used_threshold_labels:
            h = ax.axhline(
                y=nivå_vis,
                color=farge,
                linestyle="--",
                linewidth=2,
                label=etikett
            )
            threshold_handles.append(h)
            threshold_labels.append(etikett)
            used_threshold_labels.append(etikett)
        else:
            ax.axhline(y=nivå_vis, color=farge, linestyle="--", linewidth=2)

    ax.set_ylabel("Temperatur (°C)")
    ax.grid(True)

    # 7) Plot utendørsdata hvis tilgjengelig
    if ax_weather is not None and not df_weather_filt.empty:
        weather_columns = [
            ("utetemp_seklima", "Seklima", "blue"),
            ("utetemp_kunak",   "Kunak",   "orange")
        ]
        for kol, label, color in weather_columns:
            if kol in df_weather_filt.columns:
                data = df_weather_filt[kol].dropna()
                if data.empty:
                    continue
                ax_weather.plot(
                    data.index,
                    data.values,
                    label=label,
                    color=color,
                    linewidth=1.5
                )
        ax_weather.set_ylabel("Utetemperatur (°C)")
        ax_weather.grid(True)
        ax_weather.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Uteklima",
            frameon=True
        )

    # 8) Bestem x‐aksens visningsintervall
    if mode == "day" and isinstance(day, pd.Timestamp):
        start_vis = day.replace(hour=0, minute=0, second=0)
        slutt_vis = start_vis + pd.Timedelta(days=1)
    elif mode == "week" and isinstance(year, int) and isinstance(week, int):
        ref = pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%Y-W%W-%w")
        start_vis = ref.replace(hour=0, minute=0, second=0)
        slutt_vis = start_vis + pd.Timedelta(days=7)
    elif mode == "month" and isinstance(year, int) and isinstance(month, int):
        start_vis = pd.to_datetime(f"{year}-{month:02d}-01")
        slutt_vis = start_vis + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    elif mode == "year" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-01-01")
        slutt_vis = pd.to_datetime(f"{year}-12-31") + pd.Timedelta(days=1)
    elif mode == "summer" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-04-01")
        slutt_vis = pd.to_datetime(f"{year}-09-30") + pd.Timedelta(days=1)
    elif mode == "winter" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-10-01")
        slutt_vis = pd.to_datetime(f"{year+1}-03-31") + pd.Timedelta(days=1)
    else:
        alle_min = min(df.index.min() for df in df_filtered_list if not df.empty)
        alle_max = max(df.index.max() for df in df_filtered_list if not df.empty)
        start_vis, slutt_vis = alle_min, alle_max + pd.Timedelta(hours=1)

    ax.set_xlim(start_vis, slutt_vis)
    if ax_weather is not None:
        ax_weather.set_xlim(start_vis, slutt_vis)

    # 9) Formater x‐akse
    if mode in ["year", "summer", "winter", "all"]:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(DateFormatter("%b"))
        ax.tick_params(axis="x", which="minor", labelsize=8, rotation=0, pad=10)
        for lbl in ax.get_xticklabels(minor=True):
            txt = lbl.get_text()
            lbl.set_text(txt[0] if txt else "")
        ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.4)
    elif mode == "month":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%d."))
    elif mode == "week":
        start_dag = start_vis.normalize()
        dag_ticks = [start_dag + pd.Timedelta(days=i) for i in range(7)] + [slutt_vis]
        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(d) for d in dag_ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter(["man","tir","ons","tor","fre","lør","søn",""]))
    elif mode == "day":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    # 10) Sett tittel kun med byggkode og periode
    periode_str = ""
    if mode == "year" and isinstance(year, int):
        periode_str = f"{year}"
    elif mode == "month" and isinstance(year, int) and isinstance(month, int):
        måned_navn = pd.to_datetime(f"{month}", format="%m").strftime("%B")
        periode_str = f"{måned_navn} {year}"
    elif mode == "week" and isinstance(year, int) and isinstance(week, int):
        start_date = pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%Y-W%W-%w")
        end_date = start_date + pd.Timedelta(days=6)
        periode_str = f"Uke {week} {year} ({start_date.strftime('%d. %b')}–{end_date.strftime('%d. %b')})"
    elif mode == "day" and isinstance(day, pd.Timestamp):
        dato_str = day.strftime("%-d. %b %Y") if os.name != "nt" else day.strftime("%#d. %b %Y")
        periode_str = f"{dato_str}"
    elif mode == "summer" and isinstance(year, int):
        periode_str = f"Sommer {year}"
    elif mode == "winter" and isinstance(year, int):
        periode_str = f"Vinter {year}-{year+1}"

    ax.set_title(f"Bygg {byggkode} – Temperatur for {periode_str}")

    # 11) Legend for rom
    if room_lines:
        room_legend = ax.legend(
            room_lines,
            room_labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Rom",
            frameon=True
        )
        ax.add_artist(room_legend)

    # 12) Legend for tersklene
    if threshold_handles:
        ax.legend(
            threshold_handles,
            threshold_labels,
            loc="lower left",
            bbox_to_anchor=(1, 0),
            title="Grenser",
            frameon=True
        )

    # 13) Juster marger manuelt
    if ax_weather is not None:
        fig.subplots_adjust(left=0.07, right=0.80, top=0.92, bottom=0.10, hspace=0.15)
    else:
        fig.subplots_adjust(left=0.07, right=0.80, top=0.92, bottom=0.10)

    plt.show()


def plot_humidity(
    df_list: List[pd.DataFrame],
    mode: str = "year",
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[pd.Timestamp] = None,
    df_weather: Optional[pd.DataFrame] = None,
    byggkode: str = "",
    romnavn: List[str] = []
) -> None:
    """
    Plotter luftfuktighet for én eller flere rom i df_list, inkludert utendørsdata.
    Rom‐legend bygges ved å samle linjeobjekter for hvert rom direkte i room_lines.
    Tittel inkluderer kun byggkode.
    """

    # 1) Filtrer innendørsdata
    df_filtered_list = filter_data(df_list, mode, year, month, week, day)
    if not df_filtered_list:
        print("❌ Ingen innendørs fuktighetsdata i den valgte perioden.")
        return

    # 2) Finn terskler for fuktighet
    hum_col = "Luftfuktighet (%)"
    grenser = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
    thresholds = [
        (grenser["optimal_min"], "green",  "Optimal fuktighet"),
        (grenser["optimal_max"], "green",  "Optimal fuktighet"),
        (grenser["critical_min"], "red",   "Kritisk fuktighet"),
        (grenser["critical_max"], "red",   "Kritisk fuktighet")
    ]

    # 3) Filtrer utendørsdata
    show_weather = False
    df_weather_filt = pd.DataFrame()
    if df_weather is not None:
        df_weather_filt = filter_weather(df_weather, mode, year, month, week, day)
        if not df_weather_filt.empty:
            show_weather = True

    # 4) Lag figur/aks(er)
    if show_weather:
        fig, (ax, ax_weather) = plt.subplots(
            nrows=2, ncols=1, sharex=True,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15}
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax_weather = None

    romfarger = cm.get_cmap("tab10")
    room_lines = []
    room_labels = []

    # 5) Plot innendørs fuktighet
    for idx, df2 in enumerate(df_filtered_list, start=1):
        if hum_col not in df2.columns or df2.empty:
            continue

        gap_threshold = pd.Timedelta(hours=1)
        df_sort = df2.sort_index().copy()
        df_sort["Time Diff"] = df_sort.index.to_series().diff()
        df_sort.loc[df_sort["Time Diff"] > gap_threshold, hum_col] = pd.NA
        df_sort.drop(columns=["Time Diff"], inplace=True)

        start = df_sort.index.min()
        slutt = df_sort.index.max()
        full_idx = pd.date_range(start=start, end=slutt, freq="1h")
        series = df_sort[hum_col].reindex(full_idx)

        navn = romnavn[idx-1] if idx-1 < len(romnavn) else f"Rom {idx}"
        line, = ax.plot(
            series.index,
            series.values,
            label=navn,
            color=romfarger((idx - 1) % 10),
            linewidth=2
        )
        room_lines.append(line)
        room_labels.append(navn)

        if mode == "day":
            ax.scatter(
                series.index,
                series.values,
                color=romfarger((idx - 1) % 10),
                s=10,
                zorder=5
            )

    # 6) Tegn terskellinjer
    used_threshold_labels = []
    threshold_handles = []
    threshold_labels = []
    for nivå, farge, etikett in thresholds:
        nivå_vis = nivå
        if etikett not in used_threshold_labels:
            h = ax.axhline(
                y=nivå_vis,
                color=farge,
                linestyle="--",
                linewidth=2,
                label=etikett
            )
            threshold_handles.append(h)
            threshold_labels.append(etikett)
            used_threshold_labels.append(etikett)
        else:
            ax.axhline(y=nivå_vis, color=farge, linestyle="--", linewidth=2)

    ax.set_ylabel("Luftfuktighet (%)")
    ax.grid(True)

    # 7) Plot utendørsdata hvis tilgjengelig
    if ax_weather is not None and not df_weather_filt.empty:
        weather_columns = [
            ("ute_rh_seklima", "Seklima", "blue"),
            ("ute_rh_kunak",   "Kunak",   "orange")
        ]
        for kol, label, color in weather_columns:
            if kol in df_weather_filt.columns:
                data = df_weather_filt[kol].dropna()
                if data.empty:
                    continue
                ax_weather.plot(
                    data.index,
                    data.values,
                    label=label,
                    color=color,
                    linewidth=1.5
                )
        ax_weather.set_ylabel("Uteluftfuktighet (%)")
        ax_weather.grid(True)
        ax_weather.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Uteklima",
            frameon=True
        )

    # 8) Bestem x‐aksens visningsintervall
    if mode == "day" and isinstance(day, pd.Timestamp):
        start_vis = day.replace(hour=0, minute=0, second=0)
        slutt_vis = start_vis + pd.Timedelta(days=1)
    elif mode == "week" and isinstance(year, int) and isinstance(week, int):
        ref = pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%Y-W%W-%w")
        start_vis = ref.replace(hour=0, minute=0, second=0)
        slutt_vis = start_vis + pd.Timedelta(days=7)
    elif mode == "month" and isinstance(year, int) and isinstance(month, int):
        start_vis = pd.to_datetime(f"{year}-{month:02d}-01")
        slutt_vis = start_vis + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    elif mode == "year" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-01-01")
        slutt_vis = pd.to_datetime(f"{year}-12-31") + pd.Timedelta(days=1)
    elif mode == "summer" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-04-01")
        slutt_vis = pd.to_datetime(f"{year}-09-30") + pd.Timedelta(days=1)
    elif mode == "winter" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-10-01")
        slutt_vis = pd.to_datetime(f"{year+1}-03-31") + pd.Timedelta(days=1)
    else:
        alle_min = min(df.index.min() for df in df_filtered_list if not df.empty)
        alle_max = max(df.index.max() for df in df_filtered_list if not df.empty)
        start_vis, slutt_vis = alle_min, alle_max + pd.Timedelta(hours=1)

    ax.set_xlim(start_vis, slutt_vis)
    if ax_weather is not None:
        ax_weather.set_xlim(start_vis, slutt_vis)

    # 9) Formater x‐akse
    if mode in ["year", "summer", "winter", "all"]:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(DateFormatter("%b"))
        ax.tick_params(axis="x", which="minor", labelsize=8, rotation=0, pad=10)
        for lbl in ax.get_xticklabels(minor=True):
            txt = lbl.get_text()
            lbl.set_text(txt[0] if txt else "")
        ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.4)
    elif mode == "month":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%d."))
    elif mode == "week":
        start_dag = start_vis.normalize()
        dag_ticks = [start_dag + pd.Timedelta(days=i) for i in range(7)] + [slutt_vis]
        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(d) for d in dag_ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter(["man","tir","ons","tor","fre","lør","søn",""]))
    elif mode == "day":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    # 10) Sett tittel kun med byggkode og periode
    periode_str = ""
    if mode == "year" and isinstance(year, int):
        periode_str = f"{year}"
    elif mode == "month" and isinstance(year, int) and isinstance(month, int):
        måned_navn = pd.to_datetime(f"{month}", format="%m").strftime("%B")
        periode_str = f"{måned_navn} {year}"
    elif mode == "week" and isinstance(year, int) and isinstance(week, int):
        start_date = pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%Y-W%W-%w")
        end_date = start_date + pd.Timedelta(days=6)
        periode_str = f"Uke {week} {year} ({start_date.strftime('%d. %b')}–{end_date.strftime('%d. %b')})"
    elif mode == "day" and isinstance(day, pd.Timestamp):
        dato_str = day.strftime("%-d. %b %Y") if os.name != "nt" else day.strftime("%#d. %b %Y")
        periode_str = f"{dato_str}"
    elif mode == "summer" and isinstance(year, int):
        periode_str = f"Sommer {year}"
    elif mode == "winter" and isinstance(year, int):
        periode_str = f"Vinter {year}-{year+1}"

    ax.set_title(f"Bygg {byggkode} – Luftfuktighet for {periode_str}")

    # 11) Legend for rom
    if room_lines:
        room_legend = ax.legend(
            room_lines,
            room_labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Rom",
            frameon=True
        )
        ax.add_artist(room_legend)

    # 12) Legend for tersklene
    if threshold_handles:
        ax.legend(
            threshold_handles,
            threshold_labels,
            loc="lower left",
            bbox_to_anchor=(1, 0),
            title="Grenser",
            frameon=True
        )

    # 13) Juster marger manuelt
    if ax_weather is not None:
        fig.subplots_adjust(left=0.07, right=0.80, top=0.92, bottom=0.10, hspace=0.15)
    else:
        fig.subplots_adjust(left=0.07, right=0.80, top=0.92, bottom=0.10)

    plt.show()

def plot_air_quality_variable(
    df_list: List[pd.DataFrame],
    variable: str,
    mode: str = "year",
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[pd.Timestamp] = None,
    df_weather: Optional[pd.DataFrame] = None,
    byggkode: Union[str, int, List] = "",
    romnavn: List[str] = []
) -> None:
    """
    Plotter luftkvalitetsvariabler (CO2, TVOC, PM osv.) som tidsserie for alle rom.
    Henter varsel- og kritisk grense fra THRESHOLDS_WARN/THRESHOLDS_CRITICAL, og legger dem
    som horisontale linjer. Y-aksen begrenses til 2× høyeste terskelverdi.
    Tittelen viser kun én byggkode (uten ledende 0 eller lister).
    """

    # 1) Filtrer innendørsdata i valgt periode
    df_filtered_list = filter_data(df_list, mode, year, month, week, day)
    if not df_filtered_list:
        print(f"❌ Ingen data for '{variable}' i den valgte perioden.")
        return

    # 2) Hent terskelverdier fra config
    warn_value     = THRESHOLDS_WARN.get(variable)
    critical_value = THRESHOLDS_CRITICAL.get(variable)
    if warn_value is None or critical_value is None:
        print(f"❌ Ingen terskelverdier definert for '{variable}'.")
        return

    # Finn høyeste terskelverdi og sett øvre grense for y-akse (2×)
    maks_terskel = max(warn_value, critical_value)
    y_upper = maks_terskel * 2

    # Bygg liste med terskellinjer (først varsel, så kritisk)
    thresholds = [
        (warn_value,     "orange", "Varselgrense"),
        (critical_value, "red",    "Kritisk grense")
    ]

    # 3) Opprett figur/akse
    fig, ax = plt.subplots(figsize=(12, 6))
    romfarger = cm.get_cmap("tab10")

    room_lines = []
    room_labels = []

    # 4) Plot hver romserie (kolonnen 'variable') som tidsserie
    for idx, df2 in enumerate(df_filtered_list, start=1):
        if variable not in df2.columns or df2.empty:
            continue

        # a) Fjern hull > 1 time → NaN
        gap_thr = pd.Timedelta(hours=1)
        df_sort = df2.sort_index().copy()
        df_sort["Time Diff"] = df_sort.index.to_series().diff()
        df_sort.loc[df_sort["Time Diff"] > gap_thr, variable] = pd.NA
        df_sort.drop(columns=["Time Diff"], inplace=True)

        # b) Reindekser til timeserie
        start = df_sort.index.min()
        slutt = df_sort.index.max()
        full_idx = pd.date_range(start=start, end=slutt, freq="1h")
        series = df_sort[variable].reindex(full_idx)

        # c) Finn romnavn (hvis gitt), ellers fallback "Rom {idx}"
        navn = romnavn[idx-1] if idx-1 < len(romnavn) else f"Rom {idx}"
        line, = ax.plot(
            series.index,
            series.values,
            label=navn,
            color=romfarger((idx - 1) % 10),
            linewidth=2
        )
        room_lines.append(line)
        room_labels.append(navn)

        if mode == "day":
            ax.scatter(
                series.index,
                series.values,
                color=romfarger((idx - 1) % 10),
                s=10,
                zorder=5
            )

    ax.set_ylabel(variable)
    ax.grid(True)

    # 5) Tegn terskellinjer
    threshold_handles = []
    threshold_labels = []
    brukt = set()
    for nivå, farge, etikett in thresholds:
        if etikett not in brukt:
            h = ax.axhline(
                y=nivå,
                color=farge,
                linestyle="--",
                linewidth=1.5,
                label=etikett
            )
            threshold_handles.append(h)
            threshold_labels.append(etikett)
            brukt.add(etikett)
        else:
            ax.axhline(y=nivå, color=farge, linestyle="--", linewidth=1.5)

    # 6) Begrens y-akse til 0–2×maks terskel
    ax.set_ylim(0, y_upper)

    # 7) Bestem x‐aksens visningsintervall
    if mode == "day" and isinstance(day, pd.Timestamp):
        start_vis = day.replace(hour=0, minute=0, second=0)
        slutt_vis = start_vis + pd.Timedelta(days=1)
    elif mode == "week" and isinstance(year, int) and isinstance(week, int):
        ref = pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%Y-W%W-%w")
        start_vis = ref.replace(hour=0, minute=0, second=0)
        slutt_vis = start_vis + pd.Timedelta(days=7)
    elif mode == "month" and isinstance(year, int) and isinstance(month, int):
        start_vis = pd.to_datetime(f"{year}-{month:02d}-01")
        slutt_vis = start_vis + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    elif mode == "year" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-01-01")
        slutt_vis = pd.to_datetime(f"{year}-12-31") + pd.Timedelta(days=1)
    elif mode == "summer" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-04-01")
        slutt_vis = pd.to_datetime(f"{year}-09-30") + pd.Timedelta(days=1)
    elif mode == "winter" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-10-01")
        slutt_vis = pd.to_datetime(f"{year+1}-03-31") + pd.Timedelta(days=1)
    else:
        alle_min = min(df.index.min() for df in df_filtered_list if not df.empty)
        alle_max = max(df.index.max() for df in df_filtered_list if not df.empty)
        start_vis, slutt_vis = alle_min, alle_max + pd.Timedelta(hours=1)

    ax.set_xlim(start_vis, slutt_vis)

    # 8) Formater x‐akse etter periode
    if mode in ["year", "summer", "winter", "all"]:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(DateFormatter("%b"))
        ax.tick_params(axis="x", which="minor", labelsize=8, rotation=0, pad=10)
        for lbl in ax.get_xticklabels(minor=True):
            txt = lbl.get_text()
            lbl.set_text(txt[0] if txt else "")
        ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.4)
    elif mode == "month":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%d."))
    elif mode == "week":
        start_dag = start_vis.normalize()
        dag_ticks = [start_dag + pd.Timedelta(days=i) for i in range(7)] + [slutt_vis]
        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(d) for d in dag_ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter(["man","tir","ons","tor","fre","lør","søn",""]))
    elif mode == "day":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    # 9) Sett tittel med kun ett byggnummer
    if isinstance(byggkode, (list, tuple)) and len(byggkode) > 0:
        # Dersom byggkode ble sendt inn som liste, trekk ut første element
        første = byggkode[0]
        bygge_nummer = int(første) if str(første).isdigit() else første
    else:
        # Fjern ledende nuller hvis byggkode er streng med tall
        if isinstance(byggkode, str) and byggkode.isdigit():
            bygge_nummer = int(byggkode)
        else:
            bygge_nummer = byggkode

    if mode == "year" and isinstance(year, int):
        ax.set_title(f"Bygg {bygge_nummer} – {variable} for {year}")
    elif mode == "winter" and isinstance(year, int):
        ax.set_title(f"Bygg {bygge_nummer} – {variable} for Vinter {year}-{year+1}")
    elif mode == "summer" and isinstance(year, int):
        ax.set_title(f"Bygg {bygge_nummer} – {variable} for Sommer {year}")
    elif mode == "month" and isinstance(year, int) and isinstance(month, int):
        måned_navn = pd.to_datetime(f"{month}", format="%m").strftime("%B")
        ax.set_title(f"Bygg {bygge_nummer} – {variable} for {måned_navn} {year}")
    elif mode == "week" and isinstance(year, int) and isinstance(week, int):
        start_date = pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%Y-W%W-%w")
        end_date = start_date + pd.Timedelta(days=6)
        ax.set_title(
            f"Bygg {bygge_nummer} – {variable} for Uke {week} {year} "
            f"({start_date.strftime('%d. %b')}–{end_date.strftime('%d. %b')})"
        )
    elif mode == "day" and isinstance(day, pd.Timestamp):
        dato_str = day.strftime("%-d. %b %Y") if os.name != "nt" else day.strftime("%#d. %b %Y")
        ax.set_title(f"Bygg {bygge_nummer} – {variable} for {dato_str}")

    # 10) Legend for rom‐seriene
    if room_lines:
        rom_legend = ax.legend(
            room_lines,
            room_labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Rom",
            frameon=True
        )
        ax.add_artist(rom_legend)

    # 11) Legend for terskellinjene
    if threshold_handles:
        ax.legend(
            threshold_handles,
            threshold_labels,
            loc="lower left",
            bbox_to_anchor=(1, 0),
            title="Grenser",
            frameon=True
        )

    # 12) Juster marger manuelt (unngå tight_layout-advarsel)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.92, bottom=0.10)

    plt.show()



# ──────────────────────────────────────────────────────────────────────────────
#  4) plot_all_rooms_variable
# ──────────────────────────────────────────────────────────────────────────────
def plot_all_rooms_variable(df_list: List[pd.DataFrame], variable: str) -> None:
    """
    Plotter én variabel for alle rom over hele perioden (mode="all").
    - Tids‐gaps (>1 time uten måling) gir hull i linjene.
    - Ingen rom‐legend; kun terskellinjers legend.
    - Terskler per variabel.
    - y_max = 3 * kritisk nivå for alle andre enn temperatur og fuktighet.
    - Temperatur: y_min=10, y_max=35.
    - Luftfuktighet: autolimit (ingen fast y_bunn).
    - CO2/Formaldehyd/PM: y_min=0, y_max=3×kritisk nivå.
    - X‐akse: Major ticks = år, Minor ticks = månedsbokstav.
    """

    terskler = []
    y_label = variable
    y_min = None
    y_max = None

    if variable == "Temperatur (°C)":
        dag_maks  = THRESHOLDS_TEMPERATURE["day"]["max"]
        dag_min   = THRESHOLDS_TEMPERATURE["day"]["min"]
        natt_maks = THRESHOLDS_TEMPERATURE["night"]["max"]
        natt_min  = THRESHOLDS_TEMPERATURE["night"]["min"]

        y_label = "Temperatur (°C)"
        y_min, y_max = 10, 35

        terskler = [
            (dag_maks,  "red",    "Maksimal"),
            ((dag_min + natt_maks) / 2, "orange", "Dag/Natt "),
            (natt_min,  "blue",   "Minimal")
        ]

    elif variable == "Luftfuktighet (%)":
        grenser = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        optimal_min  = grenser["optimal_min"]
        optimal_max  = grenser["optimal_max"]
        kritisk_min  = grenser["critical_min"]
        kritisk_max  = grenser["critical_max"]

        y_label = "Luftfuktighet (%)"
        y_min, y_max = None, None

        terskler = [
            (optimal_max, "green", "Optimal fuktighet"),
            (optimal_min, "green", None),
            (kritisk_max, "red", "Kritisk fuktighet"),
            (kritisk_min, "red", None)
        ]

    else:
        warn_value     = THRESHOLDS_WARN.get(variable)
        critical_value = THRESHOLDS_CRITICAL.get(variable)
        if warn_value is None or critical_value is None:
            print(f"❌ Ingen terskelverdier definert for '{variable}'.")
            return

        y_label = variable
        y_min = 0
        y_max = 2 * critical_value

        terskler = [
            (critical_value, "red",   "Kritisk nivå"),
            (warn_value,     "green", "Optimal nivå")
        ]

    fig, ax = plt.subplots(figsize=(12, 6))
    romfarger = plt.cm.get_cmap("tab20")

    # Plot rom‐serier med hull for gap >1 time
    for idx, df2 in enumerate(df_list, start=1):
        if df2.empty:
            continue

        kolonne = "Value" if "Value" in df2.columns else variable
        if kolonne not in df2.columns:
            continue

        df_sort = df2.sort_index().copy()
        if not isinstance(df_sort.index, pd.DatetimeIndex):
            df_sort = set_datetime_index(df_sort)

        df_sort["tdiff"] = df_sort.index.to_series().diff()
        df_sort.loc[df_sort["tdiff"] > pd.Timedelta(hours=1), kolonne] = pd.NA
        df_sort.drop(columns=["tdiff"], inplace=True)

        start = df_sort.index.min()
        slutt = df_sort.index.max()
        full_idx = pd.date_range(start=start, end=slutt, freq="1h")
        series = df_sort[kolonne].reindex(full_idx)

        ax.plot(
            series.index,
            series.values,
            color=romfarger((idx - 1) % 20),
            linewidth=1
        )

    # Terskellinjer i synkende rekkefølge
    terskler_sorted = sorted(terskler, key=lambda x: x[0], reverse=True)
    brukt_label = set()
    for verdi, farge, etikett in terskler_sorted:
        if etikett is None:
            ax.axhline(y=verdi, color=farge, linestyle="--", linewidth=1.5)
        else:
            if etikett not in brukt_label:
                ax.axhline(y=verdi, color=farge, linestyle="--", linewidth=1.5, label=etikett)
                brukt_label.add(etikett)
            else:
                ax.axhline(y=verdi, color=farge, linestyle="--", linewidth=1.5)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_title(f"{variable} – Tidsserie for alle rom (hele perioden)", fontsize=14, pad=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("")

    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_formatter(DateFormatter("%b"))

    ax.tick_params(axis="x", which="major", labelsize=10, rotation=0, pad=15)
    ax.tick_params(axis="x", which="minor", labelsize=8, rotation=0, pad=10)
    for lbl in ax.get_xticklabels(minor=True):
        txt = lbl.get_text()
        lbl.set_text(txt[0] if txt else "")

    ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.4)

    for child in ax.get_children():
        if isinstance(child, Legend):
            labels = [h.get_label() for h in child.legendHandles]
            if any("Rom" in lbl for lbl in labels):
                child.remove()

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Terskler", frameon=True)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  5) plot_all_pm_subplots
# ──────────────────────────────────────────────────────────────────────────────
def plot_all_pm_subplots(pm_data: Dict[str, List[pd.DataFrame]]) -> None:
    """
    Lager én figur med 4 subplots (2x2), én for hver PM‐variabel:
      - PM 1.0 (µg/m³)
      - PM 2.5 (µg/m³)
      - PM 4.0 (µg/m³)
      - PM 10  (µg/m³)
    """
    pm_vars = [
        "PM 1.0 (µg/m³)",
        "PM 2.5 (µg/m³)",
        "PM 4.0 (µg/m³)",
        "PM 10 (µg/m³)"
    ]
    y_min, y_max = 0, 150

    terskler_pm = {}
    for var in pm_vars:
        warn_val     = THRESHOLDS_WARN.get(var)
        critical_val = THRESHOLDS_CRITICAL.get(var)
        if warn_val is None or critical_val is None:
            terskler_pm[var] = []
        else:
            terskler_pm[var] = [
                (critical_val, "red",   "Kritisk nivå"),
                (warn_val,     "green", "Optimal nivå")
            ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    romfarger = plt.cm.get_cmap("tab20")

    for ax, var in zip(axes_flat, pm_vars):
        df_list = pm_data.get(var, [])
        if not df_list:
            ax.text(0.5, 0.5, f"Ingen data for\n{var}", ha="center", va="center", fontsize=12)
            ax.set_title(var)
            ax.set_ylim(y_min, y_max)
            continue

        for idx, df2 in enumerate(df_list, start=1):
            if df2.empty:
                continue

            kolonne = "Value" if "Value" in df2.columns else var
            if kolonne not in df2.columns:
                continue

            df_sort = df2.sort_index().copy()
            if not isinstance(df_sort.index, pd.DatetimeIndex):
                df_sort = set_datetime_index(df_sort)

            df_sort["tdiff"] = df_sort.index.to_series().diff()
            gap_threshold = pd.Timedelta(hours=1)
            df_sort.loc[df_sort["tdiff"] > gap_threshold, kolonne] = pd.NA
            df_sort.drop(columns=["tdiff"], inplace=True)

            start = df_sort.index.min()
            slutt = df_sort.index.max()
            full_idx = pd.date_range(start=start, end=slutt, freq="1h")
            series = df_sort[kolonne].reindex(full_idx)

            ax.plot(
                series.index,
                series.values,
                color=romfarger((idx - 1) % 20),
                linewidth=0.8
            )

        ters = terskler_pm.get(var, [])
        ters_sorted = sorted(ters, key=lambda x: x[0], reverse=True)
        brukt_label = set()
        for verdi, farge, etikett in ters_sorted:
            if etikett not in brukt_label:
                ax.axhline(y=verdi, color=farge, linestyle="--", linewidth=1.2, label=etikett)
                brukt_label.add(etikett)
            else:
                ax.axhline(y=verdi, color=farge, linestyle="--", linewidth=1.2)

        ax.set_title(var, fontsize=12, pad=6)
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_ylabel("µg/m³")

        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_minor_formatter(DateFormatter("%b"))
        ax.tick_params(axis="x", which="major", labelsize=10, rotation=0, pad=8)
        ax.tick_params(axis="x", which="minor", labelsize=8, rotation=0, pad=4)
        for lbl in ax.get_xticklabels(minor=True):
            txt = lbl.get_text()
            lbl.set_text(txt[0] if txt else "")

        ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.3)

        for child in ax.get_children():
            if isinstance(child, Legend):
                labels = [h.get_label() for h in child.legendHandles]
                if any("Rom" in lbl for lbl in labels):
                    child.remove()

    handles_all, labels_all = [], []
    for var in pm_vars:
        for verdi, farge, etikett in terskler_pm.get(var, []):
            if etikett not in labels_all:
                h = plt.Line2D([], [], color=farge, linestyle="--", linewidth=1.2)
                handles_all.append(h)
                labels_all.append(etikett)

    fig.legend(
        handles_all,
        labels_all,
        title="Terskler",
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        frameon=True
    )

    fig.supxlabel("Dato (år og månedsbokstav)", fontsize=12, y=0.02)
    fig.supylabel("PM (µg/m³)", x=0.02, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.show()

# ────────────── Dekningsgrad per rom (tekst-tabell) ──────────────
def dekningsgrad_per_rom(df_list: List[pd.DataFrame], romnavn_list: List[str]) -> pd.DataFrame:
    """
    Beregner dekningsgrad for hver romserie i df_list.
    Returnerer en DataFrame med kolonnene:
      ['Room','Start date','End date','Days','Hours with data','Coverage (%)']
    """
    resultater = []

    for i, df in enumerate(df_list):
        romnavn = romnavn_list[i]
        if df.empty or df.index.inferred_type != "datetime64":
            resultater.append((romnavn, "No data", "-", "-", "-", "-"))
            continue

        start = df.index.min().floor("h")
        slutt = df.index.max().ceil("h")
        totalt_timer = int((slutt - start).total_seconds() / 3600)

        df["hour"] = df.index.floor("h")
        antall_timer_med_data = df["hour"].nunique()

        dekning = min(100, (antall_timer_med_data / totalt_timer) * 100) if totalt_timer else 0

        resultater.append((
            romnavn,
            start.strftime("%Y-%m-%d"),
            slutt.strftime("%Y-%m-%d"),
            (slutt - start).days + 1,
            f"{antall_timer_med_data}/{totalt_timer}",
            f"{dekning:.1f}"
        ))

    return pd.DataFrame(
        resultater,
        columns=["Room", "Start date", "End date", "Days", "Hours with data", "Coverage (%)"]
    )


# ────────────── Datadeknings‐Gantt for alle rom ──────────────
def vis_datadekning_per_rom(
    byggliste: List[str] = ["01", "02", "04", "05", "07", "08"],
    mappe: Path = INNEKLIMA_DIR
) -> None:
    """
    Tegner en Gantt-lignende oversikt over hvilke dager hvert rom har data på tvers av bygg.
    Henter CSV for hvert bygg/rom via fetch_csv og set_datetime_index.
    """
    oversikt = []

    for bygg in byggliste:
        romdata, romnavn, antall_rom = fetch_csv(directory=mappe, building_number=bygg)
        romdata = [set_datetime_index(df) for df in romdata]

        for navn, df in zip(romnavn, romdata):
            if df.empty or df.index.inferred_type != "datetime64":
                continue
            # (Her kan du fjerne filtrering på rom > 5 hvis du vil vise alle rom.)
            df["Dato"] = df.index.date
            unike_dager = sorted(df["Dato"].unique())
            for dag in unike_dager:
                start = pd.to_datetime(dag)
                slutt = start + pd.Timedelta(days=1)
                oversikt.append({
                    "Bygg": f"B{bygg}",
                    "Rom":  f"R{navn}",
                    "Start": start,
                    "Slutt": slutt
                })

    df = pd.DataFrame(oversikt)
    if df.empty:
        print("❌ Ingen data å vise.")
        return

    df["Etikett"] = df["Bygg"] + "-" + df["Rom"]
    df.sort_values(by=["Etikett", "Start"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    bygg_farger = {
        bygg: farge for bygg, farge in zip(
            sorted(df["Bygg"].unique()),
            cm.get_cmap("tab10").colors
        )
    }

    fig, ax = plt.subplots(
        figsize=(12, max(5, len(df["Etikett"].unique()) * 0.5))
    )
    etiketter_sortert = sorted(df["Etikett"].unique(), reverse=True)
    etikett_idx = { etikett: idx for idx, etikett in enumerate(etiketter_sortert) }

    # Tegn årsskifte-linjer i bakgrunnen (lav zorder):
    min_dato = df["Start"].min()
    max_dato = df["Slutt"].max()
    år_start = min_dato.year
    år_slutt = max_dato.year
    for år in range(år_start + 1, år_slutt + 1):
        overgang = pd.to_datetime(f"{år}-01-01")
        ax.axvline(
            overgang,
            color="0.85",
            linewidth=0.8,
            zorder=0
        )

    # Tegn datalinjer for hver dag/rom
    for _, row in df.iterrows():
        y = etikett_idx[row["Etikett"]]
        ax.plot(
            [row["Start"], row["Slutt"]],
            [y, y],
            linewidth=3,
            color=bygg_farger[row["Bygg"]],
            zorder=1
        )

    ax.set_yticks(list(etikett_idx.values()))
    ax.set_yticklabels(list(etikett_idx.keys()))

    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_minor_formatter(DateFormatter("%b"))
    ax.tick_params(axis="x", which="minor", labelsize=8, rotation=0, pad=10)
    for lbl in ax.get_xticklabels(minor=True):
        txt = lbl.get_text()
        lbl.set_text(txt[0] if txt else "")
    ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.4)

    for bygg, farge in bygg_farger.items():
        ax.plot([], [], label=bygg, color=farge, linewidth=6)
    ax.legend(title="Bygg", loc="upper left", bbox_to_anchor=(1, 1))

    ax.set_title("Tilgjengelighet av sensordata – bygg og romnummer")
    plt.tight_layout()
    plt.show()


def vis_dekningsgrad_alle_bygg() -> None:
    """
    Loop gjennom alle tilgjengelige bygg, beregn dekningsgrad per rom, og skriv resultatet ut.
    """

    # Hent TILGJENGELIGE_BYGG inn her (slik unngår vi top‐level sirkulær import)
    from building_analysis import TILGJENGELIGE_BYGG

    for byggkode in TILGJENGELIGE_BYGG:
        dfs, romnavn, _ = fetch_csv(building_number=byggkode)
        if not dfs:
            continue

        # Sørg for at alle data har datetime‐index før dekningsgrad‐beregnes
        for df in dfs:
            set_datetime_index(df)

        df_deg = dekningsgrad_per_rom(dfs, romnavn)

        # Oversett kolonnenavn til norsk
        df_deg.rename(columns={
            'Room':           'Rom',
            'Start date':     'Startdato',
            'End date':       'Sluttdato',
            'Hours with data': 'Timer med data',
            'Coverage (%)':   'Dekning (%)'
        }, inplace=True)

        print(f"\n-- Dekningsgrad for Bygg {byggkode} – {TILGJENGELIGE_BYGG[byggkode]} --")
        print(df_deg.to_string(index=False))

