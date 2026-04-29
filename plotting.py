# plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter, MultipleLocator

from typing import List, Optional
from pathlib import Path

from data_processing import (INNEKLIMA_DIR, fetch_csv, set_datetime_index, filter_data, filter_weather)
from config import (THRESHOLDS_TEMPERATURE, THRESHOLDS_OPTIMAL_HUMIDITY, THRESHOLDS_WARN, THRESHOLDS_CRITICAL,
                    NORWEGIAN_MONTHS)


def format_period_label(
    mode: str = "all",
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[pd.Timestamp] = None,
) -> str:
    if mode == "all":
        return "hele måleperioden"
    if mode == "year" and isinstance(year, int):
        return f"år {year}"
    if mode == "month" and isinstance(year, int) and isinstance(month, int):
        return f"{NORWEGIAN_MONTHS[month]} {year}"
    if mode == "week" and isinstance(year, int) and isinstance(week, int):
        return f"uke {week} i {year}"
    if mode == "day" and isinstance(day, pd.Timestamp):
        return day.strftime("%d.%m.%Y")
    if mode == "fall" and isinstance(year, int):
        return f"høstperioden {year}"
    if mode == "spring" and isinstance(year, int):
        return f"vårperioden {year}"
    return "valgt periode"

def build_plot_title(subject: str, metric: str, period_label: str) -> str:
    return f"{subject} – {metric} i {period_label}"

def plot_temperature(df_list: List[pd.DataFrame],
                     mode: str = "year", year: Optional[int] = None,
    month: Optional[int] = None, week: Optional[int] = None, day: Optional[pd.Timestamp] = None,
    df_weather: Optional[pd.DataFrame] = None, byggkode: str = "", romnavn: Optional[List[str]] = None,
    title_subject: Optional[str] = None) -> None:


    if romnavn is None:
        romnavn = []

    """
    Plotter temperatur for én eller flere rom i df_list, inkludert utendorsdata.
    """

    # 1) Filtrer innendorsdata
    df_filtered_list = filter_data(df_list, mode, year, month, week, day)
    if not df_filtered_list:
        print("❌ Ingen innendors temperaturdata i den valgte perioden.")
        return

    # 2) Finn terskler for temperatur
    temp_col = "Temperatur (°C)"
    thresholds = [
        (THRESHOLDS_TEMPERATURE["day"]["min"],   "black",  "Dag/natt‐grense"),
        (THRESHOLDS_TEMPERATURE["day"]["max"],   "orange", "Maks dagtemperatur"),
        (THRESHOLDS_TEMPERATURE["night"]["min"], "purple", "Min natt temperatur")
    ]

    # 3) Filtrer utendorsdata dersom tilgjengelig
    show_weather = False
    df_weather_filt = pd.DataFrame()
    if df_weather is not None:
        df_weather_filt = filter_weather(df_weather, mode, year, month, week, day)
        if not df_weather_filt.empty:
            show_weather = True

    # 4) Lag figur/akser
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

    # 5) Plot innendors temperatur for hvert rom
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

        # c) Sett label til romnavn eller “Rom {idx}”
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
    for nivaa, farge, etikett in thresholds:
        nivaa_vis = nivaa
        if etikett not in used_threshold_labels:
            h = ax.axhline(
                y=nivaa_vis,
                color=farge,
                linestyle="--",
                linewidth=2,
                label=etikett
            )
            threshold_handles.append(h)
            threshold_labels.append(etikett)
            used_threshold_labels.append(etikett)
        else:
            ax.axhline(y=nivaa_vis, color=farge, linestyle="--", linewidth=2)

    ax.set_ylabel("Temperatur (°C)")
    ax.grid(True)

    # 7) Plot utendorsdata hvis tilgjengelig
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
    elif mode == "fall" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-08-10")
        slutt_vis = pd.to_datetime(f"{year}-12-10") + pd.Timedelta(days=1)
    elif mode == "spring" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-01-06")
        slutt_vis = pd.to_datetime(f"{year}-06-06") + pd.Timedelta(days=1)
    else:
        alle_min = min(df.index.min() for df in df_filtered_list if not df.empty)
        alle_max = max(df.index.max() for df in df_filtered_list if not df.empty)
        start_vis, slutt_vis = alle_min, alle_max + pd.Timedelta(hours=1)

    ax.set_xlim(start_vis, slutt_vis)
    if ax_weather is not None:
        ax_weather.set_xlim(start_vis, slutt_vis)

    # 9) Formater x‐akse
    if mode in ["year", "fall", "spring", "all"]:
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
        ax.xaxis.set_major_formatter(FixedFormatter(["man","tir","ons","tor","fre","lor","son",""]))
    elif mode == "day":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    # 10) Sett tittel kun med byggkode og periode
    period_label = format_period_label(mode, year, month, week, day)

    if title_subject is None:
        title_subject = f"Bygg {byggkode}" if byggkode else (""
                                                             ""
                                                            )

    ax.set_title(build_plot_title(title_subject, "temperatur", period_label))


    # 11) Legend for tersklene
    if threshold_handles:
        ax.legend(
            threshold_handles,
            threshold_labels,
            loc="lower left",
            bbox_to_anchor=(1, 0),
            title="Grenser",
            frameon=True
        )

    # 12) Juster marger manuelt
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
    romnavn: Optional[List[str]] = None,
    title_subject: Optional[str] = None
) -> None:


    if romnavn is None:
        romnavn = []
    """
    Plotter luftfuktighet for én eller flere rom i df_list, inkludert utendorsdata.
    Rom‐legend bygges ved aa samle linjeobjekter for hvert rom direkte i room_lines.
    Tittel inkluderer kun byggkode.
    """

    # 1) Filtrer innendorsdata
    df_filtered_list = filter_data(df_list, mode, year, month, week, day)
    if not df_filtered_list:
        print("❌ Ingen innendors fuktighetsdata i den valgte perioden.")
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

    # 3) Filtrer utendorsdata
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

    # 5) Plot innendors fuktighet
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
    for nivaa, farge, etikett in thresholds:
        nivaa_vis = nivaa
        if etikett not in used_threshold_labels:
            h = ax.axhline(
                y=nivaa_vis,
                color=farge,
                linestyle="--",
                linewidth=2,
                label=etikett
            )
            threshold_handles.append(h)
            threshold_labels.append(etikett)
            used_threshold_labels.append(etikett)
        else:
            ax.axhline(y=nivaa_vis, color=farge, linestyle="--", linewidth=2)

    ax.set_ylabel("Relative luftfuktighet (%)")
    ax.grid(True)

    # 7) Plot utendorsdata hvis tilgjengelig
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
    elif mode == "fall" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-08-10")
        slutt_vis = pd.to_datetime(f"{year}-12-10") + pd.Timedelta(days=1)
    elif mode == "spring" and isinstance(year, int):
        start_vis = pd.to_datetime(f"{year}-01-06")
        slutt_vis = pd.to_datetime(f"{year}-06-06") + pd.Timedelta(days=1)
    else:
        alle_min = min(df.index.min() for df in df_filtered_list if not df.empty)
        alle_max = max(df.index.max() for df in df_filtered_list if not df.empty)
        start_vis, slutt_vis = alle_min, alle_max + pd.Timedelta(hours=1)

    ax.set_xlim(start_vis, slutt_vis)
    if ax_weather is not None:
        ax_weather.set_xlim(start_vis, slutt_vis)

    # 9) Formater x‐akse
    if mode in ["year", "fall", "spring", "all"]:
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
        ax.xaxis.set_major_formatter(FixedFormatter(["man","tir","ons","tor","fre","lor","son",""]))
    elif mode == "day":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    # 10) Sett tittel kun med byggkode og periode
    period_label = format_period_label(mode, year, month, week, day)

    if title_subject is None:
        title_subject = f"Bygg {byggkode}" if byggkode else "Valgte rom"

    ax.set_title(build_plot_title(title_subject, "Relative luftfuktighet (%)", period_label))


    # 11) Legend for tersklene
    if threshold_handles:
        ax.legend(
            threshold_handles,
            threshold_labels,
            loc="lower left",
            bbox_to_anchor=(1, 0),
            title="Grenser",
            frameon=True
        )

    # 12) Juster marger manuelt
    if ax_weather is not None:
        fig.subplots_adjust(left=0.07, right=0.80, top=0.92, bottom=0.10, hspace=0.15)
    else:
        fig.subplots_adjust(left=0.07, right=0.80, top=0.92, bottom=0.10)

    plt.show()



def plot_all_rooms_variable(
    df_list: List[pd.DataFrame],
    variable: str,
    mode: str = "all",
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[pd.Timestamp] = None,
    title_subject: str = "Valgte rom",
) -> None:


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

    # Terskellinjer i synkende rekkefolge
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

    period_label = format_period_label(mode, year, month, week, day)
    ax.set_title(build_plot_title(title_subject, variable, period_label), fontsize=14, pad=12)

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


    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Terskler", frameon=True)
    plt.tight_layout()
    plt.show()


def plot_threshold_scatter(
    scatter_df: pd.DataFrame,
    variable: str,
    threshold: float,
    direction: str,
    scope_label: str,
    title: str,
    start_vis: Optional[pd.Timestamp] = None,
    slutt_vis: Optional[pd.Timestamp] = None,
) -> None:
    if scatter_df.empty:
        retningstekst = "over" if direction == "above" else "under"
        print(f"❌ Ingen målinger {retningstekst} {threshold:g} for {variable} i valgt datasett.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = scatter_df.copy()

    if plot_df["Bygg"].nunique() == 1:
        group_col = "Rom"
        legend_title = "Rom"
        plot_df[group_col] = plot_df[group_col].fillna("Ukjent rom")
    else:
        group_col = "Bygg"
        legend_title = "Bygg"

    grupper = sorted(plot_df[group_col].dropna().unique())
    cmap = cm.get_cmap("tab10", max(1, len(grupper)))

    for idx, gruppe in enumerate(grupper):
        subset = plot_df[plot_df[group_col] == gruppe]
        ax.scatter(
            subset["Tid"],
            subset["Verdi"],
            label=gruppe,
            color=cmap(idx),
            s=18,
            alpha=0.75
        )

    threshold_label = "Øvre grense" if direction == "above" else "Nedre grense"
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"{threshold_label}: {threshold:g}"
    )

    if start_vis is not None and slutt_vis is not None:
        ax.set_xlim(start_vis, slutt_vis)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_ylabel(variable)
    ax.set_xlabel("")
    ax.set_title(f"{title}\n{scope_label}")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title=legend_title,
        frameon=True
    )

    fig.subplots_adjust(left=0.07, right=0.80, top=0.90, bottom=0.12)
    plt.show()


def plot_room_data_availability(state, mappe: Path = INNEKLIMA_DIR):

    byggliste = state["buildings"]
    oversikt = []

    for bygg in byggliste:
        romdata, romnavn, antall_rom = fetch_csv(directory=mappe, building_number=bygg)
        romdata = [set_datetime_index(df) for df in romdata]

        for navn, df in zip(romnavn, romdata):
            valgte_rom = state.get("rooms_by_building", {}).get(str(bygg))
            if valgte_rom is not None and str(navn) not in valgte_rom:
                continue

        for navn, df in zip(romnavn, romdata):
            filtrert_liste = filter_data(
                [df],
                mode=state["mode"],
                year=state["year"],
                month=state["month"],
                week=state["week"],
                day=state["day"]
            )

            if not filtrert_liste:
                continue

            df_filtrert = filtrert_liste[0]

            if df_filtrert.empty or df_filtrert.index.inferred_type != "datetime64":
                continue

            df_filtrert["Dato"] = df_filtrert.index.date
            unike_dager = sorted(df_filtrert["Dato"].unique())

            for dag in unike_dager:
                start = pd.to_datetime(dag)
                slutt = start + pd.Timedelta(days=1)
                oversikt.append({
                    "Bygg": f"{bygg}",
                    "Rom": f"{navn}",
                    "Start": start,
                    "Slutt": slutt
                })

    df = pd.DataFrame(oversikt)
    if df.empty:
        print("❌ Ingen data å vise for valgt datasett.")
        return

    df["Etikett"] = "B" + df["Bygg"] + "-R" + df["Rom"]
    df.sort_values(by=["Etikett", "Start"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cmap = cm.get_cmap("tab10", len(sorted(df["Bygg"].unique())))

    bygg_farger = {
        bygg: cmap(i)
        for i, bygg in enumerate(sorted(df["Bygg"].unique()))
    }

    fig, ax = plt.subplots(
        figsize=(12, max(5, len(df["Etikett"].unique()) * 0.5))
    )

    etiketter_sortert = sorted(df["Etikett"].unique(), reverse=True)
    etikett_idx = {etikett: idx for idx, etikett in enumerate(etiketter_sortert)}

    min_dato = df["Start"].min()
    max_dato = df["Slutt"].max()
    year_start = min_dato.year
    year_slutt = max_dato.year

    for år in range(year_start + 1, year_slutt + 1):
        overgang = pd.to_datetime(f"{år}-01-01")
        ax.axvline(
            overgang,
            color="0.85",
            linewidth=0.8,
            zorder=0
        )

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
    period_label = format_period_label(
        state["mode"],
        state["year"],
        state["month"],
        state["week"],
        state["day"],
    )

    subject = (
        f"Bygg {state['buildings'][0]}"
        if len(state["buildings"]) == 1
        else "Valgte bygg"
    )

    ax.set_title(build_plot_title(subject, "datadekning per rom", period_label))

    plt.tight_layout()
    plt.show()

def plot_building_boxplot(variable: str, building_data, building_labels, scope_label):
    if not building_data:
        print(f"Ingen data funnet for {variable} i valgt datasett.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    building_data = building_data[::-1]
    building_labels = building_labels[::-1]
    ax.boxplot(building_data, tick_labels=building_labels, vert=False)

    draw_thresholds(ax, variable)

    ax.set_xlabel(variable)

    if variable in ("Temperatur (°C)", "Luftfuktighet (%)"):
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis="x", which="minor", length=4)
        ax.grid(axis="x", which="major", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

    ax.set_title(f"{variable} – {scope_label}")

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_pm_boxplots(pm_plot_data, scope_label):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (variable, building_data, building_labels) in zip(axes, pm_plot_data):
        if not building_data:
            plt.suptitle(f"Luftpartikkelnivå – {scope_label}")

            ax.axis("off")
            continue

        building_data = building_data[::-1]
        building_labels = building_labels[::-1]
        ax.boxplot(building_data, tick_labels=building_labels, vert=False)

        draw_thresholds(ax, variable)
        ax.set_title(variable)
        ax.set_xlabel("Konsentrasjon")

        ax.legend()

    plt.suptitle(f"Partikler\n{scope_label}")
    plt.tight_layout()
    plt.show()


def draw_thresholds(ax, variabel: str):
    # Luftkvalitetsvariabler
    if variabel in THRESHOLDS_CRITICAL:
        terskel = THRESHOLDS_CRITICAL[variabel]
        ax.axvline(terskel, linestyle="--", linewidth=2, label=f"Kritisk grense: {terskel}")

    # Luftfuktighet
    elif variabel == "Luftfuktighet (%)":
        crit_min = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]["critical_min"]
        crit_max = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]["critical_max"]

        ax.axvline(crit_min, linestyle="--", linewidth=2, label=f"Kritisk min: {crit_min}")
        ax.axvline(crit_max, linestyle="--", linewidth=2, label=f"Kritisk maks: {crit_max}")

    # Temperatur
    elif variabel == "Temperatur (°C)":
        dag_min = THRESHOLDS_TEMPERATURE["day"]["min"]
        dag_max = THRESHOLDS_TEMPERATURE["day"]["max"]
        natt_min = THRESHOLDS_TEMPERATURE["night"]["min"]

        ax.axvline(dag_min, linestyle=":", linewidth=2, label=f"Dag/natt: {dag_min}")
        ax.axvline(natt_min, linestyle="--", linewidth=2, label=f"Minimum: {natt_min}")
        ax.axvline(dag_max, linestyle="--", linewidth=2, label=f"Maksimum: {dag_max}")