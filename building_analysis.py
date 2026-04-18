import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List

from data_processing import fetch_csv, set_datetime_index, filter_data, fetch_weather
from plotting import plot_temperature, plot_humidity, plot_air_quality_variable
from config import (LUFTKVALITETS_VARIABLER_I_REKKE, INNEKLIMA_DIR, THRESHOLDS_TEMPERATURE, THRESHOLDS_CRITICAL,
                    THRESHOLDS_OPTIMAL_HUMIDITY, TILGJENGELIGE_BYGG, PM_VARIABLER)



def be_om_år(prompt="Skriv inn år (ÅÅÅÅ) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None
    if not (inp.isdigit() and len(inp) == 4 and 2000 <= int(inp) <= 2100):
        print("❌ Ugyldig årstall. Prøv igjen.")
        return be_om_år(prompt)
    return int(inp)

def be_om_måned(prompt="Skriv inn måned (1–12) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None
    if not (inp.isdigit() and 1 <= int(inp) <= 12):
        print("❌ Ugyldig måned. Prøv igjen.")
        return be_om_måned(prompt)
    return int(inp)

def be_om_uke(prompt="Skriv inn uke (1–53) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None
    if not (inp.isdigit() and 1 <= int(inp) <= 53):
        print("❌ Ugyldig uke. Prøv igjen.")
        return be_om_uke(prompt)
    return int(inp)

def be_om_dag(prompt="Skriv inn dato (ÅÅÅÅ-MM-DD) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None

    try:
        return datetime.strptime(inp, "%Y-%m-%d")
    except ValueError:
        print("❌ Ugyldig datoformat eller ugyldig dato. Prøv igjen.")
        return be_om_dag(prompt)


def run_building_analysis():
    # 1) Vis byggliste og la bruker velge
    print("🏢 Tilgjengelige bygg for analyse:")
    for kode, navn in TILGJENGELIGE_BYGG.items():
        print(f"  Bygg {kode} – {navn}")

    user_input = input("Velg bygg eller 'b' for å gå tilbake: ").strip().lower()
    if user_input == 'b':
        return

    byggkode = user_input.zfill(2)
    if byggkode not in TILGJENGELIGE_BYGG:
        print("❌ Ugyldig byggkode.")
        return

    # 2) Hent og forbered dataene for det valgte bygget
    dfs, romnavn = fetch_and_prepare_building_data(byggkode)
    if not dfs:
        print(f"❌ Finner ingen data for Bygg {byggkode}.")
        return

    weather_df = fetch_weather()
    velg_periode_og_variabel(dfs, romnavn, weather_df, byggkode)


def fetch_and_prepare_building_data(building_id):
    dfs, romnavn, _ = fetch_csv(INNEKLIMA_DIR, building_id)  # bruk _ for antall_rom
    df_list = [set_datetime_index(df) for df in dfs]
    return df_list, romnavn


def velg_periode_og_variabel(
    df_list: List[pd.DataFrame],
    romnavn: List[str],
    weather_df: pd.DataFrame,
    byggkode: str
):
    """
    Lar brukeren først velge periode, deretter variabel.
    - Byggkode (eksempel "01", "02" osv.) sendes inn fra run_building_analysis.
    - romnavn er en liste av strenger (f.eks. ["Rom 101", "Rom 102", ...]) og brukes i legend.
    """

    while True:
        # ── 1) Velg periode ──
        print("\n⏳ VELG TIDSPERIODE")
        print("1. År")
        print("2. Høst (Aug-Des)")
        print("3. Vår (Jan-Jun)")
        print("4. Måned")
        print("5. Uke")
        print("6. Dag")
        print("b. Tilbake til hovedmeny")
        periode_valg = input("Ditt valg: ").strip().lower()
        if periode_valg == 'b':
            return  # Gå tilbake til hovedmeny


        month = week = None
        day = None

        # ── Sett mode + tilhørende år/måned/uke/dag ──
        if periode_valg == '1':  # År
            mode = 'year'
            year = be_om_år()
            if year is None:
                continue

        elif periode_valg == '2':  # Høst (fra august ut desember)
            mode = 'fall'
            year = be_om_år()
            if year is None:
                continue

        elif periode_valg == '3':  # Vår  (fra januar ut juni)
            mode = 'spring'
            year = be_om_år()
            if year is None:
                continue

        elif periode_valg == '4':  # Måned
            mode = 'month'
            year = be_om_år()
            if year is None:
                continue
            month = be_om_måned()
            if month is None:
                continue

        elif periode_valg == '5':  # Uke
            mode = 'week'
            year = be_om_år()
            if year is None:
                continue
            week = be_om_uke()
            if week is None:
                continue

        elif periode_valg == '6':  # Dag
            mode = 'day'
            day = be_om_dag()
            if day is None:
                continue
            year = day.year
            month = day.month
            week = int(day.strftime("%W"))

        else:
            print("❌ Ugyldig valg. Prøv igjen.")
            continue

        # ── Filtrer innendørsdata etter valgt periode ──
        filtrerte_data = filter_data(df_list, mode, year, month, week, day)
        if not filtrerte_data:
            print("❌ Ingen data i den valgte perioden.")
            continue

        # ── 2) Velg variabel ──
        while True:
            print("\n📊 VELG VARIABEL")
            print("1. Temperatur (°C)")
            print("2. Luftfuktighet (%)")
            for idx, var in enumerate(LUFTKVALITETS_VARIABLER_I_REKKE, start=3):
                print(f"{idx}. {var}")
            print("d. Bytt periode")
            print("b. Bytt bygg")
            print("m. Hovedmeny")

            valg_var = input("Ditt valg: ").strip().lower()
            if valg_var == 'd':
                # Gå tilbake til å velge periode
                break
            if valg_var == 'b':
                # Gå tilbake til bygg‐valg
                return
            if valg_var == 'm':
                # Gå tilbake til hovedmeny
                return

            try:
                valg_int = int(valg_var)
            except ValueError:
                print("❌ Ugyldig valg.")
                continue

            # ── 3) Kall riktig plot‐funksjon med korrekt argumentrekkefølge ──
            if valg_int == 1:
                # Temperatur
                plot_temperature(filtrerte_data, mode, year, month, week, day, weather_df, byggkode, romnavn)

            elif valg_int == 2:
                # Luftfuktighet
                plot_humidity(filtrerte_data, mode, year, month, week, day, weather_df, byggkode, romnavn)

            elif 3 <= valg_int < 3 + len(LUFTKVALITETS_VARIABLER_I_REKKE):
                # Luftkvalitetsvariabler (CO2, Formaldehyd, TVOC, PM osv.)
                var_idx = valg_int - 3
                variable = LUFTKVALITETS_VARIABLER_I_REKKE[var_idx]
                plot_air_quality_variable(filtrerte_data, variable, mode, year, month, week, day, byggkode, romnavn)

            else:
                print("❌ Ugyldig valg.")
                continue

            # Når bruker lukker graf‐vinduet, returnerer vi hit og kan velge ny variabel
            continue


def filtrer_datointervall(df: pd.DataFrame, start_dato: str, slutt_dato: str) -> pd.DataFrame:
    start = pd.to_datetime(start_dato)
    slutt = pd.to_datetime(slutt_dato)
    return df.loc[(df.index >= start) & (df.index <= slutt)]


def samle_data_per_bygg(variabel: str, start_dato: str, slutt_dato: str):
    bygg_data = []
    bygg_labels = []

    for byggkode, byggnavn in TILGJENGELIGE_BYGG.items():
        dfs: list[pd.DataFrame]
        dfs, romnavn, _ = fetch_csv(building_number=byggkode)

        alle_verdier = []

        for df in dfs:
            df = set_datetime_index(df)
            df = filtrer_datointervall(df, start_dato, slutt_dato)

            if variabel in df.columns:
                serie: pd.Series = pd.to_numeric(df[variabel], errors="coerce")
                verdier = serie.dropna()

                if not verdier.empty:
                    alle_verdier.extend(verdier.tolist())

        if alle_verdier:
            bygg_data.append(alle_verdier)
            bygg_labels.append(f"{byggkode} - {byggnavn}")

    return bygg_data, bygg_labels

def plot_boksplott_per_bygg(variabel: str, start_dato: str, slutt_dato: str):
    bygg_data, bygg_labels = samle_data_per_bygg(variabel, start_dato, slutt_dato)

    if not bygg_data:
        print(f"Ingen data funnet for {variabel} i valgt periode.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(bygg_data, tick_labels=bygg_labels, vert=False)

    tegn_terskellinjer(ax, variabel)

    ax.set_xlabel(variabel)
    ax.set_ylabel("Bygg")
    ax.set_title(f"{variabel} \n{start_dato} til {slutt_dato}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pm_boksplott_per_bygg(start_dato: str, slutt_dato: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, variabel in zip(axes, PM_VARIABLER):
        bygg_data, bygg_labels = samle_data_per_bygg(variabel, start_dato, slutt_dato)

        if not bygg_data:
            ax.set_title(f"{variabel}\nIngen data")
            ax.axis("off")
            continue

        ax.boxplot(bygg_data, tick_labels=bygg_labels, vert=False)
        tegn_terskellinjer(ax, variabel)
        ax.set_title(variabel)
        ax.set_xlabel("Konsentrasjon")
        ax.set_ylabel("Bygg")
        ax.grid(axis="x", alpha=0.3)
        ax.legend()

    plt.suptitle(f"Partikler \n{start_dato} til {slutt_dato}")
    plt.tight_layout()
    plt.show()

def run_boxplot_alle_bygg():
    print("\n📦 BOKSPLOTT ALLE BYGG")
    print("1. Studieår 2023-08-10 til 2023-12-10")
    print("2. Studieår 2024-01-10 til 2024-06-10")
    print("3. Egendefinert periode")
    print("b. Tilbake")

    periodevalg = input("Velg periode: ").strip().lower()
    if periodevalg == "b":
        return

    if periodevalg == "1":
        start_dato = "2023-08-10"
        slutt_dato = "2023-12-10"
    elif periodevalg == "2":
        start_dato = "2024-01-10"
        slutt_dato = "2024-06-10"
    elif periodevalg == "3":
        start_dato = input("Startdato (ÅÅÅÅ-MM-DD): ").strip()
        slutt_dato = input("Sluttdato (ÅÅÅÅ-MM-DD): ").strip()
    else:
        print("❌ Ugyldig valg.")
        return

    while True:
        print("\n📊 VELG VARIABEL")
        print("1. Temperatur (°C)")
        print("2. Luftfuktighet (%)")
        print("3. CO2 (ppm)")
        print("4. Formaldehyd (µg/m³)")
        print("5. TVOC (ppb)")
        print("6. PM (alle fire i én figur)")
        print("b. Tilbake")

        valg = input("Valg: ").strip().lower()
        if valg == "b":
            return

        variabler = {
            "1": "Temperatur (°C)",
            "2": "Luftfuktighet (%)",
            "3": "CO2 (ppm)",
            "4": "Formaldehyd (µg/m³)",
            "5": "TVOC (ppb)"
        }

        if valg == "6":
            plot_pm_boksplott_per_bygg(start_dato, slutt_dato)
        elif valg in variabler:
            plot_boksplott_per_bygg(variabler[valg], start_dato, slutt_dato)
        else:
            print("❌ Ugyldig valg.")
            continue

def tegn_terskellinjer(ax, variabel: str):
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
        natt_max = THRESHOLDS_TEMPERATURE["night"]["max"]

        ax.axvline(dag_min, linestyle="--", linewidth=2, label=f"Dag/natt: {dag_min}")
        ax.axvline(natt_min, linestyle=":", linewidth=2, label=f"Minimum: {natt_min}")
        ax.axvline(dag_max, linestyle="--", linewidth=2, label=f"Maksimum: {dag_max}")
        ax.axvline(natt_max, linestyle=":", linewidth=2, )