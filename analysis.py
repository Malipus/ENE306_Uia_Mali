import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_processing import fetch_csv, set_datetime_index
from plotting import plot_all_rooms_variable
from config import (THRESHOLDS_TEMPERATURE, THRESHOLDS_OPTIMAL_HUMIDITY,
                    THRESHOLDS_WARN, THRESHOLDS_CRITICAL, TILGJENGELIGE_BYGG, VARIABLE_CHOICES)



def _collect_all_room_data(variable: str):
    """
    Henter hver rom‐DataFrame for alle bygg i TILGJENGELIGE_BYGG, setter datetime‐index,
    filtrerer bort rom som ikke har kolonnen, og returnerer liste av DataFrames.
    """
    all_dfs = []
    all_labels = []

    for bygg in TILGJENGELIGE_BYGG:
        try:
            dfs_bygg, romnavn, _ = fetch_csv(building_number=bygg)
        except Exception as e:
            print(f"⚠️ Klarte ikke hente data for bygg {bygg}: {e}")
            continue

        for df, rom in zip(dfs_bygg, romnavn):
            df2 = set_datetime_index(df)
            if variable not in df2.columns:
                continue

            # Lag ny kolonne “Value” med bare én variabel
            df3 = df2[[variable]].copy()
            df3.rename(columns={variable: "Value"}, inplace=True)

            all_dfs.append(df3)
            all_labels.append(f"B{bygg}-R{rom}")

    return all_dfs, all_labels

def run_time_series():
    """
    Hovedmeny for å plotte tidsserie for én valgt variabel på tvers av alle bygg/rom.
    Etter ett plott kan brukeren taste ny variabelkode for nytt plott,
    eller 'b' for å gå tilbake til forrige meny.
    """
    while True:
        # 1) Velg variabel
        print("\n📊 VELG VARIABEL FOR TIDSSERIEPLOTT")
        for key, var in VARIABLE_CHOICES.items():
            print(f"{key}. {var}")
        print("b. Tilbake til forrige meny")

        valg = input("Valg: ").strip().lower()
        if valg == 'b':
            return  # Gå tilbake til hovedmeny
        if valg not in VARIABLE_CHOICES:
            print("❌ Ugyldig valg. Prøv igjen.")
            continue

        variable = VARIABLE_CHOICES[valg]
        print(f"👉 Du har valgt: {variable}")

        # 2) Hent alle rom‐data for den valgte variabelen:
        dfs, rom_labels = _collect_all_room_data(variable)
        if not dfs:
            print(f"❌ Ingen data funnet for variabelen '{variable}'.")
            # Returner til toppen av loop slik at brukeren kan taste ny kode
            continue

        # 3) Plot tidsserien over hele perioden
        plot_all_rooms_variable(dfs, variable)

        continue


def run_distribution():
    """
    Hovedmeny for å plotte fordeling (histogram) for én variabel på tvers av alle rom/bygg.
    Løkker slik at brukeren kan taste nytt valg umiddelbart etter plottet lukkes.
    """
    while True:
        # 1) Velg variabel
        print("\n📊 VELG VARIABEL FOR FORDELINGSPLOTT")
        for key, var in VARIABLE_CHOICES.items():
            print(f"{key}. {var}")
        print("b. Tilbake til forrige meny")

        valg = input("Valg: ").strip().lower()
        if valg == 'b':
            return  # Ut av run_distribution tilbake til forrige meny
        if valg not in VARIABLE_CHOICES:
            print("❌ Ugyldig valg. Prøv igjen.")
            continue

        variable = VARIABLE_CHOICES[valg]
        print(f"\n👉 Du har valgt: {variable}")

        # 2) Hent alle rom‐data og samle alle verdier i én Series
        serie = _collect_all_values_for_variable(variable)
        if serie is None or serie.empty:
            print(f"❌ Ingen data funnet for variabelen '{variable}'.")
            # Fortsett for å spørre om ny variabel
            continue

        # 3) Plot histogram for valgte variabel
        _plot_distribution(serie, variable)
        # Når brukeren lukker plottvinduet, kommer vi hit og går tilbake til starten av løkka


def _collect_all_values_for_variable(variable: str) -> pd.Series:
    """
    Henter alle data for én variabel, på tvers av alle rom/bygg,
    legger dem i én stor Pandas‐Series, og returnerer denne.
    """
    samling = []
    for bygg in TILGJENGELIGE_BYGG:
        try:
            dfs_bygg, romnavn, _ = fetch_csv(building_number=bygg)
        except Exception as e:
            print(f"Kunne ikke hente data for bygg {bygg}: {e}")
            continue

        for df, rom in zip(dfs_bygg, romnavn):
            df2 = set_datetime_index(df)
            if variable not in df2.columns:
                continue
            # Fjern rader uten data
            serie = df2[variable].dropna()
            if serie.empty:
                continue
            samling.append(serie)

    if not samling:
        return pd.Series(dtype='float64')

    # Slå sammen i én Series (multi‐index blir ignorert; vi beholder bare verdier)
    samlet = pd.concat(samling, axis=0)
    return samlet


def _plot_time_series_all_rooms(dfs: list, labels: list, variable: str):
    """
    Plott tidsserie for alle romene (liste av times‐resamplede DataFrames).
    Setter terskel‐linjer automatisk basert på variable, tilpasser akser og tittel.
    """
    # 1) Finn terskel‐verdier og farger for den valgte variable
    y_label = variable
    title_prefix = f"{variable}"

    if variable == "Temperatur (°C)":
        # Dag/natt, maks dag, min natt
        day_min = THRESHOLDS_TEMPERATURE["day"]["min"]
        day_max = THRESHOLDS_TEMPERATURE["day"]["max"]
        night_min = THRESHOLDS_TEMPERATURE["night"]["min"]
        thresholds = [
            (day_min,   "black",  "Grense dag/natt"),
            (day_max,   "orange", "Maks dagtemperatur"),
            (night_min, "purple", "Min nattetemperatur")
        ]

    elif variable == "Luftfuktighet (%)":
        # Optimal + kritiske grenser
        grenser = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        thresholds = [
            (grenser["optimal_min"], "green",   "Optimal fuktighet (min)"),
            (grenser["optimal_max"], "green",   "Optimal fuktighet (max)"),
            (grenser["critical_min"], "red",    "Kritisk fuktighet (min)"),
            (grenser["critical_max"], "red",    "Kritisk fuktighet (max)")
        ]

    else:
        # Luftkvalitets‐variable (CO2, TVOC, PM osv.) bruker varsels + kritiske terskler
        warn_value = THRESHOLDS_WARN.get(variable)
        crit_value = THRESHOLDS_CRITICAL.get(variable)
        if warn_value is not None and crit_value is not None:
            thresholds = [
                (warn_value, "orange", "Varselgrense"),
                (crit_value, "red",    "Kritisk grense")
            ]
        else:
            thresholds = []

    # 2) Bygg en felles plott‐figur
    fig, ax = plt.subplots(figsize=(12, 6))

    # 3) Tegn én kurve per rom
    for df2, label in zip(dfs, labels):
        # df2 har bare én kolonne kalt "Value"
        ax.plot(df2.index, df2["Value"], label=label, linewidth=1)

    # 4) Tegn terskel‐linjer horisontalt
    for lvl, farge, etikett in thresholds:
        ax.axhline(y=lvl, color=farge, linestyle='--', linewidth=1.5, label=etikett)

    # 5) Legg på tittel
    ax.set_title(f"{title_prefix} – Tidsserie for alle rom")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Dato/tid")

    # 6) Legg til legend UTENFOR aksen til høyre
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Rom og terskler")

    # 7) Stram opp layout slik det er plass til legend‐boksen
    fig.subplots_adjust(right=0.75)

    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def _plot_distribution(serie: pd.Series, variable: str):
    """
    Plott fordeling av én variabel på tvers av alle rom/bygg, med faste bin‐bredder
    og manuelle x‐aksebegrensninger (ZOOM ved hjelp av set_xlim).
    Terskelverdiene hentes fra config og vises som vertikale streker.
    Y‐aksen starter på 0.

    - Temperatur: 10–35 °C, 1 °C per søyle (heltall på x‐aksen)
    - Luftfuktighet: 15–75 %, 2 % per søyle
    - CO₂: 400–1200 ppm, 30 ppm per søyle (med 5 ppm venstremarg)
    - Formaldehyd: 0–125 µg/m³, 20 µg/m³ per søyle
    - TVOC: 0–1500 ppb, 30 ppb per søyle
    - PM 1.0: 0–40 µg/m³, 5 µg/m³ per søyle
    - PM 2.5: 0–40 µg/m³, 5 µg/m³ per søyle
    - PM 4:  0–40 µg/m³, 25 µg/m³ per søyle
    - PM 10: 0–110 µg/m³, 5 µg/m³ per søyle
    """

    # 1) Hent terskelverdier
    terskelverdier = []
    if variable == "Temperatur (°C)":
        day_min   = THRESHOLDS_TEMPERATURE["day"]["min"]
        day_max   = THRESHOLDS_TEMPERATURE["day"]["max"]
        night_min = THRESHOLDS_TEMPERATURE["night"]["min"]
        terskelverdier.extend([
            (day_min,   "black",  "Dag/natt‐grense"),
            (day_max,   "orange", "Maks dagtemperatur"),
            (night_min, "purple", "Min nattetemperatur")
        ])

    elif variable == "Luftfuktighet (%)":
        grenser = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        terskelverdier.extend([
            (grenser["optimal_min"], "green", "Optimal fuktighet"),
            (grenser["optimal_max"], "green", "Optimal fuktighet"),
            (grenser["critical_min"], "red",   "Kritisk fuktighet"),
            (grenser["critical_max"], "red",   "Kritisk fuktighet")
        ])

    else:
        warn_value = THRESHOLDS_WARN.get(variable)
        crit_value = THRESHOLDS_CRITICAL.get(variable)
        if warn_value is not None:
            terskelverdier.append((warn_value, "orange", "Varselgrense"))
        if crit_value is not None:
            terskelverdier.append((crit_value, "red", "Kritisk grense"))

    # 2) Hent data og fjern NaN
    data = serie.dropna().values
    if data.size == 0:
        print(f"❌ Ingen gyldige målinger for '{variable}'.")
        return

    # 3) Sett x_min, x_max og bins for hver variabel
    #    Vi bruker substring-sjekk for de PM-variantene med mellomrom i navnet.
    if variable == "Temperatur (°C)":
        x_min, x_max = 10, 35
        bins = np.arange(x_min, x_max + 1, 1)  # 1 °C per søyle

    elif variable == "Luftfuktighet (%)":
        x_min, x_max = 15, 75
        bins = np.arange(x_min, x_max + 2, 2)  # 2 % per søyle

    elif variable == "CO2 (ppm)":
        # Fast 400–1200 ppm; 30 ppm per søyle, med 5 ppm marg til venstre
        raw_bins = np.arange(400, 1200 + 30, 30)  # 400, 430, …, 1200
        bins = raw_bins
        x_min, x_max = 395, 1200  # litt luft til venstre (395)

    elif "Formaldehyd" in variable:  # f.eks. "Formaldehyd (µg/m³)"
        x_min, x_max = 0, 125
        bins = np.arange(x_min, x_max + 20, 20)  # 20 µg/m³ per søyle

    elif "TVOC" in variable:  # f.eks. "TVOC (ppb)"
        x_min, x_max = 0, 1500
        bins = np.arange(x_min, x_max + 30, 30)  # 30 ppb per søyle

    # NB! Støtt også "PM 1.0 (µg/m³)" med mellomrom
    elif "PM 1.0" in variable:
        x_min, x_max = 0, 40
        bins = np.arange(x_min, x_max + 5, 5)  # 5 µg/m³ per søyle

    elif "PM 2.5" in variable:
        x_min, x_max = 0, 40
        bins = np.arange(x_min, x_max + 5, 5)  # 5 µg/m³ per søyle

    elif "PM 4" in variable:
        x_min, x_max = 0, 40
        bins = np.arange(x_min, x_max + 25, 25)  # 25 µg/m³ per søyle

    elif "PM 10" in variable:
        x_min, x_max = 0, 110
        bins = np.arange(x_min, x_max + 5, 5)  # 5 µg/m³ per søyle

    else:
        print(f"❌ Ingen forhåndsdefinert oppsett for '{variable}'.")
        return

    # 4) Tegn histogrammet for *alle* data (uten range=...).
    #    Dermed kaster vi aldri datapunkter, men zoomer kun visningen etterpå.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        data,
        bins=bins,
        color="skyblue",
        edgecolor="black"
    )
    ax.set_ylabel("Antall observasjoner")
    ax.set_xlabel(variable)
    ax.set_ylim(bottom=0)

    # 5) Zoom x-aksen til ønsket intervall (bruke set_xlim)
    ax.set_xlim([x_min, x_max])

    # 6) Sørg for at terskelverdier vises som xticks
    eksisterende_ticks = list(ax.get_xticks())
    for lvl, _, _ in terskelverdier:
        if lvl < x_min:
            eksisterende_ticks.append(x_min)
        elif lvl > x_max:
            eksisterende_ticks.append(x_max)
        else:
            eksisterende_ticks.append(lvl)
    eksisterende_ticks = sorted(set(eksisterende_ticks))
    ax.set_xticks(eksisterende_ticks)
    ax.set_xticklabels([f"{int(t)}" for t in eksisterende_ticks])

    # 7) Tegn terskel‐linjer med eksakte verdier fra config (klippet for visning)
    brukt_farger = set()
    for lvl, farge, etikett in terskelverdier:
        lvl_vis = np.clip(lvl, x_min, x_max)
        label = etikett if farge not in brukt_farger else ""
        ax.axvline(
            x=lvl_vis,
            color=farge,
            linestyle="--",
            linewidth=2,
            label=label
        )
        brukt_farger.add(farge)

    # 8) Profesjonelle titler per variabel
    tittel = ""
    if variable == "Temperatur (°C)":
        tittel = "Temperaturfordeling 10–35 °C (1 °C per søyle)"
    elif variable == "Luftfuktighet (%)":
        tittel = "Fuktighetsfordeling 15 %–75 % (2 % per søyle)"
    elif variable == "CO2 (ppm)":
        tittel = "CO₂-fordeling 400–1200 ppm (30 ppm per søyle)"
    elif "Formaldehyd" in variable:
        tittel = "Formaldehydfordeling 0–125 µg/m³ (20 µg/m³ per søyle)"
    elif "TVOC" in variable:
        tittel = "TVOC-fordeling 0–1500 ppb (30 ppb per søyle)"
    elif "PM 1.0" in variable:
        tittel = "PM 1.0-fordeling 0–40 µg/m³ (5 µg/m³ per søyle)"
    elif "PM 2.5" in variable:
        tittel = "PM 2.5-fordeling 0–40 µg/m³ (5 µg/m³ per søyle)"
    elif "PM 4" in variable:
        tittel = "PM 4-fordeling 0–40 µg/m³ (25 µg/m³ per søyle)"
    elif "PM 10" in variable:
        tittel = "PM 10-fordeling 0–110 µg/m³ (5 µg/m³ per søyle)"

    ax.set_title(tittel)

    # 9) Tegn legend for terskellinjer
    ax.legend(loc="upper right", fontsize=9)

    # 10) Legg inn rutenett og tight layout
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
