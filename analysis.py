import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict, Optional
from datetime import datetime
from building_analysis import TILGJENGELIGE_BYGG
from data_processing import fetch_csv, set_datetime_index
from plotting import plot_all_rooms_variable
from config import (THRESHOLDS_TEMPERATURE, THRESHOLDS_OPTIMAL_HUMIDITY,
                    THRESHOLDS_WARN, THRESHOLDS_CRITICAL, LUFTKVALITETS_VARIABLER_I_REKKE)

# Hvilke bygg skal brukes her
TILGJENGELIGE_BYGG = ["01", "02", "04", "05", "07", "08"]

VARIABLE_CHOICES = {
    "1": "Temperatur (°C)",
    "2": "Luftfuktighet (%)",
    "3": "CO2 (ppm)",
    "4": "Formaldehyd (µg/m³)",
    "5": "TVOC (ppb)",
    "6": "PM 1.0 (µg/m³)",
    "7": "PM 2.5 (µg/m³)",
    "8": "PM 4.0 (µg/m³)",
    "9": "PM 10 (µg/m³)"
}

# ─── START AV run_timer_over_terskel ─────────────────────────────────────────
def run_timer_over_terskel():
    """
    Ber brukeren om et byggnummer, henter df_list og romnavn via fetch_csv(),
    kaller tabell_timer_over_terskel(...) og skriver ut tabellene per rom.
    """


    # 1) Velg bygg
    byggkoder = list(TILGJENGELIGE_BYGG.keys())
    while True:
        print("\n🏢 VELG BYGG FOR 'TIMER OVER TERSKEL'")
        for kode in byggkoder:
            print(f"{kode}. {TILGJENGELIGE_BYGG[kode]}")
        print("b. Tilbake til forrige meny")

        valg = input("Valg: ").strip().lower()
        if valg == 'b':
            return

        if valg.zfill(2) in byggkoder:
            valgt_bygg = valg.zfill(2)
            break
        else:
            print("❌ Ugyldig byggkode. Prøv igjen.")

    # 2) Hent alle rom‐DataFrames for dette bygget
    try:
        dfs, romnavn, _ = fetch_csv(building_number=valgt_bygg)
    except Exception as e:
        print(f"⚠️ Klarte ikke hente data for bygg {valgt_bygg}: {e}")
        return

    if not dfs:
        print(f"❌ Ingen data tilgjengelig for bygg {valgt_bygg}.")
        return

    # 3) Kall funksjonen som bygger tabeller per rom
    resultater = tabell_timer_over_terskel(dfs, romnavn)
    if not resultater:
        print("❌ Fant ingen tabeller (sannsynligvis ingen målinger).")
        return

    # 4) Skriv ut én tabell per rom
    print(f"\n📋 Timer over terskel for Bygg {valgt_bygg} – {TILGJENGELIGE_BYGG[valgt_bygg]}")
    for rom, df_tab in resultater.items():
        print(f"\n── {rom} ──")
        print(df_tab.to_string(index=False))
# ─── SLUTT AV run_timer_over_terskel ──────────────────────────────────────────

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

        # Når plottet er lukket, havner vi her – brukeren kan taste
        # ny variabelkode eller 'b' for å forlate loop.
        # (Ingen ny meny-print‐sekvens før input‐prompten dukker opp.)
        # Derfor gjør vi en enkel “fortsett” tilbake til starten av while‐loopen:
        continue

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

        # 3) Plot histogram for valgte variabel (interaktivt)
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
        except Exception:
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
    thresholds = []
    y_label = variable
    title_prefix = f"{variable}"

    # Eksempel: temperatur‐terskler:
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
    for nivå, farge, etikett in thresholds:
        ax.axhline(y=nivå, color=farge, linestyle='--', linewidth=1.5, label=etikett)

    # 5) Legg på tittel‐eksempel (du kan tilpasse ytterligere om ønskelig)
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
        rå_bins = np.arange(400, 1200 + 30, 30)  # 400, 430, …, 1200
        bins = rå_bins
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
    for nivå, _, _ in terskelverdier:
        if nivå < x_min:
            eksisterende_ticks.append(x_min)
        elif nivå > x_max:
            eksisterende_ticks.append(x_max)
        else:
            eksisterende_ticks.append(nivå)
    eksisterende_ticks = sorted(set(eksisterende_ticks))
    ax.set_xticks(eksisterende_ticks)
    ax.set_xticklabels([f"{int(t)}" for t in eksisterende_ticks])

    # 7) Tegn terskel‐linjer med eksakte verdier fra config (klippet for visning)
    brukt_farger = set()
    for nivå, farge, etikett in terskelverdier:
        nivå_vis = np.clip(nivå, x_min, x_max)
        label = etikett if farge not in brukt_farger else ""
        ax.axvline(
            x=nivå_vis,
            color=farge,
            linestyle="--",
            linewidth=2,
            label=label
        )
        brukt_farger.add(farge)

    # 8) Profesjonelle titler per variabel
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

def vis_spredningsmål():
    """
    Lar brukeren først definere en studieperiode, lagrer denne, og deretter hente spredningsmål (min, Q1, median,
    Q3, maks, IQR, std dev) for én eller flere variabler på rad. Inkluderer valg av bygg (eller "a" = alle), valg
    om å splitte pr. rom, samt ny valgmulighet: periode fra måned–år til måned–år.
    """

    # Hjelpefunksjon for å finne siste dag i en måned
    def siste_dagen_i_måned(år: int, måned: int) -> int:
        if måned == 12:
            return 31
        neste = pd.Timestamp(year=år, month=måned + 1, day=1)
        return (neste - pd.Timedelta(days=1)).day

    # 1) La brukeren velge studieperiode én gang
    while True:
        print("\n⏳ VELG STUDIEPERIODE FOR SPREDNINGSMÅL")
        print("1. År")
        print("2. Vinter (Okt–Mars)")
        print("3. Sommer (Apr–Sep)")
        print("4. Måned")
        print("5. Uke")
        print("6. Dag")
        print("7. Fra måned–år til måned–år")
        print("b. Tilbake til hovedmeny")

        ans = input("Valg: ").strip().lower()
        if ans == 'b':
            return

        mode = None
        year = month = week = None
        day = None
        # Default for custom range
        start_date = None
        end_date = None

        if ans == '1':
            mode = 'year'
            inp = input("Skriv inn år (ÅÅÅÅ): ").strip()
            if not (inp.isdigit() and len(inp) == 4):
                print("❌ Ugyldig årstall.")
                continue
            year = int(inp)

        elif ans == '2':
            mode = 'winter'
            inp = input("Skriv inn årstall for vinterstart (ÅÅÅÅ): ").strip()
            if not (inp.isdigit() and len(inp) == 4):
                print("❌ Ugyldig årstall.")
                continue
            year = int(inp)

        elif ans == '3':
            mode = 'summer'
            inp = input("Skriv inn årstall (ÅÅÅÅ): ").strip()
            if not (inp.isdigit() and len(inp) == 4):
                print("❌ Ugyldig årstall.")
                continue
            year = int(inp)

        elif ans == '4':
            mode = 'month'
            inp = input("Skriv inn år (ÅÅÅÅ): ").strip()
            if not (inp.isdigit() and len(inp) == 4):
                print("❌ Ugyldig årstall.")
                continue
            year = int(inp)
            inp2 = input("Skriv inn måned (1–12): ").strip()
            if not (inp2.isdigit() and 1 <= int(inp2) <= 12):
                print("❌ Ugyldig måned.")
                continue
            month = int(inp2)

        elif ans == '5':
            mode = 'week'
            inp = input("Skriv inn år (ÅÅÅÅ): ").strip()
            if not (inp.isdigit() and len(inp) == 4):
                print("❌ Ugyldig årstall.")
                continue
            year = int(inp)
            inp2 = input("Skriv inn uke (1–53): ").strip()
            if not (inp2.isdigit() and 1 <= int(inp2) <= 53):
                print("❌ Ugyldig uke.")
                continue
            week = int(inp2)

        elif ans == '6':
            mode = 'day'
            inp = input("Skriv inn dato (YYYY-MM-DD): ").strip()
            try:
                day = pd.to_datetime(inp, format="%Y-%m-%d")
            except Exception:
                print("❌ Ugyldig dato. Bruk format YYYY-MM-DD.")
                continue
            year = day.year
            month = day.month
            week = int(day.strftime("%W"))

        elif ans == '7':
            mode = 'range'
            # Be om start måned–år
            inp = input("Skriv inn start måned–år (YYYY-MM): ").strip()
            try:
                dato = pd.to_datetime(inp + "-01", format="%Y-%m-%d")
                start_date = dato.replace(day=1)
            except Exception:
                print("❌ Ugyldig format. Bruk YYYY-MM.")
                continue

            # Be om slutt måned–år
            inp2 = input("Skriv inn slutt måned–år (YYYY-MM): ").strip()
            try:
                dato2 = pd.to_datetime(inp2 + "-01", format="%Y-%m-%d")
                år2, mnd2 = dato2.year, dato2.month
                siste = siste_dagen_i_måned(år2, mnd2)
                end_date = dato2.replace(day=siste)
            except Exception:
                print("❌ Ugyldig format. Bruk YYYY-MM.")
                continue

            # Sjekk at slutt er etter start
            if end_date < start_date:
                print("❌ Sluttdato må være etter startdato.")
                continue
        else:
            print("❌ Ugyldig valg.")
            continue

        # Hvis vi kommer hit, har vi en gyldig periode
        break

    # 2) Velg bygg eller 'a' for alle bygg
    byggkoder = list(TILGJENGELIGE_BYGG.keys())
    while True:
        print("\n🏢 VELG BYGG FOR SPREDNINGSMÅL (eller 'a' for alle bygg)")
        for kode in byggkoder:
            print(f"{kode}. {TILGJENGELIGE_BYGG[kode]}")
        print("a. Alle bygg")
        print("b. Tilbake til hovedmeny")

        valg_bygg = input("Valg: ").strip().lower()
        if valg_bygg == 'b':
            return
        if valg_bygg == 'a':
            valgt_byggliste = byggkoder.copy()
            break
        if valg_bygg.zfill(2) in byggkoder:
            valgt_byggliste = [valg_bygg.zfill(2)]
            break
        print("❌ Ugyldig byggkode. Prøv igjen.")

    # 3) Velg om man vil splitte pr. rom
    print("\n🔀 Ønsker du spredningsmål per ROM i hvert bygg?")
    print("j. Ja")
    print("n. Nei")
    svar = input("Valg: ").strip().lower()
    split_per_rom = (svar == 'j')

    # 4) Her starter løkka for variabelvalg (samme periode/byggvalg)
    while True:
        print("\n📊 VELG VARIABEL FOR SPREDNINGSMÅL")
        for key, var in VARIABLE_CHOICES.items():
            print(f"{key}. {var}")
        print("b. Tilbake til hovedmeny")

        valg = input("Valg: ").strip().lower()
        if valg == 'b':
            return
        if valg not in VARIABLE_CHOICES:
            print("❌ Ugyldig valg. Prøv igjen.")
            continue

        variable = VARIABLE_CHOICES[valg]
        # Gi brukeren en kort oppsummering av hvilken periode som er valgt
        periode_str = ""
        if mode == "year":
            periode_str = f"{year}"
        elif mode == "winter":
            periode_str = f"Vinter {year}-{year+1}"
        elif mode == "summer":
            periode_str = f"Sommer {year}"
        elif mode == "month":
            måned_navn = pd.to_datetime(f"{year}-{month:02d}-01").strftime("%B")
            periode_str = f"{måned_navn} {year}"
        elif mode == "week":
            start_u = pd.to_datetime(f"{year}-W{week:02d}-1", format="%Y-W%W-%w")
            slutt_u = start_u + pd.Timedelta(days=6)
            periode_str = f"Uke {week} {year} ({start_u.strftime('%d. %b')}–{slutt_u.strftime('%d. %b')})"
        elif mode == "day":
            periode_str = day.strftime("%-d. %b %Y") if os.name != "nt" else day.strftime("%#d. %b %Y")
        elif mode == "range":
            periode_str = f"{start_date.strftime('%b %Y')} til {end_date.strftime('%b %Y')}"

        print(f"\n👉 Spredningsmål for: {variable}  | Periode: {periode_str}\n")

        # 5) Hent og filtrer data per bygg/rom
        total_aggregert_for_flere_bygg = []

        for kode in valgt_byggliste:
            bygnavn = TILGJENGELIGE_BYGG[kode]
            try:
                dfs_bygg, romnavn_bygg, _ = fetch_csv(building_number=kode)
            except Exception as e:
                print(f"⚠️ Klarte ikke hente data for Bygg {kode}: {e}")
                continue

            # Samle serier for hvert rom, filtrert til valgt periode
            rom_data = {}

            for df_rom, rom in zip(dfs_bygg, romnavn_bygg):
                df2 = set_datetime_index(df_rom)

                # Filtrer på periode: bruk enten filter_data eller egen range for custom
                if mode == 'range':
                    df_f = df2[(df2.index >= start_date) & (df2.index <= end_date)]
                else:
                    # Bruk det generiske filteret for year/month/week/day
                    df_f_list = filter_data([df2], mode, year, month, week, day)
                    df_f = df_f_list[0] if df_f_list else pd.DataFrame()

                if df_f.empty or variable not in df_f.columns:
                    continue

                ser = df_f[variable].dropna()
                if ser.empty:
                    continue

                nøkkel = f"Rom {rom}"
                rom_data[nøkkel] = ser

            if not rom_data:
                print(f"❌ Ingen data for '{variable}' i Bygg {kode} ({bygnavn}).")
                continue

            # 6) Beregn spredningsmål
            if split_per_rom:
                print(f"\n📈 Bygg {kode} ({bygnavn}) – {variable}, per ROM:")
                for rom_label, serie in rom_data.items():
                    minimum = serie.min()
                    q25 = serie.quantile(0.25)
                    q50 = serie.quantile(0.50)
                    q75 = serie.quantile(0.75)
                    maksimum = serie.max()
                    iqr = q75 - q25
                    std_val = serie.std()
                    ant = len(serie)

                    print(f"  • {rom_label}: "
                          f"Antall {ant:>5}, Min {minimum:.2f}, Q1 {q25:.2f}, Med {q50:.2f}, "
                          f"Q3 {q75:.2f}, Maks {maksimum:.2f}, IQR {iqr:.2f}, Std {std_val:.2f}")

                    total_aggregert_for_flere_bygg.append(serie)

            else:
                felles = pd.concat(list(rom_data.values()))
                total_aggregert_for_flere_bygg.append(felles)

                minimum = felles.min()
                q25 = felles.quantile(0.25)
                q50 = felles.quantile(0.50)
                q75 = felles.quantile(0.75)
                maksimum = felles.max()
                iqr = q75 - q25
                std_val = felles.std()
                ant = len(felles)

                print(f"\n📈 Bygg {kode} ({bygnavn}) – {variable}, samlet:")
                print(f"  Antall målinger      : {ant}")
                print(f"  Minimum              : {minimum:.2f}")
                print(f"  25%-percentil (Q1)   : {q25:.2f}")
                print(f"  Median (Q2)          : {q50:.2f}")
                print(f"  75%-percentil (Q3)   : {q75:.2f}")
                print(f"  Maksimum             : {maksimum:.2f}")
                print(f"  IQR (Q3 − Q1)        : {iqr:.2f}")
                print(f"  Standardavvik (std)   : {std_val:.2f}")

        # 7) Hvis flere bygg, vis samlet aggregert
        if len(valgt_byggliste) > 1 and total_aggregert_for_flere_bygg:
            slått_sammen = pd.concat(total_aggregert_for_flere_bygg)
            minimum = slått_sammen.min()
            q25 = slått_sammen.quantile(0.25)
            q50 = slått_sammen.quantile(0.50)
            q75 = slått_sammen.quantile(0.75)
            maksimum = slått_sammen.max()
            iqr = q75 - q25
            std_val = slått_sammen.std()
            ant = len(slått_sammen)

            print(f"\n🔢 Samlet spredningsmål for '{variable}' – Alle valgte bygg:")
            print(f"  Antall målinger      : {ant}")
            print(f"  Minimum              : {minimum:.2f}")
            print(f"  25%-percentil (Q1)   : {q25:.2f}")
            print(f"  Median (Q2)          : {q50:.2f}")
            print(f"  75%-percentil (Q3)   : {q75:.2f}")
            print(f"  Maksimum             : {maksimum:.2f}")
            print(f"  IQR (Q3 − Q1)        : {iqr:.2f}")
            print(f"  Standardavvik (std)   : {std_val:.2f}")

        # 8) Etter utskrift kan brukeren taste ny variabel direkte
        # Fortsetter i samme løkke: nye valg → nye beregninger innen samme periode/byggvalg
        continue

def tabell_timer_over_terskel(
    df_list: List[pd.DataFrame],
    romnavn: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    For hver rom‐serie i df_list (én DataFrame per rom), bygger en tabell (DataFrame)
    som viser for hver variabel:
      - "Variabel"
      - "Timer med data"
      - "Timer over terskel"
      - "Lengste overperiode"
      - "Gj.snitt overskridelse"
      - "Timer under terskel"      <-- flyttet til høyre
      - "Lengste underperiode"     <-- flyttet til høyre
      - "Gj.snitt underskudd"      <-- flyttet til høyre

    Returnerer en dict:
        { "Rom X": DataFrame, "Rom Y": DataFrame, ... }
    """
    resultat: Dict[str, pd.DataFrame] = {}

    # 1) Hent terskelverdier
    temp_min = THRESHOLDS_TEMPERATURE["day"]["min"]
    temp_max = THRESHOLDS_TEMPERATURE["day"]["max"]

    hum_opts = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
    hum_min = hum_opts["optimal_min"]
    hum_max = hum_opts["optimal_max"]

    warn_terskler = {
        var: THRESHOLDS_WARN.get(var)
        for var in LUFTKVALITETS_VARIABLER_I_REKKE
    }

    # 2) Hjelpefunksjon: finn lengste sammenhengende run av True i en bool‐serie
    def lengste_run(mask: pd.Series) -> int:
        if mask.empty or not mask.any():
            return 0
        grp = (mask != mask.shift()).cumsum()
        runs = mask.groupby(grp).sum()
        return int(runs.max())

    # 3) Gå gjennom hvert rom
    for idx, df in enumerate(df_list, start=1):
        romnummer = romnavn[idx-1] if idx-1 < len(romnavn) else str(idx)
        romlabel = f"Rom {romnummer}"
        if df is None or df.empty:
            continue

        # Sørg for DateTimeIndex og reindekser til timeintervall
        df2 = set_datetime_index(df.copy())
        start = df2.index.min()
        slutt = df2.index.max()
        full_idx = pd.date_range(start=start, end=slutt, freq="1h")
        df2 = df2.reindex(full_idx)

        rader = []

        # 3.a) Temperatur
        if "Temperatur (°C)" in df2.columns:
            series = df2["Temperatur (°C)"]
            non_na = series.dropna()
            timer_med_data = non_na.count()

            over_mask = series > temp_max
            under_mask = series < temp_min

            timer_over = int(over_mask.sum())
            timer_under = int(under_mask.sum())

            lengst_over = lengste_run(over_mask.fillna(False))
            lengst_under = lengste_run(under_mask.fillna(False))

            overskr_values = (series[over_mask] - temp_max).dropna()
            gj_snitt_over = float(overskr_values.mean()) if not overskr_values.empty else 0.0

            undersk_values = (temp_min - series[under_mask]).dropna()
            gj_snitt_under = float(undersk_values.mean()) if not undersk_values.empty else 0.0

            rader.append({
                "Variabel": "Temperatur (°C)",
                "Timer med data": timer_med_data,
                "Timer over terskel": timer_over,
                "Lengste overperiode": lengst_over,
                "Gj.snitt overskridelse": round(gj_snitt_over, 2),
                "Timer under terskel": timer_under,
                "Lengste underperiode": lengst_under,
                "Gj.snitt underskudd": round(gj_snitt_under, 2)
            })

        # 3.b) Luftfuktighet
        if "Luftfuktighet (%)" in df2.columns:
            series = df2["Luftfuktighet (%)"]
            non_na = series.dropna()
            timer_med_data = non_na.count()

            over_mask = series > hum_max
            under_mask = series < hum_min

            timer_over = int(over_mask.sum())
            timer_under = int(under_mask.sum())

            lengst_over = lengste_run(over_mask.fillna(False))
            lengst_under = lengste_run(under_mask.fillna(False))

            overskr_values = (series[over_mask] - hum_max).dropna()
            gj_snitt_over = float(overskr_values.mean()) if not overskr_values.empty else 0.0

            undersk_values = (hum_min - series[under_mask]).dropna()
            gj_snitt_under = float(undersk_values.mean()) if not undersk_values.empty else 0.0

            rader.append({
                "Variabel": "Luftfuktighet (%)",
                "Timer med data": timer_med_data,
                "Timer over terskel": timer_over,
                "Lengste overperiode": lengst_over,
                "Gj.snitt overskridelse": round(gj_snitt_over, 2),
                "Timer under terskel": timer_under,
                "Lengste underperiode": lengst_under,
                "Gj.snitt underskudd": round(gj_snitt_under, 2)
            })

        # 3.c) Øvrige luftkvalitetsvariabler
        for var in LUFTKVALITETS_VARIABLER_I_REKKE:
            if var not in df2.columns:
                continue
            terskel = warn_terskler.get(var)
            if terskel is None:
                continue

            series = df2[var]
            non_na = series.dropna()
            timer_med_data = non_na.count()

            over_mask = series > terskel

            timer_over = int(over_mask.sum())
            lengst_over = lengste_run(over_mask.fillna(False))

            overskr_values = (series[over_mask] - terskel).dropna()
            gj_snitt_over = float(overskr_values.mean()) if not overskr_values.empty else 0.0

            rader.append({
                "Variabel": var,
                "Timer med data": timer_med_data,
                "Timer over terskel": timer_over,
                "Lengste overperiode": lengst_over,
                "Gj.snitt overskridelse": round(gj_snitt_over, 2),
                "Timer under terskel": "-",
                "Lengste underperiode": "-",
                "Gj.snitt underskudd": "-"
            })

        # 4) Bygg DataFrame, og sikre at kolonnene i ønsket rekkefølge
        df_res = pd.DataFrame(rader, columns=[
            "Variabel",
            "Timer med data",
            "Timer over terskel",
            "Lengste overperiode",
            "Gj.snitt overskridelse",
            "Timer under terskel",
            "Lengste underperiode",
            "Gj.snitt underskudd"
        ])
        resultat[romlabel] = df_res

    return resultat


def tabell_prosent_over_terskel_global(
    df_list: List[pd.DataFrame],
    romnavn: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Lager to tabeller for hele datasettet (alle rom i df_list):
      1) Tabell for temperatur og luftfuktighet:
         - Prosent av timer med målinger over henholdsvis over- og under- terskel.
      2) Tabell for øvrige luftkvalitetsvariabler:
         - Prosent av timer med målinger over varsel-terskel.

    Returnerer dict med nøkler:
      "temp_humid": DataFrame for temperatur/luftfuktighet,
      "others"    : DataFrame for øvrige variabler.
    """

    # 1) Hent terskelverdier
    temp_min = THRESHOLDS_TEMPERATURE["day"]["min"]
    temp_max = THRESHOLDS_TEMPERATURE["day"]["max"]

    hum_opts = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
    hum_min = hum_opts["optimal_min"]
    hum_max = hum_opts["optimal_max"]

    warn_terskler = {
        var: THRESHOLDS_WARN.get(var)
        for var in LUFTKVALITETS_VARIABLER_I_REKKE
    }

    # 2) Samle alle måleserier per variabel
    temp_values_all = []
    humid_values_all = []
    others_values = {var: [] for var in LUFTKVALITETS_VARIABLER_I_REKKE}

    for idx, df in enumerate(df_list):
        if df is None or df.empty:
            continue

        # Sørg for DateTimeIndex og reindekser til timeintervall
        df2 = set_datetime_index(df.copy())
        start = df2.index.min()
        slutt = df2.index.max()
        full_idx = pd.date_range(start=start, end=slutt, freq="1h")
        df2 = df2.reindex(full_idx)

        # Temp
        if "Temperatur (°C)" in df2.columns:
            temp_values_all.append(df2["Temperatur (°C)"])

        # Luftfuktighet
        if "Luftfuktighet (%)" in df2.columns:
            humid_values_all.append(df2["Luftfuktighet (%)"])

        # Andre variabler
        for var in LUFTKVALITETS_VARIABLER_I_REKKE:
            if var in df2.columns:
                others_values[var].append(df2[var])

    # 3) Konkatener og fjern NaN
    temp_concat = pd.concat(temp_values_all).dropna() if temp_values_all else pd.Series(dtype=float)
    humid_concat = pd.concat(humid_values_all).dropna() if humid_values_all else pd.Series(dtype=float)

    # 4) Beregn prosent for temp/humid
    temp_total = len(temp_concat)
    temp_over = (temp_concat > temp_max).sum() if temp_total else 0
    temp_under = (temp_concat < temp_min).sum() if temp_total else 0
    temp_over_pct = round((temp_over / temp_total) * 100, 2) if temp_total else 0.0
    temp_under_pct = round((temp_under / temp_total) * 100, 2) if temp_total else 0.0

    humid_total = len(humid_concat)
    humid_over = (humid_concat > hum_max).sum() if humid_total else 0
    humid_under = (humid_concat < hum_min).sum() if humid_total else 0
    humid_over_pct = round((humid_over / humid_total) * 100, 2) if humid_total else 0.0
    humid_under_pct = round((humid_under / humid_total) * 100, 2) if humid_total else 0.0

    # 5) Lag DataFrame for temperatur og luftfuktighet
    df_temp_humid = pd.DataFrame([
        {
            "Variabel": "Temperatur (°C)",
            "Prosent over terskel": temp_over_pct,
            "Prosent under terskel": temp_under_pct
        },
        {
            "Variabel": "Luftfuktighet (%)",
            "Prosent over terskel": humid_over_pct,
            "Prosent under terskel": humid_under_pct
        }
    ])

    # 6) Beregn prosent for øvrige variabler
    rader = []
    for var, ser_list in others_values.items():
        if not ser_list:
            continue
        concat = pd.concat(ser_list).dropna()
        total = len(concat)
        terskel = warn_terskler.get(var)
        over = (concat > terskel).sum() if (terskel is not None and total) else 0
        over_pct = round((over / total) * 100, 2) if total else 0.0
        rader.append({
            "Variabel": var,
            "Prosent over terskel": over_pct
        })

    df_others = pd.DataFrame(rader, columns=["Variabel", "Prosent over terskel"])

    return {
        "temp_humid": df_temp_humid,
        "others": df_others
    }

def tabell_prosent_over_terskel_global(
    df_list: List[pd.DataFrame],
    romnavn: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Lager to tabeller for hele datasettet (alle rom i df_list):
      1) Tabell for temperatur og luftfuktighet:
         - Prosent av timer med målinger over henholdsvis over- og under- terskel.
      2) Tabell for øvrige luftkvalitetsvariabler:
         - Prosent av timer med målinger over varsel-terskel.

    Returnerer dict med nøkler:
      "temp_humid": DataFrame for temperatur/luftfuktighet,
      "others"    : DataFrame for øvrige variabler.
    """

    # 1) Hent terskelverdier
    temp_min = THRESHOLDS_TEMPERATURE["day"]["min"]
    temp_max = THRESHOLDS_TEMPERATURE["day"]["max"]

    hum_opts = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
    hum_min = hum_opts["optimal_min"]
    hum_max = hum_opts["optimal_max"]

    warn_terskler = {
        var: THRESHOLDS_WARN.get(var)
        for var in LUFTKVALITETS_VARIABLER_I_REKKE
    }

    # 2) Samle alle måleserier per variabel
    temp_values_all = []
    humid_values_all = []
    others_values = {var: [] for var in LUFTKVALITETS_VARIABLER_I_REKKE}

    for idx, df in enumerate(df_list):
        if df is None or df.empty:
            continue

        # Sørg for DateTimeIndex og reindekser til timeintervall
        df2 = set_datetime_index(df.copy())
        start = df2.index.min()
        slutt = df2.index.max()
        full_idx = pd.date_range(start=start, end=slutt, freq="1h")
        df2 = df2.reindex(full_idx)

        # Temp
        if "Temperatur (°C)" in df2.columns:
            temp_values_all.append(df2["Temperatur (°C)"])

        # Luftfuktighet
        if "Luftfuktighet (%)" in df2.columns:
            humid_values_all.append(df2["Luftfuktighet (%)"])

        # Andre variabler
        for var in LUFTKVALITETS_VARIABLER_I_REKKE:
            if var in df2.columns:
                others_values[var].append(df2[var])

    # 3) Konkatener og fjern NaN
    temp_concat = pd.concat(temp_values_all).dropna() if temp_values_all else pd.Series(dtype=float)
    humid_concat = pd.concat(humid_values_all).dropna() if humid_values_all else pd.Series(dtype=float)

    # 4) Beregn prosent for temp/humid
    temp_total = len(temp_concat)
    temp_over = (temp_concat > temp_max).sum() if temp_total else 0
    temp_under = (temp_concat < temp_min).sum() if temp_total else 0
    temp_over_pct = round((temp_over / temp_total) * 100, 2) if temp_total else 0.0
    temp_under_pct = round((temp_under / temp_total) * 100, 2) if temp_total else 0.0

    humid_total = len(humid_concat)
    humid_over = (humid_concat > hum_max).sum() if humid_total else 0
    humid_under = (humid_concat < hum_min).sum() if humid_total else 0
    humid_over_pct = round((humid_over / humid_total) * 100, 2) if humid_total else 0.0
    humid_under_pct = round((humid_under / humid_total) * 100, 2) if humid_total else 0.0

    # 5) Lag DataFrame for temperatur og luftfuktighet
    df_temp_humid = pd.DataFrame([
        {
            "Variabel": "Temperatur (°C)",
            "Prosent over terskel": temp_over_pct,
            "Prosent under terskel": temp_under_pct
        },
        {
            "Variabel": "Luftfuktighet (%)",
            "Prosent over terskel": humid_over_pct,
            "Prosent under terskel": humid_under_pct
        }
    ])

    # 6) Beregn prosent for øvrige variabler
    rader = []
    for var, ser_list in others_values.items():
        if not ser_list:
            continue
        concat = pd.concat(ser_list).dropna()
        total = len(concat)
        terskel = warn_terskler.get(var)
        over = (concat > terskel).sum() if (terskel is not None and total) else 0
        over_pct = round((over / total) * 100, 2) if total else 0.0
        rader.append({
            "Variabel": var,
            "Prosent over terskel": over_pct
        })

    df_others = pd.DataFrame(rader, columns=["Variabel", "Prosent over terskel"])

    return {
        "temp_humid": df_temp_humid,
        "others": df_others
    }


def run_prosent_over_terskel_global():
    """
    Lar brukeren velge ett bygg (eller 'a' for alle bygg), viser kun byggnummeret,
    henter alle rom‐DataFrames for det/de valgte byggene, og kaller
    tabell_prosent_over_terskel_global(...) for å skrive ut de to tabellene.
    """
    # 1) Velg bygg eller alle
    byggkoder = list(building_analysis.TILGJENGELIGE_BYGG.keys())
    while True:
        print("\n🏢 VELG BYGG FOR 'PROSENT OVER TERSKEL GLOBAL'")
        for kode in byggkoder:
            print(f"{kode}")
        print("a. Alle bygg")
        print("b. Tilbake til meny")

        valg = input("Valg: ").strip().lower()
        if valg == 'b':
            return

        if valg == 'a':
            valgt_liste = byggkoder.copy()
            break
        if valg.zfill(2) in byggkoder:
            valgt_liste = [valg.zfill(2)]
            break
        else:
            print("❌ Ugyldig byggkode. Prøv igjen.")

    # 2) Skriv ut byggnummer for det/de valgte byggene
    if len(valgt_liste) == 1:
        print(f"\nValgt bygg: {valgt_liste[0]}")
    else:
        print("\nValgte bygg:")
        for kode in valgt_liste:
            print(f"  {kode}")

    # 3) Hent DataFrames for alle valgte bygg
    samlet_dfs = []
    samlet_romnavn = []
    for kode in valgt_liste:
        try:
            dfs, romnavn, _ = fetch_csv(building_number=kode)
        except Exception as e:
            print(f"⚠️ Klarte ikke hente data for bygg {kode}: {e}")
            continue

        samlet_dfs.extend(dfs)
        samlet_romnavn.extend(romnavn)

    if not samlet_dfs:
        print("❌ Ingen data tilgjengelig for de valgte byggene.")
        return

    # 4) Kall funksjonen som lager de globale tabellene
    resultater = tabell_prosent_over_terskel_global(samlet_dfs, samlet_romnavn)

    df_temp_humid = resultater.get("temp_humid")
    df_others     = resultater.get("others")

    # 5) Skriv ut tabellen for temperatur og luftfuktighet
    if df_temp_humid is not None and not df_temp_humid.empty:
        header = "Prosent over/under terskel for Temperatur og Luftfuktighet"
        if len(valgt_liste) == 1:
            header += f" (Bygg {valgt_liste[0]})"
        else:
            header += " (Alle bygg)"
        print(f"\n📋 {header}")
        print(df_temp_humid.to_string(index=False))
    else:
        print("\n❌ Ingen temperatur- eller luftfuktdata å vise.")

    # 6) Skriv ut tabellen for øvrige variabler
    if df_others is not None and not df_others.empty:
        header2 = "Prosent over terskel for øvrige luftkvalitetsvariabler"
        if len(valgt_liste) == 1:
            header2 += f" (Bygg {valgt_liste[0]})"
        else:
            header2 += " (Alle bygg)"
        print(f"\n📋 {header2}")
        print(df_others.to_string(index=False))
    else:
        print("\n❌ Ingen luftkvalitetsdata å vise for øvrige variabler.")