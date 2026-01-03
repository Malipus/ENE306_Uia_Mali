import pandas as pd
from typing import List
from data_processing import fetch_csv, set_datetime_index, filter_data, fetch_weather
from plotting import plot_temperature, plot_humidity, plot_air_quality_variable
from config import LUFTKVALITETS_VARIABLER_I_REKKE, INNEKLIMA_DIR

TILGJENGELIGE_BYGG = {
    '01': 'Tønnevoldsgate 26, Sentrum',
    '02': 'Jon Lilletuns Vei 2A, Campus',
    '04': 'Jon Lilletuns Vei 15, Campus',
    '05': 'Jon Lilletuns Vei 17, Campus',
    '07': 'Jon Lilletuns Vei 21, Campus',
    '08': 'Jon Lilletuns Vei 23, Campus'
}


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


def run_building_analysis():
    # 1) Vis byggliste og la bruker velge
    print("🏢 Tilgjengelige bygg for analyse:")
    for kode, navn in TILGJENGELIGE_BYGG.items():
        print(f"  Bygg {kode} – {navn}")

    user_in = input("Velg bygg eller 'b' for å gå tilbake: ").strip().lower()
    if user_in == 'b':
        return

    byggkode = user_in.zfill(2)
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
        print("2. Vinter (Okt–Mars)")
        print("3. Sommer (Apr–Sep)")
        print("4. Måned")
        print("5. Uke")
        print("6. Dag")
        print("b. Tilbake til hovedmeny")
        periode_valg = input("Ditt valg: ").strip().lower()
        if periode_valg == 'b':
            return  # Gå tilbake til hovedmeny

        mode = None
        year = month = week = None
        day = None

        # ── Sett mode + tilhørende år/måned/uke/dag ──
        if periode_valg == '1':  # År
            mode = 'year'
            year = be_om_år()
            if year is None:
                continue

        elif periode_valg == '2':  # Vinter (okt–mar)
            mode = 'winter'
            year = be_om_år()
            if year is None:
                continue

        elif periode_valg == '3':  # Sommer (apr–sep)
            mode = 'summer'
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
                plot_temperature(
                    filtrerte_data,
                    mode,
                    year,
                    month,
                    week,
                    day,
                    weather_df,
                    byggkode,   # passér byggkode først
                    romnavn     # deretter liste over romnavn
                )

            elif valg_int == 2:
                # Luftfuktighet
                plot_humidity(
                    filtrerte_data,
                    mode,
                    year,
                    month,
                    week,
                    day,
                    weather_df,
                    byggkode,
                    romnavn
                )

            elif 3 <= valg_int < 3 + len(LUFTKVALITETS_VARIABLER_I_REKKE):
                # Luftkvalitetsvariabler (CO2, Formaldehyd, TVOC, PM osv.)
                var_idx = valg_int - 3
                variable = LUFTKVALITETS_VARIABLER_I_REKKE[var_idx]
                plot_air_quality_variable(
                    filtrerte_data,
                    variable,
                    mode,
                    year,
                    month,
                    week,
                    day,
                    byggkode,
                    romnavn
                )

            else:
                print("❌ Ugyldig valg.")
                continue

            # Når bruker lukker graf‐vinduet, returnerer vi hit og kan velge ny variabel
            continue

def gyldig_år(inp: str) -> bool:
    return len(inp) == 4 and inp.isdigit()

def gyldig_måned(inp: str) -> bool:
    return inp.isdigit() and 1 <= int(inp) <= 12

def gyldig_uke(inp: str) -> bool:
    return inp.isdigit() and 1 <= int(inp) <= 53
