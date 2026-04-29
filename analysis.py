import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from data_processing import fetch_csv, set_datetime_index, filter_data, fetch_weather
from plotting import (plot_all_rooms_variable,
                    plot_temperature,
                    plot_humidity,
                    plot_building_boxplot,
                    plot_pm_boxplots,
                    plot_room_data_availability,
                    plot_threshold_scatter)



from config import (THRESHOLDS_TEMPERATURE,
                    THRESHOLDS_OPTIMAL_HUMIDITY,
                    THRESHOLDS_WARN,
                    THRESHOLDS_CRITICAL,
                    VARIABLE_CHOICES,
                    TILGJENGELIGE_BYGG,
                    PM_VARIABLER)


def format_scope_label(state):
    if state["mode"] == "all":
        return "Hele perioden"
    elif state["mode"] == "year":
        return f"År {state['year']}"
    elif state["mode"] == "month":
        return f"Måned {state['month']}/{state['year']}"
    elif state["mode"] == "week":
        return f"Uke {state['week']} i {state['year']}"
    elif state["mode"] == "day" and state["day"] is not None:
        return state["day"].strftime("%Y-%m-%d")
    elif state["mode"] == "fall":
        return f"Høst {state['year']}"
    elif state["mode"] == "spring":
        return f"Vår {state['year']}"
    return "Ukjent periode"



def collect_boxplot_data_by_building(variabel: str, state):
    bygg_data = []
    bygg_labels = []

    for byggkode in state["buildings"]:
        dfs, romnavn, _ = fetch_csv(building_number=byggkode)

        for df, rom in zip(dfs, romnavn):
            if not room_is_selected(state, byggkode, rom):
                continue

        alle_verdier = []

        for df in dfs:
            df = set_datetime_index(df)

            filtrerte_liste = filter_data(
                [df],
                mode=state["mode"],
                year=state["year"],
                month=state["month"],
                week=state["week"],
                day=state["day"]
            )

            if not filtrerte_liste:
                continue

            df_filtrert = filtrerte_liste[0]

            if variabel in df_filtrert.columns:
                serie: pd.Series = pd.to_numeric(df_filtrert[variabel], errors="coerce")
                verdier = serie.dropna()

                if not verdier.empty:
                    alle_verdier.extend(verdier.tolist())

        if alle_verdier:
            bygg_data.append(alle_verdier)
            bygg_labels.append(f"Bygg {int(byggkode)}")


    return bygg_data, bygg_labels


def collect_threshold_scatter_data(variable: str, state, threshold: float, direction: str) -> pd.DataFrame:
    rows = []

    for byggkode in state["buildings"]:
        dfs, romnavn, _ = fetch_csv(building_number=byggkode)

        for df, romnavn_i in zip(dfs, romnavn):
            if not room_is_selected(state, byggkode, romnavn_i):
                continue

            df = set_datetime_index(df)

            filtrerte_liste = filter_data(
                [df],
                mode=state["mode"],
                year=state["year"],
                month=state["month"],
                week=state["week"],
                day=state["day"]
            )

            if not filtrerte_liste:
                continue

            df_filtrert = filtrerte_liste[0]

            if variable not in df_filtrert.columns:
                continue

            serie = pd.Series(
                pd.to_numeric(df_filtrert[variable], errors="coerce"),
                index=df_filtrert.index
            ).dropna()

            if serie.empty:
                continue

            if direction == "above":
                brudd = serie[serie > threshold]
            elif direction == "below":
                brudd = serie[serie < threshold]
            else:
                raise ValueError(f"Ukjent retning: {direction}")

            for tidspunkt, verdi in brudd.items():
                rows.append({
                    "Tid": tidspunkt,
                    "Verdi": float(verdi),
                    "Byggkode": byggkode,
                    "Bygg": f"Bygg {int(byggkode)}",
                    "Rom": romnavn_i
                })

    if not rows:
        return pd.DataFrame(columns=["Tid", "Verdi", "Byggkode", "Bygg", "Rom"])

    df = pd.DataFrame(rows)
    df.sort_values(by=["Tid", "Bygg", "Rom"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_threshold_scatter_specs(variable: str):
    if variable == "Temperatur (°C)":
        lower = THRESHOLDS_TEMPERATURE["night"]["min"]
        upper = THRESHOLDS_TEMPERATURE["day"]["max"]
        return [
            (lower, "below", f"Temperatur under {lower:g} °C"),
            (upper, "above", f"Temperatur over {upper:g} °C"),
        ]

    if variable == "Luftfuktighet (%)":
        grenser = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        lower = grenser["critical_min"]
        upper = grenser["critical_max"]
        return [
            (lower, "below", f"Luftfuktighet under {lower:g} %"),
            (upper, "above", f"Luftfuktighet over {upper:g} %"),
        ]

    critical = THRESHOLDS_CRITICAL.get(variable)
    if critical is None:
        return []

    return [(critical, "above", f"{variable} over kritisk grense ({critical:g})")]


def get_scope_limits(state):
    mode = state["mode"]
    year = state["year"]
    month = state["month"]
    week = state["week"]
    day = state["day"]

    if mode == "day" and isinstance(day, pd.Timestamp):
        start_vis = day.replace(hour=0, minute=0, second=0)
        slutt_vis = start_vis + pd.Timedelta(days=1)
    elif mode == "week" and isinstance(year, int) and isinstance(week, int):
        start_vis = pd.Timestamp(datetime.fromisocalendar(year, week, 1))
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
        return None, None

    return start_vis, slutt_vis


def build_boxplot_summary_df(variable: str, building_data, building_labels) -> pd.DataFrame:
    rows = []

    for label, values in zip(building_labels, building_data):
        if not values:
            continue

        series = pd.Series(values, dtype="float64").dropna()
        if series.empty:
            continue

        rows.append({
            "Bygg": label,
            "Variabel": variable,
            "Antall": int(series.count()),
            "Minimum": round(series.min(), 2),
            "Q1": round(series.quantile(0.25), 2),
            "Median": round(series.median(), 2),
            "Gjennomsnitt": round(series.mean(), 2),
            "Q3": round(series.quantile(0.75), 2),
            "Maksimum": round(series.max(), 2),
        })

    return pd.DataFrame(rows)


def print_boxplot_summary(variable: str, building_data, building_labels, scope_label: str):
    summary_df = build_boxplot_summary_df(variable, building_data, building_labels)

    print(f"\nBoxplot-verdier for {variable}")
    print(f"Periode: {scope_label}")

    if summary_df.empty:
        print("Ingen data tilgjengelig.\n")
        return

    print(summary_df.to_string(index=False))
    print()


def print_pm_boxplot_summary(pm_plot_data, scope_label: str):
    print("\nBoxplot-verdier for partikler")
    print(f"Periode: {scope_label}\n")

    for variable, building_data, building_labels in pm_plot_data:
        summary_df = build_boxplot_summary_df(variable, building_data, building_labels)

        print(variable)
        if summary_df.empty:
            print("Ingen data tilgjengelig.\n")
        else:
            print(summary_df.to_string(index=False))
            print()


def run_boxplot_menu(state):
    while True:
        print("\n📦 BOKSPLOTT")
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

        scope_label = format_scope_label(state)

        if valg == "6":
            pm_plot_data = []

            for variable in PM_VARIABLER:
                building_data, building_labels = collect_boxplot_data_by_building(variable, state)
                pm_plot_data.append((variable, building_data, building_labels))

            print_pm_boxplot_summary(pm_plot_data, scope_label)
            plot_pm_boxplots(pm_plot_data, scope_label)


        elif valg in variabler:

            variable = variabler[valg]
            building_data, building_labels = collect_boxplot_data_by_building(variable, state)

            print_boxplot_summary(variable, building_data, building_labels, scope_label)
            plot_building_boxplot(variable, building_data, building_labels, scope_label)


        else:
            print("❌ Ugyldig valg.")


def run_threshold_scatter_menu(state):
    while True:
        print("\n📍 GRENSEBRUDD (SCATTER)")
        for key, var in VARIABLE_CHOICES.items():
            print(f"{key}. {var}")
        print("b. Tilbake til hovedmeny")

        valg = input("Valg: ").strip().lower()
        if valg == "b":
            return

        if valg not in VARIABLE_CHOICES:
            print("❌ Ugyldig valg. Prøv igjen.")
            continue

        variable = VARIABLE_CHOICES[valg]
        scope_label = format_scope_label(state)
        start_vis, slutt_vis = get_scope_limits(state)
        specs = build_threshold_scatter_specs(variable)

        if not specs:
            print(f"❌ Ingen grenseverdier definert for '{variable}'.")
            continue

        for threshold, direction, title in specs:
            scatter_df = collect_threshold_scatter_data(variable, state, threshold, direction)
            plot_threshold_scatter(
                scatter_df,
                variable=variable,
                threshold=threshold,
                direction=direction,
                scope_label=scope_label,
                title=title,
                start_vis=start_vis,
                slutt_vis=slutt_vis,
            )


def print_active_scope(state):

    if state["mode"] == "all":
        periodetekst = "Hele perioden"
    elif state["mode"] == "year":
        periodetekst = f"År {state['year']}"
    elif state["mode"] == "month":
        periodetekst = f"Måned {state['month']}/{state['year']}"
    elif state["mode"] == "week":
        periodetekst = f"Uke {state['week']} i {state['year']}"
    elif state["mode"] == "day" and state["day"] is not None:
        periodetekst = state["day"].strftime("%Y-%m-%d")
    elif state["mode"] == "fall":
        periodetekst = f"Høst {state['year']}"
    elif state["mode"] == "spring":
        periodetekst = f"Vår {state['year']}"
    else:
        periodetekst = "Ukjent periode"

    print("\nAktivt datasett:")
    print(f"Periode: {periodetekst}")

    romvalg = state.get("rooms_by_building", {})

    for byggkode in state["buildings"]:
        valgte_rom = romvalg.get(byggkode)

        if valgte_rom is not None:
            print(f"Bygg {byggkode}: Rom: {','.join(valgte_rom)}")


def prompt_year(prompt="Skriv inn år (ÅÅÅÅ) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None
    if not (inp.isdigit() and len(inp) == 4 and 2000 <= int(inp) <= 2100):
        print("❌ Ugyldig årstall. Prøv igjen.")
        return prompt_year(prompt)
    return int(inp)


def prompt_month(prompt="Skriv inn måned (1–12) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None
    if not (inp.isdigit() and 1 <= int(inp) <= 12):
        print("❌ Ugyldig måned. Prøv igjen.")
        return prompt_month(prompt)
    return int(inp)


def prompt_week(prompt="Skriv inn uke (1–53) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None
    if not (inp.isdigit() and 1 <= int(inp) <= 53):
        print("❌ Ugyldig uke. Prøv igjen.")
        return prompt_week(prompt)
    return int(inp)


def prompt_day(prompt="Skriv inn dato (ÅÅÅÅ-MM-DD) eller 'b': "):
    inp = input(prompt).strip()
    if inp.lower() == "b":
        return None

    try:
        return datetime.strptime(inp, "%Y-%m-%d")
    except ValueError:
        print("❌ Ugyldig datoformat eller ugyldig dato. Prøv igjen.")
        return prompt_day(prompt)


def reset_time_scope(state):
    state["mode"] = "all"
    state["year"] = None
    state["month"] = None
    state["week"] = None
    state["day"] = None


def select_building(state):
    while True:
        print("\nBYGGVALG")
        print("1. Alle bygg")
        print("2. Velg ut bygg")
        print("b. Tilbake")

        valg = input("Velg bygg: ").strip().lower()

        if valg == "b":
            return

        if valg == "1":
            state["buildings"] = list(TILGJENGELIGE_BYGG.keys())
            print("✅ Alle bygg er valgt.")
            return

        if valg == "2":
            state["buildings"]= []
            print("\nTilgjengelige bygg:")
            for kode, navn in TILGJENGELIGE_BYGG.items():
                print(f"{kode} - {navn}")

            byggkode = input("Oppgi ønskede byggnummer: f.eks 125 ").strip().zfill(2)

            for i in byggkode:
                if i in TILGJENGELIGE_BYGG:
                    state["buildings"].append(i)
                    print(f"✅ Bygg {i} er valgt.")

            if state["buildings"]:
                return

            print("❌ Ugyldig byggkode.")
            continue

        print("❌ Ugyldig valg.")


def select_time_scope(state):
    while True:
        print("\nTIDSVALG")
        print("1. Hele perioden")
        print("2. År")
        print("3. Høst")
        print("4. Vår")
        print("5. Måned")
        print("6. Uke")
        print("7. Dag")
        print("b. Tilbake")

        valg = input("Velg periode: ").strip().lower()

        if valg == "b":
            return

        if valg == "1":
            reset_time_scope(state)
            print("✅ Hele perioden er valgt.")
            return

        if valg == "2":
            år = prompt_year()
            if år is None:
                continue
            reset_time_scope(state)
            state["mode"] = "year"
            state["year"] = år
            print(f"✅ År {år} er valgt.")
            return

        if valg == "3":
            år = prompt_year()
            if år is None:
                continue
            reset_time_scope(state)
            state["mode"] = "fall"
            state["year"] = år
            print(f"✅ Høst {år} er valgt.")
            return

        if valg == "4":
            år = prompt_year()
            if år is None:
                continue
            reset_time_scope(state)
            state["mode"] = "spring"
            state["year"] = år
            print(f"✅ Vår {år} er valgt.")
            return

        if valg == "5":
            år = prompt_year()
            if år is None:
                continue
            måned = prompt_month()
            if måned is None:
                continue
            reset_time_scope(state)
            state["mode"] = "month"
            state["year"] = år
            state["month"] = måned
            print(f"✅ Måned {måned}/{år} er valgt.")
            return

        if valg == "6":
            år = prompt_year()
            if år is None:
                continue
            uke = prompt_week()
            if uke is None:
                continue
            reset_time_scope(state)
            state["mode"] = "week"
            state["year"] = år
            state["week"] = uke
            print(f"✅ Uke {uke} i {år} er valgt.")
            return

        if valg == "7":
            dag = prompt_day()
            if dag is None:
                continue
            reset_time_scope(state)
            state["mode"] = "day"
            state["year"] = dag.year
            state["month"] = dag.month
            state["week"] = int(dag.strftime("%W"))
            state["day"] = dag
            print(f"✅ Dag {dag.strftime('%Y-%m-%d')} er valgt.")
            return

        print("❌ Ugyldig valg.")


def configure_scope(state):
    while True:
        print("\nDATASETT / SCOPE")
        print("1. Endre bygg")
        print("2. Endre rom")
        print("3. Endre periode")
        print("4. Vis aktivt scope")
        print("b. Tilbake")

        valg = input("Valg: ").strip().lower()

        if valg == "1":
            select_building(state)
        elif valg == "2":
            select_rooms(state)
        elif valg == "3":
            select_time_scope(state)
        elif valg == "4":
            print_active_scope(state)
        elif valg == "b":
            return
        else:
            print("❌ Ugyldig valg.")



def room_is_selected(state, byggkode, rom):
    valgte_rom = state.get("rooms_by_building", {}).get(str(byggkode))
    if valgte_rom is None:
        return True
    return str(rom) in [str(r) for r in valgte_rom]


def select_rooms(state):
    for byggkode in state["buildings"]:
        while True:
            print(f"\nROMVALG FOR BYGG {byggkode}")
            print(f"Byggnavn: {TILGJENGELIGE_BYGG.get(byggkode, byggkode)}")

            try:
                _, romnavn, _ = fetch_csv(building_number=byggkode)
            except Exception as e:
                print(f"⚠️ Klarte ikke hente rom for bygg {byggkode}: {e}")
                state["rooms_by_building"][byggkode] = None
                break

            tilgjengelige_rom = sorted(set(str(r) for r in romnavn), key=int)

            if not tilgjengelige_rom:
                print("⚠️ Ingen rom funnet. Bruker alle rom.")
                state["rooms_by_building"][byggkode] = None
                break

            print("Tilgjengelige rom: " + ", ".join(tilgjengelige_rom))
            inp = input(
                "Oppgi ønskede rom som sifre uten skilletegn "
                "(f.eks. 13 for rom 1 og 3). Trykk Enter for alle rom: "
            ).strip()

            if inp == "":
                state["rooms_by_building"][byggkode] = None
                print(f"✅ Alle rom i bygg {byggkode} er valgt.")
                break

            valgte_rom = []
            ugyldige = []

            for tegn in inp:
                if tegn in tilgjengelige_rom:
                    if tegn not in valgte_rom:
                        valgte_rom.append(tegn)
                else:
                    ugyldige.append(tegn)

            if not valgte_rom:
                print("❌ Du må velge minst ett gyldig rom.")
                continue

            if ugyldige:
                print(f"❌ Ugyldige rom: {', '.join(ugyldige)}")
                continue

            state["rooms_by_building"][byggkode] = valgte_rom
            print(f"✅ Valgte rom i bygg {byggkode}: {', '.join(valgte_rom)}")
            break




def _collect_all_room_data(variable: str, state, rename_to_value: bool = True):

    all_dfs = []
    all_labels = []


    for bygg in state["buildings"]:
        try:
            dfs_bygg, romnavn, _ = fetch_csv(building_number=bygg)
        except Exception as e:
            print(f"⚠️ Klarte ikke hente data for bygg {bygg}: {e}")
            continue

        for df, rom in zip(dfs_bygg, romnavn):
            if not room_is_selected(state, bygg, rom):
                continue

            df2 = set_datetime_index(df)

            filtrerte_liste = filter_data(
                [df2],
                mode=state["mode"],
                year=state["year"],
                month=state["month"],
                week=state["week"],
                day=state["day"]
            )

            if not filtrerte_liste:
                continue

            df_filtrert = filtrerte_liste[0]

            if variable not in df_filtrert.columns:
                continue

            df3 = df_filtrert[[variable]].copy()
            if rename_to_value:
                df3.rename(columns={variable: "Value"}, inplace=True)

            all_dfs.append(df3)

            if len(state["buildings"]) == 1:
                all_labels.append(f"R{rom}")
            else:
                all_labels.append(f"B{bygg}-R{rom}")
    return all_dfs, all_labels


def _spør_om_uteklima_sammenligning() -> bool:
    while True:
        valg = input("Vil du sammenligne med uteklima? (j/n): ").strip().lower()

        if valg in {"j", "ja"}:
            return True
        if valg in {"n", "nei"}:
            return False

        print("❌ Ugyldig valg. Skriv 'j' eller 'n'.")


def run_time_series(state):
    while True:
        print("\n📊 VELG VARIABEL FOR TIDSSERIEPLOTT")
        for key, var in VARIABLE_CHOICES.items():
            print(f"{key}. {var}")
        print("b. Tilbake til forrige meny")

        valg = input("Valg: ").strip().lower()
        if valg == 'b':
            return

        if valg not in VARIABLE_CHOICES:
            print("❌ Ugyldig valg. Prøv igjen.")
            continue

        variable = VARIABLE_CHOICES[valg]
        print(f"👉 Du har valgt: {variable}")

        compare_weather = False
        if variable in {"Temperatur (°C)", "Luftfuktighet (%)"}:
            compare_weather = _spør_om_uteklima_sammenligning()

        dfs, rom_labels = _collect_all_room_data(
            variable,
            state,
            rename_to_value=not compare_weather
        )

        if not dfs:
            print(f"❌ Ingen data funnet for variabelen '{variable}' i valgt datasett.")
            continue

        if compare_weather:
            try:
                weather_df = fetch_weather()
            except Exception as e:
                print(f"❌ Klarte ikke hente uteklima: {e}")
                continue

            title_subject = (
                f"Bygg {state['buildings'][0]}"
                if len(state["buildings"]) == 1
                else "Valgte bygg"
            )

            if variable == "Temperatur (°C)":
                plot_temperature(
                    dfs,
                    mode=state["mode"],
                    year=state["year"],
                    month=state["month"],
                    week=state["week"],
                    day=state["day"],
                    df_weather=weather_df,
                    romnavn=rom_labels,
                    title_subject=title_subject
                )
            else:
                plot_humidity(
                    dfs,
                    mode=state["mode"],
                    year=state["year"],
                    month=state["month"],
                    week=state["week"],
                    day=state["day"],
                    df_weather=weather_df,
                    romnavn=rom_labels,
                    title_subject=title_subject
                )
        else:
            title_subject = (
                f"Bygg {state['buildings'][0]}"
                if len(state["buildings"]) == 1
                else "Valgte bygg"
            )

            plot_all_rooms_variable(
                dfs,
                variable,
                mode=state["mode"],
                year=state["year"],
                month=state["month"],
                week=state["week"],
                day=state["day"],
                title_subject=title_subject,
            )



def run_distribution(state):
    """
    Hovedmeny for å plotte fordeling (histogram) for én variabel
    på tvers av valgte bygg og valgt periode i state.
    """
    while True:
        print("\n📊 VELG VARIABEL FOR FORDELINGSPLOTT")
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
        print(f"\n👉 Du har valgt: {variable}")

        serie = _collect_all_values_for_variable(variable, state)
        if serie is None or serie.empty:
            print(f"❌ Ingen data funnet for variabelen '{variable}' i valgt datasett.")
            continue

        _plot_distribution(serie, variable)


def _collect_all_values_for_variable(variable: str, state) -> pd.Series:
    samling = []
    for bygg in state["buildings"]:
        try:
            dfs_bygg, romnavn, _ = fetch_csv(building_number=bygg)
        except Exception as e:
            print(f"Kunne ikke hente data for bygg {bygg}: {e}")
            continue

        for df, rom in zip(dfs_bygg, romnavn):
            if not room_is_selected(state, bygg, rom):
                continue

            df2 = set_datetime_index(df)

            filtrerte_liste = filter_data(
                [df2],
                mode=state["mode"],
                year=state["year"],
                month=state["month"],
                week=state["week"],
                day=state["day"]
            )

            if not filtrerte_liste:
                continue

            df_filtrert = filtrerte_liste[0]

            if variable not in df_filtrert.columns:
                continue

            serie = df_filtrert[variable].dropna()
            if serie.empty:
                continue

            samling.append(serie)

    if not samling:
        return pd.Series(dtype="float64")

    samlet = pd.concat(samling, axis=0)
    return samlet


def _plot_distribution(serie: pd.Series, variable: str):
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

    weights = np.ones_like(data) * 100 / len(data)

    ax.hist(
        data,
        bins=bins,
        weights=weights,
        color="skyblue",
        edgecolor="black"
    )

    ax.set_ylabel("Frekvens (%)")
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


def run_data_availability(state):
    plot_room_data_availability(state)


def prompt_semester_scope(state):
    semester_state = state.copy()

    while True:
        print("\nVELG SEMESTER")
        print("1. Høst")
        print("2. Vår")
        print("b. Tilbake")

        valg = input("Velg semester: ").strip().lower()

        if valg == "b":
            return None

        if valg not in {"1", "2"}:
            print("❌ Ugyldig valg. Prøv igjen.")
            continue

        år = prompt_year()
        if år is None:
            continue

        reset_time_scope(semester_state)
        semester_state["mode"] = "fall" if valg == "1" else "spring"
        semester_state["year"] = år

        return semester_state


def get_semester_threshold_rule(variable: str):
    if variable == "Temperatur (°C)":
        return {
            "lower": THRESHOLDS_TEMPERATURE["night"]["min"],
            "upper": THRESHOLDS_TEMPERATURE["day"]["max"],
            "lower_label": "For kaldt (%)",
            "upper_label": "For varmt (%)",
            "lower_pass_label": "Oppfyller kald-kravet",
            "upper_pass_label": "Oppfyller varm-kravet",
            "two_sided": True,
        }

    if variable == "Luftfuktighet (%)":
        limits = THRESHOLDS_OPTIMAL_HUMIDITY["Humidity (%)"]
        return {
            "lower": limits["critical_min"],
            "upper": limits["critical_max"],
            "lower_label": "For tørt (%)",
            "upper_label": "For fuktig (%)",
            "lower_pass_label": "Oppfyller tørr-kravet",
            "upper_pass_label": "Oppfyller fuktig-kravet",
            "two_sided": True,
        }

    upper = THRESHOLDS_CRITICAL.get(variable)
    if upper is not None:
        return {
            "lower": None,
            "upper": upper,
            "upper_label": "Over grense (%)",
            "upper_pass_label": "Oppfyller 5 %-kravet",
            "two_sided": False,
        }

    return None


def median_exceedance(series: pd.Series, threshold: float, direction: str) -> float:
    if direction == "below":
        exceedance = (threshold - series[series < threshold]).dropna()
    elif direction == "above":
        exceedance = (series[series > threshold] - threshold).dropna()
    else:
        raise ValueError(f"Ukjent retning: {direction}")

    if exceedance.empty:
        return 0.0

    return round(float(exceedance.median()), 2)


def longest_breach_duration_hours(mask: pd.Series) -> int:
    if mask.empty or not mask.any():
        return 0

    mask = mask.astype(bool)
    change_groups = (mask != mask.shift()).cumsum()
    run_lengths = mask.groupby(change_groups).sum()

    if run_lengths.empty:
        return 0

    return int(run_lengths.max())

def max_exceedance(series: pd.Series, threshold: float, direction: str) -> float:
    if direction == "below":
        exceedance = (threshold - series[series < threshold]).dropna()
    elif direction == "above":
        exceedance = (series[series > threshold] - threshold).dropna()
    else:
        raise ValueError(f"Ukjent retning: {direction}")

    if exceedance.empty:
        return 0.0

    return round(float(exceedance.max()), 2)


def collect_semester_summary_by_building(variable: str, state, max_outside_pct: float = 5.0) -> pd.DataFrame:
    rows = []
    rule = get_semester_threshold_rule(variable)

    if rule is None:
        return pd.DataFrame()

    lower_limit = rule["lower"]
    upper_limit = rule["upper"]

    if lower_limit is None and upper_limit is None:
        return pd.DataFrame()

    for byggkode in state["buildings"]:
        try:
            dfs, romnavn, _ = fetch_csv(building_number=byggkode)
        except Exception as e:
            print(f"⚠️ Klarte ikke hente data for bygg {byggkode}: {e}")
            continue

        building_series: list[pd.Series] = []
        room_stats = []

        for df, rom in zip(dfs, romnavn):
            if not room_is_selected(state, byggkode, rom):
                continue

            df = set_datetime_index(df)

            filtrerte_liste = filter_data(
                [df],
                mode=state["mode"],
                year=state["year"],
                month=state["month"],
                week=state["week"],
                day=state["day"]
            )

            if not filtrerte_liste:
                continue

            df_filtrert = filtrerte_liste[0]

            if variable not in df_filtrert.columns:
                continue

            serie = pd.Series(
                pd.to_numeric(df_filtrert[variable], errors="coerce"),
                index=df_filtrert.index
            ).dropna()

            if serie.empty:
                continue

            building_series.append(serie)

            room_below_mask = pd.Series(False, index=serie.index)
            room_above_mask = pd.Series(False, index=serie.index)

            if lower_limit is not None:
                room_below_mask = serie < lower_limit
            if upper_limit is not None:
                room_above_mask = serie > upper_limit

            room_outside_pct = 100 * (room_below_mask | room_above_mask).mean()

            room_stats.append({
                "room": str(rom),
                "outside_pct": room_outside_pct
            })

        if not building_series:
            continue

        samlet_serie = pd.Series(pd.concat(building_series, axis=0)).dropna()
        if samlet_serie.empty:
            continue

        below_mask = pd.Series(False, index=samlet_serie.index)
        above_mask = pd.Series(False, index=samlet_serie.index)

        if lower_limit is not None:
            below_mask = samlet_serie < lower_limit

        if upper_limit is not None:
            above_mask = samlet_serie > upper_limit

        below_pct = 100 * below_mask.mean()
        above_pct = 100 * above_mask.mean()

        below_median_exceedance = 0.0
        above_median_exceedance = 0.0
        below_longest_hours = 0
        above_longest_hours = 0
        above_max_exceedance = 0.0

        if lower_limit is not None:
            below_median_exceedance = median_exceedance(samlet_serie, lower_limit, "below")
            below_longest_hours = longest_breach_duration_hours(below_mask)

        if upper_limit is not None:
            above_median_exceedance = median_exceedance(samlet_serie, upper_limit, "above")
            above_longest_hours = longest_breach_duration_hours(above_mask)
            above_max_exceedance = max_exceedance(samlet_serie, upper_limit, "above")

        worst_room = "-"
        if room_stats:
            worst_room = max(room_stats, key=lambda x: x["outside_pct"])["room"]

        row = {
            "Bygg": f"Bygg {int(byggkode)}",
        }

        if rule["two_sided"]:
            row[rule["lower_label"]] = round(below_pct, 2)
            row[rule["upper_label"]] = round(above_pct, 2)

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

            row[rule["lower_pass_label"]] = "Ja" if below_pct < max_outside_pct else "Nei"
            row[rule["upper_pass_label"]] = "Ja" if above_pct < max_outside_pct else "Nei"

        else:
            row[rule["upper_label"]] = round(above_pct, 2)
            row["Median overskridelse"] = above_median_exceedance
            row["Maks overskridelse"] = above_max_exceedance
            row["Lengste brudd (t)"] = above_longest_hours
            row[rule["upper_pass_label"]] = "Ja" if above_pct < max_outside_pct else "Nei"

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




def run_semester_analysis(state):
    semester_state = prompt_semester_scope(state)
    if semester_state is None:
        return

    print("\nSEMESTERANALYSE PER BYGG")
    print("Kriterium: mindre enn 5 % av målingene utenfor grenseverdiene")
    print(f"Periode: {format_scope_label(semester_state)}")

    for variable in VARIABLE_CHOICES.values():
        summary_df = collect_semester_summary_by_building(
            variable,
            semester_state,
            max_outside_pct=5.0
        )

        print(f"\n{variable}")
        print("-" * len(variable))

        if summary_df.empty:
            print("Ingen data funnet for denne variabelen.")
            continue

        with pd.option_context("display.max_columns", None, "display.width", 1400):
            print(summary_df.to_string(index=False))
            print()


