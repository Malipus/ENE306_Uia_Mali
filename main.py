"""
Terminalmeny for ENE306-analyseverktøyet.

main.py skal være brukergrensesnittet. Den viser menyvalg, oppdaterer aktivt
utvalg og sender analyseoppgaver videre til analysis.py.
"""

from __future__ import annotations

import re
import sys
from typing import Iterable, List

import analysis
from config import PM_VARIABLER, TILGJENGELIGE_BYGG, VARIABLE_CHOICES


def print_header(title: str) -> None:
    """Skriv en tydelig seksjonsoverskrift."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def pause() -> None:
    """La brukeren lese utskriften før menyen tegnes på nytt."""
    input("\nTrykk Enter for å gå tilbake til menyen.")


def parse_digit_selection(raw_text: str, allowed: Iterable[str]) -> List[str]:
    """Finn gyldige siffer i brukerens tekst.

    Brukeren kan skrive 125 eller 1,2,5. Begge deler tolkes likt.
    """
    allowed_set = set(allowed)
    selected = []

    for digit in re.findall(r"\d", raw_text):
        if digit in allowed_set and digit not in selected:
            selected.append(digit)

    return selected


def ask_year() -> int | None:
    """Spør etter årstall."""
    raw_year = input("Skriv årstall, for eksempel 2024: ").strip()
    if raw_year.lower() == "b":
        return None
    if raw_year.isdigit() and len(raw_year) == 4:
        return int(raw_year)
    print("Ugyldig årstall.")
    return None


def print_active_scope(scope: analysis.Scope) -> None:
    """Vis aktivt utvalg."""
    print("\nAktivt utvalg")
    print(analysis.describe_scope(scope))


def choose_buildings(scope: analysis.Scope) -> None:
    """Velg bygg som skal inngå i analysene."""
    print_header("Velg bygg")
    print("a. Alle bygg")
    for building, name in TILGJENGELIGE_BYGG.items():
        print(f"{building}. Bygg {building} - {name}")
    print("b. Tilbake")

    choice = input("\nSkriv byggnummer, for eksempel 125 eller 1,2,5: ").strip().lower()
    if choice == "b":
        return
    if choice == "a":
        scope.buildings = list(TILGJENGELIGE_BYGG.keys())
        scope.rooms_by_building.clear()
        print("Alle bygg er valgt.")
        return

    selected = parse_digit_selection(choice, TILGJENGELIGE_BYGG.keys())
    if not selected:
        print("Ingen gyldige bygg ble valgt.")
        return

    scope.buildings = selected
    scope.rooms_by_building = {
        building: rooms
        for building, rooms in scope.rooms_by_building.items()
        if building in selected
    }
    print("Valgte bygg:", ", ".join(f"Bygg {int(building)}" for building in selected))


def choose_rooms(scope: analysis.Scope) -> None:
    """Velg rom for hvert aktivt bygg."""
    print_header("Velg rom")
    print("Trykk Enter for alle rom i et bygg.")

    for building in scope.buildings:
        rooms = analysis.get_available_rooms(building)
        if not rooms:
            print(f"Bygg {building}: fant ingen rom.")
            continue

        print(f"\nBygg {int(building)} har rom: {', '.join(rooms)}")
        choice = input("Skriv romnummer, eller Enter for alle rom: ").strip()

        if not choice:
            scope.rooms_by_building.pop(str(building), None)
            continue

        selected_rooms = parse_digit_selection(choice, rooms)
        if selected_rooms:
            scope.rooms_by_building[str(building)] = selected_rooms
            print(f"Bygg {int(building)}: valgte rom {', '.join(selected_rooms)}")
        else:
            print(f"Bygg {int(building)}: ingen gyldige rom valgt, bruker alle rom.")
            scope.rooms_by_building.pop(str(building), None)


def choose_period(scope: analysis.Scope) -> None:
    """Velg tidsperiode. Nærmeste zoom-nivå i menyen er uke."""
    print_header("Velg periode")
    print("1. Hele måleperioden")
    print("2. År")
    print("3. Vår-semester")
    print("4. Høst-semester")
    print("5. Måned")
    print("6. Uke")
    print("b. Tilbake")

    choice = input("\nVelg periode: ").strip().lower()
    if choice == "b":
        return

    if choice == "1":
        scope.mode = "all"
        scope.year = scope.month = scope.week = None
        scope.day = None
        print("Valgt periode: hele måleperioden.")
        return

    year = ask_year()
    if year is None:
        return

    scope.year = year
    scope.month = None
    scope.week = None
    scope.day = None

    if choice == "2":
        scope.mode = "year"
    elif choice == "3":
        scope.mode = "spring"
    elif choice == "4":
        scope.mode = "fall"
    elif choice == "5":
        raw_month = input("Skriv måned 1-12: ").strip()
        if not raw_month.isdigit() or not 1 <= int(raw_month) <= 12:
            print("Ugyldig måned.")
            return
        scope.mode = "month"
        scope.month = int(raw_month)
    elif choice == "6":
        raw_week = input("Skriv ukenummer 1-53: ").strip()
        if not raw_week.isdigit() or not 1 <= int(raw_week) <= 53:
            print("Ugyldig uke.")
            return
        scope.mode = "week"
        scope.week = int(raw_week)
    else:
        print("Ugyldig valg.")
        return

    print(f"Valgt periode: {analysis.format_scope_label(scope)}")


def configure_scope(scope: analysis.Scope) -> None:
    """Undermeny for aktivt utvalg."""
    while True:
        print_header("Utvalg")
        print_active_scope(scope)
        print("\n1. Velg bygg")
        print("2. Velg rom")
        print("3. Velg periode")
        print("b. Tilbake")

        choice = input("\nVelg handling: ").strip().lower()
        if choice == "1":
            choose_buildings(scope)
        elif choice == "2":
            choose_rooms(scope)
        elif choice == "3":
            choose_period(scope)
        elif choice == "b":
            return
        else:
            print("Ugyldig valg.")


def choose_variable(allow_pm_group: bool = False) -> str | None:
    """Velg målevariabel.

    Når allow_pm_group=True vises luftpartikkelvariablene som ett samlet valg.
    Det gjør at PM1, PM2.5, PM4 og PM10 alltid tegnes som subplots i samme
    figur, slik at rapportfigurene blir sammenlignbare og mer ryddige.
    """
    print("\nVelg variabel:")
    for key, variable in VARIABLE_CHOICES.items():
        if allow_pm_group and variable in PM_VARIABLER:
            continue
        print(f"{key}. {variable}")
    if allow_pm_group:
        print("6. Alle luftpartikkelvariabler (PM1, PM2.5, PM4 og PM10)")
    print("b. Tilbake")

    choice = input("Valg: ").strip().lower()
    if choice == "b":
        return None
    if allow_pm_group and choice in {"6", "7", "8", "9", "pm", "p", "partikler", "luftpartikler"}:
        return "pm"

    try:
        return analysis.resolve_variable_choice(choice)
    except ValueError as error:
        print(error)
        return None


def ask_weather_comparison(variable: str) -> bool:
    """Spør om uteklima bare for temperatur og luftfuktighet."""
    if variable not in {"Temperatur (°C)", "Luftfuktighet (%)"}:
        return False

    choice = input("Vil du sammenligne med uteklima? (j/n): ").strip().lower()
    return choice in {"j", "ja"}


def run_time_series_menu(scope: analysis.Scope) -> None:
    """Kjør tidsserie fra menyen."""
    print("\nLuftpartikkelvariabler kan vises samlet som PM-subplots.")
    variable = choose_variable(allow_pm_group=True)
    if variable is None:
        return

    compare_weather = ask_weather_comparison(variable)
    saved_path = analysis.run_time_series(scope, variable, compare_weather=compare_weather)
    print(f"\nFigur lagret: {saved_path}")


def run_boxplot_menu(scope: analysis.Scope) -> None:
    """Kjør boxplot fra menyen."""
    print("\nLuftpartikkelvariabler:", ", ".join(PM_VARIABLER))
    variable = choose_variable(allow_pm_group=True)
    if variable is None:
        return

    saved_path = analysis.run_boxplot(scope, variable)
    print(f"\nFigur lagret: {saved_path}")


def run_data_availability_menu(scope: analysis.Scope) -> None:
    """Kjør datadekningsfigur fra menyen."""
    saved_path = analysis.run_data_availability(scope)
    print(f"\nFigur lagret: {saved_path}")



def choose_semester() -> tuple[str, int] | None:
    """Velg vår- eller høstsemester for samlet rapportkjøring."""
    print_header("Kjør semesterpakke")
    print("1. Vårsemester")
    print("2. Høstsemester")
    print("b. Tilbake")

    choice = input("\nVelg semester: ").strip().lower()
    if choice == "b":
        return None
    if choice not in {"1", "2"}:
        print("Ugyldig valg.")
        return None

    year = ask_year()
    if year is None:
        return None

    semester = "spring" if choice == "1" else "fall"
    return semester, year


def run_semester_package_menu(scope: analysis.Scope) -> None:
    """Kjør alle rapportfigurene for valgt semester."""
    semester_choice = choose_semester()
    if semester_choice is None:
        return

    semester, year = semester_choice
    saved_paths = analysis.run_semester_package(scope, semester, year, show=False)

    print("\nLagrede filer:")
    for path in saved_paths:
        print(f"- {path}")

def run_safely(action, pause_after: bool = True) -> bool:
    """Kjør en menyhandling med ryddig feilmelding."""
    completed = False
    try:
        action()
        completed = True
    except FileNotFoundError as error:
        print(f"Fant ikke fil eller mappe: {error}")
    except ValueError as error:
        print(f"Analysen kunne ikke gjennomføres: {error}")
    except RuntimeError as error:
        print(f"Datafeil: {error}")
    except Exception as error:
        print(f"Uventet feil: {error}")
    finally:
        if pause_after:
            pause()

    return completed


def print_main_menu(scope: analysis.Scope) -> None:
    """Vis hovedmenyen."""
    print_header("ENE306 analyseverktøy")
    print_active_scope(scope)
    print("\n1. Velg eller endre utvalg")
    print("2. Tidsseriefigur")
    print("3. Boxplot med tabellverdier")
    print("4. Datadekning per rom")
    print("5. Kjør alt for et semester")
    print("b. Avslutt")


def main() -> None:
    """Start programmet."""
    scope = analysis.make_default_scope()

    while True:
        print_main_menu(scope)
        choice = input("\nVelg handling: ").strip().lower()

        if choice == "1":
            configure_scope(scope)
        elif choice == "2":
            run_safely(lambda: run_time_series_menu(scope))
        elif choice == "3":
            run_safely(lambda: run_boxplot_menu(scope))
        elif choice == "4":
            run_safely(lambda: run_data_availability_menu(scope))
        elif choice == "5":
            completed = run_safely(lambda: run_semester_package_menu(scope), pause_after=False)
            if completed:
                print("Semesterkjøringen er ferdig. Avslutter programmet.")
                sys.exit(0)
            pause()
        elif choice == "b":
            print("Avslutter.")
            sys.exit(0)
        else:
            print("Ugyldig valg.")


if __name__ == "__main__":
    main()
