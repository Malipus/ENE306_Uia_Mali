import sys
import analysis

from config import TILGJENGELIGE_BYGG

app_state = {
    "buildings": list(TILGJENGELIGE_BYGG.keys()),
    "rooms_by_building": {},
    "mode": "all",
    "year": None,
    "month": None,
    "week": None,
    "day": None,
}


def print_main_menu():
    print("\nHOVEDMENY")
    print("1. Velg eller endre datasett")
    print("2. Vis aktivt datasett")
    print("3. Fordeling (histogram)")
    print("4. Tidsserie")
    print("5. Boxplot")
    print("6. Datadekning per rom")
    print("7. Grensebrudd (scatter)")
    print("b. Avslutt")


def main():
    while True:
        print_main_menu()
        valg = input("Velg et alternativ: ").strip().lower()
        if valg == "1":
            analysis.configure_scope(app_state)
        elif valg == "2":
            analysis.print_active_scope(app_state)
        elif valg == "3":
            analysis.run_distribution(app_state)
        elif valg == "4":
            analysis.run_time_series(app_state)
        elif valg == "5":
            analysis.run_boxplot_menu(app_state)
        elif valg == "6":
            analysis.run_data_availability(app_state)
        elif valg == "7":
            analysis.run_threshold_scatter_menu(app_state)
        elif valg == "b":
            print("Avslutter...")
            sys.exit(0)

        else:
            print("❌ Ugyldig valg. Prøv igjen.")

if __name__ == "__main__":
    main()