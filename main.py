import sys
import building_analysis
import analysis
import plotting

from data_processing import bygg_cache

def vis_hovedmeny():
    print("\n📋 HOVEDMENY")
    print("1. Utforsk data")
    print("2. Bygg-analyse")
    print("3. Datadekning per rom")
    print("4. Dekningsgrad alle bygg")
    print("5. Spredningsmål")
    print("6. Timer over terskel")
    print("7. Prosent over terskel global")
    print("b. Avslutt")

def run_data_exploration():
    while True:
        print("\n🔎 UTFORSK DATA")
        print("1. Tidsserie over én variabel (alle bygg/rom)")
        print("2. Fordeling (histogram) for én variabel (alle bygg/rom)")
        print("b. Tilbake til hovedmeny")

        valg = input("Velg et alternativ: ").strip().lower()
        if valg == "1":
            analysis.run_time_series()
        elif valg == "2":
            analysis.run_distribution()
        elif valg == "b":
            return
        else:
            print("❌ Ugyldig valg. Prøv igjen.")


def main():
    # Cache bygg‐data (leser CSV/filer inn i minnet)
    bygg_cache()

    while True:
        vis_hovedmeny()
        valg = input("Velg et alternativ: ").strip().lower()

        if valg == "1":
            run_data_exploration()

        elif valg == "2":
            building_analysis.run_building_analysis()

        elif valg == "3":
            plotting.vis_datadekning_per_rom()

        elif valg == "4":
            plotting.vis_dekningsgrad_alle_bygg()

        elif valg == "5":
            analysis.vis_spredningsmål()

        elif valg == "6":
            analysis.run_timer_over_terskel()

        elif valg == "7":
            analysis.run_prosent_over_terskel_global()

        elif valg == "b":
            print("Avslutter...")
            sys.exit(0)

        else:
            print("❌ Ugyldig valg. Prøv igjen.")

if __name__ == "__main__":
    main()