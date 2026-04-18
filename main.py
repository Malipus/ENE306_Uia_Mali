import sys
import building_analysis
import analysis
import plotting


def vis_hovedmeny():
    print("\n📋 HOVEDMENY")
    print("1. Fordeling (histogram) for én variabel (alle bygg/rom)")
    print("2. Bygg-analyse")
    print("3. Boxplot")
    print("4. Tidsserie over én variabel (alle bygg/rom)")
    print("5. Datadekning per rom")
    print("b. Avslutt")



def main():

    while True:
        vis_hovedmeny()
        valg = input("Velg et alternativ: ").strip().lower()

        if valg == "1":
            analysis.run_distribution()
        elif valg == "2":
            analysis.run_time_series()
        elif valg == "3":
            building_analysis.run_boxplot_alle_bygg()
        elif valg == "4":
            building_analysis.run_building_analysis()
        elif valg == "5":
            plotting.vis_datadekning_per_rom()
        elif valg == "b":
            print("Avslutter...")
            sys.exit(0)

        else:
            print("❌ Ugyldig valg. Prøv igjen.")

if __name__ == "__main__":
    main()