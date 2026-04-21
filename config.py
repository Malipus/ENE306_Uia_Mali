from pathlib import Path
# ------------------------------------------------------------------
# Grunnkonfigurasjon for filbaner
# ------------------------------------------------------------------

# Base‐mappe der alle CSV‐filer (inneklima, Seklima, Kunak) ligger:
BASE_DATA_DIR = Path(__file__).parent / "CSV"
INNEKLIMA_DIR = BASE_DATA_DIR      # Underkatalog for inneklima‐sensorer


# Faste baner for utendørsdata:
SEKLIMA_FILE = BASE_DATA_DIR / "Seklima.csv"
KUNAK_FILE   = BASE_DATA_DIR / "Kunak.csv"

# Prefix‐mal for inneklima‐filer:
# - Filnavn ser ut som: data_RES08xxx.csv for Bygg 08
# - Vi bruker mal: "data_RES{bygg}01" der {bygg} er to sifre


INNEKLIMA_PREFIX_TEMPLATE = "data_RES0{bygg}01"


TILGJENGELIGE_BYGG = {
    '1': 'Tønnevoldsgate 26, Sentrum',
    '2': 'Jon Lilletuns Vei 2A, Campus',
    '4': 'Jon Lilletuns Vei 15, Campus',
    '5': 'Jon Lilletuns Vei 17, Campus',
    '7': 'Jon Lilletuns Vei 21, Campus',
    '8': 'Jon Lilletuns Vei 23, Campus'
}


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

NORWEGIAN_MONTHS = {
    1: "januar",
    2: "februar",
    3: "mars",
    4: "april",
    5: "mai",
    6: "juni",
    7: "juli",
    8: "august",
    9: "september",
    10: "oktober",
    11: "november",
    12: "desember",
}


PM_VARIABLER = [
    "PM 1.0 (µg/m³)",
    "PM 2.5 (µg/m³)",
    "PM 4.0 (µg/m³)",
    "PM 10 (µg/m³)"
]


THRESHOLDS_TEMPERATURE = {
    "day":   {"min": 21, "max": 26},
    "night": {"min": 18, "max": 21},
    "night_hours": (22, 7)
}


THRESHOLDS_OPTIMAL_HUMIDITY = {
    "Humidity (%)": {
        "optimal_min": 35,
        "optimal_max": 60,
        "critical_min": 25,
        "critical_max": 70
    }
}


THRESHOLDS_WARN = {
    "CO2 (ppm)":              800,
    "Formaldehyd (µg/m³)":    70,
    "TVOC (ppb)":             204,
    "PM 1.0 (µg/m³)":         12,
    "PM 2.5 (µg/m³)":         12,
    "PM 4.0 (µg/m³)":         12,
    "PM 10 (µg/m³)":          50
}


THRESHOLDS_CRITICAL = {
    "CO2 (ppm)":              1500,
    "Formaldehyd (µg/m³)":    120,
    "TVOC (ppb)":             621,
    "PM 1.0 (µg/m³)":         35,
    "PM 2.5 (µg/m³)":         35,
    "PM 4.0 (µg/m³)":         35,
    "PM 10 (µg/m³)":          100
}


