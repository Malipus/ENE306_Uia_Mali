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
INNEKLIMA_PREFIX_TEMPLATE = "data_RES{bygg}01"


# ------------------------------------------------------------------
# Terskler og variabelnavn
# ------------------------------------------------------------------

# Oversettelser mellom KPI‐koder og fulle kolonnenavn (hvis ønskelig):
VARIABLES = {
    "vi":   "Ventilation Indicator",
    "iaq":  "IAQ Indicator",
    "ti":   "thermalIndicator",
    "tmp":  "Temperature (°C)",
    "hu":   "Humidity (%)",
    "co2":  "CO2 (ppm)",
    "hcho": "Formaldehyde (µg/m³)",
    "tvoc": "TVOC (ppb)",
    "pm1":  "PM 1.0 (µg/m³)",
    "pm2_5":"PM 2.5 (µg/m³)",
    "pm4.0":"PM 4.0 (µg/m³)",
    "pm10": "PM 10 (µg/m³)"
}

# Temperatur‐terskler (dag/natt)
THRESHOLDS_TEMPERATURE = {
    "day":   {"min": 21, "max": 26},
    "night": {"min": 18, "max": 21},
    "night_hours": (22, 7)
}

# Luftfuktighet – optimale og kritiske grenser
THRESHOLDS_OPTIMAL_HUMIDITY = {
    "Humidity (%)": {
        "optimal_min": 35,
        "optimal_max": 60,
        "critical_min": 25,
        "critical_max": 70
    }
}

# Varsels‐ og kritiske grenser for luftkvalitetsindikatorer
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

# Skreddersydd rekkefølge for luftkvalitetsvariabler
LUFTKVALITETS_VARIABLER_I_REKKE = [
    "CO2 (ppm)",
    "Formaldehyd (µg/m³)",
    "TVOC (ppb)",
    "PM 1.0 (µg/m³)",
    "PM 2.5 (µg/m³)",
    "PM 4.0 (µg/m³)",
    "PM 10 (µg/m³)"
]
