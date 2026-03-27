import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

path = r"C:\Users\MaliR\Desktop\Pycharm\Pycharm\CSV"
kolonner = []

# Date;Time;Temperature (°C);Humidity (%);CO2 (ppm);Formaldehyde (µg/m³);TVOC (ppb);
# PM 1.0 (µg/m³);PM 4.0 (µg/m³);PM 10 (µg/m³);PM 2.5 (µg/m³)


variabel = 'Humidity (%)'

for file in os.scandir(path):
    if file.is_file() and file.name.startswith("data_") and file.name.endswith(".csv"): #Henter filen, inbiot
        df = pd.read_csv(file.path, sep=";") # Leser av filen
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce") # slå sammen Date og Time



        # hent ønsket målekolonne
        midlertidig = df[["datetime", variabel]].copy()
        midlertidig[variabel] = pd.to_numeric(midlertidig[variabel], errors="coerce")
        midlertidig = midlertidig.set_index("datetime").sort_index()        # sett tid som index
        midlertidig = midlertidig.resample("30min").mean()      # resample

        # lag kolonnenavn fra filnavn
        filnavn = file.name
        res_del = filnavn.split("RES")[1].split("_")[0]

        bygg = res_del[1]
        rom = res_del[4]

        kolonnenavn = f"B0{bygg}-R{rom}"

        # gjør om til serie med riktig navn
        serie = midlertidig[variabel]
        serie.name = kolonnenavn

        kolonner.append(serie)

samlet_df = pd.concat(kolonner, axis=1)

print(samlet_df.head())

x = samlet_df["B01-R1"].dropna()

print("N =", len(x))
print("Skewness:", stats.skew(x))
print("Kurtosis:", stats.kurtosis(x))

# Shapiro på tilfeldig utvalg for normalfordeling
sample = x.sample(n=200, random_state=1)
stat, p = stats.shapiro(sample)

print("Shapiro W:", stat)
print("p-value:", p)

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(x, bins=30, edgecolor="black")
plt.title("Histogram")
plt.xlabel(variabel)
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.show()

# Relevante fordelinger for fukt
fordelinger = {
    "Normal": stats.norm,
    "Logistic": stats.logistic,
    "Gamma": stats.gamma
}

# QQ-plot for hver fordeling
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (navn, dist) in zip(axes, fordelinger.items()):
    params = dist.fit(x)

    # observerte kvantiler
    x_sort = np.sort(x)

    # teoretiske kvantiler
    p = (np.arange(1, len(x_sort) + 1) - 0.5) / len(x_sort)
    q_teo = dist.ppf(p, *params)

    # fjern eventuelle ugyldige verdier
    mask = np.isfinite(q_teo) & np.isfinite(x_sort)
    q_teo = q_teo[mask]
    x_obs = x_sort[mask]

    # plott
    ax.scatter(q_teo, x_obs, s=10)
    minv = min(q_teo.min(), x_obs.min())
    maxv = max(q_teo.max(), x_obs.max())
    ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)

    ax.set_title(f"QQ-plot mot {navn}")
    ax.set_xlabel("Teoretiske kvantiler")
    ax.set_ylabel("Observerte kvantiler")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()