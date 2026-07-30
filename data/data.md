# Data Documentation

## Overview

This document describes all data sources, processing steps, and output files used in the
Traffic-Aware Distribution Center Location project.

---

## Raw Data Sources (`data/raw/`)

### 1. Muhtarlık Adres Bilgileri (GeoJSON)
- **Source:** Istanbul Metropolitan Municipality (IBB) Open Data Portal
- **URL:** https://data.ibb.gov.tr/dataset/muhtarlik-adres-bilgileri/resource/71f75529-7fae-4a85-b05f-664c62eda422
- **Content:** Location records of all muhtarlık (neighborhood office) buildings in Istanbul
- **Fields used:** `Mahalle Adı` (neighborhood name), `İlçe Adı` (district name), `Latitude`, `Longtitude`
- **Coverage:** 963 muhtarlık records → 890 unique neighborhoods after aggregation
- **Role in model:** Provides neighborhood centroid coordinates for travel time calculation

### 2. pivot.csv
- **Source:** Turkish Statistical Institute (TUIK) — MEDAS Address-Based Population Registration System
- **URL:** https://biruni.tuik.gov.tr/medas/
- **Content:** Neighborhood-level population data for Istanbul, year 2025
- **Fields used:** Neighborhood name, district name, population count
- **Coverage:** 961 Istanbul neighborhoods
- **Role in model:** Demand weights `w_i` for each neighborhood

### 3. istanbul_tum_mahalleler.csv / istanbul_tum_ilceler.csv
- **Source:** Endeksa — Commercial Real Estate Indices
- **URL:** https://www.endeksa.com
- **Format:** semicolon-separated, values like `528 ₺/m 2` (parsed by `parse_rent`)
- **Content:**
  - `istanbul_tum_mahalleler.csv` — **neighborhood-level** rent per m² (965 rows).
    Fields used: `İlçe` (district), `Mahalle` (neighborhood), `Birim Fiyatı (₺/m2)`.
  - `istanbul_tum_ilceler.csv` — **district-level** rent per m² (38 rows), used as a
    fallback when a neighborhood has no entry. Note that its first column is named
    `Mahalle` in the export even though it holds district names.
- **Role in model:** Base rent values `r_j` used in the opening cost formula
  `f_j = α · r_j · √(Q/Q₀)`

### 4. IBB Hourly Traffic Density Data Set (fetched at runtime)
- **Source:** Istanbul Metropolitan Municipality Open Data Portal — CKAN API
- **Dataset:** "Hourly Traffic Density Data Set" — one CSV resource per month (2020–2025)
- **Resource used:** `d291989c-429d-4e61-9c70-1f76294b96b8` ("Ekim 2024 Trafik Yoğunluk Verisi")
- **Content:** Hourly records per geohash cell — `DATE_TIME`, `LATITUDE`, `LONGITUDE`, `GEOHASH`,
  `MINIMUM_SPEED`, `MAXIMUM_SPEED`, `AVERAGE_SPEED`, `NUMBER_OF_VEHICLES` (~1.76M rows/month)
- **Authentication:** None required
- **Role in model:** Provides a one-month average hourly speed profile per location, from which
  peak / off-peak / blended travel-time matrices are built (spatial mapping to neighborhoods)

> **Note:** An earlier version used the daily "Istanbul Traffic Index" resource
> (`ba47eacb-…`) with min/max as peak/off-peak proxies. Per supervisor feedback, traffic must be
> a **one-month average of hourly data** (each hour = average over all days in the month), not a
> single-day snapshot — hence the switch to the hourly density dataset above.

---

## Processing Steps (`data/prepare_data.py`)

### Step 1 — Neighborhood Coordinates
- Loaded the Muhtarlık GeoJSON (963 records)
- Multiple muhtarlık offices can share the same neighborhood; coordinates were averaged per unique (neighborhood, district) pair
- Result: **890 unique neighborhoods** with centroid coordinates

### Step 2 — Population Data
- Parsed `pivot.csv` using regex to extract neighborhood name, district, and population
- Applied Turkish-aware normalization: Python's `.upper()` maps `'i' → 'I'` instead of the correct `'İ'`, so `replace('i', 'İ')` is applied before `.upper()`
- Result: **961 neighborhoods** with population

### Step 3 — Merging Coordinates and Population
- Left-joined on normalized `(name, district)` key
- **5 neighborhoods** had mismatched names between IBB and TUIK — resolved with manual aliases:

| Muhtarlık name (IBB) | TUIK name | Reason |
|----------------------|-----------|--------|
| BAHÇEKÖY YENİMAHALLE | BAHÇEKÖY YENİ | TUIK regex stops before "Mah." suffix |
| KÜÇÜKÇAMLICA | KÜÇÜK ÇAMLICA | One word vs. two words |
| RUMELİ KAVAĞI | RUMELİKAVAĞI | Two words vs. one word |
| CUMHURİYET / BEYKOZ | CUMHURİYETKÖY | Different suffix |
| SARIYER YENİMAHALLE | SARIYER MERKEZ | Different name entirely |

- **6 neighborhoods** were not present in TUIK data at all — populations were sourced manually from TUIK 2025:

| Neighborhood | District | Population (2025) |
|--------------|----------|-------------------|
| BAĞLARÇEŞME | ESENYURT | 32,537 |
| HAVAALANI | ESENLER | 32,456 |
| MİMAR SİNAN | BÜYÜKÇEKMECE | 9,382 |
| SARAY | ÜMRANİYE | 3,460 |
| YENİŞEHİR | ÜMRANİYE | 6,584 |
| ÇİFTLİK | BEYKOZ | 5,750 |

- Final result: **890 neighborhoods** — all with coordinates and population

> **Coverage note:** Istanbul has approximately 963 official neighborhoods. The IBB Muhtarlık dataset
> contains 890 unique neighborhood locations; the remaining ~73 neighborhoods exist in TUIK population
> data but have no corresponding entry in the Muhtarlık JSON and were therefore excluded. This gives
> ~93% coverage of Istanbul's neighborhoods. Adding the missing coordinates (e.g., from a community
> GeoJSON boundary file) was deemed unnecessary for project purposes.

### Step 4 — Rent Data
- Loaded `istanbul_tum_mahalleler.csv` (neighborhood-level, 817 usable rows) and
  `istanbul_tum_ilceler.csv` (district-level, 39 districts) for rent data
- Mismatched neighborhood names were fixed with a dictionary of 19 aliases (`ALIASES_RENT`)
- Coverage after matching:

| Source | Neighborhoods | Share |
|--------|---------------|-------|
| Endeksa neighborhood-level rent | 741 | 83.3% |
| District average (fallback) | 149 | 16.7% |
| Citywide median, 385 TL/m² (last resort) | 0 | 0% |

- Resulting `rent_per_m2`: min 73, mean 425, max 1,328 TL/m²/month

### Step 5 — Hourly Speed Profiles (one-month average, spatial)
- Used one full month of **hourly** traffic density (Ekim 2024).
- For each hour of the day (0–23), the measured `AVERAGE_SPEED` is averaged over **all days in the
  month**, per geohash cell, via server-side SQL aggregation (one query per hour; `COUNT(*)` per
  group = days in month, confirming a true monthly average rather than a single day).
- This yields a 24-hour speed profile (km/h) for each of ~2,458 geohash cells across Istanbul.
- **Spatial mapping:** each neighborhood is matched to its `K=3` nearest geohash cells (Haversine),
  and their profiles are averaged. Neighborhoods farther than 5 km from any cell (≈70 remote/rural
  ones) fall back to the citywide hourly profile.
- Peak / off-peak windows are data-driven from the citywide profile:
  - **Peak** = 6 slowest hours → `[14–19]` (≈53 km/h citywide)
  - **Off-peak** = 6 fastest hours → `[0–5]` (≈60 km/h citywide)
  - **Blended** = mean over all 24 hours
- Resulting neighborhood-level mean speeds: peak ≈ 38.8, off-peak ≈ 49.3, blended ≈ 43.6 km/h.
- Per-neighborhood speeds are saved to `neighborhood_speeds.csv`.
- Fallback (API unavailable): uniform speeds peak/off-peak/blended = 40 / 50 / 45 km/h.

### Step 6 — Travel Time Matrices
- Computed pairwise Haversine distances between all 890 neighborhood centroids.
- For each O–D pair, the effective speed is the **harmonic mean** of the two endpoints' local
  speeds (correct for averaging speed over a journey split between the two regions).
- Travel time = `distance_km / effective_speed` (hours); diagonal set to 0.
- Three matrices produced from the corresponding speed profile:
  - `travel_times_peak.npy` — peak-hours speed
  - `travel_times_offpeak.npy` — off-peak (night) speed
  - `travel_times.npy` — blended (24-hour mean), used as default `t_ij`
- Mean off-diagonal times: peak ≈ 0.94 h, off-peak ≈ 0.69 h, blended ≈ 0.80 h
  (peak ≈ 39% slower than off-peak on average).

---

## Output Files (`data/processed/`)

| File | Shape / Size | Description |
|------|-------------|-------------|
| `neighborhoods.csv` | 890 rows × 7 cols | `neighborhood_id`, `name`, `district`, `lat`, `lon`, `population`, `rent_per_m2` |
| `rents.csv` | 39 rows × 2 cols | `district`, `avg_rent_per_m2` |
| `neighborhood_speeds.csv` | 890 rows × 6 cols | `neighborhood_id`, `name`, `district`, `speed_peak`, `speed_offpeak`, `speed_blended` (km/h) |
| `travel_times.npy` | (890, 890) float64 | Blended travel time matrix in hours |
| `travel_times_peak.npy` | (890, 890) float64 | Peak-hour travel time matrix in hours |
| `travel_times_offpeak.npy` | (890, 890) float64 | Off-peak travel time matrix in hours |
