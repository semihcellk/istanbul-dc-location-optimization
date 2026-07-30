#!/usr/bin/env python3
"""
Data preparation script.

Inputs  (data/raw/):
  - Muhtarlık Adres Bilgileri     : IBB muhtarlık GeoJSON (neighborhood coordinates)
  - pivot.csv                     : TUIK neighborhood populations (2025)
  - istanbul_tum_mahalleler.csv   : Endeksa neighborhood-level rent per m²
  - istanbul_tum_ilceler.csv      : Endeksa district-level rent per m² (fallback)
  - IBB Hourly Traffic Density (fetched live) : month of hourly geohash speeds

Run from the project root:
    python data/prepare_data.py

Outputs (data/processed/):
  - neighborhoods.csv          : neighborhood id, name, district, lat, lon, population (w_i)
  - rents.csv                  : district, avg rent per m² (r̄_d(j))
  - neighborhood_speeds.csv    : per-neighborhood peak/offpeak/blended speed (km/h)
  - travel_times_peak.npy      : travel time matrix — peak-hours speed (hours)
  - travel_times_offpeak.npy   : travel time matrix — off-peak (night) speed (hours)
  - travel_times.npy           : blended (24-hour mean speed)

Travel times are built from a one-month average of HOURLY traffic speeds
(see section 5), not a single day's snapshot.
"""

import json
import os
import re
import numpy as np
import pandas as pd
import requests

# Resolved from this file's location so the script works from any working
# directory, not only from the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_HERE, "raw")
OUT = os.path.join(_HERE, "processed")
os.makedirs(OUT, exist_ok=True)

# helpers
def haversine_matrix(lats, lons):
    """Pairwise haversine distance matrix for all neighborhood pairs (km)."""
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def haversine_to_points(lat0, lon0, lats, lons):
    """Haversine distance (km) from a single point to an array of points."""
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)
    lats_r, lons_r = np.radians(lats), np.radians(lons)
    dlat = lats_r - lat0_r
    dlon = lons_r - lon0_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0_r) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def normalize(s):
    return s.strip().replace('i', 'İ').upper()

# 1. Muhtarlık JSON → coordinates

print("1. Loading Muhtarlık coordinates...")
with open(f"{RAW}/Muhtarlık Adres Bilgileri", encoding="utf-8") as f:
    geojson = json.load(f)

coords = [
    {
        "name":     normalize(feat["properties"]["Mahalle Adı"]),
        "district": normalize(feat["properties"]["İlçe Adı"]),
        "lat":      feat["properties"]["Latitude"],
        "lon":      feat["properties"]["Longtitude"],
    }
    for feat in geojson["features"]
]
df_coords = pd.DataFrame(coords)
df_coords = df_coords.groupby(["name", "district"], as_index=False).agg({"lat": "mean", "lon": "mean"})
print(f"   {len(df_coords)} unique neighborhoods (after merging multiple muhtarlık per neighborhood)")

# 2. pivot.csv → population

print("2. Loading TUIK population data...")
raw_lines = open(f"{RAW}/pivot.csv", encoding="utf-8-sig").readlines()

pop_rows = []
for line in raw_lines:
    if "İstanbul(" not in line:
        continue
    parts = line.strip().split("|")
    loc_part = next((p for p in parts if "İstanbul(" in p), None)
    pop_str  = next((p for p in parts if re.match(r"^\s*[\d.]+\s*$", p)), None)
    if not loc_part or not pop_str:
        continue
    m = re.search(r"İstanbul\(([^/]+)/[^/]+/(.+?) Mah\.\)", loc_part)
    if not m:
        continue
    pop_rows.append({
        "name":       normalize(m.group(2)),
        "district":   normalize(m.group(1)),
        "population": int(float(pop_str.strip())),
    })

df_pop = pd.DataFrame(pop_rows).drop_duplicates(subset=["name", "district"])
print(f"   {len(df_pop)} neighborhoods with population")

# 3. Merge coordinates + population

print("3. Merging coordinates and population...")
df_coords["_key"] = df_coords["name"] + "|" + df_coords["district"]
df_pop["_key"]    = df_pop["name"]    + "|" + df_pop["district"]

# Manual aliases: Muhtarlık name → TUIK name (name differs between the two sources)
ALIASES = {
    "BAHÇEKÖY YENİMAHALLE|SARIYER": "BAHÇEKÖY YENİ|SARIYER",
    "KÜÇÜKÇAMLICA|ÜSKÜDAR":         "KÜÇÜK ÇAMLICA|ÜSKÜDAR",
    "RUMELİ KAVAĞI|SARIYER":        "RUMELİKAVAĞI|SARIYER",
    "CUMHURİYET|BEYKOZ":            "CUMHURİYETKÖY|BEYKOZ",
    "SARIYER YENİMAHALLE|SARIYER":  "SARIYER MERKEZ|SARIYER",
}
df_coords["_key"] = df_coords["_key"].replace(ALIASES)

df = df_coords.merge(df_pop[["_key", "population"]], on="_key", how="left").drop(columns="_key")

matched = df["population"].notna().sum()
unmatched = len(df) - matched
print(f"   Matched: {matched}, unmatched: {unmatched}")

if unmatched > 0:
    print("   Unmatched neighborhoods (no population assigned):")
    for _, r in df[df["population"].isna()].iterrows():
        print(f"     {r['name']} / {r['district']}")

# Manual population entries — sourced from TUIK 2025, looked up individually
MANUAL_POP = {
    ("BAĞLARÇEŞME", "ESENYURT"):     32537,
    ("HAVAALANI",   "ESENLER"):      32456,
    ("MİMAR SİNAN", "BÜYÜKÇEKMECE"): 9382,
    ("SARAY",       "ÜMRANİYE"):      3460,
    ("YENİŞEHİR",   "ÜMRANİYE"):      6584,
    ("ÇİFTLİK",     "BEYKOZ"):        5750,
}
for (name, district), pop in MANUAL_POP.items():
    mask = (df["name"] == name) & (df["district"] == district)
    df.loc[mask, "population"] = pop
    print(f"   Manual population set: {name} / {district} → {pop:,}")

# Drop neighborhoods with no population (cannot be used in the model)
df = df.dropna(subset=["population"]).reset_index(drop=True)
df["population"] = df["population"].astype(int)
df.index.name = "neighborhood_id"
df = df.reset_index()

print(f"   Final: {len(df)} neighborhoods")

# 4. Rent data

print("4. Processing Endeksa rent data (neighborhood + district)...")

# Parse district level data first
df_dist_rent = pd.read_csv(f"{RAW}/istanbul_tum_ilceler.csv", sep=";")
def parse_rent(val):
    """'528 ₺/m 2' → 528.0 ; '-' or malformed → NaN."""
    if pd.isna(val) or val == "-" or "₺" not in str(val):
        return np.nan
    v = str(val).split("₺")[0].replace(".", "").strip()
    try:
        return float(v)
    except ValueError:
        return np.nan

df_dist_rent["avg_rent_per_m2"] = df_dist_rent["Birim Fiyatı (₺/m2)"].apply(parse_rent)
df_dist_rent["district"] = df_dist_rent["Mahalle"].apply(normalize)
df_dist_rent = df_dist_rent.dropna(subset=["avg_rent_per_m2"])
district_rents = dict(zip(df_dist_rent["district"], df_dist_rent["avg_rent_per_m2"]))
median_district_rent = df_dist_rent["avg_rent_per_m2"].median()

# Parse neighborhood level data
df_mah_rent = pd.read_csv(f"{RAW}/istanbul_tum_mahalleler.csv", sep=";")
df_mah_rent["rent"] = df_mah_rent["Birim Fiyatı (₺/m2)"].apply(parse_rent)
df_mah_rent["name"] = df_mah_rent["Mahalle"].apply(normalize)
df_mah_rent["district"] = df_mah_rent["İlçe"].apply(normalize)
df_mah_rent["_rent_key"] = df_mah_rent["name"] + "|" + df_mah_rent["district"]
df_mah_rent = df_mah_rent.dropna(subset=["rent"])
mah_rents = dict(zip(df_mah_rent["_rent_key"], df_mah_rent["rent"]))

ALIASES_RENT = {
    "AKŞEMSETTİN|SULTANBEYLİ": "AKŞEMSEDDİN|SULTANBEYLİ",
    "ATAKÖY 2-5-6. KISIM|BAKIRKÖY": "ATAKÖY 2. 5. 6. KISIM|BAKIRKÖY",
    "AŞIKVEYSEL|ATAŞEHİR": "AŞIK VEYSEL|ATAŞEHİR",
    "KAMER HATUN|BEYOĞLU": "KAMERHATUN|BEYOĞLU",
    "KEÇECİ PİRİ|BEYOĞLU": "KEÇECİPİRİ|BEYOĞLU",
    "KUMKÖY (KİLYOS)|SARIYER": "KUMKÖY|SARIYER",
    "MERKEZ|BEYKOZ": "BEYKOZ MERKEZ|BEYKOZ",
    "MERKEZ|EYÜPSULTAN": "EYÜP MERKEZ|EYÜPSULTAN",
    "MURAT ÇESME|BÜYÜKÇEKMECE": "MURAT ÇEŞME|BÜYÜKÇEKMECE",
    "MUSTAFA KEMALPAŞA|AVCILAR": "MUSTAFA KEMAL PAŞA|AVCILAR",
    "MİMAR SİNAN|BÜYÜKÇEKMECE": "MİMARSİNAN|BÜYÜKÇEKMECE",
    "NENEHATUN|ARNAVUTKÖY": "NENE HATUN|ARNAVUTKÖY",
    "NİŞANCI|EYÜPSULTAN": "NİŞANCA|EYÜPSULTAN",
    "ORHANGAZİ|SULTANBEYLİ": "ORHAN GAZİ|SULTANBEYLİ",
    "SURURİ MEHMET EFENDİ|BEYOĞLU": "SURURİ|BEYOĞLU",
    "ÇİFTLİK|BEYKOZ": "ÇAVUŞBAŞI ÇİFTLİK|BEYKOZ",
    "İSMET PAŞA|BAYRAMPAŞA": "İSMETPAŞA|BAYRAMPAŞA",
    "İSMETPAŞA|SULTANGAZİ": "İSMET PAŞA|SULTANGAZİ",
    "PİRİPAŞA|BEYOĞLU": "PİRİ MEHMET PAŞA|BEYOĞLU",
}

# Add rent column to df
df["_rent_key"] = (df["name"] + "|" + df["district"]).replace(ALIASES_RENT)
df["rent_per_m2"] = df["_rent_key"].map(mah_rents)

missing_mah = df["rent_per_m2"].isna().sum()
print(f"   {len(df) - missing_mah} neighborhoods matched with mahalle rent")

# Fill missing mahalle rent with district rent
df.loc[df["rent_per_m2"].isna(), "rent_per_m2"] = df.loc[df["rent_per_m2"].isna(), "district"].map(district_rents)
missing_dist = df["rent_per_m2"].isna().sum()
print(f"   {missing_mah - missing_dist} filled with district rent")

# Fill remaining with median district rent
if missing_dist > 0:
    df.loc[df["rent_per_m2"].isna(), "rent_per_m2"] = median_district_rent
    print(f"   {missing_dist} filled with median district rent ({median_district_rent:.0f})")

df = df.drop(columns=["_rent_key"])
df.to_csv(f"{OUT}/neighborhoods.csv", index=False)
print(f"   Saved → {OUT}/neighborhoods.csv (with rent_per_m2)")

df_dist_rent[["district", "avg_rent_per_m2"]].to_csv(f"{OUT}/rents.csv", index=False)
print(f"   Saved → {OUT}/rents.csv (district level, for compatibility)")

# 5. IBB Hourly Traffic Density → neighborhood hourly speed profiles
#
# Methodology (per supervisor feedback): traffic is NOT taken from a single
# day. We use one full month of HOURLY traffic density and, for every hour of
# the day (0–23), average the measured AVERAGE_SPEED over all days in that
# month. This yields a representative 24-hour speed profile per location.
#
# The hourly dataset is spatial: each record is a geohash cell with lat/lon and
# the measured speed for that hour. We map every neighborhood to its K nearest
# geohash cells, giving each neighborhood its own month-averaged hourly speed
# profile (spatial mapping). Peak / off-peak / blended speeds are then derived
# from that profile.

print("5. Fetching hourly traffic density (Ekim 2024) from IBB...")
SQL_URL          = "https://data.ibb.gov.tr/api/3/action/datastore_search_sql"
TRAFFIC_RESOURCE = "d291989c-429d-4e61-9c70-1f76294b96b8"  # "Ekim 2024 Trafik Yoğunluk Verisi"
TRAFFIC_MONTH    = "Ekim 2024"
K_NEAREST        = 3      # number of nearest geohash cells averaged per neighborhood
FAR_KM           = 5.0    # beyond this, fall back to the citywide hourly profile

lats = df["lat"].values
lons = df["lon"].values
n    = len(df)

# Uniform-speed fallback (km/h) if the API is unreachable
FALLBACK = {"peak": 40.0, "offpeak": 50.0, "blended": 45.0}

def fetch_hourly_geohash_speeds(resource_id):
    """Per geohash cell, the month-averaged AVERAGE_SPEED for each hour 0–23.

    One SQL query per hour keeps every response under the datastore's 32k-row
    cap (≈2.5k geohash cells per hour). The COUNT(*) per group equals the number
    of days in the month, confirming we average the whole month, not one day.
    """
    rows = []
    for h in range(24):
        hh = f"{h:02d}"
        sql = (
            f'SELECT "GEOHASH" g, "LATITUDE" lat, "LONGITUDE" lon, '
            f'AVG("AVERAGE_SPEED") s FROM "{resource_id}" '
            f"WHERE substr(\"DATE_TIME\", 12, 2) = '{hh}' "
            f'GROUP BY "GEOHASH", "LATITUDE", "LONGITUDE"'
        )
        resp = requests.get(SQL_URL, params={"sql": sql}, timeout=120)
        resp.raise_for_status()
        for rec in resp.json()["result"]["records"]:
            rows.append((rec["g"], float(rec["lat"]), float(rec["lon"]), h, float(rec["s"])))
    return pd.DataFrame(rows, columns=["geohash", "lat", "lon", "hour", "speed"])

try:
    g = fetch_hourly_geohash_speeds(TRAFFIC_RESOURCE)
    piv = g.pivot_table(index=["geohash", "lat", "lon"], columns="hour", values="speed")
    piv = piv.reindex(columns=range(24))
    gh_speed = piv.values                              # (G, 24) km/h
    gh_speed = np.where(np.isnan(gh_speed), np.nanmean(gh_speed, axis=1, keepdims=True), gh_speed)
    gh_lat = piv.index.get_level_values("lat").to_numpy()
    gh_lon = piv.index.get_level_values("lon").to_numpy()

    # Citywide hourly profile → data-driven peak / off-peak windows
    city = np.nanmean(gh_speed, axis=0)                # (24,)
    peak_hours = np.argsort(city)[:6]                  # 6 slowest hours (rush)
    off_hours  = np.argsort(city)[-6:]                 # 6 fastest hours (night)
    city_peak, city_off, city_bl = city[peak_hours].mean(), city[off_hours].mean(), city.mean()
    print(f"   {gh_speed.shape[0]} geohash cells × 24h ({TRAFFIC_MONTH} monthly average)")
    print(f"   peak hours={sorted(int(h) for h in peak_hours)} ({city_peak:.1f} km/h), "
          f"offpeak hours={sorted(int(h) for h in off_hours)} ({city_off:.1f} km/h)")

    # Spatial mapping: each neighborhood → mean profile of K nearest geohash cells
    v_peak, v_off, v_bl = np.empty(n), np.empty(n), np.empty(n)
    n_far = 0
    for i in range(n):
        d = haversine_to_points(lats[i], lons[i], gh_lat, gh_lon)
        idx = np.argsort(d)[:K_NEAREST]
        if d[idx[0]] > FAR_KM:                         # remote area → citywide profile
            v_peak[i], v_off[i], v_bl[i] = city_peak, city_off, city_bl
            n_far += 1
        else:
            sp = gh_speed[idx].mean(axis=0)            # (24,) local hourly profile
            v_peak[i], v_off[i], v_bl[i] = sp[peak_hours].mean(), sp[off_hours].mean(), sp.mean()
    print(f"   neighborhood speeds (km/h): peak {v_peak.mean():.1f}, "
          f"offpeak {v_off.mean():.1f}, blended {v_bl.mean():.1f} "
          f"({n_far} remote neighborhoods used citywide fallback)")

except Exception as e:
    print(f"   API error ({e}); falling back to uniform speeds {FALLBACK}")
    v_peak = np.full(n, FALLBACK["peak"])
    v_off  = np.full(n, FALLBACK["offpeak"])
    v_bl   = np.full(n, FALLBACK["blended"])

# 6. Travel time matrices (from real measured speeds)
#
# For an origin–destination pair, the effective speed is the harmonic mean of
# the two endpoints' local speeds (correct for averaging speed over a journey
# split between the two regions). Travel time = distance / effective speed.

print("6. Computing travel time matrices...")
dist_km = haversine_matrix(lats, lons)

def travel_times_from_speed(v):
    """O-D travel-time matrix (hours) using harmonic-mean endpoint speeds."""
    v_eff = 2.0 / (1.0 / v[:, None] + 1.0 / v[None, :])   # (n, n) km/h
    tt = dist_km / v_eff
    np.fill_diagonal(tt, 0.0)
    return tt

tt_peak    = travel_times_from_speed(v_peak)
tt_offpeak = travel_times_from_speed(v_off)
tt_blended = travel_times_from_speed(v_bl)

np.save(f"{OUT}/travel_times_peak.npy",    tt_peak)
np.save(f"{OUT}/travel_times_offpeak.npy", tt_offpeak)
np.save(f"{OUT}/travel_times.npy",         tt_blended)

# Persist the per-neighborhood speed profile for documentation / the report
pd.DataFrame({
    "neighborhood_id": df["neighborhood_id"],
    "name":            df["name"],
    "district":        df["district"],
    "speed_peak":      np.round(v_peak, 2),
    "speed_offpeak":   np.round(v_off, 2),
    "speed_blended":   np.round(v_bl, 2),
}).to_csv(f"{OUT}/neighborhood_speeds.csv", index=False)

print(f"   Matrix size: {n}×{n}")
print(f"   Saved → travel_times_peak.npy, travel_times_offpeak.npy, travel_times.npy")
print(f"   Saved → neighborhood_speeds.csv (per-neighborhood peak/offpeak/blended km/h)")

print("\nDone. data/processed/ contents:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f"{OUT}/{f}")
    print(f"   {f}  ({size/1024:.1f} KB)")