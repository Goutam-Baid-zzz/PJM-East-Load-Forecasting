"""
Fetches historical hourly temperature for Philadelphia, PA (proxy for PJM East region)
from Open-Meteo's free historical archive API. Chunked by year to keep requests small
and reliable. Run locally, then upload the resulting CSV to s3://pjm-load-raw/.
"""

import requests
import pandas as pd
import time

LATITUDE = 39.9526
LONGITUDE = -75.1652
START_YEAR = 2002
END_YEAR = 2018  # data goes up to 2018-08-03, requesting the full year is fine, we'll trim later

OUTPUT_PATH = "weather_philadelphia.csv"

all_data = []

for year in range(START_YEAR, END_YEAR + 1):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    print(f"Fetching {start_date} to {end_date}...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "Datetime": data["hourly"]["time"],
        "temp_f": data["hourly"]["temperature_2m"],
    })
    all_data.append(df)

    time.sleep(1)  # be polite to the free API

weather_df = pd.concat(all_data, ignore_index=True)
weather_df["Datetime"] = pd.to_datetime(weather_df["Datetime"])
weather_df = weather_df.drop_duplicates(subset="Datetime").sort_values("Datetime")

weather_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(weather_df)} rows to {OUTPUT_PATH}")
print(weather_df.head())
print(weather_df.tail())
