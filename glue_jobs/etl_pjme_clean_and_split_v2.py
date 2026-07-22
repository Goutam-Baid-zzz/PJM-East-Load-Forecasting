import boto3
import pandas as pd
import numpy as np

RAW_BUCKET = "pjm-load-raw"
LOAD_KEY = "PJME_hourly.csv"
WEATHER_KEY = "weather_philadelphia.csv"
PROCESSED_BUCKET = "pjm-load-processed"
LOCAL_LOAD_PATH = "/tmp/PJME_hourly.csv"
LOCAL_WEATHER_PATH = "/tmp/weather_philadelphia.csv"

TRAIN_END = "2016-12-31 23:00:00"
VAL_END = "2017-12-31 23:00:00"

HDD_CDD_BASE_TEMP_F = 65  

s3 = boto3.client("s3")


def download_raw():
    print(f"Downloading s3://{RAW_BUCKET}/{LOAD_KEY} ...")
    s3.download_file(RAW_BUCKET, LOAD_KEY, LOCAL_LOAD_PATH)
    print(f"Downloading s3://{RAW_BUCKET}/{WEATHER_KEY} ...")
    s3.download_file(RAW_BUCKET, WEATHER_KEY, LOCAL_WEATHER_PATH)
    print("Downloads complete.")


def load_and_clean():
    df = pd.read_csv(LOCAL_LOAD_PATH, usecols=["Datetime", "PJME_MW"])
    df = df.dropna(subset=["PJME_MW"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    before = len(df)
    df = df.drop_duplicates(subset="Datetime", keep="first")
    print(f"Dropped {before - len(df)} duplicate timestamp rows (load data).")

    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.rename(columns={"PJME_MW": "PJME_MW"})
    return df


def load_weather():
    weather = pd.read_csv(LOCAL_WEATHER_PATH)
    weather["Datetime"] = pd.to_datetime(weather["Datetime"])
    before = len(weather)
    weather = weather.drop_duplicates(subset="Datetime", keep="first")
    print(f"Dropped {before - len(weather)} duplicate timestamp rows (weather data).")
    return weather


def join_weather(df, weather):
    before = len(df)
    merged = df.merge(weather, on="Datetime", how="left")
    missing_temp = merged["temp_f"].isna().sum()
    print(f"Joined weather. Rows before: {before}, after: {len(merged)}, "
          f"missing temp_f after join: {missing_temp}")

    if missing_temp > 0:
        merged["temp_f"] = merged["temp_f"].ffill()
        still_missing = merged["temp_f"].isna().sum()
        print(f"After forward-fill, still missing: {still_missing} (these rows will be dropped)")
        merged = merged.dropna(subset=["temp_f"])

    return merged


def engineer_features(df):
    df = df.set_index("Datetime")

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["lag_24h"] = df["PJME_MW"].shift(24)
    df["lag_168h"] = df["PJME_MW"].shift(168)

    df["rolling_mean_168h"] = df["PJME_MW"].shift(1).rolling(window=168).mean()
    df["rolling_std_168h"] = df["PJME_MW"].shift(1).rolling(window=168).std()

    df["hdd"] = (HDD_CDD_BASE_TEMP_F - df["temp_f"]).clip(lower=0)
    df["cdd"] = (df["temp_f"] - HDD_CDD_BASE_TEMP_F).clip(lower=0)

    df = df.dropna()
    df = df.reset_index()
    return df


def chronological_split(df):
    train = df[df["Datetime"] <= TRAIN_END]
    val = df[(df["Datetime"] > TRAIN_END) & (df["Datetime"] <= VAL_END)]
    test = df[df["Datetime"] > VAL_END]

    print(f"Train: {len(train)} rows ({train['Datetime'].min()} to {train['Datetime'].max()})")
    print(f"Val:   {len(val)} rows ({val['Datetime'].min()} to {val['Datetime'].max()})")
    print(f"Test:  {len(test)} rows ({test['Datetime'].min()} to {test['Datetime'].max()})")

    return train, val, test


def write_split(df, split_name):
    local_path = f"/tmp/{split_name}.parquet"
    df.to_parquet(local_path, engine="pyarrow", index=False)
    s3_key = f"{split_name}/{split_name}.parquet"
    s3.upload_file(local_path, PROCESSED_BUCKET, s3_key)
    print(f"Wrote s3://{PROCESSED_BUCKET}/{s3_key}")


def main():
    download_raw()
    df = load_and_clean()
    weather = load_weather()
    df = join_weather(df, weather)
    df = engineer_features(df)
    train, val, test = chronological_split(df)

    write_split(train, "train")
    write_split(val, "val")
    write_split(test, "test")

    print("ETL job complete (with weather fusion).")


if __name__ == "__main__":
    main()