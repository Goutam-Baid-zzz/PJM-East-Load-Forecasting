"""
Seeds the pjm-load-cache DynamoDB table with historical load + temperature data,
simulating what a live grid/weather feed would populate in production.

Run this locally after downloading test.parquet (and ideally val.parquet too,
for extra lookback range) from S3.
"""

import boto3
import pandas as pd
from decimal import Decimal

TABLE_NAME = "pjm-load-cache"
REGION_KEY = "PJME"
DATA_DIR = "data/processed"

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table(TABLE_NAME)


def load_data():
    # Use val + test so there's enough lookback history to compute a 168h rolling
    # window even for the earliest timestamps in the test set.
    val = pd.read_parquet(f"{DATA_DIR}/val.parquet")
    test = pd.read_parquet(f"{DATA_DIR}/test.parquet")
    df = pd.concat([val, test], ignore_index=True)
    df = df[["Datetime", "PJME_MW", "temp_f"]].sort_values("Datetime")
    return df


def seed_table(df):
    print(f"Seeding {len(df)} rows into {TABLE_NAME}...")

    with table.batch_writer() as batch:
        for i, row in df.iterrows():
            item = {
                "region": REGION_KEY,
                "datetime": row["Datetime"].strftime("%Y-%m-%dT%H:%M:%S"),
                "pjme_mw": Decimal(str(row["PJME_MW"])),
                "temp_f": Decimal(str(row["temp_f"])),
            }
            batch.put_item(Item=item)

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(df)} written...")

    print("Seeding complete.")


def main():
    df = load_data()
    seed_table(df)
    print(f"\nDate range seeded: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print("Note: predictions can only be requested for timestamps that have at least "
          "168 hours of lookback history within this seeded range.")


if __name__ == "__main__":
    main()
