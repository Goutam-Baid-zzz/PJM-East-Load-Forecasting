"""
SageMaker training entry point for PJME load forecasting.
Mirrors the exact logic validated locally in 03_baseline_modeling.py —
same features, same hyperparameters — just running on SageMaker infra
and reading data from the channels SageMaker mounts into the container.
"""

import argparse
import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

FEATURES = [
    "hour", "dayofweek", "month", "quarter", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "lag_24h", "lag_168h", "rolling_mean_168h", "rolling_std_168h",
    "temp_f", "hdd", "cdd",
]
TARGET = "pjme_mw"


def load_parquet_dir(path):
    """SageMaker mounts each S3 channel as a local directory — read the parquet file(s) in it."""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)

    # SageMaker sets these automatically based on channel names in the estimator .fit() call
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    print("Loading data...")
    train_df = load_parquet_dir(args.train)
    val_df = load_parquet_dir(args.validation)
    print(f"Train: {train_df.shape}, Val: {val_df.shape}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    print("Training...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    mape = mean_absolute_percentage_error(y_val, val_pred) * 100
    print(f"Final Val MAE: {mae:.1f} MW, MAPE: {mape:.2f}%")

    # Save model to the path SageMaker expects — everything here gets packaged into model.tar.gz
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "xgboost-model.joblib"))
    print(f"Model saved to {args.model_dir}")


if __name__ == "__main__":
    main()
