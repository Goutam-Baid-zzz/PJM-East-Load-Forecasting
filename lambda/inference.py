"""
SageMaker inference entry point for PJME load forecasting.
Loads the joblib-saved XGBoost model and handles JSON prediction requests.

Expected request format:
{
  "features": [hour, dayofweek, month, quarter, is_weekend, is_holiday,
               hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos,
               lag_24h, lag_168h, rolling_mean_168h, rolling_std_168h,
               temp_f, hdd, cdd]
}

Response format:
{"prediction": 31245.7}
"""

import os
import json
import joblib
import numpy as np

FEATURES = [
    "hour", "dayofweek", "month", "quarter", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "lag_24h", "lag_168h", "rolling_mean_168h", "rolling_std_168h",
    "temp_f", "hdd", "cdd",
]


def model_fn(model_dir):
    """Load the model — called once when the endpoint starts."""
    model_path = os.path.join(model_dir, "xgboost-model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type="application/json"):
    """Parse the incoming request into a feature array."""
    if content_type == "application/json":
        data = json.loads(request_body)
        features = data["features"]

        if len(features) != len(FEATURES):
            raise ValueError(
                f"Expected {len(FEATURES)} features, got {len(features)}. "
                f"Expected order: {FEATURES}"
            )

        return np.array(features, dtype=np.float64).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run the actual prediction."""
    prediction = model.predict(input_data)
    return prediction


def output_fn(prediction, accept="application/json"):
    """Format the response."""
    if accept == "application/json":
        result = {"prediction": float(prediction[0])}
        return json.dumps(result), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
