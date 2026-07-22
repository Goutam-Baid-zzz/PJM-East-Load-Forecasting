"""
Lambda handler: pjm-smart-predict-handler

Accepts just a target datetime and builds the full feature vector automatically:
- Calendar/cyclical features: computed directly from the timestamp
- Lag + rolling features: queried from DynamoDB (pjm-load-cache table)
- Weather features: looked up from DynamoDB (stand-in for a live forecast API
  in production — see README for this documented simplification)

Then invokes the SageMaker endpoint and returns the prediction.

Expected event body:
{"datetime": "2018-08-04T14:00:00"}
"""

import json
import os
import math
from datetime import datetime, timedelta
from decimal import Decimal

import boto3

TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "pjm-load-cache")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "pjm-load-forecast-endpoint")
REGION_KEY = "PJME"
HDD_CDD_BASE_TEMP_F = 65

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
runtime = boto3.client("sagemaker-runtime")

# US federal holidays, hardcoded for the years the seeded data covers.
# (Avoids pulling in pandas/holidays libraries just for this lookup in Lambda.)
US_HOLIDAYS_2017_2018 = {
    "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
    "2018-09-03",
}


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        target_str = body.get("datetime")

        if not target_str:
            return _response(400, {"error": "Missing 'datetime' field. Format: YYYY-MM-DDTHH:MM:SS"})

        target_dt = datetime.strptime(target_str, "%Y-%m-%dT%H:%M:%S")

        calendar_features = _build_calendar_features(target_dt)
        lag_features, missing_hours = _build_lag_and_rolling_features(target_dt)

        if missing_hours:
            return _response(
                422,
                {
                    "error": "Insufficient history in cache for this timestamp.",
                    "missing_hours": missing_hours[:5],  # sample, not the full potentially-long list
                    "hint": "Requested datetime must have a full 168-hour lookback window "
                            "available in the seeded DynamoDB cache.",
                },
            )

        weather_features = _build_weather_features(target_dt)
        if weather_features is None:
            return _response(
                422,
                {"error": f"No temperature data cached for {target_str}. "
                          "In production this would come from a live weather forecast API."},
            )

        features = (
            calendar_features
            + lag_features
            + weather_features
        )

        payload = json.dumps({"features": features})
        sm_response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        result = json.loads(sm_response["Body"].read().decode("utf-8"))

        return _response(200, {
            "datetime": target_str,
            "prediction_mw": result["prediction"],
        })

    except ValueError as e:
        return _response(400, {"error": f"Invalid datetime format: {str(e)}"})
    except Exception as e:
        return _response(500, {"error": str(e)})


def _build_calendar_features(dt):
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if dayofweek >= 5 else 0
    is_holiday = 1 if dt.strftime("%Y-%m-%d") in US_HOLIDAYS_2017_2018 else 0

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    dow_sin = math.sin(2 * math.pi * dayofweek / 7)
    dow_cos = math.cos(2 * math.pi * dayofweek / 7)

    return [
        hour, dayofweek, month, quarter, is_weekend, is_holiday,
        hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos,
    ]


def _fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _build_lag_and_rolling_features(target_dt):
    range_start = target_dt - timedelta(hours=168)
    range_end = target_dt - timedelta(hours=1)

    response = table.query(
        KeyConditionExpression=(
            boto3.dynamodb.conditions.Key("region").eq(REGION_KEY)
            & boto3.dynamodb.conditions.Key("datetime").between(_fmt(range_start), _fmt(range_end))
        )
    )
    items = {item["datetime"]: float(item["pjme_mw"]) for item in response["Items"]}

    expected_hours = [_fmt(range_start + timedelta(hours=i)) for i in range(168)]
    missing_hours = [h for h in expected_hours if h not in items]

    if missing_hours:
        return None, missing_hours

    lag_24h_key = _fmt(target_dt - timedelta(hours=24))
    lag_168h_key = _fmt(target_dt - timedelta(hours=168))

    lag_24h = items[lag_24h_key]
    lag_168h = items[lag_168h_key]

    values = list(items.values())
    rolling_mean_168h = sum(values) / len(values)
    variance = sum((v - rolling_mean_168h) ** 2 for v in values) / (len(values) - 1)
    rolling_std_168h = math.sqrt(variance)

    return [lag_24h, lag_168h, rolling_mean_168h, rolling_std_168h], None


def _build_weather_features(target_dt):
    response = table.get_item(Key={"region": REGION_KEY, "datetime": _fmt(target_dt)})
    item = response.get("Item")

    if item is None or "temp_f" not in item:
        return None

    temp_f = float(item["temp_f"])
    hdd = max(HDD_CDD_BASE_TEMP_F - temp_f, 0)
    cdd = max(temp_f - HDD_CDD_BASE_TEMP_F, 0)

    return [temp_f, hdd, cdd]


def _response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body_dict),
    }