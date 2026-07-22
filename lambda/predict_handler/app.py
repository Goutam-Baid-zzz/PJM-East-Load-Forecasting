"""
Lambda handler: pjm-predict-handler

Receives a feature vector via API Gateway, invokes the SageMaker Serverless
Inference endpoint, and returns the prediction.

Expected event body (API Gateway proxy integration):
{
  "features": [hour, dayofweek, month, quarter, is_weekend, is_holiday,
               hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos,
               lag_24h, lag_168h, rolling_mean_168h, rolling_std_168h,
               temp_f, hdd, cdd]
}
"""

import json
import os
import boto3

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "pjm-load-forecast-endpoint")

runtime = boto3.client("sagemaker-runtime")


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))

        if "features" not in body:
            return _response(400, {"error": "Missing 'features' field in request body."})

        payload = json.dumps({"features": body["features"]})

        sm_response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )

        result = json.loads(sm_response["Body"].read().decode("utf-8"))
        return _response(200, result)

    except Exception as e:
        return _response(500, {"error": str(e)})


def _response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  
        },
        "body": json.dumps(body_dict),
    }