# Cost Breakdown

This project was built and kept running end-to-end for well under $1 total,
by design. Every service was chosen specifically to avoid idle hourly
billing — the two exceptions (SageMaker notebook instance, S3 storage) are
either stoppable or negligible.

## Per-service cost model

| Service | Pricing model | Behavior in this project |
|---|---|---|
| **S3** (4 buckets) | ~$0.023/GB/month | Total data <100MB → fractions of a cent/month |
| **Glue (Python Shell job)** | Billed per DPU-second while running | Job runs in 1–3 minutes per execution → a few cents total across all runs |
| **Glue Crawler** | Billed per DPU-hour while running | Run manually, on-demand only, never scheduled → seconds of billing per run |
| **Athena** | $5 per TB scanned | Validation queries scanned KB–MB of Parquet each → effectively $0 |
| **SageMaker Notebook Instance** (`ml.t3.medium`) | ~$0.05/hour while running | **The one component that bills while idle if left running** — stopped between sessions throughout development |
| **SageMaker Training (Local Mode)** | $0 — runs on the already-billed notebook instance via Docker | No separate training-instance charge at all, which was also the reason Local Mode was used in the first place (quota wall on real training jobs) |
| **SageMaker Serverless Inference** | Pay per inference-second + per-GB-second of memory used | Scales to ~zero between requests; each invocation costs a small fraction of a cent |
| **Lambda** (3 functions active) | Free tier: 1M requests + 400,000 GB-seconds/month | Testing volume for this project is far below free tier limits → $0 |
| **API Gateway** | Free tier: 1M REST API calls/month (first 12 months) | Same — testing volume is negligible → $0 |
| **DynamoDB** (on-demand) | Pay per read/write request unit | ~14,000 seeded rows (one-time write) + light read testing → a few cents at most |
| **CloudWatch** | Free tier covers basic logging | Used only for debugging/log inspection during development → $0 |

## What actually cost money

The only genuinely metered spend across this entire build was:
1. A handful of SageMaker notebook instance hours during active development
   (stopped whenever not in use)
2. Negligible fractions of a cent from Glue job runs, Athena queries, and
   DynamoDB writes

## What this means going forward

With the notebook instance **stopped** (not deleted — it can be restarted any
time more training/experimentation is needed), the entire live system —
S3 storage, the deployed model endpoint, both Lambda functions, API Gateway,
and DynamoDB — can sit live indefinitely for **a few cents a month**, driven
almost entirely by S3 storage and DynamoDB's tiny footprint. Actual usage
(someone testing the `/forecast` API) adds only fractions of a cent per
invocation on top of that.

This cost profile was a deliberate design constraint from the start, not an
afterthought — see "Key engineering decisions" in the main README for the
specific architectural choices (Local Mode training, Serverless Inference,
skipping Step Functions/EventBridge) made to keep it this way.
