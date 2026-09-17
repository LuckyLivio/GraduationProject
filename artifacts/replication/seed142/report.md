# Day-zero feasibility run

This is a pilot with one independent training run, not final thesis evidence.

Validation-selected fixed horizon: `fixed_5`. Total runtime: 76.5 seconds.

| Method | Condition | N | Success | Collision | P95 planning ms |
|---|---|---:|---:|---:|---:|
| global_physics | damping_shift | 24 | 95.8% | 4.2% | 4.02 |
| local_identification | damping_shift | 24 | 100.0% | 0.0% | 4.79 |
| fixed_5 | damping_shift | 24 | 95.8% | 4.2% | 4.75 |
| fixed_10 | damping_shift | 24 | 95.8% | 4.2% | 6.90 |
| fixed_16 | damping_shift | 24 | 95.8% | 4.2% | 8.69 |
| adaptive | damping_shift | 24 | 91.7% | 8.3% | 10.20 |
| global_physics | in_distribution | 24 | 100.0% | 0.0% | 5.30 |
| local_identification | in_distribution | 24 | 100.0% | 0.0% | 4.97 |
| fixed_5 | in_distribution | 24 | 100.0% | 0.0% | 6.26 |
| fixed_10 | in_distribution | 24 | 100.0% | 0.0% | 7.57 |
| fixed_16 | in_distribution | 24 | 100.0% | 0.0% | 9.20 |
| adaptive | in_distribution | 24 | 100.0% | 0.0% | 9.57 |

## Boundaries

- one independent training run; ensemble members are not independent experiment runs
- pilot tests are diagnostic; do not tune on these seeds and reuse them as final paper test
- history-based physical baseline receives recent transitions; frozen learned model is stateless
- known obstacle dynamics and kinematic structure retained; not fully learned state/video prediction
- model-step budget compares ensemble members; real latency differs across predictor families
