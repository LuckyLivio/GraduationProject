# Day-zero feasibility run

This is a pilot with one independent training run, not final thesis evidence.

Validation-selected fixed horizon: `fixed_5`. Total runtime: 79.6 seconds.

| Method | Condition | N | Success | Collision | P95 planning ms |
|---|---|---:|---:|---:|---:|
| global_physics | damping_shift | 24 | 95.8% | 4.2% | 4.94 |
| local_identification | damping_shift | 24 | 100.0% | 0.0% | 5.13 |
| fixed_5 | damping_shift | 24 | 95.8% | 4.2% | 5.86 |
| fixed_10 | damping_shift | 24 | 95.8% | 4.2% | 7.39 |
| fixed_16 | damping_shift | 24 | 95.8% | 4.2% | 9.35 |
| adaptive | damping_shift | 24 | 95.8% | 4.2% | 12.39 |
| global_physics | in_distribution | 24 | 100.0% | 0.0% | 4.75 |
| local_identification | in_distribution | 24 | 100.0% | 0.0% | 4.70 |
| fixed_5 | in_distribution | 24 | 100.0% | 0.0% | 5.79 |
| fixed_10 | in_distribution | 24 | 100.0% | 0.0% | 7.53 |
| fixed_16 | in_distribution | 24 | 100.0% | 0.0% | 8.10 |
| adaptive | in_distribution | 24 | 100.0% | 0.0% | 11.79 |

## Boundaries

- one independent training run; ensemble members are not independent experiment runs
- pilot tests are diagnostic; do not tune on these seeds and reuse them as final paper test
- history-based physical baseline receives recent transitions; frozen learned model is stateless
- known obstacle dynamics and kinematic structure retained; not fully learned state/video prediction
- model-step budget compares ensemble members; real latency differs across predictor families
