# Day-zero feasibility run

This is a pilot with one independent training run, not final thesis evidence.

Validation-selected fixed horizon: `fixed_5`. Total runtime: 80.6 seconds.

| Method | Condition | N | Success | Collision | P95 planning ms |
|---|---|---:|---:|---:|---:|
| global_physics | damping_shift | 24 | 95.8% | 4.2% | 5.16 |
| local_identification | damping_shift | 24 | 100.0% | 0.0% | 5.13 |
| fixed_5 | damping_shift | 24 | 95.8% | 4.2% | 5.93 |
| fixed_10 | damping_shift | 24 | 95.8% | 4.2% | 7.54 |
| fixed_16 | damping_shift | 24 | 95.8% | 4.2% | 9.36 |
| adaptive | damping_shift | 24 | 91.7% | 8.3% | 11.12 |
| global_physics | in_distribution | 24 | 100.0% | 0.0% | 5.38 |
| local_identification | in_distribution | 24 | 100.0% | 0.0% | 5.31 |
| fixed_5 | in_distribution | 24 | 100.0% | 0.0% | 6.20 |
| fixed_10 | in_distribution | 24 | 100.0% | 0.0% | 7.31 |
| fixed_16 | in_distribution | 24 | 100.0% | 0.0% | 8.84 |
| adaptive | in_distribution | 24 | 100.0% | 0.0% | 11.04 |

## Boundaries

- one independent training run; ensemble members are not independent experiment runs
- pilot tests are diagnostic; do not tune on these seeds and reuse them as final paper test
- history-based physical baseline receives recent transitions; frozen learned model is stateless
- known obstacle dynamics and kinematic structure retained; not fully learned state/video prediction
- model-step budget compares ensemble members; real latency differs across predictor families
