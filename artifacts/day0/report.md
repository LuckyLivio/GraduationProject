# Day-zero feasibility run

This is a pilot with one independent training run, not final thesis evidence.

Validation-selected fixed horizon: `fixed_5`. Total runtime: 50.4 seconds.

| Method | Condition | N | Success | Collision | P95 planning ms |
|---|---|---:|---:|---:|---:|
| global_physics | damping_shift | 12 | 83.3% | 16.7% | 5.33 |
| local_identification | damping_shift | 12 | 100.0% | 0.0% | 4.99 |
| fixed_5 | damping_shift | 12 | 83.3% | 16.7% | 4.79 |
| fixed_10 | damping_shift | 12 | 91.7% | 8.3% | 6.30 |
| fixed_16 | damping_shift | 12 | 100.0% | 0.0% | 8.71 |
| adaptive | damping_shift | 12 | 83.3% | 16.7% | 9.67 |
| global_physics | in_distribution | 12 | 91.7% | 8.3% | 4.42 |
| local_identification | in_distribution | 12 | 100.0% | 0.0% | 5.09 |
| fixed_5 | in_distribution | 12 | 100.0% | 0.0% | 5.84 |
| fixed_10 | in_distribution | 12 | 100.0% | 0.0% | 5.92 |
| fixed_16 | in_distribution | 12 | 100.0% | 0.0% | 8.09 |
| adaptive | in_distribution | 12 | 100.0% | 0.0% | 9.38 |

## Boundaries

- one independent training run; ensemble members are not independent experiment runs
- pilot tests are diagnostic; do not tune on these seeds and reuse them as final paper test
- history-based physical baseline receives recent transitions; frozen learned model is stateless
- known obstacle dynamics and kinematic structure retained; not fully learned state/video prediction
- model-step budget compares ensemble members; real latency differs across predictor families
