"""Run the complete day-zero feasibility experiment with held-out seeds.

python -m foresight.experiment --output artifacts/day0 --episodes 12 --epochs 80
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .data import collect_dataset
from .env import NavigationEnv, DT, ROBOT_RADIUS, OBSTACLE_RADIUS
from .model import train_world_model, HybridWorldModel, transition_targets
from .planner import PhysicsPredictor, MPCPlanner


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def compact(value):
    if isinstance(value, np.ndarray):
        return np.round(value, 4).tolist()
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    return value


def window_samples(data, horizon=16, limit=160, seed=47):
    indices = np.arange(len(data['states']) - horizon + 1)
    valid = data['episode_ids'][indices] == data['episode_ids'][indices + horizon - 1]
    indices = indices[valid]
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [(data['states'][i], data['actions'][i:i + horizon],
             np.concatenate([data['states'][i:i+1], data['next_states'][i:i+horizon]], axis=0))
            for i in indices[:limit]]


def prediction_evaluation(model, data, global_damping):
    samples = window_samples(data)
    metrics = []
    for name, predictor in [('global_physics', PhysicsPredictor(damping=global_damping)), ('learned_world_model', model)]:
        errors = {h: [] for h in (1, 5, 10, 16)}
        for state, actions, truth in samples:
            rollout = predictor.rollout(state, actions[None])[..., :2].mean(axis=0)[0]
            for h in errors:
                errors[h].append(float(np.sum((rollout[h] - truth[h, :2]) ** 2)))
        metrics.extend({'model': name, 'horizon': h, 'rmse': float(np.sqrt(np.mean(values))),
                        'windows': len(values), 'unit': 'map units'} for h, values in errors.items())
    return metrics


def calibrate_threshold(model, validation):
    disagreements, errors = [], []
    for state, actions, truth in window_samples(validation, limit=160):
        predicted = model.rollout(state, actions[None])[:, 0]
        mean = predicted.mean(0)
        spread = np.sqrt(np.mean(np.sum((predicted[..., :2] - mean[None, :, :2]) ** 2, axis=-1), axis=0))
        for h in (5, 10, 16):
            disagreements.append(float(spread[h]))
            errors.append(float(np.linalg.norm(mean[h, :2] - truth[h, :2])))
    # This is a validation-only heuristic. Not calibrated probability/coverage.
    threshold = max(.003, float(np.quantile(disagreements, .65)))
    if np.std(disagreements) > 1e-10 and np.std(errors) > 1e-10:
        correlation = float(np.corrcoef(disagreements, errors)[0, 1])
    else:
        correlation = None
    return threshold, {'rule': '65th percentile of ensemble position spread on held-out validation windows',
                       'threshold': threshold, 'pearson_spread_error': correlation,
                       'samples': len(errors), 'mean_position_error': float(np.mean(errors)),
                       'is_calibrated_probability': False}


def run_episode(method, model, global_damping, threshold, budget, seed, scenario, scale, replay=False):
    if method == 'global_physics':
        predictor = PhysicsPredictor(damping=global_damping)
    elif method == 'local_identification':
        predictor = PhysicsPredictor(damping=global_damping, adaptive=True)
    else:
        predictor = model
    predictor.reset()
    horizon = int(method.split('_')[-1]) if method.startswith('fixed_') else 10
    planner = MPCPlanner(predictor, horizon=horizon, adaptive=method == 'adaptive',
                         budget=budget, seed=seed + 70000, uncertainty_threshold=threshold)
    env = NavigationEnv(seed=seed, scenario=scenario, damping_scale=scale)
    state = env.reset()
    timings, calls, horizons, frames, actions = [], [], [], [], []
    path_length, minimum_clearance, fallbacks = 0., float('inf'), 0
    terminal_info = {}
    for step in range(100):
        action, info = planner.act(state)
        next_state, reward, terminated, truncated, terminal_info = env.step(action)
        predictor.observe(state, action, next_state)
        timings.append(float(info['planning_ms']))
        calls.append(int(info['model_steps']))
        horizons.append(int(info['horizon']))
        fallbacks += int(bool(info.get('fallback', False)))
        path_length += float(np.linalg.norm(next_state[:2] - state[:2]))
        minimum_clearance = min(minimum_clearance, float(terminal_info.get('min_clearance', 0)))
        actions.append(np.asarray(action).tolist())
        if replay:
            frame = {key: info[key] for key in ['horizon','planning_ms','model_steps','disagreement','predicted_path','candidate_paths'] if key in info}
            frame['state'] = state
            frame['action'] = action
            if 'candidate_paths' in frame:
                frame['candidate_paths'] = frame['candidate_paths'][:3]
            frames.append(compact(frame))
        state = next_state
        if terminated or truncated:
            break
    if replay:
        frames.append({'state': compact(state), 'action': [0, 0], 'horizon': 0, 'planning_ms': 0,
                       'model_steps': 0, 'disagreement': 0, 'predicted_path': [], 'candidate_paths': []})
    row = {'method': method, 'condition': 'in_distribution' if scale == 1 else 'damping_shift',
           'seed': seed, 'scenario': scenario, 'damping_scale': scale,
           'success': bool(terminal_info.get('success', False)),
           'collision': bool(terminal_info.get('collision', False)),
           'timeout': bool(terminal_info.get('timeout', False)), 'steps': len(timings),
           'planning_ms': timings, 'model_steps': calls, 'horizons': horizons,
           'path_length': path_length, 'minimum_clearance': minimum_clearance, 'fallback_count': fallbacks}
    episode = {'id': f'{method}-{scale}-{seed}', **{k: row[k] for k in ['method','condition','seed','success','collision']}, 'frames': frames}
    return row, episode


def summarize(rows):
    output = []
    for condition in sorted({r['condition'] for r in rows}):
        for method in dict.fromkeys(r['method'] for r in rows):
            subset = [r for r in rows if r['method'] == method and r['condition'] == condition]
            if not subset:
                continue
            times = [t for r in subset for t in r['planning_ms']]
            calls = [c for r in subset for c in r['model_steps']]
            output.append({'method': method, 'condition': condition, 'episodes': len(subset),
                           'success_rate': float(np.mean([r['success'] for r in subset])),
                           'collision_rate': float(np.mean([r['collision'] for r in subset])),
                           'timeout_rate': float(np.mean([r['timeout'] for r in subset])),
                           'mean_steps': float(np.mean([r['steps'] for r in subset])),
                           'planning_ms_p50': float(np.median(times)), 'planning_ms_p95': float(np.quantile(times, .95)),
                           'mean_model_steps': float(np.mean(calls)),
                           'horizon_counts': {str(h): sum(r['horizons'].count(h) for r in subset) for h in (5,10,16)}})
    return output


def run(args):
    output = Path(args.output)
    if (output / 'protocol.json').exists():
        raise FileExistsError(f'Experiment already exists: {output}. Choose a new --output directory to preserve evidence.')
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    seeds = {'training_data': 1100, 'validation_data': 5100, 'prediction_test': 9100,
             'planner_validation': 12000, 'navigation_test': args.navigation_seed}
    config = {'stage': 'pilot', 'training_seed': args.seed, 'training_episodes': args.train_episodes,
              'validation_episodes': 50, 'prediction_test_episodes': 50, 'test_episodes_per_condition': args.episodes,
              'budget': args.budget, 'horizons': [5,10,16], 'epochs': args.epochs, 'dt': DT,
              'damping_scales': [1.0,1.7], 'split_seeds': seeds, 'threshold_rule': 'validation spread percentile 0.65',
              'model_type': 'hybrid learned damping with known integration/obstacle dynamics',
              'main_question': 'Does learned spatial damping improve prediction/control; does disagreement-driven horizon add benefit?',
              'limitations': ['one independent training run; ensemble members are not independent experiment runs',
                             'pilot tests are diagnostic; do not tune on these seeds and reuse them as final paper test',
                             'history-based physical baseline receives recent transitions; frozen learned model is stateless',
                             'known obstacle dynamics and kinematic structure retained; not fully learned state/video prediction',
                             'model-step budget compares ensemble members; real latency differs across predictor families']}
    try:
        config['git_commit_before_run'] = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    except Exception:
        config['git_commit_before_run'] = None
    config['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('foresight').glob('*.py'))}
    write_json(output / 'protocol.json', config)
    print('Collecting disjoint episode datasets...', flush=True)
    train = collect_dataset(episodes=args.train_episodes, seed=seeds['training_data'])
    validation = collect_dataset(episodes=50, seed=seeds['validation_data'])
    test = collect_dataset(episodes=50, seed=seeds['prediction_test'])
    data_path = Path('data/processed')
    data_path.mkdir(parents=True, exist_ok=True)
    for name, data in [('train',train),('validation',validation),('prediction_test',test)]:
        np.savez_compressed(data_path / f'day0_{name}.npz', **data)
    print(f"Training with {len(train['states'])} transitions...", flush=True)
    model, training = train_world_model(train, validation, seed=args.seed, epochs=args.epochs)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path('checkpoints') / f'{output.name}_seed{args.seed}.npz'
    model.save(checkpoint)
    training['checkpoint'] = str(checkpoint)
    training['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training['all_training_transitions'] = len(train['states'])
    training['validation_transitions'] = len(validation['states'])
    global_damping = training['global_damping_fit']
    threshold, calibration = calibrate_threshold(model, validation)
    metrics = prediction_evaluation(model, test, global_damping)
    write_json(output / 'training.json', training)
    write_json(output / 'prediction.json', {'metrics': metrics, 'calibration': calibration})
    print(f'Validation-derived spread threshold={threshold:.5f}', flush=True)
    # Choose fixed horizon on separate planner-validation seeds before test.
    validation_rows = []
    for horizon in (5,10,16):
        for i in range(4):
            row, _ = run_episode(f'fixed_{horizon}', model, global_damping, threshold, args.budget,
                                 12000 + i, ['open','crossing','slalom'][i % 3], 1.)
            validation_rows.append(row)
    val_summary = summarize(validation_rows)
    chosen = max(val_summary, key=lambda r:(r['success_rate'], -r['collision_rate'], -r['mean_steps']))['method']
    write_json(output / 'planner-validation.json', {'results': val_summary, 'selected_fixed_method': chosen, 'episodes': validation_rows})
    print(f'Validation-selected fixed baseline: {chosen}', flush=True)
    methods = ['global_physics','local_identification','fixed_5','fixed_10','fixed_16','adaptive']
    rows, replays = [], []
    for scale in (1.,1.7):
        for method in methods:
            for i in range(args.episodes):
                row, replay = run_episode(method, model, global_damping, threshold, args.budget,
                                          args.navigation_seed+i, ['open','crossing','slalom'][i % 3], scale, replay=(i == 1))
                rows.append(row)
                if replay['frames']:
                    replays.append(replay)
            print(f"evaluated {method:22s} scale={scale}: {sum(r['success'] for r in rows[-args.episodes:])}/{args.episodes} success", flush=True)
            write_json(output / 'episodes.json', rows)
    results = summarize(rows)
    adaptive_id = next(r for r in results if r['method']=='adaptive' and r['condition']=='in_distribution')
    fixed_id = next(r for r in results if r['method']==chosen and r['condition']=='in_distribution')
    gains = adaptive_id['success_rate'] - fixed_id['success_rate']
    decision = {'status': 'needs_more_evidence',
                'text': '模型训练、规划与真实回放已形成闭环。自适应规划是否有效，需结合强物理基线和更多独立训练种子判断。',
                'selected_fixed_method': chosen, 'adaptive_success_delta_id': gains,
                'independent_training_runs': 1, 'is_final_thesis_result': False}
    summary = {'stage':'pilot','generated_at':datetime.now(timezone.utc).isoformat(),
               'training':training,'prediction_metrics':metrics,'calibration':calibration,
               'results':results,'decision':decision,'config':config,
               'total_seconds':time.perf_counter()-start}
    write_json(output / 'summary.json', summary)
    # Compact replay JSON remains an actual trace, never synthesized trajectories.
    (output / 'replays.json').write_text(json.dumps({'episodes':replays,'meta':{'map_size':10,'robot_radius':ROBOT_RADIUS,'obstacle_radius':OBSTACLE_RADIUS,'dt':DT}},ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    report = ['# Day-zero feasibility run', '', 'This is a pilot with one independent training run, not final thesis evidence.', '',
              f"Validation-selected fixed horizon: `{chosen}`. Total runtime: {summary['total_seconds']:.1f} seconds.", '',
              '| Method | Condition | N | Success | Collision | P95 planning ms |', '|---|---|---:|---:|---:|---:|']
    for row in results:
        report.append(f"| {row['method']} | {row['condition']} | {row['episodes']} | {row['success_rate']:.1%} | {row['collision_rate']:.1%} | {row['planning_ms_p95']:.2f} |")
    report.extend(['', '## Boundaries', '', *['- '+x for x in config['limitations']]])
    (output / 'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Finished: {output}/summary.json in {summary["total_seconds"]:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/local-run')
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--train-episodes',type=int,default=240)
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--budget',type=int,default=4500)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--navigation-seed',type=int,default=20000)
    parser.add_argument('--checkpoint',default=None)
    args=parser.parse_args()
    if args.episodes < 2 or args.train_episodes < 10 or args.epochs < 1 or args.budget < 1000:
        parser.error('Require episodes>=2, train-episodes>=10, epochs>=1, budget>=1000')
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
