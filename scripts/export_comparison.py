"""Save all six predefined comparison demonstrations; no winner selection.

This is a deterministic diagnostic set, not a new independent benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from threadpoolctl import threadpool_limits
from foresight.comparison import build_comparison
from foresight.model import HybridWorldModel


def run(output):
    output = ROOT / output
    if (output / 'summary.json').exists():
        raise FileExistsError('Choose a new output directory to preserve existing evidence')
    output.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / 'artifacts/models/seed142.npz'
    base = HybridWorldModel.load(model_path)
    calibration = json.loads((ROOT / 'artifacts/gated-study/calibration.json').read_text(encoding='utf-8'))
    threshold = calibration['models']['142']['threshold']
    global_damping = json.loads((ROOT / 'artifacts/day0/summary.json').read_text(encoding='utf-8'))['training']['global_damping_fit']
    rows = []
    for condition in ('nominal', 'global_shift'):
        for action in ('cruise', 'coast', 'brake'):
            result = build_comparison(base, threshold, global_damping, seed=2026,
                                      condition=condition, action_mode=action, horizon=16)
            (output / f'{condition}-{action}.json').write_text(
                json.dumps(result, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
            for row in result['methods']:
                rows.append({'condition': condition, 'action_mode': action,
                             'method': row['id'], 'endpoint_error': row['endpoint_error'],
                             'position_rmse': row['position_rmse'],
                             'executed_steps': result['executed_steps']})
            print(condition, action, [(r['id'], round(r['endpoint_error'], 4)) for r in result['methods']])
    result = {
        'scope': 'six predefined interactive diagnostics; not held-out benchmark evidence or seed selection',
        'seed': 2026, 'training_seed': 142, 'horizon': 16,
        'checkpoint_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256((ROOT / 'foresight/comparison.py').read_bytes()).hexdigest(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'results': rows,
    }
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/comparison-demo')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
