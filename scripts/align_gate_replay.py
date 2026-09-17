"""Derive temporally aligned display frames without changing raw study evidence.

The first study runner recorded score AFTER observing a transition beside the
BEFORE-transition gate and scale. Metrics are unaffected. Preserve the raw file
and make the display-only correction explicit and reproducible.
"""
import argparse
import hashlib
import json
from pathlib import Path


def align(directory):
    source = directory / 'replays.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    for episode in data['episodes']:
        previous_score = 0.0
        for frame in episode['frames']:
            if 'gate_score' not in frame:
                continue
            after = frame.get('gate_score_after', frame['gate_score'])
            frame['gate_score_after'] = after
            frame['gate_score'] = previous_score
            previous_score = after
    data['meta']['display_alignment'] = {
        'source': 'replays.json',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'change': 'gate_score shifted to pre-action evidence; original retained as gate_score_after',
        'scope': 'display only; no states, actions, trajectories, success or error metrics modified',
    }
    target = directory / 'replays-aligned.json'
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('artifacts/gated-study'))
    align(parser.parse_args().input)
