"""Three independently initialized ensembles on a fresh, frozen test panel.

The training dataset is held constant to measure training randomness. Outcomes
share paired scene seeds; 72 run-scene observations are not 72 unique layouts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=24)
    parser.add_argument('--output',default='artifacts/replication')
    args = parser.parse_args()
    folder = Path(args.output)
    if (folder/'protocol.json').exists():
        raise FileExistsError(f'Experiment already exists: {folder}. Choose a new --output directory.')
    folder.mkdir(parents=True,exist_ok=True)
    protocol = {'training_seeds':[142,242,342], 'navigation_seed_start':30000,
                'episodes_per_condition_per_training_seed':args.episodes,
                'data':'same frozen collection seeds as day0; train data fixed across training seeds',
                'frozen_hypothesis':'Ensemble-disagreement horizon selection adds value over validation-selected fixed horizon at same model-step budget.',
                'independence':'three independent training runs; each run trains a three-member ensemble; shared paired test layouts',
                'day0_used_to_change_hyperparameters':False,
                'status':'predeclared-before-replication'}
    (folder/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    summaries=[]
    for seed in protocol['training_seeds']:
        subprocess.run([sys.executable,'-u','-m','foresight.experiment','--output',str(folder/f'seed{seed}'),
                        '--seed',str(seed),'--navigation-seed','30000','--episodes',str(args.episodes),
                        '--train-episodes','240','--epochs','80','--budget','4500'],check=True)
        summaries.append(json.loads((folder/f'seed{seed}'/'summary.json').read_text(encoding='utf-8')))
    aggregate=[]
    for first in summaries[0]['results']:
        rows=[next(r for r in s['results'] if r['method']==first['method'] and r['condition']==first['condition']) for s in summaries]
        aggregate.append({'method':first['method'],'condition':first['condition'],
                          'unique_test_layouts':args.episodes,'independent_training_runs':3,
                          'episodes_across_runs':sum(r['episodes'] for r in rows),
                          'success_rates_by_training_seed':[r['success_rate'] for r in rows],
                          'mean_success_rate':sum(r['success_rate'] for r in rows)/3,
                          'collision_rates_by_training_seed':[r['collision_rate'] for r in rows],
                          'mean_collision_rate':sum(r['collision_rate'] for r in rows)/3,
                          'planning_ms_p95_by_training_seed':[r['planning_ms_p95'] for r in rows]})
    result={'protocol':protocol,'results':aggregate,
            'selected_fixed_methods':[s['decision']['selected_fixed_method'] for s in summaries],
            'total_test_episode_runs':3*args.episodes*2*6,
            'note':'No superiority claim from pooled episode counts. Analyze paired scenes and training-seed variability.'}
    (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
