"""Exploratory residual calibration after a negative horizon-selection result.

Frozen trained ensemble; exactly one new mechanism and fresh diagnostic seeds.
This does not establish novelty or general robustness.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from foresight.env import NavigationEnv
from foresight.model import HybridWorldModel
from foresight.planner import PhysicsPredictor, MPCPlanner
from foresight.residual import ResidualCalibratedModel
from foresight.data import collect_dataset
from foresight.experiment import summarize, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/residual-probe')
    args=parser.parse_args()
    output=ROOT/args.output
    if (output/'protocol.json').exists():
        raise FileExistsError(f'Experiment already exists: {output}. Choose a new --output directory.')
    output.mkdir(parents=True,exist_ok=True)
    day0=json.loads((ROOT/'artifacts/day0/summary.json').read_text(encoding='utf-8'))
    base=HybridWorldModel.load(ROOT/'artifacts/day0/model.npz')
    protocol={'stage':'exploratory_after_negative_result','model':'fixed day0 seed42 ensemble',
              'new_mechanism':'5-transition residual WLS multiplicative correction of learned spatial damping',
              'horizon':10,'budget':4500,'scales':[1.,1.7,2.2],'scene_seeds':list(range(40000,40018)),
              'prediction_data_seed':45100,'prediction_windows_per_scale':60,
              'known_scale_access':False,'observe_overhead':'logged separately; added for end-to-end controller time',
              'claim':'candidate engineering mechanism; novelty and generality unproven'}
    write_json(output/'protocol.json',protocol)
    methods=['frozen_learned','residual_calibrated','local_identification']
    def predictor(name):
        if name=='frozen_learned': return base
        if name=='residual_calibrated': return ResidualCalibratedModel(base)
        return PhysicsPredictor(damping=day0['training']['global_damping_fit'],adaptive=True)
    rows=[]
    for scale in protocol['scales']:
        for name in methods:
            for i,seed in enumerate(protocol['scene_seeds']):
                model=predictor(name); model.reset()
                env=NavigationEnv(seed=seed,scenario=['open','crossing','slalom'][i%3],damping_scale=scale)
                state=env.reset(seed=seed)
                planner=MPCPlanner(model,horizon=10,budget=4500,seed=seed+70000)
                times,calls,updates,horizons,scales=[],[],[],[],[]
                for step in range(100):
                    action,info=planner.act(state)
                    ns,_,done,truncated,terminal=env.step(action)
                    started=time.perf_counter(); model.observe(state,action,ns)
                    update_ms=(time.perf_counter()-started)*1000
                    times.append(float(info['planning_ms'])+update_ms)
                    updates.append(update_ms); calls.append(info['model_steps']); horizons.append(10)
                    scales.append(float(getattr(model,'current_scale',1)))
                    state=ns
                    if done or truncated:break
                rows.append({'method':name,'condition':f'damping_x{scale:g}','seed':seed,'success':terminal['success'],
                             'collision':terminal['collision'],'timeout':terminal['timeout'],'steps':len(times),
                             'planning_ms':times,'observation_update_ms':updates,'model_steps':calls,'horizons':horizons,
                             'estimated_scales':scales,'calibration_model_evaluations':getattr(model,'calibration_model_evaluations',0)})
            print(f'{name} scale={scale}: {sum(r["success"] for r in rows[-18:])}/18',flush=True)
    prediction=[]
    for scale in protocol['scales']:
        data=collect_dataset(episodes=25,seed=45100,damping_scale=scale)
        indices=[i for i in range(5,len(data['states'])-16) if data['episode_ids'][i-5]==data['episode_ids'][i+15]]
        np.random.default_rng(7).shuffle(indices)
        for name in methods:
            model=predictor(name);errors=[]
            for i in indices[:60]:
                model.reset()
                for j in range(i-5,i):model.observe(data['states'][j],data['actions'][j],data['next_states'][j])
                rollout=model.rollout(data['states'][i],data['actions'][i:i+16][None]).mean(0)[0]
                errors.append([float(np.sum((rollout[h,:2]-data['next_states'][i+h-1,:2])**2)) for h in (1,5,10,16)])
            prediction.extend({'method':name,'scale':scale,'horizon':h,'rmse':float(np.sqrt(np.mean(np.asarray(errors)[:,k]))),
                               'windows':len(errors)} for k,h in enumerate((1,5,10,16)))
    write_json(output/'episodes.json',rows)
    result={'protocol':protocol,'results':summarize(rows),'prediction':prediction,'episode_runs':len(rows),
            'note':'One frozen ensemble; 18 paired layouts. Update time included, logical calibration evaluations reported separately. No superiority/novelty claim.'}
    write_json(output/'summary.json',result)
    print('Completed exploratory residual probe',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
