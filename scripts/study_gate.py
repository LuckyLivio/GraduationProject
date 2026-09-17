"""Frozen protocol for nominal-preserving gate calibration and boundary checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from foresight.benchmark import BenchmarkWorld, CONDITIONS, SHIFT_STEP, RECOVER_STEP, sensed_state, collect_prediction_episodes
from foresight.data import collect_dataset
from foresight.model import HybridWorldModel
from foresight.residual import ResidualCalibratedModel
from foresight.gated import GatedCalibratedModel, MatchedHistoryPhysicsModel, gate_threshold_from_validation
from foresight.planner import MPCPlanner
from foresight.experiment import write_json, compact

METHODS=('frozen_learned','residual_calibrated','gated_calibrated','local_identification')
TRAINING_SEEDS=(142,242,342)


def make_model(name,base,threshold,global_damping):
    if name=='frozen_learned':return base
    if name=='residual_calibrated':return ResidualCalibratedModel(base)
    if name=='gated_calibrated':return GatedCalibratedModel(base,threshold=threshold)
    return MatchedHistoryPhysicsModel(damping=global_damping)


def episode(method,base,threshold,global_damping,condition,seed,replay=False):
    model=make_model(method,base,threshold,global_damping);model.reset()
    env=BenchmarkWorld(seed,('open','crossing','slalom')[seed % 3],condition)
    truth=env.reset(seed=seed);observed=sensed_state(truth,condition,seed,0)
    planner=MPCPlanner(model,horizon=10,budget=4500,seed=seed+90000)
    rows=[];frames=[]
    for t in range(100):
        # Diagnostic next-state prediction is already in the selected rollout;
        # it incurs no extra planning/model evaluations.
        decision_started=time.perf_counter()
        action,info=planner.act(observed)
        decision_ms=(time.perf_counter()-decision_started)*1000
        pre_gate=bool(getattr(model,'gate_active',False))
        pre_scale=float(getattr(model,'current_scale',1))
        pre_score=float(getattr(model,'gate_score',0))
        next_truth,_,terminated,truncated,terminal=env.step(action)
        next_observed=sensed_state(next_truth,condition,seed,t+1)
        started=time.perf_counter();model.observe(observed,action,next_observed)
        update_ms=(time.perf_counter()-started)*1000
        rows.append({'step':t,'gate_before':pre_gate,'gate_after':bool(getattr(model,'gate_active',False)),
                     'scale_before':pre_scale,'scale_after':float(getattr(model,'current_scale',1)),
                     'raw_scale':float(getattr(model,'raw_scale',getattr(model,'current_scale',1))),
                     'gate_score':float(getattr(model,'gate_score',0)),
                     'effective_scale':terminal['applied_scale'],
                     'one_step_position_error':float(np.linalg.norm(np.asarray(info['predicted_path'][1])[:2]-next_truth[:2])),
                     'planning_ms':info['planning_ms'],'decision_ms':decision_ms,'update_ms':update_ms,
                     'controller_ms':decision_ms+update_ms,'model_steps':info['model_steps'],
                     'update_member_evals':int(getattr(model,'last_calibration_model_evaluations',0))})
        if replay:
            frames.append(compact({'state':truth,'observed_state':observed,'action':action,
                                   'gate_active':pre_gate,'learned_scale':pre_scale,
                                   'gate_score':pre_score,'gate_score_after':rows[-1]['gate_score'],
                                   **{k:info[k] for k in ('predicted_path','candidate_paths','horizon','planning_ms','model_steps','disagreement')}}))
        truth,observed=next_truth,next_observed
        if terminated or truncated:break
    if replay:frames.append({'state':compact(truth),'action':[0,0],'horizon':0,'predicted_path':[],'candidate_paths':[]})
    row={'method':method,'condition':condition,'seed':seed,'scenario':env.scenario,
         'success':terminal['success'],'collision':terminal['collision'],'timeout':terminal['timeout'],
         'steps':len(rows),'trace':rows}
    return row,{'id':f'{method}-{condition}-{seed}','method':method,'condition':condition,'seed':seed,
                'success':terminal['success'],'collision':terminal['collision'],'frames':frames}


def aggregate(rows,episodes):
    results=[]
    for condition in CONDITIONS:
        for method in METHODS:
            subset=[r for r in rows if r['method']==method and r['condition']==condition]
            counts=[sum(r['success'] for r in subset if r['training_seed']==s) for s in TRAINING_SEEDS]
            traces=[t for r in subset for t in r['trace']]
            results.append({'method':method,'condition':condition,'episodes_per_training_seed':episodes,
                            'independent_training_runs':0 if method=='local_identification' else 3,
                            'repeated_evaluation_panels':3,'unique_test_layouts':episodes,
                            'success_counts_by_seed':counts,'mean_success_rate':float(np.mean(counts)/episodes),
                            'collision_counts_by_seed':[sum(r['collision'] for r in subset if r['training_seed']==s) for s in TRAINING_SEEDS],
                            'mean_collision_rate':float(np.mean([r['collision'] for r in subset])),
                            'controller_ms_p95':float(np.quantile([t['controller_ms'] for t in traces],.95)),
                            'mean_model_steps':float(np.mean([t['model_steps'] for t in traces])),
                            'mean_update_member_evals':float(np.mean([t['update_member_evals'] for t in traces])),
                            'gate_active_step_fraction':float(np.mean([t['gate_after'] for t in traces])) if method=='gated_calibrated' else None,
                            'any_gate_activation_episodes':sum(any(t['gate_after'] for t in r['trace']) for r in subset) if method=='gated_calibrated' else None,
                            'total_episode_runs':len(subset)})
    return results


def detection_summary(rows):
    output=[]
    for row in rows:
        if row['method']!='gated_calibrated' or row['condition'] not in ('global_shift','switch_recover'):continue
        start=0 if row['condition']=='global_shift' else SHIFT_STEP
        trace=row['trace']
        end=len(trace) if row['condition']=='global_shift' else min(RECOVER_STEP,len(trace))
        was_active_before=bool(trace[start-1]['gate_after']) if start and len(trace)>=start else False
        after=[t['step'] for t in trace if start<=t['step']<end and t['gate_after']]
        recovery=[t['step'] for t in trace if t['step']>=RECOVER_STEP and not t['gate_after']]
        active_on_return=bool(trace[RECOVER_STEP-1]['gate_after']) if len(trace)>=RECOVER_STEP else None
        output.append({'training_seed':row['training_seed'],'condition':row['condition'],'seed':row['seed'],
                       'exposed_shift':len(trace)>start,'already_active_before_shift':was_active_before,
                       'detection_observations':None if not after or was_active_before else after[0]-start+1,
                       'shift_observation_count':max(0,end-start),
                       'exposed_recovery':row['condition']=='switch_recover' and len(trace)>RECOVER_STEP,
                       'active_before_recovery':active_on_return if row['condition']=='switch_recover' else None,
                       'recovery_observations':recovery[0]-RECOVER_STEP+1 if row['condition']=='switch_recover' and recovery and active_on_return else None})
    return output


def prediction_rows(models,calibration,global_damping,traces_by_condition):
    rows=[]
    for training_seed,base in models.items():
        threshold=calibration['models'][str(training_seed)]['threshold']
        for condition,traces in traces_by_condition.items():
            for method in METHODS:
                model=make_model(method,base,threshold,global_damping)
                for trace in traces:
                    model.reset()
                    states,obs,actions=trace['states'],trace['observations'],trace['actions']
                    for i in range(len(actions)):
                        if i>=8 and i%4==0 and i+16<=len(actions):
                            prediction=model.rollout(obs[i],actions[i:i+16][None]).mean(0)[0]
                            for h in (1,5,10,16):
                                rows.append({'training_seed':training_seed,'method':method,'condition':condition,
                                             'episode_seed':trace['seed'],'start_step':i,'horizon':h,
                                             'position_squared_error':float(np.sum((prediction[h,:2]-states[i+h,:2])**2)),
                                             'gate_active':bool(getattr(model,'gate_active',False))})
                        model.observe(obs[i],actions[i],obs[i+1])
    return rows


def run(args):
    output=ROOT/args.output
    if (output/'protocol.json').exists():raise FileExistsError(f'Choose new output directory; {output} already contains evidence')
    output.mkdir(parents=True,exist_ok=True)
    model_dir=ROOT/'artifacts/models';model_dir.mkdir(parents=True,exist_ok=True)
    models={}
    for s in TRAINING_SEEDS:
        dest=model_dir/f'seed{s}.npz'
        if not dest.exists():shutil.copyfile(ROOT/f'checkpoints/seed{s}_seed{s}.npz',dest)
        models[s]=HybridWorldModel.load(dest)
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'stage':'gate_validation','training_seeds':list(TRAINING_SEEDS),
              'models':'frozen ensembles trained before this gate study; no retraining or selection on this test panel',
              'hypothesis':'preserve nominal prediction while retaining global-shift adaptation; inspect failure under local/noisy/recovering conditions',
              'threshold_rule':'nominal-validation q99 of absolute log WLS scale, floor .03; persistence3,min_samples3,off_ratio.6',
              'validation_collection_seed':60100,'validation_episodes':50,'navigation_seeds':list(range(70000,70000+args.episodes)),
              'prediction_seeds':list(range(75000,75024)), 'conditions':list(CONDITIONS),'methods':list(METHODS),
              'condition_definitions':{'nominal':'scale1','global_shift':'scale1.7','local_patch':'1+1.2*exp(-((x-5.2)/1.3)^2-((y-5)/3)^2)',
                                      'switch_recover':'scale1.7 for transitions8..23 then scale1','sensor_noise':'scale1, iid robot position sigma.005 and velocity sigma.02; other state exact'},
              'horizon':10,'budget':4500,'update_budget':'separately count neural member evaluations, controller time includes observe updates',
              'valid_filter':'all online methods use model.transition_targets; retain five valid past transitions',
              'prediction_measure':'same actions, open loop; online histories exclude future; truth for evaluator only',
              'seed_layout_convention':'explicit reset(seed), first layout',
              'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'foresight').glob('*.py'))},
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checkpoint_sha256':{str(s):hashlib.sha256((model_dir/f'seed{s}.npz').read_bytes()).hexdigest() for s in TRAINING_SEEDS},
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'cautions':['validation percentile not trajectory false-alarm probability','three seeds and paired repeated layouts, not pooled iid samples','noise test is nominal noisy dynamics only','no threshold changes after inspecting test results']}
    write_json(output/'protocol.json',protocol)
    started=time.perf_counter()
    validation=collect_dataset(episodes=50,seed=60100)
    # Verify no exact episode seed collisions with saved train/validation/test.
    sets=[]
    for path in sorted((ROOT/'data/processed').glob('day0_*.npz')):
        with np.load(path,allow_pickle=False) as data:sets.extend(data['episode_seeds'].tolist())
    assert not set(validation['episode_seeds'])&set(sets)
    assert not set(protocol['navigation_seeds']+protocol['prediction_seeds'])&set(sets+validation['episode_seeds'].tolist())
    calibration={'models':{},'demo_training_seed':142,'validation_seed':60100}
    for s,base in models.items():
        threshold,metadata=gate_threshold_from_validation(base,validation)
        calibration['models'][str(s)]={'threshold':float(threshold),**metadata}
        print(f'Frozen threshold seed{s}: {threshold:.5f}',flush=True)
    write_json(output/'calibration.json',calibration)
    global_damping=json.loads((ROOT/'artifacts/day0/summary.json').read_text(encoding='utf-8'))['training']['global_damping_fit']
    rows=[];replays=[]
    for s,base in models.items():
        seed_rows=[]
        for condition in CONDITIONS:
            for method in METHODS:
                method_rows=[]
                for seed in protocol['navigation_seeds']:
                    row,replay=episode(method,base,calibration['models'][str(s)]['threshold'],global_damping,condition,seed,
                                       replay=s==142 and seed==70000 and method=='gated_calibrated')
                    row['training_seed']=s;rows.append(row);seed_rows.append(row);method_rows.append(row)
                    if replay['frames']:replays.append(replay)
                print(f'{s} {condition:15s} {method:22s} {sum(r["success"] for r in method_rows)}/{args.episodes}',flush=True)
        write_json(output/f'seed{s}-episodes.json',seed_rows)
    traces={c:collect_prediction_episodes(c) for c in CONDITIONS}
    pred=prediction_rows(models,calibration,global_damping,traces)
    write_json(output/'prediction-windows.json',pred)
    pred_summary=[]
    for condition in CONDITIONS:
        for method in METHODS:
            for h in (1,5,10,16):
                selected=[r for r in pred if r['condition']==condition and r['method']==method and r['horizon']==h]
                values=[float(np.sqrt(np.mean([r['position_squared_error'] for r in selected if r['training_seed']==s]))) for s in TRAINING_SEEDS]
                pred_summary.append({'condition':condition,'method':method,'horizon':h,'rmse_by_training_seed':values,
                                     'mean_rmse':float(np.mean(values)),'unique_episode_count':len({r['episode_seed'] for r in selected}),
                                     'windows_per_seed':len(selected)//3})
    summary={'stage':'gate_validation','training_seeds':list(TRAINING_SEEDS),'total_episode_runs':len(rows),
             'results':aggregate(rows,args.episodes),'prediction':pred_summary,'calibration':calibration,
             'total_seconds':time.perf_counter()-started,
             'decision':{'text':'门控是否保留原分布精度、何时失效，请结合全部条件与逐种子结果判断；不据此宣称原创或普遍更优。'}}
    write_json(output/'summary.json',summary)
    write_json(output/'detection.json',detection_summary(rows))
    (output/'replays.json').write_text(json.dumps({'episodes':replays,'meta':{'map_size':10,'dt':.15,'robot_radius':.18,'obstacle_radius':.38}},ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(f'Gate study finished: {len(rows)} episodes, {summary["total_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/gated-study')
    parser.add_argument('--episodes',type=int,default=18)
    args=parser.parse_args()
    if args.episodes<3:parser.error('episodes must be at least3')
    with threadpool_limits(limits=1):run(args)
