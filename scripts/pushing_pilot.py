"""Reproducible development pilot, not a confirmatory thesis benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from foresight.pushing_env import PushParams, simulate_push, sample_action, sample_state, wrap_angle
from foresight.pushing_model import PushWorldModel, training_arrays
from foresight.pushing_baselines import identify_parameters, predict_physics, goal_cost


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(count, steps, seed_start, ood_half=False):
    states = np.zeros((count, steps+1, 6), np.float32)
    actions = np.zeros((count, steps, 3), np.float32)
    contacts = np.zeros((count, steps), bool)
    parameters = np.zeros((count, 4), np.float64)
    for episode in range(count):
        rng = np.random.default_rng(seed_start+episode)
        p = [rng.uniform(.1, .8), rng.uniform(.15, .95), *rng.uniform(-.06, .06, 2)]
        if ood_half and episode >= count//2:
            p[0], p[2] = rng.choice([.05, .9]), rng.choice([-.07, .07])
        params = PushParams(*p)
        parameters[episode] = p
        state = sample_state(rng)
        states[episode, 0] = state
        for t in range(steps):
            action = sample_action(rng)
            state, info = simulate_push(state, action, params)
            states[episode, t+1] = state
            actions[episode, t] = action
            contacts[episode, t] = info['contact']
        if (episode+1) % 200 == 0:
            print(json.dumps({'collect':seed_start, 'episodes':episode+1}), flush=True)
    return dict(states=states, actions=actions, contacts=contacts, parameters=parameters)


def generate(args):
    if (args.output/'manifest.json').exists():
        raise ValueError('Output already contains a dataset manifest; choose a new version')
    start = time.perf_counter()
    args.data.mkdir(parents=True, exist_ok=True)
    entries = {}
    for split, count, seed in [('train', args.train_episodes, 52000), ('validation', 120, 62000), ('development', 40, 71000)]:
        data = collect(count, 20, seed, split=='development')
        path = args.data/f'{split}.npz'
        np.savez_compressed(path, **data)
        entries[split] = {'episodes':count, 'steps_per_episode':20, 'seed_start':seed,
                          'sha256':sha(path), 'contact_fraction':float(data['contacts'].mean())}
    base = PushParams(.4, .5, 0, 0)
    initial = np.zeros(6)
    action_sequence = np.array([[0, .55, .7], [2, -.45, .6], [1, .3, .5]])
    variants = [('nominal', base), ('damping_low', PushParams(.05,.5,0,0)),
                ('damping_high', PushParams(.9,.5,0,0)), ('friction_low', PushParams(.4,.1,0,0)),
                ('friction_high', PushParams(.4,1,0,0)), ('cog_left', PushParams(.4,.5,-.06,0)),
                ('cog_right', PushParams(.4,.5,.06,0))]
    paths = {name:predict_physics(initial, action_sequence, p).tolist() for name,p in variants}
    sensitivity = {'initial':initial.tolist(), 'actions':action_sequence.tolist(), 'paths':paths,
                   'parameters':{name:asdict(p) for name,p in variants},
                   'note':'Same observed start and actions. One-factor changes; this does not prove unique parameter identification.'}
    write_json(args.output/'sensitivity.json', sensitivity)
    write_json(args.output/'manifest.json', {'protocol':'docs/pushing-pilot-protocol.md', 'splits':entries,
               'generation_seconds':time.perf_counter()-start, 'parameters_used_for_training':False,
               'note':'Development screening; not final held-out thesis evaluation.'})
    print(json.dumps({'generated':entries, 'seconds':time.perf_counter()-start}), flush=True)


def tensors(data, device):
    return [torch.as_tensor(a, device=device) for a in data]


def train(args):
    tr = np.load(args.data/'train.npz')
    va = np.load(args.data/'validation.npz')
    arrays = training_arrays(tr['states'], tr['actions'])
    valid_arrays = training_arrays(va['states'], va['actions'])
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    torch.set_num_threads(4)
    train_t, valid_t = tensors(arrays, device), tensors(valid_arrays, device)
    args.weights.mkdir(parents=True, exist_ok=True)
    runs=[]
    for seed in args.seeds:
        for use_history in [False, True]:
            kind = 'history' if use_history else 'no_history'
            path = args.weights/f'{kind}-{seed}.pt'
            if path.exists():
                raise ValueError(f'Will not overwrite checkpoint: {path}')
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = PushWorldModel(use_history)
            model.set_normalization(arrays[0], arrays[2])
            model.to(device)
            optimizer=torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
            best_loss=float('inf'); best_weights=None; curves=[]
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            for update in range(1, args.updates+1):
                model.train()
                indices=torch.as_tensor(rng.integers(len(arrays[0]), size=512), device=device)
                x,h,y=[t[indices] for t in train_t]
                prediction=model(x,h)
                loss=((prediction-(y-model.y_mean)/model.y_std)**2).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nnorm=torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                if update==1 or update%100==0:
                    model.eval()
                    with torch.inference_mode():
                        x,h,y=valid_t
                        vl=float(((model(x,h)-(y-model.y_mean)/model.y_std)**2).mean())
                    record={'update':update, 'train_loss':float(loss.detach()), 'validation_loss':vl,
                            'elapsed_seconds':time.perf_counter()-started}
                    curves.append(record)
                    if vl<best_loss:
                        best_loss=vl;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        best_update=update
                    print(json.dumps({'model':kind,'seed':seed,**record}), flush=True)
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            meta={'kind':kind,'seed':seed,'updates':args.updates,'best_update':best_update,
                  'best_validation_loss':best_loss,'training_seconds':elapsed,
                  'parameters':sum(p.numel() for p in model.parameters()),'device':device,
                  'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
                  'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0,
                  'samples_per_second':args.updates*512/elapsed,'curve':curves}
            model.load_state_dict(best_weights);model.save(path,meta)
            # Small, portable parameters are published without pickle.
            export=args.output/f'{kind}-{seed}.npz'
            np.savez_compressed(export, **{k:v.numpy() for k,v in best_weights.items()})
            meta['checkpoint_sha256']=sha(path);meta['export_sha256']=sha(export)
            runs.append(meta)
            write_json(args.output/'training.json', {'runs':runs,'data_manifest_sha256':sha(args.output/'manifest.json')})


def error_metrics(prediction, truth):
    residual=np.asarray(prediction)[1:]-np.asarray(truth)[1:]
    residual[:,2]=wrap_angle(residual[:,2])
    return {'position_rmse':float(np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=-1)))),
            'angle_mae':float(np.mean(np.abs(residual[:,2]))),
            'omega_rmse':float(np.sqrt(np.mean(residual[:,5]**2)))}


def evaluate(args):
    data=np.load(args.data/'development.npz')
    states,actions=data['states'],data['actions']
    torch.set_num_threads(1)
    models={(kind,seed):PushWorldModel.load(args.weights/f'{kind}-{seed}.pt') for seed in args.seeds for kind in ['no_history','history']}
    rows=[]; cases=[]; started=time.perf_counter()
    for episode in range(len(states)):
        hs,ha=states[episode,:7], actions[episode,:6]
        state=hs[-1]; future=actions[episode,6:16]; truth=states[episode,6:17]
        identified, diag=identify_parameters(hs,ha)
        physics=predict_physics(state,future,identified)
        stationary=np.tile(state,(11,1))
        constant=stationary.copy()
        constant[:,:2]+=np.arange(11)[:,None]*.5*state[3:5]
        constant[:,2]=wrap_angle(state[2]+np.arange(11)*.5*state[5])
        shared={'sysid':physics,'stationary':stationary,'constant_velocity':constant}
        predictions={}
        for seed in args.seeds:
            for kind in ['no_history','history']:
                model=models[kind,seed]
                predictions[kind,seed]=model.rollout(state,future,hs,ha)
            # Another episode's history: intentionally wrong context, same forecast inputs.
            other=(episode+1)%len(states)
            predictions['wrong_history',seed]=models['history',seed].rollout(
                state,future,states[other,:7],actions[other,:6])
        for (kind,seed),prediction in predictions.items():
            for horizon in [1,5,10]:
                rows.append({'episode':episode,'split':'in_distribution' if episode<20 else 'out_of_distribution',
                             'method':kind,'seed':seed,'horizon':horizon,
                             **error_metrics(prediction[:horizon+1],truth[:horizon+1])})
        for kind,prediction in shared.items():
            for horizon in [1,5,10]:
                rows.append({'episode':episode,'split':'in_distribution' if episode<20 else 'out_of_distribution',
                             'method':kind,'seed':None,'horizon':horizon,
                             **error_metrics(prediction[:horizon+1],truth[:horizon+1])})
        first=args.seeds[0]
        cases.append({'id':str(episode),'label':f'Development {episode}', 'history_count':6,
                      'params_display':asdict(PushParams(*data['parameters'][episode])),
                      'initial':state.tolist(),'actions':future.tolist(),'truth':truth.tolist(),
                      'predictions':{**{k:predictions[k,first].tolist() for k in ['no_history','history']}, 'sysid':physics.tolist()},
                      'metrics':{**{k:error_metrics(predictions[k,first],truth) for k in ['no_history','history']}, 'sysid':error_metrics(physics,truth)},
                      'identification':diag})
        print(json.dumps({'evaluation_episode':episode,'seconds':time.perf_counter()-started}),flush=True)
    summary=[]
    for split in ['all','in_distribution','out_of_distribution']:
        for method in ['no_history','history','wrong_history','sysid','stationary','constant_velocity']:
            for horizon in [1,5,10]:
                rr=[r for r in rows if r['method']==method and r['horizon']==horizon and (split=='all' or r['split']==split)]
                summary.append({'split':split,'method':method,'horizon':horizon,
                    'independent_scenarios':len({r['episode'] for r in rr}),
                    **{k:float(np.mean([r[k] for r in rr])) for k in ['position_rmse','angle_mae','omega_rmse']}})
    # Action check: compare four commanded faces against independent true execution.
    action_checks=[]
    for episode in range(8):
        state=states[episode,6];hs=states[episode,:7];ha=actions[episode,:6]
        alternatives=np.array([[f,.4,.65] for f in range(4)])
        true=np.array([simulate_push(state,a,PushParams(*data['parameters'][episode]))[0] for a in alternatives])
        model=models['history',args.seeds[0]]
        pred=model.predict_batch(np.tile(state,(4,1)),alternatives,model.history_array(hs,ha))
        td=true[:,:2]-state[:2];pd=pred[:,:2]-state[:2]
        cosine=np.sum(td*pd,axis=1)/(np.linalg.norm(td,axis=1)*np.linalg.norm(pd,axis=1)+1e-9)
        action_checks.append({'episode':episode,'mean_displacement_cosine':float(cosine.mean()),
                              'true':true.tolist(),'predicted':pred.tolist(),'actions':alternatives.tolist()})
    write_json(args.output/'prediction.json', {'rows':rows,'summary':summary,'action_checks':action_checks,'elapsed_seconds':time.perf_counter()-started})
    worst=int(np.argmax([c['metrics']['history']['position_rmse'] for c in cases]))
    selected=[0]+([worst] if worst!=0 else [])
    cases[0]['label']='固定首例 · 开发场景 0'
    if worst:cases[worst]['label']=f'历史模型位置误差最大例 · 场景 {worst}'
    demo={'metadata':{'title':'接触推物 · 自训练首轮','status':'开发筛查，尚未验证主动试探创新',
                     'description':'模型预测状态渲染；推杆宏动作重定位；精确状态输入。',
                     'training_samples':json.loads((args.output/'manifest.json').read_text())['splits']['train']['episodes']*20,
                     'training_seed':args.seeds[0],'test_episodes':40},
          'geometry':{'half_width':.25,'half_height':.18},'cases':[cases[i] for i in selected],
          'planning':[], 'summary':{'sample_count':40,'methods':[r for r in summary if r['split']=='all' and r['horizon']==10]}}
    write_json(args.output/'all_cases.json',cases)
    write_json(args.output/'demo.json',demo)
    print(json.dumps({'prediction_summary':[r for r in summary if r['split']=='all' and r['horizon']==10]}),flush=True)


def candidate_actions():
    return np.array([[face,offset,speed] for face in range(4) for offset in [-.65,0,.65] for speed in [.2,.55,.9]])


def plan_action(state,goal,predict_batch,horizon=2):
    """Shared two-step beam search: 36 first actions, six beams, 36 second."""
    actions=candidate_actions()
    first=predict_batch(np.tile(state,(len(actions),1)),actions)
    cost=goal_cost(first[:,None,:],goal)
    elite=np.argsort(cost)[:6]
    starts=np.repeat(first[elite],len(actions),axis=0)
    commands=np.tile(actions,(len(elite),1))
    second=predict_batch(starts,commands)
    score=goal_cost(second[:,None,:],goal)+.12*np.repeat(cost[elite],len(actions))
    best=int(np.argmin(score))
    return actions[elite[best//len(actions)]], {'model_calls':len(first)+len(second),'forecast':[state.tolist(),first[elite[best//len(actions)]].tolist(),second[best].tolist()]}


def planning(args):
    data=np.load(args.data/'development.npz'); torch.set_num_threads(1)
    models={kind:PushWorldModel.load(args.weights/f'{kind}-{args.seeds[0]}.pt') for kind in ['no_history','history']}
    rows=[];first_runs=[];started=time.perf_counter()
    # Fixed indices established in code, no ranking or winner selection.
    for episode in [0,1,2,3,20,21,22,23]:
        initial=data['states'][episode,6].copy();params=PushParams(*data['parameters'][episode])
        rng=np.random.default_rng(82000+episode)
        goal=initial[:3]+np.array([rng.uniform(.25,.5),rng.uniform(-.3,.3),rng.uniform(-.6,.6)])
        for method in ['oracle','sysid','no_history','history']:
            states=[initial.copy()];actions=[];traces=[];latencies=[];calls=[]
            hs=list(data['states'][episode,:7].copy());ha=list(data['actions'][episode,:6].copy())
            for step in range(12):
                tick=time.perf_counter()
                fit_calls=0
                if method in ['oracle','sysid']:
                    p=params
                    if method=='sysid':
                        p,fit=identify_parameters(np.array(hs[-7:]),np.array(ha[-6:]));fit_calls=fit['model_calls']
                    def predict_batch(ss,aa):
                        return np.array([simulate_push(s,a,p)[0] for s,a in zip(ss,aa)])
                else:
                    model=models[method];context=model.history_array(hs,ha)
                    def predict_batch(ss,aa):return model.predict_batch(ss,aa,context)
                action,info=plan_action(states[-1],goal,predict_batch)
                latencies.append((time.perf_counter()-tick)*1000)
                next_state,_=simulate_push(states[-1],action,params)
                states.append(next_state);actions.append(action);hs.append(next_state);ha.append(action)
                traces.append(info['forecast']);calls.append(info['model_calls']+fit_calls)
                pe=float(np.linalg.norm(next_state[:2]-goal[:2]));ae=float(abs(wrap_angle(next_state[2]-goal[2])))
                if pe<.08 and ae<.18:break
            row={'episode':episode,'method':method,'label':{'oracle':'真实参数参考','sysid':'在线物理辨识','no_history':'无历史模型','history':'历史世界模型'}[method],
                 'initial':initial.tolist(),'goal':goal.tolist(),'states':np.array(states).tolist(),
                 'actions':np.array(actions).tolist(),'forecasts':traces,'success':pe<.08 and ae<.18,
                 'position_error':pe,'angle_error':ae,'steps':len(actions),'planning_ms':latencies,'model_calls':calls,
                 'success_definition':'Pose threshold at macro endpoint; not a stopped-state criterion. Six common initial history pushes are additional shared setup.'}
            rows.append(row)
            if episode==0:first_runs.append(row)
            print(json.dumps({'planning':episode,'method':method,'success':row['success'],'position':pe,'angle':ae}),flush=True)
    summary={m:{'success':sum(r['success'] for r in rows if r['method']==m),'episodes':8,
                'mean_position_error':float(np.mean([r['position_error'] for r in rows if r['method']==m])),
                'mean_steps':float(np.mean([r['steps'] for r in rows if r['method']==m])),
                'median_planning_ms':float(np.median([v for r in rows if r['method']==m for v in r['planning_ms']]))} for m in ['oracle','sysid','no_history','history']}
    write_json(args.output/'planning.json',{'rows':rows,'summary':summary,'elapsed_seconds':time.perf_counter()-started})
    demo=json.loads((args.output/'demo.json').read_text(encoding='utf-8'));demo['planning']=first_runs
    demo['summary']['planning']=summary;write_json(args.output/'demo.json',demo)
    print(json.dumps({'planning_summary':summary}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['generate','train','evaluate','planning'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/pushing-pilot')
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed/pushing-pilot')
    parser.add_argument('--weights',type=Path,default=ROOT/'checkpoints/pushing-pilot')
    parser.add_argument('--train-episodes',type=int,default=1200)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    globals()[args.stage](args)
