"""Controlled continued one-step versus open-loop multi-step training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from foresight.pushing_env import PushParams,simulate_push,wrap_angle
from foresight.pushing_model import PushWorldModel,training_arrays
from foresight.pushing_sequence import rollout_torch,pose_loss
from foresight.pushing_baselines import identify_parameters,predict_physics
from scripts.pushing_pilot import collect,error_metrics,plan_action,sha,write_json

SEEDS=[17,29,43]
LABELS={'original':'首轮历史模型','teacher':'继续单步训练','multistep':'多步展开训练','sysid':'在线物理辨识','oracle':'真实参数参考'}


def windows(data,horizon=5):
    states,actions=data['states'],data['actions']
    _,hh,_=training_arrays(states,actions)
    hh=hh.reshape(len(states),len(actions[0]),6,16)
    starts=np.arange(6,len(actions[0])-horizon+1)
    truth=np.stack([states[:,t:t+horizon+1] for t in starts],axis=1).reshape(-1,horizon+1,6)
    commands=np.stack([actions[:,t:t+horizon] for t in starts],axis=1).reshape(-1,horizon,3)
    history=hh[:,starts].reshape(-1,6,16)
    return truth,commands,history


def prepare(args):
    if (args.output/'manifest.json').exists():raise ValueError('Output exists; select a new directory')
    args.data.mkdir(parents=True,exist_ok=True)
    old_manifest=json.loads((ROOT/'artifacts/pushing-pilot/manifest.json').read_text(encoding='utf-8'))
    entries={}
    for split in ['train','validation']:
        path=args.training_data/f'{split}.npz'
        if not path.exists():
            info=old_manifest['splits'][split]
            args.training_data.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**collect(info['episodes'],info['steps_per_episode'],info['seed_start']))
        entries[split]={'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),'sha256':sha(path),
                        'matches_published':sha(path)==old_manifest['splits'][split]['sha256']}
        if not entries[split]['matches_published']:raise ValueError(f'{split} dataset differs from published pilot')
    fresh=collect(80,20,91000,ood_half=True)
    path=args.data/'evaluation.npz';np.savez_compressed(path,**fresh)
    entries['evaluation']={'episodes':80,'steps':20,'seed_start':91000,'sha256':sha(path),'contact_fraction':float(fresh['contacts'].mean())}
    initial={str(seed):sha(ROOT/f'artifacts/pushing-pilot/history-{seed}.npz') for seed in SEEDS}
    write_json(args.output/'manifest.json',{'datasets':entries,'initial_weights_sha256':initial,'protocol_sha256':sha(ROOT/'docs/pushing-multistep-protocol.md'),
              'frozen_config':{'seeds':SEEDS,'updates':args.updates,'batch':256,'train_horizon':5,'validation_horizon':10,'lr':.0002},
              'notes':'Fresh development panel; repeated training seeds are not independent environment samples.'})


def validate(model,validation):
    # eval mode disables cuDNN GRU training workspaces; gradients never needed.
    model.eval()
    truth,actions,history=validation
    with torch.inference_mode():
        pred=rollout_torch(model,truth[:,0],actions,history)
        score=float(pose_loss(pred,truth))
    return score


def train(args):
    destination=args.output/'training.json'
    if destination.exists():raise ValueError('Training already exists; choose a new output')
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    torch.set_num_threads(4)
    dataset=np.load(args.training_data/'train.npz');valid=np.load(args.training_data/'validation.npz')
    tx,ta,th=[torch.as_tensor(x,device=device) for x in windows(dataset)]
    _,vh,_=training_arrays(valid['states'],valid['actions'])
    validation=[torch.as_tensor(x,device=device) for x in [valid['states'][:,6:17],valid['actions'][:,6:16],vh.reshape(120,20,6,16)[:,6]]]
    records=[]
    for seed in SEEDS:
        for method in ['teacher','multistep']:
            path=args.output/f'{method}-{seed}.npz'
            if path.exists():raise ValueError(f'Will not overwrite {path}')
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed+40000)
            model=PushWorldModel.load(ROOT/f'artifacts/pushing-pilot/history-{seed}.npz',device=device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.00001)
            base=validate(model,validation);best=base;best_update=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            started=time.perf_counter();curve=[{'update':0,'validation_loss':base,'elapsed_seconds':0}]
            for update in range(1,args.updates+1):
                ids=torch.as_tensor(rng.integers(len(tx),size=256),device=device)
                truth,actions,history=tx[ids],ta[ids],th[ids]
                model.train()
                pred=rollout_torch(model,truth[:,0],actions,history,teacher_states=truth if method=='teacher' else None)
                loss=pose_loss(pred,truth)
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                if update%100==0:
                    score=validate(model,validation)
                    row={'update':update,'train_loss':float(loss.detach()),'validation_loss':score,'elapsed_seconds':time.perf_counter()-started}
                    curve.append(row)
                    if score<best:
                        best=score;best_update=update;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    print(json.dumps({'method':method,'seed':seed,**row}),flush=True)
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            np.savez_compressed(path,**{k:v.numpy() for k,v in best_state.items()})
            records.append({'method':method,'seed':seed,'updates':args.updates,'best_update':best_update,'initial_validation_loss':base,
                'best_validation_loss':best,'training_seconds':elapsed,'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0,
                'parameters':sum(p.numel() for p in model.parameters()),'trained_model_steps':args.updates*256*5,
                'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'weights_sha256':sha(path),'curve':curve})
            write_json(destination,{'runs':records,'manifest_sha256':sha(args.output/'manifest.json')})


def model_for(args,method,seed):
    path=ROOT/f'artifacts/pushing-pilot/history-{seed}.npz' if method=='original' else args.output/f'{method}-{seed}.npz'
    return PushWorldModel.load(path)


def evaluate(args):
    if (args.output/'prediction.json').exists():raise ValueError('Existing prediction results')
    data=np.load(args.data/'evaluation.npz');s,a=data['states'],data['actions']
    torch.set_num_threads(1)
    models={(method,seed):model_for(args,method,seed) for method in ['original','teacher','multistep'] for seed in SEEDS}
    rows=[];cases=[]
    for e in range(80):
        hs,ha=s[e,:7],a[e,:6];future=a[e,6:16];truth=s[e,6:17]
        parameters,ident=identify_parameters(hs,ha)
        predictions={('sysid',None):predict_physics(hs[-1],future,parameters)}
        for key,model in models.items():predictions[key]=model.rollout(hs[-1],future,hs,ha)
        for (method,seed),pred in predictions.items():
            for horizon in [1,5,10]:
                rows.append({'episode':e,'split':'id' if e<40 else 'ood','method':method,'seed':seed,'horizon':horizon,
                              **error_metrics(pred[:horizon+1],truth[:horizon+1])})
        cases.append({'id':str(e),'label':f'新场景 {e}','initial':hs[-1].tolist(),'truth':truth.tolist(),
                      'actions':future.tolist(),'history_count':6,'identification':ident,
                      'predictions':{m:predictions[m,17].tolist() for m in ['original','teacher','multistep']}|{'sysid':predictions['sysid',None].tolist()}})
        if e%20==19:print(json.dumps({'evaluated':e+1}),flush=True)
    summary=[]
    for split in ['all','id','ood']:
        for method in ['original','teacher','multistep','sysid']:
            for horizon in [1,5,10]:
                selected=[r for r in rows if r['method']==method and r['horizon']==horizon and (split=='all' or r['split']==split)]
                summary.append({'method':method,'label':LABELS[method],'split':split,'horizon':horizon,'independent_scenarios':len({r['episode'] for r in selected}),
                     **{key:float(np.mean([r[key] for r in selected])) for key in ['position_rmse','angle_mae','omega_rmse']}})
    worst=int(np.argmax([error_metrics(c['predictions']['multistep'],c['truth'])['position_rmse'] for c in cases]))
    cases[0]['label']='固定首例 · 新场景 0';cases[worst]['label']=f'多步模型最大误差例 · 新场景 {worst}' if worst else cases[0]['label']
    write_json(args.output/'prediction.json',{'rows':rows,'summary':summary})
    write_json(args.output/'all_cases.json',cases)
    write_json(args.output/'demo.json',{'status':'新场景上的多步训练复核','description':'同一训练数据、初始模型、预测次数；所有方法评估时均不读取真实未来。',
        'summary':[r for r in summary if r['split']=='all'],'cases':[cases[e] for e in dict.fromkeys([0,worst])],'planning_summary':[]})
    print(json.dumps({'summary':[r for r in summary if r['split']=='all' and r['horizon']==10]}),flush=True)


def planning(args):
    if (args.output/'planning.json').exists():raise ValueError('Existing planning results')
    data=np.load(args.data/'evaluation.npz');torch.set_num_threads(1)
    models={(m,seed):model_for(args,m,seed) for m in ['original','teacher','multistep'] for seed in SEEDS}
    rows=[]
    for e in [0,1,2,3,4,5,40,41,42,43,44,45]:
        initial=data['states'][e,6];params=PushParams(*data['parameters'][e])
        rng=np.random.default_rng(102000+e);direction=rng.uniform(-np.pi,np.pi);distance=rng.uniform(.25,.5)
        goal=initial[:3]+[distance*np.cos(direction),distance*np.sin(direction),rng.uniform(-.6,.6)]
        for method,seed in [('oracle',None),('sysid',None)]+list(models):
            path=[initial.copy()];actions=[];times=[];costs=[]
            hs=list(data['states'][e,:7].copy());ha=list(data['actions'][e,:6].copy())
            for step in range(12):
                start=time.perf_counter();fit_calls=0
                if method in ['oracle','sysid']:
                    p=params
                    if method=='sysid':p,diag=identify_parameters(np.array(hs[-7:]),np.array(ha[-6:]));fit_calls=diag['model_calls']
                    def predict(ss,aa):return np.array([simulate_push(x,u,p)[0] for x,u in zip(ss,aa)])
                else:
                    model=models[method,seed];context=model.history_array(hs,ha)
                    def predict(ss,aa):return model.predict_batch(ss,aa,context)
                action,diag=plan_action(path[-1],goal,predict)
                times.append((time.perf_counter()-start)*1000);costs.append(diag['model_calls']+fit_calls)
                nxt,_=simulate_push(path[-1],action,params);path.append(nxt);actions.append(action);hs.append(nxt);ha.append(action)
                pe=float(np.linalg.norm(nxt[:2]-goal[:2]));ae=float(abs(wrap_angle(nxt[2]-goal[2])))
                if pe<.08 and ae<.18:break
            rows.append({'episode':e,'method':method,'seed':seed,'goal':goal.tolist(),'states':np.array(path).tolist(),'actions':np.array(actions).tolist(),
                         'position_error':pe,'angle_error':ae,'success':pe<.08 and ae<.18,'steps':len(actions),'planning_ms':times,'model_calls':costs})
        print(json.dumps({'planning_scene':e,'completed_runs':len(rows)}),flush=True)
    summary=[]
    for method in ['oracle','sysid','original','teacher','multistep']:
        selected=[r for r in rows if r['method']==method]
        summary.append({'method':method,'label':LABELS[method],'success':sum(r['success'] for r in selected),'episodes':len(selected),
                        'independent_scenarios':12,'training_seeds':3 if method in ['original','teacher','multistep'] else 0,
                        'mean_steps':float(np.mean([r['steps'] for r in selected])),
                        'median_planning_ms':float(np.median([v for r in selected for v in r['planning_ms']]))})
    write_json(args.output/'planning.json',{'rows':rows,'summary':summary,'note':'Pose thresholds only, no stopped-state criterion. Six shared setup pushes excluded.'})
    demo=json.loads((args.output/'demo.json').read_text(encoding='utf-8'));demo['planning_summary']=summary
    write_json(args.output/'demo.json',demo)
    print(json.dumps({'planning_summary':summary}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','evaluate','planning'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/local-multistep-run')
    parser.add_argument('--training-data',type=Path,default=ROOT/'data/processed/pushing-pilot')
    parser.add_argument('--data',type=Path)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    args.data=args.data or ROOT/'data/processed'/args.output.name
    globals()[args.stage](args)
