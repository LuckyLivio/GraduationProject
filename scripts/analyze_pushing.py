"""Audit saved pilot measurements, derive figures and publish action branches."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from foresight.pushing_model import PushWorldModel


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'artifacts/pushing-pilot'
    prediction=load(folder/'prediction.json');matched=load(folder/'matched-prediction.json')
    planning=load(folder/'planning.json');training=load(folder/'training.json')
    manifest=load(folder/'manifest.json');cases=load(folder/'all_cases.json')
    for split,info in manifest['splits'].items():
        assert digest(ROOT/f'data/processed/pushing-pilot/{split}.npz')==info['sha256']
    for run in training['runs']+load(folder/'matched-training.json')['runs']:
        path=folder/f"{run['kind']}-{run['seed']}.npz"
        assert digest(path)==run['export_sha256']
        restored=PushWorldModel.load(path)
        assert sum(p.numel() for p in restored.parameters())==run['parameters']
    # Recalculate first-seed visual cases from arrays, not displayed metrics.
    for case in cases:
        truth=np.asarray(case['truth'])[1:]
        for method,path in case['predictions'].items():
            error=np.asarray(path)[1:,:2]-truth[:,:2]
            rmse=float(np.sqrt((error**2).sum(1).mean()))
            assert np.isclose(rmse,case['metrics'][method]['position_rmse'])
    assert len(planning['rows'])==32
    for row in planning['rows']:
        actual=np.asarray(row['states'][-1]);goal=np.asarray(row['goal'])
        pe=float(np.linalg.norm(actual[:2]-goal[:2]));ae=float(abs((actual[2]-goal[2]+np.pi)%(2*np.pi)-np.pi))
        assert np.isclose(pe,row['position_error']) and np.isclose(ae,row['angle_error'])
        assert row['success']==bool(pe<.08 and ae<.18)
    combined=prediction['rows']+matched['rows']
    # Pair scene means across seeds; do not treat 3 seeds as extra environments.
    rng=np.random.default_rng(90617);bootstrap=rng.integers(0,40,(5000,40))
    paired=[]
    for horizon in [1,5,10]:
        h=np.array([np.mean([r['position_rmse'] for r in combined if r['method']=='history' and r['horizon']==horizon and r['episode']==e]) for e in range(40)])
        n=np.array([np.mean([r['position_rmse'] for r in combined if r['method']=='matched_no_history' and r['horizon']==horizon and r['episode']==e]) for e in range(40)])
        difference=h-n;ci=np.quantile(difference[bootstrap].mean(1),[.025,.975])
        paired.append({'horizon':horizon,'scene_mean_difference_m':float(difference.mean()),
                       'bootstrap95_scene_difference_m':ci.tolist(),'history_relative_reduction':float(1-h.mean()/n.mean()),
                       'independent_scenarios':40,'note':'Exploratory CI over scenes, conditional on 3 fitted seeds; not full training-seed uncertainty.'})
    # Derive explicit four-action branches from previously saved action checks.
    demo=load(folder/'demo.json');data=np.load(ROOT/'data/processed/pushing-pilot/development.npz')
    demo['action_branches']=[]
    for check in prediction['action_checks'][:2]:
        e=check['episode']
        demo['action_branches'].append({'episode':e,'initial':data['states'][e,6].tolist(),'history_count':6,
            'alternatives':[{'label':label,'action':a,'predicted':p,'actual':t} for label,a,p,t in zip(
                ['从左侧推','从右侧推','从下侧推','从上侧推'],check['actions'],check['predicted'],check['true'])]})
    demo['summary']['capacity_control']=matched['summary']
    demo['metadata']['macro_duration']=.5
    demo['metadata']['training_episodes']=1200
    demo['metadata']['validation_episodes']=120
    write(folder/'demo.json',demo)
    # Thesis-ready static figure uses raw measurements, not illustrative scores.
    methods=['no_history','matched_no_history','history','sysid']
    labels=['No history (18.6k)','No history (42.6k)','History GRU (42.5k)','Physics identification']
    colors=['#cd925d','#a5aab0','#268a7b','#7a67ac']
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for method,label,color in zip(methods,labels,colors):
        vals=[np.mean([r['position_rmse'] for r in combined if r['method']==method and r['horizon']==h]) for h in [1,5,10]]
        axes[0].plot([1,5,10],vals,'o-',label=label,color=color)
    axes[0].set(xlabel='Prediction horizon (macro actions)',ylabel='Mean episode position RMSE (m)',title='40 development scenes, 3 training seeds')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    for run in training['runs']:
        color=colors[2] if run['kind']=='history' else colors[0]
        axes[1].plot([r['update'] for r in run['curve']],[r['validation_loss'] for r in run['curve']],color=color,alpha=.7)
    axes[1].set(xlabel='Gradient updates',ylabel='Normalized one-step validation MSE',title='Validation only selects checkpoints',yscale='log');axes[1].grid(alpha=.2)
    names=['oracle','sysid','no_history','history']
    axes[2].bar(['Oracle','SysID','No history','History'],[planning['summary'][m]['mean_steps'] for m in names],color=['#777',colors[3],colors[0],colors[2]])
    axes[2].set(ylabel='Mean executed macro actions',title='8/8 pose success for every method')
    axes[2].text(.5,.97,'6 shared setup pushes excluded; not stopped-state success',ha='center',va='top',fontsize=7,transform=axes[2].transAxes)
    fig.savefig(folder/'overview.png',dpi=170);fig.savefig(folder/'overview.pdf');plt.close(fig)
    code=['foresight/pushing_env.py','foresight/pushing_baselines.py','foresight/pushing_model.py','scripts/pushing_pilot.py','scripts/analyze_pushing.py']
    write(folder/'audit.json',{'checks':{'dataset_hashes':True,'portable_weight_hashes':True,'portable_parameter_counts':True,
            'all_visual_case_position_metrics':True,'all_planning_success_and_errors':True},
            'paired_capacity_control':paired,'source_sha256':{p:digest(ROOT/p) for p in code},
            'note':'Code hashes describe delivered implementation, including portable loading and post-pilot capacity control.'})
    print(json.dumps({'checks':'passed','paired_capacity_control':paired}))


if __name__=='__main__':main()
