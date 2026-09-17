'use strict';

const METHODS = [
  {id:'no_history',name:'无历史世界模型',color:'#efb979',tag:'STATE + ACTION',badge:'自训练模型',description:'只根据当前状态和未来动作预测，不读取此前交互。'},
  {id:'history',name:'有历史世界模型',color:'#77eed1',tag:'HISTORY + STATE + ACTION',badge:'自训练模型',description:'读取已发生的交互，结合当前状态与未来动作预测。'},
  {id:'sysid',name:'在线物理辨识',color:'#b9b4f2',tag:'IDENTIFY + SIMULATE',badge:'物理基线',description:'从相同交互历史估计动力学，再利用物理模型预测。'},
];
const METHOD_MAP = new Map(METHODS.map(method=>[method.id,method]));
const $ = id=>document.getElementById(id);
const state = {data:null,currentCase:null,frame:0,playing:false,lastTime:0,planningFrame:0,planningPlaying:false,camera:null,planningCamera:null,request:0};
const cards = new Map();
const planningCards = [];
const branchCards = [];
let branchCamera = null;
const radians = 180/Math.PI;
const faces = ['左侧','右侧','下侧','上侧'];
const finite = value=>typeof value==='number'&&Number.isFinite(value);
const fmt = (value,digits=3)=>finite(value)?value.toFixed(digits):'—';
const count = value=>finite(value)?value.toLocaleString('zh-CN'):'未记录';
const wrapAngle = value=>Math.atan2(Math.sin(value),Math.cos(value));
const methodName = id=>METHOD_MAP.get(id)?.name||({wrong_history:'错误历史对照',stationary:'保持静止',constant_velocity:'恒定速度外推'})[id]||id;

for(const method of METHODS){
  const article=document.createElement('article');
  article.className='method-card';article.style.setProperty('--method-color',method.color);
  article.innerHTML=`<div class="card-head"><div class="card-kicker"><span>${method.tag}</span><span>${method.badge}</span></div><h3>${method.name}</h3><p>${method.description}</p></div><div class="canvas-wrap"><canvas aria-label="${method.name}预测与实际箱子状态"></canvas><span class="canvas-tag">SHARED ACTIONS / 共享动作</span><div class="canvas-empty">等待真实实验数据</div></div><div class="card-metrics"><div class="metric"><span>此刻位置偏差</span><strong class="position-error">—<small>m</small></strong></div><div class="metric"><span>此刻朝向偏差</span><strong class="angle-error">—<small>°</small></strong></div></div><div class="card-footer"><span>整段位置 RMSE <strong class="position-rmse">—</strong></span><span>整段角度 MAE <strong class="angle-mae">—</strong></span></div>`;
  $('prediction-grid').append(article);
  cards.set(method.id,{...method,element:article,canvas:article.querySelector('canvas')});
}

function stepDuration(){return finite(state.data?.metadata?.macro_duration)?state.data.metadata.macro_duration:.5;}
function maxFrame(){return Math.max(0,(state.currentCase?.truth?.length||1)-1);}
function planningMaxFrame(){return Math.max(0,...planningCards.map(card=>card.data.states.length-1));}
function geometry(){return{halfWidth:state.data?.geometry?.half_width??.25,halfHeight:state.data?.geometry?.half_height??.18};}

function interpolate(sequence,frame){
  if(!sequence?.length)return null;
  const index=Math.min(Math.floor(frame),sequence.length-1),next=Math.min(index+1,sequence.length-1),t=Math.max(0,frame-index);
  return sequence[index].map((value,dimension)=>dimension===2?value+wrapAngle(sequence[next][dimension]-value)*t:value+(sequence[next][dimension]-value)*t);
}

function setPlaying(playing){state.playing=playing;$('play').textContent=playing?'Ⅱ 暂停预测':'▶ 播放预测';}
function setPlanningPlaying(playing){state.planningPlaying=playing;$('planning-play').textContent=playing?'Ⅱ 暂停执行':'▶ 回放闭环执行';}
function setFrame(frame){
  state.frame=Math.max(0,Math.min(maxFrame(),Number(frame)));
  $('timeline').value=state.frame;
  $('frame-label').textContent=`${fmt(state.frame,1)} / ${maxFrame()} 步`;
  $('time-label').textContent=`${fmt(state.frame*stepDuration(),2)} s`;
  const current=state.currentCase;
  const actionIndex=Math.min(Math.floor(state.frame),Math.max(0,(current?.actions?.length||1)-1));
  const action=current?.actions?.[actionIndex];
  $('action-label').textContent=action?`第 ${actionIndex+1} 步 · ${faces[action[0]]||'未知边'}推动 · 偏移 ${fmt(action[1],2)} · ${fmt(action[2],2)} m/s`:'无后续动作';
  renderPredictions();
}
function setPlanningFrame(frame){
  state.planningFrame=Math.max(0,Math.min(planningMaxFrame(),Number(frame)));
  $('planning-timeline').value=state.planningFrame;
  $('planning-frame').textContent=`${fmt(state.planningFrame,1)} / ${planningMaxFrame()} 步`;
  renderPlanning();
}

function isState(value){return Array.isArray(value)&&value.length>=6&&value.slice(0,6).every(finite);}
function validSequence(value){return Array.isArray(value)&&value.length>0&&value.every(isState);}
function validateData(data){
  if(!data||!Array.isArray(data.cases)||!data.cases.length)throw new Error('回放文件尚未包含有效诊断案例。');
  for(const item of data.cases){
    if(!validSequence(item.truth))throw new Error('真实轨迹数据不完整，请重新导出实验。');
    if(!Array.isArray(item.actions)||item.actions.length!==item.truth.length-1||item.actions.some(action=>!Array.isArray(action)||action.length!==3||!action.every(finite)||!Number.isInteger(action[0])||action[0]<0||action[0]>3))throw new Error('宏动作与真实轨迹长度或格式不一致。');
    for(const method of METHODS){if(!validSequence(item.predictions?.[method.id])||item.predictions[method.id].length!==item.truth.length)throw new Error(`${method.name}的预测轨迹缺失或长度不一致。`);}
  }
  if(data.planning!==undefined&&(!Array.isArray(data.planning)||data.planning.some(item=>!validSequence(item.states)||!Array.isArray(item.goal)||item.goal.length<3||!item.goal.slice(0,3).every(finite))))throw new Error('闭环规划回放格式不完整。');
  return data;
}

async function loadData(){
  const request=++state.request;setPlaying(false);setPlanningPlaying(false);$('reload').disabled=true;
  $('load-message').textContent='正在读取本机生成的实验回放…';$('data-status').classList.remove('error');
  try{
    const response=await fetch('../artifacts/pushing-pilot/demo.json',{cache:'no-store'});
    if(!response.ok)throw new Error(response.status===404?'还没有生成回放文件。训练与评估完成后，点击“重新加载”。':`读取实验数据失败（HTTP ${response.status}）。`);
    const data=validateData(await response.json());if(request!==state.request)return;
    state.data=data;document.body.classList.add('loaded');
    const metadata=data.metadata||{};
    $('stage-label').textContent=metadata.stage_label||'首轮可行性试验';
    $('load-message').textContent=[metadata.status,metadata.description].filter(Boolean).join(' · ')||'已加载真实生成的模型预测与环境执行。当前为小规模开发试验，单个案例不代表总体优势。';
    const samples=metadata.training_samples??metadata.train_samples??data.summary?.training_samples;
    const episodes=metadata.training_episodes??metadata.train_episodes;
    $('training-count').textContent=finite(samples)?`${count(samples)} 条转移`:finite(episodes)?`${count(episodes)} 条轨迹`:'未记录';
    const evaluated=metadata.test_episodes??metadata.validation_episodes??data.summary?.n_episodes??data.summary?.sample_count;
    $('evaluation-count').textContent=finite(evaluated)?`${count(evaluated)} 个案例`:'见实验记录';
    $('case-count').textContent=`${data.cases.length} 个诊断 / ${(data.planning||[]).length} 条规划`;
    $('action-duration').textContent=`${fmt(stepDuration(),2)} s / 宏步`;
    $('case-select').replaceChildren(...data.cases.map((item,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=item.label||item.id||`案例 ${index+1}`;return option;}));
    ['case-select','play','reset','timeline'].forEach(id=>{$(id).disabled=false;});
    renderEvidence();buildPlanning();buildBranches();selectCase(0);
  }catch(error){
    if(request!==state.request)return;
    $('data-status').classList.add('error');$('load-message').textContent=error.message;
    if(!state.data)$('stage-label').textContent='等待真实结果导出';
  }finally{if(request===state.request)$('reload').disabled=false;}
}

function selectCase(index){
  state.currentCase=state.data.cases[index];setPlaying(false);
  const item=state.currentCase;
  state.camera=makeCamera([item.truth,...METHODS.map(method=>item.predictions[method.id])]);
  $('timeline').max=maxFrame();
  $('history-label').textContent=finite(item.history_count)?`${item.history_count} 次已发生交互`:'未记录';
  const parameterLabels={mass:'质量',friction:'摩擦',mu:'摩擦系数',damping:'阻尼',cog_x:'重心偏移 x',cog_y:'重心偏移 y',seed:'场景种子'};
  const params=Object.entries(item.params_display||{}).map(([key,value])=>`${parameterLabels[key]||key} ${finite(value)?fmt(value,2):String(value)}`).join(' · ');
  $('case-description').textContent=item.description||(params?`评估器所见：${params}`:'共享初始状态、交互历史和后续动作。');
  $('case-note').textContent=item.selection_note||item.selection_provenance||'案例按实验导出的标签呈现。固定案例用于常规检查，大误差案例用于诊断失败；两者都不能代替整体统计。';
  cards.forEach(card=>{
    card.element.querySelector('.canvas-empty').hidden=true;
    const metrics=item.metrics?.[card.id]||{};
    card.element.querySelector('.position-rmse').textContent=`${fmt(metrics.position_rmse)} m`;
    card.element.querySelector('.angle-mae').textContent=`${fmt(finite(metrics.angle_mae)?metrics.angle_mae*radians:null,1)}°`;
  });
  setFrame(maxFrame());
}

function makeCamera(sequences,extra=[]){
  const points=sequences.flat().concat(extra).filter(point=>Array.isArray(point)&&finite(point[0])&&finite(point[1]));
  if(!points.length)return{x0:-1,x1:1,y0:-1,y1:1};
  let x0=Math.min(...points.map(point=>point[0]))-.55,x1=Math.max(...points.map(point=>point[0]))+.55;
  let y0=Math.min(...points.map(point=>point[1]))-.55,y1=Math.max(...points.map(point=>point[1]))+.55;
  if(x1-x0<1.7){const middle=(x1+x0)/2;x0=middle-.85;x1=middle+.85;}
  if(y1-y0<1.7){const middle=(y1+y0)/2;y0=middle-.85;y1=middle+.85;}
  return{x0,x1,y0,y1};
}

function canvasContext(canvas,camera){
  const bounds=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  if(!bounds.width||!bounds.height)return null;
  const width=bounds.width,height=bounds.height;
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
  const scale=Math.min((width-42)/(camera.x1-camera.x0),(height-42)/(camera.y1-camera.y0));
  const cx=(camera.x0+camera.x1)/2,cy=(camera.y0+camera.y1)/2;
  return{ctx,width,height,scale,point:point=>[width/2+(point[0]-cx)*scale,height/2-(point[1]-cy)*scale]};
}

function grid(view,camera){
  const{ctx,point,width,height,scale}=view;
  const step=scale>180?.2:scale>75?.5:1;
  ctx.save();ctx.strokeStyle='#27433777';ctx.lineWidth=.6;
  for(let x=Math.ceil(camera.x0/step)*step;x<=camera.x1;x+=step){const from=point([x,camera.y0]),to=point([x,camera.y1]);ctx.beginPath();ctx.moveTo(...from);ctx.lineTo(...to);ctx.stroke();}
  for(let y=Math.ceil(camera.y0/step)*step;y<=camera.y1;y+=step){const from=point([camera.x0,y]),to=point([camera.x1,y]);ctx.beginPath();ctx.moveTo(...from);ctx.lineTo(...to);ctx.stroke();}
  const length=step*scale;ctx.strokeStyle='#678e7199';ctx.beginPath();ctx.moveTo(14,height-16);ctx.lineTo(14+length,height-16);ctx.moveTo(14,height-19);ctx.lineTo(14,height-13);ctx.moveTo(14+length,height-19);ctx.lineTo(14+length,height-13);ctx.stroke();ctx.fillStyle='#749a7c';ctx.font='8px Consolas,monospace';ctx.fillText(`${step} m`,16+length,height-13);
  ctx.fillStyle='#52775b';ctx.fillText('x →',width-37,height-14);ctx.restore();
}

function path(view,sequence,color,dashed=false,alpha=1,until=sequence.length-1){
  const{ctx,point}=view;if(!sequence?.length)return;
  const end=Math.min(Math.floor(until),sequence.length-1);ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=1.65;ctx.setLineDash(dashed?[5,4]:[]);ctx.beginPath();ctx.moveTo(...point(sequence[0]));for(let index=1;index<=end;index++)ctx.lineTo(...point(sequence[index]));if(until>end)ctx.lineTo(...point(interpolate(sequence,until)));ctx.stroke();ctx.restore();
}

function box(view,item,color,options={}){
  if(!item)return;const{ctx,point}=view;const{halfWidth,halfHeight}=geometry();
  const cos=Math.cos(item[2]),sin=Math.sin(item[2]);
  const local=(x,y)=>point([item[0]+cos*x-sin*y,item[1]+sin*x+cos*y]);
  const corners=[[-halfWidth,-halfHeight],[halfWidth,-halfHeight],[halfWidth,halfHeight],[-halfWidth,halfHeight]].map(p=>local(...p));
  ctx.save();ctx.globalAlpha=options.alpha??1;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=options.width??1.8;ctx.setLineDash(options.dashed?[5,4]:[]);ctx.beginPath();ctx.moveTo(...corners[0]);corners.slice(1).forEach(p=>ctx.lineTo(...p));ctx.closePath();
  if(!options.outline){ctx.globalAlpha=(options.alpha??1)*.1;ctx.fill();ctx.globalAlpha=options.alpha??1;}ctx.stroke();ctx.setLineDash([]);
  const center=local(0,0),tip=local(halfWidth*.69,0);ctx.beginPath();ctx.moveTo(...center);ctx.lineTo(...tip);ctx.stroke();ctx.beginPath();ctx.arc(...center,2,0,Math.PI*2);ctx.fill();
  if(options.label){ctx.fillStyle=color;ctx.font='9px "Microsoft YaHei",sans-serif';ctx.fillText(options.label,corners[3][0],Math.min(...corners.map(p=>p[1]))-9);}ctx.restore();
}

function pusher(view,item,action,color){
  if(!item||!action)return;const{halfWidth,halfHeight}=geometry();
  const[face,offset]=action;const contacts=[[-halfWidth,offset*halfHeight],[halfWidth,offset*halfHeight],[offset*halfWidth,-halfHeight],[offset*halfWidth,halfHeight]],normals=[[1,0],[-1,0],[0,1],[0,-1]];
  const contact=contacts[face],normal=normals[face];if(!contact||!normal)return;
  const cos=Math.cos(item[2]),sin=Math.sin(item[2]);
  const world=(x,y)=>view.point([item[0]+cos*x-sin*y,item[1]+sin*x+cos*y]);
  const start=world(contact[0]-normal[0]*.15,contact[1]-normal[1]*.15),end=world(contact[0]-normal[0]*.02,contact[1]-normal[1]*.02),marker=world(...contact);
  const{ctx}=view;ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=.65;ctx.lineCap='round';ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(...start);ctx.lineTo(...end);ctx.stroke();ctx.lineWidth=1;ctx.beginPath();ctx.arc(...marker,4,0,Math.PI*2);ctx.stroke();ctx.restore();
}

function renderPredictions(){
  const item=state.currentCase;if(!item)return;
  const truth=interpolate(item.truth,state.frame),showTruth=$('show-truth').checked;
  const actionIndex=Math.min(Math.floor(state.frame),item.actions.length-1);
  for(const card of cards.values()){
    const predicted=interpolate(item.predictions[card.id],state.frame),view=canvasContext(card.canvas,state.camera);if(!view)continue;
    grid(view,state.camera);
    path(view,item.predictions[card.id],card.color,true,.3);
    path(view,item.predictions[card.id],card.color,true,1,state.frame);
    if(showTruth){path(view,item.truth,'#ecf2e9',false,.18);path(view,item.truth,'#ecf2e9',false,.75,state.frame);}
    box(view,item.truth[0],'#729780',{alpha:.35,outline:true});
    if(actionIndex>=0)pusher(view,item.predictions[card.id][actionIndex],item.actions[actionIndex],card.color);
    box(view,predicted,card.color,{width:2.1});
    if(showTruth){
      const{ctx,point}=view;ctx.save();ctx.strokeStyle='#eebf88';ctx.globalAlpha=.65;ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(...point(predicted));ctx.lineTo(...point(truth));ctx.stroke();ctx.restore();box(view,truth,'#f0f3e9',{outline:true,width:1.65});
    }
    card.element.querySelector('.position-error').replaceChildren(document.createTextNode(fmt(Math.hypot(predicted[0]-truth[0],predicted[1]-truth[1]))),unit('m'));
    card.element.querySelector('.angle-error').replaceChildren(document.createTextNode(fmt(Math.abs(wrapAngle(predicted[2]-truth[2]))*radians,1)),unit('°'));
  }
}

function unit(value){const element=document.createElement('small');element.textContent=value;return element;}

function buildBranches(){
  const rows=state.data.action_branches||[];
  $('branch-cases').replaceChildren();
  if(!rows.length){branchCards.length=0;return;}
  rows.forEach((row,index)=>{const button=document.createElement('button');button.type='button';button.textContent=`固定场景 ${row.episode??index}`;button.setAttribute('aria-pressed',String(index===0));button.addEventListener('click',()=>selectBranch(index));$('branch-cases').append(button);});
  selectBranch(0);
}

function selectBranch(index){
  const row=state.data.action_branches[index];
  branchCards.length=0;$('branch-grid').replaceChildren();
  const alternatives=(row.alternatives||[]).filter(item=>isState(item.predicted)&&isState(item.actual)&&Array.isArray(item.action)&&item.action.length===3);
  if(!isState(row.initial)||!alternatives.length){const empty=document.createElement('div');empty.className='planning-empty';empty.textContent='该场景的动作分支记录不完整，请重新导出。';$('branch-grid').append(empty);return;}
  branchCamera=makeCamera([[row.initial],...alternatives.map(item=>[item.predicted,item.actual])]);
  [...$('branch-cases').children].forEach((button,i)=>button.setAttribute('aria-pressed',String(index===i)));
  $('branch-history').textContent=`开发场景 ${row.episode??index} · ${row.history_count??'—'} 次相同历史交互`;
  const colors=['#77eed1','#efb979','#8bc8f0','#b9b4f2'];
  alternatives.forEach((item,i)=>{
    const article=document.createElement('article');article.className='method-card branch-card';article.style.setProperty('--method-color',colors[i%colors.length]);
    article.innerHTML='<div class="card-head"><div class="card-kicker"><span></span><span>0.5 s</span></div><h3></h3><p></p></div><div class="canvas-wrap"><canvas></canvas></div><div class="card-metrics"><div class="metric"><span>预测位置误差</span><strong class="position-error"></strong></div><div class="metric"><span>预测角度误差</span><strong class="angle-error"></strong></div></div>';
    article.querySelector('.card-kicker>span').textContent=`ACTION ${String(i+1).padStart(2,'0')}`;
    article.querySelector('h3').textContent=item.label||`推法 ${i+1}`;
    article.querySelector('.card-head p').textContent=`${faces[item.action[0]]||'未知边'} · 偏移 ${fmt(item.action[1],2)} · ${fmt(item.action[2],2)} m/s`;
    article.querySelector('.position-error').append(document.createTextNode(fmt(Math.hypot(item.predicted[0]-item.actual[0],item.predicted[1]-item.actual[1]))),unit('m'));
    article.querySelector('.angle-error').append(document.createTextNode(fmt(Math.abs(wrapAngle(item.predicted[2]-item.actual[2]))*radians,1)),unit('°'));
    const canvas=article.querySelector('canvas');canvas.setAttribute('aria-label',`${item.label||`推法 ${i+1}`}从同一初态的预测和实际结果`);
    $('branch-grid').append(article);branchCards.push({canvas,initial:row.initial,data:item,color:colors[i%colors.length]});
  });
  renderBranches();
}

function renderBranches(){
  for(const card of branchCards){
    const view=canvasContext(card.canvas,branchCamera);if(!view)continue;
    grid(view,branchCamera);
    const item=card.data;
    path(view,[card.initial,item.predicted],card.color,true,.8);
    path(view,[card.initial,item.actual],'#eef2e9',false,.45);
    box(view,card.initial,'#8c9e90',{alpha:.7,outline:true});
    pusher(view,card.initial,item.action,card.color);
    box(view,item.predicted,card.color,{width:2});
    box(view,item.actual,'#f0f3e9',{outline:true,width:1.7});
  }
}

function buildPlanning(){
  planningCards.length=0;$('planning-grid').replaceChildren();
  const rows=state.data.planning||[];
  $('planning-grid').classList.toggle('four-methods',rows.length===4);
  if(!rows.length){const empty=document.createElement('div');empty.className='planning-empty';empty.textContent='当前导出包含预测检验，尚未包含闭环规划结果。';$('planning-grid').append(empty);$('planning-controls').hidden=true;return;}
  state.planningCamera=makeCamera(rows.map(item=>item.states),rows.map(item=>item.goal));
  for(const row of rows){
    const method=METHOD_MAP.get(row.method)||{name:row.label||row.method||'规划方法',color:'#86bdae'};
    const article=document.createElement('article');article.className='method-card planning-card';article.style.setProperty('--method-color',method.color);
    const result=typeof row.success==='boolean'?(row.success?'位姿达标':'未达标'):'未记录判定';
    article.innerHTML='<div class="card-head"><div class="card-kicker"><span>CLOSED-LOOP EXECUTION</span><span class="result-badge"></span></div><h3></h3></div><div class="canvas-wrap"><canvas></canvas><span class="canvas-tag">虚线框：目标姿态</span></div><div class="card-metrics"><div class="metric"><span>终点位置误差</span><strong class="position-error"></strong></div><div class="metric"><span>终点角度误差</span><strong class="angle-error"></strong></div></div>';
    article.querySelector('h3').textContent=row.label||method.name;
    const badge=article.querySelector('.result-badge');badge.textContent=result;badge.classList.toggle('success',row.success===true);badge.classList.toggle('failure',row.success===false);
    article.querySelector('.position-error').append(document.createTextNode(fmt(row.position_error)),unit('m'));
    article.querySelector('.angle-error').append(document.createTextNode(fmt(finite(row.angle_error)?row.angle_error*radians:null,1)),unit('°'));
    const canvas=article.querySelector('canvas');canvas.setAttribute('aria-label',`${row.label||method.name}闭环推物实际执行路径`);
    $('planning-grid').append(article);planningCards.push({...method,data:row,element:article,canvas});
  }
  $('planning-controls').hidden=false;$('planning-timeline').max=planningMaxFrame();setPlanningFrame(planningMaxFrame());
}

function renderPlanning(){
  for(const card of planningCards){
    const view=canvasContext(card.canvas,state.planningCamera);if(!view)continue;
    grid(view,state.planningCamera);const row=card.data;
    box(view,[...row.goal,0,0,0],'#b0d7b8',{dashed:true,outline:true,label:'目标'});
    path(view,row.states,card.color,false,.2);path(view,row.states,card.color,false,.95,state.planningFrame);
    box(view,row.states[0],card.color,{alpha:.25,outline:true});
    const actionIndex=Math.min(Math.floor(state.planningFrame),row.states.length-2);
    if(actionIndex>=0&&state.planningFrame<row.states.length-1)pusher(view,row.states[actionIndex],row.actions?.[actionIndex],card.color);
    box(view,interpolate(row.states,state.planningFrame),'#eef2e9',{width:1.8});
  }
}

function renderEvidence(){
  const data=state.data,summary=data.summary||{},metadata=data.metadata||{};
  $('summary-description').textContent=summary.description||summary.note||'此处为导出文件包含的评估汇总。未提供的指标保持空缺；页面不从精选回放推断总体优势。';
  const tableContainer=$('summary-table');tableContainer.replaceChildren();
  let rows=summary.methods;
  if(rows&&!Array.isArray(rows)&&typeof rows==='object')rows=Object.entries(rows).map(([method,metrics])=>({method,...metrics}));
  if(Array.isArray(rows)&&rows.length){
    rows=rows.slice();
    const capacity=(summary.capacity_control||[]).find(row=>row.horizon===10);
    if(capacity)rows.push({...capacity,method:'capacity_matched',label:'无历史 MLP（匹配容量 42.6k）'});
    const table=document.createElement('table');table.innerHTML='<thead><tr><th>方法</th><th>位置 RMSE（m）</th><th>角度 MAE（°）</th><th>案例数</th></tr></thead>';
    const tbody=document.createElement('tbody');for(const row of rows){const tr=document.createElement('tr');[row.label||methodName(row.method),fmt(row.position_rmse),fmt(finite(row.angle_mae)?row.angle_mae*radians:null,1),count(row.independent_scenarios??row.n_episodes??row.sample_count??summary.n_episodes??summary.sample_count)].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});tbody.append(tr);}table.append(tbody);tableContainer.append(table);
  }
  const details=$('source-details');details.replaceChildren();
  const entries=[['数据来源','artifacts/pushing-pilot/demo.json'],['阶段',metadata.status||metadata.stage_label||'可行性试验'],['模型',metadata.model_description||'自训练对象状态预测模型'],['训练种子',metadata.training_seed??metadata.seed],['生成时间',metadata.generated_at],['划分说明',metadata.split_description],['角度单位','原始数据为弧度，界面转换为度'],['动作定义','边侧、相对接触偏移、推杆速度；每步推动后滑行']];
  for(const[label,value]of entries){if(value===undefined||value===null)continue;const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=String(value);details.append(dt,dd);}
}

$('case-select').addEventListener('change',event=>selectCase(Number(event.target.value)));
$('show-truth').addEventListener('change',renderPredictions);
$('reload').addEventListener('click',loadData);
$('play').addEventListener('click',()=>{if(!state.playing&&state.frame>=maxFrame())setFrame(0);setPlaying(!state.playing);});
$('reset').addEventListener('click',()=>{setPlaying(false);setFrame(0);});
$('timeline').addEventListener('input',event=>{setPlaying(false);setFrame(event.target.value);});
$('planning-play').addEventListener('click',()=>{if(!state.planningPlaying&&state.planningFrame>=planningMaxFrame())setPlanningFrame(0);setPlanningPlaying(!state.planningPlaying);});
$('planning-timeline').addEventListener('input',event=>{setPlanningPlaying(false);setPlanningFrame(event.target.value);});
window.addEventListener('resize',()=>{renderPredictions();renderPlanning();renderBranches();});
document.addEventListener('visibilitychange',()=>{state.lastTime=0;});

function animate(now){
  const elapsed=state.lastTime?Math.min(.1,(now-state.lastTime)/1000):0;state.lastTime=now;
  const step=elapsed*Number($('speed').value)/stepDuration();
  if(state.playing){setFrame(state.frame+step);if(state.frame>=maxFrame())setPlaying(false);}
  if(state.planningPlaying){setPlanningFrame(state.planningFrame+step);if(state.planningFrame>=planningMaxFrame())setPlanningPlaying(false);}
  requestAnimationFrame(animate);
}
loadData();requestAnimationFrame(animate);
