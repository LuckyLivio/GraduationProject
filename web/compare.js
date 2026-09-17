'use strict';

const METHODS = [
  {id:'global_physics',name:'全局常数物理模型',badge:'传统模型',color:'#efb979',description:'整个空间使用同一个估计阻尼'},
  {id:'local_identification',name:'在线物理辨识',badge:'强基线',color:'#bcb4f3',description:'最近 5 次有效反馈，估计当前阻尼'},
  {id:'frozen_learned',name:'学习世界模型',badge:'空间先验',color:'#73d8f0',description:'学习位置与阻尼的关系，提前推演'},
  {id:'gated_calibrated',name:'世界模型 + 门控校准',badge:'研究方法',color:'#79edbb',description:'保留空间规律，变化后用反馈修正'},
];
const $ = (id) => document.getElementById(id);
const state = {view:'prediction',data:null,frame:16,playing:false,lastTime:0,accumulator:0,request:0,controller:null,camera:null};
const cards = new Map();
const number = (value, digits=3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const distance = value => Number.isFinite(value) ? (value===0?'0.000':value<.001?value.toPrecision(2):value.toFixed(3)) : '—';

for (const method of METHODS) {
  const article = document.createElement('article');
  article.className='method-card'; article.dataset.method=method.id;
  article.style.setProperty('--method-color',method.color);
  article.innerHTML=`<div class="card-heading"><div><div class="method-name"><i></i><h2>${method.name}</h2><span class="model-badge">${method.badge}</span></div><p class="card-explanation">${method.description}</p></div><div class="card-metric"><span class="metric-label">16 步终点误差</span><strong class="metric-value">—<small>m</small></strong></div></div><div class="canvas-wrap"><canvas class="world-canvas" aria-label="${method.name}的预测与实际轨迹"></canvas><span class="canvas-tag">COMMON VIEW / 共享视野</span><div class="canvas-empty">等待真实推演</div></div><div class="card-footer"><span class="footer-left">整段 RMSE <strong>— m</strong></span><span class="footer-right">模型准备中</span></div>`;
  $('comparison-grid').append(article);
  cards.set(method.id,{...method,element:article,canvas:article.querySelector('canvas')});
}

function setPlaying(playing) {
  state.playing=playing;
  state.accumulator=0;
  $('play').textContent=playing?'Ⅱ 暂停对照':'▶ 播放对照';
  $('play').setAttribute('aria-label',playing?'暂停同步轨迹':'播放同步轨迹');
}

function maxFrame() {
  if (!state.data) return 16;
  if (state.view==='prediction') return state.data.actual_path.length-1;
  return Math.max(...state.data.methods.map(method=>method.actual_path.length-1));
}

function frameState(method, frame=state.frame) {
  return method.frames[Math.min(frame,method.frames.length-1)];
}

function setFrame(frame) {
  state.frame=Math.max(0,Math.min(maxFrame(),Math.round(frame)));
  $('timeline').value=state.frame;
  $('frame-label').textContent=`${String(state.frame).padStart(2,'0')} / ${String(maxFrame()).padStart(2,'0')} 步`;
  $('time-label').textContent=`${(state.frame*(state.data?.dt||.15)).toFixed(2)} s`;
  renderCards();
}

function updateMode() {
  const navigation=state.view==='navigation';
  document.body.classList.toggle('navigation',navigation);
  document.querySelectorAll('[data-view]').forEach(button=>{
    const selected=button.dataset.view===state.view;
    button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));
  });
  $('action-control').hidden=navigation;
  $('scenario-control').hidden=!navigation;
  $('main-title').innerHTML=navigation?'同一片场地，<em>四种决策。</em>':'同一个动作，<em>四种未来。</em>';
  $('intro-copy').textContent=navigation?'起点、目标和环境相同，让四种模型各自规划。同步看清预测怎样变成真实行动。':'先给四个模型完全相同的动作，再让真实环境执行。白线揭晓：谁真正预见了未来？';
  $('experiment-span').textContent=navigation?'相同场景 · 独立规划':'16 步 · 2.4 秒';
  document.querySelector('.experiment-mark small').textContent=navigation?'四种方法各自选择并执行动作':'共享起点、历史与未来动作';
  $('gap-legend').textContent=navigation?'预测是当前计划，后续会重新规划':'同一时刻的位置偏差';
  document.querySelector('.distance-key').hidden=navigation;
  $('show-terrain').parentElement.hidden=navigation;
  $('field-evidence').hidden=navigation;
  $('scope-note').textContent=navigation?'相同场景不等于相同动作。此处比较闭环导航表现，后续实际轨迹含重新规划，不能直接作为预测误差。模型仅学习位置→阻尼，障碍运动和运动积分使用已知规律。':'当前学习位置→阻尼；障碍运动与运动积分使用已知规律。此处检验学到的空间规律对预测的作用，属于轻量混合世界模型；单次诊断不能代替多种子实验。';
  $('seed').disabled=!navigation;
  $('seed').parentElement.hidden=!navigation;
  $('seed').title=navigation?'用于生成四种方法共享的导航场景':'预测使用固定诊断路线；种子不改变固定路线';
}

async function runComparison() {
  const request=++state.request;
  state.controller?.abort(); state.controller=new AbortController();
  setPlaying(false);state.data=null;
  document.body.classList.add('busy');
  $('run').disabled=true;$('play').disabled=true;$('timeline').disabled=true;$('reset').disabled=true;
  $('run-status').classList.remove('error');
  $('run-status').textContent=state.view==='prediction'?'执行相同动作，核对预测…':'四种方法正在独立导航，请稍候…';
  $('verdict-title').textContent='正在计算，等待真实结果';
  $('verdict-text').textContent=state.view==='prediction'?'四个模型共享完全相同的历史反馈和未来动作；预测过程中不再更新模型。':'正在生成四条独立规划的真实轨迹。完成后可用共享时间轴对比，计时来自轨迹生成时。';
  cards.forEach(card=>{card.element.querySelector('.canvas-empty').hidden=false;card.element.querySelector('.canvas-empty').textContent='正在运行真实模型…';});
  try {
    const response=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},signal:state.controller.signal,body:JSON.stringify({view:state.view,condition:$('condition').value,action_mode:$('action-mode').value,seed:Number($('seed').value),scenario:$('scenario').value,horizon:16})});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||`请求失败（${response.status}）`);
    if(request!==state.request)return;
    if(!Array.isArray(data.methods)||data.methods.length!==4)throw new Error('对照结果不完整，请检查服务日志');
    state.data=data;state.camera=makeCamera(data);
    document.body.classList.remove('busy');
    $('run').disabled=false;$('play').disabled=false;$('timeline').disabled=false;$('reset').disabled=false;
    $('timeline').max=maxFrame();
    $('run-status').textContent=state.view==='prediction'?'计算完成 · 固定诊断路线':'生成完成 · 四条真实导航轨迹';
    cards.forEach(card=>card.element.querySelector('.canvas-empty').hidden=true);
    setFrame(maxFrame());renderVerdict();renderFields();
  } catch(error) {
    if(request!==state.request||error.name==='AbortError')return;
    document.body.classList.remove('busy');$('run').disabled=false;
    $('run-status').classList.add('error');$('run-status').textContent=error.message;
    $('verdict-title').textContent='当前无法生成对照';
    $('verdict-text').textContent='请确认本地服务已启动，模型文件可用，再点击“重新推演”。';
    cards.forEach(card=>{card.element.querySelector('.canvas-empty').hidden=false;card.element.querySelector('.canvas-empty').textContent='暂无结果';});
  }
}

function makeCamera(data) {
  if(data.view==='navigation')return {x0:-.3,x1:10.3,y0:-.3,y1:10.3};
  const points=[...data.actual_path,...data.history.slice(-4),...data.methods.flatMap(method=>method.predicted_path)];
  let x0=Math.min(...points.map(p=>p[0]))-.3,x1=Math.max(...points.map(p=>p[0]))+.3;
  let y0=Math.min(...points.map(p=>p[1]))-.3,y1=Math.max(...points.map(p=>p[1]))+.3;
  const minimumWidth=2.1;
  if(x1-x0<minimumWidth){const middle=(x0+x1)/2;x0=middle-minimumWidth/2;x1=middle+minimumWidth/2;}
  const targetHeight=Math.max(1.0,(x1-x0)*.32),middle=(y0+y1)/2;
  y0=middle-targetHeight/2;y1=middle+targetHeight/2;
  return{x0,x1,y0,y1};
}

function canvasContext(canvas) {
  const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  if(!rect.width||!rect.height)return null;
  if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){
    canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  }
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
  return{ctx,width:rect.width,height:rect.height};
}

function projection(camera,width,height,pad=20,square=false) {
  let drawWidth=width-pad*2,drawHeight=height-pad*2;
  let x=pad,y=pad;
  if(square){const side=Math.min(drawWidth,drawHeight);x=(width-side)/2;y=(height-side)/2;drawWidth=side;drawHeight=side;}
  return{point:p=>[x+(p[0]-camera.x0)/(camera.x1-camera.x0)*drawWidth,y+drawHeight-(p[1]-camera.y0)/(camera.y1-camera.y0)*drawHeight],x,y,width:drawWidth,height:drawHeight};
}

function terrainColor(value,alpha=1) {
  const t=Math.max(0,Math.min(1,(value-.2)/2.3));
  const colors=[[19,51,56],[39,99,91],[133,153,93],[232,183,116]],n=t*3,i=Math.min(2,Math.floor(n)),a=n-i;
  return`rgba(${colors[i].map((v,j)=>Math.round(v*(1-a)+colors[i+1][j]*a)).join(',')},${alpha})`;
}

function paintTerrain(ctx,project,field,values,scale=1,constant=null) {
  const xs=field.x,ys=field.y;
  const dx=xs.length>1?xs[1]-xs[0]:1,dy=ys.length>1?ys[1]-ys[0]:1;
  ctx.save();ctx.beginPath();ctx.rect(project.x,project.y,project.width,project.height);ctx.clip();
  for(let iy=0;iy<ys.length;iy++)for(let ix=0;ix<xs.length;ix++){
    const value=constant??values[iy]?.[ix];
    if(!Number.isFinite(value))continue;
    const p1=project.point([xs[ix]-dx/2,ys[iy]+dy/2]),p2=project.point([xs[ix]+dx/2,ys[iy]-dy/2]);
    ctx.fillStyle=terrainColor(value*scale,.52);ctx.fillRect(p1[0],p1[1],p2[0]-p1[0]+.5,p2[1]-p1[1]+.5);
  }
  ctx.restore();
}

function paintGrid(ctx,project,camera) {
  const increment=state.view==='navigation'?2:(camera.x1-camera.x0>5?1:.5);
  ctx.strokeStyle='#c0e8da13';ctx.lineWidth=.6;
  ctx.font='9px monospace';ctx.fillStyle='#77998c';
  for(let x=Math.ceil(camera.x0/increment)*increment;x<camera.x1;x+=increment){
    const p=project.point([x,camera.y0]),q=project.point([x,camera.y1]);
    ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();
    ctx.fillText(`${Number(x.toFixed(1))}`,p[0]-4,project.y+project.height+13);
  }
  if(state.view==='navigation')for(let y=0;y<=10;y+=2){const p=project.point([0,y]),q=project.point([10,y]);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();}
  ctx.textAlign='right';ctx.fillText('m',project.x+project.width+12,project.y+project.height+13);ctx.textAlign='left';
}

function paintPath(ctx,project,path,color,width=2,dash=[]) {
  if(!path?.length)return;
  ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.lineJoin='round';ctx.lineCap='round';ctx.beginPath();
  path.forEach((p,i)=>{const pixel=project.point(p);if(i===0)ctx.moveTo(...pixel);else ctx.lineTo(...pixel);});ctx.stroke();ctx.restore();
}

function dot(ctx,p,color,radius=4,ring=false) {
  ctx.beginPath();ctx.arc(p[0],p[1],radius,0,Math.PI*2);
  if(ring){ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();}else{ctx.fillStyle=color;ctx.fill();}
}

function paintObstacles(ctx,project,world) {
  for(let i=6;i<18;i+=4){
    const p=project.point(world.slice(i,i+2)),r=Math.abs(project.point([world[i]+.38,world[i+1]])[0]-p[0]);
    ctx.fillStyle='#ef9c8736';ctx.strokeStyle='#d6907d';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fill();ctx.stroke();
  }
  const goal=project.point(world.slice(4,6));dot(ctx,goal,'#9cdd8c',7,true);dot(ctx,goal,'#9cdd8c',2);
}

function renderPrediction(card,method,context) {
  const {ctx,width,height}=context,data=state.data,project=projection(state.camera,width,height,22);
  if($('show-terrain').checked&&data.field){
    const learned=method.id==='frozen_learned'||method.id==='gated_calibrated';
    paintTerrain(ctx,project,data.field,data.field.learned,learned?(method.learned_scale||1):1,learned?null:method.local_damping);
  }
  paintGrid(ctx,project,state.camera);
  const real=data.actual_path.slice(0,state.frame+1),prediction=method.predicted_path.slice(0,state.frame+1);
  paintPath(ctx,project,data.history.slice(-4),'#a7bcb844',1,[2,4]);
  paintPath(ctx,project,data.actual_path,'#e8eee929',1.2);
  paintPath(ctx,project,method.predicted_path,`${card.color}66`,1.3,[6,5]);
  paintPath(ctx,project,real,'#edf7ee',3.5);
  paintPath(ctx,project,prediction,card.color,2.1,[5,4]);
  const actual=real.at(-1),predicted=prediction.at(-1);
  const a=project.point(actual),p=project.point(predicted);
  const error=Math.hypot(actual[0]-predicted[0],actual[1]-predicted[1]);
  if(state.frame>0&&error>.005){
    const offset=-20;
    ctx.strokeStyle='#f1aa89';ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(a[0],a[1]+offset);ctx.lineTo(p[0],p[1]+offset);ctx.lineTo(...p);ctx.stroke();ctx.setLineDash([]);
    ctx.font='10px monospace';ctx.fillStyle='#e5b69b';ctx.textAlign='center';ctx.fillText(`${distance(error)} m`,(a[0]+p[0])/2,Math.min(a[1],p[1])+offset-7);ctx.textAlign='left';
  }
  dot(ctx,project.point(data.snapshot),'#91b6a5',5,true);
  dot(ctx,a,'#edf7ee',5);dot(ctx,p,'#0d1d22',5);dot(ctx,p,card.color,5,true);dot(ctx,p,card.color,2);
  ctx.font='9px sans-serif';ctx.fillStyle='#8eac9e';const origin=project.point(data.snapshot);ctx.fillText('共同起点',origin[0]-19,origin[1]+25);
  card.element.querySelector('.metric-label').textContent=state.frame===maxFrame()?`${maxFrame()} 步终点误差`:`第 ${state.frame} 步位置误差`;
  card.element.querySelector('.metric-value').innerHTML=`${distance(error)}<small>m</small>`;
  card.element.querySelector('.footer-left').innerHTML=`整段 RMSE <strong>${distance(method.position_rmse)} m</strong>`;
  const right=card.element.querySelector('.footer-right');
  if(method.id==='gated_calibrated'){
    right.className=`footer-right gate-status${method.gate_active?' on':''}`;
    right.innerHTML=`<i></i>${method.gate_active?'校准开启':'保留先验'} · <strong>${number(method.learned_scale,2)}×</strong>`;
  }else if(method.id==='local_identification'||method.id==='global_physics'){
    right.className='footer-right';right.innerHTML=`估计阻尼 <strong>${number(method.local_damping,2)}</strong>`;
  }else{
    right.className='footer-right';right.textContent='模型冻结 · 保留空间变化';
  }
}

function renderNavigation(card,method,context) {
  const {ctx,width,height}=context,project=projection(state.camera,width,height,17,true);
  paintGrid(ctx,project,state.camera);
  const frame=frameState(method),currentStep=Math.min(state.frame,method.actual_path.length-1);
  const current=method.actual_path[currentStep],completed=state.frame>=method.actual_path.length-1;
  paintObstacles(ctx,project,current);
  paintPath(ctx,project,method.actual_path.slice(0,currentStep+1),'#edf7ee',2);
  if(!completed&&frame)paintPath(ctx,project,frame.predicted_path,card.color,1.8,[4,3]);
  const pos=project.point(current);dot(ctx,pos,'#0d1d22',6);dot(ctx,pos,card.color,6,true);dot(ctx,pos,card.color,3);
  dot(ctx,project.point(state.data.initial_state),'#90b6a0',4,true);
  const dist=Math.hypot(current[4]-current[0],current[5]-current[1]);
  const initial=state.data.initial_state,initialDist=Math.hypot(initial[4]-initial[0],initial[5]-initial[1]);
  const progress=Math.max(0,Math.min(100,(1-dist/initialDist)*100));
  const status=completed?(method.success?'抵达目标':method.collision?'发生碰撞':'未在时限内抵达'):`${Math.round(progress)}%`;
  card.element.querySelector('.metric-label').textContent=completed?'本回合结果':'直线距离进度';
  card.element.querySelector('.metric-value').textContent=status;
  card.element.querySelector('.footer-left').innerHTML=completed?`总计 <strong>${currentStep} 步</strong> · 最小间距 <strong>${number(method.min_clearance,2)} m</strong>`:`已执行 <strong>${currentStep} 步</strong> · 距目标 <strong>${number(dist,2)} m</strong>`;
  const right=card.element.querySelector('.footer-right');right.className='footer-right';
  right.innerHTML=`生成时平均 <strong>${number(method.mean_controller_ms,1)} ms</strong>`;
}

function renderCards() {
  if(!state.data)return;
  for(const method of state.data.methods){
    const card=cards.get(method.id);if(!card)continue;
    const context=canvasContext(card.canvas);if(!context)continue;
    card.element.querySelector('.canvas-tag').textContent=state.view==='prediction'?'SAME ACTIONS / 共同动作 · 共享局部视野':'SAME SCENE / 同场景 · 独立决策';
    if(state.view==='prediction')renderPrediction(card,method,context);else renderNavigation(card,method,context);
  }
}

function renderVerdict() {
  const data=state.data;
  if(data.view==='navigation'){
    const successes=data.methods.filter(m=>m.success);
    $('verdict-title').textContent=successes.length===4?'这一次，四种方法都能完成导航。':`本次 ${successes.length} / 4 种方法抵达目标。`;
    $('verdict-text').textContent='导航会不断重新规划，预测更准未必带来更高成功率。这里展示各自真实执行的路径；预测能力请切换到“同动作预测检验”比较。';
    return;
  }
  const byId=Object.fromEntries(data.methods.map(m=>[m.id,m]));
  const sorted=[...data.methods].sort((a,b)=>a.position_rmse-b.position_rmse),best=sorted[0];
  const bestName=METHODS.find(m=>m.id===best.id).name;
  const difference=sorted[1].position_rmse-best.position_rmse;
  $('verdict-title').textContent=difference<.001?'本次最优两种方法的整段预测误差接近。':`本次同动作检验：${bestName}预测最接近真实。`;
  const physics=distance(byId.global_physics.position_rmse),local=distance(byId.local_identification.position_rmse),learned=distance(byId.frozen_learned.position_rmse),gated=distance(byId.gated_calibrated.position_rmse);
  $('verdict-text').textContent=`整段位置 RMSE：常数物理 ${physics} m，在线辨识 ${local} m，学习模型 ${learned} m，门控校准 ${gated} m。${data.condition==='nominal'?'本例让机器人进入阻尼不同的位置，检验模型能否提前利用空间规律。':'本例改变整个环境的阻尼，检验已有空间规律能否通过行动反馈修正。'}当前为固定诊断路线，不代表所有场景。`;
}

function renderFields() {
  const data=state.data;if(data?.view!=='prediction'||!data.field||!$('field-evidence').open)return;
  const camera={x0:0,x1:10,y0:0,y1:10};
  for(const [id,values] of [['truth-field',data.field.truth],['learned-field',data.field.learned]]){
    const context=canvasContext($(id));if(!context)continue;
    const{ctx,width,height}=context,project=projection(camera,width,height,18,true);
    paintTerrain(ctx,project,data.field,values);
    paintPath(ctx,project,data.history,'#d0e5d0aa',1.5,[2,3]);paintPath(ctx,project,data.actual_path,'#f1f6e8',2);
    dot(ctx,project.point(data.snapshot),'#f1f6e8',3);
    ctx.fillStyle='#a8c8b7';ctx.font='9px monospace';ctx.fillText('0',project.x-9,project.y+project.height+12);ctx.fillText('10 m',project.x+project.width-25,project.y+project.height+12);
  }
  const seed=data.metadata?.model_training_seed??142;
  $('field-source').textContent=`训练种子 ${seed} · 白线为本次诊断路线 · 前 ${data.metadata?.warmup_steps??data.history_actions.length} 步历史用于各方法共享的观察，预测段不更新模型。${data.condition==='global_shift'?'左图含 1.7× 环境变化；右图仍是冻结空间先验，门控方法按反馈估计倍率。':''}`;
}

$('controls').addEventListener('submit',event=>{event.preventDefault();if($('controls').reportValidity())runComparison();});
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{if(state.view===button.dataset.view)return;state.view=button.dataset.view;updateMode();runComparison();}));
$('play').addEventListener('click',()=>{if(!state.data)return;if(state.frame>=maxFrame())setFrame(0);setPlaying(!state.playing);});
$('reset').addEventListener('click',()=>{setPlaying(false);setFrame(0);});
$('timeline').addEventListener('input',()=>{setPlaying(false);setFrame(Number($('timeline').value));});
$('show-terrain').addEventListener('change',renderCards);
$('field-evidence').addEventListener('toggle',renderFields);
window.addEventListener('resize',()=>{renderCards();renderFields();});
// Redraw after responsive layout settles, including canvases below the viewport.
const canvasObserver=new ResizeObserver(()=>{renderCards();renderFields();});
document.querySelectorAll('.canvas-wrap,.field-pair figure').forEach(element=>canvasObserver.observe(element));
window.addEventListener('keydown',event=>{
  if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(event.target.tagName)||!state.data)return;
  if(event.code==='Space'){event.preventDefault();$('play').click();}
  if(event.code==='ArrowRight'){event.preventDefault();setPlaying(false);setFrame(state.frame+1);}
  if(event.code==='ArrowLeft'){event.preventDefault();setPlaying(false);setFrame(state.frame-1);}
});

function animate(time) {
  if(state.playing&&state.data){
    state.accumulator+=Math.min(time-state.lastTime,250)*Number($('speed').value);
    const interval=(state.data.dt||.15)*1000;
    while(state.accumulator>=interval){state.accumulator-=interval;setFrame(state.frame+1);if(state.frame>=maxFrame()){setPlaying(false);break;}}
  }
  state.lastTime=time;requestAnimationFrame(animate);
}
updateMode();runComparison();requestAnimationFrame(animate);
