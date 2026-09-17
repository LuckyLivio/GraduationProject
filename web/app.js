"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries(["data-status","experiment-notice","notice-title","notice-text","reload-data","world-canvas","world-container","scene-empty","scene-coordinate","scene-outcome","scene-caption","scene-caption-text","method-select","method-description","condition-select","episode-select","frame-current","frame-total","timeline","play-button","reset-button","playback-speed","frame-horizon","horizon-viz","frame-latency","frame-model-steps","frame-disagreement","step-label","show-candidates","show-actual","chart-metric","evidence-empty","evidence-content","metric-chart","results-body","evidence-scope","data-provenance"].map(id => [id, $(id)]));
  ["mode-live","mode-replay","mode-status","live-fields","live-scenario","live-seed","live-damping","damping-value","view-label","actual-legend","actual-option","inspector-note","evidence-condition","frame-scale-row","frame-learned-scale","frame-calibration-row","frame-calibration-ms","scale-estimate-note"].forEach(id=>ui[id]=$(id));
  ["apply-dynamics","trust-panel","gate-mode","gate-score","gate-meter","gate-explanation","raw-scale","gate-model","gate-evidence-status"].forEach(id=>ui[id]=$(id));
  const methods = {
    gated_calibrated: ["门控校准 · 变化后再修正", "只有持续的变化证据超过独立验证集确定的阈值，才启用在线校准。使用本轮训练模型，与历史方法的模型版本不同。"],
    residual_calibrated: ["持续校准 · 首轮学习模型", "使用首轮训练模型，持续利用近期实际转移修正阻尼倍率；不重新训练网络。与本轮门控方法的公平对比请看实验报告。"],
    global_physics: ["全局物理参数基线", "使用训练数据拟合的全局物理参数进行预测和规划。"],
    local_identification: ["局部参数辨识基线", "从近期状态转移中辨识局部动力学参数，用于后续预测。"],
    fixed_5: ["固定视野 · 5 步", "使用轻量混合世界模型，以固定 5 步视野推演候选动作。"],
    fixed_10: ["固定视野 · 10 步", "使用轻量混合世界模型，以固定 10 步视野推演候选动作。"],
    fixed_16: ["固定视野 · 16 步", "使用轻量混合世界模型，以固定 16 步视野推演候选动作。"],
    adaptive: ["不确定性自适应规划", "根据集成模型的预测分歧调节规划视野。检验它能否在相同预算下改善表现。"],
    adaptive_horizon: ["不确定性自适应规划", "根据模型分歧动态调整预测长度。"],
    fixed_short: ["固定短视野", "使用固定的短预测范围，观察短期决策的收益与局限。"],
    fixed_long: ["固定长视野", "使用固定的长预测范围，观察长期规划与误差累积的取舍。"],
    fixed_medium: ["固定中视野", "使用固定的中等预测范围进行模型预测控制。"],
    oracle: ["真实动力学上界", "规划时使用真实动力学，作为模型误差影响的参考。"],
    oracle_mpc: ["真实动力学上界", "规划时使用真实动力学，作为模型误差影响的参考。"],
    reactive: ["反应式基线", "依据当前状态直接选择动作，不通过学习的世界模型展开预测。"],
    random: ["随机动作基线", "随机选择动作，用于检查任务难度。"],
  };
  const conditions = {id:"训练分布内", nominal:"训练分布内", in_distribution:"训练分布内", iid:"训练分布内", ood:"分布外扰动", damping_shift:"阻尼变化", friction_shift:"摩擦变化", shifted:"动力学变化", dynamics_shift:"动力学变化", ood_damping:"阻尼变化", ood_wind:"外力扰动", wind:"外力扰动", easy:"基础场景", hard:"困难场景"};
  const state = {summary:null, episodes:[], meta:{map_size:10,robot_radius:.18,obstacle_radius:.38}, episode:null, frame:0, playing:false, lastTime:0, accumulator:0, error:null, retry:null,mode:"replay",transform:null,live:{ready:false,session:null,frame:null,frames:[],step:0,done:false,pending:null,timer:null,generation:0,methods:[]}};
  const ctx = ui["world-canvas"].getContext("2d");
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const number = (value,digits=1) => finite(value) ? value.toLocaleString("zh-CN", {maximumFractionDigits:digits,minimumFractionDigits:digits}) : "—";
  const percent = value => finite(value) ? `${number(value*100)}%` : "—";
  const methodName = key => methods[key]?.[0] || String(key || "未命名方法");
  const conditionName = key => conditions[key] || String(key || "未命名工况");
  const currentFrame = () => state.mode==="live"?state.live.frame:state.episode?.frames?.[state.frame];
  function element(tag,className,text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
  function options(select,values,label,preferred){select.replaceChildren(...values.map(value=>{const option=element("option","",label(value));option.value=String(value);return option;}));select.disabled=!values.length;if(values.some(v=>String(v)===String(preferred)))select.value=String(preferred);}
  function setPlay(playing){state.playing=Boolean(playing&&(state.mode==="live"?state.live.session&&!state.live.done:state.episode?.frames?.length>1));state.accumulator=0;ui["play-button"].replaceChildren(document.createTextNode(state.playing?"Ⅱ ":"▶ "),element("span","",state.playing?"暂停":state.mode==="live"?"开始":"播放"));ui["play-button"].setAttribute("aria-label",state.playing?"暂停运行":state.mode==="live"?"开始实时推理":"播放回放");if(!state.playing)clearTimeout(state.live.timer);else if(state.mode==="live")void liveStep();}
  function chooseMethod(){
    const method=ui["method-select"].value;
    ui["method-description"].textContent=methods[method]?.[1]||"展示该方法记录的真实决策过程；方法定义请参阅实验配置。";
    if(state.mode==="live"){void resetLive();renderEvidence();return;}
    const values=[...new Set(state.episodes.filter(ep=>ep.method===method).map(ep=>ep.condition))];
    options(ui["condition-select"],values,conditionName,ui["condition-select"].value);chooseCondition();
  }
  function chooseCondition(){
    const episodes=state.episodes.filter(ep=>ep.method===ui["method-select"].value&&ep.condition===ui["condition-select"].value);
    options(ui["episode-select"],episodes.map(ep=>ep.id),id=>{const ep=episodes.find(item=>item.id===id);return `Seed ${ep.seed??"—"} · ${ep.success?"到达目标":ep.collision?"发生碰撞":"未到达"}`;},ui["episode-select"].value);
    ui["evidence-condition"].value=ui["condition-select"].value;
    chooseEpisode();renderEvidence();
  }
  function chooseEpisode(){
    state.episode=state.episodes.find(ep=>String(ep.id)===ui["episode-select"].value)||null;state.frame=0;setPlay(false);
    const total=state.episode?.frames?.length||0;
    ui.timeline.max=String(Math.max(0,total-1));ui.timeline.disabled=total<2;ui["play-button"].disabled=total<2;ui["reset-button"].disabled=!total;ui["frame-total"].textContent=String(Math.max(0,total-1)).padStart(3,"0");
    ui["scene-empty"].hidden=Boolean(total);ui["scene-caption"].hidden=!total;
    const outcome=ui["scene-outcome"];outcome.className="outcome";
    if(state.episode){outcome.classList.add("visible");if(state.episode.success)outcome.textContent="回合结果 · 到达目标";else if(state.episode.collision){outcome.textContent="回合结果 · 碰撞";outcome.classList.add("collision");}else{outcome.textContent="回合结果 · 未到达";outcome.classList.add("timeout");}}
    updateFrame();
  }
  function updateFrame(){
    const frame=currentFrame();const position=state.mode==="live"?state.live.step:state.frame;ui.timeline.value=String(position);ui["frame-current"].textContent=String(position).padStart(3,"0");ui["step-label"].textContent=`STEP ${frame?String(position).padStart(3,"0"):"—"}`;
    ui["frame-horizon"].textContent=number(frame?.horizon,0);ui["frame-latency"].textContent=number(frame?.planning_ms,2);ui["frame-model-steps"].textContent=number(frame?.model_steps,0);ui["frame-disagreement"].textContent=number(frame?.disagreement,5);
    ui["frame-scale-row"].hidden=state.mode!=="live"||!finite(frame?.learned_scale);ui["frame-learned-scale"].textContent=finite(frame?.learned_scale)?`${number(frame.learned_scale,3)}×`:"—";
    ui["scale-estimate-note"].hidden=ui["frame-scale-row"].hidden;ui["scale-estimate-note"].textContent=`倍率根据本步执行后的观测校准，供下一次决策使用${finite(frame?.scale_updates)?`；已接受 ${number(frame.scale_updates,0)} 次更新`:""}。`;
    ui["frame-calibration-row"].hidden=state.mode!=="live"||!finite(frame?.calibration_ms);ui["frame-calibration-ms"].textContent=finite(frame?.calibration_ms)?`${number(frame.calibration_ms,2)} ms`:"—";
    const gated=state.mode==="live"&&typeof frame?.gate_active==="boolean";
    ui["trust-panel"].hidden=!gated;
    ui["apply-dynamics"].disabled=state.mode!=="live"||!state.live.session||state.live.done||Boolean(state.live.controlsDisabled);
    if(gated){
      ui["trust-panel"].classList.toggle("calibrating",frame.gate_active);
      ui["gate-mode"].textContent=frame.gate_active?"启用校准":"保留先验";
      ui["gate-score"].textContent=`${number(frame.gate_score,3)} / ${number(frame.gate_threshold,3)}`;
      ui["gate-meter"].style.width=`${Math.min(100,Math.max(0,frame.gate_score/Math.max(frame.gate_threshold,1e-9)*50))}%`;
      ui["raw-scale"].textContent=`${number(frame.raw_scale,3)}×`;
      ui["gate-explanation"].textContent=frame.gate_active?"持续偏差已触发校准，下一步预测将使用校准倍率。证据回落后恢复先验。":"当前使用原有模型。校准倍率的绝对对数偏差连续超阈值，才触发修正；这不是置信概率。";
      ui["gate-model"].textContent=`本轮模型 · 训练种子 ${frame.model_training_seed??"—"} · 诊断在本步执行后更新`;
    }
    const maxHorizon=state.mode==="live"?16:Math.max(1,...(state.episode?.frames||[]).map(f=>finite(f.horizon)?f.horizon:0));
    const blocks=Array.from({length:16},(_,i)=>{const block=element("i",frame&&i<16*(frame.horizon||0)/maxHorizon?"active":"");block.style.height=`${12+i*1.4}px`;return block;});ui["horizon-viz"].replaceChildren(...blocks);
    if(frame){const candidates=frame.candidate_paths?.length||0;ui["scene-caption-text"].textContent=frame.predicted_path?.length?`推演 ${frame.horizon??frame.predicted_path.length-1} 步未来${candidates?` · 展示 ${candidates} 条候选轨迹`:""} · 执行当前动作后重新规划`:state.mode==="live"?state.live.done?"本回合已结束 · 重置场景后可再次运行":"点击开始，观察模型实时预测 · 点击地图设置目标":"当前决策未记录模型预测轨迹";}
    drawWorld();
  }
  function seek(value){state.frame=Math.max(0,Math.min((state.episode?.frames?.length||1)-1,Math.round(value)));state.accumulator=0;updateFrame();}
  function renderEvidence(){
    const all=state.summary?.results||[];const condition=ui["evidence-condition"].value||all[0]?.condition;
    const results=all.filter(row=>row.condition===condition);
    ui["evidence-empty"].hidden=Boolean(results.length);ui["evidence-content"].hidden=!results.length;
    if(!results.length){ui["evidence-empty"].textContent=all.length?"当前工况暂无汇总指标，请选择其他工况。":"主实验结果尚未生成。这里不填入演示数值。";return;}
    const metric=ui["chart-metric"].value;const isRate=metric.endsWith("_rate");const max=isRate?1:Math.max(...results.map(row=>finite(row[metric])?row[metric]:0),1);
    ui["metric-chart"].replaceChildren(...results.map(row=>{const item=element("div",`chart-item${row.method===ui["method-select"].value?" selected":""}`);const label=element("div","chart-label");label.append(element("span","",methodName(row.method)),element("strong","",isRate?percent(row[metric]):`${number(row[metric],metric==="mean_model_steps"?0:1)}${metric==="planning_ms_p95"?" ms":""}`));const track=element("div","chart-track");const fill=element("div","chart-fill");fill.style.width=`${Math.max(0,Math.min(100,(row[metric]||0)/max*100))}%`;track.append(fill);item.append(label,track);return item;}));
    ui["results-body"].replaceChildren(...results.map(row=>{const tr=element("tr",row.method===ui["method-select"].value?"selected":"");[methodName(row.method),number(row.episodes,0),percent(row.success_rate),percent(row.collision_rate),percent(row.timeout_rate),number(row.mean_steps),`${number(row.planning_ms_p50,2)} / ${number(row.planning_ms_p95,2)} ms`,number(row.mean_model_steps,0)].forEach(value=>tr.append(element("td","",value)));return tr;}));
    const count=results.reduce((sum,row)=>sum+(row.episodes||0),0);ui["evidence-scope"].textContent=`${conditionName(condition)} · ${results.length} 种方法 · ${count} 次评估（各方法回合数见表）`;
  }
  function path(points,transform,{color,width=1,dash=[],alpha=1}={}){
    if(!Array.isArray(points)||points.length<2)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.globalAlpha=alpha;ctx.setLineDash(dash);ctx.lineJoin="round";ctx.lineCap="round";ctx.beginPath();let started=false;for(const s of points){if(!Array.isArray(s)||!finite(s[0])||!finite(s[1]))continue;const p=transform(s);if(!started){ctx.moveTo(p[0],p[1]);started=true;}else ctx.lineTo(p[0],p[1]);}ctx.stroke();ctx.restore();
  }
  function drawWorld(){
    const canvas=ui["world-canvas"],bounds=ui["world-container"].getBoundingClientRect();const dpr=Math.min(window.devicePixelRatio||1,2);const w=bounds.width,h=bounds.height;if(!w||!h)return;
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle="#0a171d";ctx.fillRect(0,0,w,h);
    const map=finite(state.meta.map_size)?state.meta.map_size:10;const scale=Math.min((w-80)/map,(h-86)/map);const size=scale*map;const left=(w-size)/2,top=(h-size)/2-4;const xy=s=>[left+s[0]*scale,top+size-s[1]*scale];
    state.transform={left,top,size,scale,map};
    const gradient=ctx.createRadialGradient(w*.5,h*.48,5,w*.5,h*.48,w*.55);gradient.addColorStop(0,"#12302b35");gradient.addColorStop(1,"#0a171d00");ctx.fillStyle=gradient;ctx.fillRect(0,0,w,h);
    ctx.lineWidth=1;for(let i=0;i<=map*2;i++){const pos=i*scale/2;ctx.strokeStyle=i%2?"#16293275":"#20353d90";ctx.beginPath();ctx.moveTo(left+pos,top);ctx.lineTo(left+pos,top+size);ctx.moveTo(left,top+pos);ctx.lineTo(left+size,top+pos);ctx.stroke();}
    ctx.strokeStyle="#3c585c";ctx.lineWidth=1;ctx.strokeRect(left,top,size,size);ctx.strokeStyle="#81aaa0";const tick=8;[[left,top,1,1],[left+size,top,-1,1],[left,top+size,1,-1],[left+size,top+size,-1,-1]].forEach(([x,y,dx,dy])=>{ctx.beginPath();ctx.moveTo(x+dx*tick,y);ctx.lineTo(x,y);ctx.lineTo(x,y+dy*tick);ctx.stroke();});
    ctx.font="8px Consolas,monospace";ctx.fillStyle="#49656f";ctx.textAlign="center";for(let i=0;i<=map;i+=2){ctx.fillText(String(i),left+i*scale,top+size+14);ctx.textAlign="right";ctx.fillText(String(i),left-9,top+size-i*scale+3);ctx.textAlign="center";}ctx.fillStyle="#607e85";ctx.textAlign="left";ctx.fillText("x",left+size+13,top+size+3);ctx.fillText("y",left-2,top-12);
    const frame=currentFrame();if(!frame?.state||frame.state.length<6)return;const s=frame.state;ctx.save();ctx.beginPath();ctx.rect(left-2,top-2,size+4,size+4);ctx.clip();
    const goal=xy([s[4],s[5]]);ctx.strokeStyle="#c3d4a7";ctx.fillStyle="#c3d4a711";ctx.lineWidth=1;ctx.beginPath();ctx.arc(...goal,scale*.38,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.save();ctx.translate(...goal);ctx.rotate(Math.PI/4);ctx.strokeRect(-4,-4,8,8);ctx.restore();
    if(ui["show-candidates"].checked)(frame.candidate_paths||[]).forEach(points=>path([s,...points],xy,{color:"#4ea095",alpha:.38,width:1}));
    const history=(state.mode==="live"?state.live.frames:state.episode.frames.slice(0,state.frame+1)).map(f=>f.state);path(history,xy,{color:"#d4e0df",width:1.8,alpha:.85});
    if(state.mode==="replay"&&ui["show-actual"].checked){const ahead=Math.max(1,frame.horizon||((frame.predicted_path?.length||1)-1));path(state.episode.frames.slice(state.frame,state.frame+ahead+1).map(f=>f.state),xy,{color:"#f1c27b",dash:[4,5],width:1.6,alpha:.85});}
    if(frame.predicted_path?.length){path([s,...frame.predicted_path],xy,{color:"#61e4c2",width:6,alpha:.075});path([s,...frame.predicted_path],xy,{color:"#61e4c2",width:1.9});const end=xy(frame.predicted_path[frame.predicted_path.length-1]);ctx.beginPath();ctx.arc(...end,3.2,0,Math.PI*2);ctx.fillStyle="#61e4c2";ctx.fill();}
    const obstacleRadius=(state.meta.obstacle_radius??.38)*scale;
    for(let i=6;i+3<s.length;i+=4){const p=xy([s[i],s[i+1]]);ctx.fillStyle="#ec857d12";ctx.strokeStyle="#ec857d78";ctx.lineWidth=1;ctx.beginPath();ctx.arc(...p,obstacleRadius,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle="#ec857d";ctx.beginPath();ctx.arc(...p,3,0,Math.PI*2);ctx.fill();const vx=s[i+2],vy=s[i+3];if(finite(vx)&&finite(vy)){const velocity=Math.hypot(vx,vy);if(velocity>.02){const length=Math.min(25,Math.max(9,velocity*scale*.5));const dx=vx/velocity,dy=-vy/velocity;ctx.strokeStyle="#ec857d8a";ctx.beginPath();ctx.moveTo(p[0]+dx*obstacleRadius,p[1]+dy*obstacleRadius);const end=[p[0]+dx*(obstacleRadius+length),p[1]+dy*(obstacleRadius+length)];ctx.lineTo(...end);ctx.lineTo(end[0]-dx*4-dy*2.5,end[1]-dy*4+dx*2.5);ctx.moveTo(...end);ctx.lineTo(end[0]-dx*4+dy*2.5,end[1]-dy*4-dx*2.5);ctx.stroke();}}}
    const robot=xy(s),rr=(state.meta.robot_radius??.18)*scale;ctx.fillStyle="#61e4c215";ctx.beginPath();ctx.arc(...robot,rr+7,0,Math.PI*2);ctx.fill();ctx.fillStyle="#ddf9ef";ctx.strokeStyle="#61e4c2";ctx.lineWidth=2;ctx.beginPath();ctx.arc(...robot,Math.max(rr,4),0,Math.PI*2);ctx.fill();ctx.stroke();
    const angle=Math.atan2(-(s[3]||frame.action?.[1]||0),s[2]||frame.action?.[0]||1);ctx.save();ctx.translate(...robot);ctx.rotate(angle);ctx.fillStyle="#0c554a";ctx.beginPath();ctx.moveTo(3,0);ctx.lineTo(-2,-2.5);ctx.lineTo(-2,2.5);ctx.closePath();ctx.fill();ctx.restore();ctx.restore();
    ctx.font="8px Consolas,monospace";ctx.fillStyle="#c3d4a7";ctx.textAlign="center";ctx.fillText("GOAL",goal[0],goal[1]-scale*.38-8);ctx.fillStyle="#9ddac9";ctx.textAlign="left";const labelX=Math.min(left+size-48,Math.max(left,robot[0]+rr+8));const labelY=Math.min(top+size-6,Math.max(top+12,robot[1]-rr-6));ctx.fillText("AGENT",labelX,labelY);
  }
  function updateSummary(){
    const summary=state.summary;if(!summary)return;
    const decision=summary.decision||{};const status=String(decision.status||"pending");
    ui["experiment-notice"].dataset.state=/^(go|promising|positive|pass|proceed|viable)$/.test(status)?"positive":/^(no_go|negative|fail|pivot)$/.test(status)?"negative":"pending";
    ui["notice-title"].textContent=summary.stage==="pilot"?"历史结果 · 第一轮选题验证":summary.stage==="smoke"?"最小闭环检查":"实验记录";
    ui["notice-text"].textContent=decision.text||"已载入实验指标。请结合工况、样本量和方法定义解读结果。";
    const date=summary.generated_at?new Date(summary.generated_at):null;const stamp=date&&!Number.isNaN(date.valueOf())?date.toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false}):"";
    ui["data-provenance"].textContent=state.mode==="live"?"上方：本地模型实时推理 · 下方：已记录的独立离线实验":`${stamp?`数据生成 ${stamp} · `:""}实验记录回放 · 非在线模型推理`;
  }
  async function liveRequest(endpoint,payload){
    const response=await fetch(`/api/${endpoint}`,{method:payload?"POST":"GET",headers:payload?{"Content-Type":"application/json"}:{},body:payload?JSON.stringify(payload):undefined,cache:"no-store"});
    if(!response.ok){let message=`请求失败（${response.status}）`;try{const body=await response.json();message=body.error||message;}catch{}throw new Error(message);}
    const result=await response.json();if(result.error)throw new Error(result.error);return result;
  }
  async function liveOperation(endpoint,payload){
    const previous=state.live.pending||Promise.resolve();
    const request=previous.catch(()=>{}).then(()=>liveRequest(endpoint,payload));state.live.pending=request;
    try{return await request;}finally{if(state.live.pending===request)state.live.pending=null;}
  }
  function liveControls(disabled){state.live.controlsDisabled=disabled;ui["play-button"].disabled=disabled;ui["reset-button"].disabled=disabled;ui["method-select"].disabled=disabled;ui["apply-dynamics"].disabled=disabled||!state.live.session||state.live.done;}
  function showLiveError(error){setPlay(false);liveControls(false);ui["mode-status"].textContent=`实时推理不可用：${error.message}`;ui["play-button"].disabled=!state.live.session||state.live.done;}
  async function setMode(mode){
    if(mode===state.mode)return;
    setPlay(false);state.live.generation++;state.mode=mode;document.body.classList.toggle("live-mode",mode==="live");
    ["live","replay"].forEach(name=>{ui[`mode-${name}`].classList.toggle("active",mode===name);ui[`mode-${name}`].setAttribute("aria-pressed",String(mode===name));});
    ui["live-fields"].hidden=mode!=="live";ui["actual-legend"].hidden=mode==="live";ui["actual-option"].hidden=mode==="live";ui["view-label"].textContent=mode==="live"?"本地模型实时推理":"真实实验回放";
    ui["reset-button"].classList.toggle("live-reset",mode==="live");ui["reset-button"].textContent=mode==="live"?"重置场景":"↺";ui["reset-button"].setAttribute("aria-label",mode==="live"?"重置场景":"回到回合起点");
    ui["inspector-note"].textContent=mode==="live"?"↳ 显示最近一次决策的动作前状态与预测；下一次决策刷新真实位置。候选未来来自模型实时计算。":"↳ 实际后续含重新规划，偏差也包含动作改变，不能直接当作同动作预测误差。";
    if(mode==="replay"){
      ui["mode-status"].textContent="正在查看已记录的真实实验";const available=[...new Set(state.episodes.map(ep=>ep.method))];options(ui["method-select"],available,methodName,ui["method-select"].value);chooseMethod();updateSummary();return;
    }
    ui["mode-status"].textContent="连接本地模型服务…";ui.timeline.disabled=true;ui.timeline.max="1";ui.timeline.value="0";ui["frame-total"].textContent="LIVE";state.live.frame=null;ui["scene-empty"].hidden=true;ui["scene-caption"].hidden=false;ui["scene-outcome"].className="outcome";liveControls(true);updateFrame();
    try{const status=await liveOperation("status");if(state.mode!=="live")return;if(!status.ready)throw new Error(status.message||"世界模型尚未准备好");state.live.ready=true;state.live.methods=(status.methods||["adaptive","fixed_5","fixed_10","fixed_16","global_physics","local_identification"]).filter(method=>method!=="gated_calibrated"||status.gated_ready);options(ui["method-select"],state.live.methods,methodName,status.gated_ready?"gated_calibrated":ui["method-select"].value||"adaptive");await resetLive();}
    catch(error){if(state.mode==="live")showLiveError(error);}
  }
  async function resetLive(){
    if(state.mode!=="live")return;setPlay(false);const generation=++state.live.generation;liveControls(true);ui["mode-status"].textContent="正在重置场景与规划器…";
    const seed=Number(ui["live-seed"].value);const payload={method:ui["method-select"].value||"adaptive",seed:Number.isFinite(seed)?Math.max(0,Math.floor(seed)):2026,scenario:ui["live-scenario"].value,damping_scale:Number(ui["live-damping"].value)};
    try{const result=await liveOperation("reset",payload);if(generation!==state.live.generation||state.mode!=="live")return;state.live.session=result.session_id;state.live.frame=result.frame;state.live.frames=[result.frame];state.live.step=0;state.live.done=false;ui["scene-empty"].hidden=true;ui["scene-caption"].hidden=false;ui["scene-outcome"].className="outcome";ui.timeline.max="1";ui["frame-total"].textContent="LIVE";ui["mode-status"].textContent="场景已就绪 · 点击开始，也可点击地图修改目标";ui["data-provenance"].textContent="上方：本地模型实时推理 · 下方：已记录的独立离线实验";ui["method-description"].textContent=methods[payload.method]?.[1]||"使用选定方法实时规划。";liveControls(false);updateFrame();renderEvidence();}
    catch(error){if(generation===state.live.generation&&state.mode==="live")showLiveError(error);}
  }
  async function liveStep(){
    if(!state.playing||state.mode!=="live"||state.live.done||state.live.pending)return;
    const generation=state.live.generation;const start=performance.now();
    try{const result=await liveOperation("step",{session_id:state.live.session});if(generation!==state.live.generation||state.mode!=="live")return;
      state.live.frame=result.frame;state.live.step=result.step??state.live.step+1;state.live.frames.push(result.frame);state.live.done=Boolean(result.done);ui.timeline.max=String(Math.max(1,state.live.step));ui["mode-status"].textContent=`本地实时推理 · ${methodName(ui["method-select"].value)} · 已执行 ${state.live.step} 步`;
      if(state.live.done){setPlay(false);ui["play-button"].disabled=true;ui["play-button"].textContent="已结束";const end={...result.frame,state:result.next_state||result.frame.state,action:[0,0],horizon:0,predicted_path:[],candidate_paths:[]};state.live.frame=end;state.live.frames.push(end);const outcome=ui["scene-outcome"];outcome.className=`outcome visible${result.collision?" collision":result.success?"":" timeout"}`;outcome.textContent=result.success?"实时回合 · 到达目标":result.collision?"实时回合 · 发生碰撞":"实时回合 · 已结束";ui["mode-status"].textContent="本回合结束 · 重置场景后可再次运行";}
      updateFrame();
      if(state.playing)state.live.timer=setTimeout(()=>void liveStep(),Math.max(0,(state.meta.dt||.15)*1000-(performance.now()-start)));
    }catch(error){if(generation===state.live.generation&&state.mode==="live")showLiveError(error);}
  }
  async function setLiveGoal(event){
    if(state.mode!=="live"||!state.live.session||!state.transform||state.live.done||state.live.controlsDisabled)return;const bounds=ui["world-canvas"].getBoundingClientRect(),t=state.transform;const x=(event.clientX-bounds.left-t.left)/t.scale,y=t.map-(event.clientY-bounds.top-t.top)/t.scale;if(x<.25||y<.25||x>t.map-.25||y>t.map-.25)return;
    const resume=state.playing;setPlay(false);const generation=++state.live.generation;liveControls(true);ui["mode-status"].textContent="正在设置新目标…";
    try{const result=await liveOperation("goal",{session_id:state.live.session,x,y});if(generation!==state.live.generation||state.mode!=="live")return;state.live.frame=result.frame;state.live.done=false;state.live.frames.push(result.frame);ui["scene-outcome"].className="outcome";ui["mode-status"].textContent=`目标已更新为 (${x.toFixed(1)}, ${y.toFixed(1)}) · ${resume?"继续规划":"点击开始"}`;liveControls(false);updateFrame();if(resume)setPlay(true);}
    catch(error){if(generation===state.live.generation&&state.mode==="live")showLiveError(error);}
  }
  async function applyDynamics(){
    if(state.mode!=="live"||!state.live.session||state.live.done||ui["apply-dynamics"].disabled)return;
    const resume=state.playing;setPlay(false);const generation=state.live.generation;liveControls(true);
    ui["mode-status"].textContent="正在改变环境阻尼…";
    try{
      // Queue behind any running step. Its result remains visible and the
      // planner keeps all history; only the subsequent physics is changed.
      const result=await liveOperation("dynamics",{session_id:state.live.session,damping_scale:Number(ui["live-damping"].value)});
      if(generation!==state.live.generation||state.mode!=="live")return;
      ui["mode-status"].textContent=`环境阻尼已变为 ${number(result.damping_scale,2)}× · 机器人将在后续行动中感知变化`;
      liveControls(false);if(resume)setPlay(true);
    }catch(error){if(generation===state.live.generation&&state.mode==="live")showLiveError(error);}
  }
  async function loadData(){
    ui["reload-data"].disabled=true;clearTimeout(state.retry);
    const stamp=Date.now();const read=async name=>{const response=await fetch(`../artifacts/day0/${name}.json?t=${stamp}`,{cache:"no-store"});if(!response.ok)throw new Error(`${name}: HTTP ${response.status}`);return response.json();};
    const [summary,replays,gateStudy]=await Promise.allSettled([read("summary"),read("replays"),fetch(`../artifacts/gated-study/summary.json?t=${stamp}`,{cache:"no-store"}).then(response=>{if(!response.ok)throw new Error("Gate study pending");return response.json();})]);
    if(gateStudy.status==="fulfilled"){
      const study=gateStudy.value;
      ui["gate-evidence-status"].textContent=`${study.training_seeds?.length??"—"} 次独立训练 · ${number(study.total_episode_runs,0)} 个方法运行回合。${study.decision?.text||"完整统计、消融与局限见本轮实验报告。"}`;
    }
    if(summary.status==="fulfilled"){state.summary=summary.value;updateSummary();const conditions=[...new Set((state.summary.results||[]).map(row=>row.condition))];if(conditions.length)options(ui["evidence-condition"],conditions,conditionName,ui["evidence-condition"].value);}
    if(replays.status==="fulfilled"){
      state.meta={...state.meta,...replays.value.meta};state.episodes=(replays.value.episodes||[]).filter(ep=>Array.isArray(ep.frames)&&ep.frames.length&&ep.frames.every(f=>Array.isArray(f.state)&&f.state.length>=6));
      const available=[...new Set(state.episodes.map(ep=>ep.method))];const previous=ui["method-select"].value;const preferred=available.includes(previous)?previous:available.find(name=>String(name).includes("adaptive"))||available[0];
      if(state.mode==="replay"){options(ui["method-select"],available,methodName,preferred);chooseMethod();}ui["scene-coordinate"].textContent=`ENV / ${state.meta.map_size} × ${state.meta.map_size}`;
    }
    renderEvidence();
    const ready=Boolean(state.summary||state.episodes.length);ui["data-status"].classList.toggle("ready",ready);ui["data-status"].replaceChildren(element("i"),document.createTextNode(state.episodes.length?"真实实验数据已载入":state.summary?"指标已载入 · 等待回放":"等待实验产物"));
    if(!ready){ui["notice-title"].textContent="等待第一轮实验";ui["notice-text"].textContent=location.protocol==="file:"?"请通过项目 HTTP 服务打开此页面，以读取本地实验数据。":"尚未找到实验产物。页面将自动重试，训练和评估进度请查看运行终端。";state.retry=setTimeout(loadData,15000);}
    else if(!state.episodes.length){ui["notice-text"].textContent+= " 回放数据尚不可用。";state.retry=setTimeout(loadData,15000);}
    ui["reload-data"].disabled=false;drawWorld();
  }
  ui["method-select"].addEventListener("change",chooseMethod);ui["condition-select"].addEventListener("change",chooseCondition);ui["episode-select"].addEventListener("change",chooseEpisode);ui["chart-metric"].addEventListener("change",renderEvidence);ui["evidence-condition"].addEventListener("change",renderEvidence);ui["reload-data"].addEventListener("click",loadData);
  ui["mode-live"].addEventListener("click",()=>void setMode("live"));ui["mode-replay"].addEventListener("click",()=>void setMode("replay"));ui["live-damping"].addEventListener("input",()=>{ui["damping-value"].textContent=`${Number(ui["live-damping"].value).toFixed(2)}×`;});ui["world-canvas"].addEventListener("click",event=>void setLiveGoal(event));
  ui["apply-dynamics"].addEventListener("click",()=>void applyDynamics());
  ui["play-button"].addEventListener("click",()=>{if(state.mode==="live"){if(state.live.done){void resetLive();return;}setPlay(!state.playing);return;}if(state.frame>=state.episode.frames.length-1)seek(0);setPlay(!state.playing);});ui["reset-button"].addEventListener("click",()=>{if(state.mode==="live"){void resetLive();return;}setPlay(false);seek(0);});ui.timeline.addEventListener("input",()=>{if(state.mode==="live")return;setPlay(false);seek(Number(ui.timeline.value));});ui["show-candidates"].addEventListener("change",drawWorld);ui["show-actual"].addEventListener("change",drawWorld);
  document.addEventListener("keydown",event=>{if(/^(INPUT|SELECT|TEXTAREA|BUTTON|A)$/.test(event.target.tagName)||(state.mode==="replay"&&!state.episode))return;if(event.code==="Space"){event.preventDefault();if(state.mode==="replay"&&state.frame>=state.episode.frames.length-1)seek(0);setPlay(!state.playing);}else if(state.mode==="replay"&&(event.code==="ArrowLeft"||event.code==="ArrowRight")){event.preventDefault();setPlay(false);seek(state.frame+(event.code==="ArrowLeft"?-1:1));}});
  new ResizeObserver(drawWorld).observe(ui["world-container"]);
  function tick(time){const elapsed=state.lastTime?Math.min(time-state.lastTime,250):0;state.lastTime=time;if(state.mode==="replay"&&state.playing&&state.episode){const fps=finite(state.meta.replay_fps)&&state.meta.replay_fps>0?state.meta.replay_fps:finite(state.meta.dt)&&state.meta.dt>0?1/state.meta.dt:10;state.accumulator+=elapsed*Number(ui["playback-speed"].value);const duration=1000/fps;if(state.accumulator>=duration){const steps=Math.floor(state.accumulator/duration);state.accumulator%=duration;state.frame=Math.min(state.frame+steps,state.episode.frames.length-1);updateFrame();if(state.frame>=state.episode.frames.length-1)setPlay(false);}}requestAnimationFrame(tick);}
  drawWorld();loadData();requestAnimationFrame(tick);
})();
