# 运行与复现

## 快速体验

Python 3.10+，无需 GPU。仓库内 `artifacts/day0/model.npz` 是首轮真实训练的 3 成员轻量模型，约 17KB。运行时使用 NumPy；训练阶段才需要 PyTorch。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-demo.txt
.\.venv\Scripts\python scripts/serve.py --port 8765
```

浏览器打开 `http://127.0.0.1:8765/web/`。实时模式调用本机 Python 推理；回放模式读取保存的 JSON。若端口已占用，改成 `--port 8766` 并打开对应地址。

先体验新对照页 `http://127.0.0.1:8765/web/compare.html`：四方法预测同一动作序列，白线为真实执行，彩线为模型预测。切换正常/整体变化、推进/滑行/制动，直接比较误差；再切换导航，观察同场景各自规划。种子用于导航，预测诊断使用明确的固定共同初态。结果按需生成并在服务器短缓存，页面动画是同步回放。[比较口径与六组真实读数](comparison-lab.md)。

实时模式支持三种场景、八种方法、阻尼倍率与点击修改目标。选择门控校准时，加载复核种子 142 的冻结模型及独立验证阈值；可在途中施加阻尼变化而不清空历史，也可重置开始新回合。它是二维仿真导航，不连接真实机器人。网页中的离线成绩仍为清楚标注的首轮六方法实验，不随交互操作改变。

演示示例：门控方法、open 场景、种子 2026、阻尼 1.0，运行到约第 8 步并暂停；将阻尼改为 1.7，点击“施加阻尼变化”后继续。观察门控从保留先验到启用校准。显示的是实际启停状态和 `abs(log(scale))` 分数，不是变化概率。动力学修改只作用于仿真器，模型需通过后续转移自行发现变化。

## 训练与实验

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m foresight.experiment --output artifacts/local-run --episodes 12 --train-episodes 240 --epochs 80 --budget 4500 --seed 42 --navigation-seed 20000
```

运行会先保存协议，再采集数据、训练、选择固定视野、测试并导出回放。不要覆盖已提交的实验文件夹。默认首轮训练实现使用 CPU，以降低小型网络调度开销；无需安装 CUDA 版本 PyTorch。

复核三个独立训练运行：

```powershell
.\.venv\Scripts\python scripts/replicate.py --episodes 24 --output artifacts/local-replication
```

该命令训练种子为 142/242/342；每个运行有 3 个集成成员，训练数据固定，测试场景配对。共 864 个方法运行回合，但不是 864 个独立环境布局。已保存的协议见 `artifacts/replication/protocol.json`。脚本会拒绝覆盖已有协议目录；如需新的实验批次，指定新的输出目录。

复现后续残差校准探索（使用冻结的首轮模型）：

```powershell
.\.venv\Scripts\python scripts/probe_residual.py --output artifacts/local-residual-probe
```

## 结果文件

复现门控研究（仓库已带冻结模型，运行无需重训；输出到新目录）：

```powershell
.\.venv\Scripts\python scripts/study_gate.py --output artifacts/local-gated-study --episodes 18
.\.venv\Scripts\python scripts/analyze_gate.py --input artifacts/local-gated-study
```

对应预先固定的[门控协议](gated-protocol.md)。默认 `artifacts/gated-study` 已有证据，脚本拒绝覆盖。新研究的 `calibration.json` 记录验证阈值，`seed*-episodes.json` 记录逐步门控与导航结果，`prediction-windows.json` 记录同动作完整 16 步窗口，`detection.json` 记录检出和未检出。提交的小模型来自原复核训练，无需本地原始检查点即可运行；若要重新训练，请按复核流程生成自己的模型，并单独保留新模型来源。

| 文件 | 含义 |
| --- | --- |
| `protocol.json` | 方法、预算、数据种子、环境条件；新运行还记录源代码哈希 |
| `training.json` | 成员验证损失、训练时间与模型说明 |
| `prediction.json` | 相同动作序列的开环位置误差；验证分歧阈值 |
| `planner-validation.json` | 固定视野的选择依据，独立于导航测试 |
| `episodes.json` | 每局成功/碰撞、逐步耗时、预算和视野 |
| `summary.json` | 汇总指标和边界说明 |
| `replays.json` | 真实状态、候选未来与选中轨迹的回放 |

训练数据写入本地 `data/processed`，模型写入 `checkpoints`，均默认不进入 Git。首轮小型展示模型作为可复现演示资源另存到 `artifacts/day0/model.npz`，随代码发布。

生成首轮派生图表：

```powershell
.\.venv\Scripts\python scripts/analyze_pilot.py --input artifacts/day0
```

## 解释结果的约束

- 学到的是位置相关阻尼；动作积分与障碍运动仍为已知物理结构。不能称为从图像学会完整世界。
- 原始实验函数在构造环境后再调用一次 `reset()`，因此种子对应 RNG 生成的第二个布局；各方法一致且可复现。实时接口显式 `reset(seed=seed)` 对应第一个布局，所以同数值种子的实时场景不保证等于原始离线回放。
- 模型预测的是当时的一整段候选动作，实际执行随后会重规划。展示轨迹的分离不等同于模型误差；离线误差指标使用相同动作序列。
- 固定模型仅输入位置，无法从同一个位置直接知道隐藏阻尼倍率变化；集成一致不保证正确。
- 逻辑模型步数匹配不等于延迟匹配。当前首轮延迟统计记录规划部分，未包含物理参数更新和网络/UI 延迟；不据此宣称完整系统实时安全保证。
- 当前实验是选题诊断与复核；若据此开发新方法，后续论文评测必须保留新的未用于调参的测试集。

当前开发环境实际验证：Python 3.10.9、NumPy 1.21.4、PyTorch 2.10.0.dev20251124+cu130、Matplotlib 3.8.0。依赖文件给出兼容下限，不表示所有组合已实测。
