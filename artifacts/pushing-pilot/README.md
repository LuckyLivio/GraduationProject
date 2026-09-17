# 已运行的接触推物开发筛查

日期：2026-09-17。结论与所有范围限制见[结果报告](../../docs/pushing-pilot-findings.md)；[初始协议](../../docs/pushing-pilot-protocol.md)。此目录不包括视觉视频模型或主动试探算法的实验。

- `manifest.json`：训练/验证/开发数据种子、数量、接触比例与压缩数据哈希。
- `sensitivity.json`：相同初始状态/动作下的参数单因素检查。
- `training.json`、`matched-training.json`：9个实际训练运行的验证曲线、更新数、计时、GPU与峰值张量分配。
- `*.npz`：9个小模型的可移植张量参数，使用 `PushWorldModel.load` 读取，不需 pickle。
- `prediction.json`：40场景、3模型种子的预测结果，含强物理辨识、错配历史、动作分支检查。
- `matched-prediction.json`：首轮后补充的同容量无历史控制，保留原始结果。
- `planning.json`：8固定场景×4方法的真实执行、每步候选预测、预算与时延；达到位姿不代表停稳。
- `all_cases.json`：全部40场景的预测/真实动作回放；页面只展示固定首例与最大历史误差例。
- `demo.json`：从实际记录派生的网页数据，包括固定场景0/1的四动作未来。
- `overview.png/pdf`：来自原始数值的可导出图表。
- `audit.json`：数据/参数哈希与回放指标复算、探索性场景配对区间、交付源码哈希。
- `ui-check.json`：Edge真实浏览器检查结果。

首次训练前提交：`bde9e25`。容量控制、portable权重读取与页面在后续交付提交。开发筛查不能作为最终论文盲测；之后调参需独立确认集。
