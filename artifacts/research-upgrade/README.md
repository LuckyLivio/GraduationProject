# 升级选题的检索与来源记录

访问日期：2026-09-17。对应总报告：[world-model-upgrade-decision.md](../../docs/world-model-upgrade-decision.md)。本目录是文献核验材料，不含新模型训练结果。

| 文件 | 用途与阅读范围 |
|---|---|
| `discovery-queries.json`、`focused-discovery.json` | OpenAlex 检索发现记录；搜索命中本身不等于已核验论文结论 |
| `interaction-sources.json` | 推物、对象模型、主动物性推断的摘要、指定正文章节与作者仓库摘录 |
| `pusht-source.json` | Diffusion Policy / DINO-WM 的 PushT 源码、提交号、动作/观察/成功协议与 Pymunk 文档；只读核验，未运行 |
| `memory-sources.json` | PlaNet、Plan2Explore、Memory Maze、DreamerV3、NWM 的摘要/作者说明；未通读全部正文 |
| `visual-sources.json` | DIAMOND、IRIS、Dreamer4、DINO-WM、V-JEPA2 等来源、指定算力章节与搜索限制 |
| `navigation-primary.json` | NWM、NavWM 摘要及 NWM 作者 README |
| `navigation-details.json` | NWM/NavWM 指定正文段落；另含图式主动感知论文的 OpenAlex 摘要，后者是二手索引内容，不冒充全文 |
| `nearest-memory-navigation.json` | UniWM 早期版本与 NWM 项目资料；UniWM 当前题名/设置以后一文件为准 |
| `uniwm-current.json` | UniWM 作者仓库与 arXiv v3 的记忆、训练和资源说明 |

各 JSON 记录请求 URL、状态、获取时间（部分在文件级）、响应哈希及相应摘录或解析内容。哈希用于核对当次响应，不代表作者结论得到独立复现。保留了 arXiv API 406、搜索引擎拦截/无关结果等失败；未用无关命中作证据。

原始整页下载置于 Git 忽略的 `outputs/` 中；本目录只提交紧凑访问记录和选题所需摘录。尚未运行外部作者代码。代码可访问不等于本机可复现，配置中的 GPU 数量不一定是最低要求。
