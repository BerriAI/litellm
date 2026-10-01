# LiteLLM Dashboard 中英文术语表（Glossary）

> 维护者：Agent 2（localization-designer）
> 状态：设计稿，待 G0 评审；Agent 2 负责复核与更新
> 约定：本术语表是**唯一权威词条来源**。所有 v1 相关译文必须与本表一致；出现同义词漂移即视为缺陷。`不可译` = 作为专名/数据保留英文，不进入翻译资源或不做翻译。

---

## 1. 产品核心术语（v1 高频）

| EN | ZH-CN | 备注 |
|---|---|---|
| LiteLLM | LiteLLM | 品牌名，不译 |
| Virtual Key | 虚拟密钥 | 产品专名；左导航"Virtual Keys"→"虚拟密钥"，页内统一 |
| API Key | API 密钥 | 与 Virtual Key 区分；两者都不译 API 缩写 |
| Model | 模型 | — |
| Model ID | 模型 ID | "ID"不译 |
| Model Name | 模型名称 | — |
| Model Provider | 模型提供商 | 或"模型供应商"，全站统一为"提供商" |
| Deployment | 部署 | 复数场景用计数+量词，如"2 个部署" |
| Endpoint | 端点 | "Models + Endpoints"→"模型与端点" |
| Router | 网关 | 指 LiteLLM Router 产品语义；"Router Settings"→"网关设置" |
| Proxy | 代理 | LiteLLM 代理/网关环境；"Proxy Base URL"→"代理基础 URL" |
| Budget | 预算 | — |
| Budget Limit | 预算上限 | — |
| Spend / Cost | 花费 / 成本 | `spend`（已花费）与 `cost`（成本/费用）字段语义，UI 文案统一：指"已花多少钱"用"花费"，页面标题/条目用"成本"。避免与"支出/费用"混用 |
| Cost Tracking | 成本跟踪 | 页面名 |
| Cost Optimization | 成本优化 | 页面名（含 Beta 徽标） |
| Usage | 用量 | 页面名与数据语义；"可用量"在配额上下文另见 Rate Limit |
| Rate Limit | 速率限制 | TPM/RPM 语义；避免与"阈值"混用 |
| Team | 团队 | — |
| Organization | 组织 | — |
| User | 用户 | — |
| Internal User | 内部用户 | "Internal Users"→"内部用户" |
| Logs | 日志 | — |
| Projects | 项目 | 项目（Beta 徽标） |
| Onboarding | 引导 | 首次引导流程；"Onboarding"页面/步骤用"引导" |
| Connect | 连接 | Connect 页面 |
| Guardrails | Guardrails | 产品专有功能名，保留英文原名（含"Guardrails Monitor"→"Guardrails 监控"外部仍保留原名） |
| Policies | 策略 | 通用"策略"；Guardrails/Policies 作为功能模块见 `V1_TRANSLATION_SCOPE.md`（v1 低优先级，默认不翻） |
| Settings | 设置 | — |
| Admin Settings | 管理设置 | — |
| Router Settings | 网关设置 | — |
| UI Theme | 界面主题 | — |
| MCP Server | MCP 服务器 | "MCP"不译 |
| Vector Store | 向量存储 | — |
| Playground | 演示台 | 或保留"Playground"；v1 低优先级，默认不翻，备注决策 |
| Agent | Agent | v1 低优先级；作为技术专名可保留，或在明确语境译"智能体"，未定不外扩 |
| Prompt | Prompt | 保留英文（行业惯用） |

## 2. 导航与页面结构词

| EN | ZH-CN | 备注 |
|---|---|---|
| AI Gateway | AI 网关 | — |
| Observability | 可观测性 | — |
| Access Control | 访问控制 | — |
| Developer Tools | 开发工具 | — |
| Search Tools | 搜索工具 | — |
| Tool Policies | 工具策略 | — |
| Tag Management | 标签管理 | — |
| API Reference | API 参考 | — |
| Response Cache | 响应缓存 | 或"缓存"；与"Cache"页面语义对应 |
| Learning Resources | 学习资料 | — |
| Experimental | 实验功能 | — |
| Old Usage | 旧版用量 | — |
| Usage | 用量 | — |
| Models + Endpoints | 模型与端点 | — |
| Skills | Skill | v1 低优先级；保留英文 |
| Workflow Runs | 工作流运行 | — |
| Memory | 记忆 | — |

## 3. 通用 UI 动作词

| EN | ZH-CN | 备注 |
|---|---|---|
| Save | 保存 | — |
| Cancel | 取消 | — |
| Delete | 删除 | 确认框/按钮全站统一 |
| Remove | 移除 | 多用于从列表移除关联；避免与 Delete 混用 |
| Create | 创建 | — |
| Add | 添加 | — |
| Edit | 编辑 | — |
| Update | 更新 | — |
| Enable / Disable | 启用 / 停用 | 与"开启/关闭"统一，全站用"启用/停用" |
| Generate | 生成 | — |
| Copy | 复制 | — |
| Close | 关闭 | — |
| Confirm | 确认 | — |
| Back | 返回 | — |
| Next | 下一步 | — |
| Submit | 提交 | — |
| Search | 搜索 | — |
| Filter | 筛选 | — |
| Export | 导出 | — |
| Import | 导入 | — |
| Refresh | 刷新 | — |
| Save Changes | 保存更改 | — |
| Sign in / Log in | 登录 | 页面/表单统一"登录" |
| Sign out / Log out | 登出 | — |
| Reset | 重置 | — |
| Clear | 清空 | — |

## 4. 表单与状态词

| EN | ZH-CN | 备注 |
|---|---|---|
| Name | 名称 | — |
| Description | 描述 | — |
| Required | 必填 | — |
| Optional | 可选 | label 用"（可选）"后缀 |
| Enabled | 已启用 | — |
| Disabled | 已停用 | — |
| Active / Inactive | 启用中 / 停用 | — |
| Loading… | 加载中… | — |
| Error | 错误 | — |
| Success | 成功 | — |
| None | 无 | — |
| All | 全部 | 筛选器"All"→"全部" |
| Selected | 已选择 | 计数插值：`已选择 {{count}} 个模型` |
| Default | 默认 | — |
| Custom | 自定义 | — |

## 5. 时间 / 用量 / 数值词

| EN | ZH-CN | 备注 |
|---|---|---|
| requests | 请求 | 量词"个"：`{{count}} 个请求` |
| tokens | Token | 保留英文复数语义，"Token"不译 |
| per minute | 每分钟 | TPM 语义，缩写 TPM/RPM 保留 |
| month / this month | 本月 | 与"此月/当月"统一为"本月" |
| last 24 hours | 过去 24 小时 | — |
| total | 合计 | 表格总计列 |
| average | 平均 | — |
| limit | 上限/限制 | 上下文定："Budget Limit"用"预算上限"；"Rate Limit"用"速率限制" |

## 6. 校验 / 空状态 / 提示词（示例对照）

| EN | ZH-CN | 备注 |
|---|---|---|
| This field is required | 此项为必填 | 校验提示 |
| Invalid email address | 邮箱地址无效 | — |
| No results found | 未找到结果 | 列表空态 |
| Something went wrong | 出错了，请重试 | 通用错误 |
| Unable to connect | 无法连接 | — |
| Please try again | 请重试 | — |

---

## 词条统计与约定

- 本表共计 **63 条**词条（第 1 节 32 + 第 2 节 14 + 第 3 节 25 + 第 4 节 14 + 第 5 节 12 + 第 6 节 7，去重后实际 63）。
- 所有新增译名必须先入表再由开发使用；Agent 2 在 Wave 4A 复核。
- "不可译"列入表内的专名（LiteLLM、Guardrails、MCP、Token、Prompt、VM 等）作为约定记录，防止误翻。
