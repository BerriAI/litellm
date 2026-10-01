# v1 翻译范围清单（V1 Translation Scope）

> 维护者：Agent 2（localization-designer）
> 状态：设计稿，待 G0 评审冻结；与 `I18N_MULTI_AGENT_PLAN.md` §1.2/§6.2 及 `DECISIONS.md` D8 对齐
> 说明：本文档逐项列出 v1 各页面/组件"**需翻译**"或"**明确不翻译**"及原因，作为 Agent 5/6 改造与 Review 的依据。数据/协议内容是否翻译遵循 `LOCALIZATION_SPEC.md` §4。

---

## 1. v1 需翻译范围（逐项）

### 1.1 全局壳层

| 组件/区域 | 需翻译? | 原因 / 说明 |
|---|---|---|
| `leftnav.tsx` 导航菜单项与分组标签 | 需翻译 | 全局入口；`menuGroups` 的 `groupLabel`、`label` 需 I18N 化（含分组标签如 AI Gateway/Observability 等）。`menuGroups` 由 `page_utils.ts` 消费，改造时保留导出结构，仅语义化 label |
| `leftnav.tsx` 面包屑（`getBreadcrumb`）/ Navbar 标题 | 需翻译 | 与导航一致，防止两处不一致 |
| `leftnav.tsx` aria-label（折叠/展开侧边栏） | 需翻译 | a11y 文案（`LOCALIZATION_SPEC.md` §7） |
| Navbar 用户菜单 / `SidebarAccountMenu` | 需翻译 | 显示名、菜单项、提示文案 |
| `navbar.tsx` 其余可见菜单与搜索占位 | 需翻译 | 依实际文案盘点 |
| 语言切换器（新组件） | 需翻译 | 自身 aria-label 与菜单项随语言渲染 |

### 1.2 登录与引导

| 页面 | 需翻译? | 说明 |
|---|---|---|
| `src/app/login/LoginPage.tsx` | 需翻译 | 登录表单、提交、错误提示、SSO 入口 |
| `src/app/onboarding/`（OnboardingForm/FormBody/Loading/Error 等） | 需翻译 | 引导步骤、表单、加载/错误视图 |
| `src/app/connect/` | 需翻译 | Connect 页面 |
| MCP OAuth 相关回跳/授权提示 UI | 需翻译 | 见 `LOCALE_NAVIGATION_BEHAVIOR.md`（注意整页跳转语言恢复） |

### 1.3 Models 与 API Keys

| 页面/组件 | 需翻译? | 说明 |
|---|---|---|
| Models and Endpoints（`/models`）列表、创建/编辑表单、过滤、空态、校验 | 需翻译 | 静态与列表类文案；`model-hub`/AI Hub 候选页属低优先级，v1 不强求 |
| API Keys（`/api-keys`）列表、创建弹窗、删除确认、用量展示 | 需翻译 | 含确认框、空态、校验 |
| 相关复用组件（`key_value_input`、`copy button` 等） | 需翻译 | 复用组件文案由调用方传入或按需本地化 |

### 1.4 Usage / Cost / Budget

| 页面/组件 | 需翻译? | 说明 |
|---|---|---|
| Usage（`/usage`）筛选、表格头、图表说明、空态 | 需翻译 | 数值/货币用 i18n 格式化 |
| Cost Tracking（`/cost-tracking` / `cost-optimization`） | 需翻译 | 成本跟踪页面；数值格式化随 i18n |
| Budgets（`/budgets`）创建/编辑、限额设置、列表 | 需翻译 | 含"Budget Limit"等术语（见术语表） |

---

## 2. 明确不翻译范围（逐项 + 原因）

遵循 `LOCALIZATION_SPEC.md` §4 与方案 §1.2：

| 项 | 不翻译原因 |
|---|---|
| 模型名 / Model ID / Deployment / Provider 名 | 用户/后端数据，非 UI 文案 |
| API 字段、请求/响应键、JSON 结构 | 协议内容 |
| 日志原文、错误消息、错误堆栈（stack trace） | 需原样排障，保留原文 |
| 代码示例、curl、终端命令、config.yaml、.env | 技术内容 |
| URL、路径、文件名 | 技术标识 |
| API / URL / SSO / MCP / OAuth / REST / ID / JSON 等缩写 | 通用技术缩写 |
| 品牌名 LiteLLM / OpenAI / Anthropic | 品牌 |
| 角色/权限代码值（admin、internal_user 等） | 代码值，仅显示名可翻译 |
| Beta / New 徽标 | 视觉因子，保留英文 |
| 构建期 `<title>` / metadata description | D8，v1 保持英文，不做多语言 SEO |
| Guardrails / Policies / Teams / Users / Orgs / Projects / Logs / Playground / Prompts 等整模块文案（未列入 v1 功能批次） | 属 v2+ 候选（方案 §5.7），v1 范围外不被动外扩 |
| Python SDK、API 文档、代码示例翻译 | 方案 §1.2 明确排除 |
| 后端 FastAPI 错误响应全面国际化 | 方案 §1.2 排除 |

---

## 3. 与 §6.2 状态清单对应

本清单即 `V1_SCOPE_MANIFEST.md`（Agent 0 维护）中 v1 行的**文案盘点输入**。Agent 5/6 完成任务时在清单相应行填写 EN/ZH-CN 盘点，Agent 7 填测试/UI 检查证据，Agent 2 在 Wave 4A 做术语与中文体验复核。

## 4. 边界红线

- 未列入第 1 节的功能页面文案，v1 一律**不翻译**（保持英文），不要顺手外扩造成半成品。
- 第 2 节不翻译项，任何 Agent 不得翻译；发现误翻即退回原 Owner，并作为缺陷记录。
