# LiteLLM Dashboard 中英文国际化多智能体实施方案

## 文档版本信息

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.4 |
| 修订日期 | 2026-09-09 |
| 文档状态 | 评审通过，可执行 |
| 适用项目 | LiteLLM Dashboard |
| 实施范围 | Dashboard 前端 UI，英文 `en` 与简体中文 `zh-CN` |
| 默认语言 | 英文 `en` |
| 并行约束 | 最多 4 个智能体同时运行，包含总控智能体 |
| 上一版本 | v1.3（评审通过，可执行） |

### v1.4 变更摘要

1. 明确 namespace 同时对应资源文件名和翻译 key 的冒号前缀。
2. 固定 namespace 注册表、资源加载映射和类型声明归 Agent 4 管理。
3. 固定 `package.json`、`package-lock.json` 的单一写入者，禁止并行 worktree 产生锁文件分叉。
4. 明确下游 worktree 从 G1 集成基线创建或 rebase，并使用 `npm ci` 安装独立 `node_modules`。
5. 增加简体中文计数与量词策略，禁止机械复制英文复数分支。
6. 明确 `<Trans>` 只能在资源同步就绪或统一 ready 门禁之后渲染。
7. 将首屏、`<Trans>`、刷新恢复、英文回退和整页跳转验证固化为具名证据文件。
8. 明确 Agent 7 在 Wave 1 获取平台候选提交进行验证时不得修改依赖文件。

---

## 1. 目标与范围

### 1.1 目标

在不改变现有路由和业务行为的前提下，为 LiteLLM Dashboard 增加：

- 英文 `en` 与简体中文 `zh-CN`。
- 页面内即时语言切换。
- 刷新后保留语言偏好。
- 中文缺失时回退英文。
- 导航、登录、引导和高频业务页面优先中文化。
- 可持续扩展的字典、测试与协作机制。

### 1.2 v1 范围

v1 包含：

- i18n 基础设施。
- 语言切换器和语言偏好保存。
- Navbar、Leftnav、用户菜单等全局壳层。
- Login、Onboarding、Connect、MCP OAuth。
- Models and Endpoints、API Keys。
- Usage、Cost Tracking、Budgets。
- 对应的自动化测试、静态导出构建验证和中文布局检查。

v1 不包含：

- Python SDK、API 文档和代码示例的翻译。
- 模型返回内容、日志原文和错误堆栈翻译。
- 所有 1400 余个 TSX 文件的一次性全量改造。
- FastAPI 后端错误响应的全面国际化。
- 后端 httpOnly Cookie 语言协商。
- 构建期 `<title>` 与 metadata description 的多语言 SEO 输出；v1 保持英文。
- Guardrails、Policies、Playground、Prompts 等低优先级模块的全量翻译。

---

## 2. 已确认的工程现状

- 前端目录：`ui/litellm-dashboard`。
- 技术栈：Next.js 16、React 19、TypeScript。
- 使用 App Router。
- `next.config.mjs` 配置 `output: "export"`，属于静态导出。
- 项目原始基线没有成熟的 i18n 框架。
- 根 Provider 链位于 `src/app/layout.tsx`。
- 导航主要位于 `src/components/leftnav.tsx` 和 `src/components/navbar.tsx`。
- 项目对单元、集成和 E2E 测试有明确分类要求。
- 不允许无路径运行完整 Vitest 测试集，只运行改动相关测试。

### 2.1 当前工作树状态

截至 2026-09-08，除本方案文档 `I18N_MULTI_AGENT_PLAN.md` 外，代码工作树不存在已修改或未跟踪的 i18n 实现文件，代码基线干净。

此前的候选实现从未进入 commit，现已清理。Wave 0 从干净代码基线开展技术设计和 PoC，不存在需要继承或修正的候选实现。

---

## 3. 技术方案原则

### 3.1 候选技术路线

- 候选库：`i18next` + `react-i18next`。
- 默认语言：`en`。
- 中文语言标识：`zh-CN`。
- 英文资源为真源和最终回退资源。
- 字典按业务 namespace 组织。
- 国际化运行时代码统一放在 `src/i18n/**`。
- 翻译资源统一放在 `src/locales/{en,zh-CN}/**`。
- 普通字符串使用 `t(key)`；包含链接、强调或嵌套结构时使用 `<Trans>`。
- 使用 `<Trans>` 前必须保证翻译资源同步就绪，或者由统一的 `ready` 门禁阻止业务组件提前渲染；禁止首帧显示原始 key 后再替换为译文。

最终技术方案必须经过 Baseline Review 和 PoC 验证后由总控智能体批准。

### 3.2 静态导出约束

由于使用 `output: "export"`：

- v1 不依赖 Next.js 服务端 locale 路由。
- v1 不将 httpOnly Cookie 作为前端必需能力。
- 语言初始化和切换主要在客户端完成。
- `<html lang>` 由客户端在语言确定后同步更新。
- `suppressHydrationWarning` 只能抑制 hydration 警告，不能防止首屏语言闪烁。
- 首屏闪烁必须通过 PoC 评估，可选策略包括初始化脚本、语言就绪门禁或接受短暂切换。

### 3.3 语言偏好优先级

建议优先级如下，最终以架构评审结论为准：

```text
用户主动选择
→ 已确认的用户级 UI 设置（如果后端确实支持）
→ 普通 SameSite Cookie 或 localStorage
→ 浏览器语言
→ 英文 en
```

管理员全局 UI 设置是否覆盖用户主动选择，必须作为产品决策单独确认。

Login、SSO 和 OAuth 可能发生整页跳转。PoC 必须验证跳转返回后仍能从同源 Cookie 或 localStorage 恢复语言；`LOCALIZATION_SPEC.md` 必须明确期望行为，测试方案必须覆盖登录成功回跳、SSO 回跳和 MCP OAuth 回跳。

### 3.4 key 命名规范

```text
<namespace>:<domain>.<page>.<element>[.<state>]
```

示例：

- `navigation:group.observability`
- `navigation:item.teams`
- `common:action.delete`
- `auth:login.submit`
- `models:form.name.label`
- `usage:tabs.overview`

规则：

- namespace 同时决定资源文件名和 key 前缀：`<namespace>` 对应 `src/locales/<locale>/<namespace>.json`，组件引用时使用 `<namespace>:<key>`；例如 `common.json` 对应 `common:action.delete`。
- 使用语义 key，不使用完整英文句子作为 key。
- 英文和中文字典必须保持相同 key 集合。
- 插值使用 `{{variable}}`。
- 英文根据 i18next/CLDR 规则处理 `one`、`other` 等复数形式。
- 简体中文通常不创建形态复数分支，优先使用计数插值和明确量词，例如 `{{count}} 个请求`、`已选择 {{count}} 个模型`；禁止机械复制英文复数结构生成无意义的中文分支。
- 计数测试至少覆盖 `count=0`、`count=1`、`count=2`。
- 模型名、API 参数、日志原文、代码示例原则上不翻译。
- 扫描工具只报告候选硬编码，不自动生成最终 key，不自动改写代码。

---

## 4. 多智能体组织结构

逻辑角色可以超过 4 个，但同一时刻最多运行 4 个智能体，其中包含总控智能体。

Agent 0 映射为当前主智能体，本身占用 1 个并发槽位；每个波次最多再启动 3 个执行智能体。不得把 Agent 0 视为槽位外角色后额外启动第 4 个执行智能体。

```text
Agent 0  总控、任务调度与 Review（全程）
├── Agent 1  国际化技术设计
├── Agent 2  产品、交互与本地化设计
├── Agent 3  测试架构与验收设计
├── Agent 4  国际化平台开发
├── Agent 5  导航、登录与公共 UI 开发
├── Agent 6  业务功能开发（按模块轮换）
├── Agent 7  持续测试与质量工程
└── Agent 8  集成与发布验收
```

---

## 5. 智能体职责

### 5.1 Agent 0：`i18n-lead-reviewer`

角色：总体负责人、项目调度、技术负责人、Reviewer。

职责：

1. 维护总体计划、任务板、风险和决策记录。
2. 划分文件及 namespace Ownership。
3. 给每个智能体下发自包含任务单。
4. Review 技术方案、本地化规范、测试方案和代码。
5. 处理跨 Agent 接缝和冲突。
6. 执行阶段门禁，决定通过、修改后通过或驳回。
7. 控制合并顺序和发布范围。
8. 汇总最终 Review 与发布结论。

固定交付物：

- `MASTER_PLAN.md`
- `TASK_BOARD.md`
- `FILE_OWNERSHIP.md`
- `DECISIONS.md`
- `REVIEW_REPORT.md`
- `RELEASE_CHECKLIST.md`

约束：Agent 0 不承担大规模业务开发，避免自己实现、自己 Review。

### 5.2 Agent 1：`i18n-architect`

角色：国际化技术设计智能体。

职责：

- 分析现有前端架构、Provider 模式，并调查 UI Settings `language` 字段的权限、作用域和数据语义。
- 验证 `react-i18next` 与静态导出构建兼容性。
- 设计 Provider、locale 初始化、偏好保存、英文回退和资源加载。
- 设计 `<html lang>` 同步和首屏策略。
- 完成最小 PoC。
- 输出 ADR 和开发接口契约。
- 在平台开发完成后执行 ADR 符合性复核。

交付物：

- `I18N_TECH_DESIGN.md`
- `I18N_ADR.md`
- `POC_REPORT.md`：记录静态导出构建、首屏策略对比、普通 `t()` 与 `<Trans>` 首次渲染、刷新恢复、英文回退、整页跳转恢复及浏览器验证证据。
- 技术风险清单

### 5.3 Agent 2：`localization-designer`

角色：产品、交互与本地化设计智能体。

职责：

- 设计语言切换器位置、状态和交互。
- 定义首次进入和后续访问时的语言规则。
- 明确 Login、SSO、MCP OAuth 整页跳转返回后的语言保持规则。
- 明确构建期 `<title>` 和 metadata description 在 v1 保持英文，不由功能开发智能体被动扩展范围。
- 编写中英文术语表。
- 定义不可翻译内容。
- 定义按钮、表单、错误提示和空状态的中文风格。
- 检查中文长度对导航、表格和弹窗的影响。
- 审核各功能域翻译质量。

交付物：

- `LOCALIZATION_SPEC.md`
- `GLOSSARY_EN_ZH.md`
- `LANGUAGE_SWITCHER_SPEC.md`
- `V1_TRANSLATION_SCOPE.md`
- `LOCALE_NAVIGATION_BEHAVIOR.md`：记录首次访问、手动切换、Login、SSO、MCP OAuth 回跳及存储不可用时的期望与降级行为。

### 5.4 Agent 3：`qa-architect`

角色：测试方案设计智能体。

职责：

- 制订单元、集成和 E2E 测试边界。
- 制定中英文验收用例。
- 设计 key 一致性和硬编码扫描规则。
- 设计中文布局检查矩阵。
- 明确自动化与人工验证范围。
- 为平台和功能开发准备测试接口。

交付物：

- `I18N_TEST_PLAN.md`
- `TEST_CASES.md`
- 自动化测试任务列表
- 回归测试矩阵

### 5.5 Agent 4：`i18n-platform-developer`

角色：国际化基础设施开发智能体。

允许修改：

- `src/i18n/**`：`I18nProvider`、初始化配置、locale 偏好、类型和资源注册；国际化运行时代码不再分散到 `src/contexts` 或 `src/lib`。
- `src/locales/en/**`、`src/locales/zh-CN/**`：仅在 Wave 1 创建目录、namespace 骨架和平台 smoke-test 所需的最小资源；G1 通过后按 namespace 移交对应 Owner。
- `src/app/layout.tsx`。
- `package.json`、`package-lock.json`。
- 平台层对应测试。

职责：

- 根据通过评审的 ADR，从干净代码基线实现国际化基础设施。
- 实现语言初始化、切换、持久化和英文回退。
- 实现 `<html lang>` 同步。
- 保证翻译资源同步就绪，或实现统一的 `ready` 渲染门禁，使普通 `t()` 与 `<Trans>` 首帧行为一致。
- 建立 namespace 和类型约束。
- 完成基础设施测试和静态导出构建验证。
- 提交 `PLATFORM_VALIDATION_REPORT.md`，记录实现相对 ADR 的符合性和实际验证证据。

禁止批量修改业务页面，禁止修改生成文件 `src/lib/http/schema.d.ts`。

### 5.6 Agent 5：`i18n-shell-auth-developer`

角色：导航、登录和公共入口开发智能体。

负责范围：

- Navbar、Leftnav、用户菜单及其他全局壳层。
- Login、Onboarding、Connect、MCP OAuth。
- `src/locales/en/common.json`、`src/locales/zh-CN/common.json`。
- `navigation`、`auth` 及上述页面对应的 namespace 和测试。

职责：

- 按本地化规范改造用户可见文案。
- 同步处理 `aria-label`、`title`、placeholder 和校验提示。
- 补充相关单元或集成测试。
- 检查中文布局。

### 5.7 Agent 6：`i18n-feature-developer`

角色：业务功能开发智能体，按波次领取单一功能域。

v1 功能批次：

1. Models and Endpoints、API Keys。
2. Usage、Cost Tracking、Budgets。

v2+ 候选批次（不属于 v1 范围，不进入 v1 完成定义）：

3. Teams、Users、Organizations、Projects。
4. Guardrails、Policies、Logs。
5. Playground、Prompts、Agents、Skills。
6. MCP Servers、Caching、Vector Stores 等其他页面。

每次任务必须列出明确目录和文件清单，不允许使用“其余未归属目录”。

### 5.8 Agent 7：`i18n-qa`

角色：持续测试和质量工程智能体。

职责：

- 从平台开发阶段开始持续测试。
- 实现 key 集合一致性和硬编码报告工具。
- 运行改动相关的单元及集成测试。
- 验证切换、刷新、回退、插值和 `<html lang>`。
- 验证中文布局和可访问性文案。
- 提交缺陷并退回原代码 Owner。
- 不直接大规模修改产品代码。

### 5.9 Agent 8：`i18n-integration-release`

角色：集成和发布验收智能体。

职责：

- 执行跨模块回归。
- 汇总测试证据和遗留问题。
- 验证 `lint`、格式和静态导出构建。
- 准备发布与回滚清单。
- 不绕过原 Owner 修改大范围业务代码。

---

## 6. 文件和字典 Ownership

Agent 0 在任务开始前维护完整的 `FILE_OWNERSHIP.md`。

基本分配：

| 范围 | Owner |
|---|---|
| `src/i18n/**`、Provider、locale 工具 | Agent 4 |
| `src/locales/{en,zh-CN}/**` 目录及最小骨架 | Wave 1 为 Agent 4；G1 后按 namespace 移交 |
| `src/locales/{en,zh-CN}/common.json` 及公共 UI 文案 | G1 后为 Agent 5；Agent 2 审核译文，新增通用 key 需 Agent 0 Review |
| `navigation` namespace及导航组件 | Agent 5 |
| `auth` namespace及登录引导组件 | Agent 5 |
| `models`、`apiKeys` namespace及页面 | 当期 Agent 6 |
| `usage`、`cost`、`budgets` namespace及页面 | 当期 Agent 6 |
| 测试公共工具和质量报告 | Agent 7 |
| 发布检查和汇总报告 | Agent 8 |

规则：

1. 同一文件同一时间只有一个 Owner。
2. 功能 Agent 只写自己的 namespace，不共同编辑一个大型 JSON。
3. `src/components/ui/**` 原则上保持无业务文案，由调用方传入文本。
4. `src/utils/**` 不设置单一全目录 Owner，按具体文件划分。
5. 修改非本人文件前必须向 Agent 0 提交变更请求。
6. QA 发现产品缺陷后退回原 Owner，不直接全局修复。
7. Agent 0 决定公共文件修改人和合并顺序。
8. Agent 4 在 Wave 1 只创建字典目录、namespace 注册和 smoke-test 所需最小资源，不批量填写业务文案。
9. G1 通过后，Agent 0 在 `FILE_OWNERSHIP.md` 中记录 `common`、`navigation`、`auth` 等资源文件向 Agent 5 的正式移交。
10. `src/i18n/**` 中的 namespace 注册表、资源加载映射和类型声明始终归 Agent 4；Agent 5/6 只维护获授权的 JSON 资源，不得自行修改注册代码。
11. 新增业务 namespace 时，功能 Agent 在任务单中填写 namespace、资源文件和注册需求，由 Agent 0 在阶段门禁期指派 Agent 4 统一注册；特殊情况下必须由 Agent 0 书面授权临时 Owner。

### 6.1 并行工作区与分支策略

为了保证并行开发的改动可隔离、可 Review、可回退，开发智能体不得直接在同一个工作区并发修改代码。

1. Agent 0 维护主集成工作区和集成分支。
2. Agent 4、Agent 5、每个 Agent 6 任务实例和 Agent 7 使用独立 Git worktree 与独立任务分支。
3. 分支命名建议：`i18n/<wave>-<agent>-<scope>`，例如 `i18n/w2-agent6-models`。
4. 每个任务只提交其 Ownership 范围内的文件，不夹带无关格式化或其他 Agent 的改动。
5. Agent 完成任务后提交 commit hash、diff 摘要、测试证据和遗留问题。
6. Agent 0 针对独立变更集执行 Review；`CHANGES_REQUESTED` 由原 Agent 在原 worktree 修复。
7. Review 通过后，Agent 0 按“平台层 → 公共层 → 功能层 → 测试层”顺序合并。
8. 合并出现冲突时，由 Agent 0 指定唯一 Owner 处理，禁止多个 Agent 同时修复同一冲突。
9. 每次阶段门禁通过后记录集成基线 commit，作为下一波工作的共同起点。
10. `package.json`、`package-lock.json` 默认只允许 Agent 4 在平台 worktree 修改；其他 Agent 不得执行会改写依赖声明或锁文件的操作。后续确需新增依赖时，由 Agent 0 指定唯一临时 Owner。
11. Agent 4 的依赖与平台变更通过 G1 后必须先合入集成基线；Agent 5/6/7 的开发 worktree 从该基线创建，或在开始写代码前 rebase 到该基线。
12. 下游 worktree 使用 `npm ci` 按已批准的锁文件安装依赖，不使用 `npm install` 生成各自的锁文件改动。
13. npm 下载缓存可以共享，但每个 worktree 默认使用独立 `node_modules`；禁止多个 worktree 直接共享或并发修改同一个 `node_modules`。
14. Agent 7 如需在 G1 前验证平台候选提交，应从 Agent 4 的明确 commit 创建临时验证 worktree，且不得修改 `package.json` 或 `package-lock.json`。

### 6.2 v1 范围状态清单

Agent 2 建立 `V1_SCOPE_MANIFEST.md`，Agent 0 维护状态，Agent 7 填写测试和 UI 检查证据。

| 路由/组件 | Owner | 文案盘点 | EN | ZH-CN | 测试 | UI 检查 | Review |
|---|---|---|---|---|---|---|---|
| Navbar / Leftnav | Agent 5 | 待完成 | 待完成 | 待完成 | 待执行 | 待执行 | 待审核 |
| Login / Onboarding | Agent 5 | 待完成 | 待完成 | 待完成 | 待执行 | 待执行 | 待审核 |
| Connect / MCP OAuth | Agent 5 | 待完成 | 待完成 | 待完成 | 待执行 | 待执行 | 待审核 |
| Models and Endpoints | Agent 6 | 待完成 | 待完成 | 待完成 | 待执行 | 待执行 | 待审核 |
| API Keys | Agent 6 | 待完成 | 待完成 | 待完成 | 待执行 | 待执行 | 待审核 |
| Usage / Cost Tracking | Agent 6A | 待完成 | 待完成 | 待完成 | 待执行 | 待执行 | 待审核 |
| Budgets | Agent 6B | 待完成 | 待完成 | 待完成 | 待执行 | 待执行 | 待审核 |

G2、G3 必须以对应范围的清单项全部完成作为通过条件，不以“Agent 已报告完成”代替验收证据。

---

## 7. 执行波次与阶段门禁

### Wave 0：Baseline Review 与并行设计

同时运行：

- Agent 0：建立计划、任务板和 Ownership。
- Agent 1：核查干净代码基线，分析现有工程模式并完成技术设计与最小 PoC。
- Agent 2：完成语言交互、术语和翻译规范。
- Agent 3：完成测试策略和验收用例。

门禁 G0：

- 除本方案文档外，代码工作树无本地修改且不残留候选 i18n 实现。
- 技术 ADR 已通过 Review。
- `POC_REPORT.md` 与 `LOCALE_NAVIGATION_BEHAVIOR.md` 已提交并通过 Review。
- `zh-CN`、语言优先级、偏好存储和首屏策略已决策。
- 构建期 `<title>` 和 metadata description 在 v1 保持英文的范围边界已记录。
- Login、SSO、MCP OAuth 整页跳转后的语言保持规则已写入本地化与测试方案。
- v1 页面和翻译范围已冻结。
- 测试策略已通过 Review。

### Wave 1：平台能力与测试基础

同时运行：

- Agent 0：Review 和协调。
- Agent 4：根据通过评审的 ADR，从零搭建平台实现。
- Agent 7：基于 Agent 4 的明确候选 commit 实现或执行平台测试和静态检查，不修改依赖声明及锁文件。
- Agent 5：只读盘点导航、登录文案，在任务交付物中拟定 key；G1 前不创建或修改源码及字典文件。

门禁 G1：

- Provider、语言切换和偏好保存可用。
- 英文回退有效。
- `<html lang>` 同步正确。
- 首屏策略已从“初始化脚本、语言就绪门禁、接受短暂切换”等候选项中明确选定，写入 ADR，并附构建及浏览器验证证据。
- Login、SSO、MCP OAuth 整页跳转后的语言恢复 PoC 已通过，或已形成经 Agent 0 批准的明确限制与降级方案。
- 普通 `t()` 与 `<Trans>` 均在资源就绪后渲染，首次渲染不暴露原始 key。
- 定向测试和 `npm run build` 通过。
- `PLATFORM_VALIDATION_REPORT.md` 已提交。
- Agent 1 完成 ADR 符合性复核。
- Agent 0 完成代码 Review。

### Wave 2：第一批功能并行开发

同时运行：

- Agent 0：Review 和冲突处理。
- Agent 5：导航、Login、Onboarding、Connect、MCP OAuth。
- Agent 6：Models and Endpoints、API Keys。
- Agent 7：持续测试已经提交的模块。

门禁 G2：

- 全局壳层和核心入口支持中英文。
- Models、API Keys 主流程支持中英文。
- `V1_SCOPE_MANIFEST.md` 中本波次对应项目全部完成。
- 相关测试通过。
- 中文布局不存在阻塞性问题。
- Agent 0 Review 通过。

### Wave 3：第二批功能并行开发

同时运行：

- Agent 0：Review 和任务调度。
- Agent 6A：Usage、Cost Tracking。
- Agent 6B：Budgets，以及 Budgets 范围内的测试和中文布局检查。
- Agent 7：持续测试与缺陷回归。

Agent 6A、6B 是同一角色的两个任务实例，必须拥有完全不相交的文件列表。

门禁 G3：

- v1 业务范围全部完成。
- `V1_SCOPE_MANIFEST.md` 全部项目具有测试、UI 检查和 Review 证据。
- 中英文 key 集一致。
- 无 P0、P1 缺陷。
- 定向测试、lint、格式和构建通过。

### Wave 4A：集成与发布验收

同时运行：

- Agent 0：最终 Review 和发布决策。
- Agent 2：术语与中文体验复核。
- Agent 7：完整回归和质量报告。
- Agent 8：发布检查、证据汇总和回滚清单。

如果 Wave 4A 发现需要修改产品代码的缺陷，暂停 Agent 2、Agent 8，并进入 Wave 4B。

### Wave 4B：定向缺陷修复（按需启动）

同时运行：

- Agent 0：判定缺陷 Owner、优先级并 Review 修复。
- 原开发 Owner 1：修复自身范围内缺陷。
- 原开发 Owner 2：修复另一个不重叠范围内缺陷；没有第二组缺陷时不启动。
- Agent 7：定向回归并更新测试报告。

修复通过后恢复 Wave 4A，由 Agent 2、Agent 8 完成最终体验和发布复核。

缺陷进入 `TASK_BOARD.md` 后按以下顺序排队：P0 → P1 → 阻塞同一门禁的 P2 → 其他 P2。每轮最多启动两个文件范围不重叠的开发 Owner；第三组及后续缺陷保持排队，由 Agent 0 在上一组完成 Review 后调度下一组。

门禁 G4：

- v1 完成定义全部满足。
- P0、P1 缺陷清零。
- P2 缺陷具有明确处置结论。
- 发布和回滚清单完成。
- Agent 0 给出 `APPROVED` 结论。

---

## 8. 标准任务单

Agent 0 下发的每个任务必须包含：

```markdown
## 目标
任务需要实现的可验证结果。

## 输入与依赖
依赖的 ADR、规范、接口和前置提交。

## 文件范围
允许修改的精确目录和文件。

## Namespace 变更
是否新增 namespace、资源文件路径、是否需要修改注册表、注册代码 Owner。

## 禁止范围
明确禁止修改的文件和行为。

## 交付物
代码、字典、测试和报告。

## 验收标准
功能结果、测试文件和检查命令。

## 风险与接缝
可能与其他任务冲突的地方。

## 回报格式
worktree/分支、commit hash、改动文件、翻译 key、测试结果、遗留问题和待 Review 决策。
```

进入 Review 时必须提供：

- worktree 路径和任务分支名称。
- commit hash 或边界明确的 diff。
- 修改文件列表。
- 新增、修改和删除的翻译 key。
- 实际执行的测试及工程检查命令。
- 测试结果和失败信息。
- 未执行的检查及原因。
- 已知限制和跨 Agent 接缝。

---

## 9. Review 与缺陷回流机制

Agent 0 对每个开发任务执行两轮 Review。

### 9.1 设计符合性 Review

- 是否符合 ADR 和语言优先级。
- 是否遵守术语表和翻译范围。
- 是否遵守 namespace 与文件 Ownership。
- 是否误翻译模型名、API 参数、日志或代码示例。

### 9.2 代码与质量 Review

- 是否正确处理动态插值和复数。
- 是否具有英文回退。
- 是否覆盖可访问性文案。
- 是否增加匹配风险的测试。
- 是否改变原有业务行为。
- 是否出现无意义的大范围格式化。
- 是否具有可复现的测试证据。

Review 结论：

- `APPROVED`
- `CHANGES_REQUESTED`
- `BLOCKED`

未经 `APPROVED` 的任务不得进入集成基线。

QA 发现缺陷后的流程：

```text
QA 提交复现步骤和证据
→ Agent 0 判定 Owner 与优先级
→ 原开发 Owner 修复
→ QA 定向回归
→ Agent 0 Review 并关闭
```

---

## 10. 测试与工程门禁

测试重点：

- 默认语言正确。
- 切换语言即时生效。
- 刷新后语言保持。
- 缺失中文时回退英文。
- 不显示原始翻译 key。
- `<html lang>` 与当前语言一致。
- 中文按钮、菜单、表格和弹窗不溢出。
- 切换语言不丢失表单状态。
- Login、SSO、MCP OAuth 整页跳转并返回后语言偏好保持。
- 动态数量、日期、数字和货币正确本地化。
- API 数据、模型名、日志和代码示例不被误翻译。
- 构建期 `<title>` 与 metadata description 在 v1 保持英文，且没有被功能 Agent 意外改写。

工程检查：

```bash
cd ui/litellm-dashboard
npm run lint
npm run format:check
npm run build
```

补充规则：

- Vitest 必须指定受影响的测试文件，不执行无路径的完整测试集。
- 测试命令必须由 Agent 3 根据项目 Vitest 配置确认，并在任务回报中记录准确命令和结果。
- 只有修改后端路由或响应模型时才运行 `npm run gen:api`。
- 若清理了已有 ESLint suppression，使用项目规定的 `eslint . --prune-suppressions` 并检查差异。
- 不假设或新增未经项目确认的 `eslint-budgets.json`。

---

## 11. v1 完成定义

v1 只有同时满足以下条件才算完成：

1. 支持 `en` 与 `zh-CN`。
2. 语言切换即时生效，刷新后偏好保留。
3. 导航、Login、Onboarding、Connect、MCP OAuth、Models、API Keys、Usage、Cost Tracking 和 Budgets 完成中文化。
4. 中文缺失时回退英文，不直接显示翻译 key。
5. 中文界面无严重截断、重叠和遮挡。
6. 切换语言不清空表单或触发异常请求。
7. Login、SSO、MCP OAuth 整页跳转并返回后语言偏好保持，或存在经批准且已记录的限制与降级方案。
8. 构建期 `<title>` 与 metadata description 保持英文，不在 v1 中产生半完成的多语言 SEO 行为。
9. 模型名、API 字段、日志和代码示例未被错误翻译。
10. 改动相关的单元及集成测试通过。
11. `lint`、`format:check` 和静态导出构建通过。
12. Agent 7 提交测试报告。
13. Agent 2 完成术语与中文体验复核。
14. Agent 0 完成最终 Review 并给出 `APPROVED`。
15. `V1_SCOPE_MANIFEST.md` 全部项目状态为完成，并附有测试、UI 检查和 Review 证据。

---

## 12. 下一步

1. 启动 Wave 0，由 Agent 0 建立 `MASTER_PLAN.md`、`TASK_BOARD.md`、`FILE_OWNERSHIP.md` 和 `DECISIONS.md`。
2. Agent 1 从干净代码基线完成技术设计、PoC 与 ADR，并验证静态导出构建。
3. Agent 2、Agent 3 并行完成本地化规范、范围清单和测试方案。
4. Agent 0 Review Wave 0 全部交付物并执行 G0 门禁。
5. G0 通过后，为 Agent 4、Agent 5、Agent 7 创建独立任务 worktree 和分支，启动 Wave 1。
6. 后续严格按照“1 个总控 + 3 个执行智能体”的并发限制滚动执行。
