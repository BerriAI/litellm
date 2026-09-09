# LiteLLM Dashboard 本地化规范（Localization Spec）

> 维护者：Agent 2（localization-designer）
> 状态：设计稿，待 G0 评审
> 范围：LiteLLM Dashboard 前端 UI 的 `en` / `zh-CN` 本地化
> 关联：`I18N_MULTI_AGENT_PLAN.md`（§3.3/§3.4/§5.2）、`DECISIONS.md`、`LANGUAGE_SWITCHER_SPEC.md`、`GLOSSARY_EN_ZH.md`

本文档定义语言的**交互规则**与**中文文案风格标准**。它回答"什么时候用什么语言"以及"中文应该怎么写得专业"。具体每个页面/组件翻不翻、翻哪些，见 `V1_TRANSLATION_SCOPE.md`；语言切换器的 UI 细节见 `LANGUAGE_SWITCHER_SPEC.md`。

---

## 1. 语言规则

### 1.1 语言标识与优先级

- 语言代码：`en`（默认）、`zh-CN`。
- `en` 是**真源与最终回退**；任何缺少 `zh-CN` key 的字符串一律回退英文，绝不允许显示原始 key。
- 语言优先级（D5，已定）：
  ```
  用户主动选择
  → 已确认的用户级 UI 设置（如后端支持，v1 默认不启用）
  → cookie + localStorage
  → 浏览器语言
  → en
  ```
- 语言偏好判定完全在客户端完成（静态导出约束，D4）。

### 1.2 首次进入

- 无任何已存偏好时：读取浏览器语言。若 `navigator.language` 以 `zh` 开头（`zh`、`zh-CN`、`zh-TW`、`zh-Hans` 等均视为中文系），则初始为 `zh-CN`；否则为 `en`。
- 首次进入**不强制弹出语言选择对话框**。以浏览器语言为默认即可；用户可随时通过切换器更改。
- 首次进入后（即一旦存在已存偏好），优先使用已存偏好，不再读取浏览器语言。

### 1.3 手动切换

- 切换**即时生效**，无需刷新、无需整页跳转；当前页面所有文案立即切换。
- 切换时**保留表单状态、滚动位置、导航展开状态**，不允许因语言切换重置表单或触发异常请求（v1 完成定义第 6 条）。
- 切换后立即写入 cookie + localStorage（行为见 `LANGUAGE_SWITCHER_SPEC.md`）。

### 1.4 整页跳转与回跳（Login/SSO/MCP OAuth）

详细行为与降级见 `LOCALE_NAVIGATION_BEHAVIOR.md`。摘要：

- Login 提交、SSO 登录、MCP OAuth 授权可能发生整页跳转（起跳自同源页面，回跳到同源页面）。
- 跳转返回后必须从**同源 cookie 或 localStorage** 恢复语言，语言不应回落到英文。
- 若同源存储不可用（隐私模式、禁 cookie），允许回退英文，但不允许出现语言"闪烁"到错误语言的体验缺陷。

### 1.5 构建期元数据

- v1 构建期 `<title>` 与 metadata description **保持英文**（D8），不做多语言 SEO。功能 Agent 不得被动扩大此范围。

---

## 2. 中文文案风格规范

目标：**专业、自然、克制、无翻译腔**。面向开发者/运维人员（Dashboard 用户以工程师为主），用词应准确、简洁，避免口语化和过度文学化。

### 2.1 通用原则

1. **以用户动作和对象为先**。按钮用"动词 + 宾语"最短形式（见 2.2）。
2. **不用翻译腔**：避免"为了……我们将会……"、"请注意"、"请务必"等冗余；避免英文逐字直译的语序污染。
3. **控制长度**：中文比英文短，但信息密度高。菜单项/按钮 ≤ 6 字为佳，超过 8 字注意换行与截断（见 §3）。
4. **术语一致**：一律采用 `GLOSSARY_EN_ZH.md` 中的词条，禁止同义词漂移（如"Spend"一会翻"花费"一会翻"支出"）。
5. **敬语从简**：界面动作祈使句即可，不加"您"的过度堆叠；保留必要礼貌但不每一句都加。
6. **标点**：中文使用全角标点（`，`、`。`、`：`、`？`），英文/数字/型号与中文混排时两侧不加空格；数字、单位用半角（`10 次` 中空格与否按项目 UI 惯例统一）。
7. **占位符与插值**：不译 `{{variable}}`、`{{count}}` 内的变量名；品牌名 LiteLLM、OpenAI 等不译。

### 2.2 按钮文案

- 格式：**动词（+ 宾语）**，祈使句。
- 通用按钮术语（复用 `GLOSSARY_EN_ZH.md` + `common.json`）：
  | EN | ZH-CN |
  |---|---|
  | Save | 保存 |
  | Cancel | 取消 |
  | Delete | 删除 |
  | Create | 创建 |
  | Add | 添加 |
  | Edit | 编辑 |
  | Update | 更新 |
  | Enable / Disable | 启用 / 停用 |
  | New Virtual Key | 新建虚拟密钥 |
  | Generate | 生成 |
  | Copy | 复制 |
  | Close | 关闭 |
  | Confirm | 确认 |
  - 含受保护动作（删除、重置、清空）的按钮在确认弹窗中保持同一译名，不要换成同义"移除/清除"造成语义漂移。

### 2.3 表单文案

- **Label**：名词性短语，简洁。如 `Deployment Name` → `部署名称`、`Base URL` → `基础 URL`（URL 保留英文，见 §4）。
- **Placeholder**：用"请输入 / 请选择 / 例如"引导的示例或提示。如 `Enter model ID` → `输入模型 ID`；`Select team` → `选择团队`。
- **校验提示**：指出问题 + 期望，避免命令式指责。
  - 示例：`This field is required` → `此项为必填`；`Key must be at least 8 characters` → `密钥至少需要 8 个字符`。
- **可选/必填**：`Required` → `必填`；可选字段不要写"可选"二字堆满表单，仅在视觉区分或用 `（可选）` 后缀在 label 上。
- **分段标题/说明**：`form.helper` 类说明文本可译，但保留关键英文专有名词（API、URL、SSO）。

### 2.4 错误提示

- **技术性错误**（后端返回、非 4xx 校验错误）：尽量提供"发生了什么 + 可做什么"两层。
- **不可直翻后端消息**：后端英语错误原文**不翻译**（见 §4）。即使附带中文说明，也应保留原文以便排障。
- 措辞平实。示例：
  - `Unable to load models.` → `无法加载模型。`
  - `Your session has expired.` → `会话已过期。`
  - `Failed to create Virtual Key.` → `创建虚拟密钥失败。`
  - `Insufficient permission.` → `权限不足。`（不写"您的权限不够"）
- 避免把错误写成"出错了"这类无信息量提示；能带对象就带对象（`删除团队失败`）。

### 2.5 空状态（Empty State）

- 空状态 = 说明 + 一句引导（一个主导 CTA）。
- 避免"这里空空如也"这种口语；用客观陈述 + 引导。
  - 示例：`No Virtual Keys yet. Create one to get started.` → `暂无虚拟密钥。点击"新建虚拟密钥"开始使用。`
  - `No usage data for the selected period.` → `所选时间段内暂无用量数据。`
- 引导动词与真实按钮文字保持一致（"新建虚拟密钥"）。

### 2.6 成功提示 / 通知（Toast）

- 过去式主谓短句。如 `Virtual Key created.` → `虚拟密钥已创建。`、`Settings saved.` → `设置已保存。`
- 需要引用对象名称/ID 时用插值：`Model "{{name}}" saved.` → `模型"{{name}}"已保存。`

---

## 3. 中文长度对布局的影响与处理

中文单字约占英文一个字符 ~2 倍视觉宽度，但信息密度高，实际渲染宽度通常**更短或相当**。真正的风险是：

1. 英文长复合词被较长的中文词组替换；
2. 筛选器、表格列头、弹窗标题空间不足导致换行/溢出。

### 3.1 导航（Leftnav / Navbar / 面包屑）

- 菜单项普遍 ≤ 6 字（如 `Virtual Keys`→`虚拟密钥`、`Models + Endpoints`→`模型与端点`），通常放得下。
- 侧边栏已具备 `truncate`（`flex-1 truncate`），中文过长时应保持单行截断，并补 `title` 展示全称。
- 分组标签（AI GATEWAY 等）v1 也翻译；groupLabel 是配置常量，需随 `menuGroups` 一起 I18N 化，而非仅改渲染层。
- 面包屑 section/title 同样取翻译，需与导航使用同一套 key，避免两处不一致。

### 3.2 表格

- 列头名词、数值列（用量/成本/花费用数字格式化，见 `V1_TRANSLATION_SCOPE.md` 插值注意事项）。
- 表格单元格回退策略：**列头可换行，单元格优先截断 + `title` 全称**；避免单元格内文字撑爆列宽。
- 数据列（模型 ID、团队名、时间戳）」多为数据，不翻译，按 §4 处理。

### 3.3 弹窗 / 抽屉（Modal / Drawer）

- 标题 ≤ 20 字；超长标题强制换行并提供 `line-clamp` 或在布置上预留两行。
- 按钮组"取消/保存"不加长词缀，保持最短形式。
- 弹窗内说明文本允许自然换行，但禁止水平溢出。

### 3.4 全体回退策略（截断 / 换行 / 省略号）

| 场景 | 策略 |
|---|---|
| 导航菜单单行 | `truncate`（省略号） + `title` 全称 |
| 表格单元格 | 截断 + `title`；数值列右对齐 |
| 弹窗标题 | 允许换行（白空间关键字 `[normal, pre-wrap]`），必要时 `line-clamp-2` |
| 按钮 | 不换行；过长时缩短译名，不拼接 |
| 标签/Badge（Beta、New） | 不翻译品牌徽标；`v{major.minor}` 保持 |

---

## 4. 不可翻译内容清单

以下内容**明确不翻译**（作为资源值原样保留或直接放数据字段，不进翻译 key）。Agent 5/6 必须遵守，Review 时逐条核查：

1. **模型名 / Model ID / Deployment 名 / Provider 名**：`gpt-4o`、`claude-3-5-sonnet`、`openai`、`bedrock` 等；用户自定义的 Deployment/模型名也是数据，不译。
2. **API 字段名 / 参数名**：`key`、`model`、`spend`、`tpm_limit`、`rpm_limit`、`metadata`、`created_at`；接口返回的键与 JSON 结构不译。
3. **请求/响应体、日志原文**：后端返回的原始日志、错误消息、堆栈（stack trace）、trace 内容保持原样。
4. **代码示例与终端命令**：curl 片段、Python/JS 代码、API 请求示例不译；但代码**上方/下方的说明文字**可译。
5. **URL、路径、文件名**：`https://docs.litellm.ai`、`/api/`、`config.yaml`、`.env` 等。
6. **广泛公认的英文技术缩写**：API、URL、SSO、MCP、OAuth、REST、HTTP、SQL、ID、JSON、SDK。
7. **品牌名**：LiteLLM、OpenAI、Anthropic、Virtual Key（作为产品专名，见下表备注）、Guardrails 的专有规则名。
8. **数值、日期、货币、单位**：数字格式化走 i18n 的 number/date/currency，不手工拼翻译；`{{count}}` 由计数插值负责。
9. **角色名 / 权限标识**：`admin`、`internal_user`、`team_admin` 等代码值保持英文（仅其人类可读显示名按术语表翻译）。
10. **徽标与 Beta/New 标记**：`Beta`、`New` 徽标不译（视觉因子，保持英文原样）。

> 原则：**翻译的是界面文案（UI string），不是数据（data string）与协议内容（protocol content）**。无法安全区分时，优先看它是否来自后端/用户数据 —— 是则不译。

---

## 5. zh-CN 计数与量词策略

遵循 D9：**不建立形态复数分支**，用"计数插值 + 量词"。i18next 复数字符串里 `_one/_other` 仅用于英文；`zh-CN` 用一个字符串承载 `{{count}}` + 量词即可。

量词选取规则：

- 可数离散对象：`个`（模型、密钥、请求、团队、用户、条目）。
- 键值/集合类（如导入项）：可用 `个` 或直接按对象数合身处理。
- 时间/次数：`次`（请求次数、重试次数）。
- 金额/用量单位化：不写量词，直接 `{{count}}` + 单位（`{{count}} 请求`、`{{count}} 个请求`，口语一致性以术语表为准）。
- `0` 时量词仍正常（`0 个请求` 在中文是自然的，不需要单独复数分支）。

### 5.1 具体示例

| 场景 | EN（_one/_other） | ZH-CN（单字符串） |
|---|---|---|
| 请求数 | `{{count}} request` / `{{count}} requests` | `{{count}} 个请求` |
| 选中模型 | `{{count}} model selected` / selected | `已选择 {{count}} 个模型` |
| 待办项 | `{{count}} item` / items | `{{count}} 个条目` |
| 无选中 | — | `已选择 0 个模型`（中文自然，无需分支） |
| 重试 | `Retry ({{count}})` | `重试（{{count}} 次）` |

### 5.2 测试要求

计数测试至少覆盖 `count=0`、`count=1`、`count=2`（方案 §3.4）。对 `zh-CN` 断言渲染结果等于"插值 + 量词"形态（如 `1 个请求`、`2 个请求`），**不**断言存在 `_one/_other` 分支。

---

## 6. 英文原句 → 规范中文译文示例

给开发 Agent 5/6 的参照样本（来自 v1 高频界面）。

| # | EN（原句） | ZH-CN（规范译文） | 说明 |
|---|---|---|---|
| 1 | `Create new virtual key` | 新建虚拟密钥 | 动词+宾语，最短祈使形式 |
| 2 | `No Virtual Keys found. Click "New Virtual Key" to create your first one.` | 暂无虚拟密钥。点击"新建虚拟密钥"创建你的第一个密钥。 | 空状态客观陈述 + 引导 CTA，CTA 文字与真实按钮一致 |
| 3 | `This will permanently delete this model and cannot be undone.` | 此操作将永久删除该模型，且无法撤销。 | 机构条理清晰，动作对象明确 |
| 4 | `You've spent $12.34 on Usage this month.` | 本月用量已花费 $12.34。 | 过去式主谓短句，接数值/货币（货币由 i18n 格式化） |
| 5 | `Drop any model to see how it compares.` | 拖入任意模型查看对比结果。 | 祈使引导，无翻译腔 |
| 6 | `Your changes have not been saved.` | 更改尚未保存。 | 客观陈述，不指责 |

> 译名对照一律以 `GLOSSARY_EN_ZH.md` 为准，本示例中的词汇（虚拟密钥、用量、花费、拖入、对比）与术语表保持一致。

---

## 7. 可访问性（a11y）文案规范

- 所有 `aria-label`、`title`、`alt`、placeholder 中的用户可见文案**同样翻译**（方案 §5.5），与可见文本用同一 key。
  - 例：当前 `aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}` → `展开侧边栏` / `收起侧边栏`。
- 语言切换器自身提供 `aria-label`（见 `LANGUAGE_SWITCHER_SPEC.md`）。
- 图标按钮（无可见文字）必须本地化 `aria-label`；纯装饰图标保持 `aria-hidden`。
- 保留英文专有名词的缩写不影响屏幕阅读器核心语义。

---

## 8. 对开发 Agent 的强制要求（Agent 5/6）

1. **只翻 UI 文案**，不翻 §4 清单内容（模型名、API 字段、日志、代码、URL、缩写、品牌）。
2. **key 与字典一致性**：中英文 key 集合必须一致；缺 zh-CN 时回退 en，不回退失败显示 key。
3. **写 key 不写句子到代码**：用语义 key（`navigation:item.virtualKeys`），不把英文句子当 key。
4. `<Trans>` 用于含链接/强调/嵌套，`t()` 用于纯字符串；使用 `<Trans>` 前确保资源就绪（ready 门禁，避免首帧闪 key）。
5. **同一译名贯穿全局**：按钮、空状态、确认框、Toast 对同一动作/对象用词必须一致（对照术语表）。
6. **数值/日期/货币用 i18n 格式化**，不硬编码。
7. **翻译 a11y 文案**（aria-label/title/placeholder）。
8. **不改变业务行为**：翻译不改路由、不改校验逻辑、不改提交逻辑，仅替换展示文案。
9. 交付时在 `V1_SCOPE_MANIFEST.md`（Agent 0 维护）对应行填 EN/ZH-CN 文案盘点；译名异常情况上报 Agent 2 复核。

---

## 9. 术语表引用

术语定义与单一词条见 `GLOSSARY_EN_ZH.md`。若本规范内个别短语与术语表冲突，以 `GLOSSARY_EN_ZH.md` 为准，并提请 Agent 2 修订。
