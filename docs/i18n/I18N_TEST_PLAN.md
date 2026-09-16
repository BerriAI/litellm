# i18n 测试总体策略（I18N_TEST_PLAN）

> 角色：Agent 3 `qa-architect`（测试架构与验收设计）
> 基线：`I18N_MULTI_AGENT_PLAN.md` §10 测试与工程门禁、§11 完成定义
> 实施：Agent 7 `i18n-qa` 在 Wave 1+ 按本文档落地
> 状态：G0 待评审

## 0. 目标与边界

本文件定义 LiteLLM Dashboard `en` / `zh-CN` 国际化的测试总体策略：三层测试边界、key 一致性与硬编码扫描、中文布局检查、可访问性文案覆盖，以及自动化与人工程度的边界。

**纯设计输入**：本文件与配套 `TEST_CASES.md`、`TEST_TOOLS.md`、`REGRESSION_MATRIX.md` 只规定"测什么、怎么测、如何验收"，不修改任何产品代码，不实际执行测试。实际执行由 Agent 7 在 Wave 1+ 负责。

## 1. 三层测试边界（与项目现有 tier 对齐）

### 1.1 事实校准

项目 `vitest.config.ts` 定义 4 个 vitest project：

| vitest project | 环境 | include | 对应"概念层" |
|---|---|---|---|
| `unit` | node | `src/**/*.test.ts`, `tests/**/*.test.ts`（非渲染） | 逻辑单元（无 DOM） |
| `component` | jsdom | `src/**/*.test.tsx`, `tests/**/*.test.tsx`（非 `*.integration.test.tsx`） | 组件单元（毫秒级） |
| `integration` | jsdom | `*.integration.test.tsx`（含 `tests/**`） | 集成（秒级） |
| `types` | tsc | `*.test-d.ts` / `*.test-d.tsx` | 类型测试（`npm run test:types`） |

CLAUDE.md 提到的三档（unit / integration / E2E）与 vitest 的 `unit`+`component`+`integration` 需在测试文档中统一术语：**本文档"单元层"泛指 vitest `component` project 的 jsdom 单组件测试（`*.test.tsx`），"逻辑单元层"特指 `unit` project 的 node 逻辑测试（`*.test.ts`）**。E2E 为 Playwright（见 §1.4）。

工程命令（方案 §10）：

```bash
cd ui/litellm-dashboard
npm run lint           # eslint .
npm run format:check   # prettier 校验
npm run build          # next build（静态导出）
npm run test:types     # vitest run --project types
```

Vitest 定向执行：只跑改动相关文件；单元/组件层毫秒级、集成秒级；禁止无路径全量 `vitest run`。

### 1.2 单元层（component project，`*.test.tsx`）

i18n 场景中，单元层只负责"单一模块、单一微行为"，依赖以 double 替换。

i18n 场景下的单元层**边界**：
- **locale 工具函数**（纯函数，放 `unit` project 的 `*.test.ts`）：语言选择合并逻辑（优先级合并）、`<html lang>` 计算、浏览器语言映射、首选项写入 cookie/localStorage 的序列化与解析。
- **插值/复数/格式化纯函数**（`*.test.ts`）：`count=0/1/2`、日期、数字、货币的格式化结果，无 DOM。
- **单个 UI 小组件**（`*.test.tsx`）：语言切换器按钮、单个翻译标注组件（占位）、量词插值组件，使用 i18next 的 `initReactI18next` 测试实例注入最小字典（en+zh），断言渲染文本。**不渲染整页、不跨组件**。
- 计数器本地化组件（D9）：`count=0` → `0 个`，`count=1` → `1 个`，`count=2` → `2 个`。

单元层不做：整页断言、真实网络、`<html>` 全局状态、路由跳转。

### 1.3 集成层（`*.integration.test.tsx`）

集成层渲染**真实组件树**，仅 stub 网络边界，验证"单元测试无法触达的接线"。

i18n 场景下的集成层**边界**：
- 根应用/页面级内容：`I18nProvider` 包裹下，Navbar / Leftnav / Login / Models / Usage 等页面在切换语言后**即时**更新（真实组件树 + Provider 链）。
- 语言切换不丢失表单状态：在集成树中填写字段 → 切换语言 → 断言字段值仍在、未触发无关请求。
- 动态插值与复数在组件级布线正确（由单元层证明逻辑，集成层证明"字段→key→渲染"链路）。
- `<Trans>` 组件在资源就绪后渲染（不先出原始 key）。

集成层不做：真实浏览器、真实 `document.title` 持久化之外的行为、整页导航刷新行为。

**命名**：`Foo.integration.test.tsx`，`vitest run --project integration <path>`。

### 1.4 E2E 层（Playwright，`tests/e2e/ui/`）

E2E 覆盖"真实浏览器 + 真实静态导出站点/活 proxy"下的端到端行为。**当前仓库不存在 Playwright 基础设施**（无 `playwright.config.*`、无 `tests/e2e/ui/`、无 `@playwright/test` 依赖），这是 Agent 7 在 Wave 1 必须补齐的平台缺口（需要 `package.json` 变更时，须由 Agent 0 指派唯一 Owner，遵守 §6.1 依赖单一写入者约束）。

E2E 负责的 i18n 行为：
- **默认语言**：首次访问（无偏好）→ 英文 `en`。
- **切换即时生效**：真实浏览器点击切换 → `<html lang>` 更新 → 页面文本更新。
- **刷新保持**：切换 → `page.reload()` → 语言保持。
- **整页跳转恢复**：Login / SSO / MCP OAuth 整页跳转并返回后语言保持（同源 cookie/localStorage 恢复）。
- `<html lang>` 与当前语言一致。
- 缺 key 回退英文、不显示原始 key。
- 构建期 `<title>` / meta description 保持英文。

E2E 在静态导出下运行（`npm run build` 产生 `out/`）。Playwright 配置不在 vitest 内；作为独立阶段与 `npm run build` 串联，在 Wave 4A 完整回归使用。

### 1.5 三层分工示意

| 行为 | 单元 | 集成 | E2E |
|---|---|---|---|
| 插值/复数/格式化正确性 | ✅ 逻辑纯函数 | — | 抽查 |
| 切换语言组件渲染 | ✅ | ✅ 真实树 | ✅ 真实浏览器 |
| 表单状态保持 | — | ✅ | 抽查 |
| 刷新语言保持 | — | 🔸（jsdom 有限） | ✅ |
| 整页跳转恢复 | — | — | ✅ |
| `<html lang>` | ✅ 计算函数 | ✅ Provider 接线 | ✅ 浏览器 DOM |
| `<title>`/meta 英文 | — | — | ✅ |

## 2. key 一致性与硬编码扫描

### 2.1 key 集合一致性校验

自动校验 `en` 与 `zh-CN`（以及后续任何 locale）的字典 key 集合完全一致，即对所有 namespace：

```
keys(en/<ns>.json) == keys(zh-CN/<ns>.json)
```

- 语义 key（`common:action.delete`）必须成对存在；存在偏置即失败，连同**缺 key 清单**一起输出，退出码非 0。
- 同时校验"被引用的 key ⊆ 已声明 key"（组件 `t('ns:key')` / `useTranslation('ns')` 引用 vs 字典存在性），防止静态分析漏检的悬空 key 在运行时回退英文却被误当业务缺失。
- 结构与值类型也必须一致（嵌套层级、插值变量 `{{x}}` 集合一致，避免 en 有 `{{count}}` 而 zh 漏掉）。
- 运行方式：独立 node 脚本（见 `TEST_TOOLS.md`），纳入 CI 或门禁 G3 的"中英文 key 集一致"。

### 2.2 硬编码字符串扫描（只报告、不自动改写）

按方案 §3.4：扫描工具**只报告候选硬编码，不自动生成 key，不自动改写代码**。Agent 7 实现时严格遵守该约束——扫描结果进入报告，人工/产品决定是否改写。

扫描器规则（对 TSX/TS 源码）：
- 识别疑似承载用户可见文案的 `JSXText`（非注释、非空白、非纯符号）。
- 识别 `aria-label`、`title`、`placeholder`、`alt` 等可访问性/文案属性中的字符串字面量。
- 识别 `t('...')` 之外 `useMessage`/`useToast`/`toast(...)` 等常见未国际化文案入口（以项目现状校准）。
- 排除：模型名/API 字段/日志/代码示例引用、路径、正则、CSS 类名、`data-slot`、纯数值/符号、标识符。
- 输出：文件、行、候选文案、所属 namespace 建议（仅供人工参考），**不自动写入字典**。

扫描器产出人工 review 清单，用于判断"应改为 `t()` 的遗漏硬编码"。

### 2.3 误翻译保护

- 反向断言：模型名、API 字段、日志原文、代码示例在 zh 与 en 下渲染一致（不被翻译）。见 `TEST_CASES.md` TC 对应项。
- 扫描器补充规则：对 `en.json` / `zh-CN.json` 的 value 做"疑似把模型名/字段名当值"的启发式标注（如值与标识符/模型名称重合），仅供人工复核。

## 3. 中文布局检查矩阵

中文文案通常比英文短，但存在换行、量词、长词（如"成本跟踪"、英文品牌名混排）导致溢出/截断/重叠的风险。逐模块（导航/表格/弹窗/表单/全局壳层）检查：

| 区域 | 检查维度 | 断言/观察项 |
|---|---|---|
| Navbar / Leftnav | 溢出、折行、截断 | 菜单项不换行错位、品牌/Menu 图标不重叠；窄屏不横向溢出 |
| 表格（Models/API Keys/Usage/Cost/Budgets） | 列宽截断、单元格换行 | 中文表头不截断语义、值不溢出列边界、tooltip 不遮挡 |
| 弹窗/Modal | 溢出、空白、遮挡 | 标题/正文/按钮在视口内、无横向滚动条溢出、Close 按钮可点 |
| 表单 | label 对齐、placeholder 长度 | label 不折行错位、error 提示不截断、placeholder 不溢出控件 |
| 全局壳层 | 用户菜单、语言切换器 | 中文菜单项不截断，切换器标签完整 |

检查粒度：
- **自动化**：E2E/组件层断言"内容可读但不溢出容器"（无横向滚动、`scrollWidth <= clientWidth`、关键文本 `toBeVisible`），以及对特定元素做快照对比。
- **人工**：跨语言并列渲染的视觉走查（Agent 2 术语体验复核 + Agent 7 UI 检查），覆盖窄屏与中文长字符串用例；`V1_SCOPE_MANIFEST.md` 的 `UI 检查` 栏填写证据。

矩阵逐项仍落地于 `REGRESSION_MATRIX.md`。

## 4. 可访问性文案覆盖要求

i18n 不只是可见文本；`aria-label`、`title`、`placeholder`、`alt` 等必须随语言本地化（方案 §5.6 同步处理要求）：

- **每处用户可见文本都应有可访问等效**：图标按钮、关闭按钮、导航折叠按钮等必须提供本地化的 `aria-label`（`aria-hidden` 或纯装饰除外）。
- **辅助文本本地化**：`title`/tooltip、`placeholder`、校验 error message、`role="meter"` 的 `aria-valuenow`（用 `toHaveAttribute` 断言，避免 `toHaveValue` 误用——CLAUDE.md 约定）均本地化。
- **语序不破坏可访问性**：`aria-label` 与可见文本一致；`aria-labelledby` 指向本地化 label。
- **测试断言**：使用可访问查询（`getByRole`/`getByLabel`/`getByText`）优先，其次 `data-slot`；不使用 `eslint --fix` 批量修正 testing-library/jest-dom（CLAUDE.md 约定），`toHaveTextContent` 传 plain string 子串匹配。
- **英文回退对 a11y 文案同样生效**：zh 缺失时 `aria-label` 等回退英文，不显示原始 key。

## 5. 默认语言 / 刷新 / 首屏策略的测试侧重

- **默认语言**：无任何偏好时 = `en`；这是最终回退，任何层级测试都以此为基线。
- **刷新保持**（E2E）：`page.reload()` 后仍为当前语言；核实存储介质（SameSite cookie 或 localStorage——CLAUDE.md 禁 token，但语言偏好非敏感；偏好介质以架构决策为准）。
- **首屏闪烁**：`<Trans>`/`t()` 在资源就绪后渲染，首帧不暴露原始 key（方案 §3.4/§10）。E2E 首次导航截图断言不出现 `namespace:...` 原始 key 文本。
- **`<html lang>`**：与当前语言一致（`en` 或 `zh-CN`）；组件层与 E2E 均断言。

## 6. 自动化 / 人工边界总结

| 事项 | 自动化 | 人工 |
|---|---|---|
| 单元/组件逻辑 | ✅ | — |
| 集成布线 | ✅ | — |
| E2E（切换/刷新/跳转/lang/title） | ✅ | — |
| key 集合一致性 | ✅（脚本，CI） | — |
| 硬编码扫描 | ✅（脚本报告） | ▲ 决定是否改写 |
| 中文布局视觉 | 🔸（溢出断言） | ▲ 跨语言走查 |
| 翻译质量/术语 | — | ▲ Agent 2 复核 |
| 误翻译判断 | 🔸（启发式） | ▲ 人工复核 |

## 7. 测试数据与工具接口约定

- 测试用最小字典：单元/集成层注入 `en`+`zh-CN` 的最小命名空间子集，不足最小集时直接断言英文回退。
- 时间固定：日期/货币断言使用固定 locale 与固定时间源（fake timer），避免时区/运行时刻不稳定。
- 存储抽象：偏好读写针对于可注入的 storage/cookie 抽象 double，避免测试污染真实 localStorage。

## 8. 门禁映射

- **G0**：本策略与 `TEST_CASES.md` 通过 Review。
- **G1**：平台层测试（切换/回退/`<html lang>`/首屏/整页跳转）与 key 一致性脚本具备；`npm run build` + 定向 vitest 通过。
- **G2/G3**：各模块测试、UI 检查、`V1_SCOPE_MANIFEST.md` 证据齐，中英文 key 集一致。
- **G4**：`REGRESSION_MATRIX.md` 全量回归通过（Agent 7 报告 + Agent 8 汇总）。
