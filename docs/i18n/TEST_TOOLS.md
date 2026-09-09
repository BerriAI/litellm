# i18n 自动化测试任务清单与工具设计（TEST_TOOLS.md）

> 角色：Agent 3 `qa-architect`；实施：Agent 7 `i18n-qa`
> 原则：扫描工具**只报告、不自动改写**（方案 §3.4/§11）；测试只跑改动相关文件；Vitest 命令由 Agent 3 依配置确认（方案 §10）

## 0. 运行说明（Agent 7 前置校准）

- 定向 vitest：`cd ui/litellm-dashboard && npx vitest run --project <unit|component|integration> <path>`；type：`npm run test:types`。
- 禁止无路径全量 `npm run test`（380 文件、CI 才跑全量）。
- 工程检查：`npm run lint`、`npm run format:check`、`npm run build`。
- **E2E Playwright 基础设施当前不存在**（无 `playwright.config.*`、无 `@playwright/test`、无 `tests/e2e/ui/`）。Agent 7 在 Wave 1 搭建；涉及 `package.json`/锁文件变更时必须由 Agent 0 指派唯一 Owner（方案 §6.1）。

## 1. 自动化任务清单

| 编号 | 任务 | 形态 | 何时运行 | 对应用例 |
|---|---|---|---|---|
| T-01 | 搭建 Playwright 基础（config、`tests/e2e/ui/`、live proxy fixture） | E2E 基建 | Wave 1 | TC-01~04,06~08,16~18,22,26 |
| T-02 | key 集合一致性校验脚本 | CLI（node） | G3 门禁/CI、每次字典变更 | TC-25 |
| T-03 | 硬编码字符串扫描脚本（只报告） | CLI（node） | 每波次、G3/G4 | — |
| T-04 | 语言初始化/偏好/格式化逻辑单元测试 | `*.test.ts`（unit） | 平台 Wave 1 | TC-02,05,09~13,19 |
| T-05 | 语言切换器/量词/单组件渲染测试 | `*.test.tsx`（component） | 各功能波 | TC-03,09,10,23 |
| T-06 | Provider+页面集成测试（切换、表单保持、a11y、误翻译） | `*.integration.test.tsx` | 各功能波 | TC-03,14,20,21,23,24 |
| T-07 | 静态导出产物 key 泄漏/`<title>` 英文检查 | 构建后脚本/E2E | G1、G4 | TC-22,26 |
| T-08 | 中文布局溢出自动化断言（+人工走查入口） | `integration`/`e2e` | 各功能波、Wave 4A | TC-24 |
| T-09 | `V1_SCOPE_MANIFEST.md` 测试/UI 检查证据回填 | 人工+半自动 | G2/G3/G4 | — |
| T-10 | 完整回归矩阵执行并出报告 | Playwright+vitest | Wave 4A | REGRESSION_MATRIX |

## 2. key 集合一致性校验脚本（T-02）

**目标**：证明 en/zh（及新 locale）key 集、结构与插值变量一致；组件引用的 key 均存在。

**设计**：
- 输入：`src/locales/en/**/*.json`、`src/locales/zh-CN/**/*.json`。
- 对每个 namespace 比对：key 集合（递归扁平化，含冒号前缀）对称差、嵌套结构形状、每叶子的 `{{var}}` 插值变量集合。
- 组件引用校验：AST 扫描 `t('ns:key')`、`useTranslation('ns')`、`<Trans>` 与静态字典，找出引用但未声明的 key（悬空引用）。
- 输出：无差异 → 退出 0；有差异 → 列出缺失/多余 key、结构/插值差异、悬空引用清单，退出非 0。
- 挂接：G3 门禁"中英文 key 集一致"；可加入 CI lint 前置。

**命令形态**：
```bash
node scripts/i18n/check-keys.mjs                 # 或通过 vitest 一个用例驱动（TC-25）
```

## 3. 硬编码字符串扫描脚本（T-03）——只报告、不自动改写

**目标**：发现"漏改为 `t()` 的用户可见文案"，辅助人工补翻译。**禁止自动生成 key、禁止自动改写代码**。

**设计**（对 TSX/TS 源码 AST）：
- JSX 文案：非空白/非注释/非纯符号的 `JSXText`。
- 属性字面量：`aria-label`、`title`、`placeholder`、`alt` 中指向文案的字符串常量。
- 未国际化入口：`toast(...)`、`useMessage`、`notify(...)` 等（以项目现状校准 white-list）。
- 排除白名单：模型名/API 字段引用（`model`、`langsmith` 等字典值来源）、路径、正则、CSS 类、`data-slot`、纯数字/符号、标识符、已 `t()` 包裹的。
- 输出：文件、行号、候选文案、疑似命中形式（文案属性/JSX 文本/消息入口），按文件分组；**附带"仅报告"明确标注**，不写字典。

**报告消费**：Agent 7 输出人工 review 清单 → 产品/功能 Owner 决定是否改写；禁止工具自动化改写（方案 §3.4 硬约束）。

## 4. 中文布局检查：自动化 / 人工边界

**自动化（T-08）**：
- 组件/集成断言：中文渲染下对关键容器断言 `scrollWidth <= clientWidth`（无横向溢出）、关键文本 `toBeVisible`、无遮挡（元素间不重叠）。
- E2E：在窄视口（如 1280、1024、768）下对 Navbar/Leftnav/表格/弹窗断言同样条件，并做关键帧截图。
- 限制：视觉"美观/语义完整"无法纯自动化判定，自动只兜底"不溢出、不截断语义、可读可见"。

**人工（配合 Agent 2）**：
- 跨语言（en/zh）并列视觉走查，覆盖中文长字符串与窄屏。
- 截图证据回填 `V1_SCOPE_MANIFEST.md` 的 `UI 检查` 栏。
- `TEST_CASES.md` TC-24 标记为自动化+人工混合。

## 5. 静态导出产物检查（T-07）

- 构建后扫描 `out/**/*.html`：
  - 用户可见文本（剔除 script/style/内联 JSON 字典）不含 `namespace:key` 原始 key 形式（TC-26）。
  - `<title>` 与 `meta[name=description]` 为稳定英文、非原始 key、非半中文（TC-22）。
- 挂接：G1（首屏/构建验证）与 G4（发布验收）。

## 6. E2E 基建要点（T-01）

- 目录：`tests/e2e/ui/`（Playwright spec，与 vitest 分离）。
- 目标：静态导出 `out/` 或 live proxy；语言切换/刷新/整页跳转（Login/SSO/MCP OAuth）的偏好恢复需可稳定复现（测试账号或 mock）。
- `package.json` 变更（新增 `@playwright/test`）须经 Agent 0 指定唯一 Owner，遵守 §6.1。

## 7. Agent 7 建议实现优先级

1. **T-02 key 一致性校验**（成本低、门禁 G3 硬性、最易踩线）。
2. **T-03 硬编码扫描**（只报告，贯穿所有波次，辅助功能 Agent 自查）。
3. **T-01 E2E 基建 + 平台 smoke**（默认语言/切换/刷新/`<html lang>`/回退/不显示原始 key）。
4. **T-07 静态产物 key 泄漏 + `<title>` 英文检查**（G1 门禁）。
5. **T-04/T-05/T-06 各层测试**随功能波补齐，T-08 布局断言进入各模块验收，T-10 在 Wave 4A 全量回归。
