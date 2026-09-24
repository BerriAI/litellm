# LiteLLM Dashboard i18n — DECISIONS（决策记录）

> 维护者：Agent 0。记录所有影响方案/范围的决策。`状态`: 已定 / 待定 / 待 PoC。

## 已定决策（来自 I18N_MULTI_AGENT_PLAN.md v1.4）

| # | 决策 | 结论 | 状态 |
|---|---|---|---|
| D1 | 国际化库 | `i18next` + `react-i18next` | 已定 |
| D2 | 默认/目标语言 | 默认 `en`，目标 `zh-CN`；缺 key 回退 en | 已定 |
| D3 | 运行时/资源目录 | 运行时 `src/i18n/**`，资源 `src/locales/{en,zh-CN}/**` | 已定 |
| D4 | 静态导出约束 | 不依赖服务端 locale 路由；不将 httpOnly cookie 作前端必需；切换主要在客户端 | 已定 |
| D5 | 语言偏好优先级 | 用户选择 → 用户级 UI 设置(若支持) → cookie/localStorage → 浏览器语言 → en | 已定 |
| D6 | 语言偏好存储 | cookie + localStorage 双层；SSR 首屏靠客户端收敛 | 已定 |
| D7 | 首屏策略 | 从"初始化脚本/就绪门禁/接受短暂切换"候选中**必须选定 1 种并经 PoC 验证** | **待 PoC** |
| D8 | 构建期 `<title>`/meta | v1 保持英文，不做多语言 SEO | 已定 |
| D9 | 复数量词 | zh 不建形态复数分支，用计数插值 + 量词（`{{count}} 个`） | 已定 |
| D10 | 整页跳转语言保持 | Login/SSO/MCP OAuth 回跳后需从 cookie/localStorage 恢复 | 待 PoC 验证 |
| D11 | `common` namespace Owner | Wave1 由 A4 建骨架，G1 后移交 A5 | 已定 |
| D12 | 依赖文件写入者 | `package.json`/lock 默认仅 A4 | 已定 |
| D13 | 文档目录 | 所有设计/验收文档统一放 `docs/i18n/` | 已定（本轮） |
| D14 | 并发槽位 | 1 总控 + 最多 3 执行；Agent 0 即主智能体 | 已定 |

## 待定 / 需产品确认

| # | 待决策项 | 说明 | 归属 |
|---|---|---|---|
| P1 | 管理员全局 UI 设置是否覆盖用户主动语言选择 | Wave 0 调查：后端**存在**部分 UI settings，但 `UM settings.language` 等价字段**不在** `ALLOWED_UI_SETTINGS_FIELDS` 白名单，即**无用户级 language 设置**。故 v1 语言偏好优先级实际为：用户选择 → cookie/localStorage → 浏览器语言 → en。 | A1 已调查并写入 ADR-07（Accepted） |
| P2 | `UI settings.language` 是否存在且进 v1 | **不存在**（只读调查确认），v1 不依赖后端 language 字段。 | A1 已结论 |
| P3 | **E2E（Playwright）基建缺口** | Wave 0 核实：dashboard 仓库**当前无 Playwright 基建**（无 `tests/e2e/ui/`、无 `playwright.config.*`、无 `@playwright/test`）。方案 §10 的多项 E2E（首屏/刷新/整页回跳）依赖该层级。**已定：选 a —— Wave 1 补齐 E2E 基建（引入 Playwright）**；会触及 `package.json`，按 §6.1 规则 10 由 A0 指定**唯一写入者（默认 Agent 4）**。 | A0 决策 **已定=选项a** |
| P4 | `@playwright` 依赖写入者 | 已随 P3=a 确定：Wave 1 由 **Agent 4** 作为 `package.json` 唯一写入者引入 Playwright 依赖；A0 在任务单中书面授权。 | A0 决策 **已定=Agent 4** |
| P5 | **语言偏好存储键名统一**（G0 Review 发现） | 设计文档间不一致：`I18N_ADR.md`/`I18N_TECH_DESIGN.md`/`POC_REPORT.md` 用 `dashboard.locale`；`LANGUAGE_SWITCHER_SPEC.md`/`LOCALE_NAVIGATION_BEHAVIOR.md` 用 `litellm.locale`。**已定：统一为 `litellm.locale`**（与现有候选实现历史一致、Agent 2 文档多数采用）；A4 在 `src/i18n/localePreferences.ts` 以单一常量导出，A5/6 不直接操作存储。 | A0 决策 **已定=litellm.locale** |
