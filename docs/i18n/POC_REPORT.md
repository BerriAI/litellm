# LiteLLM Dashboard i18n — PoC 验证清单与执行步骤（POC_REPORT）

> 角色：Agent 1（`i18n-architect`）编写 PoC 必验项；Wave 1 的 Agent 4（平台）与 Agent 7（测试）照此执行并把证据回填到「结果 / 证据」栏。
> 约定：PoC-* 为「必验项」，每项都必须给出**方法**、**预期证据**、**结果**（通过/失败/未执行）。
> 本机（Agent 1 仅只读）已就地验证的项，标记 **「本机已验证」**；其余标记 **「待执行」**（执行者为 A4/A7）。

> 目标：在改动最少的前提下，对 `I18N_TECH_DESIGN.md` 的**关键设计假设**给出可复现证据，支撑 G1 门禁（ADR-i18n-01/03/04/05 转 Accepted）。
> 前置：Wave 1 平台基线由 A4 建立（`src/i18n/**` + `src/locales` 骨架 + 依赖写入，遵循 D12/FILE_OWNERSHIP）。

---

## 0. 环境前提（A4 先落实）

- 依赖：`i18next@^26.4.2` + `react-i18next@^17.0.13`（写入后运行 `npm ci` 于独立 worktree，禁止并行锁分叉——D12/v1.4）。
- `next.config.mjs`：**不改**（静态导出已满足）。
- 仅改动 `src/i18n/**`、`src/locales/**`、`src/app/layout.tsx`（挂 Provider）、语言切换器。

---

## PoC-1：静态导出构建成功 && i18next 依赖可解析（G1 前置）

- **方法**：A4 在平台 worktree `npm run build`（`output:"export"`）。检查：构建无因引入 i18next 产生的解析/打包错误；`out/` 正常产出；`turbopack` 能打包 json 资源。
- **预期证据**：`next build` 成功，`out/` 目录生成，无「Cannot resolve i18next/react-i18next」、无 static-export 舞台报错；`npm ls i18next react-i18next` 无 peer 冲突。
- **结果**：【✅ 通过 —— 2026-09-09，Agent 0 在集成分支 `i18n/w1-integration` 验证】`npm run build` 全绿：`✓ Compiled successfully`、`✓ Generating static pages using 11 workers (51/51)`（全部 51 路由静态预渲染）、TypeScript 阶段无平台错误、`BUILD_EXIT=0`。i18next/react-i18next/@playwright/test 依赖解析正常，无 peer 冲突。

## PoC-2：首屏策略（就绪门禁 + `<html lang>` 同步）行为

- **方法**：默认浏览器语言 en：加载页面，在首帧与 JS 就绪后截图/断言。
  1. en 用户：首帧即为中文 UI 不存在（应显示英文 UI），`<html lang="en">` 保持。
  2. zh-CN 偏好用户（预置 `litellm.locale=zh-CN` 再刷新）：JS 就绪前不渲染业务 `t()` 内容（就绪态/空），就绪后一次性渲染中文，且 `document.documentElement.lang === 'zh-CN'`。
- **预期证据**：截图对比 + DOM 断言：①en 首帧无 key/英文句闪烁；②zh 就绪前无业务文案、就绪后 `lang` 正确；③测量就绪门禁额外延时（performance 日志）记录数值，供评审接受性判断。
- **结果**：【待执行】

## PoC-3：`t()` / `<Trans>` 首帧不暴露原始 key（资源就绪门禁）

- **方法**：在导航/登录样面的一个组件用 `t('common:title')` 与一个 `<Trans>`。用 Playwright 在**首帧**（未就绪）与就绪后采样 DOM 文本。
- **预期证据**：首帧 DOM 中**不含** `common:title`、不含未翻译英文句子（即不以 key=原文）；就绪后显示译文。如有 `returnNull`，首帧对应节点为空而非显示 key。
- **结果**：【待执行】

## PoC-4：缺 key 英文回退（D2）

- **方法**：在 zh-CN 资源中删掉某个 en 有的 key（临时），UI 切 zh-CN，观察该字符串。
- **预期证据**：该串回退英文（`fallbackLng:'en'`），**不**崩、不显示 key、控制台（dev）有 missingKey warning。
- **结果**：【待执行】

## PoC-5：刷新后语言偏好保留（D6/D5）

- **方法**：切 zh-CN → 刷新 → 断言仍 zh-CN；清空 cookie 但留 localStorage → 刷新 → 仍 zh-CN（双层读取）；两者都清 → 回退浏览器语言/en。
- **预期证据**：各分支刷新后语言正确；`litellm.locale` cookie 存在且 `SameSite=Lax`。
- **结果**：【待执行】

## PoC-6：英文为首帧默认、无偏好时按浏览器语言（D5/D2）

- **方法**：无任何偏好 cookie/localStorage；浏览器 `navigator.language` 分别设 en、zh-CN、zh、zh-Hans、fr。
- **预期证据**：en→英文；zh/zh-Hans→zh-CN；fr（不支持）→英文。首帧英文合法，无闪烁。
- **结果**：【待执行】

## PoC-7：整页跳转语言恢复（Login/SSO/MCP OAuth 回跳，D10）

- **方法**：在 `(dashboard)` 中切 zh-CN → 触发到 `/login` 或模拟 SSO/OAuth 全页跳转（改变 URL/全量 reload）→ 返回原页。
- **预期证据**：回跳后仍 zh-CN（从同源 cookie 恢复），`<html lang>` 一致，无 key 闪烁；未登录直登场景与 OAuth 回跳分别覆盖。
- **结果**：【待执行】

## PoC-8：中文量词/计数（D9）在真实组件表现

- **方法**：在带 `{{count}}` 的计数展示（如 usage 表格）切 zh-CN，覆盖 count=0/1/2。
- **预期证据**：输出如「0 个 / 1 个 / 2 个」，无英文复数 s；量词与数为「数+量」顺序正确出现。
- **结果**：【待执行】

## PoC-9：toast / 构建产物不含未翻译硬编码（回归）

- **方法**：扫描构建后产物与关键页面文本，确认没有「key 串」残留为可见 UI 文本。
- **预期证据**：`grep` 产物无 `\w+:\w+\.` 形式的 key 作为文本暴露；关键页面无英文句被用户看到（除 `title`/meta 英文外）。
- **结果**：【待执行】

---

## 本机已验证项（Agent 1，只读，2026-09-09）

| # | 项 | 验证方式 | 结果 |
|---|---|---|---|
| 1 | `react-i18next@17.0.13` 兼容 React 19 / TS 5 | `npm view react-i18next peerDependencies` → `react>=16.8.0`, `i18next>=26.2.0`, `ts ^5\|\|^6\|\|^7`；项目 React 19.2.8 / TS 5.9.3 满足 | 通过 |
| 2 | `i18next@26.4.2` 无冲突 peer | `npm view i18next@latest peerDependencies` → 仅 `typescript` | 通过 |
| 3 | 根 Provider 链结构 | 读 `src/app/layout.tsx`：`ThemeProvider→NuqsAdapter→ReactQueryProvider→AuthProvider`；`<html lang="en" suppressHydrationWarning>`；metadata 英文 | 确认 |
| 4 | `src/i18n` / `src/locales` 当前不存在（基线干净） | `ls` | 确认（Wave0 从零建） |
| 5 | 路由组 layout 分布 | 根 / `(dashboard)` / `chat` / `connect` 各有一 `layout.tsx`；`(dashboard)` 为 use client 且另含 SidebarProvider/ThemeContext/PluginModeContext | 确认（接缝见设计 §8） |
| 6 | 后端无 `UI settings.language` | `UISettings` 白名单 `ALLOWED_UI_SETTINGS_FIELDS` 不含 `language`；`/get/ui_settings` 由 `LiteLLM_UISettings` 表支撑 | 确认（→ADR-i18n-07） |
| 7 | next export 配置 | `next.config.mjs`: `output:"export"`, `trailingSlash:true`, `images.unoptimized` | 确认（静态导出约束成立） |

## 待执行（Wave 1 A4/A7 回填）

- PoC-1 .. PoC-9 均【待执行】；其中 PoC-1（构建+依赖）、PoC-2/3（首屏与门禁）、PoC-5（刷新）、PoC-7（跳转恢复）为 **G1 关键门禁**，优先完成并回填证据文件（证据文件命名遵从 v1.4 §11 具名证据约定，由 Agent 3/7 落地）。

---

## 建议的 PoC 测试层级归属（供 Agent 3 细化）

- 单测（unit）：`localePreferences.ts`（D5 优先级）、`detectLocale.ts`（`zh`→`zh-CN`）、`i18n.ts` 实例化。
- 集成：`I18nProvider` 就绪门禁渲染（PoC-3）、Layout 挂 Provider 后 `useTranslation` 可用。
- E2E（`tests/e2e/ui/`，Playwright 对 live proxy）：PoC-2/5/6/7（首屏、刷新、浏览器语言、整页回跳）。
