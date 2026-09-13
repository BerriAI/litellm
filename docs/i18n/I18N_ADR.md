# LiteLLM Dashboard i18n — 架构决策记录（I18N_ADR）

> 维护者：Agent 1（`i18n-architect`）· Wave 0
> 状态：`Proposed`（待 Agent 0 Baseline Review 后置为 `Accepted`；涉 PoC 条目待验证后转 `Accepted`）
> **G0 Review（Agent 0，2026-09-09）：APPROVED。** ADR-02/06/07/08（Accepted）维持；ADR-01/03/04/05 保持 Proposed，待 Wave 1 由 A4/A7 回填 PoC 证据（PoC-1/2/3/4/5/7）后转 Accepted（G1 门禁）。
> 编号约定：ADR-i18n-01 .. N。与 `DECISIONS.md`（D1–D14）及技术设计 `I18N_TECH_DESIGN.md` 相互引用。

---

## ADR-i18n-01：i18n 库选型

- **状态**：Proposed（待 G0 通过）
- **背景**：静态导出（`output:"export"`）、Next.js 16、React 19；需要运行时切换 `en`/`zh-CN`、缺 key 回退英文、类型安全的 key。
- **决策**：采用 `i18next` + `react-i18next`（`i18next@^26.4.2` + `react-i18next@^17.0.13`）。
- **理由**：`react-i18next` 纯客户端 Context/Hook，不依赖 Next.js SSR/服务端钩子，契合静态导出；peer 依赖 `react>=16.8.0` 满足 React 19.2.8；`i18next` 无 react peer。生态成熟、`<Trans>`/复数/插值完备。
- **后果**：
  - 积极：低成本接入、类型增强可行、中文量词策略（D9）可自然实现。
  - 消极/风险：新增两份依赖，须由 A4 单一写入（D12）；需 `knip`/lint 放行；类型资源的维护由 A4 承担。

## ADR-i18n-02：静态导出下不采用服务端 locale 路由 / 不在服务端解析 locale

- **状态**：Accepted（继承 D4）
- **背景**：`output:"export"` 无 Node 运行时，服务端无法协商 locale；Next app-router 的 locale 机制（`generateStaticParams`、`headers()`、`cookies()`）在纯静态导出下不可用或无效。
- **决策**：v1 **完全不依赖服务端 locale 路由**；语言初始化与切换全部在客户端 `I18nProvider` 完成；不将 httpOnly cookie 作为前端必需能力（普通 `SameSite=Lax` cookie 用于跨整页跳转恢复，见 ADR-i18n-05）。
- **理由**：静态导出别无选择；客户端收敛成本低，且与「首帧英文合法（D2/D8）」矛盾小。
- **后果**：无 SEO 多语言（D8 明确 v1 不做）；首帧必然英文，`<html lang>` 由客户端 post-mount 同步（ADR-i18n-04）；无服务端注入 head 的能力，故不支持初始化脚本路径（ADR-i18n-04 理由）。

## ADR-i18n-03：资源同步就绪门禁（禁止首帧暴露 key）

- **状态**：Proposed（待 PoC 佐证）
- **背景**：§3.1 要求：禁止首帧显示原始 key 后再替换为译文。
- **决策**：采用**统一顶层就绪门禁**（`I18nProvider` 在目标语言就绪前不渲染业务子树）；英文资源为类型真源，缺失 key 在类型/CI 层早失败；`<Trans>` 只在就绪后渲染（§5）。
- **理由**：就绪门禁能同时覆盖 `t()` 与 `<Trans>`，避免每个组件自行 `if(!ready)` 的碎片化；结合 key=语义（非句子）的规范，从根源杜绝「key 即原文被暴露」。
- **后果**：
  - 积极：首帧要么是合法英文，要么是短暂的就绪态，无 key 闪烁。
  - 消极：引入极短的渲染门禁；需保证 en 是同步就绪（不改后端），zh-CN 是注册的静态资源内切换（同步）。
  - 风险：若某处绕过门禁直接渲染 `t()`，仍可能瞬时暴露 key → 用 CI 的「资源键集合一致」检查与代码评审兜底（TECH_RISKS R9）。

## ADR-i18n-04：首屏策略选定「语言就绪门禁 + 挂载后同步 `<html lang>`」（不用初始化脚本、不接受短暂切换）

- **状态**：Proposed（G0/G1 门禁要求必须选定；PoC 验证）
- **背景**：候选 3 种：初始化脚本 / 语言就绪门禁 / 接受短暂切换（D7）。
- **决策**：选定**语言就绪门禁**，并在 `changeLanguage` 完成后同步 `document.documentElement.lang`。
- **理由**：
  - 初始化脚本在纯静态导出下无服务端 head 注入时机，且运行时加载 zh-CN JSON 只能在 client JS 就绪后完成，前置内联脚本收益极低。
  - 接受短暂切换制造「英文按键闪烁」，违反 §3.1 精神与首屏体验。
  - 门禁方案与「首帧英文本就合法」天然一致，切换收敛到偏好语言只需一次、且行为可测。
- **后果**：`<html lang>` 首帧 `en`，post-mount 更新为目标语言（`document.documentElement.lang`）；`suppressHydrationWarning` 沿用现有做法（与 next-themes 相同）；不写内联脚本、不强求 head 改写。需 PoC 量化门禁延迟可接受（POC_REPORT PoC-2/3）。

## ADR-i18n-05：语言偏好采用「SameSite=Lax cookie + localStorage 双层」且取显式写入者为准

- **状态**：Proposed（D6 + 本 ADR 澄清）
- **背景**：Login/SSO/MCP OAuth 可能整页跳转，跳转返回后需恢复语言（D10）；刷新需保留（§1.1）。
- **决策**：偏好键 `dashboard.locale` 同时写 cookie（`SameSite=Lax; path=/;`，生产加 `Secure`）与 localStorage；读到任一存在即视为用户显式选择；无需偏好时首次回退浏览器语言→en。
- **理由**：
  - cookie：跨整页（同源）跳转天然携带，SSO 回跳同域可恢复（D10）。
  - `SameSite=Lax`：允许顶层导航回跳带 cookie，又不阻断，优于 Strict 的极端。（若 SSO 走第三方域且需带 cookie，再评估 `None`，属后续风险 R8。）
  - localStorage 双写：供无 cookie 上下文（如纯静态托管其他子路径/隐私模式兜底）与客户端快速读取。
- **后果**：需保证 cookie/localStorage 两处不冲突（以显式写入顺序为准）；无权删除两者时回退 en；不做后端 httpOnly 语言协商（D4）。

## ADR-i18n-06：构建期 `<title>` / metadata 在 v1 保持英文，不做多语言 SEO

- **状态**：Accepted（继承 D8）
- **背景**：静态导出下无服务端 metadata 动态化；`next/metadata` 多语言需通过路由/静态生成实现，成本高、非 UI 功能。
- **决策**：`src/app/layout.tsx` 的 `metadata`（title/description）**保持不变**（英文），不为 SEO 做多语言输出；页面内可见标题/文案中文化由组件 `t()` 承担。
- **理由**：v1 范围不含多语言 SEO（§1.2）；避免在静态导出下为 `title` 引入额外复杂度。
- **后果**：搜索引擎看到的 title 为英文；若未来需要，另立 ADR 引入 per-locale 静态 `title` 生成。

## ADR-i18n-07：v1 不使用后端 `UI settings.language`（字段不存在）

- **状态**：Accepted（P1/P2 调查结论）
- **背景**：P1/P2 待 A1 调查 `UI settings.language` 是否存在/是否进 v1。
- **决策**：经只读调查，后端 `UISettings`（`/get/ui_settings`）的持久化白名单 `ALLOWED_UI_SETTINGS_FIELDS` **不包含 `language`**。故 v1 **不采用**后端全局语言设置；管理员覆盖用户选择的产品决策在 v1 不生效。
- **理由**：字段不存在于持久化白名单 → 即使前端写入也不会被保存/下发；强行使用无意义且会引入两段式异步。
- **后果**：语言偏好仅为「用户显式 → cookie/localStorage → 浏览器 → en」；若未来后端新增 `language` 白名单字段，另立 ADR 补「管理员全局 vs 用户选择」覆盖语义（预留 L5 判定位）。

## ADR-i18n-08：`src/i18n/**` 与 namespace 注册表由 Agent 4 永久独占，功能 Agent 不共享编辑大 JSON

- **状态**：Accepted（继承 FILE_OWNERSHIP / D3）
- **背景**：多智能体并行，`i18n` 初始化代码与资源注册是共享单点，并行编辑易冲突。
- **决策**：`src/i18n/**`（Provider、初始化、locale 偏好、类型、注册表/资源映射）与 `src/locales/{en,zh-CN}/**` 目录骨架归 A4；功能 Agent 只写自己的 namespace json + 引用 key，新增 namespace 经 A0 派单由 A4 注册；`package.json`/lock 仅 A4 写。
- **理由**：避免共享单点写冲突（§6/D12），保证注册表唯一真源。
- **后果**：功能 Agent 接入需等待 A4 完成平台基线（G1），形成 Wave 依赖；须遵守 FILE_OWNERSHIP 规则 1/6。

---

## ADR 状态汇总（供 Agent 0 回填）

| ADR | 主题 | 状态 | 依赖 PoC |
|---|---|---|---|
| 01 | i18n 库选型 | Proposed | PoC-1（兼容安装） |
| 02 | 静态导出无服务端 locale | Accepted | 无 |
| 03 | 资源就绪门禁（禁暴 key） | Proposed | PoC-3/4 |
| 04 | 首屏策略=就绪门禁+`lang` 同步 | Proposed | PoC-2/3 |
| 05 | cookie+localStorage 双层偏好 | Proposed | PoC-5/7 |
| 06 | title/meta v1 英文 | Accepted | 无 |
| 07 | v1 不用后端 language | Accepted | 无（只读调查） |
| 08 | 注册表 A4 独占 | Accepted | 无 |
