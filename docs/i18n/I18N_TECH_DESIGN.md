# LiteLLM Dashboard i18n — 技术方案设计（I18N_TECH_DESIGN）

> 文档角色：Agent 1（`i18n-architect`）· Wave 0 交付 · 状态：**待 Baseline Review → 待 PoC 验证**
> 基线：`I18N_MULTI_AGENT_PLAN.md` v1.4（§3、§5.2、D1–D14）
> 范围：纯技术设计；本文档 **不修改任何产品代码**，所有文件路径均为设计目标而非已落地实现。

---

## 0. 阅读指引

| 章节 | 内容 | 对应门禁/决策 |
|---|---|---|
| §1 | 技术选型兼容性结论（静态导出 + i18next + React 19） | D1、D4 |
| §2 | Provider 层级 + `src/i18n/**` 模块划分 | D3 |
| §3 | locale 初始化 与《语言偏好优先级》实现映射 | D5、D6 |
| §4 | `<html lang>` 同步 + **首屏策略（选定 1 种）** | D7、G0/G1 门禁 |
| §5 | 普通 `t()` 与 `<Trans>` 的资源就绪门禁 | §3.1、G1 |
| §6 | `src/locales/{en,zh-CN}/**` 组织 与 namespace 注册 | D3、§3.3/3.4 |
| §7 | 后端 `UI settings.language` 调查结论（P1/P2） | P1、P2 |
| §8 | 与既有 Provider/路由组 layout 的接缝 | — |

---

## 1. 技术选型兼容性结论

### 1.1 结论

**`i18next` + `react-i18next` 与 `output: "export"`（静态导出）+ Next.js 16 + React 19 完全兼容，可作为 v1 方案。** 理由如下：

1. `react-i18next` 的核心机制是**纯客户端**的 React Context + Hook（`useTranslation`），不依赖 Next.js 的任何服务端渲染钩子（`headers()`、`cookies()`、middleware、app-router locale 约定）。它既不需要 SSR 数据注入，也不要求在 node 侧执行。
2. 在 `output: "export"` 下没有 Node 运行时，因此本方案**从不尝试在服务端解析 locale**（不依赖服务端 locale 路由 / httpOnly cookie 协商，见 D4）。locale 初始化完全在客户端 `I18nProvider`（`use client`）的挂载期完成。
3. 静态 HTML 首帧默认按英文渲染（D8：构建期 title/meta 保持英文）。`<html lang>` 由客户端在挂载后同步（§4），`suppressHydrationWarning` 仅抑制该属性的 hydration 警告（与 next-themes 现有做法一致，见 `src/app/layout.tsx`）。

### 1.2 已验证版本依据（本机只读核查，2026-09-09）

| 项 | 值 | 依据 |
|---|---|---|
| 项目 React | `19.2.8` | `ui/litellm-dashboard/package.json` |
| 项目 Next | `16.2.11` | 同上 |
| 项目 TS | `5.9.3` | 同上 |
| `react-i18next` 最新 | `17.0.13`（2026-09-01 发布） | `npm view` |
| `react-i18next` peerDeps | `react: >= 16.8.0`, `i18next: >= 26.2.0`, `typescript: ^5\|\|^6\|\|^7` | `npm view react-i18next peerDependencies` |
| `i18next` 最新 | `26.4.2` | `npm view i18next version` |
| `i18next` peerDeps | 无 react peer（仅 `typescript`） | `npm view i18next@latest peerDependencies` |

**兼容性判定**：React 19.2.8 ≥ 16.8.0，故 `react-i18next@17` 与 `i18next@26` 满足 peer 约束。**建议锁定范围**：`react-i18next@^17.0.13` + `i18next@^26.4.2`（Wave 1 由 Agent 4 写入，遵循 D12 单一写入者）。不采用旧版 15/16 行，避免为规避问题而引入历史缺陷。

> 说明：以上为**只读** `npm view`（不安装、不改依赖）。实际安装验证由 Wave 1 Agent 4 承担（见 `POC_REPORT.md` PoC-1）。

---

## 2. Provider 层级与 `src/i18n/**` 模块划分

### 2.1 Provider 放置层级

`I18nProvider` 放**根 `src/app/layout.tsx`，作为最外层 Provider**（在既有 `ThemeProvider` 之外或与其并列的最上方）。理由：

- 根 layout 渲染 `<html>`，`I18nProvider` 在此可统一同步 `document.documentElement.lang`（§4）。
- 所有路由组（`(dashboard)`、`chat`、`connect`、`login`、`onboarding` 等）都经由根 layout，各自无需重复挂 Provider。
- `I18nProvider` 是 `use client` 组件；因 Next.js 中根 layout 本身是 server component，需用一个 `"use client"` 的子组件包装。页面内部不含服务端下发的动态 locale，故无需用全局 server/client provider 分支。

设计目标示意（**非当前代码，禁止修改**）：

```tsx
// src/app/layout.tsx（设计目标）
<html lang="en" suppressHydrationWarning>
  <body className={inter.className}>
    <I18nProvider>          {/* 新增：最外层，use client 包装 */}
      <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
        <NuqsAdapter>
          <ReactQueryProvider>
            <AuthProvider>{children}</AuthProvider>
            <Toaster />
          </ReactQueryProvider>
        </NuqsAdapter>
      </ThemeProvider>
    </I18nProvider>
  </body>
</html>
```

### 2.2 `src/i18n/**` 模块划分（Ownership：Agent 4，永久，见 FILE_OWNERSHIP）

```
src/i18n/
├── index.ts            # 公开出口：导出 I18nProvider 与 getI18n/useI18n 便捷 API
├── I18nProvider.tsx    # "use client" Provider：初始化实例 + 语言就绪门禁 + <html lang> 同步
├── i18n.ts             # createI18n(): 构建/复用 i18next 单例（resources、fallbackLng、supportedLngs）
├── localePreferences.ts# 读/写语言偏好（cookie + localStorage 双层 + 浏览器语言嗅探）——纯函数，可单测
├── detectLocale.ts     # 语言解析/规范化（'zh'→'zh-CN'、大小写、支持列表校验）
├── types.d.ts          # 全局声明增强：模块扩展 'i18next' 的 CustomTypeOptions（资源键类型安全）
└── resources/registry.ts  # === namespace 注册表 + 资源加载映射（唯一真源）=== 见 §6.2
```

职责边界：

- `i18n.ts` 持有 i18next **单例**的生命周期（`initReactI18next`、`init`）。为保证 SSR 静态构建与 HMR 稳定，导出一个惰性 `getI18n()` 而非模块顶层 `init`（避免重复初始化）。
- `localePreferences.ts` 不依赖 React，纯逻辑（读 cookie/localStorage/navigator），供单测覆盖 D5 优先级。
- `resources/registry.ts` 是**唯一的 namespace 注册表 + 资源加载映射**：声明「locale ↔ namespace ↔ json 路径」，并导出给 `i18n.ts` 构造 resources。功能 Agent（A5/A6）**只往 json 加 key，不改此注册表**（FILE_OWNERSHIP 关键规则 1/6）。
- `types.d.ts` 用 `CustomTypeOptions`（`defaultNS`、`resources`）让 `t()` 的 key 具备类型约束；英文资源为类型真源（D2：en 为真源）。

---

## 3. locale 初始化 与《语言偏好优先级》实现映射

### 3.1 优先级实现映射（D5 → 代码）

规范化后的优先级链（`detectLocale.ts + localePreferences.ts`）：

```text
用户主动选择(显式写入偏好存储)
 → 用户级 UI 设置(后端 UI settings.language —— v1 不采用，见 §7)
 → SameSite cookie / localStorage（双层）
 → 浏览器语言（navigator.languages，经 supportedLngs 过滤）
 → 英文 en（兜底）
```

实现要点：

- **用户主动选择 vs 残留存储的区分**：用户在切换器显式选择语言时，`setLocale` 同时写 **cookie + localStorage** 并 `changeLanguage`。读取时把「cookie/localStorage 中存在显式写入的 locale」视为「用户主动选择」。为区分「显式选择」与「自动探测回写」，可用单独的 marker（如 cookie 值带 `explicit` 标记，或 localStorage 键与探测键分离）。**v1 建议**：统一用一个偏好键（如 `dashboard.locale`），只要该键存在即为用户选择；首次无键时才走浏览器语言。该判定需在 §3.2 明确，避免「用户从不手选」时被浏览器语言误判为主动选择。
- **cookie 属性**：`SameSite=Lax`（非 Strict，避免 Okta/SSO 回跳丢失）、`path=/`、非 httpOnly（前端需读，D4 允许）。语言不敏感隐私，不设 Secure 亦可在 http 下工作；生产建议 `Secure`（见 TECH_RISKS R8）。
- **无后端用户级设置**：P1/P2 结论为 v1 不使用 `UI settings.language`（§7），故优先级链简化为「显式选择 → cookie/localStorage → 浏览器语言 → en」。
- **i18next 配置**：`supportedLngs: ['en','zh-CN']`、`fallbackLng: 'en'`、`nonExplicitSupportedLngs: false` + 在 `detectLocale` 中把 `navigator.language` 规范化为 `zh-CN`（`'zh'`/`'zh-Hans'`→`'zh-CN'`；其余→en）。load 路径由 registry 提供静态 resources，不使用 `loadPath` 网络加载（静态导出下目录外资源不可用）。

### 3.2 关键判定汇总（供 PoC 与后续澄清）

| # | 判定 | v1 默认 | 依据 |
|---|---|---|---|
| L1 | 偏好键唯一性 | 单一键 `dashboard.locale`，存在即视为显式选择 | §3.1 |
| L2 | 浏览器语言含义 | 仅首次（无偏好键）时使用 | §3.1 |
| L3 | cookie 属性 | `SameSite=Lax; path=/;`（生产加 `Secure`） | §3.1 / R8 |
| L4 | 根 layout 的 `lang` 静态默认 | `en`（与静态 HTML 一致） | §4 |
| L5 | 管理员全局覆盖用户选择 | v1 不实现（无该字段） | §7 / D8 |

---

## 4. `<html lang>` 同步 与 首屏策略（G0/G1 门禁，必须选定）

### 4.1 首屏策略：**选定「语言就绪门禁 + 挂载后同步 `<html lang>`」**，不采用初始化脚本、不接受短暂切换。

从候选「初始化脚本 / 语言就绪门禁 / 接受短暂切换」中，**选定「语言就绪门禁」（并且不在 `<head>` 内注入内联脚本）**。理由：

- **初始化脚本（在静态导出下基本无效）**：该类脚本需在 next 的 `<script>` 阶段于头部内联以在 paint 前改写 `lang`/文本。但静态导出没有服务端可注入的 head 时机，且「切换到中文」需要**运行时加载 zh-CN 资源**——这只能在 client JS 就绪后通过 i18next 完成，内联脚本无法在现代打包下前置加载 JSON 资源并替换整棵 textContent。故初始化脚本收益低。
- **接受短暂切换**：会把「首帧英文→JS 就绪后切中文」的闪烁当作常态，直接违反 D7「禁止首屏显示原始 key 后再替换」的精神（§5 进一步禁止 key 暴露）。不可接受。
- **语言就绪门禁**最符合静态导出约束：首帧即是合法的英文 UI（D2 默认语言为 en，D8 title 为英文），不存在「错误语言闪烁」；JS 就绪后 i18next 收敛到偏好语言并同步 `<html lang>`。就绪门禁只是**短于 paint 的少量额外延时**，且保证「看到的就是最终语言」或「简短英文首帧后一次切换」。

### 4.2 就绪门禁实现草图（设计目标）

`I18nProvider` 在挂载时：

1. 读取偏好（§3）→ 得到目标语言 `target`。
2. 若 `target === 'en'`：`i18next.changeLanguage('en')`，`<html lang>` 已为 `en`，**无需门禁**，直接渲染 children。
3. 若 `target === 'zh-CN'`：`await i18next.changeLanguage('zh-CN')`（resources 已在 `init` 时注册，changeLanguage 为同步内部切换完的 `t` 就绪）,当 `i18n.isInitialized && i18n.language === 'zh-CN'` 时为「就绪」。就绪前渲染一个**轻量就绪态**（如 `aria-busy` 的空壳/骨架，不渲染依赖 `t()` 的业务内容），就绪后渲染 children 并设置 `document.documentElement.lang = 'zh-CN'`。

```tsx
// src/i18n/I18nProvider.tsx（设计目标）
"use client";
export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(() => resolveTarget() === "en"); // en 直接就绪
  useEffect(() => {
    const target = resolveTarget();            // localePreferences + detectLocale
    getI18n().then(async (i18n) => {
      if (target === "zh-CN") await i18n.changeLanguage("zh-CN");
      document.documentElement.lang = i18n.language;   // 与 <html lang> 同步
      setReady(true);
    });
  }, []);
  if (!ready) return null;      // 或轻量就绪态；首帧为 en 时立即渲染
  return children;              // 实际会包一层 I18nextProvider/useSSR(false)
}
```

要点：
- **`<html lang>` 同步**：`document.documentElement.lang = i18n.language`，在 `changeLanguage` 完成后、children 渲染前执行，使浏览器/屏幕阅读器解析到的语言与实际内容一致。
- 不依赖内联脚本、不动 `metadata`（D8 保持英文 title）。
- 就绪态渲染 null 或骨架，**不渲染任何 `t()`/`<Trans>` 内容**，天然杜绝「key 先暴露」（§5）。

### 4.3 PoC 验证

首屏切换的判定证据与测量由 Wave 1 执行（见 `POC_REPORT.md` PoC-2/PoC-3）：重点验证「en 默认首帧合法」、「切 zh-CN 后 `<html lang>` 更新」、「就绪门禁下业务内容不在就绪前出现」。

---

## 5. 普通 `t()` 与 `<Trans>` 的资源就绪门禁

### 5.1 统一门禁原则

**禁止首帧暴露原文 key。** 凡渲染翻译内容的组件，必须保证 i18next 资源就绪后其 `t`/`<Trans>` 才被求值。承接 §4.2 的顶层就绪门禁，业务组件**默认**就处于就绪之后（因为 `I18nProvider` 就绪前不渲染 children）。在此基础上再加两层保障：

1. **资源键集合约束**：英文 `en/*.json` 是类型与资源真源（D2），`t()` 的 key 受 `CustomTypeOptions` 类型校验。**只允许引用已注册 namespace 的 key**——key 不存在会在类型层报错，从而无法「引用一个没资源的 key」（见 §6 的 en/zh 键一致性检查 + `test:types`）。
2. **运行时兜底**：i18next 对缺失 key 默认回退 `fallbackLng='en'`；若 en 亦缺，`returnNull`/`returnEmptyString` 策略应配置为**返回 key 本身或空串且不渲染原文句子**，并在开发模式打 warning 以便早发现。**绝不允许把「英文句子」作为 key**（§3.4 语义 key 规范），从根上防止「key=原文→暴露原文」的伪就绪。

### 5.2 `<Trans>` 的门禁

- `<Trans>` 依赖资源同步就绪（§3.1：必须在资源同步就绪或统一 ready 门禁之后渲染）。
- 因 §4.2 顶层门禁已保证业务子树在就绪后才挂载，`<Trans>` 不会有「先渲染占位再替换」窗口。
- 若个别模块需要在自己内部再等，使用 `useTranslation` 的 `ready` 标志：`const { t, ready } = useTranslation('ns'); if (!ready) return skeleton;`。**推荐所有激进的内联文案一次性接入统一门禁，避免每个组件各自开花**；组件级 `ready` 仅用于少数在就绪后有异步依赖的场景。
- `<Trans>` 复数/插值注意事项：zh 不建形态复数分支（D9），用 `{{count}}` 计数插值；`<Trans>` 内嵌组件（如带链接）需保证 `i18n` 与组件 props 的 children 均就绪。

### 5.3 断言/测试

- 集成测试以「首帧（未就绪）不出现 key 原文 / 不出现未翻译句」为断言（见 POC_REPORT PoC-4 与测试方案章节；由 Agent 3 细化）。

---

## 6. `src/locales/{en,zh-CN}/**` 组织 与 namespace 注册

### 6.1 目录组织

```
src/locales/
├── en/
│   ├── common.json      # 全局公共文案（A4 Wave1 骨架 → G1 后 A5）
│   ├── navigation.json
│   ├── auth.json
│   ├── models.json
│   ├── apiKeys.json
│   ├── usage.json
│   ├── cost.json
│   └── budgets.json
└── zh-CN/
    └── (同名 json，键集合与 en 一致)
```

- **namespace = 文件名 = key 冒号前缀**（§3.4：如 `common:action.delete` 对应 `common.json` 的 `action.delete`）。
- 英文与中文**键集合必须一致**（§3.4；用 CI 键集合 diff 校验，见 R9）。
- 中文按语义组织，不机械复制英文复数结构（D9）。

### 6.2 namespace 注册表 / 资源加载映射（唯一真源，Agent 4 永久 Owner）

`src/i18n/resources/registry.ts` 以数组/映射声明全部 namespace 及各 locale 的静态资源对象（`import common from "@/locales/en/common.json"`），供 `i18n.ts` 构造 `resources`。**往该注册表新增/删除 namespace 只有 Agent 4 可改**（FILE_OWNERSHIP 规则 6）。功能 Agent 新增业务 namespace 时在任务单填注册需求，由 A0 指派 A4 注册。

登记形式（设计目标）：

```ts
// src/i18n/resources/registry.ts
export const NAMESPACES = ["common", "navigation", "auth", "models", "apiKeys", "usage", "cost", "budgets"] as const;
export type Namespace = (typeof NAMESPACES)[number];
export const RESOURCES = {
  en: { common, navigation, auth, models, apiKeys, usage, cost, budgets },   // 惰性 import
  "zh-CN": { common, navigation, auth, models, apiKeys, usage, cost, budgets },
} as const satisfies Record<Locale, Record<Namespace, object>>;
```

`types.d.ts` 引用 `RESOURCES['en']` 作为 `CustomTypeOptions['resources']`，保证 `t('common:action.delete')` 有类型。

---

## 7. 后端 `UI settings.language` 调查结论（P1/P2）

本机只读调查（`litellm/` 后端代码，2026-09-09）：

- **`UISettings` 模型**（`litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py`）是代理级（管理员/全局）UI 配置，经 `/get/ui_settings`、`/update/ui_settings` 暴露，落在 `LiteLLM_UISettings` 表（`litellm/repositories/table_repositories.py:UISettingsRepository`，表名 `LiteLLM_UISettings` 见 `litellm/proxy/_types.py`）。
- 该模型有 `model_config = ConfigDict(extra="allow")`（可携带额外字段），但**持久化受 `ALLOWED_UI_SETTINGS_FIELDS` 白名单约束**，而该白名单**不包含 `language` 字段**。白名单现有字段如 `disable_model_add_for_internal_users`、`enabled_ui_pages_internal_users`、`enable_chat_ui` 等，均与语言无关。
- 即：**后端当前不存在可用的 `UI settings.language` 全局语言设置**。即便请求携带 `language`，也不在白名单内，不会被持久化/下发。

**结论：**
- **P2（该字段是否真实存在）**：**不存在**。`language` 不在 `UISettings` 白名单。
- **P1（管理员全局设置是否覆盖用户主动选择）**：**v1 不实现该覆盖**，因为字段不存在。优先级链退化见 §3.1。若未来后端新增该字段，需产品决策「管理员全局 vs 用户选择」的覆盖语义，再补实现（本设计已预留 L5 判定位）。
- 这同时**简化了 v1**：无需「用户偏好 vs 管理员全局」的合并/剥离逻辑，也避免在静态导出下引入「先取全局再取用户」的两段式异步。

---

## 8. 与既有 Provider / 路由组 layout 的接缝

- 根 `layout.tsx`：现为 `ThemeProvider → NuqsAdapter → ReactQueryProvider → AuthProvider`。`I18nProvider` 追加为**最外层**（§2.1）。**不移动**现有 Provider 顺序，避免影响 next-themes 首帧打 class 的既有行为。
- `(dashboard)/layout.tsx`：`use client` 路由组，另含 `SidebarProvider`、`ThemeContext`、`PluginModeContext` 等，均用到 `useAuth`。它们处于 `AuthProvider` 之下，`I18nProvider` 在最外层 → 这些组件可安全调用 `useTranslation`。**注意**：该 layout 自身是 `use client`，若其中出现硬编码文案需在 Wave 2 起中文化（A5/A6 职责）。
- `chat/layout.tsx`、`connect/layout.tsx` 独立于 `(dashboard)`，仍归根 layout 的 `I18nProvider` 管理，无需各自挂 Provider。
- **`<Trans>` 与 next-themes 的首帧 class 互不干扰**：`suppressHydrationWarning` 已覆盖 `<html>` 的未知属性（theme class / lang），本方案不加额外 suppress。

---

## 9. 依赖变更清单（仅供 Wave 1 Agent 4 执行，本设计不落地）

| 变更 | 目标值 | 写入者 |
|---|---|---|
| 新增依赖 | `i18next@^26.4.2`、`react-i18next@^17.0.13` | A4（D12） |
| 类型依赖 | 无需额外类型包（自带 TS 类型） | A4 |
| `next.config.mjs` | **不改**（静态导出已满足） | — |

---

## 附录 A：门禁对照

| 门禁 | 本设计的对应物 |
|---|---|
| G0（基线评审） | 本文档 §1–§8 供评审；D7 首屏策略已选定（§4.1） |
| G1（平台能力） | §6.2 注册表/资源映射/类型归 A4；PoC 清单（POC_REPORT）由 A4/A7 执行并回填 |
