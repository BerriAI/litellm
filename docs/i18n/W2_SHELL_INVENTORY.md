# W2 Shell & Auth 文案盘点清单（Wave 1 只读产出）

> 作者：Agent 5（`i18n-shell-auth-developer`）
> 阶段：Wave 1 只读盘点（G1 前未创建/修改任何 `src/**`、`src/locales/**`、`package*.json`、`layout.tsx`）
> 范围：全局壳层（Leftnav/Navbar/ThemeToggle/SidebarAccountMenu/SidebarUsageCard）+ 认证引导（Login/Onboarding/Connect/MCP OAuth）
> 依据：`LOCALIZATION_SPEC.md`（§3.4 key 规范、§4 不可翻译、§7 a11y）、`GLOSSARY_EN_ZH.md`、`V1_TRANSLATION_SCOPE.md`、`I18N_MULTI_AGENT_PLAN.md` §3.4/§5.6/§6
> 归属 namespace：`navigation`（导航/壳层）、`auth`（登录/引导/连接）、`common`（通用动作/复用 key）

## 0. 环境说明（重要接缝）

- 本盘点在**干净主工作区**（`ui/litellm-dashboard`）进行，且 **`src/locales/{en,zh-CN}` namespace 骨架在本次盘点工作区中并不存在**（属 Wave 1 Agent 4 交付物，尚未合入主工作区）。
- 因此"是否已存在(common?)"一栏：`common:action.*` / `common:actions.*` 等 key 依据 `LOCALIZATION_SPEC.md` §2.2、`GLOSSARY_EN_ZH.md` §3 与方案 §6 所述骨架命名**推断存在**；**本次盘点未能对 `common.json` 物理校验**。Wave 2 动手前必须由 Agent 5 在拿到移交后的 `common.json` 时逐条核对真实 key（本清单提供备选，供核准）。
- key 命名严格遵循 §3.4：`<namespace>:<domain>.<page>.<element>[.<state>]`；语义 key，不用整句当 key。
- 大量 key 同时用于可见文本与 a11y（aria-label/title/placeholder），二者同 key（§7）。

## 1. 关键接缝风险（先读这节）

| # | 风险 | 位置 | 说明 / 处理 |
|---|---|---|---|
| R1 | **`menuGroups` 是导出共享配置** | `leftnav.tsx`（119-363 行）被 `page_utils.ts`(getAvailablePages) 与 `getBreadcrumb`(leftnav 内部) 消费 | `groupLabel`/`label` 中文化必须**保留导出结构与 key/page/route/roles 字段不变**，仅把 label 换成 i18n key 引用。`page_utils.ts` 用 `item.label`/`group.groupLabel` 做字符串拼接（`${group.groupLabel} > ${parentLabel}`），且 `page_metadata.ts` 存 `pageDescriptions` 英文。Wave 2 需为 `page_utils`/`page_metadata` 也 I18N 化（否则 UI Settings 可见性列表仍显英文）。**这是跨模块共享 key 的核心接缝。** |
| R2 | **面包屑与导航共用同一套 key** | `getBreadcrumb`（leftnav 408-419 行）→ `DashboardHeader.tsx`(`title`) | 面包屑 title 必须与导航 label 同一 key，避免两处不一致（§3.1）。`SECTION_DISPLAY` 常量（391-397 行）已把大写分组映射为显示名，需一并 I18N 化。 |
| R3 | **同一动作词在多个组件重复**（Logout/Connect/Disconnect/Beta 等） | 见 §3/§6 | 必须统一走 `common`/同 namespace 单 key，禁止各组件造同义 key。 |
| R4 | **登录/SSO/OAuth 整页跳转语言恢复** | login、mcp/oauth/callback | 规范 §1.4：跳转返回需从同源 cookie/localStorage 恢复语言。属行为不改（仅文案），Wave 2 只换文案，不碰跳转/校验/路由逻辑。 |
| R5 | **Beta/New/vX.Y.Z 徽标与品牌不译** | 各处 | `Beta`、`LiteLLM`、`v{version}`、`AUTO_REDIRECT_UI_LOGIN_TO_SSO`、`DISABLE_ADMIN_UI` 等属 §4 不可翻译，**不进 key**。 |
| R6 | **Code/命令/环境变量/URL 不译** | login SSO 提示、default credentials | 代码片段、`MASTER_KEY`、`admin`、链接 URL 保持英文原文（§4 第 3/4/5 条）。 |

## 2. 盘点覆盖度

| 文件 | 状态 | 备注 |
|---|---|---|
| `src/components/leftnav.tsx` | 已全文盘点 | menuGroups + breadcrumb + a11y |
| `src/components/navbar.tsx`（顶层 Navbar） | 已全文盘点 | 仅 logo/badge/aria-label |
| `src/components/Navbar/BlogDropdown.tsx` | 已盘点 | 含错误/空态 |
| `src/components/Navbar/DocsLink.tsx` | 已盘点 | `Docs` |
| `src/components/Navbar/CommunityEngagementButtons.tsx` | 已盘点 | aria-label + tooltip |
| `src/components/Navbar/NotificationsBell.tsx` | 已盘点 | 弹窗正文 + aria-label |
| `src/components/Navbar/UserDropdown.tsx` | 已盘点 | 用户菜单 |
| `src/components/Navbar/WorkerDropdown.tsx` | 已盘点 | aria-label + empty |
| `src/components/Navbar/ViewSwitcher.tsx` | 已盘点 | AI Gateway/Chat 切换 |
| `src/components/Navbar/navDisplayName.ts` | 已盘点 | `Account` 回退文案 |
| `src/components/ThemeToggle/ThemeToggle.tsx` | 已盘点 | aria-label/title |
| `src/components/SidebarAccountMenu/SidebarAccountMenu.tsx` | 已盘点 | 用户菜单（与 UserDropdown 高度重复） |
| `src/components/SidebarUsageCard.tsx` | 已盘点 | Enterprise usage 卡 |
| `src/components/page_utils.ts` / `page_metadata.ts` | 已盘点（消费方） | R1 接缝 |
| `src/app/login/LoginPage.tsx` + `page.tsx` | 已全文盘点 | 登录表单/SSO/禁用态 |
| `src/app/onboarding/*`（Form/FormBody/Loading/Error/page） | 已盘点 | 引导 |
| `src/app/connect/*` + `src/components/chat/{MCPAppsPanel,ConnectFlowBanner,MCPConnectPicker,MCPCredentialsTab}` | 已盘点 | Connect 页 |
| `src/app/mcp/oauth/callback/page.tsx` | 已盘点 | OAuth 回跳 UI |
| **不在 v1 范围（只读参考，不拟 key）** | `Chat`(chat UI)、`MCP Servers` 配置页、`Tools/Vector Stores`、`Skills/Agents/Guardrails` 等 | 属 v2+ 候选（方案 §5.7）。但 **leftnav 中的 `menuGroups` label 属全局壳层，即使指向 v2 页面也仍由 leftnav 统一 I18N**（仅渲染入口，非页面功能翻译）。 |

## 3. 全局壳层文案盘点

### 3.1 leftnav.tsx — menuGroups（组/项，含面包屑，共享配置）

分组 groupLabel（`SECTION_DISPLAY` 显示名，key 与导航一致，供面包屑 section 复用）：

| 文件 | 位置 | 原文 EN | 拟定 key | namespace | 是否已存在(common?) | 备注 |
|---|---|---|---|---|---|---|
| leftnav.tsx | 119 | AI GATEWAY → "AI Gateway" | `navigation:group.aiGateway` | navigation | 否 | 分组标签；页面大写、SECTION_DISPLAY 用显示名，key 共用一套 |
| leftnav.tsx | 197 | OBSERVABILITY → "Observability" | `navigation:group.observability` | navigation | 否 | §3.4 示例 key |
| leftnav.tsx | 229 | ACCESS CONTROL → "Access Control" | `navigation:group.accessControl` | navigation | 否 | 术语表 §2 "Access Control"→访问控制 |
| leftnav.tsx | 262 | DEVELOPER TOOLS → "Developer Tools" | `navigation:group.developerTools` | navigation | 否 | 术语表 §2 |
| leftnav.tsx | 320 | SETTINGS → "Settings" | `navigation:group.settings` | navigation | 否 | 术语表 Settings→设置 |

AI GATEWAY 组 items：

| leftnav | 121 | Virtual Keys | `navigation:item.virtualKeys` | navigation | 否 | 术语表：Virtual Key→虚拟密钥 |
| leftnav | 126 | Playground | `navigation:item.playground` | navigation | 否 | v1 低优先级，**建议保留英文**（术语表备注）；key 仍拟好备用 |
| leftnav | 133 | Models + Endpoints | `navigation:item.modelsAndEndpoints` | navigation | 否 | 术语表：模型与端点 |
| leftnav | 142 | Agentic | `navigation:item.agentic` | navigation | 否 | Agent——v1 低优先级；见术语表 |
| leftnav | 146 | Agents | `navigation:item.agents` | navigation | 否 | v2+ 页面，但入口由壳层统一渲染 |
| leftnav | 154 | Workflow Runs | `navigation:item.workflowRuns` | navigation | 否 | 术语表 §2 |
| leftnav | 161 | Memory | `navigation:item.memory` | navigation | 否 | 术语表 §2：记忆 |
| leftnav | 167 | MCP Servers | `navigation:item.mcpServers` | navigation | 否 | "MCP"不译 |
| leftnav | 168 | Skills | `navigation:item.skills` | navigation | 否 | v2+，保留英文（术语表） |
| leftnav | 169 | Guardrails | `navigation:item.guardrails` | navigation | 否 | 产品专名不译（术语表），key 备用 |
| leftnav | 174 | Policies | `navigation:item.policies` | navigation | 否 | v2+ |
| leftnav | 179 | Tools | `navigation:item.tools` | navigation | 否 | v2+ |
| leftnav | 183 | Search Tools | `navigation:item.searchTools` | navigation | 否 | 术语表 §2：搜索工具 |
| leftnav | 184 | Vector Stores | `navigation:item.vectorStores` | navigation | 否 | v2+，向量存储 |
| leftnav | 188 | Tool Policies | `navigation:item.toolPolicies` | navigation | 否 | 工具策略 |

OBSERVABILITY / ACCESS CONTROL / DEVELOPER TOOLS / SETTINGS 组 items：

| leftnav | 205 | Usage | `navigation:item.usage` | navigation | 否 | 术语表：用量 |
| leftnav | 214 | Cost Optimization | `navigation:item.costOptimization` | navigation | 否 | 术语表：成本优化；含 `<BetaBadge/>`（不译，见 R5） |
| leftnav | 218 | Logs | `navigation:item.logs` | navigation | 否 | 日志（v2+ 页面，入口照译） |
| leftnav | 222 | Guardrails Monitor | `navigation:item.guardrailsMonitor` | navigation | 否 | Guardrails 专名不译 |
| leftnav | 231 | Teams | `navigation:item.teams` | navigation | 否 | §3.4 示例 key；v2+ 页面入口 |
| leftnav | 235 | Projects | `navigation:item.projects` | navigation | 否 | 含 BetaBadge（不译） |
| leftnav | 243 | Internal Users | `navigation:item.internalUsers` | navigation | 否 | 术语表：内部用户 |
| leftnav | 247 | Organizations | `navigation:item.organizations` | navigation | 否 | 组织 |
| leftnav | 253 | Access Groups | `navigation:item.accessGroups` | navigation | 否 | 访问组 |
| leftnav | 258 | Budgets | `navigation:item.budgets` | navigation | 否 | 预算 |
| leftnav | 264 | API Reference | `navigation:item.apiReference` | navigation | 否 | 术语表 §2：API 参考 |
| leftnav | 265 | AI Hub | `navigation:item.aiHub` | navigation | 否 | 品牌/产品名，可保留（建议按术语表复核） |
| leftnav | 270 | Learning Resources | `navigation:item.learningResources` | navigation | 否 | 术语表 §2：学习资料 |
| leftnav | 276 | Response Cache | `navigation:item.responseCache` | navigation | 否 | 术语表 §2：响应缓存 |
| leftnav | 284 | Experimental | `navigation:item.experimental` | navigation | 否 | 术语表 §2：实验功能 |
| leftnav | 289 | Prompts | `navigation:item.prompts` | navigation | 否 | 保留英文（术语表） |
| leftnav | 297 | API Playground | `navigation:item.apiPlayground` | navigation | 否 | 建议保留"API Playground" |
| leftnav | 303 | Tag Management | `navigation:item.tagManagement` | navigation | 否 | 术语表 §2：标签管理 |
| leftnav | 311 | Old Usage | `navigation:item.oldUsage` | navigation | 否 | 术语表 §2：旧版用量 |
| leftnav | 326 | Settings | `navigation:item.settings` | navigation | 否 | 设置 |
| leftnav | 333 | Router Settings | `navigation:item.routerSettings` | navigation | 否 | 术语表：网关设置 |
| leftnav | 340 | Logging & Alerts | `navigation:item.loggingAndAlerts` | navigation | 否 | 日志与告警 |
| leftnav | 347 | Admin Settings | `navigation:item.adminSettings` | navigation | 否 | 管理设置 |
| leftnav | 354 | Cost Tracking | `navigation:item.costTracking` | navigation | 否 | 术语表：成本跟踪 |
| leftnav | 358 | UI Theme | `navigation:item.uiTheme` | navigation | 否 | 术语表：界面主题 |

leftnav a11y / 其余：

| leftnav | 606 | aria-label="LiteLLM home" | `navigation:brand.homeAriaLabel` | navigation | 否 | a11y；"LiteLLM"品牌不译，仅"home"语义 |
| leftnav | 607 | alt="LiteLLM" | （品牌，不译） | — | — | §4 品牌不译 |
| leftnav | 631 | aria-label "Expand sidebar"/"Collapse sidebar" | `navigation:sidebar.expand` / `navigation:sidebar.collapse` | navigation | 否 | §7 示例；navbar 同文案（R3） |
| leftnav | 622 | v{version} | （不译） | — | — | §4 `v{major.minor}` |

### 3.2 navbar.tsx（顶层）

| navbar | 77 | title "Expand sidebar"/"Collapse sidebar" | `navigation:sidebar.expand` / `navigation:sidebar.collapse` | navigation | 否 | 与 leftnav 同 key（R3） |
| navbar | 93 | alt="LiteLLM Brand" | （品牌，不译） | — | — | §4 |
| navbar | 109 | title="Thanks for using LiteLLM!" | `navigation:navbar.thanksMessage` | navigation | 否 | 提示 title |
| navbar | 142 | aria-label="Product documentation" | `navigation:navbar.productDocsAria` | navigation | 否 | a11y |

### 3.3 Navbar 子组件

**BlogDropdown.tsx**（Blog 菜单）：

| BlogDropdown | 触发按钮 | Blog | `navigation:blog.title` | navigation | 否 | 下拉触发器可见文本 |
| BlogDropdown | loading aria | aria-label="loading" | `common:aria.loading` | common | 否 | a11y；多组件复用（R3），可放 common |
| BlogDropdown | 错误态 | Failed to load posts | `navigation:blog.loadError` | navigation | 否 | toast/错误 |
| BlogDropdown | 错误态按钮 | Retry | `common:action.retry` | common | 拟共存（common:actions/action） | §2.2 通用动作；Wave 2 核对现有 key（retry 并入 common） |
| BlogDropdown | 空态 | No posts available | `navigation:blog.empty` | navigation | 否 | 空态 |
| BlogDropdown | 底部链接 | View all posts | `navigation:blog.viewAll` | navigation | 否 | |
| @formatDate | — | (toLocaleDateString "en-US" 硬编码) | `navigation:blog.date` 区域化 | navigation | 否 | **接缝**：硬编码 `en-US` locale，Wave 2 需 `t()` 区域化（R6/i18n 格式化） |

**DocsLink.tsx**：

| DocsLink | 可见文本 | Docs | `navigation:docs.title` | navigation | 否 | 与 Blog 对应 |

**CommunityEngagementButtons.tsx**（aria-label + tooltip，§7 同翻）：

| CommunityEngagementButtons | aria-label/tooltip | Join Slack | `navigation:community.joinSlack` | navigation | 否 | 品牌 Slack 不译主体，"加入 Slack" |
| CommunityEngagementButtons | aria-label/tooltip | LiteLLM on GitHub | `navigation:community.github` | navigation | 否 | 品牌 LiteLLM 不译 |
| CommunityEngagementButtons | ButtonGroup aria | Community links | `navigation:community.groupAria` | navigation | 否 | a11y |

**NotificationsBell.tsx**（弹窗正文 + a11y）：

| NotificationsBell | popover aria | aria-label="Notifications" | `navigation:notifications.triggerAria` | navigation | 否 | a11y |
| NotificationsBell | PopoverTitle | LiteLLM Auto Router | `navigation:notifications.autoRouterTitle` | navigation | 否 | 产品名 LiteLLM 不译；"Auto Router"可译"自动路由" |
| NotificationsBell | PopoverDescription | Route every request to the cheapest model... | `navigation:notifications.autoRouterBody` | navigation | 否 | 说明文本 |
| NotificationsBell | 按钮 | Read the docs | `navigation:notifications.readDocs` | navigation | 否 | |
| NotificationsBell | 按钮 | Mark as read | `navigation:notifications.markAsRead` | navigation | 否 | |

**UserDropdown.tsx**（navbar 与 sidebar 变体共用）：

| UserDropdown | 触发 aria | `Account menu — {role} — signed in as {email}` | `auth:account.triggerAria` | auth | 否 | 插值 `{{role}}`/`{{email}}`；模板字符串 → 语义 key+插值 |
| UserDropdown | aria | (同上，SidebarAccountMenu 变体) | `auth:account.triggerAria` | auth | 否 | 两处共用（R3） |
| UserDropdown | 徽标 | Premium / Standard | `auth:account.tier.premium` / `auth:account.tier.standard` | auth | 否 | 订阅等级 |
| UserDropdown | tooltip | Upgrade to Premium for advanced features | `auth:account.tier.upgradeTooltip` | auth | 否 | |
| UserDropdown | 行 | User ID | `auth:account.userId` | auth | 否 | |
| UserDropdown | 行 | Role | `auth:account.role` | auth | 否 | |
| UserDropdown | copy | Copy User ID | `common:action.copyUserId` | common | 拟存 | 复制类（R3）；Wave 2 核对 |
| UserDropdown | 开关 | Hide New Feature Indicators | `auth:account.toggle.hideNew` | auth | 否 | |
| UserDropdown | aria | Toggle hide new feature indicators | `auth:account.toggle.hideNewAria` | auth | 否 | a11y |
| UserDropdown | 开关 | Hide All Prompts | `auth:account.toggle.hidePrompts` | auth | 否 | |
| UserDropdown | aria | Toggle hide all prompts | `auth:account.toggle.hidePromptsAria` | auth | 否 | |
| UserDropdown | 开关 | Hide Blog Posts | `auth:account.toggle.hideBlog` | auth | 否 | |
| UserDropdown | aria | Toggle hide blog posts | `auth:account.toggle.hideBlogAria` | auth | 否 | |
| UserDropdown | 开关 | Hide Bouncing Icon | `auth:account.toggle.hideBouncing` | auth | 否 | |
| UserDropdown | aria | Toggle hide bouncing icon | `auth:account.toggle.hideBouncingAria` | auth | 否 | |
| UserDropdown | 按钮 | Logout | `auth:account.logout` | auth | 否 | 术语表 §3：登出 |

**WorkerDropdown.tsx**：

| WorkerDropdown | ComboboxInput aria | aria-label="Worker" | `navigation:worker.ariaLabel` | navigation | 否 | a11y |
| WorkerDropdown | 空态 | No matching workers | `navigation:worker.empty` | navigation | 否 | |

**ViewSwitcher.tsx**（AI Gateway / Chat 切换）：

| ViewSwitcher | 默认/项 | AI Gateway | `navigation:view.aiGateway` | navigation | 否 | 术语表 §2：AI 网关 |
| ViewSwitcher | 项/active | Chat | `navigation:view.chat` | navigation | 否 | |
| ViewSwitcher | disabled 说明 | Admins can enable in Settings | `navigation:view.chatDisabled` | navigation | 否 | |

**navDisplayName.ts**：

| navDisplayName | 回退值 | "Account" | `auth:account.fallbackName` | auth | 否 | 当无 email/id 时的显示名 |

### 3.4 ThemeToggle / SidebarAccountMenu / SidebarUsageCard

**ThemeToggle.tsx**：

| ThemeToggle | aria-label/title | Switch to dark mode (beta) | `navigation:theme.toDark` | navigation | 否 | §7；"(beta)"保留或并入 |
| ThemeToggle | aria-label/title | Switch to light mode | `navigation:theme.toLight` | navigation | 否 | |

**SidebarAccountMenu.tsx**（用户菜单，与 UserDropdown 高度重复 → 共用 key，R3）：

| SidebarAccountMenu | 头部 | LiteLLM | （品牌不译） | — | — | §4 |
| SidebarAccountMenu | title | Thanks for using LiteLLM! | `navigation:navbar.thanksMessage` | navigation | 否 | 与 navbar 同 key（R3） |
| SidebarAccountMenu | 触发 aria | Account menu — ... | `auth:account.triggerAria` | auth | 否 | 与 UserDropdown 同 key |
| SidebarAccountMenu | 行 | Tier | `auth:account.tier.label` | auth | 否 | |
| SidebarAccountMenu | 徽标 | Premium / Standard | `auth:account.tier.premium` / `auth:account.tier.standard` | auth | 否 | 同上 |
| SidebarAccountMenu | title | Upgrade to Premium for advanced features | `auth:account.tier.upgradeTooltip` | auth | 否 | |
| SidebarAccountMenu | 行 | Role | `auth:account.role` | auth | 否 | |
| SidebarAccountMenu | 行 | Email | `auth:account.email` | auth | 否 | |
| SidebarAccountMenu | 行 | User ID | `auth:account.userId` | auth | 否 | 与 UserDropdown 同 key |
| SidebarAccountMenu | copy | Copy email / Copy user ID | `common:action.copyEmail` / `common:action.copyUserId` | common | 拟存 | |
| SidebarAccountMenu | toggle×4 | Hide New Feature Indicators / Hide All Prompts / Hide Blog Posts / Hide Bouncing Icon | `auth:account.toggle.*` | auth | 否 | 与 UserDropdown 同 key |
| SidebarAccountMenu | toggle aria×4 | Toggle hide ... | `auth:account.toggle.*Aria` | auth | 否 | 与 UserDropdown 同 key |
| SidebarAccountMenu | 按钮 | Logout | `auth:account.logout` | auth | 否 | |

**SidebarUsageCard.tsx**（侧边栏用量卡）：

| SidebarUsageCard | collapsed title | Enterprise usage | `navigation:usageCard.title` | navigation | 否 | |
| SidebarUsageCard | 副标题 | Active plan | `navigation:usageCard.activePlan` | navigation | 否 | license 无过期日期时 |
| SidebarUsageCard | 标签 | Seats / Teams | `navigation:usageCard.seats` / `navigation:usageCard.teams` | navigation | 否 | 数据标签 |
| SidebarUsageCard | 加载 | Loading… | `common:loading.ellipsis` | common | 拟存 | §4 加载中 |
| SidebarUsageCard | aria | (aria-valuetext `{used} of {total}`) | `navigation:usageCard.valuetext` | navigation | 否 | a11y 插值 |

**page_utils.ts / page_metadata.ts（消费方，R1）**：

| page_utils | 51 | "No description available" | `common:error.noDescription` | common | 否 | 兜底文本 |
| page_metadata | — | `pageDescriptions` 英文说明 | （每页 description key） | navigation | 否 | **需 I18N 化**，否则 UI Settings 可见性列表显英文（R1）；提供 `navigation:pageDesc.*` 系列 |

## 4. 认证/引导/连接文案盘点（auth / navigation / common）

### 4.1 LoginPage.tsx（含 SSO / 禁用 / 表单）

| LoginPage | 校验 | Please enter your username | `auth:login.usernameRequired` | auth | 否 | 校验提示（zod message） |
| LoginPage | 校验 | Please enter your password | `auth:login.passwordRequired` | auth | 否 | |
| LoginPage | 标题 | 🚅 LiteLLM | （品牌不译） | — | — | |
| LoginPage | 副标题 | Login | `auth:login.title` | auth | 否 | |
| LoginPage | 副标题 | Access your LiteLLM Admin UI. | `auth:login.subtitle` | auth | 否 | "LiteLLM Admin UI"品牌不译 |
| LoginPage | alert | Default Credentials | `auth:login.defaultCredsTitle` | auth | 否 | |
| LoginPage | alert 说明 | By default, Username is `admin` and Password is ... | `auth:login.defaultCredsBody` | auth | 否 | `admin`/`MASTER_KEY` 代码值不译（§4） |
| LoginPage | 链接 | Check the documentation | `auth:login.checkDocs` | auth | 否 | |
| LoginPage | alert | Admin UI Disabled | `auth:login.adminDisabledTitle` | auth | 否 | |
| LoginPage | alert 说明 | The Admin UI has been disabled by the administrator... | `auth:login.adminDisabledBody` | auth | 否 | |
| LoginPage | alert | SSO 提示标题 | Single Sign-On (SSO) is enabled... | `auth:login.ssoNoticeTitle` | auth | 否 | 含 `AUTO_REDIRECT_UI_LOGIN_TO_SSO` 代码不译 |
| LoginPage | alert 关闭 aria | aria-label="Close" | `common:action.close` | common | 拟存 | §2.2 |
| LoginPage | 字段 | Worker | `auth:login.workerLabel` | auth | 否 | |
| LoginPage | placeholder | Choose a worker to connect to | `auth:login.workerPlaceholder` | auth | 否 | |
| LoginPage | 字段 | Username | `auth:login.usernameLabel` | auth | 否 | |
| LoginPage | placeholder | Enter your username | `auth:login.usernamePlaceholder` | auth | 否 | |
| LoginPage | 字段 | Password | `auth:login.passwordLabel` | auth | 否 | |
| LoginPage | placeholder | Enter your password | `auth:login.passwordPlaceholder` | auth | 否 | |
| LoginPage | 提交中 | Logging in... | `auth:login.submitting` | auth | 否 | |
| LoginPage | 按钮 | Login | `auth:login.submit` | auth | 否 | §3.4 示例 key |
| LoginPage | 按钮 | Login with SSO | `auth:login.ssoSubmit` | auth | 否 | |
| LoginPage | tooltip | Please configure SSO to log in with SSO. | `auth:login.ssoNotConfigured` | auth | 否 | |
| LoginPage | loading aria | aria-label="loading" | `common:aria.loading` | common | 否 | 与 BlogLoading 同 key（R3） |
| — | 后端 error | `loginMutation.error.message` | （后端消息不翻，§4） | — | — | 保留原文展示，不拟 key |

### 4.2 Onboarding（引导）

**OnboardingForm.tsx**：

| OnboardingForm | claim 失败 | Failed to start session. Please try again. | `auth:onboarding.startSessionError` | auth | 否 | |
| OnboardingForm | claim 失败 | Failed to submit. Please try again. | `auth:onboarding.submitError` | auth | 否 | |

**OnboardingFormBody.tsx**：

| OnboardingFormBody | 标题 | Reset Password / Sign Up | `auth:onboarding.action.resetPassword` / `auth:onboarding.action.signUp` | auth | 否 | 动作标签，两处按钮共用 |
| OnboardingFormBody | 标题 | 🚅 LiteLLM | （品牌不译） | — | — | |
| OnboardingFormBody | 说明(reset) | Reset your password to access Admin UI. | `auth:onboarding.resetDesc` | auth | 否 | |
| OnboardingFormBody | 说明(signup) | Claim your user account to login to Admin UI. | `auth:onboarding.signupDesc` | auth | 否 | |
| OnboardingFormBody | alert | SSO | `auth:onboarding.ssoTitle` | auth | 否 | |
| OnboardingFormBody | alert | SSO is under the Enterprise Tier. | `auth:onboarding.ssoBody` | auth | 否 | |
| OnboardingFormBody | 按钮 | Get Free Trial | `auth:onboarding.freeTrial` | auth | 否 | |
| OnboardingFormBody | 字段 | Email Address | `auth:onboarding.emailLabel` | auth | 否 | |
| OnboardingFormBody | 字段 | Password | `auth:onboarding.passwordLabel` | auth | 否 | |
| OnboardingFormBody | 说明(reset) | Enter your new password | `auth:onboarding.passwordResetDesc` | auth | 否 | |
| OnboardingFormBody | 说明(signup) | Create a password for your account | `auth:onboarding.passwordSignupDesc` | auth | 否 | |
| OnboardingFormBody | 校验 | password required to sign up | `auth:onboarding.passwordRequired` | auth | 否 | |
| OnboardingFormBody | loading aria | aria-label="loading" | `common:aria.loading` | common | 否 | |

**OnboardingLoadingView / OnboardingErrorView / page**：

| OnboardingLoadingView | aria | Loading invitation | `auth:onboarding.loadingAria` | auth | 否 | a11y |
| OnboardingErrorView | 标题 | Failed to load invitation | `auth:onboarding.loadError` | auth | 否 | |
| OnboardingErrorView | 说明 | The invitation link may be invalid or expired. | `auth:onboarding.loadErrorDesc` | auth | 否 | |
| OnboardingErrorView | 链接 | Back to Login | `auth:onboarding.backToLogin` | auth | 否 | 术语表 Back→返回 |
| page.tsx (onboarding) | Suspense | Loading... | `common:loading.ellipsis` | common | 拟存 | 与 SidebarUsageCard 同 key |

### 4.3 Connect 页与 MCP OAuth

**connect/page.tsx**：无硬编码文案（业务状态由子组件承载），不拟 key。

**ConnectFlowBanner.tsx**：

| ConnectFlowBanner | 标题 | Connect your MCP servers to {{client}} | `auth:connectFlow.title` | auth | 否 | 插值 `{{client}}`（clientLabel） |
| ConnectFlowBanner | 说明 | Authorize the servers you want to use below, then click Finish connecting... | `auth:connectFlow.subtitle` | auth | 否 | |
| ConnectFlowBanner | 按钮 | Finish connecting | `auth:connectFlow.finish` | auth | 否 | 关键 consent 动作文案 |
| ConnectFlowBanner | 标签 | My client is on a remote or SSH machine | `auth:connectFlow.remoteClient` | auth | 否 | |

**MCPAppsPanel.tsx**（Connect 应用网格）：

| MCPAppsPanel | 按钮 | Connect / Connecting… / Disconnect | `auth:connectFlow.connect` / `auth:connectFlow.connecting` / `auth:connectFlow.disconnect` | auth | 否 | R3：与 MCPCredentialsTab 共用 |
| MCPAppsPanel | 徽标 | Not supported on this connection | `auth:connectFlow.unsupported` | auth | 否 | |
| MCPAppsPanel | 标题 | MCP Servers | `navigation:item.mcpServers` | navigation | 否 | 与 leftnav 同 key（R3/R1） |
| MCPAppsPanel | 说明 | Click a server to see its tools and connect | `auth:connectFlow.listHint` | auth | 否 | |
| MCPAppsPanel | 说明 | Browse tools, authenticate once, use in chat | `auth:connectFlow.browseHint` | auth | 否 | |
| MCPAppsPanel | 加载 | Loading tools... | `auth:connectFlow.loadingTools` | auth | 否 | |
| MCPAppsPanel | 计数 | {{count}} tool(s) available | `auth:connectFlow.toolsCount` | auth | 否 | 复数插值（zh 用 `{{count}} 个工具`） |
| MCPAppsPanel | placeholder | Search servers... | `auth:connectFlow.searchPlaceholder` | auth | 否 | |
| MCPAppsPanel | tab | All | `common:filter.all` | common | 拟存 | §4 表单词 All→全部；核对 common |
| MCPAppsPanel | tab | Connected (n) | `auth:connectFlow.tab.connected` | auth | 否 | 插值计数 |
| MCPAppsPanel | 空态 | No MCP servers are available to this connection yet... | `auth:connectFlow.emptyConnectMode` | auth | 否 | |
| MCPAppsPanel | 空态 | No MCP servers configured. Add servers in Tools -> MCP Servers. | `auth:connectFlow.emptyNoServer` | auth | 否 | |
| MCPAppsPanel | 空态 | No servers connected yet. | `auth:connectFlow.emptyConnected` | auth | 否 | |
| MCPAppsPanel | 空态 | No servers match your search. | `auth:connectFlow.emptySearch` | auth | 否 | |
| MCPAppsPanel | 返回 | Back | `common:action.back` | common | 拟存 | 术语表 Back→返回 |
| MCPAppsPanel | 兜底 | MCP server | `auth:connectFlow.serverFallback` | auth | 否 | description 缺省 |
| MCPAppsPanel | 详情标题 | Information | `auth:connectFlow.infoTitle` | auth | 否 | |
| MCPAppsPanel | 详情行 | Server ID / Transport / Status | `auth:connectFlow.detail.serverId` / `transport` / `status` | auth | 否 | |
| MCPAppsPanel | 状态值 | Connected / Not connected | `auth:connectFlow.status.connected` / `status.notConnected` | auth | 否 | |
| MCPAppsPanel | 标题 | Available Tools | `auth:connectFlow.toolsTitle` | auth | 否 | |
| MCPAppsPanel | 空态 | No tools available | `auth:connectFlow.noTools` | auth | 否 | |
| MCPAppsPanel | Beta | Beta | （不译，§4） | — | — | |
| MCPAppsPanel | toast | Could not load tools for {{server}} | `auth:connectFlow.toolsLoadError` | auth | 否 | 插值 |

**MCPConnectPicker.tsx**：

| MCPConnectPicker | 空态 | No MCP servers configured | `auth:connectFlow.emptyNoServer` | auth | 否 | 与 MCPAppsPanel 同 key（R3） |
| MCPConnectPicker | toast | Could not load tools for {{server}} — it will be excluded... | `auth:connectFlow.toolsLoadErrorExcluded` | auth | 否 | 插值，含 `—` 说明 |

**MCPCredentialsTab.tsx**（App 凭据表）：

| MCPCredentialsTab | 标题 | App Credentials | `auth:credentials.title` | auth | 否 | |
| MCPCredentialsTab | 说明 | Your stored OAuth connections; used automatically in chat | `auth:credentials.subtitle` | auth | 否 | |
| MCPCredentialsTab | 列头 | App / Connected / Status / Actions | `auth:credentials.col.app` / `col.connected` / `col.status` / `col.actions` | auth | 否 | |
| MCPCredentialsTab | 空态 | No connections yet | `auth:credentials.empty` | auth | 否 | |
| MCPCredentialsTab | 空态说明 | Go to Integrations and click Connect to authorize an MCP server | `auth:credentials.emptyHint` | auth | 否 | 含"Integrations"/"Connect"强调 |
| MCPCredentialsTab | 时间 | just now | `auth:credentials.justNow` | auth | 否 | |
| MCPCredentialsTab | 状态 | Does not expire / Expired / Expires in {{n}}d/h/m | `auth:credentials.expiry.*` | auth | 否 | 插值 |
| MCPCredentialsTab | title | Revoke connection | `auth:credentials.revokeTitle` | auth | 否 | a11y title |
| MCPCredentialsTab | 弹窗标题 | Revoke connection? | `auth:credentials.revokeDialogTitle` | auth | 否 | |
| MCPCredentialsTab | 弹窗说明 | This removes the stored OAuth credential for {{name}}... | `auth:credentials.revokeDialogBody` | auth | 否 | 插值 |
| MCPCredentialsTab | 按钮 | Cancel / Revoke | `common:action.cancel` / `auth:credentials.revokeConfirm` | common/auth | 拟存/新增 | Cancel 用 common（§2.2） |
| MCPCredentialsTab | toast | Failed to revoke connection. Please try again. | `auth:credentials.revokeError` | auth | 否 | |

**mcp/oauth/callback/page.tsx**：

| oauth callback | 标题 | LiteLLM MCP OAuth | `navigation:mcpOAuth.title` | navigation | 否 | 回调中间页（§1.4 整页跳转语言恢复关切） |
| oauth callback | 说明 | Authorization complete. You may close this window and return to the LiteLLM dashboard. | `navigation:mcpOAuth.complete` | navigation | 否 | |
| oauth callback | 说明 | If the window does not close automatically, everything is still saved—you can close it manually. | `navigation:mcpOAuth.manualClose` | navigation | 否 | |
| oauth callback | Suspense | Loading... | `common:loading.ellipsis` | common | 拟存 | |

## 5. 复用既有 common key 清单（Wave 2 需逐条核对 `common.json` 真实 key）

> 下列 key 依据规范 §2.2 / 术语表 §3 推断骨架已提供（`common:action.*`、`common:actions.*` 命名差异需以实际骨架为准）。**复用，不重复造；若骨架 key 命名不同（e.g. `actions.save` vs `action.save`），以骨架为准并同步本表。** 新增通用 key 须 Agent 0 Review（方案 §6）。

| 复用 key（拟定） | UI 位 | 说明 |
|---|---|---|
| `common:action.close` | LoginPage alert 关闭 aria | Close |
| `common:action.cancel` | MCPCredentialsTab Cancel | Cancel |
| `common:action.back` | MCPAppsPanel Back | Back（可能新增，视 common 是否含 back） |
| `common:action.retry` | BlogDropdown Retry | Retry |
| `common:action.copy` / `copyEmail` / `copyUserId` | UserDropdown / SidebarAccountMenu | Copy 类（核对 common 是否已含 copy） |
| `common:aria.loading` | Login / Onboarding / Blog loading aria | 多组件复用 |
| `common:loading.ellipsis` | SidebarUsageCard / onboarding / mcp callback | "Loading…" |
| `common:filter.all` | MCPAppsPanel tab "All" | All→全部 |
| `common:error.noDescription` | page_utils 兜底 | 可新增 |

## 6. 需新增到 `navigation` / `auth` 的 key 汇总（供 Wave 2 使用）

### navigation namespace（导航/壳层）

- 分组 key（5）：`navigation:group.{aiGateway,observability,accessControl,developerTools,settings}`
- 菜单项 label key（~40）：`navigation:item.{virtualKeys,playground,modelsAndEndpoints,agentic,agents,workflowRuns,memory,mcpServers,skills,guardrails,policies,tools,searchTools,vectorStores,toolPolicies,usage,costOptimization,logs,guardrailsMonitor,teams,projects,internalUsers,organizations,accessGroups,budgets,apiReference,aiHub,learningResources,responseCache,experimental,prompts,apiPlayground,tagManagement,oldUsage,settings,routerSettings,loggingAndAlerts,adminSettings,costTracking,uiTheme}`
- 面包屑/共享：`navigation:group.*` 复用；`navigation:pageDesc.<page>` 系列（page_metadata 说明）
- 侧栏/导航 a11y：`navigation:sidebar.expand/collapse`、`navigation:brand.homeAriaLabel`、`navigation:navbar.thanksMessage`、`navigation:navbar.productDocsAria`
- Navbar 子组件：`navigation:blog.{title,loadError,empty,viewAll,date}`、`navigation:docs.title`、`navigation:community.{joinSlack,github,groupAria}`、`navigation:notifications.{triggerAria,autoRouterTitle,autoRouterBody,readDocs,markAsRead}`、`navigation:worker.{ariaLabel,empty}`、`navigation:view.{aiGateway,chat,chatDisabled}`、`navigation:theme.{toDark,toLight}`、`navigation:usageCard.{title,activePlan,seats,teams,valuetext}`
- MCP OAuth 中间页：`navigation:mcpOAuth.{title,complete,manualClose}`

### auth namespace（登录/引导/连接）

- 登录：`auth:login.{usernameRequired,passwordRequired,title,subtitle,defaultCredsTitle,defaultCredsBody,checkDocs,adminDisabledTitle,adminDisabledBody,ssoNoticeTitle,workerLabel,workerPlaceholder,usernameLabel,usernamePlaceholder,passwordLabel,passwordPlaceholder,submitting,submit,ssoSubmit,ssoNotConfigured}`
- 账号菜单：`auth:account.{triggerAria,fallbackName,tier.label,tier.premium,tier.standard,tier.upgradeTooltip,userId,role,email,toggle.hideNew/hidePrompts/hideBlog/hideBouncing 及 *Aria,logout}`
- 引导：`auth:onboarding.{startSessionError,submitError,action.resetPassword,action.signUp,resetDesc,signupDesc,ssoTitle,ssoBody,freeTrial,emailLabel,passwordLabel,passwordResetDesc,passwordSignupDesc,passwordRequired,loadingAria,loadError,loadErrorDesc,backToLogin}`
- Connect：`auth:connectFlow.{title,subtitle,finish,remoteClient,connect,connecting,disconnect,unsupported,listHint,browseHint,loadingTools,toolsCount,searchPlaceholder,tab.connected,emptyConnectMode,emptyNoServer,emptyConnected,emptySearch,serverFallback,infoTitle,detail.serverId/transport/status,status.connected/notConnected,toolsTitle,noTools,toolsLoadError,toolsLoadErrorExcluded}`
- 凭据：`auth:credentials.{title,subtitle,col.app/connected/status/actions,empty,emptyHint,justNow,expiry.*,revokeTitle,revokeDialogTitle,revokeDialogBody,revokeConfirm,revokeError}`

## 7. 覆盖度结论

- **已完整盘点并拟 key**：leftnav、navbar、Navbar/**（Blog,Docs,Community,Notifications,UserDropdown,WorkerDropdown,ViewSwitcher,navDisplayName）、ThemeToggle、SidebarAccountMenu、SidebarUsageCard、page_utils/page_metadata（消费方）、login、onboarding 全部、connect（含 chat/MCPAppsPanel、ConnectFlowBanner、MCPConnectPicker、MCPCredentialsTab）、mcp/oauth/callback。
- **未盘点（v2+ 不在 v1 范围，仅入口经 leftnav 拟 key）**：MCP Servers 配置页、Tools/Vector Stores、Agents/Skills、Guardrails、Chat UI 等整模块（方案 §5.7）。其 leftnav 入口 label 已由壳层统一覆盖。
- **命名捷径**：`auth:connectFlow.*` / `auth:credentials.*` / `navigation:pageDesc.*` 均为新增，未与既有工程冲突（本次工作区无 `src/locales`，无法做字符串比对）。
