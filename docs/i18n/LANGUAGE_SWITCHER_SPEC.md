# 语言切换器详细设计（Language Switcher Spec）

> 维护者：Agent 2（localization-designer）
> 状态：设计稿，待 G0 评审
> 关联：`I18N_MULTI_AGENT_PLAN.md`（§1.1/D5/D6）、`DECISIONS.md`、`LOCALIZATION_SPEC.md` §1、`LOCALE_NAVIGATION_BEHAVIOR.md`

本文档定义语言切换器的**入口位置、显示形式、状态、交互、持久化与可访问性**。它是供 Agent 5（壳层开发）实现的交互契约。

---

## 1. 入口位置（建议）

v1 只有一个入口，放在**顶部导航栏（Navbar）右侧**，靠近用户/账号菜单，以便登录后各页都能发现。

- 首选位置：`navbar.tsx` 右端、用户菜单图标左侧，作为独立图标按钮。
- 备选：若 Navbar 空间受限（静态导出、窄屏），回退到 `SidebarAccountMenu` 账号弹层内放置文字切换。
- **登录/引导页（Login、Onboarding、Connect、MCP OAuth）也必须有入口**，因为用户可能在这些页面首次选择语言。建议在这些页面顶部/右上角放置同一图标按钮（复用同一组件），与 Dashboard 主体入口共用组件逻辑。

> 实现要求：切换器组件做成可复用的独立组件（如 `src/components/LanguageSwitcher.tsx`），在 Navbar/账号菜单/登录页引用，避免多份复制。

## 2. 显示文案

- 按钮仅显示图标 + 当前语言的**目标语言缩写**，用于明确"点一下会变成什么"，同时按钮自身可在弹层中切换。
- 展示为 **「中文」/「EN」**：
  - 当前为 `en` 时，按钮显示 **中文**（提示可切到中文）。
  - 当前为 `zh-CN` 时，按钮显示 **EN**（提示可切到英文）。
- 展开菜单（Popover/Menu）时，提供两行完整选项：`English` 与 `简体中文（zh-CN）`，当前语言行加选中态（勾选/高亮）。
- 若空间允许，可在按钮前显示一个地球图标（`lucide` `Globe`/`Languages`），并在展开菜单用图标区分。

### 2.1 为什么按钮显示"目标语言"

图标 + 目标语言缩写最节省空间且语义清晰（常见国际化 UI 惯例，如 GitHub/Google）。同时菜单内提供明确的"当前语言"选中态，避免歧义。

### 2.2 空态/兜底

- 若某些组件在资源未就绪时不渲染切换器，则**优先不显示**而不是显示错误语言；资源就绪后由统一 ready 门禁放行（D7）。
- 切换器自身文案（`aria-label`、菜单项"English / 简体中文"）用当前语言渲染，保证用户总能用自己当前看到的语言操作它。

## 3. 状态

| 状态 | 表现 |
|---|---|
| 当前 `en` | 按钮显示"中文"；菜单中 `English` 高亮为选中 |
| 当前 `zh-CN` | 按钮显示"EN"；菜单中 `简体中文（zh-CN）` 高亮为选中 |
| 加载中/资源未就绪 | 按钮不闪烁、不显示 key；就绪后出现 |
| 存储不可用（隐私/禁 cookie） | 会话内仍可切换（见 `LOCALE_NAVIGATION_BEHAVIOR.md`），但刷新后回退 |

## 4. 交互

1. 点击按钮打开菜单（Popover/Menu）。
2. 选择目标语言项：
   - **即时生效**：当前页所有文案立即切换，无需刷新或整页跳转（`LOCALIZATION_SPEC.md` §1.3）。
   - 同时更新 `<html lang>`（`lang="zh-CN"` / `lang="en"`）。
   - 关闭菜单。
3. 切换**不**重置表单、导航展开、滚动位置；不触发额外请求。

## 5. 持久化行为

- 切换成功后**同步写入**：
  1. **localStorage**：键 `litellm.locale`（约定，见下），值 `en` | `zh-CN`。
  2. **SameSite cookie**：同名 cookie，`SameSite=Lax`、`path=/`，一个较长的有效期（如 365 天），用于跨整页跳转/新标签恢复；不设为 httpOnly（前端需读）。
- 存储键建议统一为常量 `litellm.locale`，由 Agent 4 在 `src/i18n/**` 提供读写函数，Agent 5/6 不直接操作存储。
- **优先级**：读取时优先 user 主动选择 → cookie/localStorage → 浏览器语言 → en（见 `LOCALIZATION_SPEC.md` §1.1）。cookie 与 localStorage 两者都存在时以较新写入者为准（实现按 D5/D6 由 Agent 4 定，建议以 localStorage 为准，cookie 用于跨页恢复）。
- 写入失败（隐私模式）不报错、不阻塞切换，静默降级。

## 6. 可访问性

- 触发器按钮必须带 `aria-label`，内容随当前语言本地化：
  - `en` 下：`Change language` / `选择语言`（用当前语言渲染，即 en 时英文标签，zh 时中文标签）。
  - zh 下：`切换语言`。
  - 推荐：`aria-label="Switch language"`（en）/ `"切换语言"`（zh-CN）。
- 菜单用 `role="menu"`/`menuitemradio` 或 Popover 语义，支持键盘操作（方向键选择、Enter 确认、Esc 关闭）。
- 选中项要有 `aria-checked` 或视觉高亮 + `aria-current`。
- 焦点管理：打开时焦点移至菜单，关闭后返回触发器。

## 7. 测试要点（供 Agent 7 参考）

- 切换即时生效（DOM 文案变化且无整页刷新）。
- 刷新后保持（cookie/localStorage 恢复）。
- `<html lang>` 与当前语言一致。
- 切换不清空表单/不触发请求。
- 当前语言在菜单中正确高亮。
- aria-label 随语言变化。
- 存储不可用时降级行为（见 `LOCALE_NAVIGATION_BEHAVIOR.md`）。
