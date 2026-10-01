# LiteLLM Dashboard i18n — FILE_OWNERSHIP

> 维护者：Agent 0。同一文件同一时刻只有一个 Owner。规则见 `I18N_MULTI_AGENT_PLAN.md` §6。

## 目录级 Ownership

| 范围 | Owner | 备注 |
|---|---|---|
| `src/i18n/**`（Provider、初始化、locale 偏好、类型、**namespace 注册表/资源映射/类型声明**） | **Agent 4（永久）** | A5/6/7 不得自行修改注册代码 |
| `src/locales/{en,zh-CN}/**` 目录及最小骨架 | Wave 1 为 A4；G1 后按 namespace 移交 | A4 在 Wave 1 只建骨架 + smoke-test 最小资源 |
| `src/locales/{en,zh-CN}/common.json` + 公共 UI 文案 | G1 后为 A5 | A2 审核译文；新增通用 key 需 A0 Review |
| `navigation` namespace + 导航组件 | A5 | — |
| `auth` namespace + 登录引导组件 | A5 | — |
| `models`、`apiKeys` namespace + 页面 | 当期 A6 | Wave 2 |
| `usage`、`cost`、`budgets` namespace + 页面 | 当期 A6A/A6B | Wave 3 |
| 测试公共工具 + 质量报告 | A7 | — |
| 发布检查 + 汇总报告 | A8 | Wave 4 |
| 设计/文档（docs/i18n/**） | Agent 0 统筹，各 Agent 自写 | 文档归 docs/i18n/ |

## 关键规则

1. 同一文件同一时间仅一个 Owner。
2. 功能 Agent 只写自己的 namespace，不共同编辑一个大 JSON。
3. `src/components/ui/**` 原则上无业务文案，由调用方传入文本。
4. `src/utils/**` 不设全目录 Owner，按文件划分。
5. `package.json`/`package-lock.json`：默认只允许 **Agent 4** 在平台 worktree 修改；确需改依赖由 A0 指定唯一临时 Owner。
6. 新增业务 namespace：功能 Agent 在任务单填注册需求，由 A0 指派 A4 统一注册，或 A0 书面授权。

## 移交记录

| 时间 | 资源 | 由 → 到 | 确认 |
|---|---|---|---|
| （待 G1） | `src/locales/{en,zh-CN}/common|navigation|auth*` | Agent 4 → Agent 5 | A0 |
