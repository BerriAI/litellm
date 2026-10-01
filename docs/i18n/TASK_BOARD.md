# LiteLLM Dashboard i18n — TASK_BOARD（多智能体进展看板）

> 维护者：Agent 0。**每启动/交付/Review 一个 Agent 即更新本表**。这是你查看"谁在工作、进展到哪"的单点真相。
> 状态取值：待命 / 进行中 / 待 Review / 已完成 / 被阻塞

## Wave 0 — 并行设计（G0 门禁前）

| Agent | 名称 | 状态 | worktree/分支 | 交付物 | 最后更新 |
|---|---|---|---|---|---|
| A1 | i18n-architect | **已完成 ✅（G0 Review 通过）** | （纯设计） | I18N_TECH_DESIGN / I18N_ADR / POC_REPORT / TECH_RISKS | 2026-09-09 G0 |
| A2 | localization-designer | **已完成 ✅（G0 Review 通过）** | （纯设计） | LOCALIZATION_SPEC / GLOSSARY_EN_ZH(63条) / LANGUAGE_SWITCHER_SPEC / V1_TRANSLATION_SCOPE / LOCALE_NAVIGATION_BEHAVIOR | 2026-09-09 G0 |
| A3 | qa-architect | **已完成 ✅（G0 Review 通过）** | （纯设计） | I18N_TEST_PLAN / TEST_CASES(26) / TEST_TOOLS / REGRESSION_MATRIX | 2026-09-09 G0 |

**G0 门禁结论（Agent 0，2026-09-09）：APPROVED**
- 首屏策略已选定「就绪门禁 + `<html lang>` 同步」（ADR-04，非"再评估"）。
- 后端无 `UI settings.language`（ADR-07 Accepted）。
- 构建期 `<title>`/meta 保持英文边界已记录（ADR-06 Accepted）。
- 整页跳转语言保持规则已写入本地化与测试方案（LOCALE_NAVIGATION_BEHAVIOR + TEST_CASES TC-16..18）。
- E2E 基建缺口已定：选 **a（Wave 1 补齐 Playwright）**，见 DECISIONS P3/P4。
- **G0 Review 发现 1 个需修正点（P5）**：语言偏好存储键名跨文档不一致（`dashboard.locale` vs `litellm.locale`）。**已定统一为 `litellm.locale`**，由 A4 以单一常量实现，A5/6 不直接操作存储。
- G0 Review 确认 POC_REPORT 9 项必验项已具备（待 Wave 1 A4/A7 回填证据）。

## Wave 1 — 平台能力与测试基础

| Agent | 名称 | 状态 | worktree/分支 | 交付物 | 最后更新 |
|---|---|---|---|---|---|
| A4 | i18n-platform-developer | **实现完成 ✅；build 验证进行中** | i18n/w1-agent4-platform | src/i18n/** + 8 namespace 骨架 + layout + LanguageSwitcher + Playwright基建 + 依赖写入 + 30单测/7集成测试通过 | 2026-09-09 |
| A5 | i18n-shell-auth-developer | **只读盘点完成 ✅** | —（只读） | W2_SHELL_INVENTORY.md（~170 key，25 文件） | 2026-09-09 |
| A7 | i18n-qa | **工具开发完成 ✅** | i18n/w1-agent7-qa | scripts/i18n/check-keys + scan-hardcoded + 测试（待 A4 集成后正式跑 vitest） | 2026-09-09 |

**Wave 1 集成基线（Agent 0，本地，未 push）**
- 分支 `i18n/w1-integration`（主 worktree）= 基线 `31e3a76d3f` → 平台 `37dd0678f6`（含你的修复 `6321c8b52d`）→ A7 QA `fff2c41c44` 合并。
- A4 平台 33 单测 + 7 集成全过；A7 QA 19 测试全过（**修复 1 处 A7 测试漏引号 typo**）。
- A7 的 check-keys 对 A4 8 namespace 端到端 **PASS**。
- **未决**：集成分支 `npm run build` 验证中（后台）；PLATFORM_VALIDATION_REPORT、PoC 回填待补；push 因无 GitHub 凭据暂缓（用户决定不 push）。

## Wave 2 — 第一批功能

| Agent | 名称 | 状态 | worktree/分支 | 交付物 | 最后更新 |
|---|---|---|---|---|---|
| A5 | i18n-shell-auth-developer | 待命 | i18n/w2-agent5-shell | 壳层 + common/navigation/auth namespace | - |
| A6 | i18n-feature-developer | 待命 | i18n/w2-agent6-models | Models + API Keys | - |
| A7 | i18n-qa | 待命 | i18n/w2-agent7-qa | 持续测试 | - |

## Wave 3 — 第二批功能

| Agent | 名称 | 状态 | worktree/分支 | 交付物 | 最后更新 |
|---|---|---|---|---|---|
| A6A | i18n-feature-developer (Usage/Cost) | 待命 | i18n/w3-agent6a-usage | Usage + Cost Tracking | - |
| A6B | i18n-feature-developer (Budgets) | 待命 | i18n/w3-agent6b-budgets | Budgets | - |
| A7 | i18n-qa | 待命 | i18n/w3-agent7-qa | 回归 | - |

## Wave 4 — 集成验收

| Agent | 名称 | 状态 | 交付物 | 最后更新 |
|---|---|---|---|---|
| A2 | localization-designer | 待命 | 术语/中文体验复核 | - |
| A7 | i18n-qa | 待命 | 完整回归 + 质量报告 | - |
| A8 | i18n-integration-release | 待命 | 发布/回滚清单 | - |

## 缺陷队列
（Wave 4B 按 P0 → P1 → 阻塞门禁 P2 → 其他 P2 排序）
（空）
