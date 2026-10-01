# LiteLLM Dashboard 中英文国际化 — MASTER_PLAN

> 总控：Agent 0（`i18n-lead-reviewer`） · 由当前主智能体承担
> 基线：`I18N_MULTI_AGENT_PLAN.md` v1.4（评审通过，可执行）
> 目标：Dashboard 前端 UI 支持 `en` / `zh-CN`，导航、登录、Models、API Keys、Usage、Cost、Budgets 优先中文化。

## 波次总览

| 波次 | 内容 | 涉及角色 | 门禁 |
|---|---|---|---|
| Wave 0 | 基线评审 + 并行设计 | A0 + A1/A2/A3 | G0 |
| Wave 1 | 平台能力与测试基础 | A0 + A4/A5(只读)/A7 | G1 |
| Wave 2 | 第一批功能并行开发 | A0 + A5/A6/A7 | G2 |
| Wave 3 | 第二批功能并行开发 | A0 + A6A/A6B/A7 | G3 |
| Wave 4A/4B | 集成验收 + 定向缺陷修复 | A0 + A2/A7/A8(及开发 Owner) | G4 |

## 并发约束
同刻最多 4 个智能体（1 总控 Agent 0 + 最多 3 执行子代理）。子代理按波次由 Agent 0 拉起，完成即回收。

## 当前状态
（由 Agent 0 在每波次更新）

- [ ] Wave 0 进行中
- [ ] Wave 1 待开始
- [ ] Wave 2 待开始
- [ ] Wave 3 待开始
- [ ] Wave 4 待开始

## 完成定义
见 `I18N_MULTI_AGENT_PLAN.md` §11（v1 完成定义 15 条）。
