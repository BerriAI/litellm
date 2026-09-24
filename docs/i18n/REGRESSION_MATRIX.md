# i18n 回归测试矩阵（REGRESSION_MATRIX.md）

> 角色：Agent 3 `qa-architect`；执行：Agent 7 `i18n-qa`（Wave 4A 完整回归）
> 依据：`I18N_MULTI_AGENT_PLAN.md` §11 完成定义、§10 测试重点；`TEST_CASES.md`
> 用法：按"页面 × 关键行为"逐格勾选。行为列引用 TC 编号；页面以 v1 范围为准。每格填：`✅ 通过 / ❌ 失败(#缺陷) / ➖ 不适用 / (m) 人工`。

## 行为列（横轴）编号

| 列 | 行为 | 主要 TC | 层级 |
|---|---|---|---|
| B1 | 默认语言 en | TC-01/02 | e2e/unit |
| B2 | 切换即时生效 + `<html lang>` | TC-03/08 | e2e/component |
| B3 | 刷新保持 | TC-04/05 | e2e/unit |
| B4 | 缺 key 回退英文、不显示原始 key | TC-06/07 | e2e/component |
| B5 | 动态数量/日期/数字/货币本地化（含 count=0/1/2） | TC-09~13 | unit |
| B6 | 模型名/API 字段/日志/代码示例不误翻 | TC-20/21 | integration |
| B7 | a11y 文案本地化（aria-label/title/placeholder/error） | TC-23 | component/integration |
| B8 | 中文布局不溢出/截断/遮挡 | TC-24 | integration/e2e+(m) |
| B9 | 切换语言不清空表单/不触发异常请求 | TC-14/15 | integration/e2e |
| B10 | 整页跳转（Login/SSO/MCP OAuth）回跳语言保持 | TC-16~19 | e2e |

## 矩阵（页面 × 行为）

| 页面/区域 | Owner | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | Review |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Navbar | Agent 5 | | | | | | | | | | ➖ | |
| Leftnav | Agent 5 | | | | | | | | | | ➖ | |
| 用户菜单 / 全局壳层 | Agent 5 | | | | | | | | | | ➖ | |
| 语言切换器 | Agent 5 | | | | | | | | | ➖ | | |
| Login | Agent 5 | | | | | | | | | | | |
| Onboarding | Agent 5 | | | | | | | | | | | |
| Connect | Agent 5 | | | | | | | | | | | |
| MCP OAuth | Agent 5 | | | | | | | | | | | |
| Models and Endpoints | Agent 6 | | | | | | | | | | | |
| API Keys | Agent 6 | | | | | | | | | | | |
| Usage | Agent 6A | | | | | | | | | | | |
| Cost Tracking | Agent 6A | | | | | | | | | | | |
| Budgets | Agent 6B | | | | | | | | | | | |
| 全局：构建期 `<title>`/meta 英文 | Agent 4/8 | | | | | ➖ | | | | ➖ | ➖ | |

## 平台层回归（不绑定单页面）

| 项目 | 说明 | TC | 结果 |
|---|---|---|---|
| 默认语言 en（无偏好） | 全站 | TC-01 | |
| `<html lang>` 一致 | 全局 | TC-08 | |
| 刷新保持 | 全局 | TC-04 | |
| 缺 key 英文回退 | 全局 | TC-06 | |
| 不显示原始 key（首帧/全程） | 全局 | TC-07/26 | |
| 静态产物 key 泄漏 / `<title>` 英文 | `out/` | TC-22/26 | |
| en/zh key 集合一致 | 全部 namespace | TC-25 | |
| 硬编码扫描报告 | 源码 | T-03 | |

## 门禁映射

- 本矩阵在 **Wave 4A** 全量执行（Agent 7 完成 + Agent 8 汇总 + Agent 2 体验复核 + Agent 0 终审）。
- 对应完成定义（§11）：支持双语言(1)、切换即时+刷新保持(2)、v1 页面中文化(3)、回退不露 key(4)、中文无严重截断重叠(5)、切换不清表单(6)、整页跳转保持(7)、`<title>`/meta 英文(8)、模型名/API 不误翻(9)、相关测试通过(10)、lint/format/build(11)。
- 任何 `❌` 须挂缺陷进 `TASK_BOARD.md`，P0/P1 清零后才满足 G4。
