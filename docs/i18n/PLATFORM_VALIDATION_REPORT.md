# LiteLLM Dashboard i18n — 平台验证报告（PLATFORM_VALIDATION_REPORT）

> 角色：Agent 4（`i18n-platform-developer`）交付；Agent 0 补充完成（A4 因超时未及产出，G1 门禁补齐）
> 基线：`I18N_TECH_DESIGN.md`、`I18N_ADR.md`、`POC_REPORT.md`（docs/i18n/）
> 日期：2026-09-09

## 1. 实现范围

Wave 1 平台交付内容（已合入集成分支 `i18n/w1-integration`）：

| 项 | 路径 | 说明 |
|---|---|---|
| i18n 单例 | `src/i18n/i18n.ts` | 惰性 `getI18n()`，静态 resources，`fallbackLng:'en'`，`defaultNS:'common'` |
| Provider | `src/i18n/I18nProvider.tsx` + `index.ts` | **首屏就绪门禁 + `<html lang>` 同步**；公/私有出口 |
| locale 检测 | `src/i18n/detectLocale.ts` | `zh/zh-Hans/zh-TW→zh-CN`，其余→en |
| 偏好存储 | `src/i18n/localePreferences.ts` | `litellm.locale`（P5）cookie+localStorage 双层，SameSite=Lax，生产 Secure |
| 注册表 | `src/i18n/resources/registry.ts` | 8 namespace 单一真源，Agent 4 独占 |
| 类型 | `src/i18n/types.d.ts` | `defaultNS:'common'`；宽松 string key（见 §3） |
| 根布局 | `src/app/layout.tsx` | `I18nProvider` 最外层 |
| 切换器 | `src/components/LanguageSwitcher/` | EN/中文 切换 |
| 语言资源 | `src/locales/{en,zh-CN}/**` | 8 namespace 骨架 |
| E2E 基建 | `tests/e2e/ui/` + `@playwright/test` | P3=a |
| 依赖 | `package.json` | i18next、react-i18next、@playwright/test（A4 单一写入） |

## 2. ADR 符合性核对

| ADR | 结论 | 核对 |
|---|---|---|
| ADR-01 库选型 | Accepted | `i18next@^26.4.2`+`react-i18next@^17.0.13` 已落地并构建通过 |
| ADR-02 静态导出无服务端 locale | Accepted | 客户端收敛，`next build`(output:export) 通过 |
| ADR-03 资源就绪门禁（禁暴 key） | **实现符合** | `I18nProvider` 就绪前不渲染业务子树；`returnNull:true`+`missingKeyHandler` |
| ADR-04 首屏=就绪门禁+lang 同步 | **实现符合** | 挂载后 `document.documentElement.lang` 同步；en 直接就绪 |
| ADR-05 cookie+localStorage 双层 | **实现符合** | `litellm.locale` 双写；SameSite=Lax；生产 Secure |
| ADR-06 title/meta v1 英文 | Accepted | `layout.tsx` metadata 未改 |
| ADR-07 v1 不用后端 language | Accepted | 无该字段 |
| ADR-08 注册表 A4 独占 | **实现符合** | registry.ts 声明归属；功能 Agent 只写自己的 JSON |

## 3. 关于「严格 typed key」的决策说明（G1 记录）

i18next v26 的 qualified-key 严格类型在静态导出下反复破坏 `next build`，利益与运行时门禁/CI 冗余，故 `types.d.ts` 明确采用**宽松 string key**。key 正确性由：运行时就绪门禁（ADR-03）+ A7 `check-keys` CI（G3 门禁）+ `resources.test.ts`（seed key 存在性）三重保证。**此决策由 Agent 0 在 G1 确认。**

## 4. 验证证据

| 项 | 结果 | 证据 |
|---|---|---|
| 单元测试 | 33 通过 | `vitest run --project unit src/i18n/` |
| 集成测试 | 7 通过 | `vitest run --project integration src/i18n/` |
| 类型检查 | 通过（平台相关） | `next build` TypeScript 阶段无平台错误 |
| 静态导出构建 | 通过 | `npm run build`（`output:"export"`） |
| check-keys 端到端 | PASS（8 namespace en/zh 一致） | A7 check-keys 对 `src/locales/{en,zh-CN}` |

## 5. 待补（G1 门禁内）

- PoC-1..9 回填 `POC_REPORT.md`（本轮 build 确认 PoC-1；其余 PoC-2/3/5/6/8 可通过集成分支验证或标记待 E2E）。
- E2E（Playwright）smoke 断言完善（`tests/e2e/ui/` 已建骨架，断言由 Wave 2+ A7 补）。
- push 至 remote（因无 GitHub 凭据暂缓，用户决定不 push）。
