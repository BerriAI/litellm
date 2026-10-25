import { describe, expect, it } from "vitest";
import { translate, type Translate } from "@/i18n/language";
import {
  getTierLabelsError,
  getMissingTiersError,
  getPlanModeTierError,
  getKeywordTierRulesError,
  getHeuristicV2SuccessThresholdError,
  getReminderMarkersError,
  getClassifierPluginTimeoutError,
  getClassifierModelError,
  getClassifierReasoningEffortError,
  getSemanticConfigError,
  dryRunRejection,
} from "./build_complexity_router_config";
import { getCustomTierRowsError, type TierRow } from "./tier_rows";
import { getForecastConfigError } from "./forecast_classifier_config";
import { customDimensionsError } from "./custom_dimensions";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const zh: Translate = (message, values) => translate("zh-CN", message, values);
const rows: TierRow[] = [{ id: "custom", name: "Home", definition: "User text", models: [] }];
const config: ComplexityRouterConfigValue = {
  classifier_type: "llm",
  tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
};

describe("自动路由本地校验翻译", () => {
  it.each([
    ["层级重名", () => getTierLabelsError({ SIMPLE: "MEDIUM" }, zh)],
    ["缺少模型", () => getMissingTiersError(rows, zh)],
    ["计划模式", () => getPlanModeTierError("custom", rows, zh)],
    ["空关键词", () => getKeywordTierRulesError([{ id: "rule", tier: "Home", keywords: [] }], rows, zh)],
    ["概率阈值", () => getHeuristicV2SuccessThresholdError(2, zh)],
    ["提醒标签", () => getReminderMarkersError([{ open: "", close: "close" }], zh)],
    ["插件超时", () => getClassifierPluginTimeoutError("custom", 0, zh)],
    ["分类模型", () => getClassifierModelError(config, zh)],
    [
      "推理强度",
      () =>
        getClassifierReasoningEffortError(
          { ...config, classifier_llm_config: { model: "Home", timeout_ms: 1, reasoning_effort: "high" } },
          [{ model_group: "Home", supported_reasoning_efforts: ["low"] }] as never,
          zh,
        ),
    ],
    [
      "语义模型",
      () =>
        getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: undefined, keywordTierRules: [] }, zh),
    ],
    ["自定义层级", () => getCustomTierRowsError({ tiers: rows, fallback_tier_id: "custom" }, zh)],
    [
      "预测配置",
      () =>
        getForecastConfigError(
          { ...config, classifier_type: "capability", classifier_llm_config: { model: "Home", timeout_ms: 0 } },
          zh,
        ),
    ],
    ["自定义维度", () => customDimensionsError([{ id: "a", name: "", weight: 0.2 }], undefined, zh)],
    ["本地拒绝回退", () => dryRunRejection({ valid: false }, zh)],
  ])("%s错误以中文显示", (_label, validate) => {
    expect(validate()).toMatch(/[\u3400-\u9fff]/);
  });

  it("动态层级与服务端诊断保持原文", () => {
    expect(getMissingTiersError(rows, zh)).toContain("Home");
    expect(getPlanModeTierError("custom", rows, zh)).toContain("Home");
    expect(dryRunRejection({ valid: false, error: "Home" }, zh)).toBe("Home");
  });
});
