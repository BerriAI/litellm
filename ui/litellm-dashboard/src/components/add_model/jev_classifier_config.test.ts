import { describe, expect, it } from "vitest";
import { defaultJevClassifierConfig, hydrateOssClassifier, jevClassifierConfigSchema } from "./jev_classifier_config";

describe("hydrateOssClassifier", () => {
  it("keeps a stored Databricks classifier instead of falling back to Jev", () => {
    const stored = {
      provider: "databricks",
      model: "ai_decide",
      timeout_ms: 4200,
      circuit_breaker_enabled: true,
      circuit_breaker_cooldown_seconds: 50,
    };

    const hydrated = hydrateOssClassifier({ classifier_type: "oss_classifier", opensource_classifier_config: stored });

    expect(hydrated).toEqual({ classifier_type: "jev", jev_classifier_config: stored });
  });

  it("falls back to Jev only when the stored classifier cannot be parsed", () => {
    const hydrated = hydrateOssClassifier({
      classifier_type: "oss_classifier",
      opensource_classifier_config: { provider: "databricks", model: "databricks-openjev-qwen35-4b" },
    });

    expect(hydrated.jev_classifier_config).toEqual(defaultJevClassifierConfig());
  });
});

describe("jevClassifierConfigSchema", () => {
  it("starts a Databricks classifier on ai_decide, which passes as is", () => {
    const fresh = defaultJevClassifierConfig("databricks");

    expect(fresh).toEqual({ provider: "databricks", model: "ai_decide", timeout_ms: 3000 });
    expect(jevClassifierConfigSchema.safeParse(fresh).success).toBe(true);
  });

  it.each(["databricks-openjev-qwen35-4b", "serving-endpoints/ai_decide", "ai-decide", "AI_DECIDE"])(
    "rejects the Databricks model %j because the ai-decide route serves only ai_decide",
    (model) => {
      expect(jevClassifierConfigSchema.safeParse({ provider: "databricks", model }).success).toBe(false);
    },
  );
});
