import {
  buildComplexityRouterConfig,
  getSemanticConfigError,
  BuildComplexityRouterConfigParams,
} from "./build_complexity_router_config";

const tiers = {
  SIMPLE: "gpt-4o-mini",
  MEDIUM: "gpt-4o",
  COMPLEX: "claude-sonnet-4",
  REASONING: "o1-preview",
};

const baseParams: BuildComplexityRouterConfigParams = {
  tiers,
  classifierType: "heuristic",
  classifierLlmConfig: undefined,
  customTechnicalKeywords: [],
  keywordTierRules: [],
  semanticMatchingEnabled: false,
  embeddingModel: undefined,
  matchThreshold: 0.5,
};

describe("buildComplexityRouterConfig", () => {
  it("emits only tiers and classifier_type when nothing else is configured", () => {
    const config = buildComplexityRouterConfig(baseParams);
    expect(config).toEqual({ tiers, classifier_type: "heuristic" });
  });

  it("includes classifier_llm_config only when classifier_type is llm", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "llm",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    });
    expect(config.classifier_type).toBe("llm");
    expect(config.classifier_llm_config).toEqual({ model: "gpt-4o-mini", timeout_ms: 3000 });
  });

  it("omits classifier_llm_config when classifier_type is heuristic even if config lingers in state", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "heuristic",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    });
    expect(config.classifier_llm_config).toBeUndefined();
  });

  it("sends keyword_tier_rules with their per-tier targeting preserved (not flattened)", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      customTechnicalKeywords: ["udp"],
      keywordTierRules: [
        { id: "r1", keywords: ["deploy to k8s"], tier: "REASONING" },
        { id: "r2", keywords: ["invoice", "refund"], tier: "SIMPLE" },
      ],
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.keyword_tier_rules).toEqual([
      { keywords: ["deploy to k8s"], tier: "REASONING" },
      { keywords: ["invoice", "refund"], tier: "SIMPLE" },
    ]);
    // custom technical keywords stay their own list, not merged with rule keywords
    expect(config.custom_technical_keywords).toEqual(["udp"]);
    expect(config.semantic_keyword_matching).toBeUndefined();
  });

  it("includes semantic fields only when semantic matching is enabled", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["k8s"], tier: "REASONING" }],
      semanticMatchingEnabled: true,
      embeddingModel: "openai/text-embedding-3-small",
      matchThreshold: 0.42,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.semantic_keyword_matching).toBe(true);
    expect(config.embedding_model).toBe("openai/text-embedding-3-small");
    expect(config.match_threshold).toBe(0.42);
  });

  it("omits semantic fields when the toggle is off even if an embedding model lingers in state", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["k8s"], tier: "REASONING" }],
      semanticMatchingEnabled: false,
      embeddingModel: "openai/text-embedding-3-small",
      matchThreshold: 0.42,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.semantic_keyword_matching).toBeUndefined();
    expect(config.embedding_model).toBeUndefined();
    expect(config.match_threshold).toBeUndefined();
  });

  it("omits empty optional lists", () => {
    const config = buildComplexityRouterConfig(baseParams);
    expect(config.custom_technical_keywords).toBeUndefined();
    expect(config.keyword_tier_rules).toBeUndefined();
  });

  it("trims keywords and drops rules left empty, so unfilled rows never 400 the backend", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [
        { id: "r1", keywords: [" deploy to k8s ", "", "  "], tier: "REASONING" },
        { id: "r2", keywords: [], tier: "COMPLEX" }, // seeded by "Add keyword rule", never filled
        { id: "r3", keywords: ["   "], tier: "SIMPLE" }, // whitespace only
      ],
    };
    const config = buildComplexityRouterConfig(params);
    // r1 keeps only its real keyword (trimmed); r2 and r3 are dropped entirely.
    expect(config.keyword_tier_rules).toEqual([{ keywords: ["deploy to k8s"], tier: "REASONING" }]);
  });

  it("omits keyword_tier_rules entirely when every rule is empty", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["", "  "], tier: "COMPLEX" }],
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.keyword_tier_rules).toBeUndefined();
  });
});

describe("getSemanticConfigError", () => {
  const rule = { id: "r1", keywords: ["k8s"], tier: "REASONING" as const };

  it("returns null when semantic matching is disabled (even with gaps)", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: false, embeddingModel: undefined, keywordTierRules: [] }),
    ).toBeNull();
  });

  it("errors when enabled without an embedding model", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: undefined, keywordTierRules: [rule] }),
    ).toMatch(/embedding model/i);
  });

  it("errors when enabled with an embedding model but no keyword tier rules", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: "voyage-3-5", keywordTierRules: [] }),
    ).toMatch(/keyword tier rule/i);
  });

  it("errors when a rule has no non-empty keywords", () => {
    const emptyRule = { id: "r2", keywords: ["", "  "], tier: "SIMPLE" as const };
    expect(
      getSemanticConfigError({
        semanticMatchingEnabled: true,
        embeddingModel: "voyage-3-5",
        keywordTierRules: [emptyRule],
      }),
    ).toMatch(/at least one keyword/i);
  });

  it("returns null when enabled with both an embedding model and rules", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: "voyage-3-5", keywordTierRules: [rule] }),
    ).toBeNull();
  });
});
