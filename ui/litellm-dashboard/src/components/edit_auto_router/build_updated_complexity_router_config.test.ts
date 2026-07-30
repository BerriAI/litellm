import { describe, expect, it } from "vitest";

import { buildUpdatedComplexityRouterConfig, type KeywordMatchingState } from "./edit_auto_router_modal";

const STORED = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
  classifier_type: "heuristic",
  keyword_tier_rules: [{ keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  escalation_keywords: ["urgent", "outage"],
  semantic_keyword_matching: true,
  embedding_model: "voyage-4-large",
  match_threshold: 0.72,
  // A key no UI control owns; it must survive every save untouched.
  some_future_backend_key: { nested: true },
};

const FORM_VALUE = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
  classifier_type: "heuristic" as const,
};

const hydratedState: KeywordMatchingState = {
  keywordTierRules: [{ id: "stored-0", keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  escalationKeywords: ["urgent", "outage"],
  semanticMatchingEnabled: true,
  embeddingModel: "voyage-4-large",
  matchThreshold: 0.72,
};

describe("buildUpdatedComplexityRouterConfig keyword matching", () => {
  it("round-trips an untouched edit without changing any keyword-matching value", () => {
    // Opening the modal hydrates state from STORED; saving with nothing changed must be a
    // no-op. These keys are now MANAGED, so a hydration bug silently wipes them.
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, hydratedState);

    expect(result.keyword_tier_rules).toEqual([{ keywords: ["invoice", "refund"], tier: "MEDIUM" }]);
    expect(result.escalation_keywords).toEqual(["urgent", "outage"]);
    expect(result.semantic_keyword_matching).toBe(true);
    expect(result.embedding_model).toBe("voyage-4-large");
    expect(result.match_threshold).toBe(0.72);
  });

  it("preserves keys no control owns", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, hydratedState);
    expect(result.some_future_backend_key).toEqual({ nested: true });
  });

  it("persists an edited keyword rule", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, {
      ...hydratedState,
      keywordTierRules: [{ id: "stored-0", keywords: ["chargeback"], tier: "COMPLEX" }],
    });

    expect(result.keyword_tier_rules).toEqual([{ keywords: ["chargeback"], tier: "COMPLEX" }]);
  });

  it("drops a rule left empty rather than shipping one the backend 400s on", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, {
      ...hydratedState,
      keywordTierRules: [{ id: "new-1", keywords: ["   "], tier: "SIMPLE" }],
    });

    expect(result.keyword_tier_rules).toBeUndefined();
  });

  it("removes the semantic trio when the toggle is turned off", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, {
      ...hydratedState,
      semanticMatchingEnabled: false,
    });

    expect(result.semantic_keyword_matching).toBeUndefined();
    expect(result.embedding_model).toBeUndefined();
    expect(result.match_threshold).toBeUndefined();
  });

  it("carries stored keyword matching through untouched when the caller owns no such state", () => {
    // Any caller that does not render these controls must not have its values dropped just
    // because the keys are listed as managed.
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE);

    expect(result.keyword_tier_rules).toEqual([{ keywords: ["invoice", "refund"], tier: "MEDIUM" }]);
    expect(result.escalation_keywords).toEqual(["urgent", "outage"]);
    expect(result.semantic_keyword_matching).toBe(true);
    expect(result.embedding_model).toBe("voyage-4-large");
    expect(result.match_threshold).toBe(0.72);
  });
});
