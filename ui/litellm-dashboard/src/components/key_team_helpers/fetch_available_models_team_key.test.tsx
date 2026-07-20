import { describe, expect, it } from "vitest";

import {
  getModelDisplayName,
  resolveModelBudgetOptions,
  unfurlWildcardModelsInList,
} from "./fetch_available_models_team_key";

const ALL_MODELS = ["claude-sonnet-4-6", "claude-opus-4-8", "openai/gpt-4o"];

describe("getModelDisplayName", () => {
  it("should return display label for all proxy models", () => {
    expect(getModelDisplayName("all-proxy-models")).toBe("All Proxy Models");
  });

  it("should return provider-wide label for wildcard models", () => {
    expect(getModelDisplayName("openai/*")).toBe("All openai models");
  });
});

describe("resolveModelBudgetOptions", () => {
  it("returns all proxy models when team models is empty", () => {
    expect(resolveModelBudgetOptions([], ALL_MODELS)).toEqual(ALL_MODELS);
  });

  it("returns all proxy models when team uses all-proxy-models", () => {
    expect(resolveModelBudgetOptions(["all-proxy-models"], ALL_MODELS)).toEqual(ALL_MODELS);
  });

  it("returns specific team models when explicitly set", () => {
    expect(resolveModelBudgetOptions(["claude-sonnet-4-6"], ALL_MODELS)).toEqual(["claude-sonnet-4-6"]);
  });
});

describe("unfurlWildcardModelsInList", () => {
  it("expands provider wildcards into concrete models", () => {
    expect(unfurlWildcardModelsInList(["openai/*"], ALL_MODELS)).toEqual(["openai/*", "openai/gpt-4o"]);
  });
});
