import { describe, expect, it } from "vitest";
import {
  EMPTY_DECISION_CATALOG,
  buildDecisionCatalog,
  decisionModelsSublabel,
  decisionProviderNames,
  isDecisionMode,
  isDecisionSelection,
} from "./decisionModels";

const COST_MAP = {
  "typesafe/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
  "typesafe/jev-preview": { litellm_provider: "typesafe", mode: "evaluation" },
  "gpt-6-luna": {
    litellm_provider: "openai",
    mode: "chat",
    supported_endpoints: ["/v1/chat/completions", "/v1/decisions"],
  },
  "gpt-5.5": { litellm_provider: "openai", mode: "chat", supported_endpoints: ["/v1/chat/completions"] },
  "perplexity/pplx-decider-v1-27b": { litellm_provider: "perplexity", mode: "evaluation" },
  "perplexity/sonar-pro": { litellm_provider: "perplexity", mode: "chat" },
  "systemone-only": { litellm_provider: "somevendor", supported_endpoints: ["/v1/systemone"] },
  sample_spec: { max_tokens: "set to max_output_tokens if provider specifies it" },
};

describe("isDecisionMode", () => {
  it("is true only for the evaluation mode", () => {
    expect(isDecisionMode("evaluation")).toBe(true);
    expect(isDecisionMode("chat")).toBe(false);
    expect(isDecisionMode(null)).toBe(false);
    expect(isDecisionMode(undefined)).toBe(false);
  });
});

describe("buildDecisionCatalog", () => {
  const catalog = buildDecisionCatalog(COST_MAP);

  it("collects evaluation models and models that list a decision endpoint", () => {
    expect([...catalog.models].sort()).toEqual(
      [
        "gpt-6-luna",
        "perplexity/pplx-decider-v1-27b",
        "systemone-only",
        "typesafe/jev-latest",
        "typesafe/jev-preview",
      ].sort(),
    );
  });

  it("strips the provider prefix from the names shown per provider", () => {
    expect(catalog.providers.get("typesafe")?.names).toEqual(["jev-latest", "jev-preview"]);
    expect(catalog.providers.get("perplexity")?.names).toEqual(["pplx-decider-v1-27b"]);
    expect(catalog.providers.get("openai")?.names).toEqual(["gpt-6-luna"]);
  });

  it("marks a provider decision-only when every one of its models is a decision model", () => {
    expect(catalog.providers.get("typesafe")?.decisionOnly).toBe(true);
    expect(catalog.providers.get("somevendor")?.decisionOnly).toBe(true);
    expect(catalog.providers.get("perplexity")?.decisionOnly).toBe(false);
    expect(catalog.providers.get("openai")?.decisionOnly).toBe(false);
  });

  it("leaves providers without decision models out", () => {
    expect(catalog.providers.has("anthropic")).toBe(false);
  });

  it("returns an empty catalog for a missing or malformed cost map", () => {
    expect(buildDecisionCatalog(undefined)).toEqual(EMPTY_DECISION_CATALOG);
    expect(buildDecisionCatalog("not a map")).toEqual(EMPTY_DECISION_CATALOG);
    expect(buildDecisionCatalog({ broken: { mode: "evaluation" } })).toEqual(EMPTY_DECISION_CATALOG);
  });
});

describe("decisionModelsSublabel", () => {
  const catalog = buildDecisionCatalog(COST_MAP);

  it("lists the provider's decision models", () => {
    expect(decisionModelsSublabel(catalog, "typesafe")).toBe("Decision models: jev-latest, jev-preview");
  });

  it("is undefined for a provider with no decision models", () => {
    expect(decisionModelsSublabel(catalog, "anthropic")).toBeUndefined();
  });
});

describe("isDecisionSelection", () => {
  const catalog = buildDecisionCatalog(COST_MAP);

  it("is true for a decision-only provider before any model is picked", () => {
    expect(isDecisionSelection(catalog, "typesafe", [])).toBe(true);
  });

  it("waits for a decision model on a provider that also serves chat models", () => {
    expect(isDecisionSelection(catalog, "perplexity", [])).toBe(false);
    expect(isDecisionSelection(catalog, "perplexity", ["perplexity/sonar-pro"])).toBe(false);
    expect(isDecisionSelection(catalog, "perplexity", ["perplexity/sonar-pro", "perplexity/pplx-decider-v1-27b"])).toBe(
      true,
    );
  });

  it("is true for a chat-mode model that lists a decision endpoint", () => {
    expect(isDecisionSelection(catalog, "openai", ["gpt-6-luna"])).toBe(true);
    expect(isDecisionSelection(catalog, "openai", ["gpt-5.5"])).toBe(false);
  });

  it("is false with no provider and no models", () => {
    expect(isDecisionSelection(catalog, undefined, [])).toBe(false);
  });
});

describe("decisionProviderNames", () => {
  it("lists each provider with decision models by its display name, sorted", () => {
    expect(decisionProviderNames(buildDecisionCatalog(COST_MAP))).toEqual([
      "OpenAI",
      "Perplexity",
      "Somevendor",
      "TypeSafe",
    ]);
  });

  it("is empty for an empty catalog", () => {
    expect(decisionProviderNames(EMPTY_DECISION_CATALOG)).toEqual([]);
  });
});
