import { describe, expect, it } from "vitest";
import {
  EMPTY_DECISION_CATALOG,
  buildDecisionCatalog,
  buildDecisionGarden,
  displayMode,
  expandModeFilterValues,
  isDecisionMode,
  isDecisionSelection,
  parseProxyDecisionProviders,
} from "./decisionModels";

const COST_MAP = {
  "typesafe/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
  "typesafe/jev-preview": { litellm_provider: "typesafe", mode: "evaluation" },
  "cloudflare/clef": { litellm_provider: "cloudflare", mode: "decisions" },
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
  it("is true for both the decisions and evaluation modes", () => {
    expect(isDecisionMode("decisions")).toBe(true);
    expect(isDecisionMode("evaluation")).toBe(true);
    expect(isDecisionMode("chat")).toBe(false);
    expect(isDecisionMode(null)).toBe(false);
    expect(isDecisionMode(undefined)).toBe(false);
  });
});

describe("displayMode", () => {
  it("shows every decision mode spelling as decisions and leaves other modes alone", () => {
    expect(displayMode("evaluation")).toBe("decisions");
    expect(displayMode("decisions")).toBe("decisions");
    expect(displayMode("chat")).toBe("chat");
    expect(displayMode("embedding")).toBe("embedding");
    expect(displayMode(null)).toBeUndefined();
    expect(displayMode(undefined)).toBeUndefined();
  });
});

describe("expandModeFilterValues", () => {
  it("widens a decisions selection to every mode spelling the route stores", () => {
    expect(expandModeFilterValues(["decisions"])).toEqual(["evaluation", "decisions"]);
    expect(expandModeFilterValues(["chat", "decisions"])).toEqual(["chat", "evaluation", "decisions"]);
    expect(expandModeFilterValues(["chat"])).toEqual(["chat"]);
  });
});

describe("buildDecisionGarden", () => {
  const garden = buildDecisionGarden(COST_MAP, null);

  it("lists each decision-mode model once, one per provider before any provider's second", () => {
    expect(garden).toEqual([
      { model: "cloudflare/clef", provider: "cloudflare", route: "decisions" },
      { model: "perplexity/pplx-decider-v1-27b", provider: "perplexity", route: "decisions" },
      { model: "typesafe/jev-latest", provider: "typesafe", route: "decisions" },
      { model: "typesafe/jev-preview", provider: "typesafe", route: "decisions" },
    ]);
  });

  it("puts self-hosted models that only serve their own route after every /v1/decisions model", () => {
    const rows = buildDecisionGarden(
      {
        "aaa/self-hosted": { litellm_provider: "aaa", mode: "evaluation", supported_endpoints: ["/v1/systemone"] },
        "typesafe/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
        "typesafe/jev-preview": { litellm_provider: "typesafe", mode: "evaluation" },
        "zzz/decider": { litellm_provider: "zzz", mode: "evaluation", supported_endpoints: ["/v1/decisions"] },
      },
      null,
    );
    expect(rows.map((row) => [row.model, row.route])).toEqual([
      ["typesafe/jev-latest", "decisions"],
      ["zzz/decider", "decisions"],
      ["typesafe/jev-preview", "decisions"],
      ["aaa/self-hosted", "self-hosted"],
    ]);
  });

  it("shows a provider's shortest model name first, so an alias outranks its long twin", () => {
    const rows = buildDecisionGarden(
      {
        "cloudflare/@cf/cloudflare/clef": { litellm_provider: "cloudflare", mode: "evaluation" },
        "cloudflare/clef": { litellm_provider: "cloudflare", mode: "evaluation" },
        "databricks/ai_decide": { litellm_provider: "databricks", mode: "evaluation" },
      },
      null,
    );
    expect(rows.map((row) => row.model)).toEqual([
      "cloudflare/clef",
      "databricks/ai_decide",
      "cloudflare/@cf/cloudflare/clef",
    ]);
  });

  it("leaves chat models out even when they serve a decision endpoint", () => {
    expect(garden.map((row) => row.model)).not.toContain("gpt-6-luna");
    expect(garden.map((row) => row.model)).not.toContain("systemone-only");
  });

  it("marks a model from a provider this proxy cannot serve on /v1/decisions as needing a newer LiteLLM, last", () => {
    const rows = buildDecisionGarden(
      {
        "aaa/self-hosted": { litellm_provider: "aaa", mode: "evaluation", supported_endpoints: ["/v1/systemone"] },
        "newvendor/decider": { litellm_provider: "newvendor", mode: "evaluation" },
        "typesafe/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
      },
      new Set(["typesafe"]),
    );
    expect(rows.map((row) => [row.model, row.route])).toEqual([
      ["typesafe/jev-latest", "decisions"],
      ["aaa/self-hosted", "self-hosted"],
      ["newvendor/decider", "needs-upgrade"],
    ]);
  });

  it("is empty for a missing or malformed cost map", () => {
    expect(buildDecisionGarden(undefined, null)).toEqual([]);
    expect(buildDecisionGarden("not a map", null)).toEqual([]);
  });
});

describe("parseProxyDecisionProviders", () => {
  it("reads the decision providers this proxy serves, and nothing from a response without them", () => {
    expect(parseProxyDecisionProviders({ decision_model_providers: ["typesafe", "cloudflare"] })).toEqual(
      new Set(["typesafe", "cloudflare"]),
    );
    expect(parseProxyDecisionProviders({ supported_modes: ["pre_call"] })).toBeNull();
    expect(parseProxyDecisionProviders(undefined)).toBeNull();
  });
});

describe("buildDecisionCatalog", () => {
  const catalog = buildDecisionCatalog(COST_MAP);

  it("collects evaluation models and models that list a decision endpoint", () => {
    expect([...catalog.models].sort()).toEqual(
      [
        "cloudflare/clef",
        "gpt-6-luna",
        "perplexity/pplx-decider-v1-27b",
        "systemone-only",
        "typesafe/jev-latest",
        "typesafe/jev-preview",
      ].sort(),
    );
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
