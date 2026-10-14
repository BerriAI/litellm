import { describe, expect, it } from "vitest";
import {
  expandFallbackSelection,
  fallbackModelLabel,
  ModelProviders,
  providerWildcardValues,
  toFallbackModelOption,
} from "./providerWildcards";

const models = ["claude-sonnet-4-5", "anthropic/claude-opus-4-1", "openai/gpt-4o", "gpt-4o-mini"];
const modelProviders: ModelProviders = { "claude-sonnet-4-5": ["anthropic"], "gpt-4o-mini": ["openai"] };

describe("providerWildcardValues", () => {
  it("offers one wildcard per provider found in model metadata or a model name prefix", () => {
    expect(providerWildcardValues(models, modelProviders)).toEqual(["anthropic/*", "openai/*"]);
  });
});

describe("fallbackModelLabel", () => {
  it("names provider wildcards and the global wildcard in plain words", () => {
    expect(fallbackModelLabel("openai/*")).toBe("All OpenAI models");
    expect(fallbackModelLabel("*")).toBe("All models");
    expect(fallbackModelLabel("openai/gpt-4o")).toBe("openai/gpt-4o");
  });

  it("keeps the raw wildcard searchable as the sublabel", () => {
    expect(toFallbackModelOption("openai/*")).toEqual({
      label: "All OpenAI models",
      value: "openai/*",
      sublabel: "openai/*",
    });
  });
});

describe("expandFallbackSelection", () => {
  it("expands a provider wildcard into every model group of that provider", () => {
    expect(
      expandFallbackSelection(["anthropic/*"], models, modelProviders, { primaryModel: "openai/*", maxFallbacks: 10 }),
    ).toEqual(["claude-sonnet-4-5", "anthropic/claude-opus-4-1"]);
  });

  it("drops duplicates and the primary model from the expanded chain", () => {
    expect(
      expandFallbackSelection(["gpt-4o-mini", "openai/*"], models, modelProviders, {
        primaryModel: "openai/gpt-4o",
        maxFallbacks: 10,
      }),
    ).toEqual(["gpt-4o-mini"]);
  });

  it("keeps a wildcard that is itself a configured model group", () => {
    expect(
      expandFallbackSelection(["anthropic/*"], [...models, "anthropic/*"], modelProviders, {
        primaryModel: null,
        maxFallbacks: 10,
      }),
    ).toEqual(["anthropic/*"]);
  });

  it("caps the expanded chain at the max fallback count", () => {
    expect(
      expandFallbackSelection(["anthropic/*"], models, modelProviders, {
        primaryModel: "openai/gpt-4o",
        maxFallbacks: 1,
      }),
    ).toEqual(["claude-sonnet-4-5"]);
  });
});
