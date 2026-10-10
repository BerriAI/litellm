import { describe, expect, it } from "vitest";

import {
  availableProviders,
  browseModels,
  BROWSE_PAGE_SIZE,
  FEATURED_MODELS,
  formatPerMillion,
  formatTokens,
  LAUNCHES,
  toCatalog,
  visibleLaunches,
} from "./discoverContent";

const costMap = {
  sample_spec: { litellm_provider: "one of https://docs.litellm.ai/docs/providers", mode: "chat" },
  "openai/*": { litellm_provider: "openai", mode: "chat" },
  "gpt-6.1-sol": { litellm_provider: "openai", mode: "chat", max_input_tokens: 922_000, input_cost_per_token: 2e-6 },
  "claude-haiku-5-5": { litellm_provider: "anthropic", mode: "chat", max_input_tokens: 1_000_000 },
  "text-embedding-3-small": { litellm_provider: "openai", mode: "embedding", max_tokens: 8191 },
  "whisper-1": { litellm_provider: "openai", mode: "audio_transcription" },
  "o9-responses": { litellm_provider: "openai", mode: "responses", max_input_tokens: 400_000 },
  ...Object.fromEntries(
    Array.from({ length: 15 }, (_, i) => [
      `bedrock-chat-${i}`,
      { litellm_provider: "bedrock", mode: "chat", max_input_tokens: 1000 * (i + 1) },
    ]),
  ),
};

describe("visibleLaunches", () => {
  it("sorts every launch newest first", () => {
    const dates = visibleLaunches(LAUNCHES, "all").map((l) => l.publishedOn);
    expect(dates).toEqual([...dates].sort().reverse());
    expect(dates).toHaveLength(LAUNCHES.length);
  });

  it("keeps only the requested kind", () => {
    const models = visibleLaunches(LAUNCHES, "model");
    expect(models.length).toBeGreaterThan(0);
    expect(models.every((l) => l.kind === "model")).toBe(true);
  });
});

describe("toCatalog", () => {
  it("drops sample_spec and wildcard keys and types the fields", () => {
    const catalog = toCatalog(costMap);
    expect(catalog.map((m) => m.name)).not.toContain("sample_spec");
    expect(catalog.map((m) => m.name)).not.toContain("openai/*");
    expect(catalog.find((m) => m.name === "gpt-6.1-sol")).toEqual({
      name: "gpt-6.1-sol",
      provider: "openai",
      mode: "chat",
      contextWindow: 922_000,
      inputCostPerToken: 2e-6,
      outputCostPerToken: null,
    });
    expect(catalog.find((m) => m.name === "text-embedding-3-small")?.contextWindow).toBe(8191);
  });

  it("returns nothing while the cost map has not loaded", () => {
    expect(toCatalog(undefined)).toEqual([]);
  });
});

describe("browseModels", () => {
  const catalog = toCatalog(costMap);

  it("shows the featured models in curated order when nothing is filtered", () => {
    const names = browseModels(catalog, { query: "", provider: null }).map((m) => m.name);
    expect(names).toEqual(FEATURED_MODELS.filter((name) => name in costMap));
    expect(names).toEqual(["gpt-6.1-sol", "claude-haiku-5-5"]);
  });

  it("filters by provider, keeps chat and responses modes only, biggest context first, capped to a page", () => {
    const openai = browseModels(catalog, { query: "", provider: "openai" }).map((m) => m.name);
    expect(openai).toEqual(["gpt-6.1-sol", "o9-responses"]);
    const bedrock = browseModels(catalog, { query: "", provider: "bedrock" });
    expect(bedrock).toHaveLength(BROWSE_PAGE_SIZE);
    expect(bedrock[0].name).toBe("bedrock-chat-14");
  });

  it("searches every mode by name, case insensitive, within the chosen provider", () => {
    expect(browseModels(catalog, { query: "WHISPER", provider: null }).map((m) => m.name)).toEqual(["whisper-1"]);
    expect(browseModels(catalog, { query: "whisper", provider: "anthropic" })).toEqual([]);
  });
});

describe("availableProviders", () => {
  it("lists only featured providers present in the catalog, in the featured order", () => {
    expect(availableProviders(toCatalog(costMap))).toEqual(["openai", "anthropic", "bedrock"]);
  });
});

describe("formatters", () => {
  it("prices per million tokens and abbreviates context windows", () => {
    expect(formatPerMillion(2e-6)).toBe("$2.00");
    expect(formatPerMillion(null)).toBe("-");
    expect(formatTokens(1_000_000)).toBe("1.0M");
    expect(formatTokens(922_000)).toBe("922K");
    expect(formatTokens(null)).toBe("-");
  });
});
