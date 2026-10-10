import { describe, expect, it } from "vitest";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import {
  fallbackChainOptions,
  fallbackPrimaryLabel,
  primaryModelOptions,
  resolveFallbackProvider,
} from "./fallbackOptions";

const modelGroups: ModelGroup[] = [
  { model_group: "gpt-4", providers: ["openai"] },
  { model_group: "claude-3-opus", providers: ["anthropic"] },
  { model_group: "bedrock/us.anthropic.claude-opus-4-8", providers: ["bedrock"] },
];

describe("primaryModelOptions", () => {
  it("lists provider wildcards first with friendly labels, then model groups", () => {
    const options = primaryModelOptions(modelGroups);

    expect(options.slice(0, 3)).toEqual([
      { value: "anthropic/*", label: "All Anthropic Models" },
      { value: "bedrock/*", label: "All Amazon Bedrock Models" },
      { value: "openai/*", label: "All OpenAI Models" },
    ]);
    expect(options.slice(3).map((option) => option.value)).toEqual([
      "bedrock/us.anthropic.claude-opus-4-8",
      "claude-3-opus",
      "gpt-4",
    ]);
  });

  it("derives a provider wildcard from the model group prefix when providers are absent", () => {
    const options = primaryModelOptions([{ model_group: "openai/gpt-6-astra" }]);

    expect(options[0]).toEqual({ value: "openai/*", label: "All OpenAI Models" });
    expect(options).toHaveLength(2);
  });

  it("ignores prefixes that are not known provider slugs", () => {
    const options = primaryModelOptions([{ model_group: "my-router/custom-tier" }]);

    expect(options).toEqual([{ value: "my-router/custom-tier", label: "my-router/custom-tier" }]);
  });

  it("folds a model group literally named openai/* into the friendly option", () => {
    const options = primaryModelOptions([{ model_group: "openai/*", providers: ["openai"] }]);

    expect(options).toEqual([{ value: "openai/*", label: "All OpenAI Models" }]);
  });
});

describe("fallbackChainOptions", () => {
  it("excludes wildcard entries, which can never be fallback targets", () => {
    const options = fallbackChainOptions([...modelGroups, { model_group: "openai/*", providers: ["openai"] }]);

    expect(options.map((option) => option.value)).toEqual([
      "bedrock/us.anthropic.claude-opus-4-8",
      "claude-3-opus",
      "gpt-4",
    ]);
  });
});

describe("resolveFallbackProvider", () => {
  it("prefers the cost map litellm_provider", () => {
    expect(resolveFallbackProvider("gpt-4", { "gpt-4": { litellm_provider: "azure" } }, modelGroups)).toBe("azure");
  });

  it("falls back to the model group providers when the cost map misses", () => {
    expect(resolveFallbackProvider("claude-3-opus", null, modelGroups)).toBe("anthropic");
  });

  it("resolves provider wildcards and prefixed names from the prefix", () => {
    expect(resolveFallbackProvider("anthropic/*", null, [])).toBe("anthropic");
    expect(resolveFallbackProvider("openai/gpt-6-astra", null, [])).toBe("openai");
    expect(resolveFallbackProvider("bedrock/us.anthropic.claude-opus-4-8", null, [])).toBe("bedrock");
  });

  it("returns empty when nothing resolves", () => {
    expect(resolveFallbackProvider("custom-group", null, [])).toBe("");
  });
});

describe("fallbackPrimaryLabel", () => {
  it("renders provider wildcards as All <Provider> Models", () => {
    expect(fallbackPrimaryLabel("openai/*")).toBe("All OpenAI Models");
    expect(fallbackPrimaryLabel("anthropic/*")).toBe("All Anthropic Models");
  });

  it("keeps concrete names and unknown wildcards as-is", () => {
    expect(fallbackPrimaryLabel("gpt-4")).toBe("gpt-4");
    expect(fallbackPrimaryLabel("custom/*")).toBe("custom/*");
  });
});
