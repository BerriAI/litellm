import { describe, expect, it } from "vitest";
import { validateModelName } from "./model_name_validation";

const ANTHROPIC_MODELS = [
  "claude-3-5-sonnet-20240620",
  "claude-3-opus-20240229",
  "claude-opus-4-20250514",
  "claude-sonnet-4-20250514",
];

const OPENAI_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"];

describe("validateModelName", () => {
  it("flags a bare provider keyword like 'claude' as misnamed and offers known model suggestions", () => {
    const result = validateModelName("claude", ANTHROPIC_MODELS);
    expect(result.isMisnamed).toBe(true);
    expect(result.suggestions).toContain("claude-3-5-sonnet-20240620");
    expect(result.suggestions.length).toBeGreaterThan(0);
    expect(result.suggestions.length).toBeLessThanOrEqual(5);
  });

  it("flags 'gpt' against an OpenAI catalog", () => {
    const result = validateModelName("gpt", OPENAI_MODELS);
    expect(result.isMisnamed).toBe(true);
    expect(result.suggestions).toEqual(expect.arrayContaining(["gpt-4", "gpt-4-turbo", "gpt-4o"]));
  });

  it("is case insensitive when matching exact known models", () => {
    expect(validateModelName("CLAUDE-3-OPUS-20240229", ANTHROPIC_MODELS).isMisnamed).toBe(false);
  });

  it("does not flag an exact match in the catalog", () => {
    expect(validateModelName("claude-3-5-sonnet-20240620", ANTHROPIC_MODELS)).toEqual({
      isMisnamed: false,
      suggestions: [],
    });
  });

  it("does not flag a fully qualified custom model that does not look like a known prefix", () => {
    expect(validateModelName("my-finetuned-llama", ANTHROPIC_MODELS)).toEqual({
      isMisnamed: false,
      suggestions: [],
    });
  });

  it("strips a provider prefix before matching, so 'anthropic/claude' is still flagged", () => {
    const result = validateModelName("anthropic/claude", ANTHROPIC_MODELS);
    expect(result.isMisnamed).toBe(true);
    expect(result.suggestions.length).toBeGreaterThan(0);
  });

  it("does nothing when the catalog is empty (no signal to validate against)", () => {
    expect(validateModelName("claude", [])).toEqual({ isMisnamed: false, suggestions: [] });
  });

  it("ignores whitespace-only values", () => {
    expect(validateModelName("   ", ANTHROPIC_MODELS)).toEqual({ isMisnamed: false, suggestions: [] });
  });

  it("ignores null and undefined", () => {
    expect(validateModelName(null, ANTHROPIC_MODELS)).toEqual({ isMisnamed: false, suggestions: [] });
    expect(validateModelName(undefined, ANTHROPIC_MODELS)).toEqual({ isMisnamed: false, suggestions: [] });
  });

  it("caps suggestions at five entries", () => {
    const manyClaudes = Array.from({ length: 10 }, (_, i) => `claude-${i}-foo`);
    const result = validateModelName("claude", manyClaudes);
    expect(result.suggestions).toHaveLength(5);
  });

  it("does not flag a model whose name only happens to contain the substring without a separator", () => {
    expect(validateModelName("claude", ["my-claude-fork"]).isMisnamed).toBe(false);
  });
});
