import { describe, it, expect } from "vitest";
import {
  getProviderCacheReadTokens,
  hasProviderPromptCacheHit,
  splitPromptTokens,
  deriveRate,
} from "./cacheCostBreakdown";

describe("getProviderCacheReadTokens", () => {
  it("reads OpenAI-style usage_object.prompt_tokens_details.cached_tokens", () => {
    const metadata = { usage_object: { prompt_tokens_details: { cached_tokens: 430464 } } };
    expect(getProviderCacheReadTokens(metadata)).toBe(430464);
  });

  it("reads Anthropic-style additional_usage_values.cache_read_input_tokens", () => {
    const metadata = { additional_usage_values: { cache_read_input_tokens: 1200 } };
    expect(getProviderCacheReadTokens(metadata)).toBe(1200);
  });

  it("prefers the OpenAI cached_tokens over the Anthropic field when both are set", () => {
    const metadata = {
      usage_object: { prompt_tokens_details: { cached_tokens: 999 } },
      additional_usage_values: { cache_read_input_tokens: 111 },
    };
    expect(getProviderCacheReadTokens(metadata)).toBe(999);
  });

  it("falls back to the Anthropic field when cached_tokens is 0", () => {
    const metadata = {
      usage_object: { prompt_tokens_details: { cached_tokens: 0 } },
      additional_usage_values: { cache_read_input_tokens: 55 },
    };
    expect(getProviderCacheReadTokens(metadata)).toBe(55);
  });

  it("returns 0 when no provider cache tokens are present", () => {
    expect(getProviderCacheReadTokens({})).toBe(0);
    expect(getProviderCacheReadTokens(undefined)).toBe(0);
    expect(getProviderCacheReadTokens(null)).toBe(0);
  });

  it("ignores non-numeric values", () => {
    const metadata = { usage_object: { prompt_tokens_details: { cached_tokens: "oops" } } };
    expect(getProviderCacheReadTokens(metadata)).toBe(0);
  });
});

describe("hasProviderPromptCacheHit", () => {
  it("is true only when provider cache tokens > 0", () => {
    expect(hasProviderPromptCacheHit({ usage_object: { prompt_tokens_details: { cached_tokens: 1 } } })).toBe(true);
    expect(hasProviderPromptCacheHit({ usage_object: { prompt_tokens_details: { cached_tokens: 0 } } })).toBe(false);
    expect(hasProviderPromptCacheHit({})).toBe(false);
  });
});

describe("splitPromptTokens", () => {
  it("splits using the issue's numbers: 430798 prompt, 430464 cached => 334 miss", () => {
    const result = splitPromptTokens(430798, 430464);
    expect(result.cacheMissTokens).toBe(334);
    expect(result.cacheHitTokens).toBe(430464);
    expect(result.promptTokens).toBe(430798);
  });

  it("treats 0 provider-cache tokens as an all-miss request", () => {
    const result = splitPromptTokens(500, 0);
    expect(result.cacheMissTokens).toBe(500);
    expect(result.cacheHitTokens).toBe(0);
  });

  it("clamps cache-hit tokens to the prompt total so miss can never go negative", () => {
    const result = splitPromptTokens(100, 250);
    expect(result.cacheHitTokens).toBe(100);
    expect(result.cacheMissTokens).toBe(0);
  });

  it("handles missing/invalid prompt counts without producing NaN", () => {
    const result = splitPromptTokens(NaN, 10);
    expect(result.promptTokens).toBe(0);
    expect(result.cacheHitTokens).toBe(0);
    expect(result.cacheMissTokens).toBe(0);
  });
});

describe("deriveRate", () => {
  it("derives the full input rate from cache-miss cost and tokens", () => {
    // 334 miss tokens * 0.00000059 = 0.00019706
    expect(deriveRate(0.00019706, 334)).toBeCloseTo(0.00000059, 12);
  });

  it("derives the discounted cache-read rate", () => {
    // 430464 hit tokens * 0.00000015 = 0.0645696
    expect(deriveRate(0.0645696, 430464)).toBeCloseTo(0.00000015, 12);
  });

  it("returns undefined when token count is 0 (no divide-by-zero)", () => {
    expect(deriveRate(1.23, 0)).toBeUndefined();
  });

  it("returns undefined when cost is missing", () => {
    expect(deriveRate(undefined, 100)).toBeUndefined();
  });

  it("returns undefined for a negative cost so no negative rate is shown", () => {
    // Guards against inconsistent backend costs where read+write cost exceeds
    // input cost, which would otherwise render a negative per-token rate.
    expect(deriveRate(-0.0001, 100)).toBeUndefined();
  });
});
